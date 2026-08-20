"""
Lightweight Open3D viewer for a completed
tools/bagviz/run_object_inference_debug.py run: shows each camera's
segmented-objects point cloud + a coordinate triad per detected object
(from that camera's frame_dir/offline_inference/pointcloud_objects_debug.ply
/ poses_objects_debug.yaml -- known objects only, unknowns were already
dropped at inference time), one window per camera, followed by one combined
window with all cameras' objects clouds + triads overlaid in the shared
base frame.

Companion to tools/bagviz/view_pointclouds.py -- same run-dir layout, same
T_base_cam resolution (frame_info.yaml, or --use-config-extrinsics to
recompute from the extrinsics YAMLs) -- but for
run_object_inference_debug.py's *objects-only* clouds/poses instead of
capture_pipeline_snapshots.py's full-scene clouds.

This is deliberately a SEPARATE script from run_object_inference_debug.py:
that one needs the full GPU inference stack and runs inside the (normally
headless) `vision` docker container, where Open3D can't open a window
(GLFW/XDG_RUNTIME_DIR errors). This script needs only numpy/open3d/pyyaml --
run it on a display-capable host, in the lightweight `bagviz` conda env,
same as view_pointclouds.py.

Usage:
    conda activate bagviz
    python -m tools.bagviz.view_object_inference_debug \\
        --run-dir outputs/bagviz/<run> --frame 0
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d
import yaml

from tools.bagviz.view_pointclouds import (
    AXIS_LEN_BASE_M,
    AXIS_LEN_OBJECT_M,
    base_cloud_unavailable_reason,
    camera_frame_dir,
    camera_triad,
    pose_dict_to_T,
    resolve_T_base_cam,
    show,
)

ALL_CAMS = ["zed2i_1", "realsense_1", "realsense_2"]

# Matches OFFLINE_SUBDIR in run_object_inference_debug.py -- duplicated
# (rather than imported) because that module pulls in the full GPU
# inference stack at import time, which this lightweight viewer must not
# need.
OFFLINE_SUBDIR = "offline_inference"


def load_objects_cloud(offline_dir: Path) -> Optional[o3d.geometry.PointCloud]:
    ply = offline_dir / "pointcloud_objects_debug.ply"
    if not ply.exists():
        return None
    pc = o3d.io.read_point_cloud(str(ply))
    return pc if len(pc.points) > 0 else None


def load_detections(offline_dir: Path) -> list[dict]:
    yml = offline_dir / "poses_objects_debug.yaml"
    if not yml.exists():
        return []
    data = yaml.safe_load(yml.read_text()) or {}
    return data.get("detections", [])


def object_triads(
    detections: list[dict],
    T_extra: Optional[np.ndarray] = None,
) -> list[o3d.geometry.TriangleMesh]:
    triads = []
    for d in detections:
        if "pose_camera" not in d:
            continue
        T = pose_dict_to_T(d["pose_camera"])
        if T_extra is not None:
            T = T_extra @ T
        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=AXIS_LEN_OBJECT_M
        )
        mesh.transform(T)
        triads.append(mesh)
    return triads


def run(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"Run dir not found: {run_dir}")

    cams = [c.strip() for c in args.cameras.split(",")] if args.cameras else ALL_CAMS
    extrinsics_yaml = Path(args.extrinsics_yaml).expanduser().resolve()
    extrinsics_realsense_yaml = Path(args.extrinsics_realsense_yaml).expanduser().resolve()

    results = []
    for cam_id in cams:
        frame_dir = camera_frame_dir(run_dir, cam_id, args.frame)
        if frame_dir is None:
            print(f"[{cam_id}] no frame_{args.frame:02d} in {run_dir} -- skipping")
            continue

        offline_dir = frame_dir / OFFLINE_SUBDIR
        cloud = load_objects_cloud(offline_dir)
        if cloud is None:
            print(
                f"[{cam_id}] pointcloud_objects_debug.ply missing/empty in {offline_dir} -- "
                f"run run_object_inference_debug.py for this frame first -- skipping"
            )
            continue

        detections = load_detections(offline_dir)
        T_base_cam, _info = resolve_T_base_cam(
            frame_dir,
            use_config_extrinsics=args.use_config_extrinsics,
            extrinsics_yaml=extrinsics_yaml,
            extrinsics_realsense_yaml=extrinsics_realsense_yaml,
        )

        results.append(dict(
            cam_id=cam_id, frame_dir=frame_dir, cloud=cloud,
            detections=detections, T_base_cam=T_base_cam,
        ))

    if not results:
        raise SystemExit("No camera has a usable objects cloud for this frame.")

    # ---- per-camera visualization (camera frame) ----------------------------
    for res in results:
        cam_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=AXIS_LEN_BASE_M
        )
        geoms = [res["cloud"], cam_origin] + object_triads(res["detections"])
        show(
            geoms,
            f"debug: {res['cam_id']} objects (frame {args.frame})",
            args.dry_run,
        )

    # ---- combined multi-camera view (base frame) -----------------------------
    base_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=AXIS_LEN_BASE_M * 1.5
    )
    combined_geoms = [base_origin]
    used_cams: list[str] = []

    for res in results:
        T_base_cam = res["T_base_cam"]
        if T_base_cam is None:
            print(
                f"[{res['cam_id']}] no T_base_cam -- "
                f"{base_cloud_unavailable_reason(res['frame_dir'])} -- "
                f"omitted from combined view"
            )
            continue

        cloud = o3d.geometry.PointCloud(res["cloud"])
        cloud.transform(T_base_cam)
        combined_geoms.append(cloud)

        triad = camera_triad(
            res["frame_dir"],
            use_config_extrinsics=args.use_config_extrinsics,
            extrinsics_yaml=extrinsics_yaml,
            extrinsics_realsense_yaml=extrinsics_realsense_yaml,
        )
        if triad is not None:
            combined_geoms.append(triad)

        combined_geoms += object_triads(res["detections"], T_extra=T_base_cam)
        used_cams.append(res["cam_id"])

    if len(used_cams) == 0:
        print("[!] no camera resolved a base-frame extrinsic -- skipping combined view")
    else:
        show(
            combined_geoms,
            f"debug: all cameras combined (frame {args.frame}) -- {used_cams}",
            args.dry_run,
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument("--run-dir", required=True,
                    help="Output dir shared with run_object_inference_debug.py.")
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--cameras", default=None,
                    help="Comma-separated cam_ids; default zed2i_1,realsense_1,realsense_2.")
    p.add_argument("--dry-run", action="store_true",
                    help="Build geometries and print counts without opening windows.")

    p.add_argument("--use-config-extrinsics", action="store_true",
                    help="Same meaning as in view_pointclouds.py.")
    p.add_argument("--extrinsics-yaml", default="config/camera_extrinsics_base.yaml")
    p.add_argument("--extrinsics-realsense-yaml", default="config/camera_extrinsics_realsense.yaml")

    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
