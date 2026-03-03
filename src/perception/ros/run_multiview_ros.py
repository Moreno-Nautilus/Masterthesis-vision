from __future__ import annotations

import time
from typing import Any, List, Optional, Tuple

import numpy as np
import rclpy

from src.perception.ros.multicam_grabber import MultiCamGrabber, CameraTopics
from src.perception.pipeline import GraspPerceptionPipeline, PipelineConfig
from src.perception.pipeline_multiview import MultiViewRunner, MultiViewConfig
from src.perception.pose_icp import load_cad_as_pointcloud, estimate_pose_icp
from src.perception.viz_plotly import save_scene_plotly_html
from src.calibration.io_extrinsics import load_extrinsics_yaml


CAMERAS = [
    CameraTopics(
        cam_id="zed2i_1",
        depth_topic="/zed2i_1/zed_node/depth/depth_registered",
        info_topic="/zed2i_1/zed_node/depth/depth_registered/camera_info",
    ),
    CameraTopics(
        cam_id="zed2i_2",
        depth_topic="/zed2i_2/zed_node/depth/depth_registered",
        info_topic="/zed2i_2/zed_node/depth/depth_registered/camera_info",
    ),
]


def _pose_candidates_in_priority() -> List[str]:
    return [
        "T_object_to_world",
        "T_world_obj",
        "T_map_obj",
        "T_base_obj",
        "world_T_obj",
        "map_T_obj",
        "base_T_obj",
        "T_w_o",
        "T_m_o",
        "T_b_o",
        "pose_world",
        "pose_in_world",
        "pose_base",
        "pose_in_base",
        "T",
        "pose",
    ]


def _to_matrix4x4(T: Any) -> np.ndarray:
    if T is None:
        raise ValueError("T is None")
    if hasattr(T, "as_matrix"):
        M = np.asarray(T.as_matrix())
    elif hasattr(T, "matrix"):
        M = np.asarray(T.matrix())
    else:
        M = np.asarray(T)
    if M.shape != (4, 4):
        raise ValueError(f"Expected 4x4 transform, got shape={M.shape} type={type(T)}")
    return M


def _transform_points(T: Any, pts_xyz: np.ndarray) -> np.ndarray:
    if pts_xyz.size == 0:
        return pts_xyz
    if hasattr(T, "transform_points"):
        return np.asarray(T.transform_points(pts_xyz))
    M = _to_matrix4x4(T)
    pts_h = np.hstack([pts_xyz, np.ones((pts_xyz.shape[0], 1), dtype=pts_xyz.dtype)])
    return (M @ pts_h.T).T[:, :3]


def _pick_pose_field(o: Any) -> Tuple[Optional[str], Optional[Any]]:
    for name in _pose_candidates_in_priority():
        if hasattr(o, name):
            return name, getattr(o, name)
    return None, None


def _line(a: np.ndarray, b: np.ndarray, n: int = 60) -> np.ndarray:
    a = np.asarray(a, dtype=float).reshape(3,)
    b = np.asarray(b, dtype=float).reshape(3,)
    ts = np.linspace(0.0, 1.0, n)
    return (1 - ts)[:, None] * a[None, :] + ts[:, None] * b[None, :]


def _base_axes(scale: float = 0.35, n: int = 60) -> list[np.ndarray]:
    o = np.zeros(3)
    return [
        _line(o, np.array([scale, 0, 0]), n),
        _line(o, np.array([0, scale, 0]), n),
        _line(o, np.array([0, 0, scale]), n),
    ]


def _frame_lines(T: Any, scale: float = 0.20, n: int = 60) -> list[np.ndarray]:
    M = _to_matrix4x4(T)
    R = M[:3, :3]
    o = M[:3, 3]
    x = o + scale * R[:, 0]
    y = o + scale * R[:, 1]
    z = o + scale * R[:, 2]
    return [_line(o, x, n), _line(o, y, n), _line(o, z, n)]


def _plane_normal_line(nrm: np.ndarray, p0: np.ndarray, scale: float = 0.25, n: int = 80) -> np.ndarray:
    nrm = np.asarray(nrm, dtype=float).reshape(3,)
    p0 = np.asarray(p0, dtype=float).reshape(3,)
    p1 = p0 + scale * nrm
    return _line(p0, p1, n)


def _bbox_diag(pts: np.ndarray) -> float:
    if pts is None or len(pts) == 0:
        return float("inf")
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    ext = maxs - mins
    return float(np.linalg.norm(ext))


def main() -> None:
    rclpy.init()

    cad_library = {
        "cube": load_cad_as_pointcloud("Data/CAD_Models/Cube.stl", scale=0.003822, center=True),
    }

    cad = cad_library["cube"]
    cad_ext = cad.max(axis=0) - cad.min(axis=0)
    cad_diag = float(np.linalg.norm(cad_ext))
    print("[CAD] cube extent [m]:", cad_ext, "diag:", cad_diag)

    pipe_cfg = PipelineConfig(
        plane_distance_threshold=0.002,
        dbscan_eps=0.03,
        dbscan_min_points=40,
        voxel_size=0.005,
        max_rms_nn=0.012,
        min_margin=1.5,
    )
    pipe = GraspPerceptionPipeline(cad_library=cad_library, cfg=pipe_cfg)

    mv_cfg = MultiViewConfig(
        voxel_size_fusion=0.003,
        stride=1,
        zmin=0.25,
        zmax=1.6,
        roi_x_min=-0.7,
        roi_x_max=0.7,
        roi_y_min=-0.7,
        roi_y_max=0.7,
        roi_z_min=0.25,
        roi_z_max=1.6,
    )
    runner = MultiViewRunner(pipe, cfg=mv_cfg)

    T_map = load_extrinsics_yaml("config/camera_extrinsics.yaml")
    grabber = MultiCamGrabber(
        cameras=CAMERAS,
        sync_slop_s=0.05,
        use_best_effort_if_unsynced=False,
        static_extrinsics_base_cam=T_map,
    )

    try:
        print("Waiting for synced multi-view set...")
        views = None
        t0 = time.time()
        while views is None:
            rclpy.spin_once(grabber, timeout_sec=0.1)
            views = grabber.get_latest_views()
            if time.time() - t0 > 10.0 and views is None:
                print("Still waiting... (check camera topics / running nodes)")
                t0 = time.time()

        print("Got synced views:", [(v.cam_id, f"{v.stamp_s:.3f}") for v in views])

        # debug geometry (axes + camera frames)
        debug_clusters: List[np.ndarray] = []
        debug_clusters += _base_axes(scale=0.35)

        cam_origins = []
        for v in views:
            debug_clusters += _frame_lines(v.T_base_cam, scale=0.20)
            cam_origins.append(v.T_base_cam.transform_point(np.zeros(3)))
        if cam_origins:
            debug_clusters.append(np.asarray(cam_origins, dtype=float))

        result = runner.run(views)
        raw = result.points_world_raw

        print("[RAW] n=", len(raw))
        mins = raw.min(axis=0); maxs = raw.max(axis=0)
        ext = maxs - mins
        print("[RAW] mins:", mins, "maxs:", maxs)
        print("[RAW] extent XYZ [m]:", ext)

        # --- Choose plane normal sign by maximizing above-band count ---
        # plane_model may not be normalized
        n0 = np.array(result.plane_model[:3], dtype=float)
        d0 = float(result.plane_model[3])
        nn = np.linalg.norm(n0) + 1e-12
        n0 = n0 / nn
        d0 = d0 / nn

        h0 = raw @ n0 + d0
        h1 = -h0  # corresponds to flipping normal & d

        h_min, h_max = 0.01, 0.12
        n_above0 = int(np.count_nonzero((h0 > h_min) & (h0 < h_max)))
        n_above1 = int(np.count_nonzero((h1 > h_min) & (h1 < h_max)))

        if n_above1 > n_above0:
            n = -n0
            d = -d0
            h = h1
        else:
            n = n0
            d = d0
            h = h0

        print(f"[UP_SIGN] above_count normal={n_above0} flipped={n_above1} -> using {'flipped' if n_above1>n_above0 else 'normal'}")
        print("[H] min/med/max:", float(h.min()), float(np.median(h)), float(h.max()))
        print("[H] quantiles 50/90/95/99%:", np.quantile(h, [0.5, 0.9, 0.95, 0.99]))

        # Plane centroid marker + normal arrow
        plane_centroid = np.mean(result.plane_points, axis=0) if result.plane_points is not None and len(result.plane_points) else np.zeros(3)
        debug_clusters.append(np.asarray([plane_centroid], dtype=float))
        debug_clusters.append(_plane_normal_line(n, plane_centroid, scale=0.25))

        # Above-band points
        mask = (h > h_min) & (h < h_max)
        above = raw[mask]
        print("[ABOVE] n=", len(above))

        save_scene_plotly_html(
            scene=raw,
            plane=result.plane_points,
            clusters=debug_clusters + [above],
            objects_world=None,
            out_path="/home/moreno/MasterThesis/multiview_above_plane_debug.html",
        )
        print("Saved /home/moreno/MasterThesis/multiview_above_plane_debug.html")

        # -------- Pick a cluster for ICP by CAD size prior --------
        icp_target = None
        clusters = result.clusters or []
        if clusters:
            diags = [(_bbox_diag(c), i, c.shape[0]) for i, c in enumerate(clusters)]
            # choose cluster whose bbox diag is closest to CAD diag, but ignore tiny clusters
            cand = []
            for diag, i, sz in diags:
                if sz < 80:
                    continue
                cand.append((abs(diag - cad_diag), diag, i, sz))
            cand.sort(key=lambda x: x[0])
            if cand:
                _, diag, i, sz = cand[0]
                icp_target = clusters[i]
                print(f"[ICP_PICK] picked cluster={i} size={sz} bbox_diag={diag:.3f} (cad_diag={cad_diag:.3f})")
            else:
                icp_target = max(clusters, key=lambda c: c.shape[0])
                print("[ICP_PICK] fallback largest cluster:", icp_target.shape[0])
        else:
            icp_target = above
            print("[ICP_PICK] no clusters, using ABOVE:", icp_target.shape[0])

        # Always-on ICP overlay
        objects_world_icp: List[Tuple[str, np.ndarray]] = []
        if icp_target is not None and icp_target.shape[0] >= 80:
            T_obs_obj, metrics = estimate_pose_icp(
                observed_points=icp_target,
                cad_points=cad_library["cube"],
                voxel_size=0.005,
            )
            M = _to_matrix4x4(T_obs_obj)
            pts_as_is = _transform_points(M, cad_library["cube"])
            pts_inv = _transform_points(np.linalg.inv(M), cad_library["cube"])
            objects_world_icp.append(("cube_icp_as_is", pts_as_is))
            objects_world_icp.append(("cube_icp_inv", pts_inv))
            print("[ICP_DEBUG] metrics:", metrics)
            print("[ICP_DEBUG] t_as_is:", M[:3, 3], "|t|=", float(np.linalg.norm(M[:3, 3])))

        out_ws = "/home/moreno/MasterThesis/multiview_workspace_with_icp.html"
        save_scene_plotly_html(
            scene=result.points_world_raw,
            plane=result.plane_points,
            clusters=debug_clusters + ([icp_target] if icp_target is not None else []) + (clusters if clusters else []),
            objects_world=objects_world_icp,
            out_path=out_ws,
        )
        print("Saved workspace ICP debug:", out_ws)

        # Detection plot (if pipeline produced objects)
        objects_world: List[Tuple[str, np.ndarray]] = []
        if getattr(result, "objects", None):
            for i, o in enumerate(result.objects):
                obj_id = getattr(o, "object_id", f"obj{i}")
                if obj_id not in cad_library:
                    continue
                pose_name, T = _pick_pose_field(o)
                if T is None:
                    continue
                M = _to_matrix4x4(T)
                objects_world.append((f"{obj_id}_as_is", _transform_points(M, cad_library[obj_id])))
                objects_world.append((f"{obj_id}_inv", _transform_points(np.linalg.inv(M), cad_library[obj_id])))

        out_det = "/home/moreno/MasterThesis/multiview_detection_debug.html"
        save_scene_plotly_html(
            scene=result.points_world_raw,
            plane=result.plane_points,
            clusters=debug_clusters + (result.clusters or []),
            objects_world=objects_world,
            out_path=out_det,
        )
        print(f"Saved detection debug: {out_det}")

    finally:
        grabber.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()