"""
Interactive Open3D viewer for a completed tools/bagviz/capture_pipeline_snapshots.py
run: shows the colored point cloud(s) + coordinate-frame triads for one sampled
frame index, in several steps:

  1. zed2i_1 alone, in the shared base frame.
  2. Every available camera in the shared base frame.
  3. Sequential ICP alignment of all available clouds, starting from identity
     for every ICP registration, followed by a fused point cloud.

ICP is performed on point clouds downsampled by 4 for speed. The resulting
transforms are printed and then applied to the original-resolution clouds for
the final fused cloud.

Reads only what capture_pipeline_snapshots.py already saved for the run --
never reopens the original bag. Camera extrinsics are loaded from config YAML
files, allowing offline tuning without re-running the capture.

Usage:
    conda activate bagviz
    python -m tools.bagviz.view_pointclouds --run-dir outputs/bagviz/<run>
    python -m tools.bagviz.view_pointclouds --run-dir outputs/bagviz/<run> --frame 1
    python -m tools.bagviz.view_pointclouds --run-dir outputs/bagviz/<run> --dry-run
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d
import yaml

try:
    from src.calibration.io_extrinsics import load_extrinsics_yaml
except Exception:
    load_extrinsics_yaml = None


AXIS_LEN_BASE_M = 0.10
AXIS_LEN_OBJECT_M = 0.05

# ICP parameters.
#
# The clouds are downsampled by a factor of 4 using every 4th point.
# The correspondence threshold therefore needs to be large enough to
# accommodate the remaining nominal calibration error between cameras.
ICP_MAX_CORRESPONDENCE_DISTANCE_M = 0.05

# Maximum ICP iterations per scale.
ICP_MAX_ITERATIONS = 100

# Camera configuration
#
# Eye-in-hand RealSense cameras: which robot_base_key and which flange
# pose topic they correspond to (for extracting flange pose from captured data).
DYNAMIC_CAM_CONFIG = {
    "realsense_1": dict(robot_base_key="robot_a", flange_pose_topic="/left/ee_pose"),
    "realsense_2": dict(robot_base_key="robot_b", flange_pose_topic="/right/ee_pose"),
}

STATIC_CAM_CONFIG = ["zed2i_1"]


_warned_no_flange_cams: set[str] = set()


def compute_T_base_cam_from_config(
    cam_id: str,
    frame_info: dict,
    extrinsics_yaml: Path,
    extrinsics_realsense_yaml: Optional[Path] = None,
) -> Optional[np.ndarray]:
    """
    Compute T_base_cam by loading camera extrinsics from config files and
    (for RealSense) recomposing with the flange pose captured for this frame.

    For static cameras (ZED): directly loads T_base_cam from extrinsics_yaml.
    For RealSense: recomposes T_base_cam = T_base_flange @ T_flange_cam(config),
    using the T_base_flange saved in frame_info.yaml at capture time and the
    (possibly just-edited) camera-to-flange offset from
    extrinsics_realsense_yaml. This is what makes offline tuning of the
    RealSense camera-to-flange offset actually visible.

    Args:
        cam_id: Camera identifier
        frame_info: Loaded frame_info.yaml dict (T_base_cam / T_base_flange)
        extrinsics_yaml: Path to camera extrinsics YAML (static cameras)
        extrinsics_realsense_yaml: Path to RealSense extrinsics YAML

    Returns:
        4x4 transformation matrix, or None if not resolvable.
    """
    if load_extrinsics_yaml is None:
        return None

    # Load extrinsics from config files
    if extrinsics_realsense_yaml is not None:
        extr_dict = load_extrinsics_yaml(extrinsics_realsense_yaml)
    else:
        extr_dict = load_extrinsics_yaml(extrinsics_yaml)

    if cam_id not in extr_dict:
        return None

    if cam_id in STATIC_CAM_CONFIG:
        # Static camera: directly convert SE3 to 4x4 matrix
        se3 = extr_dict[cam_id]
        return se3.as_matrix()

    if cam_id in DYNAMIC_CAM_CONFIG:
        # Eye-in-hand RealSense: T_base_cam = T_base_flange @ T_flange_cam.
        # T_base_flange is the flange pose that was live in the bag at
        # capture time (fixed, can't be re-derived from anything else this
        # tool has offline). T_flange_cam is the tunable camera-to-flange
        # offset, read fresh from extrinsics_realsense_yaml every call.
        T_flange_cam = extr_dict[cam_id].as_matrix()

        T_base_flange = frame_info.get("T_base_flange")

        if T_base_flange is not None:
            T_base_flange = np.asarray(T_base_flange, dtype=float)
            return T_base_flange @ T_flange_cam

        # Older captures (before T_base_flange was logged) can't recompose:
        # fall back to the T_base_cam baked in at capture time and warn once
        # per camera that edits to extrinsics_realsense_yaml won't show up.
        T_base_cam_captured = frame_info.get("T_base_cam")

        if T_base_cam_captured is None:
            return None

        if cam_id not in _warned_no_flange_cams:
            _warned_no_flange_cams.add(cam_id)
            print(
                f"[!] {cam_id}: frame_info.yaml has no T_base_flange "
                "(this capture predates that field) -- "
                "--use-config-extrinsics can't recompose this camera's "
                "pose, falling back to the T_base_cam baked in at capture "
                "time. Edits to extrinsics_realsense_yaml will NOT be "
                "reflected for this camera. Re-run "
                "capture_pipeline_snapshots.py to pick up T_base_flange "
                "logging and enable offline tuning."
            )

        return np.asarray(T_base_cam_captured, dtype=float)

    return None


def resolve_T_base_cam(
    frame_dir: Path,
    use_config_extrinsics: bool = False,
    extrinsics_yaml: Optional[Path] = None,
    extrinsics_realsense_yaml: Optional[Path] = None,
) -> tuple[Optional[np.ndarray], dict]:
    """
    Resolve T_base_cam for a frame, either from frame_info.yaml (default)
    or recomputed from the extrinsics config YAMLs (--use-config-extrinsics).

    Returns (T_base_cam or None, frame_info dict).
    """
    info_path = frame_dir / "frame_info.yaml"

    if not info_path.exists():
        return None, {}

    info = yaml.safe_load(info_path.read_text()) or {}

    if use_config_extrinsics and extrinsics_yaml is not None:
        cam_id = info.get("cam_id")

        if cam_id is None:
            return None, info

        T_base_cam = compute_T_base_cam_from_config(
            cam_id,
            info,
            extrinsics_yaml,
            extrinsics_realsense_yaml,
        )
    else:
        T_base_cam = info.get("T_base_cam")

        if T_base_cam is not None:
            T_base_cam = np.asarray(T_base_cam, dtype=float)

    return T_base_cam, info


def quat_xyzw_to_R(q: list[float]) -> np.ndarray:
    x, y, z, w = q
    n = x * x + y * y + z * z + w * w
    s = 2.0 / n if n > 0 else 0.0

    return np.array([
        [1 - s * (y * y + z * z),
         s * (x * y - w * z),
         s * (x * z + w * y)],

        [s * (x * y + w * z),
         1 - s * (x * x + z * z),
         s * (y * z - w * x)],

        [s * (x * z - w * y),
         s * (y * z + w * x),
         1 - s * (x * x + y * y)],
    ])


def pose_dict_to_T(d: dict) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = quat_xyzw_to_R(d["quat_xyzw"])
    T[:3, 3] = d["t"]
    return T


def camera_frame_dir(
    run_dir: Path,
    cam_id: str,
    frame: int,
) -> Optional[Path]:
    d = run_dir / cam_id / f"frame_{frame:02d}"
    return d if d.is_dir() else None


def load_base_cloud(
    frame_dir: Path,
) -> Optional[o3d.geometry.PointCloud]:
    """Load the cloud pre-baked into base frame at capture time."""
    ply = frame_dir / "pointcloud_raw_base.ply"

    if not ply.exists():
        return None

    pc = o3d.io.read_point_cloud(str(ply))

    return pc if len(pc.points) > 0 else None


def load_raw_cam_cloud(
    frame_dir: Path,
) -> Optional[o3d.geometry.PointCloud]:
    """Load the cloud in camera frame (before any base-frame transform)."""
    ply = frame_dir / "pointcloud_raw.ply"

    if not ply.exists():
        return None

    pc = o3d.io.read_point_cloud(str(ply))

    return pc if len(pc.points) > 0 else None


def load_cloud_in_base_frame(
    frame_dir: Path,
    use_config_extrinsics: bool = False,
    extrinsics_yaml: Optional[Path] = None,
    extrinsics_realsense_yaml: Optional[Path] = None,
) -> Optional[o3d.geometry.PointCloud]:
    """
    Load this frame's point cloud expressed in the base frame.

    Default (use_config_extrinsics=False): loads pointcloud_raw_base.ply,
    which was transformed into base frame once, at capture time, using
    whatever extrinsics were active then.

    use_config_extrinsics=True: loads the camera-frame pointcloud_raw.ply
    and transforms it at view time using T_base_cam recomputed from the
    current extrinsics_yaml / extrinsics_realsense_yaml -- so edits to
    those YAML files are actually reflected in the cloud, not just in the
    camera triad.
    """
    if not use_config_extrinsics:
        return load_base_cloud(frame_dir)

    T_base_cam, _info = resolve_T_base_cam(
        frame_dir,
        use_config_extrinsics=True,
        extrinsics_yaml=extrinsics_yaml,
        extrinsics_realsense_yaml=extrinsics_realsense_yaml,
    )

    if T_base_cam is None:
        return None

    pc = load_raw_cam_cloud(frame_dir)

    if pc is None:
        return None

    pc.transform(T_base_cam)

    return pc


def object_triads(
    frame_dir: Path,
) -> list[o3d.geometry.TriangleMesh]:
    poses_yaml = frame_dir / "poses.yaml"

    if not poses_yaml.exists():
        return []

    data = yaml.safe_load(poses_yaml.read_text()) or {}

    triads = []

    for item in data.get("pose_items", []):
        if "pose_base" not in item:
            continue

        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=AXIS_LEN_OBJECT_M
        )

        mesh.transform(pose_dict_to_T(item["pose_base"]))
        triads.append(mesh)

    return triads


def camera_triad(
    frame_dir: Path,
    use_config_extrinsics: bool = False,
    extrinsics_yaml: Optional[Path] = None,
    extrinsics_realsense_yaml: Optional[Path] = None,
) -> Optional[o3d.geometry.TriangleMesh]:
    """
    Create a coordinate-frame mesh for the camera's pose in the base frame.

    Args:
        frame_dir: Frame directory containing frame_info.yaml
        use_config_extrinsics: If True, load T_base_cam from config files
                               instead of frame_info.yaml
        extrinsics_yaml: Path to static camera extrinsics YAML
        extrinsics_realsense_yaml: Path to RealSense camera extrinsics YAML
    """
    T_base_cam, _info = resolve_T_base_cam(
        frame_dir,
        use_config_extrinsics=use_config_extrinsics,
        extrinsics_yaml=extrinsics_yaml,
        extrinsics_realsense_yaml=extrinsics_realsense_yaml,
    )

    if T_base_cam is None:
        return None

    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=AXIS_LEN_BASE_M
    )

    mesh.transform(np.asarray(T_base_cam, dtype=float))

    return mesh


def base_cloud_unavailable_reason(frame_dir: Path) -> str:
    """Why pointcloud_raw_base.ply is missing/empty for this frame."""

    info_path = frame_dir / "frame_info.yaml"

    if not info_path.exists():
        return "frame_info.yaml is missing."

    info = yaml.safe_load(info_path.read_text()) or {}

    notes = info.get("notes", [])

    if info.get("T_base_cam") is None:
        for note in notes:
            if (
                "base" in note.lower()
                and (
                    "skip" in note.lower()
                    or "no base extrinsics" in note.lower()
                )
            ):
                return note

        return (
            "T_base_cam not resolved for this frame "
            "(see frame_info.yaml notes)."
        )

    for note in notes:
        low = note.lower()

        if "depth" in low or "native-resolution rgb" in low:
            return note

    return (
        "pointcloud_raw_base.ply missing for this frame "
        "(see frame_info.yaml notes) -- T_base_cam was resolved fine, "
        "so this isn't an extrinsics problem."
    )


def show(
    geoms: list,
    window_name: str,
    dry_run: bool,
) -> None:
    n_pts = sum(
        len(g.points)
        for g in geoms
        if isinstance(g, o3d.geometry.PointCloud)
    )

    n_triads = sum(
        1
        for g in geoms
        if isinstance(g, o3d.geometry.TriangleMesh)
    )

    print(
        f"[*] {window_name}: "
        f"{n_pts} points, {n_triads} coordinate triads"
    )

    if dry_run:
        return

    o3d.visualization.draw_geometries(
        geoms,
        window_name=window_name,
    )


def filter_noise(
    cloud: o3d.geometry.PointCloud,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
) -> o3d.geometry.PointCloud:
    """
    Fast statistical outlier removal to reject noise.

    Removes points that are statistical outliers based on their distance
    to neighbors. This is a quick pre-filter to clean up noisy depth data.

    Args:
        cloud: Input point cloud
        nb_neighbors: Number of neighbors to consider for each point
        std_ratio: Standard deviation ratio threshold (higher = less aggressive)

    Returns:
        Filtered point cloud with noise removed
    """
    if len(cloud.points) == 0:
        return cloud

    try:
        filtered, _ = cloud.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio,
        )
        return filtered
    except Exception:
        # If filtering fails, return original cloud
        return cloud


def downsample_by_4(
    cloud: o3d.geometry.PointCloud,
) -> o3d.geometry.PointCloud:
    """
    Downsample by keeping every fourth point.

    This is intentionally a point-count reduction rather than voxel
    downsampling because the request is specifically to downsample
    by 4 and it avoids having to choose a scene-dependent voxel size.
    """

    points = np.asarray(cloud.points)

    if len(points) <= 4:
        return cloud

    indices = np.arange(0, len(points), 4)

    result = o3d.geometry.PointCloud()

    result.points = o3d.utility.Vector3dVector(
        points[indices]
    )

    if cloud.has_colors():
        colors = np.asarray(cloud.colors)

        if len(colors) == len(points):
            result.colors = o3d.utility.Vector3dVector(
                colors[indices]
            )

    if cloud.has_normals():
        normals = np.asarray(cloud.normals)

        if len(normals) == len(points):
            result.normals = o3d.utility.Vector3dVector(
                normals[indices]
            )

    return result


def icp_align(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    source_name: str,
    target_name: str,
) -> np.ndarray:
    """
    Align source -> target using point-to-point ICP.

    Initialization is explicitly identity, as requested.
    """

    source_ds = downsample_by_4(source)
    target_ds = downsample_by_4(target)

    print()
    print(
        f"[*] ICP: {source_name} -> {target_name}"
    )
    print(
        f"    source: {len(source.points)} -> "
        f"{len(source_ds.points)} points"
    )
    print(
        f"    target: {len(target.points)} -> "
        f"{len(target_ds.points)} points"
    )
    print(
        f"    max correspondence distance: "
        f"{ICP_MAX_CORRESPONDENCE_DISTANCE_M:.4f} m"
    )
    print("    initial transform: identity")

    if len(source_ds.points) == 0 or len(target_ds.points) == 0:
        print("    [!] empty source/target -- returning identity")
        return np.eye(4)

    result = o3d.pipelines.registration.registration_icp(
        source_ds,
        target_ds,
        ICP_MAX_CORRESPONDENCE_DISTANCE_M,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=ICP_MAX_ITERATIONS,
        ),
    )

    print(
        f"    fitness:          {result.fitness:.6f}"
    )
    print(
        f"    inlier RMSE:      {result.inlier_rmse:.6f} m"
    )
    print("    resulting transform:")
    print(result.transformation)

    return result.transformation


def print_transform(
    name: str,
    T: np.ndarray,
) -> None:
    print()
    print(f"========== {name} ==========")
    print(T)
    print("============================")


def build_icp_fused_cloud(
    clouds: list[tuple[str, o3d.geometry.PointCloud]],
) -> tuple[
    o3d.geometry.PointCloud,
    dict[str, np.ndarray],
]:
    """
    Sequentially ICP-align all clouds.

    The first cloud is the reference.

    Every subsequent cloud is registered against the cloud that has
    already been accumulated into the reference frame. Importantly,
    every ICP registration starts from identity.

    Returns:
        fused_cloud:
            Original-resolution fused cloud.
        transforms:
            Transform for every camera that was aligned.
    """

    if not clouds:
        return o3d.geometry.PointCloud(), {}

    reference_name, reference_cloud = clouds[0]

    print()
    print("============================================================")
    print("                    SEQUENTIAL ICP")
    print("============================================================")
    print(
        f"Reference cloud: {reference_name}"
    )
    print(
        f"ICP downsampling: every 4th point"
    )
    print(
        f"ICP initialization: identity for every registration"
    )
    print("============================================================")

    transforms: dict[str, np.ndarray] = {
        reference_name: np.eye(4)
    }

    # Start with the first cloud as the current reference.
    #
    # We maintain a downsampled accumulated cloud for ICP so that the
    # registration remains fast even after multiple cameras are added.
    accumulated_ds = downsample_by_4(reference_cloud)

    # Final fused cloud retains original resolution.
    fused = o3d.geometry.PointCloud(reference_cloud)

    print_transform(
        f"{reference_name} -> reference",
        np.eye(4),
    )

    for source_name, source_cloud in clouds[1:]:
        # ICP source is registered against the accumulated reference.
        #
        # Both clouds are downsampled inside icp_align(). The source
        # transform is always initialized with identity.
        T = icp_align(
            source_cloud,
            accumulated_ds,
            source_name,
            f"fused({reference_name}, previous cameras)",
        )

        transforms[source_name] = T

        print_transform(
            f"{source_name} -> reference",
            T,
        )

        # Apply the resulting transform to the original-resolution
        # source cloud before adding it to the fused cloud.
        aligned_original = o3d.geometry.PointCloud(source_cloud)
        aligned_original.transform(T)

        fused += aligned_original

        # Add the transformed source to the accumulated ICP reference.
        source_ds = downsample_by_4(source_cloud)
        source_ds.transform(T)

        accumulated_ds += source_ds

        # Keep the accumulated ICP cloud bounded in size.
        #
        # This is only the registration cloud; the final fused cloud
        # above remains at original resolution.
        accumulated_ds = downsample_by_4(accumulated_ds)

    print()
    print("============================================================")
    print("                    ICP RESULTS")
    print("============================================================")

    for name, T in transforms.items():
        print_transform(
            f"{name} -> reference",
            T,
        )

    print()
    print(
        f"[*] Final fused cloud: {len(fused.points)} points"
    )

    return fused, transforms


def show_camera_subset(
    camera_ids: list[str],
    frame_dirs: dict[str, Path],
    all_clouds: dict[str, o3d.geometry.PointCloud],
    base_origin: o3d.geometry.TriangleMesh,
    args: argparse.Namespace,
    extrinsics_yaml: Path,
    extrinsics_realsense_yaml: Path,
) -> tuple[
    list[tuple[str, o3d.geometry.PointCloud]],
    list[str],
]:
    """
    Show a subset of cameras and return their clouds for later ICP.

    Reuses the already-loaded/filtered clouds in all_clouds (built once,
    respecting --use-config-extrinsics) instead of reloading from disk.

    Returns:
        (clouds, used_camera_ids)
    """
    geoms = [base_origin]
    clouds: list[tuple[str, o3d.geometry.PointCloud]] = []
    used_cams: list[str] = []

    for cam_id in camera_ids:
        if cam_id not in frame_dirs or cam_id not in all_clouds:
            continue

        frame_dir = frame_dirs[cam_id]
        cloud = all_clouds[cam_id]

        triad = camera_triad(
            frame_dir,
            use_config_extrinsics=args.use_config_extrinsics,
            extrinsics_yaml=extrinsics_yaml,
            extrinsics_realsense_yaml=extrinsics_realsense_yaml,
        )

        geoms.append(cloud)

        if triad is not None:
            geoms.append(triad)

        used_cams.append(cam_id)
        clouds.append((cam_id, cloud))

    # Add object triads from first available camera
    for cam_id in used_cams:
        triads = object_triads(frame_dirs[cam_id])
        if triads:
            geoms += triads
            break

    if len(used_cams) > 0:
        show(
            geoms,
            f"bagviz: {' + '.join(used_cams)} (frame {args.frame})",
            args.dry_run,
        )

    return clouds, used_cams


def run(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir).expanduser().resolve()

    if not run_dir.is_dir():
        raise SystemExit(
            f"Run dir not found: {run_dir}"
        )

    extrinsics_yaml = Path(args.extrinsics_yaml).expanduser().resolve()
    extrinsics_realsense_yaml = (
        Path(args.extrinsics_realsense_yaml).expanduser().resolve()
        if args.extrinsics_realsense_yaml
        else None
    )

    all_cams = [
        "zed2i_1",
        "realsense_1",
        "realsense_2",
    ]

    base_origin = (
        o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=AXIS_LEN_BASE_M * 1.5
        )
    )

    # ------------------------------------------------------------------
    # Step 1: ZED alone
    # ------------------------------------------------------------------

    zed_dir = camera_frame_dir(
        run_dir,
        "zed2i_1",
        args.frame,
    )

    if zed_dir is None:
        raise SystemExit(
            f"No zed2i_1/frame_{args.frame:02d} in {run_dir} -- "
            f"re-run capture_pipeline_snapshots.py with more "
            f"--num-frames or check --cameras."
        )

    zed_cloud = load_cloud_in_base_frame(
        zed_dir,
        use_config_extrinsics=args.use_config_extrinsics,
        extrinsics_yaml=extrinsics_yaml,
        extrinsics_realsense_yaml=extrinsics_realsense_yaml,
    )

    if zed_cloud is None:
        raise SystemExit(
            f"{zed_dir}/pointcloud_raw_base.ply missing/empty -- "
            f"{base_cloud_unavailable_reason(zed_dir)}"
        )

    # Apply noise filtering
    zed_cloud = filter_noise(zed_cloud)

    geoms = [
        zed_cloud,
        base_origin,
    ]

    zed_triad = camera_triad(
        zed_dir,
        use_config_extrinsics=args.use_config_extrinsics,
        extrinsics_yaml=extrinsics_yaml,
        extrinsics_realsense_yaml=extrinsics_realsense_yaml,
    )

    if zed_triad is not None:
        geoms.append(zed_triad)

    geoms += object_triads(zed_dir)

    show(
        geoms,
        f"bagviz: zed2i_1 only (frame {args.frame})",
        args.dry_run,
    )

    # ------------------------------------------------------------------
    # Build frame_dirs and clouds for all available cameras
    # ------------------------------------------------------------------

    frame_dirs: dict[str, Path] = {}
    all_clouds: dict[str, o3d.geometry.PointCloud] = {}

    for cam_id in all_cams:
        frame_dir = camera_frame_dir(
            run_dir,
            cam_id,
            args.frame,
        )

        if frame_dir is None:
            print(
                f"[*] {cam_id}: no frame_{args.frame:02d} "
                f"in this run, skipping"
            )
            continue

        cloud = load_cloud_in_base_frame(
            frame_dir,
            use_config_extrinsics=args.use_config_extrinsics,
            extrinsics_yaml=extrinsics_yaml,
            extrinsics_realsense_yaml=extrinsics_realsense_yaml,
        )

        if cloud is None:
            print(
                f"[*] {cam_id}: skipped -- "
                f"{base_cloud_unavailable_reason(frame_dir)}"
            )
            continue

        # Apply noise filtering
        cloud = filter_noise(cloud)

        frame_dirs[cam_id] = frame_dir
        all_clouds[cam_id] = cloud

    # ------------------------------------------------------------------
    # Step 2: RealSense1 alone
    # ------------------------------------------------------------------

    clouds_rs1, _ = show_camera_subset(
        ["realsense_1"],
        frame_dirs,
        all_clouds,
        base_origin,
        args,
        extrinsics_yaml,
        extrinsics_realsense_yaml,
    )

    # ------------------------------------------------------------------
    # Step 3: RealSense2 alone
    # ------------------------------------------------------------------

    clouds_rs2, _ = show_camera_subset(
        ["realsense_2"],
        frame_dirs,
        all_clouds,
        base_origin,
        args,
        extrinsics_yaml,
        extrinsics_realsense_yaml,
    )

    # ------------------------------------------------------------------
    # Step 4: ZED + RealSense1
    # ------------------------------------------------------------------

    clouds_zed_rs1, _ = show_camera_subset(
        ["zed2i_1", "realsense_1"],
        frame_dirs,
        all_clouds,
        base_origin,
        args,
        extrinsics_yaml,
        extrinsics_realsense_yaml,
    )

    # ------------------------------------------------------------------
    # Step 5: ZED + RealSense2
    # ------------------------------------------------------------------

    clouds_zed_rs2, _ = show_camera_subset(
        ["zed2i_1", "realsense_2"],
        frame_dirs,
        all_clouds,
        base_origin,
        args,
        extrinsics_yaml,
        extrinsics_realsense_yaml,
    )

    # ------------------------------------------------------------------
    # Step 6: RealSense1 + RealSense2
    # ------------------------------------------------------------------

    clouds_rs1_rs2, _ = show_camera_subset(
        ["realsense_1", "realsense_2"],
        frame_dirs,
        all_clouds,
        base_origin,
        args,
        extrinsics_yaml,
        extrinsics_realsense_yaml,
    )

    # ------------------------------------------------------------------
    # Step 7: All cameras in nominal base frame
    # ------------------------------------------------------------------

    geoms = [base_origin]
    used_cams: list[str] = []

    # Keep original-resolution clouds for final ICP/fusion.
    clouds: list[
        tuple[str, o3d.geometry.PointCloud]
    ] = []

    for cam_id in all_cams:
        if cam_id not in frame_dirs:
            continue

        frame_dir = frame_dirs[cam_id]
        cloud = all_clouds[cam_id]

        triad = camera_triad(
            frame_dir,
            use_config_extrinsics=args.use_config_extrinsics,
            extrinsics_yaml=extrinsics_yaml,
            extrinsics_realsense_yaml=extrinsics_realsense_yaml,
        )

        geoms.append(cloud)

        if triad is not None:
            geoms.append(triad)

        used_cams.append(cam_id)
        clouds.append((cam_id, cloud))

    # Object triads:
    #
    # Prefer ZED's pose set, otherwise use the first available camera
    # with pose_base entries.
    added_objects = False

    for cam_id in used_cams:
        frame_dir = frame_dirs[cam_id]

        triads = object_triads(frame_dir)

        if triads:
            geoms += triads
            added_objects = True
            break

    if not added_objects:
        print(
            "[*] no camera in this frame had pose_base "
            "object triads to draw"
        )

    if len(used_cams) <= 1:
        print(
            f"[!] only {used_cams} contributed a base-frame cloud -- "
            "'combined' view is not much of a combination this frame."
        )

    show(
        geoms,
        f"bagviz: all cameras combined "
        f"(frame {args.frame}) -- {used_cams}",
        args.dry_run,
    )

    # ------------------------------------------------------------------
    # Step 8: sequential ICP + fused point cloud
    # ------------------------------------------------------------------

    if len(clouds) < 2:
        print()
        print(
            "[!] Need at least two valid camera clouds for ICP. "
            "Skipping ICP/fusion."
        )
        return

    fused_cloud, icp_transforms = build_icp_fused_cloud(
        clouds
    )

    # ------------------------------------------------------------------
    # Step 9: show ICP-fused cloud
    # ------------------------------------------------------------------

    fused_geoms = [
        fused_cloud,
        base_origin,
    ]

    # Add the original camera triads.
    #
    # These represent the nominal camera poses from the capture
    # pipeline. We intentionally do not transform them by ICP because
    # the ICP result describes a point-cloud correction, not a corrected
    # physical camera extrinsic.
    for cam_id in used_cams:
        triad = camera_triad(
            frame_dirs[cam_id],
            use_config_extrinsics=args.use_config_extrinsics,
            extrinsics_yaml=extrinsics_yaml,
            extrinsics_realsense_yaml=extrinsics_realsense_yaml,
        )

        if triad is not None:
            fused_geoms.append(triad)

    # Object triads.
    for cam_id in used_cams:
        triads = object_triads(frame_dirs[cam_id])

        if triads:
            fused_geoms += triads
            break

    show(
        fused_geoms,
        f"bagviz: ICP fused clouds "
        f"(frame {args.frame}) -- {used_cams}",
        args.dry_run,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument(
        "--run-dir",
        required=True,
        help="Output dir from capture_pipeline_snapshots.py.",
    )

    p.add_argument(
        "--frame",
        type=int,
        default=0,
        help="Frame index to view (frame_00, frame_01, ...).",
    )

    p.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Build geometries and perform ICP while printing "
            "counts/transforms, without opening windows."
        ),
    )

    p.add_argument(
        "--use-config-extrinsics",
        action="store_true",
        help=(
            "Load camera extrinsics from YAML config files instead of "
            "from frame_info.yaml. Allows offline tuning of camera "
            "parameters without re-running capture_pipeline_snapshots.py."
        ),
    )

    p.add_argument(
        "--extrinsics-yaml",
        default="config/camera_extrinsics_realsense.yaml",
        help="Path to static camera extrinsics YAML.",
    )

    p.add_argument(
        "--extrinsics-realsense-yaml",
        default="config/camera_extrinsics_realsense.yaml",
        help=(
            "Path to RealSense camera extrinsics YAML "
            "(camera-to-flange offsets)."
        ),
    )

    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())