from __future__ import annotations

from pathlib import Path
import numpy as np
import open3d as o3d

from src.utils.se3 import SE3


def _pcd(pts: np.ndarray) -> o3d.geometry.PointCloud:
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(np.asarray(pts, dtype=float))
    return p


def _rot_x(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def _rot_y(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def _rot_z(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def estimate_pose_icp(
    observed_points: np.ndarray,
    cad_points: np.ndarray,
    voxel_size: float = 0.005,
) -> tuple[SE3, dict]:
    """
    Convention / frames:

      - observed_points are points measured in the camera (or world) frame:  P_cam
      - cad_points are model points in the object frame:                    P_obj

    We return:
      - T_obj_cam (maps cam -> obj), such that:
            P_obj ≈ T_obj_cam(P_cam)

    If you need the usual vision pose (object in camera frame):
      - T_cam_obj = T_obj_cam.inverse()
      - then: P_cam_pred = T_cam_obj(P_obj)
    """
    source = _pcd(observed_points)  # cam/world
    target = _pcd(cad_points)       # obj

    # Downsample for global registration
    src_down = source.voxel_down_sample(voxel_size)
    tgt_down = target.voxel_down_sample(voxel_size)

    # Normals
    radius_normal = voxel_size * 3.0
    src_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=50)
    )
    tgt_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=50)
    )

    source.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=50)
    )
    target.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=50)
    )

    # FPFH
    radius_feature = voxel_size * 8.0
    src_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        src_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=200),
    )
    tgt_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        tgt_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=200),
    )

    # RANSAC init
    dist_thresh = voxel_size * 4.0
    ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_down,
        tgt_down,
        src_fpfh,
        tgt_fpfh,
        mutual_filter=False,
        max_correspondence_distance=dist_thresh,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist_thresh),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(80000, 1000),
    )

    T0 = np.asarray(ransac.transformation, dtype=float)

    # Multi-init candidates
    candidates: list[np.ndarray] = []

    def _add_rot_left(Rleft: np.ndarray) -> None:
        T = T0.copy()
        T[:3, :3] = Rleft @ T[:3, :3]
        candidates.append(T)

    candidates.append(T0)

    # tabletop ambiguity: yaw hypotheses
    for yaw in (-90.0, 90.0, 180.0):
        _add_rot_left(_rot_z(yaw))

    # 180° flips (common "wrong direction")
    for Rflip in (_rot_x(180.0), _rot_y(180.0), _rot_z(180.0)):
        _add_rot_left(Rflip)

    # a couple random yaw jitters (helps with partial views)
    rng = np.random.default_rng(0)
    for _ in range(2):
        yaw = float(rng.uniform(-30.0, 30.0))
        _add_rot_left(_rot_z(yaw))

    # Evaluate candidates with ICP refine
    best_T = None
    best_fit = -1.0
    best_rmse = float("inf")

    for init in candidates:
        reg1 = o3d.pipelines.registration.registration_icp(
            source,
            target,
            0.05,
            init,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        )
        reg2 = o3d.pipelines.registration.registration_icp(
            source,
            target,
            0.01,
            reg1.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        )
        fit = float(getattr(reg2, "fitness", 0.0))
        rmse = float(getattr(reg2, "inlier_rmse", np.inf))

        if (fit > best_fit) or (fit == best_fit and rmse < best_rmse):
            best_fit, best_rmse = fit, rmse
            best_T = np.asarray(reg2.transformation, dtype=float)

    if best_T is None:
        best_T = np.eye(4)

    T_obj_cam = SE3.from_matrix(best_T)

    metrics = {
        "ransac_fitness": float(getattr(ransac, "fitness", np.nan)),
        "ransac_inlier_rmse": float(getattr(ransac, "inlier_rmse", np.nan)),
        "icp_fitness": float(best_fit),
        "icp_inlier_rmse": float(best_rmse),
        "n_candidates": float(len(candidates)),
    }
    return T_obj_cam, metrics


def load_mesh(path: str | Path) -> o3d.geometry.TriangleMesh:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CAD mesh not found: {path.resolve()}")

    mesh = o3d.io.read_triangle_mesh(str(path))
    if mesh.is_empty():
        raise ValueError(f"Loaded mesh is empty: {path.resolve()}")

    if not mesh.has_triangles():
        raise ValueError(f"Mesh has no triangles (bad/empty file): {path.resolve()}")

    mesh.compute_vertex_normals()
    return mesh


def load_cad_as_pointcloud(
    path: str | Path,
    voxel_size: float = 0.005,
    n_points: int = 5000,
    scale: float = 1.0,
    center: bool = True,
) -> np.ndarray:
    mesh = load_mesh(path)
    pcd = mesh.sample_points_uniformly(number_of_points=n_points)
    pcd = pcd.voxel_down_sample(voxel_size)
    pts = np.asarray(pcd.points) * float(scale)
    if center:
        pts = pts - pts.mean(axis=0)
    return pts