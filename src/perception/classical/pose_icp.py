from __future__ import annotations

from pathlib import Path
import numpy as np
import open3d as o3d

from src.utils.se3 import SE3


def _pcd(pts: np.ndarray) -> o3d.geometry.PointCloud:
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(np.asarray(pts, dtype=float))
    return p


def estimate_pose_icp(
    observed_points: np.ndarray,   # points in some "target/obs" frame (e.g. base or cam)
    cad_points: np.ndarray,        # points in object frame
    voxel_size: float = 0.005,
) -> tuple[SE3, dict]:
    """
    Open3D convention:
      registration_icp(source, target) returns a transform that maps source -> target.

    Here:
      source = CAD points in object frame
      target = observed_points in "obs" frame

    Returns:
      T_obs_obj  (maps object-frame points into observed frame):
        P_obs ≈ T_obs_obj ⊕ P_obj
    """
    if observed_points is None or len(observed_points) < 30:
        return SE3.identity(), {
            "icp_fitness": 0.0,
            "icp_inlier_rmse": 0.0,
            "ransac_fitness": float("nan"),
            "ransac_inlier_rmse": float("nan"),
        }

    source = _pcd(cad_points)        # obj
    target = _pcd(observed_points)   # obs (base or cam)

    src_down = source.voxel_down_sample(voxel_size)
    tgt_down = target.voxel_down_sample(voxel_size)

    # Normals help FPFH; keep them
    radius_normal = voxel_size * 3.0
    src_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=50))
    tgt_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=50))

    source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=50))
    target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=50))

    radius_feature = voxel_size * 8.0
    src_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        src_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=200),
    )
    tgt_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        tgt_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=200),
    )

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

    # DEBUG-FRIENDLY ICP:
    # - point-to-point is far more robust when normals / plane / clutter are messy
    # - larger threshold allows inliers to exist
    icp_thresh = 0.03
    reg = o3d.pipelines.registration.registration_icp(
        source,
        target,
        icp_thresh,
        T0,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )

    T_obs_obj = SE3.from_matrix(np.asarray(reg.transformation))

    metrics = {
        "icp_fitness": float(reg.fitness),
        "icp_inlier_rmse": float(reg.inlier_rmse),
        "ransac_fitness": float(getattr(ransac, "fitness", np.nan)),
        "ransac_inlier_rmse": float(getattr(ransac, "inlier_rmse", np.nan)),
        "icp_thresh": float(icp_thresh),
    }

    return T_obs_obj, metrics


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