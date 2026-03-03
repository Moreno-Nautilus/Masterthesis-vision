from __future__ import annotations
import numpy as np
import open3d as o3d
from src.utils.se3 import SE3

def _pcd(pts: np.ndarray) -> o3d.geometry.PointCloud:
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(np.asarray(pts, dtype=float))
    return p

def estimate_extrinsic_icp(
    pts_cam1: np.ndarray,
    pts_cam2: np.ndarray,
    voxel_size: float = 0.01,
) -> tuple[SE3, dict]:
    """
    Returns T_cam1_cam2 (maps cam2 -> cam1):
        P_cam1 ≈ T_cam1_cam2(P_cam2)
    """

    source = _pcd(pts_cam2)  # cam2
    target = _pcd(pts_cam1)  # cam1

    # Downsample
    src_down = source.voxel_down_sample(voxel_size)
    tgt_down = target.voxel_down_sample(voxel_size)

    # Normals
    r_n = voxel_size * 3.0
    for p in (src_down, tgt_down):
        p.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=r_n, max_nn=50))

    # FPFH
    r_f = voxel_size * 8.0
    src_f = o3d.pipelines.registration.compute_fpfh_feature(
        src_down, o3d.geometry.KDTreeSearchParamHybrid(radius=r_f, max_nn=200)
    )
    tgt_f = o3d.pipelines.registration.compute_fpfh_feature(
        tgt_down, o3d.geometry.KDTreeSearchParamHybrid(radius=r_f, max_nn=200)
    )

    # RANSAC init
    dist = voxel_size * 4.0
    ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_down, tgt_down, src_f, tgt_f,
        mutual_filter=True,
        max_correspondence_distance=dist,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(80000, 1000),
    )

    T0 = np.asarray(ransac.transformation, dtype=float)

    # ICP refine (point-to-plane is best if normals exist; if unstable, switch to PointToPoint)
    source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=r_n, max_nn=50))
    target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=r_n, max_nn=50))

    icp1 = o3d.pipelines.registration.registration_icp(
        source, target, 0.05, T0,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )
    icp2 = o3d.pipelines.registration.registration_icp(
        source, target, 0.01, icp1.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )

    T = np.asarray(icp2.transformation, dtype=float)

    metrics = {
        "ransac_fitness": float(getattr(ransac, "fitness", np.nan)),
        "ransac_inlier_rmse": float(getattr(ransac, "inlier_rmse", np.nan)),
        "icp_fitness": float(getattr(icp2, "fitness", np.nan)),
        "icp_inlier_rmse": float(getattr(icp2, "inlier_rmse", np.nan)),
        "voxel_size": float(voxel_size),
    }
    return SE3.from_matrix(T), metrics