from __future__ import annotations

from pathlib import Path
import numpy as np
import open3d as o3d

from src.utils.se3 import SE3


def _pcd(pts: np.ndarray) -> o3d.geometry.PointCloud:
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(np.asarray(pts, dtype=np.float64))
    return p


def _subsample(pts: np.ndarray, max_points: int) -> np.ndarray:
    pts = np.asarray(pts)
    if pts.shape[0] <= max_points:
        return pts
    stride = max(1, int(np.ceil(pts.shape[0] / max_points)))
    return pts[::stride]


def estimate_pose_icp(
    observed_points: np.ndarray,
    cad_points: np.ndarray,
    voxel_size: float = 0.008,
) -> tuple[SE3, dict]:
    """
    Fast ICP version:
    - no FPFH
    - no RANSAC feature matching
    - centroid initialization + point-to-point ICP

    Returns:
      T_obs_obj  (maps object-frame points into observed frame)
    """
    observed_points = np.asarray(observed_points, dtype=np.float32).reshape(-1, 3)
    cad_points = np.asarray(cad_points, dtype=np.float32).reshape(-1, 3)

    if observed_points.shape[0] < 20 or cad_points.shape[0] < 20:
        return SE3.identity(), {
            "icp_fitness": 0.0,
            "icp_inlier_rmse": 1e9,
            "ransac_fitness": float("nan"),
            "ransac_inlier_rmse": float("nan"),
            "icp_thresh": 0.03,
        }

    # aggressive cap for speed
    observed_points = _subsample(observed_points, 400)
    cad_points = _subsample(cad_points, 400)

    source = _pcd(cad_points)        # object frame
    target = _pcd(observed_points)   # observed frame

    if voxel_size > 0.0:
        source = source.voxel_down_sample(voxel_size)
        target = target.voxel_down_sample(voxel_size)

    src_pts = np.asarray(source.points)
    tgt_pts = np.asarray(target.points)

    if src_pts.shape[0] < 10 or tgt_pts.shape[0] < 10:
        return SE3.identity(), {
            "icp_fitness": 0.0,
            "icp_inlier_rmse": 1e9,
            "ransac_fitness": float("nan"),
            "ransac_inlier_rmse": float("nan"),
            "icp_thresh": 0.03,
        }

    # simple translation-only init from centroids
    src_center = src_pts.mean(axis=0)
    tgt_center = tgt_pts.mean(axis=0)

    T0 = np.eye(4, dtype=np.float64)
    T0[:3, 3] = tgt_center - src_center

    icp_thresh = 0.025
    reg = o3d.pipelines.registration.registration_icp(
        source,
        target,
        icp_thresh,
        T0,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20),
    )

    T_obs_obj = SE3.from_matrix(np.asarray(reg.transformation))

    metrics = {
        "icp_fitness": float(reg.fitness),
        "icp_inlier_rmse": float(reg.inlier_rmse),
        "ransac_fitness": float("nan"),
        "ransac_inlier_rmse": float("nan"),
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
    voxel_size: float = 0.008,
    n_points: int = 3000,
    scale: float = 1.0,
    center: bool = True,
) -> np.ndarray:
    mesh = load_mesh(path)
    pcd = mesh.sample_points_uniformly(number_of_points=n_points)
    pcd = pcd.voxel_down_sample(voxel_size)
    pts = np.asarray(pcd.points, dtype=np.float32) * float(scale)
    if center and pts.shape[0] > 0:
        pts = pts - pts.mean(axis=0, keepdims=True)
    return pts