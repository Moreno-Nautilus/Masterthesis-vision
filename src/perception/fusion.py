from __future__ import annotations
import numpy as np
import open3d as o3d
from src.perception.view import View

def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if len(points) == 0:
        return points
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pcd = pcd.voxel_down_sample(voxel_size)
    return np.asarray(pcd.points)

def fuse_views_to_points_base(
    views: list[View],
    voxel_size : float = 0.005,
    stride: int = 2,
    zmin: float = 0.15,
    zmax: float = 2.0
)-> np.ndarray:
    from src.perception.backproject import depth_to_points_cam

    all_pts = []
    for v in views:
        pts_cam = depth_to_points_cam(v.depth, v.K, stride = stride, zmin = zmin, zmax = zmax)
        pts_base = v.T_base_cam.transform_points(pts_cam)
        all_pts.append(pts_base)
    
    if not all_pts:
        return np.zeros((0,3), dtype = float)

    pts = np.vstack(all_pts).astype(np.float32)
    pts = voxel_downsample(pts, voxel_size)
    return pts