"""
fused_multicam_helpers.py

Helper functions for multi-camera fused pose estimation.
Used by both init (dual-FP + ICP refinement + weighted average)
and tracking (Cutie per-cam + fused ICP).

Place this file alongside your runner, e.g.:
  src/perception/fused_multicam_helpers.py

Then import in your runner:
  from src.perception.fused_multicam_helpers import (
      lift_masked_depth_to_base,
      mesh_to_pcd_cached,
      run_icp_in_base_frame,
      weighted_average_poses,
  )
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import open3d as o3d
import trimesh


# ─── Point-cloud cache for meshes ───────────────────────────────────────

_MESH_PCD_CACHE: dict[str, o3d.geometry.PointCloud] = {}


def mesh_to_pcd_cached(
    mesh_path: str,
    mesh_scale: float,
    num_points: int = 5000,
) -> o3d.geometry.PointCloud:
    """Load mesh, sample surface points, cache result."""
    key = f"{mesh_path}_{mesh_scale}_{num_points}"
    if key in _MESH_PCD_CACHE:
        return _MESH_PCD_CACHE[key]

    mesh = trimesh.load(mesh_path, force="mesh")
    pts, _ = trimesh.sample.sample_surface(mesh, num_points)
    pts = pts.astype(np.float64) * mesh_scale

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    _MESH_PCD_CACHE[key] = pcd
    return pcd


# ─── Depth → base-frame point cloud ─────────────────────────────────────

def lift_masked_depth_to_base(
    depth: np.ndarray,
    mask: np.ndarray,
    K: np.ndarray,
    T_base_cam: np.ndarray,
    z_min: float = 0.05,
    z_max: float = 3.0,
    voxel_size: float = 0.002,
) -> Optional[o3d.geometry.PointCloud]:
    """
    Unproject masked depth pixels into 3D, transform to base frame.
    Returns an Open3D PointCloud (voxel-downsampled) or None if too few points.
    """
    mask_bool = np.asarray(mask).astype(bool)
    if mask_bool.sum() < 10:
        return None

    ys, xs = np.where(mask_bool)
    zs = depth[ys, xs].astype(np.float64)

    valid = np.isfinite(zs) & (zs > z_min) & (zs < z_max)
    if valid.sum() < 10:
        return None

    xs = xs[valid].astype(np.float64)
    ys = ys[valid].astype(np.float64)
    zs = zs[valid]

    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    pts_cam = np.stack([
        (xs - cx) * zs / fx,
        (ys - cy) * zs / fy,
        zs,
    ], axis=1)  # (N, 3) in camera frame

    T = np.asarray(T_base_cam, dtype=np.float64).reshape(4, 4)
    pts_base = (T[:3, :3] @ pts_cam.T).T + T[:3, 3]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_base)

    if voxel_size > 0 and len(pcd.points) > 100:
        pcd = pcd.voxel_down_sample(voxel_size)

    return pcd


def merge_point_clouds(
    clouds: list[o3d.geometry.PointCloud],
    voxel_size: float = 0.002,
) -> Optional[o3d.geometry.PointCloud]:
    """Merge multiple point clouds and voxel-downsample."""
    if not clouds:
        return None
    merged = o3d.geometry.PointCloud()
    for pcd in clouds:
        merged += pcd
    if voxel_size > 0 and len(merged.points) > 100:
        merged = merged.voxel_down_sample(voxel_size)
    return merged


# ─── ICP in base frame ──────────────────────────────────────────────────

def run_icp_in_base_frame(
    scene_pcd: o3d.geometry.PointCloud,
    model_pcd: o3d.geometry.PointCloud,
    T_base_object_init: np.ndarray,
    max_correspondence_dist: float = 0.05,
    max_iteration: int = 30,
) -> tuple[np.ndarray, float, float]:
    """
    Run point-to-point ICP in base frame.

    source = model points (object-local frame, transformed by T_init)
    target = scene points (already in base frame)

    Returns:
        T_base_object_refined (4x4 float32), fitness, rmse
    """
    T_init = np.asarray(T_base_object_init, dtype=np.float64).reshape(4, 4)

    result = o3d.pipelines.registration.registration_icp(
        source=model_pcd,
        target=scene_pcd,
        max_correspondence_distance=max_correspondence_dist,
        init=T_init,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=max_iteration,
        ),
    )

    T_refined = np.asarray(result.transformation, dtype=np.float32).reshape(4, 4)
    return T_refined, float(result.fitness), float(result.inlier_rmse)


# ─── Chamfer distance for pose evaluation ────────────────────────────────

def chamfer_distance_one_way(
    model_pcd: o3d.geometry.PointCloud,
    scene_pcd: o3d.geometry.PointCloud,
    T_base_object: np.ndarray,
) -> float:
    """
    Compute mean nearest-neighbor distance from transformed model to scene.
    Lower = better alignment.
    """
    T = np.asarray(T_base_object, dtype=np.float64).reshape(4, 4)
    model_pts = np.asarray(model_pcd.points)
    transformed = (T[:3, :3] @ model_pts.T).T + T[:3, 3]

    transformed_pcd = o3d.geometry.PointCloud()
    transformed_pcd.points = o3d.utility.Vector3dVector(transformed)

    dists = transformed_pcd.compute_point_cloud_distance(scene_pcd)
    return float(np.mean(dists))


# ─── Weighted pose averaging ────────────────────────────────────────────

def rotation_matrix_to_quaternion_wxyz(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation to quaternion [w, x, y, z]."""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    trace = np.trace(R)
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / (np.linalg.norm(q) + 1e-12)


def quaternion_wxyz_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
    q = np.asarray(q, dtype=np.float64).reshape(4)
    q = q / (np.linalg.norm(q) + 1e-12)
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two quaternions [w,x,y,z]."""
    q0 = q0 / (np.linalg.norm(q0) + 1e-12)
    q1 = q1 / (np.linalg.norm(q1) + 1e-12)

    dot = np.dot(q0, q1)
    if dot < 0:
        q1 = -q1
        dot = -dot

    dot = min(dot, 1.0)

    if dot > 0.9995:
        result = q0 + t * (q1 - q0)
        return result / (np.linalg.norm(result) + 1e-12)

    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)

    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return s0 * q0 + s1 * q1


def weighted_average_poses(
    poses: list[np.ndarray],
    weights: list[float],
) -> np.ndarray:
    """
    Compute weighted average of SE(3) poses.

    Translation: weighted mean.
    Rotation: SLERP between the two (for 2 poses), or pick highest-weight
              rotation for 3+ poses (full Karcher mean is overkill here).

    Args:
        poses: list of 4x4 transforms
        weights: corresponding non-negative weights (will be normalized)
    Returns:
        4x4 averaged transform
    """
    if len(poses) == 1:
        return poses[0].copy()

    w = np.array(weights, dtype=np.float64)
    w = w / (w.sum() + 1e-12)

    # Weighted translation average
    avg_t = np.zeros(3, dtype=np.float64)
    for T, wi in zip(poses, w):
        avg_t += wi * T[:3, 3].astype(np.float64)

    # Rotation: SLERP for 2 poses, highest-weight for more
    if len(poses) == 2:
        q0 = rotation_matrix_to_quaternion_wxyz(poses[0][:3, :3])
        q1 = rotation_matrix_to_quaternion_wxyz(poses[1][:3, :3])
        # SLERP parameter = weight of second pose
        avg_q = slerp(q0, q1, float(w[1]))
        avg_R = quaternion_wxyz_to_rotation_matrix(avg_q)
    else:
        best_idx = int(np.argmax(w))
        avg_R = poses[best_idx][:3, :3].astype(np.float64)

    T_avg = np.eye(4, dtype=np.float32)
    T_avg[:3, :3] = avg_R.astype(np.float32)
    T_avg[:3, 3] = avg_t.astype(np.float32)
    return T_avg