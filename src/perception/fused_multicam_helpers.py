from __future__ import annotations

from collections import deque
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
    # Cache keyed on (path, scale, point count) — sampling is the expensive part.
    key = f"{mesh_path}_{mesh_scale}_{num_points}"
    if key in _MESH_PCD_CACHE:
        return _MESH_PCD_CACHE[key]

    # Sample a surface point cloud and scale it to meters.
    mesh = trimesh.load(mesh_path, force="mesh")
    pts, _ = trimesh.sample.sample_surface(mesh, num_points)
    pts = pts.astype(np.float64) * mesh_scale

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    _MESH_PCD_CACHE[key] = pcd
    return pcd


# ─── Depth hole-filling ─────────────────────────────────────────────────

def fill_depth_holes_in_mask(
    depth: np.ndarray,
    mask: np.ndarray,
    kernel: int = 5,
) -> np.ndarray:
    """
    Fill zero / non-finite depth holes that fall inside `mask` using a
    median blur, leaving outside-mask pixels untouched.
    """
    import cv2

    # Holes = zero/non-finite depth inside the mask; nothing to do if there are none.
    mask_b = np.asarray(mask).astype(bool)
    d = np.asarray(depth, dtype=np.float32)
    bad = ((~np.isfinite(d)) | (d <= 0.0)) & mask_b
    if not bad.any():
        return d

    # uint16 millimetres for medianBlur (which is what it supports for k > 3).
    d_clean = np.where(np.isfinite(d) & (d > 0), d, 0.0)
    d_mm = np.clip(d_clean * 1000.0, 0.0, 65535.0).astype(np.uint16)

    # Odd kernel in [3,5], median-blur, back to meters.
    k = max(3, min(int(kernel), 5))
    if k % 2 == 0:
        k += 1
    d_blurred_mm = cv2.medianBlur(d_mm, k)
    d_blurred = d_blurred_mm.astype(np.float32) / 1000.0

    # Only overwrite the hole pixels; leave valid depth untouched.
    out = d.copy()
    out[bad] = d_blurred[bad]
    return out


# ─── Depth → base-frame point cloud ─────────────────────────────────────

def lift_masked_depth_to_base(
    depth: np.ndarray,
    mask: np.ndarray,
    K: np.ndarray,
    T_base_cam: np.ndarray,
    z_min: float = 0.05,
    z_max: float = 3.0,
    voxel_size: float = 0.002,
    mask_morph_close_kernel: int = 0,
    mask_interior_erosion: int = 0,
) -> Optional[o3d.geometry.PointCloud]:
    """
    Unproject masked depth pixels into 3D, transform to base frame.
    Returns an Open3D PointCloud (voxel-downsampled) or None if too few points.
    """
    mask_bool = np.asarray(mask).astype(bool)
    if mask_bool.sum() < 10:
        return None

    # Optional mask cleanup: close holes / erode the border inward.
    if mask_morph_close_kernel > 0 or mask_interior_erosion > 0:
        import cv2
        m_u8 = mask_bool.astype(np.uint8)
        if mask_morph_close_kernel > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (int(mask_morph_close_kernel), int(mask_morph_close_kernel)),
            )
            m_u8 = cv2.morphologyEx(m_u8, cv2.MORPH_CLOSE, kernel)
        if mask_interior_erosion > 0:
            ekernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (int(mask_interior_erosion), int(mask_interior_erosion)),
            )
            m_u8 = cv2.erode(m_u8, ekernel)
        mask_bool = m_u8.astype(bool)
        if mask_bool.sum() < 10:
            return None

    # Keep only masked pixels whose depth is finite and within range.
    ys, xs = np.where(mask_bool)
    zs = depth[ys, xs].astype(np.float64)

    valid = np.isfinite(zs) & (zs > z_min) & (zs < z_max)
    if valid.sum() < 10:
        return None

    xs = xs[valid].astype(np.float64)
    ys = ys[valid].astype(np.float64)
    zs = zs[valid]

    # Pinhole un-projection to camera-frame 3D points.
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    pts_cam = np.stack([
        (xs - cx) * zs / fx,
        (ys - cy) * zs / fy,
        zs,
    ], axis=1)  # (N, 3) in camera frame

    # Transform into the base frame via the camera extrinsic.
    T = np.asarray(T_base_cam, dtype=np.float64).reshape(4, 4)
    pts_base = (T[:3, :3] @ pts_cam.T).T + T[:3, 3]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_base)

    # Voxel-downsample for a consistent density.
    if voxel_size > 0 and len(pcd.points) > 100:
        pcd = pcd.voxel_down_sample(voxel_size)

    return pcd


def merge_point_clouds(
    clouds: list[o3d.geometry.PointCloud],
    voxel_size: float = 0.002,
) -> Optional[o3d.geometry.PointCloud]:
    """Merge multiple point clouds and voxel-downsample (unweighted, original behavior)."""
    if not clouds:
        return None
    # Concatenate all clouds, then optionally voxel-downsample.
    merged = o3d.geometry.PointCloud()
    for pcd in clouds:
        merged += pcd
    if voxel_size > 0 and len(merged.points) > 100:
        merged = merged.voxel_down_sample(voxel_size)
    return merged


def merge_point_clouds_weighted(
    clouds: list[o3d.geometry.PointCloud],
    cam_positions_base: list[np.ndarray],
    voxel_size: float = 0.002,
    distance_exponent: float = 2.0,
) -> Optional[o3d.geometry.PointCloud]:
    """
    Merge point clouds with distance-based weighting.

    """
    if not clouds:
        return None
    if len(clouds) == 1:
        return clouds[0]

    all_points = []
    all_weights = []

    # Weight each point by inverse distance to its camera (closer depth = more trusted).
    for pcd, cam_pos in zip(clouds, cam_positions_base):
        pts = np.asarray(pcd.points)
        if len(pts) == 0:
            continue

        cam_pos = np.asarray(cam_pos, dtype=np.float64).reshape(3)
        dists = np.linalg.norm(pts - cam_pos, axis=1)
        weights = 1.0 / (dists ** distance_exponent + 1e-6)

        all_points.append(pts)
        all_weights.append(weights)

    if not all_points:
        return None

    all_points_arr = np.concatenate(all_points, axis=0)
    all_weights_arr = np.concatenate(all_weights, axis=0)

    if len(all_points_arr) < 10:
        return None

    # Weighted voxel merge: accumulate weighted points per voxel, then divide.
    if voxel_size > 0:
        voxel_indices = np.floor(all_points_arr / voxel_size).astype(np.int64)

        voxel_dict: dict[tuple, tuple[np.ndarray, float]] = {}
        for i in range(len(all_points_arr)):
            key = tuple(voxel_indices[i])
            pt = all_points_arr[i]
            w = all_weights_arr[i]
            if key in voxel_dict:
                accum_pt, accum_w = voxel_dict[key]
                voxel_dict[key] = (accum_pt + pt * w, accum_w + w)
            else:
                voxel_dict[key] = (pt * w, w)

        result_pts = np.array([
            accum_pt / accum_w
            for accum_pt, accum_w in voxel_dict.values()
        ], dtype=np.float64)
    else:
        result_pts = all_points_arr

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(result_pts)
    return pcd


# ─── ICP in base frame ──────────────────────────────────────────────────

def run_icp_in_base_frame(
    scene_pcd: o3d.geometry.PointCloud,
    model_pcd: o3d.geometry.PointCloud,
    T_base_object_init: np.ndarray,
    max_correspondence_dist: float = 0.05,
    max_iteration: int = 30,
    variant: str = "point_to_point",
    normal_radius: float = 0.01,
    relative_fitness: float = 1e-6,
    relative_rmse: float = 1e-6,
) -> tuple[np.ndarray, float, float]:
    """
    Run ICP in base frame.

    source = model points (object-local frame, transformed by T_init)
    target = scene points (already in base frame)

    """
    T_init = np.asarray(T_base_object_init, dtype=np.float64).reshape(4, 4)

    # Pick the cost function; point-to-plane needs scene normals.
    if variant == "point_to_plane":
        if not scene_pcd.has_normals() and len(scene_pcd.points) >= 10:
            scene_pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30)
            )
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    elif variant == "point_to_point":
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    else:
        raise ValueError(f"Unknown ICP variant: {variant!r} (expected point_to_point or point_to_plane)")

    # Register model→scene from the init pose; returns the refined transform + metrics.
    result = o3d.pipelines.registration.registration_icp(
        source=model_pcd,
        target=scene_pcd,
        max_correspondence_distance=max_correspondence_dist,
        init=T_init,
        estimation_method=estimation,
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=max_iteration,
            relative_fitness=relative_fitness,
            relative_rmse=relative_rmse,
        ),
    )

    T_refined = np.asarray(result.transformation, dtype=np.float32).reshape(4, 4)
    return T_refined, float(result.fitness), float(result.inlier_rmse)


def evaluate_icp_in_base_frame(
    scene_pcd: o3d.geometry.PointCloud,
    model_pcd: o3d.geometry.PointCloud,
    T_base_object: np.ndarray,
    max_correspondence_dist: float = 0.05,
) -> tuple[float, float]:
    """Return (fitness, inlier_rmse) for the given transform without running
    ICP iterations.
    """
    # Score a fixed transform's fit (no iterations) — fitness + inlier RMSE.
    T = np.asarray(T_base_object, dtype=np.float64).reshape(4, 4)
    result = o3d.pipelines.registration.evaluate_registration(
        source=model_pcd,
        target=scene_pcd,
        max_correspondence_distance=max_correspondence_dist,
        transformation=T,
    )
    return float(result.fitness), float(result.inlier_rmse)


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
    # Transform the model by the pose, then average nearest-neighbour distance to the scene.
    T = np.asarray(T_base_object, dtype=np.float64).reshape(4, 4)
    model_pts = np.asarray(model_pcd.points)
    transformed = (T[:3, :3] @ model_pts.T).T + T[:3, 3]

    transformed_pcd = o3d.geometry.PointCloud()
    transformed_pcd.points = o3d.utility.Vector3dVector(transformed)

    dists = transformed_pcd.compute_point_cloud_distance(scene_pcd)
    return float(np.mean(dists))


# ─── Median Pose Buffer (temporal outlier filter, zero compute) ──────────

class MedianPoseBuffer:
    """
    Holds the last N accepted poses and outputs the median.

    This provides temporal smoothing that is robust to single-frame outliers
    without adding any compute cost. The median is computed component-wise
    on the translation, and the rotation closest to the median translation
    is used (to avoid quaternion averaging complexities).
    """

    def __init__(self, window_size: int = 3):
        self.window_size = window_size
        self._buffer: deque[np.ndarray] = deque(maxlen=window_size)

    def push(self, T_base: np.ndarray) -> None:
        """Add a new accepted pose to the buffer."""
        self._buffer.append(np.asarray(T_base, dtype=np.float32).reshape(4, 4).copy())

    def get_median(self) -> Optional[np.ndarray]:
        """Return the median pose from the buffer, or None if empty."""
        if not self._buffer:
            return None
        if len(self._buffer) == 1:
            return self._buffer[0].copy()

        # Component-wise median translation.
        translations = np.array([T[:3, 3] for T in self._buffer])
        median_t = np.median(translations, axis=0)

        # Use rotation from pose closest to median translation
        dists = np.linalg.norm(translations - median_t, axis=1)
        best_idx = int(np.argmin(dists))

        result = self._buffer[best_idx].copy()
        result[:3, 3] = median_t.astype(np.float32)
        return result

    def is_ready(self) -> bool:
        return len(self._buffer) >= 2

    def reset(self) -> None:
        self._buffer.clear()

    @property
    def count(self) -> int:
        return len(self._buffer)


# ─── Weighted pose averaging ────────────────────────────────────────────

def rotation_matrix_to_quaternion_wxyz(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation to quaternion [w, x, y, z]."""
    # Standard branchy R→quaternion (pick the largest diagonal term for stability).
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

    # Take the shorter arc.
    dot = np.dot(q0, q1)
    if dot < 0:
        q1 = -q1
        dot = -dot

    dot = min(dot, 1.0)

    # Near-parallel: fall back to (normalized) linear interpolation.
    if dot > 0.9995:
        result = q0 + t * (q1 - q0)
        return result / (np.linalg.norm(result) + 1e-12)

    # Otherwise interpolate along the great-circle arc.
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
    """
    if len(poses) == 1:
        return poses[0].copy()

    # Normalize the weights.
    w = np.array(weights, dtype=np.float64)
    w = w / (w.sum() + 1e-12)

    # Translation = weighted mean.
    avg_t = np.zeros(3, dtype=np.float64)
    for T, wi in zip(poses, w):
        avg_t += wi * T[:3, 3].astype(np.float64)

    # Rotation: SLERP for two poses, else just take the highest-weighted one.
    if len(poses) == 2:
        q0 = rotation_matrix_to_quaternion_wxyz(poses[0][:3, :3])
        q1 = rotation_matrix_to_quaternion_wxyz(poses[1][:3, :3])
        avg_q = slerp(q0, q1, float(w[1]))
        avg_R = quaternion_wxyz_to_rotation_matrix(avg_q)
    else:
        best_idx = int(np.argmax(w))
        avg_R = poses[best_idx][:3, :3].astype(np.float64)

    T_avg = np.eye(4, dtype=np.float32)
    T_avg[:3, :3] = avg_R.astype(np.float32)
    T_avg[:3, 3] = avg_t.astype(np.float32)
    return T_avg