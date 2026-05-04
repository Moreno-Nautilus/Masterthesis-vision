"""
multicam_fusion.py — Multi-camera point cloud fusion for improved 6D pose estimation.

Pipeline:
  1. Back-project per-object masked depth → 3D point cloud in base frame
  2. Match objects across cameras by DINO label + 3D centroid proximity
  3. ICP-align per-object clouds from different cameras
  4. Fuse (concatenate + voxel downsample) into a single cloud
  5. Render fused cloud back into reference camera's depth image
  6. Feed fused depth + reference RGB/mask/K to FoundationPose

Place this file alongside run_pipeline_track.py:
  src/perception/ros/learn_runners/multicam_fusion.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import open3d as o3d


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PerCamDetection:
    """One detected object from one camera."""
    cam_id: str
    object_id: str          # DINO label
    dino_score: float
    mask: np.ndarray        # (H, W) bool
    mask_area: int
    bbox_xyxy: tuple[int, int, int, int]
    rgb: np.ndarray         # full camera RGB
    depth: np.ndarray       # full camera depth (meters)
    K: np.ndarray           # 3x3 intrinsics
    T_base_cam: np.ndarray  # 4x4 cam-to-base extrinsic
    centroid_base: np.ndarray | None = None  # (3,) computed lazily
    cloud_base: np.ndarray | None = None     # (N, 3) in base frame


@dataclass
class FusedDetection:
    """Result of matching + fusing one object across cameras."""
    object_id: str
    detections: list[PerCamDetection]  # 1 or 2 cams
    fused_cloud_base: np.ndarray       # (N, 3) in base frame, voxel-downsampled
    ref_cam_idx: int                   # index into detections for reference camera
    fused_depth: np.ndarray | None = None  # (H, W) rendered into ref cam
    fused_mask: np.ndarray | None = None   # (H, W) bool, reprojected mask


# ---------------------------------------------------------------------------
# Back-projection: masked depth → 3D points in base frame
# ---------------------------------------------------------------------------

def backproject_masked_depth(
    depth: np.ndarray,
    mask: np.ndarray,
    K: np.ndarray,
    T_base_cam: np.ndarray,
    min_depth: float = 0.05,
    max_depth: float = 3.0,
) -> np.ndarray:
    """
    Back-project masked depth pixels into 3D points in base frame.

    Returns (N, 3) float32 array. May be empty if no valid depth.
    """
    mask_bool = mask.astype(bool)
    vs, us = np.where(mask_bool)
    if len(vs) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    d = depth[vs, us].astype(np.float32)

    # Filter invalid depth
    valid = np.isfinite(d) & (d > min_depth) & (d < max_depth)
    if valid.sum() == 0:
        return np.zeros((0, 3), dtype=np.float32)

    vs = vs[valid].astype(np.float32)
    us = us[valid].astype(np.float32)
    d = d[valid]

    # Unproject to camera frame
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    x_cam = (us - cx) * d / fx
    y_cam = (vs - cy) * d / fy
    z_cam = d

    pts_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)  # (N, 3)

    # Transform to base frame
    R = T_base_cam[:3, :3].astype(np.float32)
    t = T_base_cam[:3, 3].astype(np.float32)
    pts_base = (pts_cam @ R.T) + t  # (N, 3)

    return pts_base


def compute_cloud_centroid(pts: np.ndarray) -> np.ndarray:
    """Compute centroid of point cloud. Returns (3,)."""
    if len(pts) == 0:
        return np.zeros(3, dtype=np.float32)
    return pts.mean(axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Cross-camera object matching
# ---------------------------------------------------------------------------

def match_detections_across_cameras(
    detections_by_cam: dict[str, list[PerCamDetection]],
    max_centroid_distance: float = 0.08,
) -> list[list[PerCamDetection]]:
    """
    Match detected objects across cameras by DINO label + 3D centroid proximity.

    Args:
        detections_by_cam: cam_id -> list of detections (with centroid_base populated)
        max_centroid_distance: max Euclidean distance in base frame to consider a match

    Returns:
        List of matched groups. Each group is a list of 1-2 PerCamDetections
        representing the same physical object seen from different cameras.
    """
    cam_ids = list(detections_by_cam.keys())
    if len(cam_ids) < 2:
        # Single camera: every detection is its own group
        groups = []
        for dets in detections_by_cam.values():
            for d in dets:
                groups.append([d])
        return groups

    # For now, handle exactly 2 cameras
    cam_a, cam_b = cam_ids[0], cam_ids[1]
    dets_a = detections_by_cam[cam_a]
    dets_b = list(detections_by_cam[cam_b])  # copy so we can mark used

    matched_groups: list[list[PerCamDetection]] = []
    used_b: set[int] = set()

    # Group detections by DINO label
    labels_a: dict[str, list[int]] = {}
    for i, d in enumerate(dets_a):
        labels_a.setdefault(d.object_id, []).append(i)

    labels_b: dict[str, list[int]] = {}
    for j, d in enumerate(dets_b):
        labels_b.setdefault(d.object_id, []).append(j)

    # For each label that appears in both cameras, match by centroid proximity
    for label in labels_a:
        if label not in labels_b:
            # Only seen in cam_a
            for i in labels_a[label]:
                matched_groups.append([dets_a[i]])
            continue

        indices_a = labels_a[label]
        indices_b = [j for j in labels_b[label] if j not in used_b]

        if not indices_b:
            for i in indices_a:
                matched_groups.append([dets_a[i]])
            continue

        # Build cost matrix based on centroid distance
        cost = np.full((len(indices_a), len(indices_b)), np.inf, dtype=np.float32)
        for ia, idx_a in enumerate(indices_a):
            ca = dets_a[idx_a].centroid_base
            if ca is None:
                continue
            for ib, idx_b in enumerate(indices_b):
                cb = dets_b[idx_b].centroid_base
                if cb is None:
                    continue
                cost[ia, ib] = float(np.linalg.norm(ca - cb))

        # Greedy matching (sufficient for small numbers of objects)
        matched_a: set[int] = set()
        matched_b_local: set[int] = set()

        while True:
            if cost.size == 0:
                break
            min_val = cost.min()
            if min_val > max_centroid_distance:
                break
            ia_best, ib_best = np.unravel_index(cost.argmin(), cost.shape)
            if ia_best in matched_a or ib_best in matched_b_local:
                cost[ia_best, ib_best] = np.inf
                continue

            idx_a = indices_a[ia_best]
            idx_b = indices_b[ib_best]
            matched_groups.append([dets_a[idx_a], dets_b[idx_b]])
            matched_a.add(ia_best)
            matched_b_local.add(ib_best)
            used_b.add(idx_b)
            cost[ia_best, :] = np.inf
            cost[:, ib_best] = np.inf

        # Unmatched from cam_a for this label
        for ia, idx_a in enumerate(indices_a):
            if ia not in matched_a:
                matched_groups.append([dets_a[idx_a]])

    # Labels only in cam_b (not processed above)
    for label in labels_b:
        if label in labels_a:
            continue
        for j in labels_b[label]:
            if j not in used_b:
                matched_groups.append([dets_b[j]])

    # Unmatched cam_b detections from shared labels
    for j, d in enumerate(dets_b):
        if j not in used_b:
            # Check it wasn't already added
            already_added = any(
                any(dd is d for dd in group)
                for group in matched_groups
            )
            if not already_added:
                matched_groups.append([d])

    return matched_groups


# ---------------------------------------------------------------------------
# ICP alignment of per-object clouds
# ---------------------------------------------------------------------------

def icp_align_clouds(
    cloud_source: np.ndarray,
    cloud_target: np.ndarray,
    max_correspondence_distance: float = 0.025,
    voxel_size: float = 0.001,
) -> tuple[np.ndarray, float, float]:
    """
    Align cloud_source to cloud_target using ICP.

    Both clouds should be in the same frame (base frame) already,
    so this corrects residual calibration error.

    Args:
        cloud_source: (N, 3) source points
        cloud_target: (M, 3) target points
        max_correspondence_distance: ICP max distance (meters)
        voxel_size: downsample before ICP for speed

    Returns:
        (aligned_source, fitness, rmse)
        aligned_source: (N, 3) transformed source points
        fitness: ICP fitness (0-1, higher = more inliers)
        rmse: ICP RMSE in meters
    """
    if len(cloud_source) < 10 or len(cloud_target) < 10:
        # Not enough points for ICP, return as-is
        return cloud_source, 0.0, float('inf')

    pcd_src = o3d.geometry.PointCloud()
    pcd_src.points = o3d.utility.Vector3dVector(cloud_source.astype(np.float64))

    pcd_tgt = o3d.geometry.PointCloud()
    pcd_tgt.points = o3d.utility.Vector3dVector(cloud_target.astype(np.float64))

    # Downsample for ICP speed
    if voxel_size > 0:
        pcd_src_ds = pcd_src.voxel_down_sample(voxel_size)
        pcd_tgt_ds = pcd_tgt.voxel_down_sample(voxel_size)
    else:
        pcd_src_ds = pcd_src
        pcd_tgt_ds = pcd_tgt

    # Estimate normals for point-to-plane if enough points
    if len(pcd_src_ds.points) > 30 and len(pcd_tgt_ds.points) > 30:
        pcd_src_ds.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=20))
        pcd_tgt_ds.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=20))
        icp_method = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    else:
        icp_method = o3d.pipelines.registration.TransformationEstimationPointToPoint()

    # Identity init (clouds are already approximately aligned via extrinsics)
    init_transform = np.eye(4, dtype=np.float64)

    result = o3d.pipelines.registration.registration_icp(
        source=pcd_src_ds,
        target=pcd_tgt_ds,
        max_correspondence_distance=max_correspondence_distance,
        init=init_transform,
        estimation_method=icp_method,
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=50,
        ),
    )

    T_icp = np.asarray(result.transformation, dtype=np.float64)
    fitness = float(result.fitness)
    rmse = float(result.inlier_rmse)

    # Apply ICP transform to the FULL (non-downsampled) source
    R_icp = T_icp[:3, :3]
    t_icp = T_icp[:3, 3]
    aligned = (cloud_source.astype(np.float64) @ R_icp.T + t_icp).astype(np.float32)

    return aligned, fitness, rmse


# ---------------------------------------------------------------------------
# Cloud fusion: concatenate + voxel downsample
# ---------------------------------------------------------------------------

def fuse_and_downsample(
    clouds: list[np.ndarray],
    voxel_size: float = 0.001,
) -> np.ndarray:
    """
    Concatenate multiple point clouds and voxel-downsample.

    Args:
        clouds: list of (N_i, 3) arrays
        voxel_size: voxel size in meters (0.001 = 1mm)

    Returns:
        (M, 3) fused and downsampled points
    """
    valid = [c for c in clouds if len(c) > 0]
    if not valid:
        return np.zeros((0, 3), dtype=np.float32)

    merged = np.concatenate(valid, axis=0)

    if voxel_size <= 0 or len(merged) < 10:
        return merged.astype(np.float32)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(merged.astype(np.float64))
    pcd_ds = pcd.voxel_down_sample(voxel_size)

    return np.asarray(pcd_ds.points, dtype=np.float32)


# ---------------------------------------------------------------------------
# Render fused cloud back into a camera's depth image
# ---------------------------------------------------------------------------

def render_fused_depth(
    cloud_base: np.ndarray,
    K: np.ndarray,
    T_base_cam: np.ndarray,
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Project fused point cloud (in base frame) into a camera to create
    a synthetic depth image and mask.

    Args:
        cloud_base: (N, 3) points in base frame
        K: 3x3 camera intrinsics
        T_base_cam: 4x4 base-to-camera extrinsic (cam_to_base transform)
        image_shape: (H, W) of the target image

    Returns:
        (depth_img, mask_img)
        depth_img: (H, W) float32, depth in meters (0 where no projection)
        mask_img: (H, W) bool, True where a point was projected
    """
    H, W = image_shape

    if len(cloud_base) == 0:
        return np.zeros((H, W), dtype=np.float32), np.zeros((H, W), dtype=bool)

    # Transform from base to camera frame
    # T_base_cam is cam-to-base, so we need its inverse: base-to-cam
    T_cam_base = np.linalg.inv(T_base_cam.astype(np.float64))
    R_cb = T_cam_base[:3, :3]
    t_cb = T_cam_base[:3, 3]

    pts_cam = (cloud_base.astype(np.float64) @ R_cb.T + t_cb).astype(np.float32)

    # Filter points behind camera
    valid = pts_cam[:, 2] > 0.01
    pts_cam = pts_cam[valid]

    if len(pts_cam) == 0:
        return np.zeros((H, W), dtype=np.float32), np.zeros((H, W), dtype=bool)

    # Project to pixel coordinates
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    u = (pts_cam[:, 0] * fx / pts_cam[:, 2] + cx).astype(np.float32)
    v = (pts_cam[:, 1] * fy / pts_cam[:, 2] + cy).astype(np.float32)
    z = pts_cam[:, 2]

    # Round to pixel indices
    ui = np.round(u).astype(np.int32)
    vi = np.round(v).astype(np.int32)

    # Filter out-of-bounds
    in_bounds = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    ui = ui[in_bounds]
    vi = vi[in_bounds]
    z = z[in_bounds]

    if len(ui) == 0:
        return np.zeros((H, W), dtype=np.float32), np.zeros((H, W), dtype=bool)

    # Z-buffer: keep closest point per pixel
    depth_img = np.zeros((H, W), dtype=np.float32)
    mask_img = np.zeros((H, W), dtype=bool)

    # Sort by depth (farthest first, so closest overwrites)
    order = np.argsort(-z)
    ui = ui[order]
    vi = vi[order]
    z = z[order]

    depth_img[vi, ui] = z
    mask_img[vi, ui] = True

    return depth_img, mask_img


# ---------------------------------------------------------------------------
# Top-level fusion pipeline
# ---------------------------------------------------------------------------

@dataclass
class FusionConfig:
    max_centroid_distance: float = 0.08     # meters, for cross-cam matching
    icp_max_correspondence: float = 0.025   # meters
    icp_voxel_size: float = 0.001           # meters, downsample before ICP
    fusion_voxel_size: float = 0.001        # meters, final downsample
    min_cloud_points: int = 50              # skip fusion if too few points
    icp_min_fitness: float = 0.10           # below this, skip ICP correction
    min_depth: float = 0.05
    max_depth: float = 3.0
    debug_dir: str = ""                     # set to a path to dump debug PLY/PNG
    debug_enabled: bool = False


# ---------------------------------------------------------------------------
# Debug visualization
# ---------------------------------------------------------------------------

def _save_debug_cloud(
    path: str,
    cloud: np.ndarray,
    color: tuple[int, int, int] = (128, 128, 128),
) -> None:
    """Save point cloud as PLY with uniform color."""
    if len(cloud) == 0:
        return
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud.astype(np.float64))
    colors = np.tile(np.array(color, dtype=np.float64) / 255.0, (len(cloud), 1))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(path, pcd)


def _save_debug_clouds_colored(
    path: str,
    clouds_with_colors: list[tuple[np.ndarray, tuple[int, int, int]]],
) -> None:
    """Save multiple point clouds as a single PLY, each with its own color."""
    all_pts = []
    all_colors = []
    for cloud, color in clouds_with_colors:
        if len(cloud) == 0:
            continue
        all_pts.append(cloud)
        c = np.array(color, dtype=np.float64) / 255.0
        all_colors.append(np.tile(c, (len(cloud), 1)))

    if not all_pts:
        return

    pts = np.concatenate(all_pts, axis=0)
    cols = np.concatenate(all_colors, axis=0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(cols)
    o3d.io.write_point_cloud(path, pcd)


def _save_debug_depth_image(path: str, depth: np.ndarray) -> None:
    """Save depth map as a colorized PNG for visual inspection."""
    valid = depth[depth > 0]
    if len(valid) == 0:
        cv2.imwrite(path, np.zeros_like(depth, dtype=np.uint8))
        return
    d_min, d_max = float(valid.min()), float(valid.max())
    if d_max - d_min < 1e-6:
        d_max = d_min + 0.1

    norm = np.zeros_like(depth, dtype=np.uint8)
    mask = depth > 0
    norm[mask] = np.clip(
        ((depth[mask] - d_min) / (d_max - d_min) * 255), 0, 255
    ).astype(np.uint8)
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    colored[~mask] = 0
    cv2.imwrite(path, colored)


def build_per_cam_detections(
    selections_by_cam: dict[str, list],  # cam_id -> list of CandidateSelection
    views_by_cam: dict[str, object],     # cam_id -> View
    T_base_cam_map: dict[str, object],   # cam_id -> SE3 or 4x4
    cfg: FusionConfig | None = None,
) -> dict[str, list[PerCamDetection]]:
    """
    Convert per-camera CandidateSelections into PerCamDetections with
    back-projected clouds and centroids.
    """
    cfg = cfg or FusionConfig()
    result: dict[str, list[PerCamDetection]] = {}

    for cam_id, selections in selections_by_cam.items():
        view = views_by_cam[cam_id]
        T_bc = T_base_cam_map[cam_id]

        # Convert SE3 to matrix if needed
        if hasattr(T_bc, 'as_matrix'):
            T_bc = T_bc.as_matrix()
        elif hasattr(T_bc, 'matrix'):
            T_bc = T_bc.matrix
        T_bc = np.asarray(T_bc, dtype=np.float32).reshape(4, 4)

        K = np.asarray(view.K, dtype=np.float32).reshape(3, 3)
        dets = []

        for sel in selections:
            if sel.object_id == "unknown":
                continue

            cloud = backproject_masked_depth(
                depth=view.depth,
                mask=sel.candidate.mask,
                K=K,
                T_base_cam=T_bc,
                min_depth=cfg.min_depth,
                max_depth=cfg.max_depth,
            )
            centroid = compute_cloud_centroid(cloud) if len(cloud) > 0 else None

            det = PerCamDetection(
                cam_id=cam_id,
                object_id=sel.object_id,
                dino_score=float(sel.score),
                mask=sel.candidate.mask,
                mask_area=int(sel.candidate.mask.sum()),
                bbox_xyxy=sel.candidate.bbox_xyxy,
                rgb=view.rgb,
                depth=view.depth,
                K=K,
                T_base_cam=T_bc,
                centroid_base=centroid,
                cloud_base=cloud,
            )
            dets.append(det)

        result[cam_id] = dets

    return result


def fuse_matched_group(
    group: list[PerCamDetection],
    cfg: FusionConfig | None = None,
) -> FusedDetection | None:
    """
    Given a matched group of detections (1 or 2 cameras for same object),
    fuse their point clouds and render back into the reference camera.

    Returns None if fusion fails (too few points, etc.)
    """
    cfg = cfg or FusionConfig()

    if not group:
        return None

    # Pick reference camera: the one with the larger mask
    ref_idx = max(range(len(group)), key=lambda i: group[i].mask_area)
    ref = group[ref_idx]

    if len(group) == 1:
        # Single camera, no fusion needed — just pass through
        cloud = group[0].cloud_base
        if cloud is None or len(cloud) < cfg.min_cloud_points:
            return None

        return FusedDetection(
            object_id=ref.object_id,
            detections=group,
            fused_cloud_base=cloud,
            ref_cam_idx=ref_idx,
            fused_depth=None,   # will use original depth
            fused_mask=None,    # will use original mask
        )

    # Multi-camera fusion
    clouds = []
    for det in group:
        if det.cloud_base is not None and len(det.cloud_base) >= cfg.min_cloud_points:
            clouds.append((det, det.cloud_base))

    if len(clouds) < 2:
        # Only one camera had enough points, treat as single-cam
        cloud = clouds[0][1] if clouds else ref.cloud_base
        if cloud is None or len(cloud) < cfg.min_cloud_points:
            return None
        return FusedDetection(
            object_id=ref.object_id,
            detections=group,
            fused_cloud_base=cloud,
            ref_cam_idx=ref_idx,
            fused_depth=None,
            fused_mask=None,
        )

    # ICP: align non-reference clouds to reference cloud
    ref_cloud = group[ref_idx].cloud_base
    aligned_clouds = [ref_cloud]

    for i, det in enumerate(group):
        if i == ref_idx:
            continue

        t0 = time.time()
        aligned, fitness, rmse = icp_align_clouds(
            cloud_source=det.cloud_base,
            cloud_target=ref_cloud,
            max_correspondence_distance=cfg.icp_max_correspondence,
            voxel_size=cfg.icp_voxel_size,
        )
        elapsed_ms = (time.time() - t0) * 1000

        if fitness >= cfg.icp_min_fitness:
            aligned_clouds.append(aligned)
            print(
                f"[FUSION] ICP {det.object_id} {det.cam_id}->{group[ref_idx].cam_id}: "
                f"fitness={fitness:.3f} rmse={rmse*1000:.1f}mm time={elapsed_ms:.0f}ms"
            )
        else:
            # ICP failed — still include the raw cloud (better than nothing)
            aligned_clouds.append(det.cloud_base)
            print(
                f"[FUSION] ICP POOR {det.object_id} {det.cam_id}: "
                f"fitness={fitness:.3f}, using raw cloud"
            )

    # Fuse and downsample
    fused = fuse_and_downsample(aligned_clouds, voxel_size=cfg.fusion_voxel_size)

    if len(fused) < cfg.min_cloud_points:
        return None

    # Render fused cloud back into reference camera
    fused_depth, fused_mask = render_fused_depth(
        cloud_base=fused,
        K=ref.K,
        T_base_cam=ref.T_base_cam,
        image_shape=ref.depth.shape[:2],
    )

    # --- Debug saves ---
    if cfg.debug_enabled and cfg.debug_dir:
        import os
        ts = int(time.time() * 1000)
        obj_dir = os.path.join(cfg.debug_dir, ref.object_id)
        os.makedirs(obj_dir, exist_ok=True)

        # Per-camera clouds (different colors) + fused result
        cam_colors = [(255, 80, 80), (80, 80, 255), (80, 255, 80)]
        pre_icp_parts = []
        for ci, det in enumerate(group):
            c = cam_colors[ci % len(cam_colors)]
            if det.cloud_base is not None and len(det.cloud_base) > 0:
                pre_icp_parts.append((det.cloud_base, c))
                _save_debug_cloud(
                    os.path.join(obj_dir, f"{ts}_{det.cam_id}_raw.ply"),
                    det.cloud_base, c,
                )

        # Pre-ICP overlay (both cameras raw)
        if len(pre_icp_parts) > 1:
            _save_debug_clouds_colored(
                os.path.join(obj_dir, f"{ts}_pre_icp_overlay.ply"),
                pre_icp_parts,
            )

        # Post-ICP / fused cloud
        _save_debug_cloud(
            os.path.join(obj_dir, f"{ts}_fused.ply"),
            fused, (0, 200, 0),
        )

        # Post-ICP overlay (ref + aligned)
        post_icp_parts = []
        post_icp_parts.append((ref_cloud, cam_colors[0]))
        for ci, ac in enumerate(aligned_clouds[1:], 1):
            post_icp_parts.append((ac, cam_colors[ci % len(cam_colors)]))
        _save_debug_clouds_colored(
            os.path.join(obj_dir, f"{ts}_post_icp_overlay.ply"),
            post_icp_parts,
        )

        # Rendered depth
        if fused_depth is not None:
            _save_debug_depth_image(
                os.path.join(obj_dir, f"{ts}_fused_depth.png"),
                fused_depth,
            )

        # Original depth from ref cam (for comparison)
        _save_debug_depth_image(
            os.path.join(obj_dir, f"{ts}_ref_original_depth.png"),
            ref.depth,
        )

        # Fused mask
        if fused_mask is not None:
            mask_vis = (fused_mask.astype(np.uint8) * 255)
            cv2.imwrite(os.path.join(obj_dir, f"{ts}_fused_mask.png"), mask_vis)

        print(f"[FUSION DEBUG] Saved debug for {ref.object_id} -> {obj_dir}")

    return FusedDetection(
        object_id=ref.object_id,
        detections=group,
        fused_cloud_base=fused,
        ref_cam_idx=ref_idx,
        fused_depth=fused_depth,
        fused_mask=fused_mask,
    )


def run_multicam_fusion(
    selections_by_cam: dict[str, list],
    views_by_cam: dict[str, object],
    T_base_cam_map: dict[str, object],
    cfg: FusionConfig | None = None,
) -> list[FusedDetection]:
    """
    Top-level fusion: takes per-camera DINO selections and views,
    returns fused detections ready for FoundationPose.

    Args:
        selections_by_cam: cam_id -> list of CandidateSelection (from DINO)
        views_by_cam: cam_id -> View object
        T_base_cam_map: cam_id -> extrinsic (SE3 or 4x4)
        cfg: fusion parameters

    Returns:
        List of FusedDetection, each ready for FoundationPose
    """
    cfg = cfg or FusionConfig()

    dets_by_cam = build_per_cam_detections(
        selections_by_cam, views_by_cam, T_base_cam_map, cfg,
    )

    matched_groups = match_detections_across_cameras(
        dets_by_cam, max_centroid_distance=cfg.max_centroid_distance,
    )

    results = []
    for group in matched_groups:
        if not group:
            continue
        ref_idx = max(range(len(group)), key=lambda i: group[i].mask_area)
        results.append(FusedDetection(
            object_id=group[ref_idx].object_id,
            detections=group,
            fused_cloud_base=np.zeros((0, 3), dtype=np.float32),
            ref_cam_idx=ref_idx,
        ))
    return results