"""
multicam_fusion.py — Multi-camera point cloud fusion for improved 6D pose estimation.

Pipeline:
  1. Back-project per-object masked depth → 3D point cloud in base frame
  2. Cluster detections across cameras by 3D centroid proximity (geometry-first),
     then arbitrate the class label by summed DINO-score voting within each cluster
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
    object_id: str          # DINO label (winner per-cam; may be overwritten by cross-cam arbitration)
    dino_score: float       # final selection-ranking score (mixes raw DINO + area/fill priors)
    mask: np.ndarray        # (H, W) bool
    mask_area: int
    bbox_xyxy: tuple[int, int, int, int]
    rgb: np.ndarray         # full camera RGB
    depth: np.ndarray       # full camera depth (meters)
    K: np.ndarray           # 3x3 intrinsics
    T_base_cam: np.ndarray  # 4x4 cam-to-base extrinsic
    centroid_base: np.ndarray | None = None  # (3,) computed lazily
    cloud_base: np.ndarray | None = None     # (N, 3) in base frame
    # Raw per-class DINO scores carried from CandidateSelection — used by
    # cross-camera label arbitration so each member contributes a vote for
    # every candidate label, not just its own self-pick.
    scores_by_object: dict[str, float] = field(default_factory=dict)


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

def _arbitrate_label(
    group: list[PerCamDetection],
    min_margin: float = 0.10,
) -> tuple[str, bool, float, str, float, float]:
    """Pick the class label for a matched group by summed raw-DINO voting.

    Each member contributes its *raw per-class* DINO score (from
    `scores_by_object`) for every candidate label, not just its own
    self-pick. This is critical: if cam_a's raw DINO had
    {cooling_screw: 0.85, cooling_f: 0.30} and cam_b had
    {cooling_screw: 0.05, cooling_f: 0.70}, the right answer is cooling_f
    (sum 1.00 > 0.90), even though both cams' self-picks were cooling_screw
    and cooling_f respectively.

    `dino_score` (the selection-ranking heuristic that mixes raw DINO with
    area/fill priors) is only used as a fallback for the member's self-pick
    when `scores_by_object` is empty.

    Returns:
        (winner_label, ambiguous_flag, winner_score, runner_label,
        runner_score, margin). When `ambiguous_flag` is True, the
        top-1 and top-2 summed scores are within `min_margin` of each
        other — the caller should drop the cluster rather than commit to a
        coin-flip class assignment.
    """
    if not group:
        return "", False, 0.0, "", 0.0, 0.0

    candidate_labels: set[str] = set()
    for d in group:
        sbo = getattr(d, "scores_by_object", None) or {}
        candidate_labels.update(sbo.keys())
        if d.object_id:
            candidate_labels.add(d.object_id)
    if not candidate_labels:
        return group[0].object_id, False, float(group[0].dino_score), "", 0.0, float(group[0].dino_score)

    sums: dict[str, float] = {lbl: 0.0 for lbl in candidate_labels}
    for d in group:
        sbo = getattr(d, "scores_by_object", None) or {}
        for lbl in candidate_labels:
            if lbl in sbo:
                sums[lbl] += float(sbo[lbl])
            elif lbl == d.object_id:
                sums[lbl] += float(d.dino_score)

    ranked = sorted(sums.items(), key=lambda kv: (-kv[1], kv[0]))
    winner_lbl, winner_score = ranked[0]
    if len(ranked) > 1:
        runner_lbl, runner_score = ranked[1]
        margin = float(winner_score - runner_score)
        ambiguous = margin < float(min_margin)
    else:
        runner_lbl = ""
        runner_score = 0.0
        margin = float(winner_score)
        ambiguous = False
    return winner_lbl, ambiguous, float(winner_score), runner_lbl, float(runner_score), margin


def match_detections_across_cameras(
    detections_by_cam: dict[str, list[PerCamDetection]],
    max_centroid_distance: float = 0.08,
    label_arbitration_min_margin: float = 0.05,
    label_arbitration_singleton_margin: float = 0.01,
) -> list[list[PerCamDetection]]:
    """
    Cluster detections across cameras by 3D centroid proximity (geometry-first),
    then arbitrate the class label per cluster by summed DINO-score voting.

    This handles the failure mode where cam1 labels a physical object as
    cooling_screw while cam2 labels the same physical object as cooling_f:
    both detections still cluster together by centroid, vote on the label,
    and produce a single fused detection with the winning label written back
    onto every member's object_id.

    Args:
        detections_by_cam: cam_id -> list of detections (with centroid_base populated)
        max_centroid_distance: max Euclidean distance in base frame to consider a match

    Returns:
        List of matched groups. Each group is a list of 1-2 PerCamDetections
        representing the same physical object seen from different cameras.
        Every detection's `object_id` is overwritten with the arbitrated label
        so downstream code (mesh lookup, FP, tracking init) sees one consistent
        class per cluster.
    """
    cam_ids = list(detections_by_cam.keys())
    if len(cam_ids) < 2:
        # Single camera: every detection is its own group (label unchanged)
        groups = []
        for dets in detections_by_cam.values():
            for d in dets:
                groups.append([d])
        return groups

    # For now, handle exactly 2 cameras
    cam_a, cam_b = cam_ids[0], cam_ids[1]
    dets_a = detections_by_cam[cam_a]
    dets_b = detections_by_cam[cam_b]

    # Geometry-first cost matrix between ALL cam_a x cam_b detections,
    # independent of DINO label. Label disagreement is handled by voting
    # after clustering, not by partitioning the search space upfront.
    na, nb = len(dets_a), len(dets_b)
    cost = np.full((na, nb), np.inf, dtype=np.float32)
    for ia in range(na):
        ca = dets_a[ia].centroid_base
        if ca is None:
            continue
        for ib in range(nb):
            cb = dets_b[ib].centroid_base
            if cb is None:
                continue
            cost[ia, ib] = float(np.linalg.norm(ca - cb))

    matched_groups: list[list[PerCamDetection]] = []
    matched_a: set[int] = set()
    matched_b: set[int] = set()
    # Detections that were geometrically paired but dropped for label ambiguity.
    # Crucially these MUST NOT fall through to the singleton path, or we re-
    # introduce the phantom-duplicate-track failure mode the matcher exists
    # to prevent.
    dropped_a: set[int] = set()
    dropped_b: set[int] = set()

    # Greedy minimum-distance assignment (one-to-one). Fine for small N.
    while cost.size > 0:
        min_val = float(cost.min())
        if not np.isfinite(min_val) or min_val > max_centroid_distance:
            break
        ia_best, ib_best = np.unravel_index(cost.argmin(), cost.shape)
        ia_best, ib_best = int(ia_best), int(ib_best)
        if ia_best in matched_a or ib_best in matched_b:
            cost[ia_best, ib_best] = np.inf
            continue
        group = [dets_a[ia_best], dets_b[ib_best]]
        winner, ambiguous, winner_score, runner, runner_score, margin = _arbitrate_label(
            group, min_margin=label_arbitration_min_margin,
        )
        if ambiguous:
            if margin >= float(label_arbitration_singleton_margin):
                winner_members = [d for d in group if d.object_id == winner]
                keep = max(
                    winner_members or group,
                    key=lambda d: float(d.dino_score),
                )
                drop_labels = [
                    f"{d.cam_id}={d.object_id}({d.dino_score:.2f})"
                    for d in group if d is not keep
                ]
                print(
                    f"[FUSION] LABEL WEAK at "
                    f"{tuple(np.round(dets_a[ia_best].centroid_base, 3))}: "
                    f"{cam_a}={dets_a[ia_best].object_id}({dets_a[ia_best].dino_score:.2f}) "
                    f"+ {cam_b}={dets_b[ib_best].object_id}({dets_b[ib_best].dino_score:.2f}) "
                    f"votes {winner}={winner_score:.2f} {runner}={runner_score:.2f} "
                    f"margin={margin:.2f} -> KEEP {keep.cam_id}={keep.object_id} "
                    f"as single-cam, DROP {', '.join(drop_labels)}"
                )
                matched_groups.append([keep])
                matched_a.add(ia_best)
                matched_b.add(ib_best)
                cost[ia_best, :] = np.inf
                cost[:, ib_best] = np.inf
                continue
            print(
                f"[FUSION] LABEL AMBIGUOUS at "
                f"{tuple(np.round(dets_a[ia_best].centroid_base, 3))}: "
                f"{cam_a}={dets_a[ia_best].object_id}({dets_a[ia_best].dino_score:.2f}) "
                f"+ {cam_b}={dets_b[ib_best].object_id}({dets_b[ib_best].dino_score:.2f}) "
                f"votes {winner}={winner_score:.2f} {runner}={runner_score:.2f} "
                f"margin={margin:.2f} -> DROP cluster "
                f"(margin < {label_arbitration_singleton_margin:.2f})"
            )
            dropped_a.add(ia_best)
            dropped_b.add(ib_best)
            # Mark both endpoints consumed so neither becomes a singleton.
            matched_a.add(ia_best)
            matched_b.add(ib_best)
            cost[ia_best, :] = np.inf
            cost[:, ib_best] = np.inf
            continue
        loser_labels = {d.object_id for d in group if d.object_id != winner}
        if loser_labels:
            print(
                f"[FUSION] LABEL ARBITRATION at "
                f"{tuple(np.round(dets_a[ia_best].centroid_base, 3))}: "
                f"{cam_a}={dets_a[ia_best].object_id}({dets_a[ia_best].dino_score:.2f}) "
                f"+ {cam_b}={dets_b[ib_best].object_id}({dets_b[ib_best].dino_score:.2f}) "
                f"votes {winner}={winner_score:.2f} {runner}={runner_score:.2f} "
                f"margin={margin:.2f} -> {winner}"
            )
        for d in group:
            d.object_id = winner
        matched_groups.append(group)
        matched_a.add(ia_best)
        matched_b.add(ib_best)
        cost[ia_best, :] = np.inf
        cost[:, ib_best] = np.inf

    # Singletons: detections never matched across cams keep their original
    # label. Detections dropped for ambiguity are NOT singletonised.
    for ia in range(na):
        if ia not in matched_a and ia not in dropped_a:
            matched_groups.append([dets_a[ia]])
    for ib in range(nb):
        if ib not in matched_b and ib not in dropped_b:
            matched_groups.append([dets_b[ib]])

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
    # Cross-camera label arbitration is tiered by summed per-class DINO
    # vote margin:
    #   margin >= label_arbitration_min_margin: fuse under the winner label
    #   singleton_margin <= margin < min_margin: keep the winner-side detection
    #       as single-cam and consume the loser to avoid duplicate phantoms
    #   margin < singleton_margin: drop both as a true coin flip
    label_arbitration_min_margin: float = 0.05
    label_arbitration_singleton_margin: float = 0.01
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
                scores_by_object={
                    str(k): float(v)
                    for k, v in (getattr(sel, "scores_by_object", {}) or {}).items()
                },
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
        dets_by_cam,
        max_centroid_distance=cfg.max_centroid_distance,
        label_arbitration_min_margin=cfg.label_arbitration_min_margin,
        label_arbitration_singleton_margin=cfg.label_arbitration_singleton_margin,
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
