from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# Data structures

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
    scores_by_object: dict[str, float] = field(default_factory=dict)


@dataclass
class FusedDetection:
    """Result of matching + fusing one object across cameras."""
    object_id: str
    detections: list[PerCamDetection]
    ref_cam_idx: int                   # index into detections for reference camera


# Back-projection: masked depth → 3D points in base frame
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
    # Masked pixel coordinates and their depths.
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

    # Unproject to camera frame (pinhole model)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    x_cam = (us - cx) * d / fx
    y_cam = (vs - cy) * d / fy
    z_cam = d

    pts_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)  # (N, 3)

    # Transform to base frame via the camera extrinsic
    R = T_base_cam[:3, :3].astype(np.float32)
    t = T_base_cam[:3, 3].astype(np.float32)
    pts_base = (pts_cam @ R.T) + t  # (N, 3)

    return pts_base


def compute_cloud_centroid(pts: np.ndarray) -> np.ndarray:
    """Compute centroid of point cloud. Returns (3,)."""
    if len(pts) == 0:
        return np.zeros(3, dtype=np.float32)
    return pts.mean(axis=0).astype(np.float32)


# Cross-camera object matching

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

    # Collect every label any member scored (plus each member's self-pick).
    candidate_labels: set[str] = set()
    for d in group:
        sbo = getattr(d, "scores_by_object", None) or {}
        candidate_labels.update(
            lbl for lbl, score in sbo.items()
            if np.isfinite(float(score))
        )
        if d.object_id:
            candidate_labels.add(d.object_id)
    if not candidate_labels:
        fallback = float(group[0].dino_score)
        if not np.isfinite(fallback):
            fallback = 0.0
        return group[0].object_id, False, fallback, "", 0.0, fallback

    # Sum each label's score across all members (raw per-class, or self-pick fallback).
    sums: dict[str, float] = {lbl: 0.0 for lbl in candidate_labels}
    for d in group:
        sbo = getattr(d, "scores_by_object", None) or {}
        for lbl in candidate_labels:
            if lbl in sbo:
                score = float(sbo[lbl])
                if np.isfinite(score):
                    sums[lbl] += score
            elif lbl == d.object_id:
                score = float(d.dino_score)
                if np.isfinite(score):
                    sums[lbl] += score

    # Winner = highest summed vote; flag ambiguous when top-2 are within the margin.
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


def _label_score_vector(det: PerCamDetection) -> dict[str, float]:
    """Per-class DINO score vector for one detection, falling back to its
    self-pick (object_id -> dino_score) when raw per-class scores are absent."""
    sbo = getattr(det, "scores_by_object", None) or {}
    if sbo:
        return {k: float(v) for k, v in sbo.items()}
    if det.object_id:
        return {det.object_id: float(det.dino_score)}
    return {}


def _label_agreement(det: PerCamDetection, cluster: list[PerCamDetection]) -> float:
    """Cosine similarity in [0, 1] between a detection's per-class DINO score
    vector and the cluster's aggregated (summed) score vector. 1.0 = same class
    with aligned confidence, 0.0 = disjoint classes. Falls back to exact
    object_id equality when raw per-class scores are unavailable on either side.
    """
    a = _label_score_vector(det)
    b: dict[str, float] = {}
    for m in cluster:
        for k, v in _label_score_vector(m).items():
            b[k] = b.get(k, 0.0) + v
    if not a or not b:
        return 1.0 if det.object_id and any(det.object_id == m.object_id for m in cluster) else 0.0
    labels = set(a) | set(b)
    va = np.array([a.get(l, 0.0) for l in labels], dtype=np.float64)
    vb = np.array([b.get(l, 0.0) for l in labels], dtype=np.float64)
    na, nb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.clip(np.dot(va, vb) / (na * nb), 0.0, 1.0))


def _cloud_bbox_diagonal(cloud: np.ndarray | None) -> float:
    """Bounding-box diagonal of a point cloud in metres. Returns 0 if too few points."""
    if cloud is None or len(cloud) < 4:
        return 0.0
    span = cloud.max(axis=0) - cloud.min(axis=0)
    return float(np.linalg.norm(span))


def match_detections_across_cameras(
    detections_by_cam: dict[str, list[PerCamDetection]],
    max_centroid_distance: float = 0.08,
    label_arbitration_min_margin: float = 0.05,
    label_arbitration_singleton_margin: float = 0.01,
    match_ambiguity_margin: float = 0.0,
    label_match_penalty_weight: float = 0.0,
    cloud_extent_gate_scale: float = 0.5,
) -> list[list[PerCamDetection]]:
    """
    Cluster detections across N cameras by 3D centroid proximity (geometry-
    first), then arbitrate the class label per cluster by summed DINO-score
    voting.

    """
    # With one camera there is nothing to match across — each detection stands alone.
    cam_ids = list(detections_by_cam.keys())
    if len(cam_ids) < 2:
        # Single camera: every detection is its own group (label unchanged)
        groups = []
        for dets in detections_by_cam.values():
            for d in dets:
                groups.append([d])
        return groups

    def _cluster_centroid(cluster: list[PerCamDetection]):
        pts = [d.centroid_base for d in cluster if d.centroid_base is not None]
        if not pts:
            return None
        return np.mean(np.stack(pts, axis=0), axis=0)

    # ── Phase 1: geometry-first incremental clustering across cameras ──
    clusters: list[list[PerCamDetection]] = []
    for cam_id in cam_ids:
        dets = detections_by_cam.get(cam_id, [])
        if not dets:
            continue
        if not clusters:
            # First non-empty camera seeds one cluster per detection.
            clusters = [[d] for d in dets]
            continue

        centroids = [_cluster_centroid(cl) for cl in clusters]
        # Representative cloud diagonal per cluster (largest-mask member).
        cluster_diags = [
            _cloud_bbox_diagonal(
                max(cl, key=lambda d: d.mask_area).cloud_base
            )
            for cl in clusters
        ]
        # Build a (detection × cluster) cost matrix of gated centroid distances.
        m, k = len(dets), len(clusters)
        cost = np.full((m, k), np.inf, dtype=np.float32)
        for i, d in enumerate(dets):
            ci = d.centroid_base
            if ci is None:
                continue
            diag_i = _cloud_bbox_diagonal(d.cloud_base) if cloud_extent_gate_scale > 0.0 else 0.0
            for j, cc in enumerate(centroids):
                if cc is None:
                    continue
                dist = float(np.linalg.norm(ci - cc))
                # Per-pair adaptive gate: large objects (big cloud diagonal)
                # get a wider merge window than small ones.
                if cloud_extent_gate_scale > 0.0:
                    adaptive_gate = max(
                        max_centroid_distance,
                        max(diag_i, cluster_diags[j]) * cloud_extent_gate_scale,
                    )
                else:
                    adaptive_gate = max_centroid_distance
                if dist > adaptive_gate:
                    continue  # leave cost[i, j] = inf, pair is ineligible
                if label_match_penalty_weight > 0.0:
                    dist += label_match_penalty_weight * (
                        1.0 - _label_agreement(d, clusters[j])
                    )
                cost[i, j] = dist

        # Greedily assign cheapest detection↔cluster pairs (≤1 det per cam per cluster).
        joined: set[int] = set()        # det idx -> joined an existing cluster
        used_cluster: set[int] = set()  # cluster idx -> already took a det
        dropped: set[int] = set()       # det idx -> dropped for ambiguity
        while cost.size > 0:
            min_val = float(cost.min())
            if not np.isfinite(min_val):
                break
            i_best, j_best = np.unravel_index(cost.argmin(), cost.shape)
            i_best, j_best = int(i_best), int(j_best)
            if i_best in joined or j_best in used_cluster:
                cost[i_best, j_best] = np.inf
                continue
            # Geometric ambiguity guard: another cluster near-tied for this det?
            if match_ambiguity_margin > 0.0:
                row = cost[i_best, :].copy()
                row[j_best] = np.inf
                second = float(row.min())
                if (np.isfinite(second)
                        and (second - min_val) < match_ambiguity_margin):
                    print(
                        f"[FUSION] MATCH AMBIGUOUS at "
                        f"{tuple(np.round(dets[i_best].centroid_base, 3))}: "
                        f"{cam_id}={dets[i_best].object_id} near two clusters "
                        f"(d1={min_val:.3f} d2={second:.3f}, "
                        f"margin<{match_ambiguity_margin:.3f}) -> DROP (no fuse)"
                    )
                    cost[i_best, :] = np.inf
                    dropped.add(i_best)
                    continue
            clusters[j_best].append(dets[i_best])
            joined.add(i_best)
            used_cluster.add(j_best)
            cost[i_best, :] = np.inf
            cost[:, j_best] = np.inf

        # Unmatched, non-dropped detections start their own cluster.
        for i in range(m):
            if i not in joined and i not in dropped:
                clusters.append([dets[i]])

    # ── Phase 2: per-cluster label arbitration (summed DINO voting) ──
    matched_groups: list[list[PerCamDetection]] = []
    for group in clusters:
        if not group:
            continue
        if len(group) == 1:
            # Single-camera detection: keep its original label untouched.
            matched_groups.append(group)
            continue

        # Vote on the cluster's class label.
        winner, ambiguous, winner_score, runner, runner_score, margin = _arbitrate_label(
            group, min_margin=label_arbitration_min_margin,
        )
        members_str = " + ".join(
            f"{d.cam_id}={d.object_id}({d.dino_score:.2f})" for d in group
        )
        pos = _cluster_centroid(group)
        pos_str = tuple(np.round(pos, 3)) if pos is not None else None

        # Ambiguous vote: keep only the winning-label members if the margin is
        # at least the singleton threshold, otherwise drop the whole cluster.
        if ambiguous:
            if margin >= float(label_arbitration_singleton_margin):
                winner_members = [d for d in group if d.object_id == winner]
                if winner_members:
                    keep = winner_members
                    for d in keep:
                        d.object_id = winner
                else:
                    keep = [max(group, key=lambda d: float(d.dino_score))]
                keep_ids = {id(d) for d in keep}
                drop_labels = [
                    f"{d.cam_id}={d.object_id}({d.dino_score:.2f})"
                    for d in group if id(d) not in keep_ids
                ]
                print(
                    f"[FUSION] LABEL WEAK at {pos_str}: {members_str} "
                    f"votes {winner}={winner_score:.2f} {runner}={runner_score:.2f} "
                    f"margin={margin:.2f} -> KEEP {winner} ({len(keep)} cam(s)), "
                    f"DROP {', '.join(drop_labels) or 'none'}"
                )
                matched_groups.append(list(keep))
                continue
            print(
                f"[FUSION] LABEL AMBIGUOUS at {pos_str}: {members_str} "
                f"votes {winner}={winner_score:.2f} {runner}={runner_score:.2f} "
                f"margin={margin:.2f} -> DROP cluster "
                f"(margin < {label_arbitration_singleton_margin:.2f})"
            )
            continue

        # Clear winner: overwrite every member's label with the arbitrated class.
        loser_labels = {d.object_id for d in group if d.object_id != winner}
        if loser_labels:
            print(
                f"[FUSION] LABEL ARBITRATION at {pos_str}: {members_str} "
                f"votes {winner}={winner_score:.2f} {runner}={runner_score:.2f} "
                f"margin={margin:.2f} -> {winner}"
            )
        for d in group:
            d.object_id = winner
        matched_groups.append(group)

    return matched_groups


# 
# Top-level fusion pipeline

@dataclass
class FusionConfig:
    max_centroid_distance: float = 0.08     # meters, for cross-cam matching
    min_depth: float = 0.05
    max_depth: float = 3.0
  
    label_arbitration_min_margin: float = 0.05
    label_arbitration_singleton_margin: float = 0.01
    
    match_ambiguity_margin: float = 0.0
    
    label_match_penalty_weight: float = 0.0
   
    cloud_extent_gate_scale: float = 0.5


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

        # Accept either an SE3 object or a raw 4x4 for the extrinsic.
        if hasattr(T_bc, 'as_matrix'):
            T_bc = T_bc.as_matrix()
        elif hasattr(T_bc, 'matrix'):
            T_bc = T_bc.matrix
        T_bc = np.asarray(T_bc, dtype=np.float32).reshape(4, 4)

        K = np.asarray(view.K, dtype=np.float32).reshape(3, 3)
        dets = []

        # Turn each accepted DINO selection into a PerCamDetection with its cloud.
        for sel in selections:
            if sel.object_id == "unknown":
                continue
            dino_score = float(sel.score)
            if not np.isfinite(dino_score):
                print(
                    f"[FUSION] DROP {cam_id}={sel.object_id}: "
                    f"non-finite selection score {dino_score}"
                )
                continue
            scores_by_object = {
                str(k): float(v)
                for k, v in (getattr(sel, "scores_by_object", {}) or {}).items()
                if np.isfinite(float(v))
            }

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
                dino_score=dino_score,
                mask=sel.candidate.mask,
                mask_area=int(sel.candidate.mask.sum()),
                bbox_xyxy=sel.candidate.bbox_xyxy,
                rgb=view.rgb,
                depth=view.depth,
                K=K,
                T_base_cam=T_bc,
                centroid_base=centroid,
                cloud_base=cloud,
                scores_by_object=scores_by_object,
            )
            dets.append(det)

        result[cam_id] = dets

    return result


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

    # 1) Per-camera detections + clouds → 2) cross-camera match/label → 3) FusedDetections.
    dets_by_cam = build_per_cam_detections(
        selections_by_cam, views_by_cam, T_base_cam_map, cfg,
    )

    matched_groups = match_detections_across_cameras(
        dets_by_cam,
        max_centroid_distance=cfg.max_centroid_distance,
        label_arbitration_min_margin=cfg.label_arbitration_min_margin,
        label_arbitration_singleton_margin=cfg.label_arbitration_singleton_margin,
        match_ambiguity_margin=cfg.match_ambiguity_margin,
        label_match_penalty_weight=cfg.label_match_penalty_weight,
        cloud_extent_gate_scale=cfg.cloud_extent_gate_scale,
    )

    # Reference camera per object = the one with the largest mask.
    results = []
    for group in matched_groups:
        if not group:
            continue
        ref_idx = max(range(len(group)), key=lambda i: group[i].mask_area)
        results.append(FusedDetection(
            object_id=group[ref_idx].object_id,
            detections=group,
            ref_cam_idx=ref_idx,
        ))
    return results
