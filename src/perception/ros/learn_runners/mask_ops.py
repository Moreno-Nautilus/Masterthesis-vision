"""Mask / bbox / ROI-polygon helpers for the multicam init + tracking pipeline.

Extracted from run_pipeline_track_multicam_realsense.py.
"""
from __future__ import annotations

import cv2
import numpy as np

from src.perception.learned.SAM.sam_segmentation import SAMMaskCandidate
from src.perception.ros.learn_runners.track_types import ObjectTrackState


def parse_polygon_string(s: str) -> np.ndarray:
    # Empty/whitespace => no ROI; resolved to the full frame at use time.
    if not s or not s.strip():
        return np.zeros((0, 2), dtype=np.int32)
    vals = [int(v.strip()) for v in s.split(",")]
    if len(vals) % 2 != 0:
        raise ValueError(f"Polygon string must have even number of values: {s}")
    return np.array(vals, dtype=np.int32).reshape(-1, 2)


# Remove duplicate same-class states that occupy nearly the same camera-frame position.
def nms_by_position(
    states: list[ObjectTrackState],
    position_threshold: float = 0.05,
) -> list[ObjectTrackState]:
    if len(states) <= 1:
        return states

    by_class: dict[str, list[ObjectTrackState]] = {}
    for s in states:
        by_class.setdefault(s.object_id, []).append(s)

    kept: list[ObjectTrackState] = []
    for _, obj_states in by_class.items():
        if len(obj_states) == 1:
            kept.extend(obj_states)
            continue

        obj_states = sorted(obj_states, key=lambda x: x.dino_score, reverse=True)
        keep_mask = [True] * len(obj_states)

        for i in range(len(obj_states)):
            if not keep_mask[i]:
                continue
            if obj_states[i].T_object_camera is None:
                continue
            pos_i = obj_states[i].T_object_camera[:3, 3]
            tid_i = getattr(obj_states[i], "track_id", "") or ""

            for j in range(i + 1, len(obj_states)):
                if not keep_mask[j]:
                    continue
                if obj_states[j].T_object_camera is None:
                    continue
                tid_j = getattr(obj_states[j], "track_id", "") or ""
                if tid_i and tid_j and tid_i != tid_j:
                    continue
                pos_j = obj_states[j].T_object_camera[:3, 3]
                dist = np.linalg.norm(pos_i - pos_j)
                if dist < position_threshold:
                    keep_mask[j] = False

        kept.extend([s for s, k in zip(obj_states, keep_mask) if k])

    return kept


# Crop RGB and mask around a bbox, keeping a little context for DINO/memory crops.
def bbox_crop_with_local_mask(
    rgb: np.ndarray,
    mask: np.ndarray,
    bbox_xyxy: tuple[int, int, int, int],
    pad_frac: float = 0.15,
    min_pad_px: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = [int(v) for v in bbox_xyxy]

    bw = x1 - x0
    bh = y1 - y0

    pad_x = max(min_pad_px, int(round(bw * pad_frac)))
    pad_y = max(min_pad_px, int(round(bh * pad_frac)))

    x0p = max(0, x0 - pad_x)
    y0p = max(0, y0 - pad_y)
    x1p = min(w, x1 + pad_x)
    y1p = min(h, y1 + pad_y)

    return (
        rgb[y0p:y1p, x0p:x1p].copy(),
        mask[y0p:y1p, x0p:x1p].copy(),
    )


def upscale_crop_if_small(
    rgb: np.ndarray,
    mask: np.ndarray,
    min_side: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Bicubic-upscale a small crop so DINOv2's input-size downsample
    doesn't throw away detail.
    """
    if min_side <= 0:
        return rgb, mask
    h, w = rgb.shape[:2]
    short = min(h, w)
    if short == 0 or short >= min_side:
        return rgb, mask
    scale = float(min_side) / float(short)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    rgb_up = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    mask_up = cv2.resize(
        mask.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST
    ).astype(mask.dtype)
    return rgb_up, mask_up


# Filter obviously over-large SAM masks before expensive DINO scoring.
def reject_large_masks(
    masks: list[SAMMaskCandidate],
    h: int,
    w: int,
    max_mask_area_ratio: float,
    max_bbox_area_ratio: float,
) -> list[SAMMaskCandidate]:
    img_area = float(h * w)
    out = []
    for c in masks:
        x0, y0, x1, y1 = c.bbox_xyxy
        if float(c.area) / img_area > max_mask_area_ratio:
            continue
        if float((x1 - x0) * (y1 - y0)) / img_area > max_bbox_area_ratio:
            continue
        out.append(c)
    return out


# Keep masks whose bbox center lies inside the camera ROI polygon.
def reject_outside_roi_polygon(
    masks: list[SAMMaskCandidate],
    polygon: np.ndarray,
) -> list[SAMMaskCandidate]:
    kept = []
    for m in masks:
        x0, y0, x1, y1 = m.bbox_xyxy
        cx = (x0 + x1) // 2
        cy = (y0 + y1) // 2
        if cv2.pointPolygonTest(polygon, (float(cx), float(cy)), False) >= 0:
            kept.append(m)
    return kept


# Drop masks that mostly touch the image border; these are usually background spills.
def reject_border_masks(
    masks: list[SAMMaskCandidate],
    border_px: int,
    max_border_fraction: float,
) -> list[SAMMaskCandidate]:
    out = []
    for c in masks:
        m = c.mask
        h, w = m.shape[:2]

        bp = min(border_px, h // 2, w // 2)
        if bp <= 0:
            out.append(c)
            continue

        border_pixels = (
            m[:bp, :].sum() + m[-bp:, :].sum()
            + m[bp:-bp, :bp].sum()
            + m[bp:-bp, -bp:].sum()
        )
        if c.area == 0:
            continue
        if float(border_pixels) / float(c.area) > max_border_fraction:
            continue
        out.append(c)
    return out


def pad_mask_for_fp(mask: np.ndarray, pad_px: int = 5) -> np.ndarray:
    """Dilate mask by pad_px pixels to give FP more context around the object."""
    if pad_px <= 0:
        return mask
    kernel = np.ones((2 * pad_px + 1, 2 * pad_px + 1), np.uint8)
    dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
    return dilated.astype(mask.dtype)


def mask_depth_coverage(
    depth: np.ndarray,
    mask: np.ndarray,
    zmin: float = 0.05,
    zmax: float = 3.0,
) -> float:
    """Fraction of mask pixels with finite depth inside the valid range."""
    mask_bool = mask.astype(bool)
    n_mask = int(mask_bool.sum())
    if n_mask == 0:
        return 0.0
    d = depth[mask_bool]
    valid = np.isfinite(d) & (d > zmin) & (d < zmax)
    return float(valid.sum()) / float(n_mask)


def bbox_containment_ratio(
    inner: tuple[int, int, int, int],
    outer: tuple[int, int, int, int],
) -> float:
    """How much of `inner` is covered by `outer`."""
    ix0, iy0, ix1, iy1 = inner
    ox0, oy0, ox1, oy1 = outer

    ax0 = max(ix0, ox0)
    ay0 = max(iy0, oy0)
    ax1 = min(ix1, ox1)
    ay1 = min(iy1, oy1)

    iw = max(0, ax1 - ax0)
    ih = max(0, ay1 - ay0)
    inter = iw * ih
    inner_area = max(1, (ix1 - ix0) * (iy1 - iy0))

    return float(inter) / float(inner_area)


def bbox_iou_xyxy(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    area_a = max(1, (ax1 - ax0) * (ay1 - ay0))
    area_b = max(1, (bx1 - bx0) * (by1 - by0))
    union = area_a + area_b - inter
    return float(inter) / float(union) if union > 0 else 0.0


def dedup_masks_by_bbox_iou(
    masks: list[SAMMaskCandidate],
    iou_thresh: float = 0.7,
    containment_thresh: float = 0.9,
) -> list[SAMMaskCandidate]:
    """Greedy bbox-level dedup, keeping larger masks first."""
    out = []
    masks_sorted = sorted(masks, key=lambda m: m.area, reverse=True)

    for m in masks_sorted:
        keep = True
        for k in out:
            if bbox_iou_xyxy(m.bbox_xyxy, k.bbox_xyxy) > iou_thresh:
                keep = False
                break
            if bbox_containment_ratio(m.bbox_xyxy, k.bbox_xyxy) > containment_thresh:
                keep = False
                break
        if keep:
            out.append(m)

    return out


def crop_rgb_to_polygon_bbox(
    rgb: np.ndarray,
    polygon: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Crop an RGB frame to the ROI bbox and shift polygon coords into crop space."""
    h, w = rgb.shape[:2]
    xs = polygon[:, 0]
    ys = polygon[:, 1]

    x0 = max(0, int(xs.min()))
    y0 = max(0, int(ys.min()))
    x1 = min(w, int(xs.max()) + 1)
    y1 = min(h, int(ys.max()) + 1)

    rgb_crop = rgb[y0:y1, x0:x1].copy()
    polygon_crop = polygon.copy()
    polygon_crop[:, 0] -= x0
    polygon_crop[:, 1] -= y0

    return rgb_crop, polygon_crop.astype(np.int32), x0, y0


def lift_crop_masks_to_full_image(
    crop_masks: list[SAMMaskCandidate],
    full_h: int,
    full_w: int,
    x0: int,
    y0: int,
) -> list[SAMMaskCandidate]:
    """Map masks produced on the ROI crop back into full-image coordinates."""
    lifted = []
    for c in crop_masks:
        full_mask = np.zeros((full_h, full_w), dtype=c.mask.dtype)
        h, w = c.mask.shape[:2]
        full_mask[y0:y0 + h, x0:x0 + w] = c.mask

        bx0, by0, bx1, by1 = c.bbox_xyxy
        lifted.append(
            SAMMaskCandidate(
                mask=full_mask,
                bbox_xyxy=(bx0 + x0, by0 + y0, bx1 + x0, by1 + y0),
                area=int(c.area),
                score=float(c.score),
                prompt_score=c.prompt_score,
            )
        )
    return lifted
