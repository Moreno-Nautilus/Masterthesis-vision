from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from src.perception.view import View


# ------------------------------------------
# Fast NumPy voxel downsampling (no Open3D)
# ------------------------------------------

def _voxel_unique_indices(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if points.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)

    q = np.floor(points / float(voxel_size)).astype(np.int32)
    _, idx = np.unique(q, axis=0, return_index=True)
    idx.sort()
    return idx


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if len(points) == 0 or voxel_size <= 0.0:
        return np.asarray(points, dtype=np.float32)

    pts = np.asarray(points, dtype=np.float32)
    idx = _voxel_unique_indices(pts, voxel_size)
    return pts[idx]


def voxel_downsample_with_colors(
    points: np.ndarray, colors_rgb_u8: np.ndarray, voxel_size: float
) -> Tuple[np.ndarray, np.ndarray]:
    if len(points) == 0 or voxel_size <= 0.0:
        return (
            np.asarray(points, dtype=np.float32),
            np.asarray(colors_rgb_u8, dtype=np.uint8),
        )

    pts = np.asarray(points, dtype=np.float32)
    cols = np.asarray(colors_rgb_u8, dtype=np.uint8)

    idx = _voxel_unique_indices(pts, voxel_size)
    return pts[idx], cols[idx]


# -----------------------------
# RGB mask
# -----------------------------

@dataclass
class RGBMaskConfig:
    mode: str = "none"
    min_v: int = 30
    min_chroma: int = 12
    min_v_chroma: int = 25
    roi_x0: int = 0
    roi_y0: int = 0
    roi_x1: int = 0
    roi_y1: int = 0


def rgb_mask(rgb: Optional[np.ndarray], cfg: RGBMaskConfig) -> Optional[np.ndarray]:
    if cfg.mode == "none":
        return None
    if rgb is None:
        return None

    img = np.asarray(rgb)
    if img.ndim != 3 or img.shape[2] != 3:
        return None

    h, w, _ = img.shape

    if cfg.mode == "brightness":
        v = img.max(axis=2)
        return v >= int(cfg.min_v)

    if cfg.mode == "chroma":
        mx = img.max(axis=2).astype(np.int16)
        mn = img.min(axis=2).astype(np.int16)
        chroma = mx - mn
        return (mx >= int(cfg.min_v_chroma)) & (chroma >= int(cfg.min_chroma))

    if cfg.mode == "roi":
        x0 = int(np.clip(cfg.roi_x0, 0, w))
        x1 = int(np.clip(cfg.roi_x1, 0, w))
        y0 = int(np.clip(cfg.roi_y0, 0, h))
        y1 = int(np.clip(cfg.roi_y1, 0, h))
        m = np.zeros((h, w), dtype=bool)
        if x1 > x0 and y1 > y0:
            m[y0:y1, x0:x1] = True
        return m

    return None


# ---------------------------------------
# Fast fused points with colors
# ---------------------------------------

def fuse_views_to_points_base_with_colors(
    views: list[View],
    voxel_size: float = 0.005,
    stride: int = 2,
    zmin: float = 0.15,
    zmax: float = 2.0,
    rgb_mask_cfg: RGBMaskConfig = RGBMaskConfig(),
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Returns:
      pts_base: (N,3) float32
      cols_rgb: (N,3) uint8 aligned with pts_base or None
    """
    all_pts: list[np.ndarray] = []
    all_cols: list[np.ndarray] = []
    have_any_rgb = False

    for v in views:
        depth = np.asarray(v.depth, dtype=np.float32)
        H, W = depth.shape[:2]

        # grid sample
        us = np.arange(0, W, stride, dtype=np.int32)
        vs = np.arange(0, H, stride, dtype=np.int32)
        uu, vv = np.meshgrid(us, vs, indexing="xy")
        uu = uu.reshape(-1)
        vv = vv.reshape(-1)

        d = depth[vv, uu]
        valid = np.isfinite(d) & (d >= float(zmin)) & (d <= float(zmax))

        m_rgb = rgb_mask(v.rgb, rgb_mask_cfg)
        if m_rgb is not None:
            valid &= m_rgb[vv, uu]

        if not np.any(valid):
            continue

        uu = uu[valid]
        vv = vv[valid]
        d = d[valid]

        fx = float(v.K[0, 0])
        fy = float(v.K[1, 1])
        cx = float(v.K[0, 2])
        cy = float(v.K[1, 2])

        x = (uu.astype(np.float32) - cx) * d / fx
        y = (vv.astype(np.float32) - cy) * d / fy
        z = d
        pts_cam = np.stack((x, y, z), axis=1)

        pts_base = v.T_base_cam.transform_points(pts_cam).astype(np.float32)
        all_pts.append(pts_base)

        if v.rgb is not None:
            rgb = np.asarray(v.rgb)
            if rgb.ndim == 3 and rgb.shape[2] == 3 and rgb.dtype == np.uint8:
                cols = rgb[vv, uu, :]
                have_any_rgb = True
            else:
                cols = np.zeros((pts_base.shape[0], 3), dtype=np.uint8)
        else:
            cols = np.zeros((pts_base.shape[0], 3), dtype=np.uint8)

        all_cols.append(cols)

    if not all_pts:
        return np.zeros((0, 3), dtype=np.float32), None

    pts = np.vstack(all_pts).astype(np.float32)

    cols = None
    if have_any_rgb:
        cols = np.vstack(all_cols).astype(np.uint8)
        if cols.shape[0] != pts.shape[0]:
            cols = None

    if voxel_size > 0.0:
        if cols is None:
            pts = voxel_downsample(pts, voxel_size)
        else:
            pts, cols = voxel_downsample_with_colors(pts, cols, voxel_size)

    return pts, cols


# backward-compatible API
def fuse_views_to_points_base(
    views: list[View],
    voxel_size: float = 0.005,
    stride: int = 2,
    zmin: float = 0.15,
    zmax: float = 2.0,
) -> np.ndarray:
    pts, _cols = fuse_views_to_points_base_with_colors(
        views,
        voxel_size=voxel_size,
        stride=stride,
        zmin=zmin,
        zmax=zmax,
        rgb_mask_cfg=RGBMaskConfig(mode="none"),
    )
    return pts