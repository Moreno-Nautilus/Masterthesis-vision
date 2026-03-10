from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import open3d as o3d

from src.perception.view import View


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if len(points) == 0:
        return points
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pcd = pcd.voxel_down_sample(voxel_size)
    return np.asarray(pcd.points)


def voxel_downsample_with_colors(
    points: np.ndarray, colors_rgb_u8: np.ndarray, voxel_size: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Downsample points and colors together using Open3D.
    Colors are assumed uint8 RGB in [0,255].
    """
    if len(points) == 0:
        return points, colors_rgb_u8

    pts = np.asarray(points, dtype=np.float64)
    cols = np.asarray(colors_rgb_u8, dtype=np.float64)
    if cols.max() > 1.0:
        cols = cols / 255.0

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pcd.colors = o3d.utility.Vector3dVector(cols)
    pcd = pcd.voxel_down_sample(voxel_size)

    pts_ds = np.asarray(pcd.points, dtype=np.float32)
    cols_ds = np.asarray(pcd.colors, dtype=np.float32)
    cols_ds_u8 = np.clip(np.round(cols_ds * 255.0), 0, 255).astype(np.uint8)
    return pts_ds, cols_ds_u8


# -----------------------------
# Option 2: RGB mask (classical)
# -----------------------------

@dataclass
class RGBMaskConfig:
    mode: str = "none"
    # For mode="brightness":
    min_v: int = 30                 # keep pixels with max(R,G,B) >= min_v
    # For mode="chroma":
    min_chroma: int = 12            # keep pixels with max-min >= min_chroma
    min_v_chroma: int = 25          # also require max >= min_v_chroma
    # For mode="roi":
    roi_x0: int = 0
    roi_y0: int = 0
    roi_x1: int = 0
    roi_y1: int = 0


def rgb_mask(rgb: Optional[np.ndarray], cfg: RGBMaskConfig) -> Optional[np.ndarray]:
    """
    Returns boolean mask HxW (True = keep pixel), or None if no mask.
    """
    if cfg.mode == "none":
        return None
    if rgb is None:
        return None

    img = np.asarray(rgb)
    if img.ndim != 3 or img.shape[2] != 3:
        # unexpected format -> skip masking rather than crashing
        return None

    h, w, _ = img.shape

    if cfg.mode == "brightness":
        v = img.max(axis=2)  # [0..255]
        return v >= int(cfg.min_v)

    if cfg.mode == "chroma":
        mx = img.max(axis=2).astype(np.int16)
        mn = img.min(axis=2).astype(np.int16)
        chroma = mx - mn
        return (mx >= int(cfg.min_v_chroma)) & (chroma >= int(cfg.min_chroma))

    if cfg.mode == "roi":
        # keep only a 2D ROI in the image
        x0 = int(np.clip(cfg.roi_x0, 0, w))
        x1 = int(np.clip(cfg.roi_x1, 0, w))
        y0 = int(np.clip(cfg.roi_y0, 0, h))
        y1 = int(np.clip(cfg.roi_y1, 0, h))
        m = np.zeros((h, w), dtype=bool)
        if x1 > x0 and y1 > y0:
            m[y0:y1, x0:x1] = True
        return m

    # unknown mode -> no mask
    return None


# ---------------------------------------
# Option 1 (+2): fuse points WITH colors
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
      cols_rgb: (N,3) uint8 (always aligned with pts_base) or None if no rgb anywhere
    """
    all_pts: list[np.ndarray] = []
    all_cols: list[np.ndarray] = []
    have_any_rgb = False

    for v in views:
        depth = np.asarray(v.depth)
        H, W = depth.shape[:2]

        # sample pixels on grid
        us = np.arange(0, W, stride, dtype=np.int32)
        vs = np.arange(0, H, stride, dtype=np.int32)
        uu, vv = np.meshgrid(us, vs)
        uu = uu.reshape(-1)
        vv = vv.reshape(-1)

        d = depth[vv, uu].astype(np.float32)
        valid = np.isfinite(d) & (d >= float(zmin)) & (d <= float(zmax))

        # Option 2: RGB mask in pixel domain
        m_rgb = rgb_mask(v.rgb, rgb_mask_cfg)
        if m_rgb is not None:
            valid &= m_rgb[vv, uu]

        uu = uu[valid]
        vv = vv[valid]
        d = d[valid]
        if d.size == 0:
            continue

        # backproject (guaranteed 1:1 with uu,vv)
        fx = float(v.K[0, 0]); fy = float(v.K[1, 1])
        cx = float(v.K[0, 2]); cy = float(v.K[1, 2])

        x = (uu.astype(np.float32) - cx) * d / fx
        y = (vv.astype(np.float32) - cy) * d / fy
        z = d
        pts_cam = np.stack([x, y, z], axis=1)

        pts_base = v.T_base_cam.transform_points(pts_cam)
        all_pts.append(pts_base)

        # Option 1: attach colors (ALWAYS create an array with same N as pts_base)
        if v.rgb is not None:
            rgb = np.asarray(v.rgb)
            if rgb.ndim == 3 and rgb.shape[2] == 3 and rgb.dtype == np.uint8:
                cols = rgb[vv, uu, :]                 # (N,3) uint8
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
        # sanity
        if cols.shape[0] != pts.shape[0]:
            cols = None

    # Downsample (points + colors together if colors exist)
    if voxel_size > 0.0:
        if cols is None:
            pts = voxel_downsample(pts, voxel_size).astype(np.float32)
        else:
            pts, cols = voxel_downsample_with_colors(pts, cols, voxel_size)

    return pts, cols

# -----------------------------
# Backward-compatible old API
# -----------------------------

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