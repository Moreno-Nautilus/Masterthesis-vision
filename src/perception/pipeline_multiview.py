from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from src.perception.view import View
from src.perception.pipeline import GraspPerceptionPipeline, SceneResult
from src.perception.fusion import fuse_views_to_points_base_with_colors, RGBMaskConfig


@dataclass
class MultiViewConfig:
    # Fusion / backprojection settings
    voxel_size_fusion: float = 0.003
    stride: int = 1
    zmin: float = 0.25
    zmax: float = 1.6

    # Option 2: RGB masking (classical)
    rgb_mask: RGBMaskConfig = RGBMaskConfig(mode="chroma", min_chroma=15, min_v_chroma=25)
    # Set to RGBMaskConfig(mode="none") to disable masking quickly.

    # EARLY ROI crop in "base" frame (currently cam1 if cam1 extrinsic is identity)
    roi_x_min: float = -0.8
    roi_x_max: float = 0.8
    roi_y_min: float = -0.8
    roi_y_max: float = 0.8
    roi_z_min: float = 0.2
    roi_z_max: float = 1.6


class MultiViewRunner:
    def __init__(self, pipe: GraspPerceptionPipeline, cfg: MultiViewConfig | None = None):
        self.pipe = pipe
        self.cfg = cfg or MultiViewConfig()

        # handy for later debug/publishing (optional)
        self.last_colors_rgb: np.ndarray | None = None

    def run(self, views: list[View]) -> SceneResult:
        pts_base_raw, cols_rgb = fuse_views_to_points_base_with_colors(
            views,
            voxel_size=self.cfg.voxel_size_fusion,
            stride=self.cfg.stride,
            zmin=self.cfg.zmin,
            zmax=self.cfg.zmax,
            rgb_mask_cfg=self.cfg.rgb_mask,
        )

        # --- EARLY ROI CROP (pre-plane) ---
        if pts_base_raw.size != 0:
            m = (
                (pts_base_raw[:, 0] > self.cfg.roi_x_min) & (pts_base_raw[:, 0] < self.cfg.roi_x_max) &
                (pts_base_raw[:, 1] > self.cfg.roi_y_min) & (pts_base_raw[:, 1] < self.cfg.roi_y_max) &
                (pts_base_raw[:, 2] > self.cfg.roi_z_min) & (pts_base_raw[:, 2] < self.cfg.roi_z_max)
            )
            pts_base_raw = pts_base_raw[m]
            # cols_rgb = cols_rgb[m]
            if cols_rgb is not None:
                cols_rgb = cols_rgb[m]
        # --- END ROI CROP ---

        # store for debugging (optional)
        self.last_colors_rgb = cols_rgb

        return self.pipe.run(pts_base_raw)