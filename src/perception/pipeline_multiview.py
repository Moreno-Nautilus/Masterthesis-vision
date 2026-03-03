from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from src.perception.view import View
from src.perception.fusion import fuse_views_to_points_base
from src.perception.pipeline import GraspPerceptionPipeline, SceneResult


@dataclass
class MultiViewConfig:
    # Fusion / backprojection settings
    voxel_size_fusion: float = 0.003
    stride: int = 1
    zmin: float = 0.25
    zmax: float = 1.6

    # EARLY ROI crop in "base" frame (currently cam1 if cam1 extrinsic is identity)
    roi_x_min: float = -0.7
    roi_x_max: float = 0.7
    roi_y_min: float = -0.7
    roi_y_max: float = 0.7
    roi_z_min: float = 0.25
    roi_z_max: float = 1.6


class MultiViewRunner:
    def __init__(self, pipe: GraspPerceptionPipeline, cfg: MultiViewConfig | None = None):
        self.pipe = pipe
        self.cfg = cfg or MultiViewConfig()

    def run(self, views: list[View]) -> SceneResult:
        pts_base_raw = fuse_views_to_points_base(
            views,
            voxel_size=self.cfg.voxel_size_fusion,
            stride=self.cfg.stride,
            zmin=self.cfg.zmin,
            zmax=self.cfg.zmax,
        )

        # --- EARLY ROI CROP (pre-plane) ---
        if pts_base_raw.size != 0:
            m = (
                (pts_base_raw[:, 0] > self.cfg.roi_x_min) & (pts_base_raw[:, 0] < self.cfg.roi_x_max) &
                (pts_base_raw[:, 1] > self.cfg.roi_y_min) & (pts_base_raw[:, 1] < self.cfg.roi_y_max) &
                (pts_base_raw[:, 2] > self.cfg.roi_z_min) & (pts_base_raw[:, 2] < self.cfg.roi_z_max)
            )
            pts_base_raw = pts_base_raw[m]
        # --- END ROI CROP ---

        return self.pipe.run(pts_base_raw)