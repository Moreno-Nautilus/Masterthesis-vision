from __future__ import annotations

from dataclasses import dataclass, field
import time
import numpy as np

from src.perception.view import View
from src.perception.pipeline_fast import GraspPerceptionPipeline, SceneResult
from src.perception.fusion_fast import fuse_views_to_points_base_with_colors, RGBMaskConfig


@dataclass
class MultiViewConfig:
    voxel_size_fusion: float = 0.005
    stride: int =  1

    zmin: float = 0.30
    zmax: float = 1.05

    #rgb_mask: RGBMaskConfig = field(
    #    default_factory=lambda: RGBMaskConfig(mode="chroma", min_chroma=15, min_v_chroma=25))
    rgb_mask: RGBMaskConfig = field(
        default_factory=lambda: RGBMaskConfig(mode="chroma", min_chroma=15, min_v_chroma=25)
    )
    roi_x_min: float = -0.35
    roi_x_max: float = 0.45
    roi_y_min: float = -0.35
    roi_y_max: float = 0.45
    roi_z_min: float = 0.30
    roi_z_max: float = 1.05

    max_points_after_roi: int = 22000


class MultiViewRunner:
    def __init__(self, pipe: GraspPerceptionPipeline, cfg: MultiViewConfig | None = None):
        self.pipe = pipe
        self.cfg = cfg or MultiViewConfig()

        self.last_colors_rgb: np.ndarray | None = None
        self.last_points_world_raw: np.ndarray | None = None
        self.last_points_world_roi: np.ndarray | None = None

    def _cap_points(
        self, pts: np.ndarray, cols: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        if pts.shape[0] <= self.cfg.max_points_after_roi:
            return pts, cols

        stride = max(1, int(np.ceil(pts.shape[0] / self.cfg.max_points_after_roi)))
        pts = pts[::stride]
        if cols is not None:
            cols = cols[::stride]
        return pts, cols

    def run(self, views: list[View]) -> SceneResult:
        t0 = time.perf_counter()


        debug_pts_base_raw, _ = fuse_views_to_points_base_with_colors(
            views,
            voxel_size=self.cfg.voxel_size_fusion,
            stride=self.cfg.stride,
            zmin=self.cfg.zmin,
            zmax=self.cfg.zmax,
            rgb_mask_cfg=RGBMaskConfig(mode="none"),
        )
        debug_pts_base_raw = np.asarray(debug_pts_base_raw, dtype=np.float32).reshape(-1, 3)


        
        pts_base_raw, cols_rgb = fuse_views_to_points_base_with_colors(
            views,
            voxel_size=self.cfg.voxel_size_fusion,
            stride=self.cfg.stride,
            zmin=self.cfg.zmin,
            zmax=self.cfg.zmax,
            rgb_mask_cfg=self.cfg.rgb_mask,
        )

        t1 = time.perf_counter()

        pts_base_raw = np.asarray(pts_base_raw, dtype=np.float32).reshape(-1, 3)
        self.last_points_world_raw = pts_base_raw

        if pts_base_raw.size == 0:
            self.last_points_world_roi = np.zeros((0, 3), dtype=np.float32)
            self.last_colors_rgb = cols_rgb
            result = self.pipe.run(pts_base_raw)
            result.debug_points_world_raw = debug_pts_base_raw
            t2 = time.perf_counter()
            print(
                f"[TIMING pipeline_multiview_fast] "
                f"fusion={(t1 - t0) * 1000:.1f} ms | "
                f"roi+cap=0.0 ms | "
                f"pipe={(t2 - t1) * 1000:.1f} ms | "
                f"raw_pts=0 roi_pts=0"
            )
            return result

        m = (
            (pts_base_raw[:, 0] > self.cfg.roi_x_min)
            & (pts_base_raw[:, 0] < self.cfg.roi_x_max)
            & (pts_base_raw[:, 1] > self.cfg.roi_y_min)
            & (pts_base_raw[:, 1] < self.cfg.roi_y_max)
            & (pts_base_raw[:, 2] > self.cfg.roi_z_min)
            & (pts_base_raw[:, 2] < self.cfg.roi_z_max)
        )

        pts_base_roi = pts_base_raw[m]
        if cols_rgb is not None:
            cols_rgb = cols_rgb[m]

        pts_base_roi, cols_rgb = self._cap_points(pts_base_roi, cols_rgb)

        t2 = time.perf_counter()

        self.last_points_world_roi = pts_base_roi
        self.last_colors_rgb = cols_rgb

        result = self.pipe.run(pts_base_roi)

        t3 = time.perf_counter()

        result.points_world_raw = self.last_points_world_raw
        result.points_world_roi = self.last_points_world_roi
        result.debug_points_world_raw = debug_pts_base_raw


        print(
            f"[TIMING pipeline_multiview_fast] "
            f"fusion={(t1 - t0) * 1000:.1f} ms | "
            f"roi+cap={(t2 - t1) * 1000:.1f} ms | "
            f"pipe={(t3 - t2) * 1000:.1f} ms | "
            f"raw_pts={pts_base_raw.shape[0]} roi_pts={pts_base_roi.shape[0]}"
        )

        return result