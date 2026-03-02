from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from src.perception.view import View
from src.perception.fusion import fuse_views_to_points_base
from src.perception.pipeline import GraspPerceptionPipeline, SceneResult


@dataclass
class MultiViewConfig:
    voxel_size_fusion: float = 0.005
    stride: int = 2
    zmin: float = 0.15
    zmax: float = 2.0


class MultiViewRunner:
    def __init__(self, pipe: GraspPerceptionPipeline, cfg: MultiViewConfig | None = None):
        self.pipe = pipe
        self.cfg = cfg or MultiViewConfig()

    def run(self, views: list[View]) -> SceneResult:
        pts_base = fuse_views_to_points_base(
            views,
            voxel_size=self.cfg.voxel_size_fusion,
            stride=self.cfg.stride,
            zmin=self.cfg.zmin,
            zmax=self.cfg.zmax,
        )
        return self.pipe.run(pts_base)