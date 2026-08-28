"""Dataclasses for per-camera / per-object tracking state.

Extracted from run_pipeline_track_multicam_realsense.py.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.perception.learned.SAM.sam_segmentation import SAMMaskCandidate
from src.perception.ros.learn_runners.PoseKalmanFilter import PoseKalmanFilter


# Per-camera, per-object tracking state carried between ticks (pose, masks, mode, Kalman).
@dataclass
class ObjectTrackState:
    object_id: str
    mesh_path: str
    track_id: str = ""
    mode: str = "search"
    T_object_camera: Optional[np.ndarray] = None
    dino_score: float = 0.0
    lost_count: int = 0
    last_mask_area: int = 0
    degraded_count: int = 0
    id_history: deque = field(default_factory=lambda: deque(maxlen=5))
    track_pose_convention: str = "raw"
    recovery_mask: Optional[np.ndarray] = None
    last_good_mask: Optional[np.ndarray] = None
    last_good_T: Optional[np.ndarray] = None
    kalman: Optional[PoseKalmanFilter] = None

    last_logged_T_base: Optional[np.ndarray] = None
    last_logged_convention: Optional[str] = None

    def __post_init__(self):
        if self.kalman is None:
            self.kalman = PoseKalmanFilter()


# One accepted detection for an object on a camera (chosen mask + its DINO scores).
@dataclass
class CandidateSelection:
    object_id: str
    score: float
    scores_by_object: dict[str, float]
    candidate: SAMMaskCandidate
    base_scores_by_object: dict[str, float] = field(default_factory=dict)
    score_source: str = "dino_base"
    dino_class_score: float = 0.0
    dino_margin: float = 0.0


# Per-camera SAM mask-filtering parameters + ROI polygon.
@dataclass
class CameraSAMParams:
    min_mask_area: int
    min_bbox_side_px: int
    max_mask_area_ratio: float
    max_bbox_area_ratio: float
    border_px: int
    max_border_fraction: float
    roi_polygon: np.ndarray
