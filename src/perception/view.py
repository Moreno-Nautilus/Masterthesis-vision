from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from src.utils.se3 import SE3

# One camera's synchronized frame: image, depth, intrinsics, and its base-frame extrinsic.
@dataclass
class View:
    cam_id: str
    rgb: np.ndarray
    depth: np.ndarray
    K: np.ndarray
    T_base_cam: SE3
    stamp_s: float | None = None
