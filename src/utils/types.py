from dataclasses import dataclass
import numpy as  np
from src.utils.se3 import SE3   

@dataclass      
class DetectedObject:
    object_id: str
    point_cloud: np.ndarray
    pose_base: SE3 | None = None
    id_confidence: float
    pose_confidence: float = 0.0
    id_confidence: float = 0.0


@dataclass
class Scene:
    point_cloud: np.ndarray
    objects:  list[DetectedObject]
