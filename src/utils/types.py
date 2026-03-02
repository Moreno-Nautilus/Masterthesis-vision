from dataclasses import dataclass
import numpy as  np
from src.utils.se3 import SE3   

@dataclass      
class DetectedObject:
    object_id: str
    point_cloud: np.ndarray
    T_obj_to_world: SE3 | None = None
    pose_confidence: float = 0.0
    id_confidence: float = 0.0
    metrics: dict[str, float] | None = None


@dataclass
class Scene:
    point_cloud: np.ndarray
    objects:  list[DetectedObject]
