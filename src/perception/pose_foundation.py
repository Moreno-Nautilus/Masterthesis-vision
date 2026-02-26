import numpy as np
from src.utils.se3  import SE3


def estimate_pose_foundation( rgb_crop, depth_crop, cad_model) -> SE3:
    """
    Run  FoundationPose inference
    """