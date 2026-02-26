from src.utils.se3  import SE3

def fuse_poses(poses: list[SE3], weights: list[float]) -> SE3:
    """
    Weighted SE3 averaging.
    """