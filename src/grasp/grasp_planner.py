from grasp_library import GRASP_LIBRARY    
from src.utils.se3  import SE3


def compute_grasp_pose(object_pose: SE3, object_id: str) -> SE3:
    grasp_offset = GRASP_LIBRARY[object_id]
    return object_pose.compose(grasp_offset)