import  numpy as np
from utils.se3  import SE3         

def  estimate_pose_icp(cluster_points: np.ndarray, cad_model_points: np.ndarray) -> SE3:
    """
    Run ICP and  retuirn SE3 pose
    """