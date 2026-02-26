import  numpy as np

def depth_to_pointcloud(depth,instrinsics,extrinsics)-> np.ndarray:
    """
    Convert depth image to Nx3 ptcloud in base frame
    """