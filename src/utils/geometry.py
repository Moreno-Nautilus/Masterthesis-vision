import numpy as np

def random_rot_matrix()-> np.ndarray:
    # random rot via QR decomposition
    A = np.random.randn(3,3)
    Q,_ = np.linalg.qr(A)
    # Ensure right-handed
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q

def rotation_error_deg(R_est: np.ndarray, R_gt:np.ndarray)-> float:
    R = R_est@R_gt.T
    cos  = (np.trace(R)-1.0)/2.0
    cos = np.clip(cos, -1.0,1.0)
    return float(np.degrees(np.arccos(cos)))

