import numpy as np
from src.utils.se3 import SE3
from src.utils.geometry import random_rot_matrix, rotation_error_deg

R = random_rot_matrix()
t = np.array([0.1, -0.2, 0.3])
T = SE3(R,t)

pts = np.random.randn(100,3)
pts2 = T.transform_points(pts)

T_inv = T.inverse()
pts_back = T_inv.transform_points(pts2)


print("max point error:", np.max(np.linalg.norm(pts_back -pts, axis = 1)))
print("rot error:", rotation_error_deg(T.R, R))