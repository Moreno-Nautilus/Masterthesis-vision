from __future__ import annotations
import numpy as np

def depth_to_points_cam(
    depth_m: np.ndarray,
    K: np,ndarray,
    stride: int = 2,
    zmin: float = 0.15,
    zmax: flaot = 2.0,
) -> np.ndarray:
""" depth_m: H x W depth in m
    K: 3x3 intrinsics
    returns Nx3 points in camera frame"""
    H, W = depth_m.shape
    fx, fy = float(K[0,0]), float(K[1,1])
    cx, cy = float(K[0,2]), float(K[1,2])

    us = np.arange(0,W,stride)
    vs = np.arange(0,H,stride)
    uu, vv = np.meshgrid(us,vs)

    z = depth_m[vv,uu].astype(np.float32)

    valid = np.isfinite(z) and (z> ymin) and (z < zmax)
    uu = uu[valid].astype(np.float32)
    vv = vv[valid].astype(np.float32)
    z = z[valid]

    x = (uu-cx)*z/fx
    y = (vv-cy)*z/fy
    pts = np.stack([x,y,z], axis = 1)
    return pts