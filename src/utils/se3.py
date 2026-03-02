from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass (frozen=True)
class SE3:
    
    R: np.ndarray #3x3 Rot Matrix
    t: np.ndarray #3x1 Transl Vector

    def __post_init__(self):
        object.__setattr__(self, "R", np.asarray(self.R, dtype =float))
        object.__setattr__(self, "t", np.asarray(self.t,dtype=float).reshape(3,))
        if self.R.shape != (3,3):
            raise ValueError(f"R must be (3,3), got {self.R.shape}")
        if self.t.shape != (3,):
            raise ValueError(f"t must be (3,), got {self.t.shape}")
    
    @staticmethod
    def identity()->"SE3":
        return  SE3(np.eye(3),np.zeros(3))
    
    @staticmethod
    def from_matrix(T:np.ndarray) -> "SE3":
        T =  np.asarray(T , dtype=float)
        if T.shape != (4,4):
            raise ValueError(f"T must be (4,4), got {T.shape}")
        return SE3(T[:3, :3], T[:3, 3])
    

    def as_matrix(self) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = self.R
        T[:3, 3] = self.t
        return T
    
    def inverse(self)-> "SE3":
        R_inv = self.R.T
        t_inv = -R_inv @  self.t
        return SE3(R_inv,t_inv)
    
    def compose(self,other:"SE3") -> "SE3":
        R = self.R @ other.R
        t = self.R @ other.t +self.t
        return SE3(R,t)

    def __matmul__(self, other: "SE3") -> "SE3":
        if not isinstance(other, SE3):
            return NotImplemented
        return self.compose(other)
    
    def transform_points(self, pts: np.ndarray)-> np.ndarray:
        pts = np.asarray(pts,dtype= float)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"pts  must be (N,3),  got {pts.shape} ")
        return (self.R @ pts.T).T+ self.t

    def transform_point(self, p: np.ndarray)-> np.ndarray:
        p = np.asarray(p, dtype = float).reshape(3,)
        return self.R @ p + self.t

    def is_valid(self, atol: float = 1e-6) -> bool:
        """Checks R is a proper rot matrix within tolerance"""
        RtR = self.R.T @self.R
        if not np.allclose(RtR, np.eye(3), atol = atol):
            return False
        if not np.isfinite(self.t).all():
            return False
        det = np.linalg.det(self.R)
        return np.isfinite(det) and abs(det- 1.0) < 1e-4

    def __repr__(self)-> str:
        return f"SE3(R=\n{np.array_str(self.R, precision=3, suppress_small=True)},\n t={np.array_str(self.t, precision=3, suppress_small=True)})"