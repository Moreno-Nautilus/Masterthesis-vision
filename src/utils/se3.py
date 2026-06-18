from __future__ import annotations
import numpy as np
from dataclasses import dataclass


# Minimal immutable rigid transform (rotation + translation) in 3D.
@dataclass (frozen=True)
class SE3:

    R: np.ndarray #3x3 Rot Matrix
    t: np.ndarray #3x1 Transl Vector

    def __post_init__(self):
        # Coerce inputs to float arrays and reject anything that isn't a 3x3 R / 3-vector t.
        object.__setattr__(self, "R", np.asarray(self.R, dtype =float))
        object.__setattr__(self, "t", np.asarray(self.t,dtype=float).reshape(3,))
        if self.R.shape != (3,3):
            raise ValueError(f"R must be (3,3), got {self.R.shape}")
        if self.t.shape != (3,):
            raise ValueError(f"t must be (3,), got {self.t.shape}")

    @staticmethod
    def identity()->"SE3":
        # The no-op transform.
        return  SE3(np.eye(3),np.zeros(3))

    @staticmethod
    def from_matrix(T:np.ndarray) -> "SE3":
        # Split a 4x4 homogeneous matrix into its R and t blocks.
        T =  np.asarray(T , dtype=float)
        if T.shape != (4,4):
            raise ValueError(f"T must be (4,4), got {T.shape}")
        return SE3(T[:3, :3], T[:3, 3])


    def as_matrix(self) -> np.ndarray:
        # Pack R and t back into a 4x4 homogeneous matrix.
        T = np.eye(4)
        T[:3, :3] = self.R
        T[:3, 3] = self.t
        return T

    def inverse(self)-> "SE3":
        # Inverse rigid transform: Rᵀ with the translation rotated back.
        R_inv = self.R.T
        t_inv = -R_inv @  self.t
        return SE3(R_inv,t_inv)

    def compose(self,other:"SE3") -> "SE3":
        # Chain two transforms (self then other), i.e. self · other.
        R = self.R @ other.R
        t = self.R @ other.t +self.t
        return SE3(R,t)

    def __matmul__(self, other: "SE3") -> "SE3":
        # Let `a @ b` mean compose.
        if not isinstance(other, SE3):
            return NotImplemented
        return self.compose(other)

    def is_valid(self, atol: float = 1e-6) -> bool:
        """Checks R is a proper rot matrix within tolerance"""
        # Orthonormal (RᵀR ≈ I), finite t, and det ≈ +1 (no reflection).
        RtR = self.R.T @self.R
        if not np.allclose(RtR, np.eye(3), atol = atol):
            return False
        if not np.isfinite(self.t).all():
            return False
        det = np.linalg.det(self.R)
        return np.isfinite(det) and abs(det- 1.0) < 1e-4

    def __repr__(self)-> str:
        return f"SE3(R=\n{np.array_str(self.R, precision=3, suppress_small=True)},\n t={np.array_str(self.t, precision=3, suppress_small=True)})"
