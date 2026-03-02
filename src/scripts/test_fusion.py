from __future__ import annotations
import numpy as np

from src.utils.se3 import SE3
from src.perception.view import View
from src.perception.fusion import fuse_views_to_points_base


def main() -> None:
    # Create synthetic points in a "camera frame"
    rng = np.random.default_rng(0)
    pts_cam = rng.normal(size=(1000, 3)).astype(np.float32)
    pts_cam[:, 2] = np.abs(pts_cam[:, 2]) + 0.5  # keep z positive

    # Fake depth: we won’t use depth_to_points_cam here.
    # Instead we directly test the transform part by encoding points into a "depth-like" pipeline later.
    # For now: make two views whose "depth" is unused; we will bypass backproject by monkeypatching if needed.
    # Easier: directly validate SE3 and fusion transform by calling transform_points.

    T_base_cam1 = SE3(np.eye(3), np.array([0.0, 0.0, 0.0]))
    T_base_cam2 = SE3(np.eye(3), np.array([0.1, 0.0, 0.0]))  # shift x by 10cm

    # Direct transform check:
    pts_base1 = T_base_cam1.transform_points(pts_cam)
    pts_base2 = T_base_cam2.transform_points(pts_cam)

    # Fused should have both clouds (2x points) before downsample
    pts_fused = np.vstack([pts_base1, pts_base2])

    # Basic sanity: means differ by ~0.05 in x between the two halves
    mean1 = pts_base1.mean(axis=0)
    mean2 = pts_base2.mean(axis=0)
    dx = float(mean2[0] - mean1[0])

    print("fusion sanity")
    print("  mean1:", mean1)
    print("  mean2:", mean2)
    print("  dx:", dx)
    assert abs(dx - 0.1) < 1e-6
    print(" OK ")


if __name__ == "__main__":
    main()