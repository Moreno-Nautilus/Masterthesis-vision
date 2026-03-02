from __future__ import annotations

import numpy as np

from src.utils.se3 import SE3
from src.utils.geometry import random_rot_matrix, rotation_error_deg


def main() -> None:
    rng = np.random.default_rng(0)

    R = random_rot_matrix()
    t = np.array([0.1, -0.2, 0.3], dtype=float)
    T = SE3(R, t)

    pts = rng.normal(size=(100, 3))
    pts2 = T.transform_points(pts)

    T_inv = T.inverse()
    pts_back = T_inv.transform_points(pts2)

    max_err = float(np.max(np.linalg.norm(pts_back - pts, axis=1)))
    rot_err = float(rotation_error_deg(T.R, R))

    print("SE3 test results")
    print("  max point error:", max_err)
    print("  rot error [deg]:", rot_err)

    # hard asserts so it fails loudly
    assert max_err < 1e-9, f"Roundtrip error too large: {max_err}"
    assert rot_err < 1e-12, f"Rotation mismatch too large: {rot_err}"

    # composition sanity
    T2 = SE3(random_rot_matrix(), rng.normal(size=3))
    M = (T @ T2).as_matrix()
    M_ref = T.as_matrix() @ T2.as_matrix()
    assert np.allclose(M, M_ref, atol=1e-9), "SE3 @ composition mismatch"

    print("OK")


if __name__ == "__main__":
    main()