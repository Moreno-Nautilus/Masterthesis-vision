from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from src.utils.se3 import SE3

ROBOT_BASES_YAML = "config/robot_bases.yaml"


# Roll/pitch/yaw (degrees) -> rotation matrix, Rz*Ry*Rx convention.
# Duplicated (not imported) from src/calibration/base_to_cams_calib_3.py to
# keep this a leaf module with no dependency on src/calibration.
def _rpy_deg_to_R(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    r = np.deg2rad(roll_deg)
    p = np.deg2rad(pitch_deg)
    y = np.deg2rad(yaw_deg)

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(r), -np.sin(r)],
        [0, np.sin(r),  np.cos(r)],
    ], dtype=float)

    Ry = np.array([
        [ np.cos(p), 0, np.sin(p)],
        [0,          1, 0],
        [-np.sin(p), 0, np.cos(p)],
    ], dtype=float)

    Rz = np.array([
        [np.cos(y), -np.sin(y), 0],
        [np.sin(y),  np.cos(y), 0],
        [0,          0,         1],
    ], dtype=float)

    return Rz @ Ry @ Rx


def load_robot_bases(path: str | Path = ROBOT_BASES_YAML) -> dict[str, SE3]:
    """Returns {robot_name: T_robotA_robot} for every robot in the yaml,
    i.e. each robot's base pose expressed in robot_a's base frame."""
    cfg = yaml.safe_load(Path(path).read_text())
    out: dict[str, SE3] = {}
    for name, b in cfg["bases"].items():
        t = np.array(b["translation_xyz_m"], dtype=float)
        R = _rpy_deg_to_R(*[float(v) for v in b["rotation_rpy_deg"]])
        out[name] = SE3(R, t)
    return out


def load_active_robot(path: str | Path = ROBOT_BASES_YAML) -> str:
    cfg = yaml.safe_load(Path(path).read_text())
    return str(cfg["active_robot"])


def get_active_robot_base(path: str | Path = ROBOT_BASES_YAML) -> tuple[str, SE3]:
    """Returns (active_robot_name, T_robotA_activeRobot)."""
    cfg = yaml.safe_load(Path(path).read_text())
    active = str(cfg["active_robot"])
    bases = load_robot_bases(path)
    return active, bases[active]


def get_dual_arm_base_link(path: str | Path = ROBOT_BASES_YAML) -> SE3:
    """Returns T_robotA_baseLink: the dual-arm bringup's `base_link` frame
    (the physical midpoint between robot_a's and robot_b's bases, oriented
    the same as both -- see lbr_dual_arm.xacro, which mounts each arm at
    +/-0.42 m in Y off a shared `base_link`), expressed in robot_a's frame.

    Computed as the midpoint of robot_a and robot_b rather than hard-coded,
    so it stays correct if robot_bases.yaml's measured offset is ever
    recalibrated.
    """
    bases = load_robot_bases(path)
    t_mid = 0.5 * (bases["robot_a"].t + bases["robot_b"].t)
    return SE3(bases["robot_a"].R, t_mid)
