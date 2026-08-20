"""Shared pose-perturbation math for augmenting a small set of known-good
flange poses into a larger one, used by both capture_handeye_data.py's
`--mode augment` (hand-eye Stage A, target 10 samples/arm) and
autocalibrate_dual_realsense.py's board-pose stage (target 5 samples total).

Standard procedure (always, regardless of how many arms are connected or
which stage is calibrating): replay whatever poses are already known-good
for an arm, then -- for however many more samples are still needed to reach
the stage's target count -- repeatedly (1) perturb the MOST RECENTLY
accepted pose (last known-good one first, then whichever perturbation was
just accepted -- see sample_augmented_pose, NOT a random pick from the
whole pool) by a small random Cartesian offset, (2) drive the arm there via
IK and attempt a checkerboard detection, (3) accept it (appending it to the
pool, so it becomes the new "most recent") or discard it and perturb the
same most-recent pose again. Only the perturbation math lives here -- driving
the arm and deciding what "success" means (board detection vs something
else) differs enough between the two call sites that the retry loop itself
is written at each call site instead of forced through a shared abstraction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.utils.se3 import SE3


@dataclass
class PoseAugmentationConfig:
    max_translation_m: float = 0.03    # +/-3cm per linear axis (x, y, z independently)
    # Max magnitude of ONE combined 3D rotation (axis-angle, not per-axis
    # roll/pitch/yaw -- see perturb_se3). 12deg, not 7: composing 3
    # independent +/-7deg Euler perturbations could exceed 7deg of actual
    # combined rotation anyway, so this replaces that with an explicit,
    # correctly-bounded single-rotation magnitude instead.
    max_rotation_deg: float = 12.0
    # Gaussian sigma, as a fraction of each max -- 1/3 puts the hard clip
    # below at roughly 3-sigma, so it only trims rare outlier draws rather
    # than flattening the distribution into a uniform one.
    sigma_fraction: float = 1.0 / 3.0


def _axis_angle_to_R(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rodrigues' rotation formula: R for a rotation by `angle_rad` about
    the unit vector `axis`."""
    x, y, z = axis
    K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle_rad) * K + (1.0 - np.cos(angle_rad)) * (K @ K)


def perturb_se3(base: SE3, rng: np.random.Generator, cfg: PoseAugmentationConfig) -> SE3:
    """One randomly-perturbed candidate pose near `base`.

    Translation offset is drawn per-axis (x, y, z independently) in `base`'s
    own frame (e.g. the arm base frame T_armBase_flange already lives in)
    and added directly to base.t.

    Rotation offset is ONE random 3D rotation -- a uniformly random axis
    (not one of roll/pitch/yaw) with a magnitude clipped to
    max_rotation_deg -- NOT three independent per-axis (roll/pitch/yaw)
    perturbations composed together. That composition doesn't actually
    bound the resulting rotation to max_rotation_deg (three independent
    +/-max draws can combine to a larger net rotation) and isn't an
    isotropic sample on SO(3) either. Applied in the pose's OWN local frame
    (base.R @ R_delta, not R_delta @ base.R) so it reads as "tilt the
    flange/camera view a bit off-axis" rather than "rotate about the
    distant base origin".
    """
    t_sigma = cfg.max_translation_m * cfg.sigma_fraction
    r_sigma = cfg.max_rotation_deg * cfg.sigma_fraction

    d_t = np.clip(rng.normal(0.0, t_sigma, size=3), -cfg.max_translation_m, cfg.max_translation_m)

    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle_deg = np.clip(abs(rng.normal(0.0, r_sigma)), 0.0, cfg.max_rotation_deg)

    R_delta = _axis_angle_to_R(axis, np.deg2rad(angle_deg))
    return SE3(base.R @ R_delta, base.t + d_t)


def sample_augmented_pose(pool: list[SE3], rng: np.random.Generator, cfg: PoseAugmentationConfig) -> SE3:
    """Perturbs `pool[-1]` -- the MOST RECENT already-successful pose (the
    last known-good one on the first call, then whichever perturbation was
    just accepted, since callers append to `pool` in acceptance order) --
    and returns one perturbed candidate near it.

    Deliberately NOT a random pick from the whole pool: consecutive
    Cartesian IK targets that jump between far-apart poses in `pool` can
    force this 7-DOF redundant arm into a very different elbow/null-space
    configuration to reach each one, which MoveIt's planner can resolve
    with a large, visually pointless joint-space swing (and, closer to a
    joint limit, an even larger detour to avoid it) even though the
    Cartesian motion requested was small. Perturbing from the most recent
    pose instead keeps every step local, so consecutive targets stay close
    in both Cartesian AND joint space."""
    base = pool[-1]
    return perturb_se3(base, rng, cfg)
