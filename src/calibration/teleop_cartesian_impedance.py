"""Interactive keyboard teleop for the dual-LBR rig's custom Cartesian
impedance controllers (cartesian_impedance_lbr_one/_two) -- NOT the KUKA
Sunrise cabinet's built-in Cartesian impedance mode. Jogs the flange by
streaming incremental target_frame updates through
CartesianImpedanceDualArmClient.publish_target(), the same client
cartesian_impedance_dual_arm.py uses for scripted moves, but here driven by
live keypresses instead of a saved target -- no MoveGroup goal is ever sent.

Requires `ros2 launch lbr_dual_arm_bringup cartesian_impedance.launch.py`
already running (see cartesian_impedance_dual_arm.py's module docstring).
"MoveIt" here means the same planning-group/frame conventions as
moveit_dual_arm.py (ArmTarget, lbr_{one,two}_link_0 base frames,
arm_one_flange/arm_two_flange group names) -- not MoveGroup execution.

Jog convention: translation increments are applied in the arm's own BASE
frame (lbr_{one,two}_link_0 -- intuitive "world" directions for that arm);
rotation increments are applied in the FLANGE's own (tool) frame, i.e.
roll/pitch/yaw about the flange's current axes. This is the usual jog
convention: world-frame translate, tool-frame rotate.

    python3 -m src.calibration.teleop_cartesian_impedance
    python3 -m src.calibration.teleop_cartesian_impedance --arm left
    python3 -m src.calibration.teleop_cartesian_impedance --gain-profile holding
"""

from __future__ import annotations

import argparse
import sys
import termios
import tty

import numpy as np
import rclpy
from rclpy.node import Node

from src.calibration import gain_profiles
from src.calibration.cartesian_impedance_dual_arm import (
    ARM_KEYS,
    CartesianImpedanceDualArmClient,
    GainChangeUnsafeError,
    GainSettings,
)
from src.calibration.moveit_dual_arm import ArmTarget
from src.utils.se3 import SE3

LINEAR_STEP_M = 0.005
ANGULAR_STEP_RAD = 0.02
STEP_SCALE_FACTOR = 1.5  # '['/']' multiply/divide both steps by this

# Cycled through by the 'g' key; profile names must exist in
# config/impedance_gain_profiles.yaml (see gain_profiles.load_impedance_profile()).
GAIN_PROFILE_CYCLE = ("holding", "insertion")

TRANSLATION_KEYS = {  # key -> (axis, sign), arm base frame
    "w": ("x", +1), "s": ("x", -1),
    "d": ("y", +1), "a": ("y", -1),
    "r": ("z", +1), "f": ("z", -1),
}
ROTATION_KEYS = {  # key -> (axis, sign), flange/tool frame
    "o": ("x", +1), "u": ("x", -1),
    "k": ("y", +1), "i": ("y", -1),
    "l": ("z", +1), "j": ("z", -1),
}

HELP_TEXT = """
Cartesian impedance teleop -- press a key (no Enter needed):
  Translation (arm base frame):  w/s = +x/-x   a/d = -y/+y   r/f = +z/-z
  Rotation (tool frame):         u/o = roll -/+   i/k = pitch -/+   j/l = yaw -/+
  [ / ]   halve / double the step size
  1 / 2   select left / right arm as active (only when --arm was omitted)
  g       cycle gain profile (holding <-> insertion) on the active arm
  h       show this help again
  q       quit (holds the last-published target; controller keeps running)
"""


def read_single_key() -> str:
    """Blocking single-character read from stdin, raw mode (no Enter needed)."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


def _rot(axis: str, angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    if axis == "x":
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    if axis == "y":
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])  # "z"


def _flange_group_name(arm_key: str) -> str:
    return "arm_one_flange" if arm_key == "left" else "arm_two_flange"


def _publish(client: CartesianImpedanceDualArmClient, arm_key: str, pose: SE3) -> None:
    info = ARM_KEYS[arm_key]
    target = ArmTarget(
        group_name=_flange_group_name(arm_key),
        base_frame=info["base_frame"],
        tip_link=info["tip_frame"],
        T_armBase_flange=pose,
    )
    client.publish_target(target)


def _apply_gain_profile(client, arm_key: str, profile_name: str) -> None:
    profile = gain_profiles.load_impedance_profile(profile_name)
    client.set_gains(arm_key, GainSettings(**profile))


def _rotation_to_rpy(R: np.ndarray) -> tuple[float, float, float]:
    """Display-only roll/pitch/yaw extraction (ZYX convention) -- not used
    anywhere the actual pose math depends on it, just for human-readable
    --dry-run output."""
    pitch = np.arcsin(-np.clip(R[2, 0], -1.0, 1.0))
    roll = np.arctan2(R[2, 1], R[2, 2])
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return float(roll), float(pitch), float(yaw)


class _DryRunClient:
    """Stand-in for CartesianImpedanceDualArmClient with no ROS/hardware
    involved -- prints what would be published/set instead of touching a
    controller. Lets --dry-run exercise the exact same key-handling and pose
    math as a live run (see docs/teleop_cartesian_impedance_quickstart.md);
    it does not simulate the controller's compliant response, only the
    teleop script's own bookkeeping.
    """

    def current_flange_pose(self, arm_key: str) -> SE3:
        return SE3.identity()

    def publish_target(self, target: ArmTarget) -> None:
        t = target.T_armBase_flange.t
        roll, pitch, yaw = _rotation_to_rpy(target.T_armBase_flange.R)
        print(
            f"[dry-run] {target.group_name} -> target_frame in {target.base_frame}: "
            f"t=({t[0]:+.4f}, {t[1]:+.4f}, {t[2]:+.4f}) m  "
            f"rpy=({roll:+.3f}, {pitch:+.3f}, {yaw:+.3f}) rad",
            flush=True,
        )

    def set_gains(self, arm_key: str, gains: GainSettings) -> None:
        print(f"[dry-run] [{arm_key}] would set_gains({gains})", flush=True)


def main(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument(
        "--arm", default=None, choices=sorted(ARM_KEYS),
        help="Restrict teleop to a single arm. Omit to control both -- '1'/'2' "
             "switch which one is active.",
    )
    parser.add_argument(
        "--gain-profile", default=None, choices=GAIN_PROFILE_CYCLE,
        help="Apply this stiffness profile (config/impedance_gain_profiles.yaml) "
             "to the controlled arm(s) at startup. Defaults to the controller's "
             "own current gains if omitted.",
    )
    parser.add_argument(
        "--linear-step", type=float, default=LINEAR_STEP_M,
        help="Initial per-keypress translation step, meters.",
    )
    parser.add_argument(
        "--angular-step", type=float, default=ANGULAR_STEP_RAD,
        help="Initial per-keypress rotation step, radians.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Skip ROS/hardware entirely -- print each target pose instead of "
             "publishing it, starting from an identity pose per arm (not the real "
             "robot's current pose). For sanity-checking key bindings and step "
             "sizes without a live cartesian_impedance.launch.py bring-up.",
    )
    parsed = parser.parse_args(args=args)

    if not sys.stdin.isatty():
        print(
            "teleop_cartesian_impedance needs an interactive TTY for raw keypresses "
            "(run it directly at a terminal, not through a pipe or non-interactive shell).",
            file=sys.stderr,
        )
        sys.exit(1)

    arm_keys = (parsed.arm,) if parsed.arm is not None else tuple(ARM_KEYS)

    node = None
    if parsed.dry_run:
        client = _DryRunClient()
        print("[dry-run] no ROS/hardware involved -- poses start at identity, not the real robot.", flush=True)
    else:
        rclpy.init()
        node = Node("teleop_cartesian_impedance")
        client = CartesianImpedanceDualArmClient(node)

    targets: dict[str, SE3] = {}
    for arm_key in arm_keys:
        pose = client.current_flange_pose(arm_key)
        if pose is None:
            print(
                f"Could not read current flange pose for arm_key={arm_key!r} -- is "
                f"cartesian_impedance.launch.py running and publishing TF for "
                f"{ARM_KEYS[arm_key]['tip_frame']}?",
                file=sys.stderr,
            )
            if node is not None:
                node.destroy_node()
                rclpy.shutdown()
            sys.exit(1)
        targets[arm_key] = pose

    if parsed.gain_profile is not None:
        for arm_key in arm_keys:
            try:
                _apply_gain_profile(client, arm_key, parsed.gain_profile)
            except GainChangeUnsafeError as e:
                print(f"[{arm_key}] startup gain-profile not applied: {e}", file=sys.stderr)

    active_arm = arm_keys[0]
    linear_step = parsed.linear_step
    angular_step = parsed.angular_step
    gain_idx = GAIN_PROFILE_CYCLE.index(parsed.gain_profile) if parsed.gain_profile else -1

    print(HELP_TEXT, flush=True)
    print(
        f"Active arm: {active_arm}" + ("" if len(arm_keys) == 1 else " (press 1/2 to switch)"),
        flush=True,
    )

    try:
        while True:
            key = read_single_key()
            if key in ("q", "\x03"):  # q or Ctrl-C
                break
            elif key == "h":
                print(HELP_TEXT, flush=True)
            elif key in ("1", "2") and len(arm_keys) > 1:
                candidate = "left" if key == "1" else "right"
                if candidate in arm_keys:
                    active_arm = candidate
                    print(f"Active arm: {active_arm}", flush=True)
            elif key == "[":
                linear_step /= STEP_SCALE_FACTOR
                angular_step /= STEP_SCALE_FACTOR
                print(f"step -> linear={linear_step:.4f} m, angular={angular_step:.4f} rad", flush=True)
            elif key == "]":
                linear_step *= STEP_SCALE_FACTOR
                angular_step *= STEP_SCALE_FACTOR
                print(f"step -> linear={linear_step:.4f} m, angular={angular_step:.4f} rad", flush=True)
            elif key == "g":
                gain_idx = (gain_idx + 1) % len(GAIN_PROFILE_CYCLE)
                profile_name = GAIN_PROFILE_CYCLE[gain_idx]
                try:
                    _apply_gain_profile(client, active_arm, profile_name)
                    print(f"[{active_arm}] gain profile -> {profile_name}", flush=True)
                except GainChangeUnsafeError as e:
                    print(f"[{active_arm}] {e}", flush=True)
            elif key in TRANSLATION_KEYS:
                axis, sign = TRANSLATION_KEYS[key]
                delta = np.zeros(3)
                delta["xyz".index(axis)] = sign * linear_step
                current = targets[active_arm]
                targets[active_arm] = SE3(current.R, current.t + delta)
                _publish(client, active_arm, targets[active_arm])
            elif key in ROTATION_KEYS:
                axis, sign = ROTATION_KEYS[key]
                current = targets[active_arm]
                R_delta = _rot(axis, sign * angular_step)
                targets[active_arm] = SE3(current.R @ R_delta, current.t)
                _publish(client, active_arm, targets[active_arm])
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
