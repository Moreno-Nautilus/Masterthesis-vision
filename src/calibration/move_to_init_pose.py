"""Drives both LBR arms to the saved initial pose in config/robot_init_pose.yaml.

That file is written by capture_robot_init_pose.py (snapshot the robot's
CURRENT pose) and consumed here to drive back to it -- e.g. after a session
has moved the arms elsewhere, or at the start of one, to guarantee a known
starting configuration.

Uses the exact same mechanism as capture_handeye_data.py's --mode replay:
JointTarget / DualArmMoveitClient.move_to_joint() (moveit_dual_arm.py) --
a JointConstraint per joint, no IK involved, so this reproduces the literal
captured configuration rather than some other configuration that happens to
reach an equivalent flange pose. Both arms move through a single
simultaneous "both_arms_flange" goal by default (see --arm to restrict to
one arm), same as capture_handeye_data.py's _replay_pair().

Prerequisite: lbr_dual_arm_bringup's move_group.launch.py already running,
plus ONE of its two controller bring-ups (real hardware, not mock -- this is
meant to park the physical arms at a known pose), selected via
--control-mode:
  position (default)  -- hardware.launch.py (joint_trajectory_controller).
                          Plans+executes a single MoveGroup goal, both arms
                          at once via the "both_arms_flange" composite group
                          when --arm both (see JointTarget docstring).
  cartesian_impedance  -- cartesian_impedance.launch.py
                          (cartesian_impedance_lbr_one/_two, torque mode --
                          see src/calibration/cartesian_impedance_dual_arm.py).
                          MoveIt still PLANS the move (same collision-checked
                          planner as position mode, via plan_joint_trajectory()
                          -- plan-only, never executed by MoveGroup); the
                          impedance controller only EXECUTES it, via
                          execute_planned_trajectory(), which resamples the
                          planned waypoints and streams each one to the
                          controller as an FK'd Cartesian target + nullspace
                          bias. That controller has no both-arms composite
                          goal, so --arm both plans+runs the two arms one
                          after another rather than simultaneously (unlike
                          position mode). This mode still needs
                          move_group.launch.py running -- for the planning
                          itself, not just the readiness probe.

See scripts/launch_robots_to_init_pose.sh, which brings the right pair up
(via CONTROL_MODE) and then runs this. This script itself waits (up to
--timeout-s) for the MoveGroup action server AND a valid planning state,
since either controller bring-up additionally needs the KUKA FRI application
streaming from each pendant before any goal can actually plan -- see
docs/moveit_robot_control.md.

Required LBRServer (pendant) app settings per --control-mode -- these must
be set on the pendant BEFORE starting the FRI connection, and must match
the bring-up launch file or ros2_control_node just hangs waiting on the
wrong FRI stream; see docs/calibration_control_modes.md's "Pendant
settings" table (source of truth -- keep this in sync with it):
  position (hardware.launch.py)             -- send period 10 ms (100 Hz),
                                                FRI control mode "position",
                                                client command mode POSITION.
  cartesian_impedance (cartesian_impedance.launch.py)
                                             -- send period 1 ms (1000 Hz),
                                                FRI control mode
                                                JOINT_IMPEDANCE_CONTROL,
                                                client command mode TORQUE.

Usage:
    python3 -m src.calibration.move_to_init_pose
    python3 -m src.calibration.move_to_init_pose --arm left
    python3 -m src.calibration.move_to_init_pose --control-mode cartesian_impedance
"""

from __future__ import annotations

import argparse
from pathlib import Path

import rclpy
import yaml
from rclpy.node import Node

from src.calibration.cartesian_impedance_dual_arm import CartesianImpedanceDualArmClient
from src.calibration.flange_pose_store import ARM_KEYS
from src.calibration.moveit_dual_arm import DEFAULT_MOVE_GROUP_NAMESPACE, DualArmMoveitClient, JointTarget

DEFAULT_CONFIG_PATH = Path("config/robot_init_pose.yaml")

# hardware.launch.py's MoveGroup action server/current-state monitor can be
# up well before the arms themselves are actually plannable -- that also
# needs the FRI app started on each pendant (a manual, per-robot-controller
# step), which can take a while after this script starts. Generous default
# so the operator doesn't have to race it; overridable via --timeout-s.
DEFAULT_TIMEOUT_S = 180.0


def _load_joint_positions(config_path: Path, arm_key: str) -> dict[str, float]:
    data = yaml.safe_load(config_path.read_text())
    if arm_key not in data:
        raise KeyError(f"{config_path} has no {arm_key!r} entry.")
    joints = data[arm_key].get("joint_positions") or {}
    if not joints:
        raise ValueError(f"{config_path}[{arm_key!r}] has no joint_positions.")
    return {name: float(position) for name, position in joints.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", choices=["left", "right", "both"], default="both")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="path to robot_init_pose.yaml")
    parser.add_argument("--robot-namespace", default=DEFAULT_MOVE_GROUP_NAMESPACE)
    parser.add_argument(
        "--timeout-s", type=float, default=DEFAULT_TIMEOUT_S,
        help="how long to wait for MoveGroup + a plannable state before giving up",
    )
    parser.add_argument(
        "--control-mode", choices=["position", "cartesian_impedance"], default="position",
        help=(
            "position (default): MoveGroup-executed joint_trajectory_controller move "
            "(hardware.launch.py). cartesian_impedance: compliant move through "
            "cartesian_impedance_lbr_one/_two (cartesian_impedance.launch.py) -- see "
            "module docstring. Requires move_group.launch.py running either way."
        ),
    )
    args = parser.parse_args()

    if not args.config.exists():
        raise FileNotFoundError(
            f"{args.config} not found -- run `python3 -m src.calibration.capture_robot_init_pose` "
            f"once (with the robot at the pose you want as 'init') to create it."
        )

    arm_keys = ["left", "right"] if args.arm == "both" else [args.arm]
    targets = [
        JointTarget(
            group_name=ARM_KEYS[arm_key]["group_name"],
            joint_positions=_load_joint_positions(args.config, arm_key),
            label=f"{arm_key} init_pose",
        )
        for arm_key in arm_keys
    ]
    group_name = "both_arms_flange" if len(targets) == 2 else None

    rclpy.init()
    node = Node("move_to_init_pose")
    moveit = DualArmMoveitClient(node, namespace=args.robot_namespace)
    try:
        print(f"[{'+'.join(arm_keys)}] waiting for /{args.robot_namespace}/move_action (MoveGroup)...")
        if not moveit.wait_for_server(timeout_s=args.timeout_s):
            raise RuntimeError(
                "MoveGroup action server not available -- is lbr_dual_arm_bringup's "
                "move_group.launch.py running?"
            )

        print(f"[{'+'.join(arm_keys)}] waiting for a plannable state "
              f"(needs the FRI app streaming from each pendant)...")
        if not moveit.wait_for_valid_state_joint(targets, group_name=group_name, timeout_s=args.timeout_s):
            raise RuntimeError(
                "move_group never reported a valid/plannable state -- see the warnings above "
                "and move_group's own log."
            )

        if args.control_mode == "position":
            print(f"[{'+'.join(arm_keys)}] moving to saved init pose ({args.config})...")
            ok, record = moveit.move_to_joint(targets, group_name=group_name)
            if not ok:
                raise RuntimeError(f"MoveGroup failed to reach the saved init pose: {record}")
        else:
            # cartesian_impedance_lbr_one/_two are two independent
            # controllers (no both-arms composite goal like
            # "both_arms_flange"), so --arm both plans+runs them one at a
            # time here rather than simultaneously -- see module docstring.
            # MoveIt plans (plan_joint_trajectory, collision-checked, same
            # planner as position mode); the controller only executes
            # (execute_planned_trajectory streams the planned waypoints).
            compliant = CartesianImpedanceDualArmClient(node, namespace=args.robot_namespace)
            for arm_key, target in zip(arm_keys, targets):
                # wait_for_valid_state_joint() above only checks MoveGroup
                # can plan -- it can succeed before this arm's controller
                # (or its FRI hardware connection) actually exists. Wait for
                # the controller itself too, or execute_planned_trajectory()
                # below can hit an unavailable set_parameters service.
                print(f"[{arm_key}] waiting for cartesian_impedance controller...")
                if not compliant.wait_for_controller(arm_key, timeout_s=args.timeout_s):
                    raise RuntimeError(
                        f"cartesian_impedance controller not available for arm_key={arm_key!r} -- "
                        f"is cartesian_impedance.launch.py running, and has the FRI app been "
                        f"started on that pendant?"
                    )
                print(f"[{arm_key}] planning to saved init pose ({args.config})...")
                traj = compliant.plan_joint_trajectory([target], group_name=target.group_name)
                if traj is None:
                    raise RuntimeError(f"MoveGroup failed to plan a trajectory to the saved init pose for arm_key={arm_key!r}.")
                print(f"[{arm_key}] executing planned trajectory via cartesian impedance controller...")
                if not compliant.execute_planned_trajectory(arm_key, traj):
                    raise RuntimeError(
                        f"cartesian impedance controller failed to execute the planned trajectory for arm_key={arm_key!r} "
                        f"-- is cartesian_impedance.launch.py running (not hardware.launch.py)?"
                    )
        print(f"[{'+'.join(arm_keys)}] reached init pose.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
