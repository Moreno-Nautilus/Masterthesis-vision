"""Simulation-only smoke test for moveit_dual_arm.py's joint-space goal path
(JointTarget / move_to_joint() / plan_only_joint()) -- the "mock script"
twin of mock_reachability_check.py, but for the NEW joint-space replay
mechanism autocalibrate_dual_realsense.py now uses instead of Cartesian
ArmTarget/move_to().

IMPORTANT SCOPE NOTE: this does NOT replay real calibration poses. Those
only exist once capture_flange_poses_dual_handguided.py has actually been
run against the real hand-guided rig (joint_positions in
config/flange_poses/{left,right}.json is empty for every capture on this
checkout right now -- those captures predate that field / were made with
the old MoveIt-jogged capture_flange_poses_dual.py). Instead this script
builds SYNTHETIC joint targets from whatever configuration the mock robot
currently reports (its home/default state) plus a small per-joint
perturbation, and checks that:

  1. plan_only_joint() to the arm's OWN current configuration succeeds
     trivially, for arm_one, arm_two, AND both_arms (both_arms joint-space
     was already confirmed working in the mock_reachability_check.py
     session on 2026-07-30 -- see README_mock_calibration_reachability.md --
     this re-confirms it still holds with the new JointTarget code path).
  2. plan_only_joint() to a real (non-trivial) perturbed target succeeds,
     same three groups.
  3. --execute additionally exercises move_to_joint()'s achieved-state
     readback + outputs/calibration_debug/moveit_joint_targets/*.json save
     path end-to-end (requested vs. actually-achieved joint values should
     match closely once the mock controller settles).

Exits non-zero if any check fails. Run (needs ONLY the mock/sim MoveIt
stack -- no camera launch, no hardware.launch.py):

    ros2 launch lbr_dual_arm_bringup mock.launch.py
    ros2 launch lbr_dual_arm_bringup move_group.launch.py mode:=mock rviz:=false
    python3 -m src.calibration.mock_joint_target_check
    python3 -m src.calibration.mock_joint_target_check --execute
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from src.calibration.flange_pose_store import ARM_KEYS
from src.calibration.moveit_dual_arm import DualArmMoveitClient, JointTarget, JOINT_STATES_TOPIC

PERTURB_RAD = 0.15
JOINT_STATE_WAIT_S = 5.0
READY_TIMEOUT_S = 30.0
READY_RETRY_INTERVAL_S = 1.0


@dataclass
class _CheckResult:
    label: str
    group_name: str
    ok: bool
    error_code: int


def _arm_joint_prefix(arm_key: str) -> str:
    base_frame = ARM_KEYS[arm_key]["base_frame"]
    robot_prefix = base_frame.rsplit("_link_0", 1)[0]
    return f"{robot_prefix}_A"


class _ProbeNode(Node):
    def __init__(self) -> None:
        super().__init__("mock_joint_target_check")
        self.joint_positions: dict[str, float] = {}
        self.create_subscription(JointState, JOINT_STATES_TOPIC, self._on_joint_state, 10)

    def _on_joint_state(self, msg: JointState) -> None:
        self.joint_positions = dict(zip(msg.name, msg.position))

    def wait_for_joint_state(self, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.joint_positions:
                return True
        return False


def _arm_current_positions(node: _ProbeNode, arm_key: str) -> dict[str, float]:
    prefix = _arm_joint_prefix(arm_key)
    return {
        name: pos for name, pos in node.joint_positions.items()
        if name.startswith(prefix)
    }


def _perturbed(positions: dict[str, float], delta: float) -> dict[str, float]:
    # Alternate +/- so joints don't all move the same direction (more
    # representative of a real varied target than a uniform shove).
    return {
        name: pos + (delta if i % 2 == 0 else -delta)
        for i, (name, pos) in enumerate(positions.items())
    }


def _wait_ready_joint(
    node: Node,
    moveit: DualArmMoveitClient,
    probe_target: JointTarget,
    timeout_s: float = READY_TIMEOUT_S,
    retry_interval_s: float = READY_RETRY_INTERVAL_S,
) -> bool:
    """plan_only_joint()'s twin of DualArmMoveitClient.wait_for_valid_state()
    -- that method is Cartesian-only (needs an ArmTarget), so this repeats
    its same "retry a plan-only goal until move_group's current-state
    monitor is actually ready" pattern for the joint-space path instead.
    Goals sent too soon after move_group.launch.py starts fail with
    "Found empty JointState message" / "invalid start state" in move_group's
    own log -- a startup-timing issue, not a real planning failure."""
    deadline = time.monotonic() + timeout_s
    attempt = 0
    while True:
        attempt += 1
        ok, error_code = moveit.plan_only_joint([probe_target], group_name=probe_target.group_name)
        if ok:
            if attempt > 1:
                print(f"  move_group ready after {attempt} attempts.")
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            print(f"  move_group still not ready after {timeout_s:.0f}s (last error_code={error_code}).")
            return False
        print(f"  [attempt {attempt}] not ready yet (error_code={error_code}) -- retrying...")
        sleep_end = time.monotonic() + min(retry_interval_s, max(remaining, 0.0))
        while time.monotonic() < sleep_end:
            rclpy.spin_once(node, timeout_sec=0.05)


def _check(
    moveit: DualArmMoveitClient,
    label: str,
    targets: list[JointTarget],
    group_name: str,
    execute: bool,
) -> _CheckResult:
    if execute:
        ok, record = moveit.move_to_joint(targets, group_name=group_name)
        error_code = record["error_code"]
        if ok:
            for t in record["targets"]:
                for name, requested in t["requested"].items():
                    achieved = t["achieved"].get(name)
                    if achieved is None:
                        continue
                    err = abs(achieved - requested)
                    if err > 0.02:
                        print(
                            f"    [warn] {name}: requested={requested:.4f} "
                            f"achieved={achieved:.4f} (|err|={err:.4f}rad > 0.02rad)"
                        )
    else:
        ok, error_code = moveit.plan_only_joint(targets, group_name=group_name)
    print(f"  [{label}] group={group_name}: {'OK' if ok else f'FAIL(err={error_code})'}")
    return _CheckResult(label, group_name, ok, error_code)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--execute", action="store_true",
        help="Also execute (move the mock robot) and verify achieved-vs-requested "
             "via move_to_joint()'s save-record path. Default: plan-only.",
    )
    parser.add_argument("--robot-namespace", default="lbr_dual_arm")
    args = parser.parse_args()

    rclpy.init()
    node = _ProbeNode()
    moveit = DualArmMoveitClient(node, namespace=args.robot_namespace)

    print("Waiting for /move_action (MoveGroup) action server...")
    if not moveit.wait_for_server(timeout_s=30.0):
        raise RuntimeError(
            "MoveGroup action server not available -- is `ros2 launch lbr_dual_arm_bringup "
            "mock.launch.py` + `ros2 launch lbr_dual_arm_bringup move_group.launch.py "
            "mode:=mock` running?"
        )

    print(f"Waiting for {JOINT_STATES_TOPIC} ...")
    if not node.wait_for_joint_state(JOINT_STATE_WAIT_S):
        raise RuntimeError(f"No message on {JOINT_STATES_TOPIC} after {JOINT_STATE_WAIT_S}s.")

    left_home = _arm_current_positions(node, "left")
    right_home = _arm_current_positions(node, "right")
    if not left_home or not right_home:
        raise RuntimeError(
            f"Got joint_states but found no joints matching prefixes "
            f"{_arm_joint_prefix('left')}*/{_arm_joint_prefix('right')}* -- "
            f"got names={sorted(node.joint_positions)}"
        )
    print(f"left home ({len(left_home)} joints): { {k: round(v, 4) for k, v in left_home.items()} }")
    print(f"right home ({len(right_home)} joints): { {k: round(v, 4) for k, v in right_home.items()} }")

    left_perturbed = _perturbed(left_home, PERTURB_RAD)
    right_perturbed = _perturbed(right_home, PERTURB_RAD)

    # arm_one_flange/arm_two_flange/both_arms_flange -- the ALWAYS-bare-flange
    # groups calibration actually uses (see flange_pose_store.ARM_KEYS and
    # moveit_dual_arm.py's module docstring); arm_one/arm_two/both_arms now
    # tip at the gripper TCP instead, which calibration doesn't use.
    left_group = ARM_KEYS["left"]["group_name"]
    right_group = ARM_KEYS["right"]["group_name"]
    both_group = "both_arms_flange"

    print("\nConfirming move_group's current-state monitor is ready (plan-only probe)...")
    probe = JointTarget(group_name=left_group, joint_positions=left_home, label="probe")
    if not _wait_ready_joint(node, moveit, probe):
        raise RuntimeError(f"move_group never became ready to plan for '{left_group}' -- see log above.")

    results: list[_CheckResult] = []

    print("\n=== 1. plan-only to current (trivial, already-satisfied) joint targets ===")
    for label, positions, group in (
        ("left@home", left_home, left_group),
        ("right@home", right_home, right_group),
    ):
        t = JointTarget(group_name=group, joint_positions=positions, label=label)
        results.append(_check(moveit, label, [t], group, execute=False))
    both_home = [
        JointTarget(group_name=left_group, joint_positions=left_home, label="left@home"),
        JointTarget(group_name=right_group, joint_positions=right_home, label="right@home"),
    ]
    results.append(_check(moveit, "both@home", both_home, both_group, execute=False))

    print(f"\n=== 2. plan-only to perturbed (+/-{PERTURB_RAD}rad) joint targets ===")
    for label, positions, group in (
        ("left@perturbed", left_perturbed, left_group),
        ("right@perturbed", right_perturbed, right_group),
    ):
        t = JointTarget(group_name=group, joint_positions=positions, label=label)
        results.append(_check(moveit, label, [t], group, execute=False))
    both_perturbed = [
        JointTarget(group_name=left_group, joint_positions=left_perturbed, label="left@perturbed"),
        JointTarget(group_name=right_group, joint_positions=right_perturbed, label="right@perturbed"),
    ]
    results.append(_check(moveit, "both@perturbed", both_perturbed, both_group, execute=False))

    if args.execute:
        print(f"\n=== 3. execute to perturbed targets (verifies achieved-state readback + save) ===")
        for label, positions, group in (
            ("left@perturbed[exec]", left_perturbed, left_group),
            ("right@perturbed[exec]", right_perturbed, right_group),
        ):
            t = JointTarget(group_name=group, joint_positions=positions, label=label)
            results.append(_check(moveit, label, [t], group, execute=True))

    node.destroy_node()
    rclpy.shutdown()

    n_ok = sum(1 for r in results if r.ok)
    print(f"\n=== Summary: {n_ok}/{len(results)} joint-target checks passed ===")
    for r in results:
        if not r.ok:
            print(f"  FAIL: [{r.label}] group={r.group_name} error_code={r.error_code}")

    if n_ok < len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
