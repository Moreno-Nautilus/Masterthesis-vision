"""Simulation-only reachability check for the dual-arm RealSense
calibration poses -- answers ONE question: can MoveIt actually plan (and,
optionally, execute) motion to every pose saved by capture_handeye_data.py
(config/flange_poses/{left,right}.json), driving lbr_dual_arm_bringup's
`mock` (simulated, non-physical) robot state instead of the real arms?

This is the "mock script" version of capture_handeye_data.py's --mode replay
+ autocalibrate_dual_realsense.py: it splits the saved poses into a
"--num-handeye-poses go through a simultaneous both_arms goal" chunk and a
"--num-board-poses go through single-arm goals" chunk (mirroring how
capture_handeye_data.py's replay mode pairs poses for --arm both, and how
autocalibrate_dual_realsense.py's board-pose stage moves single-arm) purely
to exercise both group-type reachability, and uses the same MoveIt client
(moveit_dual_arm.py) -- but does NOT touch any camera topic, does NOT run
the checkerboard PnP solve, and does NOT write any of the real calibration
outputs (config/camera_extrinsics_realsense.yaml, config/base_board_pose.yaml).
It only checks: is each pose (pair) reachable? There does not need to be a
real checkerboard, real RealSense cameras, or a real robot connected at all
-- everything here is virtual.

Run (needs ONLY the mock/sim MoveIt stack -- no camera launch, no
hardware.launch.py):

    ros2 launch lbr_dual_arm_bringup mock.launch.py
    ros2 launch lbr_dual_arm_bringup move_group.launch.py mode:=mock rviz:=false

then, in a third terminal, from the Masterthesis-vision repo root:

    python3 -m src.calibration.mock_reachability_check

By default this only PLANS (never executes, never moves anything -- not even
the simulated joint state) so it's fast and side-effect free. Pass --execute
to also run each accepted plan on the mock controller (harmless -- there is
no physical robot -- but slower, and only useful if you want to watch it move
in RViz to sanity-check the trajectories, not just confirm IK/collision
feasibility).

Exits non-zero if any pose (pair) was unreachable, so this can be used as a
pre-flight gate before booking real robot time for the actual
autocalibrate_dual_realsense.py run. Results are also written to
outputs/calibration_debug/mock_reachability/<timestamp>.json.

This script (and the rest of the calibration pipeline) uses the
arm_one_flange/arm_two_flange/both_arms_flange groups exclusively -- these
are always pinned to the bare flange (lbr_{one,two}_link_ee) regardless of
lbr_dual_arm_bringup's use_gripper toggle, matching where the saved
config/flange_poses/*.json captures are anchored. See
moveit_dual_arm.py's module docstring for the full arm_one/arm_one_flange
distinction.

KNOWN LIMITATION -- the "both_arms_flange" simultaneous goal (lbr_dual_arm's
SRDF has `both_arms_flange` = `arm_one_flange` + `arm_two_flange`, a
composite of two independent chains with no combined entry in
kinematics.yaml, only per-chain KDL solvers) cannot currently plan a
Cartesian (position/orientation) goal on BOTH tip links at once: move_group's
constraint-sampling pipeline logs "IKConstraintSampler received dirty robot
state" and fails, deterministically, even for a target pose that's already
exactly satisfied (verified: a plan-only goal to each arm's own current pose,
sent through "both_arms_flange", fails 100% of the time; the identical goal
split into two single-arm "arm_one_flange"/"arm_two_flange" goals succeeds
instantly). A joint-space goal (JointConstraint on both arms' current joint
values) through "both_arms_flange" DOES succeed -- so the group and the
planning pipeline are otherwise fine; this is specifically the
Cartesian-constraint IK-sampler path for composite multi-chain groups, a
documented MoveIt2 limitation, not a property of any particular flange pose.

Because of this, this script checks each "handeye-style" pose PAIR three
ways: the literal "both_arms_flange" simultaneous goal (expected to fail --
reported separately, does not affect the pass/fail verdict), plus each arm's
own target checked alone via its single-arm group (arm_one_flange /
arm_two_flange -- these DO answer "is this pose reachable" and DO gate the
exit code). If capture_handeye_data.py's --mode replay needs to actually run
simultaneous motion (--arm both) on real hardware, `moveit_dual_arm.py`'s
"both_arms_flange" single-goal design will need a workaround first (e.g. a
dedicated combined kinematics solver for `both_arms_flange` in
kinematics.yaml, or issuing two concurrent single-arm goals instead of one
composite-group goal).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import rclpy
from rclpy.node import Node

from src.calibration.flange_pose_store import ARM_KEYS, FlangePoseCapture, load_pose_set
from src.calibration.moveit_dual_arm import ArmTarget, DualArmMoveitClient

DEFAULT_NUM_HANDEYE_POSES = 5
DEFAULT_NUM_BOARD_POSES = 2

RESULTS_DIR = Path("outputs/calibration_debug/mock_reachability")


@dataclass
class _PoseCheckResult:
    stage: str          # "handeye" (both_arms) or "board" (single arm)
    label: str          # e.g. "pair 0" or "left idx=5"
    group_name: str
    ok: bool
    error_code: int


def _arm_target(arm_key: str, capture: FlangePoseCapture) -> ArmTarget:
    arm = ARM_KEYS[arm_key]
    return ArmTarget(
        group_name=arm["group_name"], base_frame=arm["base_frame"],
        tip_link=arm["flange_frame"], T_armBase_flange=capture.T_armBase_flange,
    )


def _check(
    moveit: DualArmMoveitClient,
    targets: list[ArmTarget],
    group_name: str,
    execute: bool,
) -> tuple[bool, int]:
    if execute:
        ok = moveit.move_to(targets, group_name=group_name)
        # move_to() doesn't hand back the raw error code -- it already logged
        # it on failure -- but callers here only need ok/not-ok either way.
        return ok, 0 if ok else -1
    return moveit.plan_only(targets, group_name=group_name)


def _run_both_arms_check(
    moveit: DualArmMoveitClient,
    left_poses: list[FlangePoseCapture],
    right_poses: list[FlangePoseCapture],
    execute: bool,
) -> list[_PoseCheckResult]:
    print("\n=== both_arms_flange simultaneous reachability ===")
    print(
        "(also checking each arm's target individually via arm_one_flange/arm_two_flange -- "
        "see module docstring: this MoveIt config's 'both_arms_flange' composite group cannot plan Cartesian "
        "pose goals on both tip links at once, so the per-arm checks are what actually answer "
        "'is this pose reachable'.)"
    )
    results = []
    n_pairs = min(len(left_poses), len(right_poses))
    for i in range(n_pairs):
        left_target = _arm_target("left", left_poses[i])
        right_target = _arm_target("right", right_poses[i])

        ok_both, err_both = _check(moveit, [left_target, right_target], "both_arms_flange", execute)
        ok_left, err_left = _check(moveit, [left_target], "arm_one_flange", execute)
        ok_right, err_right = _check(moveit, [right_target], "arm_two_flange", execute)

        print(
            f"  [pair {i}] left idx={left_poses[i].idx} + right idx={right_poses[i].idx}: "
            f"both_arms_flange={'OK' if ok_both else f'FAIL(err={err_both})'}  "
            f"left_alone={'OK' if ok_left else f'FAIL(err={err_left})'}  "
            f"right_alone={'OK' if ok_right else f'FAIL(err={err_right})'}"
        )
        results.append(_PoseCheckResult("handeye_both_arms", f"pair {i}", "both_arms_flange", ok_both, err_both))
        results.append(_PoseCheckResult("handeye_left_alone", f"pair {i} left idx={left_poses[i].idx}", "arm_one_flange", ok_left, err_left))
        results.append(_PoseCheckResult("handeye_right_alone", f"pair {i} right idx={right_poses[i].idx}", "arm_two_flange", ok_right, err_right))
    return results


def _run_single_arm_check(
    moveit: DualArmMoveitClient,
    left_poses: list[FlangePoseCapture],
    right_poses: list[FlangePoseCapture],
    execute: bool,
) -> list[_PoseCheckResult]:
    print("\n=== single-arm-only reachability ===")
    results = []
    for arm_key, poses in (("left", left_poses), ("right", right_poses)):
        arm = ARM_KEYS[arm_key]
        for cap in poses:
            targets = [_arm_target(arm_key, cap)]
            ok, error_code = _check(moveit, targets, arm["group_name"], execute)
            status = "OK" if ok else f"UNREACHABLE (error_code={error_code})"
            print(f"  [{arm_key} idx={cap.idx}]: {status}")
            results.append(_PoseCheckResult("board", f"{arm_key} idx={cap.idx}", arm["group_name"], ok, error_code))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num-handeye-poses", type=int, default=DEFAULT_NUM_HANDEYE_POSES)
    parser.add_argument("--num-board-poses", type=int, default=DEFAULT_NUM_BOARD_POSES)
    parser.add_argument(
        "--execute", action="store_true",
        help="Also execute each accepted plan on the mock controller (default: plan-only, no motion at all).",
    )
    parser.add_argument(
        "--robot-namespace", default="lbr_dual_arm",
        help="Namespace move_group.launch.py was started with (default matches that launch file's default).",
    )
    args = parser.parse_args()

    left_all = load_pose_set("left").captures
    right_all = load_pose_set("right").captures
    need = args.num_handeye_poses + args.num_board_poses
    if len(left_all) < need or len(right_all) < need:
        raise RuntimeError(
            f"Need >= {need} saved poses per arm (have left={len(left_all)}, right={len(right_all)}). "
            f"Run capture_handeye_data.py first."
        )

    left_handeye = left_all[: args.num_handeye_poses]
    right_handeye = right_all[: args.num_handeye_poses]
    left_board = left_all[args.num_handeye_poses: need]
    right_board = right_all[args.num_handeye_poses: need]

    rclpy.init()
    node = Node("mock_reachability_check")
    moveit = DualArmMoveitClient(node, namespace=args.robot_namespace)

    print("Waiting for /move_action (MoveGroup) action server...")
    if not moveit.wait_for_server(timeout_s=30.0):
        raise RuntimeError(
            "MoveGroup action server not available -- is `ros2 launch lbr_dual_arm_bringup "
            "mock.launch.py` + `ros2 launch lbr_dual_arm_bringup move_group.launch.py mode:=mock` "
            "running?"
        )

    print("Confirming move_group's current-state monitor is ready (plan-only probe)...")
    # Probe with each arm's own single-arm group, NOT "both_arms_flange": both_arms_flange is
    # a composite group (arm_one_flange + arm_two_flange, no shared kinematics_solver in
    # kinematics.yaml) and its Cartesian-constraint IK sampling is broken in this MoveIt
    # install regardless of startup timing -- see module docstring. arm_one_flange/
    # arm_two_flange each have their own KDL solver and work reliably, so they're what
    # actually tells us "is move_group ready".
    if not moveit.wait_for_valid_state([_arm_target("left", left_handeye[0])], group_name="arm_one_flange"):
        raise RuntimeError(
            "move_group never became ready to plan for 'arm_one_flange' -- see the warnings above."
        )
    if not moveit.wait_for_valid_state([_arm_target("right", right_handeye[0])], group_name="arm_two_flange"):
        raise RuntimeError(
            "move_group never became ready to plan for 'arm_two_flange' -- see the warnings above."
        )

    try:
        results = _run_both_arms_check(moveit, left_handeye, right_handeye, args.execute)
        results += _run_single_arm_check(moveit, left_board, right_board, args.execute)
    finally:
        node.destroy_node()
        rclpy.shutdown()

    # "handeye_both_arms" failures are the known composite-group limitation (see module
    # docstring) -- they don't mean a pose is unreachable, so they're reported separately
    # and don't gate the exit code. Every other stage (per-arm handeye + board poses) is a
    # real Cartesian-goal reachability check on real hardware groups (arm_one_flange/
    # arm_two_flange) and does gate it.
    pose_results = [r for r in results if r.stage != "handeye_both_arms"]
    both_arms_results = [r for r in results if r.stage == "handeye_both_arms"]

    n_ok = sum(1 for r in pose_results if r.ok)
    n_total = len(pose_results)
    n_both_arms_ok = sum(1 for r in both_arms_results if r.ok)
    print(f"\n=== Summary: {n_ok}/{n_total} individual poses reachable ({'plan+execute' if args.execute else 'plan-only'}) ===")
    print(
        f"    both_arms simultaneous goal: {n_both_arms_ok}/{len(both_arms_results)} succeeded "
        f"(known MoveIt limitation for this composite group + Cartesian goals -- see docstring; "
        f"does not affect the reachability verdict above)"
    )
    for r in pose_results:
        if not r.ok:
            print(f"  UNREACHABLE: [{r.stage}] {r.label} (group={r.group_name}, error_code={r.error_code})")

    payload = json.dumps({
        "mode": "plan+execute" if args.execute else "plan_only",
        "num_handeye_poses": args.num_handeye_poses,
        "num_board_poses": args.num_board_poses,
        "results": [r.__dict__ for r in results],
    }, indent=2)
    try:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RESULTS_DIR / f"{time.strftime('%Y%m%d_%H%M%S')}.json"
        out_path.write_text(payload)
        print(f"Wrote results to: {out_path}")
    except PermissionError:
        # outputs/ is root-owned on this checkout (pre-existing, not created by this
        # script) -- don't let a debug-log write failure mask an otherwise-successful
        # reachability run; just say so and move on.
        print(
            f"Could not write results to {RESULTS_DIR} (permission denied -- outputs/ is "
            f"root-owned on this checkout). Results were still printed above."
        )

    if n_ok < n_total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
