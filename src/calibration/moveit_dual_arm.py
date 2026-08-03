"""Minimal MoveGroup action-client helper for commanding the dual-LBR rig.

Nothing in this repo sends motion commands to the robot today (see
docs/moveit_robot_control.md: "none of Masterthesis-vision's own scripts send
motion commands to the robot" -- everything only reads /iiwa/ee_pose-style
topics). src/calibration/autocalibrate_dual_realsense.py is the first thing
that needs to actually drive the arms (to the poses saved by
capture_flange_poses_dual.py / capture_flange_poses_dual_handguided.py), so
this module exists to do that through the moveit_msgs/action/MoveGroup action
directly -- no moveit_commander/moveit_py Python bindings are installed in
this environment (checked: both raise ModuleNotFoundError), but moveit_msgs
itself is, and MoveGroup is the same action move_group's own C++
MoveGroupInterface calls under the hood.

Planning groups (lbr_dual_arm_moveit_config/config/lbr_dual_arm.srdf.xacro):
  arm_one    -- chain lbr_one_link_0 -> tip (left arm, port 30200)
  arm_two    -- chain lbr_two_link_0 -> tip (right arm, port 30201)
  both_arms  -- arm_one + arm_two combined, for one simultaneous, jointly
                planned & executed goal covering both tip links at once.
  These three groups' tip depends on whichever `use_gripper` value
  lbr_dual_arm_bringup was launched with: lbr_{one,two}_gripper_tcp when
  true (the default -- the Y-gripper is the TCP for general bimanual
  motion), lbr_{one,two}_link_ee (the bare flange) when false.

  arm_one_flange / arm_two_flange / both_arms_flange -- same joint chains,
  but ALWAYS tipped at lbr_{one,two}_link_ee regardless of use_gripper.
  Hand-eye calibration (capture_flange_poses_dual*.py,
  autocalibrate_dual_realsense.py, mock_reachability_check.py) uses these
  exclusively -- flange_pose_store.ARM_KEYS["group_name"] points here --
  since the calibration math and the already-captured
  config/flange_poses/*.json are anchored to the physical flange, not
  wherever the gripper's TCP currently is. Using the _flange groups (rather
  than constraining lbr_link_ee as a non-tip link of arm_one/arm_two) also
  keeps a fast, tip-matched KDL IK solver available for Cartesian goals --
  MoveIt's KDL plugin only does closed-form IK for a group's own configured
  tip_link.

Two independent goal styles are supported:
  ArmTarget / move_to() / plan_only() -- Cartesian: a position+orientation
    constraint on the group's tip link, solved via IK. Used for
    reachability probing and the startup "is move_group ready" check.
  JointTarget / move_to_joint() / plan_only_joint() -- joint-space: a
    JointConstraint per joint, no IK at all. Used to replay a captured
    configuration exactly (see JointTarget docstring) -- this is what
    autocalibrate_dual_realsense.py's Stage A/B replay actually uses.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    Constraints,
    JointConstraint,
    MoveItErrorCodes,
    OrientationConstraint,
    PositionConstraint,
)
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive

from src.utils.se3 import SE3

# lbr_dual_arm_bringup's move_group.launch.py namespaces everything under
# its `robot_name` launch arg (default "lbr_dual_arm") -- see that launch
# file's `namespace=robot_name` on the move_group node. Override via the
# `namespace` arg below if you launched with a different `robot_name`.
DEFAULT_MOVE_GROUP_NAMESPACE = "lbr_dual_arm"

# Pose-goal tolerances -- matches the "let it settle" precision the manual
# calibration routines already rely on (io_extrinsics reprojection gates are
# tighter than this; this only needs to get the flange back close enough for
# the checkerboard to be in frame again, not to reproduce the exact original
# pose bit-for-bit).
POSITION_TOLERANCE_M = 0.005
ORIENTATION_TOLERANCE_RAD = 0.02

PLANNING_TIME_S = 10.0
NUM_PLANNING_ATTEMPTS = 5
VELOCITY_SCALING = 0.15
ACCELERATION_SCALING = 0.15

# move_group's own current-state monitor can still be "dirty" (no valid link
# transforms yet) for a short window right after move_group.launch.py comes
# up, even once wait_for_server() confirms the /move_action action server
# itself is reachable -- the action server and the planning-scene monitor's
# current-state subscription come up independently. Sending a goal in that
# window fails instantly with "IKConstraintSampler received dirty robot
# state" -> "Unable to sample any valid states for goal tree", which looks
# identical to a real planning failure (generic FAILURE error code) unless
# you go read move_group's own log.
#
# Running everything inside a Docker container makes this window less
# predictable, not just longer: DDS discovery of that /joint_states
# subscription (and of /move_action itself) over the container's network
# adds its own, variable delay on top of move_group's normal startup, so a
# fixed sleep tuned on bare metal isn't reliable here. Poll instead, with a
# generous ceiling, and stop as soon as the state is actually ready.
STARTUP_READY_TIMEOUT_S = 30.0
STARTUP_READY_RETRY_INTERVAL_S = 1.0

# joint_state_broadcaster's namespace -- must match calibration.launch.py's
# default robot_name ("lbr_dual_arm") and
# capture_flange_poses_dual_handguided.py's JOINT_STATES_TOPIC.
JOINT_STATES_TOPIC = "/lbr_dual_arm/joint_states"

# JointConstraint tolerance for move_to_joint()/plan_only_joint() -- how
# close is "reached" that joint value. Looser than a real precision
# requirement; this is about reproducing a captured configuration closely
# enough for the checkerboard to be back in frame, same rationale as
# POSITION_TOLERANCE_M/ORIENTATION_TOLERANCE_RAD above.
JOINT_TOLERANCE_RAD = 0.01

# How long to let /lbr_dual_arm/joint_states catch up to the controller's
# final state after a goal completes, before move_to_joint() reads back
# "achieved" -- avoids saving a stale mid-motion snapshot.
JOINT_STATE_SETTLE_S = 0.2

MOVE_LOG_DIR = Path("outputs/calibration_debug/moveit_joint_targets")


# One arm's motion target: which planning group/tip link, and the desired
# T_armBase_flange pose (already in that arm's OWN lbr_{one,two}_link_0
# frame -- i.e. exactly what /left/ee_pose or /right/ee_pose publishes, and
# exactly the T_base_flange stored by capture_flange_poses_dual.py).
@dataclass
class ArmTarget:
    group_name: str      # "arm_one"/"arm_two" (gripper-toggle tip) or "arm_one_flange"/"arm_two_flange" (always bare flange)
    base_frame: str       # "lbr_one_link_0" or "lbr_two_link_0"
    tip_link: str          # must match group_name's actual SRDF tip_link
    T_armBase_flange: SE3


# One arm's motion target as a literal joint configuration -- e.g. the
# joint_positions saved by capture_flange_poses_dual_handguided.py
# (FlangePoseCapture.joint_positions). Unlike ArmTarget (a Cartesian pose,
# which move_to() has MoveGroup's IK sampler solve, landing on *some*
# configuration among this redundant 7-DOF arm's whole family of solutions
# for that pose), this skips IK entirely: MoveGroup just plans a joint-space
# path to these exact values, reproducing the specific configuration that
# was actually captured, not just an equivalent flange pose.
@dataclass
class JointTarget:
    group_name: str                      # e.g. "arm_one_flange"/"arm_two_flange" (calibration); tip link is irrelevant here, joint-space goals don't use IK
    joint_positions: dict[str, float]     # joint name -> radians, e.g. {"lbr_one_A1": 0.12, ...}
    label: str = ""                       # for move-record filenames/entries, e.g. "left idx=3"


def _se3_to_pose_stamped(T: SE3, frame_id: str) -> PoseStamped:
    msg = PoseStamped()
    msg.header.frame_id = frame_id
    msg.pose.position.x = float(T.t[0])
    msg.pose.position.y = float(T.t[1])
    msg.pose.position.z = float(T.t[2])

    R = T.R
    trace = np.trace(R)
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    q = np.array([qx, qy, qz, qw])
    q /= np.linalg.norm(q) + 1e-12
    msg.pose.orientation.x = float(q[0])
    msg.pose.orientation.y = float(q[1])
    msg.pose.orientation.z = float(q[2])
    msg.pose.orientation.w = float(q[3])
    return msg


def _pose_goal_constraints(target: ArmTarget) -> Constraints:
    pose = _se3_to_pose_stamped(target.T_armBase_flange, target.base_frame)

    pos_constraint = PositionConstraint()
    pos_constraint.header.frame_id = target.base_frame
    pos_constraint.link_name = target.tip_link
    pos_constraint.target_point_offset.x = 0.0
    pos_constraint.target_point_offset.y = 0.0
    pos_constraint.target_point_offset.z = 0.0

    sphere = SolidPrimitive()
    sphere.type = SolidPrimitive.SPHERE
    sphere.dimensions = [POSITION_TOLERANCE_M]
    pos_constraint.constraint_region.primitives.append(sphere)
    pos_constraint.constraint_region.primitive_poses.append(pose.pose)
    pos_constraint.weight = 1.0

    ori_constraint = OrientationConstraint()
    ori_constraint.header.frame_id = target.base_frame
    ori_constraint.link_name = target.tip_link
    ori_constraint.orientation = pose.pose.orientation
    ori_constraint.absolute_x_axis_tolerance = ORIENTATION_TOLERANCE_RAD
    ori_constraint.absolute_y_axis_tolerance = ORIENTATION_TOLERANCE_RAD
    ori_constraint.absolute_z_axis_tolerance = ORIENTATION_TOLERANCE_RAD
    ori_constraint.weight = 1.0

    c = Constraints()
    c.position_constraints.append(pos_constraint)
    c.orientation_constraints.append(ori_constraint)
    return c


def _joint_goal_constraints(target: JointTarget) -> Constraints:
    c = Constraints()
    for name, position in target.joint_positions.items():
        jc = JointConstraint()
        jc.joint_name = name
        jc.position = float(position)
        jc.tolerance_above = JOINT_TOLERANCE_RAD
        jc.tolerance_below = JOINT_TOLERANCE_RAD
        jc.weight = 1.0
        c.joint_constraints.append(jc)
    return c


class DualArmMoveitClient:
    """Thin synchronous wrapper around move_group's MoveGroup action.

    One instance drives however many single/simultaneous goals a calibration
    run needs; each call blocks (via rclpy.spin_until_future_complete) until
    that goal's plan+execute has finished or failed.
    """

    def __init__(self, node: Node, namespace: str = DEFAULT_MOVE_GROUP_NAMESPACE):
        self._node = node
        move_action = f"/{namespace}/move_action" if namespace else "/move_action"
        self._client = ActionClient(node, MoveGroup, move_action)

        # Tracks the latest /lbr_dual_arm/joint_states, so move_to_joint()
        # can read back what was actually achieved after a goal completes.
        # Updated by rclpy spin -- _send_goal_and_wait() already spins the
        # node while waiting on a goal, so no extra spinning is needed for
        # this to stay current.
        self._joint_state_positions: dict[str, float] = {}
        self._node.create_subscription(
            JointState, JOINT_STATES_TOPIC, self._on_joint_state, 10
        )

    def _on_joint_state(self, msg: JointState) -> None:
        self._joint_state_positions = dict(zip(msg.name, msg.position))

    def wait_for_server(self, timeout_s: float = 10.0) -> bool:
        return self._client.wait_for_server(timeout_sec=timeout_s)

    def _build_goal(
        self, targets: list[ArmTarget], group_name: Optional[str], plan_only: bool
    ) -> MoveGroup.Goal:
        goal = MoveGroup.Goal()
        req = goal.request
        req.group_name = group_name or targets[0].group_name
        req.num_planning_attempts = NUM_PLANNING_ATTEMPTS
        req.allowed_planning_time = PLANNING_TIME_S
        req.max_velocity_scaling_factor = VELOCITY_SCALING
        req.max_acceleration_scaling_factor = ACCELERATION_SCALING

        for target in targets:
            req.goal_constraints.append(_pose_goal_constraints(target))

        goal.planning_options.plan_only = plan_only
        goal.planning_options.replan = False
        return goal

    def _send_goal_and_wait(self, goal: MoveGroup.Goal) -> tuple[bool, int]:
        """Returns (ok, error_code). error_code is MoveItErrorCodes.FAILURE
        (still useful to log) if the goal was rejected outright or the
        result future never resolved."""
        send_future = self._client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self._node, send_future)
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            return False, MoveItErrorCodes.FAILURE

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self._node, result_future)
        result_wrapper = result_future.result()
        if result_wrapper is None:
            return False, MoveItErrorCodes.FAILURE

        status_ok = result_wrapper.status == GoalStatus.STATUS_SUCCEEDED
        error_code = result_wrapper.result.error_code.val
        return status_ok and error_code == MoveItErrorCodes.SUCCESS, error_code

    def wait_for_valid_state(
        self,
        targets: list[ArmTarget],
        group_name: Optional[str] = None,
        timeout_s: float = STARTUP_READY_TIMEOUT_S,
        retry_interval_s: float = STARTUP_READY_RETRY_INTERVAL_S,
    ) -> bool:
        """Repeatedly plan-only (never executes/moves the robot) toward
        `targets` until move_group's current-state monitor is actually ready,
        or `timeout_s` elapses.

        Call this once, right after wait_for_server(), before the first real
        move_to() -- see STARTUP_READY_TIMEOUT_S above for why
        wait_for_server() alone isn't enough. Reuses whatever the very first
        real target is as the probe pose, so this doubles as an early,
        cheap confirmation that that goal is plannable at all (collision
        included) before committing to the full run.
        """
        if not targets:
            raise ValueError("targets must be non-empty")

        group = group_name or targets[0].group_name
        deadline = time.monotonic() + timeout_s
        attempt = 0
        while True:
            attempt += 1
            goal = self._build_goal(targets, group_name, plan_only=True)
            ok, error_code = self._send_goal_and_wait(goal)
            if ok:
                if attempt > 1:
                    self._node.get_logger().info(
                        f"move_group ready for group={group} after {attempt} attempts."
                    )
                return True

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._node.get_logger().error(
                    f"move_group still not ready for group={group} after "
                    f"{timeout_s:.0f}s (last error_code={error_code}). Giving up -- "
                    f"if this is running in a Docker container, check that move_group's "
                    f"own log doesn't show DDS discovery still in progress."
                )
                return False

            self._node.get_logger().warn(
                f"[attempt {attempt}] group={group} not plannable yet "
                f"(error_code={error_code}) -- retrying in {retry_interval_s:.1f}s "
                f"({remaining:.0f}s left)..."
            )
            deadline_sleep = min(retry_interval_s, max(remaining, 0.0))
            self._spin_sleep(deadline_sleep)

    def _spin_sleep(self, duration_s: float) -> None:
        end = time.monotonic() + duration_s
        while time.monotonic() < end:
            rclpy.spin_once(self._node, timeout_sec=0.05)

    def plan_only(self, targets: list[ArmTarget], group_name: Optional[str] = None) -> tuple[bool, int]:
        """One plan-only attempt (never executes, never moves anything --
        real or mock). Unlike wait_for_valid_state(), this does NOT retry on
        failure, so a genuine planning failure (pose unreachable / no IK
        solution / self-collision) is distinguishable from move_group simply
        not being ready yet -- use wait_for_valid_state() once at startup for
        that, then this per-pose for the actual reachability question."""
        goal = self._build_goal(targets, group_name, plan_only=True)
        return self._send_goal_and_wait(goal)

    def move_to(self, targets: list[ArmTarget], group_name: Optional[str] = None) -> bool:
        """Plan + execute one MoveGroup goal covering every target in `targets`.

        A single target -> group_name defaults to that target's own group
        ("arm_one"/"arm_two"). Two targets (one per arm) -> pass
        group_name="both_arms" explicitly for one simultaneous, jointly
        planned goal with both tip links constrained at once.
        """
        if not targets:
            raise ValueError("targets must be non-empty")

        goal = self._build_goal(targets, group_name, plan_only=False)
        ok, error_code = self._send_goal_and_wait(goal)
        if not ok:
            self._node.get_logger().error(
                f"MoveGroup goal (group={group_name or targets[0].group_name}) failed: "
                f"error_code={error_code}"
            )
        return ok

    def _build_joint_goal(
        self, targets: list[JointTarget], group_name: Optional[str], plan_only: bool
    ) -> MoveGroup.Goal:
        goal = MoveGroup.Goal()
        req = goal.request
        req.group_name = group_name or targets[0].group_name
        req.num_planning_attempts = NUM_PLANNING_ATTEMPTS
        req.allowed_planning_time = PLANNING_TIME_S
        req.max_velocity_scaling_factor = VELOCITY_SCALING
        req.max_acceleration_scaling_factor = ACCELERATION_SCALING

        for target in targets:
            req.goal_constraints.append(_joint_goal_constraints(target))

        goal.planning_options.plan_only = plan_only
        goal.planning_options.replan = False
        return goal

    def plan_only_joint(
        self, targets: list[JointTarget], group_name: Optional[str] = None
    ) -> tuple[bool, int]:
        """plan_only()'s joint-space twin -- one plan-only attempt at the
        exact given joint configuration(s), no execution, no IK involved."""
        goal = self._build_joint_goal(targets, group_name, plan_only=True)
        return self._send_goal_and_wait(goal)

    def move_to_joint(
        self, targets: list[JointTarget], group_name: Optional[str] = None, save: bool = True
    ) -> tuple[bool, dict]:
        """Plan + execute one MoveGroup goal that drives to the EXACT joint
        configuration(s) in `targets` (a JointConstraint per joint, not a
        Cartesian pose) -- skips IK entirely, so this reproduces the literal
        captured configuration, not just some configuration that reaches an
        equivalent flange pose (see JointTarget docstring).

        Regardless of success/failure, reads back whatever
        /lbr_dual_arm/joint_states currently reports for the requested
        joints after the goal completes, and -- unless save=False -- writes
        both the requested and achieved values to
        outputs/calibration_debug/moveit_joint_targets/<timestamp>.json.
        Saved on failure too: a partial/failed move is exactly the case
        you most want a permanent record of, not less than a success.

        Returns (ok, record) -- record is the same dict that gets saved, so
        callers can inspect requested-vs-achieved even when save=False or
        the write fails (outputs/ has been root-owned on this checkout
        before -- see mock_reachability_check.py's identical handling).
        """
        if not targets:
            raise ValueError("targets must be non-empty")

        goal = self._build_joint_goal(targets, group_name, plan_only=False)
        ok, error_code = self._send_goal_and_wait(goal)
        if not ok:
            self._node.get_logger().error(
                f"MoveGroup joint-space goal (group={group_name or targets[0].group_name}) "
                f"failed: error_code={error_code}"
            )

        # Let joint_states catch up to the controller's final state before
        # reading "achieved" -- avoids saving a stale mid-motion snapshot for
        # a goal that otherwise succeeded.
        self._spin_sleep(JOINT_STATE_SETTLE_S)

        record = {
            "timestamp_unix_s": time.time(),
            "group_name": group_name or targets[0].group_name,
            "ok": ok,
            "error_code": error_code,
            "targets": [
                {
                    "label": t.label,
                    "group_name": t.group_name,
                    "requested": dict(t.joint_positions),
                    "achieved": {
                        name: self._joint_state_positions.get(name)
                        for name in t.joint_positions
                    },
                }
                for t in targets
            ],
        }
        if save:
            self._save_move_record(record)
        return ok, record

    def _save_move_record(self, record: dict) -> None:
        try:
            MOVE_LOG_DIR.mkdir(parents=True, exist_ok=True)
            ts_ms = int(record["timestamp_unix_s"] * 1000) % 1000
            out_path = MOVE_LOG_DIR / f"{time.strftime('%Y%m%d_%H%M%S')}_{ts_ms:03d}.json"
            out_path.write_text(json.dumps(record, indent=2))
            self._node.get_logger().info(f"Saved joint-target move record -> {out_path}")
        except PermissionError:
            # outputs/ is root-owned on this checkout (pre-existing -- see
            # mock_reachability_check.py's identical handling). Don't let a
            # debug-log write failure mask an otherwise-successful move;
            # the record is still returned to the caller either way.
            self._node.get_logger().warn(
                f"Could not write joint-target move record to {MOVE_LOG_DIR} (permission "
                f"denied -- outputs/ is root-owned on this checkout). requested/achieved "
                f"values were still returned to the caller."
            )
