"""Minimal MoveGroup action-client helper for commanding the dual-LBR rig.

Nothing in this repo sends motion commands to the robot today (see
docs/moveit_robot_control.md: "none of Masterthesis-vision's own scripts send
motion commands to the robot" -- everything only reads /iiwa/ee_pose-style
topics). src/calibration/autocalibrate_dual_realsense.py is the first thing
that needs to actually drive the arms (to the poses saved by
capture_flange_poses_dual.py), so this module exists to do that through the
moveit_msgs/action/MoveGroup action directly -- no moveit_commander/moveit_py
Python bindings are installed in this environment (checked: both raise
ModuleNotFoundError), but moveit_msgs itself is, and MoveGroup is the same
action move_group's own C++ MoveGroupInterface calls under the hood.

Planning groups (lbr_dual_arm_moveit_config/config/lbr_dual_arm.srdf):
  arm_one    -- chain lbr_one_link_0 -> lbr_one_link_ee (left arm, port 30200)
  arm_two    -- chain lbr_two_link_0 -> lbr_two_link_ee (right arm, port 30201)
  both_arms  -- arm_one + arm_two combined, for one simultaneous, jointly
                planned & executed goal covering both tip links at once.

Only pose-goal (position + orientation constraint on the tip link) requests
are supported -- that's all autocalibrate_dual_realsense.py needs to reach a
previously-recorded flange pose.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    Constraints,
    MoveItErrorCodes,
    OrientationConstraint,
    PositionConstraint,
)
from rclpy.action import ActionClient
from rclpy.node import Node
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


# One arm's motion target: which planning group/tip link, and the desired
# T_armBase_flange pose (already in that arm's OWN lbr_{one,two}_link_0
# frame -- i.e. exactly what /left/ee_pose or /right/ee_pose publishes, and
# exactly the T_base_flange stored by capture_flange_poses_dual.py).
@dataclass
class ArmTarget:
    group_name: str      # "arm_one" or "arm_two"
    base_frame: str       # "lbr_one_link_0" or "lbr_two_link_0"
    tip_link: str          # "lbr_one_link_ee" or "lbr_two_link_ee"
    T_armBase_flange: SE3


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
