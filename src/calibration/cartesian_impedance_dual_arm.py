"""Compliant execution client for the dual-LBR rig's Cartesian impedance
controllers (cartesian_impedance_lbr_one/_two -- see
franka_ros2_ws/.../lbr_dual_arm_bringup/launch/cartesian_impedance.launch.py).

Additive alongside moveit_dual_arm.py, not a replacement: that module still
owns the position-controlled MoveGroup pipeline
(autocalibrate_dual_realsense.py etc.) untouched. This module talks to a
*different* controller bring-up (torque/effort command interface, must be
launched with cartesian_impedance.launch.py instead of hardware.launch.py)
and drives it directly over the controller's own topics -- no MoveGroup
execution involved, only (optionally) MoveGroup *planning* via
plan_joint_trajectory() below.

Two ways to reach a target, matching the controller's actual inputs:
  - move_to_cartesian()      -- a Cartesian target (ArmTarget): published
                                 straight to the controller's target_frame
                                 topic, then blocks until settled.
                                 publish_target() is the same publish without
                                 the settle-wait, for interactive/streaming
                                 callers like teleop_cartesian_impedance.py.
  - move_to_joint_compliant() -- a joint-space target (JointTarget, e.g. a
                                 hand-guided capture): FK'd via MoveIt's
                                 compute_fk service to get target_frame, AND
                                 pushed via set_nullspace_target() so the
                                 redundant arm's elbow is biased toward the
                                 captured configuration instead of an
                                 arbitrary IK solution. This is a soft pull
                                 (nullspace stiffness), not a hard
                                 constraint -- see the controller patch's
                                 comments and validate against literal
                                 captures before relying on it for
                                 precision-sensitive stages.

Nothing about the controller's compliance behavior changes except through
its own set_parameters service (no free topic for stiffness or nullspace --
only target_frame, the motion goal itself, stays a topic):
  - set_gains()            -- stiffness.*/nullspace_stiffness. Defaults
                               shipped in dual_arm_cartesian_impedance_controllers.yaml
                               are the "reasonable out of the box" values
                               (1000 N/m trans / 30 Nm/rad rot,
                               nullspace_stiffness 0.0). Per the agreed
                               design these are GAINS, so changing them is
                               only safe while the arm is stationary --
                               set_gains() enforces this itself and raises
                               GainChangeUnsafeError otherwise; callers must
                               not bypass that guard.
  - set_nullspace_target()  -- nullspace_desired_configuration only. This is
                               a TARGET (like target_frame), not a gain, so
                               it has no stationary guard -- move_to_joint_compliant()
                               and execute_planned_trajectory() both call it
                               while the arm may be actively moving. It still
                               goes exclusively through the same
                               set_parameters service, never a topic.
Both are live-rereadable every control cycle by the patched controller (see
kuka_lbr_control/controllers/cartesian_impedance_controller's
updateGainsFromParameters()), so neither needs a controller restart.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Pose, PoseStamped, TransformStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import RobotState, RobotTrajectory
from moveit_msgs.srv import GetPositionFK
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

from src.calibration.moveit_dual_arm import (
    ArmTarget,
    JointTarget,
    NUM_PLANNING_ATTEMPTS,
    PLANNING_TIME_S,
    VELOCITY_SCALING,
    ACCELERATION_SCALING,
    DEFAULT_MOVE_GROUP_NAMESPACE,
    _joint_goal_constraints,
    _se3_to_pose_stamped,
)
from src.utils.se3 import SE3

# Per-arm mapping onto the cartesian_impedance_lbr_one/_two controller
# instances (dual_arm_cartesian_impedance_controllers.yaml) -- base_frame /
# tip_frame match flange_pose_store.ARM_KEYS (left=robot_a=lbr_one,
# right=robot_b=lbr_two); joint_names is the exact, ordered joints: list
# from that yaml (must match -- it's also the order
# nullspace_desired_configuration is indexed in, see set_nullspace_target()).
ARM_KEYS = {
    "left": {
        "controller_name": "cartesian_impedance_lbr_one",
        "base_frame": "lbr_one_link_0",
        "tip_frame": "lbr_one_link_ee",
        "joint_names": [f"lbr_one_A{i}" for i in range(1, 8)],
    },
    "right": {
        "controller_name": "cartesian_impedance_lbr_two",
        "base_frame": "lbr_two_link_0",
        "tip_frame": "lbr_two_link_ee",
        "joint_names": [f"lbr_two_A{i}" for i in range(1, 8)],
    },
}

DEFAULT_NAMESPACE = DEFAULT_MOVE_GROUP_NAMESPACE  # "lbr_dual_arm"

# Convergence gates for move_to_cartesian()/move_to_joint_compliant() --
# looser than moveit_dual_arm.py's MoveGroup goal tolerances since this is a
# spring-damper settling to a pose, not a planner's terminal-state check.
POSITION_TOLERANCE_M = 0.01
ORIENTATION_TOLERANCE_RAD = 0.03
SETTLE_VELOCITY_RAD_S = 0.01  # joint speed below this counts as "not moving"
SETTLE_CONSECUTIVE_CHECKS = 5
POLL_INTERVAL_S = 0.05
DEFAULT_TIMEOUT_S = 20.0

# set_gains() stationary guard.
GAIN_CHANGE_VELOCITY_THRESHOLD_RAD_S = 0.01


class GainChangeUnsafeError(RuntimeError):
    """Raised by set_gains() when the arm is not (recently) stationary."""


def _quat_to_R(x: float, y: float, z: float, w: float) -> np.ndarray:
    n = np.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _pose_to_se3(pose: Pose) -> SE3:
    R = _quat_to_R(
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w
    )
    t = np.array([pose.position.x, pose.position.y, pose.position.z])
    return SE3(R, t)


def _transform_to_se3(tf: TransformStamped) -> SE3:
    q = tf.transform.rotation
    R = _quat_to_R(q.x, q.y, q.z, q.w)
    t = np.array(
        [tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z]
    )
    return SE3(R, t)


def _orientation_error_rad(R_a: np.ndarray, R_b: np.ndarray) -> float:
    """Angle of the relative rotation R_a^T @ R_b, in [0, pi]."""
    R_err = R_a.T @ R_b
    cos_angle = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(cos_angle))


def _arm_key_from_group_name(group_name: str) -> str:
    if group_name.startswith("arm_one"):
        return "left"
    if group_name.startswith("arm_two"):
        return "right"
    raise ValueError(
        f"Can't infer arm_key from group_name={group_name!r}; expected it to "
        f"start with 'arm_one' or 'arm_two' (see flange_pose_store.ARM_KEYS)."
    )


@dataclass
class GainSettings:
    stiffness: Optional[dict] = None  # e.g. {"trans_x": 500.0, ..., "rot_z": 30.0}
    nullspace_stiffness: Optional[float] = None


class CartesianImpedanceDualArmClient:
    """Drives cartesian_impedance_lbr_one/_two directly -- no MoveGroup
    execution. One instance serves both arms (mirrors DualArmMoveitClient's
    per-node-not-per-arm shape)."""

    def __init__(self, node: Node, namespace: str = DEFAULT_NAMESPACE):
        self._node = node
        self._namespace = namespace

        self._target_frame_pub = {}
        self._set_params_client = {}
        for arm_key, info in ARM_KEYS.items():
            base = f"/{namespace}/{info['controller_name']}"
            self._target_frame_pub[arm_key] = node.create_publisher(
                PoseStamped, f"{base}/target_frame", 1
            )
            self._set_params_client[arm_key] = node.create_client(
                SetParameters, f"{base}/set_parameters"
            )

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, node)

        self._joint_state_positions: dict[str, float] = {}
        self._joint_state_velocities: dict[str, float] = {}
        node.create_subscription(
            JointState, f"/{namespace}/joint_states", self._on_joint_state, 10
        )

        self._fk_client = node.create_client(GetPositionFK, f"/{namespace}/compute_fk")
        self._move_group_client = ActionClient(
            node, MoveGroup, f"/{namespace}/move_action"
        )

    def _on_joint_state(self, msg: JointState) -> None:
        self._joint_state_positions = dict(zip(msg.name, msg.position))
        if msg.velocity:
            self._joint_state_velocities = dict(zip(msg.name, msg.velocity))

    def _spin_sleep(self, duration_s: float) -> None:
        end = time.monotonic() + duration_s
        while time.monotonic() < end:
            rclpy.spin_once(self._node, timeout_sec=0.05)

    def _joint_velocities_settled(self, arm_key: str, threshold: float) -> bool:
        names = ARM_KEYS[arm_key]["joint_names"]
        velocities = [self._joint_state_velocities.get(n) for n in names]
        if any(v is None for v in velocities):
            return False  # no data yet -- can't confirm settled
        return all(abs(v) < threshold for v in velocities)

    # -- Readiness ---------------------------------------------------------

    def wait_for_controller(self, arm_key: str, timeout_s: float = 10.0) -> bool:
        """Waits for arm_key's cartesian_impedance_lbr_one/_two controller's
        set_parameters service to become available -- a concrete signal that
        controller is actually loaded and its node is up, as opposed to
        DualArmMoveitClient.wait_for_valid_state_joint()'s check (MoveGroup
        can produce a plan), which is a structural/MoveIt-side check that
        can succeed BEFORE this controller (or even the FRI hardware
        connection) exists -- e.g. immediately after cartesian_impedance.launch.py
        starts, using whatever placeholder joint state MoveGroup happens to
        have. Callers driving this client (not just MoveGroup) must wait on
        this too, or a call like execute_planned_trajectory() ->
        set_nullspace_target() can hit an unavailable set_parameters service
        (see cartesian_impedance_lbr_one/_two spawner logs: it blocks on
        controller_manager's own service, which itself doesn't come up
        until BOTH arms' FRI hardware has connected)."""
        return self._set_params_client[arm_key].wait_for_service(timeout_sec=timeout_s)

    # -- Cartesian pose lookup -------------------------------------------------

    def current_flange_pose(self, arm_key: str, timeout_s: float = 2.0) -> Optional[SE3]:
        info = ARM_KEYS[arm_key]
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                tf = self._tf_buffer.lookup_transform(
                    info["base_frame"], info["tip_frame"], rclpy.time.Time()
                )
                return _transform_to_se3(tf)
            except (LookupException, ConnectivityException, ExtrapolationException):
                rclpy.spin_once(self._node, timeout_sec=0.1)
        return None

    def _wait_until_settled(
        self,
        arm_key: str,
        target: SE3,
        position_tol: float,
        orientation_tol: float,
        timeout_s: float,
    ) -> bool:
        deadline = time.monotonic() + timeout_s
        consecutive = 0
        while time.monotonic() < deadline:
            current = self.current_flange_pose(arm_key, timeout_s=1.0)
            if current is not None:
                pos_err = float(np.linalg.norm(current.t - target.t))
                rot_err = _orientation_error_rad(current.R, target.R)
                settled = self._joint_velocities_settled(arm_key, SETTLE_VELOCITY_RAD_S)
                if pos_err <= position_tol and rot_err <= orientation_tol and settled:
                    consecutive += 1
                    if consecutive >= SETTLE_CONSECUTIVE_CHECKS:
                        return True
                else:
                    consecutive = 0
            self._spin_sleep(POLL_INTERVAL_S)
        return False

    # -- Goal execution ---------------------------------------------------------

    def publish_target(self, target: ArmTarget) -> None:
        """Publish target.T_armBase_flange to the arm's target_frame topic
        without waiting for it to settle -- for continuous/interactive
        streaming (see teleop_cartesian_impedance.py), where blocking on
        every incremental update would stall input handling. move_to_cartesian()
        below is this plus the settle-wait, for one-shot goals."""
        arm_key = _arm_key_from_group_name(target.group_name)
        pose_msg = _se3_to_pose_stamped(target.T_armBase_flange, target.base_frame)
        self._target_frame_pub[arm_key].publish(pose_msg)

    def move_to_cartesian(
        self,
        target: ArmTarget,
        position_tol: float = POSITION_TOLERANCE_M,
        orientation_tol: float = ORIENTATION_TOLERANCE_RAD,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> bool:
        """Publish target.T_armBase_flange to the arm's target_frame topic
        and block until the flange settles there (or timeout_s elapses)."""
        arm_key = _arm_key_from_group_name(target.group_name)
        self.publish_target(target)
        return self._wait_until_settled(
            arm_key, target.T_armBase_flange, position_tol, orientation_tol, timeout_s
        )

    def _compute_fk(self, arm_key: str, joint_positions: dict) -> Optional[SE3]:
        info = ARM_KEYS[arm_key]
        if not self._fk_client.wait_for_service(timeout_sec=5.0):
            self._node.get_logger().error("compute_fk service not available.")
            return None

        req = GetPositionFK.Request()
        req.header.frame_id = info["base_frame"]
        req.fk_link_names = [info["tip_frame"]]
        req.robot_state = RobotState()
        req.robot_state.joint_state = JointState(
            name=list(joint_positions.keys()),
            position=[float(v) for v in joint_positions.values()],
        )

        future = self._fk_client.call_async(req)
        rclpy.spin_until_future_complete(self._node, future)
        result = future.result()
        if result is None or not result.pose_stamped:
            self._node.get_logger().error("compute_fk call failed or returned no pose.")
            return None
        return _pose_to_se3(result.pose_stamped[0].pose)

    def move_to_joint_compliant(
        self,
        arm_key: str,
        target: JointTarget,
        position_tol: float = POSITION_TOLERANCE_M,
        orientation_tol: float = ORIENTATION_TOLERANCE_RAD,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> bool:
        """FK target.joint_positions (via MoveIt's compute_fk), publish the
        result as target_frame, and set the joint vector itself as the
        controller's nullspace target (via set_nullspace_target(), a
        service call) so redundancy resolution favors the captured elbow
        configuration. See module docstring: this is a soft nullspace pull,
        not a hard joint-space constraint."""
        target_pose = self._compute_fk(arm_key, target.joint_positions)
        if target_pose is None:
            return False

        info = ARM_KEYS[arm_key]
        self.set_nullspace_target(arm_key, target.joint_positions)
        self._target_frame_pub[arm_key].publish(
            _se3_to_pose_stamped(target_pose, info["base_frame"])
        )
        return self._wait_until_settled(
            arm_key, target_pose, position_tol, orientation_tol, timeout_s
        )

    # -- MoveIt-planned multi-waypoint trajectories ------------------------------

    def plan_joint_trajectory(
        self, targets: list[JointTarget], group_name: str
    ) -> Optional[RobotTrajectory]:
        """Plan-only (never executes) MoveGroup call, returning the planned
        trajectory for execute_planned_trajectory() to stream through the
        impedance controller -- MoveIt still does the planning/collision
        checking, this controller only does execution. A standalone caller
        rather than reusing DualArmMoveitClient (which discards
        result.planned_trajectory) -- moveit_dual_arm.py is left unmodified
        per the agreed additive-only scope."""
        goal = MoveGroup.Goal()
        req = goal.request
        req.group_name = group_name
        req.num_planning_attempts = NUM_PLANNING_ATTEMPTS
        req.allowed_planning_time = PLANNING_TIME_S
        req.max_velocity_scaling_factor = VELOCITY_SCALING
        req.max_acceleration_scaling_factor = ACCELERATION_SCALING
        for target in targets:
            req.goal_constraints.append(_joint_goal_constraints(target))
        goal.planning_options.plan_only = True
        goal.planning_options.replan = False

        send_future = self._move_group_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self._node, send_future)
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            return None

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self._node, result_future)
        result_wrapper = result_future.result()
        if result_wrapper is None or result_wrapper.status != GoalStatus.STATUS_SUCCEEDED:
            return None
        return result_wrapper.result.planned_trajectory

    def execute_planned_trajectory(
        self,
        arm_key: str,
        traj: RobotTrajectory,
        rate_hz: float = 20.0,
        hold_s: float = 0.3,
        timeout_s: float = 60.0,
    ) -> bool:
        """Resample traj.joint_trajectory's waypoints and stream each as an
        FK'd target_frame plus a set_nullspace_target() service call to the
        impedance controller at rate_hz, then hold the final waypoint until
        settled -- turns the controller into a compliant trajectory
        follower instead of a snap-to-pose one. set_nullspace_target() has
        no stationary guard (it's a target, not a gain -- see module
        docstring), so this works while the arm is actively moving between
        waypoints."""
        jt = traj.joint_trajectory
        if not jt.points:
            return False

        dt = 1.0 / rate_hz
        info = ARM_KEYS[arm_key]
        for point in jt.points:
            joint_positions = dict(zip(jt.joint_names, point.positions))
            target_pose = self._compute_fk(arm_key, joint_positions)
            if target_pose is None:
                return False
            self.set_nullspace_target(arm_key, joint_positions)
            self._target_frame_pub[arm_key].publish(
                _se3_to_pose_stamped(target_pose, info["base_frame"])
            )
            self._spin_sleep(dt)

        final_positions = dict(zip(jt.joint_names, jt.points[-1].positions))
        final_pose = self._compute_fk(arm_key, final_positions)
        if final_pose is None:
            return False
        return self._wait_until_settled(
            arm_key, final_pose, POSITION_TOLERANCE_M, ORIENTATION_TOLERANCE_RAD, timeout_s
        )

    # -- Runtime gains / nullspace target, both via set_parameters only -----------

    def _call_set_parameters(self, arm_key: str, params: list[Parameter]) -> None:
        if not params:
            return
        client = self._set_params_client[arm_key]
        if not client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError(
                f"set_parameters service not available for arm_key={arm_key!r} -- "
                f"is cartesian_impedance.launch.py running?"
            )
        req = SetParameters.Request(parameters=params)
        future = client.call_async(req)
        rclpy.spin_until_future_complete(self._node, future)
        result = future.result()
        if result is None or not all(r.successful for r in result.results):
            raise RuntimeError(
                f"set_parameters failed for arm_key={arm_key!r}: "
                f"{[r.reason for r in (result.results if result else [])]}"
            )

    def set_gains(
        self,
        arm_key: str,
        gains: GainSettings,
        velocity_threshold: float = GAIN_CHANGE_VELOCITY_THRESHOLD_RAD_S,
    ) -> None:
        """Live-set stiffness.*/nullspace_stiffness on the running
        controller via its set_parameters service -- the patched controller
        re-reads these every cycle, no restart needed, and there is no
        other way to change them (no free topic). These are GAINS, so only
        safe while the arm is stationary (agreed constraint); raises
        GainChangeUnsafeError if recent joint velocities aren't settled --
        callers must not bypass this by catching-and-ignoring it. For the
        nullspace TARGET (which configuration to bias toward, as opposed to
        how strongly), see set_nullspace_target() -- no stationary guard,
        since it's a moving goal like target_frame, not a gain."""
        if not self._joint_velocities_settled(arm_key, velocity_threshold):
            raise GainChangeUnsafeError(
                f"Refusing to change gains for arm_key={arm_key!r}: joint velocities "
                f"are not below {velocity_threshold} rad/s. Gain changes are only "
                f"safe while the arm is stationary."
            )

        params = []
        if gains.stiffness:
            for axis, value in gains.stiffness.items():
                params.append(
                    Parameter(
                        name=f"stiffness.{axis}",
                        value=ParameterValue(
                            type=ParameterType.PARAMETER_DOUBLE, double_value=float(value)
                        ),
                    )
                )
        if gains.nullspace_stiffness is not None:
            params.append(
                Parameter(
                    name="nullspace_stiffness",
                    value=ParameterValue(
                        type=ParameterType.PARAMETER_DOUBLE,
                        double_value=float(gains.nullspace_stiffness),
                    ),
                )
            )
        self._call_set_parameters(arm_key, params)

    def set_nullspace_target(self, arm_key: str, joint_positions: dict) -> None:
        """Live-set nullspace_desired_configuration via set_parameters --
        same service, same live-reread mechanism as set_gains(), but no
        stationary guard: this is the TARGET the nullspace spring pulls
        toward (like target_frame), not the spring's stiffness, so it's
        expected to change while the arm is moving (see
        execute_planned_trajectory()). Has no effect on its own unless
        nullspace_stiffness has also been raised above 0 via set_gains()."""
        info = ARM_KEYS[arm_key]
        try:
            ordered = [float(joint_positions[name]) for name in info["joint_names"]]
        except KeyError as e:
            raise ValueError(
                f"joint_positions is missing joint {e} -- expected all of "
                f"{info['joint_names']}."
            ) from e

        params = [
            Parameter(
                name="nullspace_desired_configuration",
                value=ParameterValue(
                    type=ParameterType.PARAMETER_DOUBLE_ARRAY, double_array_value=ordered
                ),
            )
        ]
        self._call_set_parameters(arm_key, params)
