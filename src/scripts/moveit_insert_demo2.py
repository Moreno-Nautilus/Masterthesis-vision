from __future__ import annotations

import time
import threading
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

from std_srvs.srv import Trigger
from std_msgs.msg import Header, String
from geometry_msgs.msg import PoseStamped

from moveit_msgs.srv import GetPositionIK, GetMotionPlan
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.msg import Constraints, JointConstraint, MoveItErrorCodes


FAST_QOS = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
)


def quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    q = q / (np.linalg.norm(q) + 1e-12)
    x, y, z, w = q

    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def rot_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    q = np.empty(4, dtype=np.float64)
    trace = np.trace(R)

    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        q[3] = 0.25 * s
        q[0] = (R[2, 1] - R[1, 2]) / s
        q[1] = (R[0, 2] - R[2, 0]) / s
        q[2] = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        q[3] = (R[2, 1] - R[1, 2]) / s
        q[0] = 0.25 * s
        q[1] = (R[0, 1] + R[1, 0]) / s
        q[2] = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        q[3] = (R[0, 2] - R[2, 0]) / s
        q[0] = (R[0, 1] + R[1, 0]) / s
        q[1] = 0.25 * s
        q[2] = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        q[3] = (R[1, 0] - R[0, 1]) / s
        q[0] = (R[0, 2] + R[2, 0]) / s
        q[1] = (R[1, 2] + R[2, 1]) / s
        q[2] = 0.25 * s

    q = q / (np.linalg.norm(q) + 1e-12)
    return q.astype(np.float32)


def rpy_to_rot(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    Rx = np.array([
        [1, 0, 0],
        [0, cr, -sr],
        [0, sr, cr],
    ], dtype=np.float64)

    Ry = np.array([
        [cp, 0, sp],
        [0, 1, 0],
        [-sp, 0, cp],
    ], dtype=np.float64)

    Rz = np.array([
        [cy, -sy, 0],
        [sy, cy, 0],
        [0, 0, 1],
    ], dtype=np.float64)

    return Rz @ Ry @ Rx


def pose_to_T(msg: PoseStamped) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_to_rot(
        msg.pose.orientation.x,
        msg.pose.orientation.y,
        msg.pose.orientation.z,
        msg.pose.orientation.w,
    )
    T[0, 3] = msg.pose.position.x
    T[1, 3] = msg.pose.position.y
    T[2, 3] = msg.pose.position.z
    return T


def T_to_pose(T: np.ndarray, frame_id: str, stamp) -> PoseStamped:
    msg = PoseStamped()
    msg.header = Header(frame_id=frame_id, stamp=stamp)
    msg.pose.position.x = float(T[0, 3])
    msg.pose.position.y = float(T[1, 3])
    msg.pose.position.z = float(T[2, 3])

    q = rot_to_quat_xyzw(T[:3, :3])
    msg.pose.orientation.x = float(q[0])
    msg.pose.orientation.y = float(q[1])
    msg.pose.orientation.z = float(q[2])
    msg.pose.orientation.w = float(q[3])
    return msg


class MoveItInsertStages(Node):
    def __init__(self) -> None:
        super().__init__("moveit_insert_stages")

        self.cb_group = ReentrantCallbackGroup()

        # ----- params -----
        self.declare_parameter("robot_ip", "127.0.0.1")
        self.declare_parameter("use_fake_hardware", True)
        self.declare_parameter("fake_sensor_commands", True)

        self.declare_parameter("planning_group", "fr3_arm")
        self.declare_parameter("pose_link", "fr3_link8")

        self.declare_parameter(
            "cooling_base_topic",
            "/perception/fp/pose_base/zed2i_2/cooling_base_0",
        )

        self.declare_parameter("hole_x_m", -0.0332)
        self.declare_parameter("hole_y_m", 0.020)
        self.declare_parameter("hole_z_m", 0.03)
        self.declare_parameter("hole_mirror_axis", "y")

        self.declare_parameter("preinsert_dz_m", 0.04)
        self.declare_parameter("insert_dz_m", 0.02)
        self.declare_parameter("seat_dz_m", 0.01)

        self.declare_parameter("ee_to_screw_tip_z_m", 0.06)
        self.declare_parameter("post_execute_sleep_s", 1.0)

        self.declare_parameter("ik_timeout_s", 2.0)
        self.declare_parameter("planning_time_s", 5.0)
        self.declare_parameter("velocity_scale", 0.2)
        self.declare_parameter("acceleration_scale", 0.2)

        self.robot_ip = str(self.get_parameter("robot_ip").value)
        self.use_fake_hardware = bool(self.get_parameter("use_fake_hardware").value)
        self.fake_sensor_commands = bool(self.get_parameter("fake_sensor_commands").value)

        self.planning_group = str(self.get_parameter("planning_group").value)
        self.pose_link = str(self.get_parameter("pose_link").value)
        self.cooling_base_topic = str(self.get_parameter("cooling_base_topic").value)

        hole1_offset = np.array(
            [
                float(self.get_parameter("hole_x_m").value),
                float(self.get_parameter("hole_y_m").value),
                float(self.get_parameter("hole_z_m").value),
            ],
            dtype=np.float64,
        )

        hole_mirror_axis = str(self.get_parameter("hole_mirror_axis").value)
        hole2_offset = hole1_offset.copy()

        if hole_mirror_axis == "x":
            hole2_offset[0] *= -1.0
        elif hole_mirror_axis == "y":
            hole2_offset[1] *= -1.0
        elif hole_mirror_axis == "xy":
            hole2_offset[0] *= -1.0
            hole2_offset[1] *= -1.0
        else:
            raise ValueError(f"Unsupported hole_mirror_axis='{hole_mirror_axis}'")

        self.hole_offsets = {
            "hole1": hole1_offset,
            "hole2": hole2_offset,
        }

        self.preinsert_dz_m = float(self.get_parameter("preinsert_dz_m").value)
        self.insert_dz_m = float(self.get_parameter("insert_dz_m").value)
        self.seat_dz_m = float(self.get_parameter("seat_dz_m").value)

        self.ee_to_screw_tip_z_m = float(self.get_parameter("ee_to_screw_tip_z_m").value)
        self.post_execute_sleep_s = float(self.get_parameter("post_execute_sleep_s").value)

        self.ik_timeout_s = float(self.get_parameter("ik_timeout_s").value)
        self.planning_time_s = float(self.get_parameter("planning_time_s").value)
        self.velocity_scale = float(self.get_parameter("velocity_scale").value)
        self.acceleration_scale = float(self.get_parameter("acceleration_scale").value)

        # ----- latest perception + latched targets -----
        self._lock = threading.Lock()
        self.latest_base_pose_msg: PoseStamped | None = None
        self.latched_targets: dict[str, dict[str, np.ndarray]] = {}

        # ----- fixed top-down insertion orientation -----
        self.R_insert_fixed = rpy_to_rot(np.pi, 0.0, np.pi / 2.0)

        # ----- fixed EE -> screw_tip transform -----
        self.T_ee_screw_tip = np.eye(4, dtype=np.float64)
        self.T_ee_screw_tip[2, 3] = self.ee_to_screw_tip_z_m
        self.T_screw_tip_ee = np.linalg.inv(self.T_ee_screw_tip)

        # ----- subscriptions -----
        self.sub = self.create_subscription(
            PoseStamped,
            self.cooling_base_topic,
            self._base_cb,
            FAST_QOS,
            callback_group=self.cb_group,
        )

        # ----- hole 1 services: keep original names -----
        self.srv_preinsert = self.create_service(
            Trigger,
            "/assembly/preinsert",
            self._preinsert_cb,
            callback_group=self.cb_group,
        )
        self.srv_insert = self.create_service(
            Trigger,
            "/assembly/insert",
            self._insert_cb,
            callback_group=self.cb_group,
        )
        self.srv_seat = self.create_service(
            Trigger,
            "/assembly/seat",
            self._seat_cb,
            callback_group=self.cb_group,
        )
        self.srv_retreat = self.create_service(
            Trigger,
            "/assembly/retreat",
            self._retreat_cb,
            callback_group=self.cb_group,
        )

        # ----- hole 2 services -----
        self.srv_preinsert_hole2 = self.create_service(
            Trigger,
            "/assembly/preinsert_hole2",
            self._preinsert_hole2_cb,
            callback_group=self.cb_group,
        )
        self.srv_insert_hole2 = self.create_service(
            Trigger,
            "/assembly/insert_hole2",
            self._insert_hole2_cb,
            callback_group=self.cb_group,
        )
        self.srv_seat_hole2 = self.create_service(
            Trigger,
            "/assembly/seat_hole2",
            self._seat_hole2_cb,
            callback_group=self.cb_group,
        )
        self.srv_retreat_hole2 = self.create_service(
            Trigger,
            "/assembly/retreat_hole2",
            self._retreat_hole2_cb,
            callback_group=self.cb_group,
        )

        # ----- debug publishers: keep original topics for hole 1 -----
        self.pub_hole = self.create_publisher(PoseStamped, "/assembly/hole_pose", 10)
        self.pub_preinsert = self.create_publisher(PoseStamped, "/assembly/preinsert_pose", 10)
        self.pub_insert = self.create_publisher(PoseStamped, "/assembly/insert_pose", 10)
        self.pub_seat = self.create_publisher(PoseStamped, "/assembly/seat_pose", 10)

        self.pub_preinsert_ee = self.create_publisher(PoseStamped, "/assembly/preinsert_ee_pose", 10)
        self.pub_insert_ee = self.create_publisher(PoseStamped, "/assembly/insert_ee_pose", 10)
        self.pub_seat_ee = self.create_publisher(PoseStamped, "/assembly/seat_ee_pose", 10)

        # ----- debug publishers for hole 2 -----
        self.pub_hole_hole2 = self.create_publisher(PoseStamped, "/assembly/hole_pose_hole2", 10)
        self.pub_preinsert_hole2 = self.create_publisher(PoseStamped, "/assembly/preinsert_pose_hole2", 10)
        self.pub_insert_hole2 = self.create_publisher(PoseStamped, "/assembly/insert_pose_hole2", 10)
        self.pub_seat_hole2 = self.create_publisher(PoseStamped, "/assembly/seat_pose_hole2", 10)

        self.pub_preinsert_ee_hole2 = self.create_publisher(PoseStamped, "/assembly/preinsert_ee_pose_hole2", 10)
        self.pub_insert_ee_hole2 = self.create_publisher(PoseStamped, "/assembly/insert_ee_pose_hole2", 10)
        self.pub_seat_ee_hole2 = self.create_publisher(PoseStamped, "/assembly/seat_ee_pose_hole2", 10)

        self.pub_next_stage_ee = self.create_publisher(PoseStamped, "/assembly/next_stage_ee_pose", 10)
        self.pub_next_stage_tip = self.create_publisher(PoseStamped, "/assembly/next_stage_screw_tip_pose", 10)
        self.pub_next_stage_name = self.create_publisher(String, "/assembly/next_stage_name", 10)

        # ----- MoveIt clients -----
        self.ik_client = self.create_client(
            GetPositionIK,
            "/compute_ik",
            callback_group=self.cb_group,
        )
        self.plan_client = self.create_client(
            GetMotionPlan,
            "/plan_kinematic_path",
            callback_group=self.cb_group,
        )
        self.exec_client = ActionClient(
            self,
            ExecuteTrajectory,
            "/execute_trajectory",
            callback_group=self.cb_group,
        )

        self.group_joint_names = [f"fr3_joint{i}" for i in range(1, 8)]

        if not self.ik_client.wait_for_service(timeout_sec=10.0):
            raise RuntimeError("Service /compute_ik not available")
        if not self.plan_client.wait_for_service(timeout_sec=10.0):
            raise RuntimeError("Service /plan_kinematic_path not available")
        if not self.exec_client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError("Action /execute_trajectory not available")

        self.get_logger().info(
            f"MoveItInsertStages started | group={self.planning_group} "
            f"pose_link={self.pose_link} fake={self.use_fake_hardware}"
        )
        self.get_logger().info(
            f"Hole offsets | hole1={self.hole_offsets['hole1']} | hole2={self.hole_offsets['hole2']}"
        )

    # -------------------------------------------------------------------------
    # perception + target computation
    # -------------------------------------------------------------------------

    def _base_cb(self, msg: PoseStamped) -> None:
        with self._lock:
            self.latest_base_pose_msg = msg

        targets_hole1 = self._compute_targets(msg, self.hole_offsets["hole1"])
        targets_hole2 = self._compute_targets(msg, self.hole_offsets["hole2"])

        if targets_hole1 is None or targets_hole2 is None:
            return

        stamp = self.get_clock().now().to_msg()

        # hole 1 on original topics
        self.pub_hole.publish(T_to_pose(targets_hole1["T_base_hole"], "base", stamp))
        self.pub_preinsert.publish(T_to_pose(targets_hole1["T_base_preinsert"], "base", stamp))
        self.pub_insert.publish(T_to_pose(targets_hole1["T_base_insert"], "base", stamp))
        self.pub_seat.publish(T_to_pose(targets_hole1["T_base_seat"], "base", stamp))

        self.pub_preinsert_ee.publish(T_to_pose(targets_hole1["T_base_preinsert_ee"], "base", stamp))
        self.pub_insert_ee.publish(T_to_pose(targets_hole1["T_base_insert_ee"], "base", stamp))
        self.pub_seat_ee.publish(T_to_pose(targets_hole1["T_base_seat_ee"], "base", stamp))

        # hole 2 on separate topics
        self.pub_hole_hole2.publish(T_to_pose(targets_hole2["T_base_hole"], "base", stamp))
        self.pub_preinsert_hole2.publish(T_to_pose(targets_hole2["T_base_preinsert"], "base", stamp))
        self.pub_insert_hole2.publish(T_to_pose(targets_hole2["T_base_insert"], "base", stamp))
        self.pub_seat_hole2.publish(T_to_pose(targets_hole2["T_base_seat"], "base", stamp))

        self.pub_preinsert_ee_hole2.publish(T_to_pose(targets_hole2["T_base_preinsert_ee"], "base", stamp))
        self.pub_insert_ee_hole2.publish(T_to_pose(targets_hole2["T_base_insert_ee"], "base", stamp))
        self.pub_seat_ee_hole2.publish(T_to_pose(targets_hole2["T_base_seat_ee"], "base", stamp))

    def _compute_targets(self, msg: PoseStamped, hole_offset: np.ndarray) -> dict[str, np.ndarray] | None:
        T_base_cooling_base = pose_to_T(msg)
        p_base_cooling_base = T_base_cooling_base[:3, 3]

        p_base_hole = p_base_cooling_base + hole_offset

        T_base_hole = np.eye(4, dtype=np.float64)
        T_base_hole[:3, :3] = self.R_insert_fixed
        T_base_hole[:3, 3] = p_base_hole

        T_base_preinsert = T_base_hole.copy()
        T_base_preinsert[2, 3] += self.preinsert_dz_m

        T_base_insert = T_base_hole.copy()
        T_base_insert[2, 3] += self.insert_dz_m

        T_base_seat = T_base_hole.copy()
        T_base_seat[2, 3] += self.seat_dz_m

        T_base_preinsert_ee = T_base_preinsert @ self.T_screw_tip_ee
        T_base_insert_ee = T_base_insert @ self.T_screw_tip_ee
        T_base_seat_ee = T_base_seat @ self.T_screw_tip_ee

        return {
            "T_base_hole": T_base_hole,
            "T_base_preinsert": T_base_preinsert,
            "T_base_insert": T_base_insert,
            "T_base_seat": T_base_seat,
            "T_base_preinsert_ee": T_base_preinsert_ee,
            "T_base_insert_ee": T_base_insert_ee,
            "T_base_seat_ee": T_base_seat_ee,
        }

    def _latch_targets_from_latest(self, hole_name: str) -> tuple[dict[str, np.ndarray] | None, str]:
        with self._lock:
            latest = self.latest_base_pose_msg

        if latest is None:
            return None, "No cooling base pose received yet"

        if hole_name not in self.hole_offsets:
            return None, f"Unknown hole: {hole_name}"

        targets = self._compute_targets(latest, self.hole_offsets[hole_name])
        if targets is None:
            return None, f"Failed to compute targets for {hole_name}"

        with self._lock:
            self.latched_targets[hole_name] = targets

        return targets, "ok"

    def _get_latched_targets(self, hole_name: str) -> tuple[dict[str, np.ndarray] | None, str]:
        with self._lock:
            targets = self.latched_targets.get(hole_name)

        if targets is None:
            return None, f"No latched targets for {hole_name}. Call preinsert first."

        return targets, "ok"

    # -------------------------------------------------------------------------
    # visualization / intent publishing
    # -------------------------------------------------------------------------

    def _pose_from_target_key(self, targets: dict[str, np.ndarray], key: str) -> PoseStamped:
        stamp = self.get_clock().now().to_msg()
        return T_to_pose(targets[key], "base", stamp)

    def _publish_next_intent(self, targets: dict[str, np.ndarray], next_stage: str, hole_name: str) -> None:
        stage_to_ee_key = {
            "preinsert": "T_base_preinsert_ee",
            "insert": "T_base_insert_ee",
            "seat": "T_base_seat_ee",
            "retreat": "T_base_preinsert_ee",
        }
        stage_to_tip_key = {
            "preinsert": "T_base_preinsert",
            "insert": "T_base_insert",
            "seat": "T_base_seat",
            "retreat": "T_base_preinsert",
        }

        if next_stage not in stage_to_ee_key:
            return

        stamp = self.get_clock().now().to_msg()

        ee_pose = T_to_pose(targets[stage_to_ee_key[next_stage]], "base", stamp)
        tip_pose = T_to_pose(targets[stage_to_tip_key[next_stage]], "base", stamp)

        self.pub_next_stage_ee.publish(ee_pose)
        self.pub_next_stage_tip.publish(tip_pose)
        self.pub_next_stage_name.publish(String(data=f"{hole_name}_{next_stage}"))

        self.get_logger().info(
            f"Next intended stage: {hole_name}_{next_stage} | "
            f"EE xyz=({ee_pose.pose.position.x:.4f}, "
            f"{ee_pose.pose.position.y:.4f}, "
            f"{ee_pose.pose.position.z:.4f})"
        )

    # -------------------------------------------------------------------------
    # MoveIt helpers
    # -------------------------------------------------------------------------

    def _wait_future(self, future, timeout_s: float):
        done = threading.Event()
        future.add_done_callback(lambda _: done.set())
        if not done.wait(timeout_s):
            raise TimeoutError("Timed out waiting for ROS future")
        return future.result()

    def _compute_ik(self, pose_msg: PoseStamped) -> tuple[list[float] | None, str]:
        req = GetPositionIK.Request()
        req.ik_request.group_name = self.planning_group
        req.ik_request.ik_link_name = self.pose_link
        req.ik_request.pose_stamped = pose_msg
        req.ik_request.avoid_collisions = True

        secs = int(self.ik_timeout_s)
        nsecs = int((self.ik_timeout_s - secs) * 1e9)
        req.ik_request.timeout.sec = secs
        req.ik_request.timeout.nanosec = nsecs

        fut = self.ik_client.call_async(req)
        try:
            res = self._wait_future(fut, self.ik_timeout_s + 3.0)
        except Exception as e:
            return None, f"IK call failed: {e}"

        if res is None:
            return None, "IK response was None"

        if res.error_code.val != MoveItErrorCodes.SUCCESS:
            return None, f"IK failed code={res.error_code.val}"

        name_to_pos = dict(zip(res.solution.joint_state.name, res.solution.joint_state.position))
        missing = [n for n in self.group_joint_names if n not in name_to_pos]
        if missing:
            return None, f"IK solution missing joints: {missing}"

        joints = [float(name_to_pos[n]) for n in self.group_joint_names]
        return joints, "ok"

    def _plan_joint_goal(self, joint_positions: list[float], label: str):
        req = GetMotionPlan.Request()

        mpr = req.motion_plan_request
        mpr.group_name = self.planning_group
        mpr.num_planning_attempts = 5
        mpr.allowed_planning_time = self.planning_time_s
        mpr.max_velocity_scaling_factor = self.velocity_scale
        mpr.max_acceleration_scaling_factor = self.acceleration_scale

        mpr.start_state.is_diff = True

        goal = Constraints()
        goal.name = label

        for joint_name, pos in zip(self.group_joint_names, joint_positions):
            jc = JointConstraint()
            jc.joint_name = joint_name
            jc.position = float(pos)
            jc.tolerance_above = 0.001
            jc.tolerance_below = 0.001
            jc.weight = 1.0
            goal.joint_constraints.append(jc)

        mpr.goal_constraints.append(goal)

        fut = self.plan_client.call_async(req)
        try:
            res = self._wait_future(fut, self.planning_time_s + 5.0)
        except Exception as e:
            return None, f"planning call failed: {e}"

        if res is None:
            return None, "planning response was None"

        ec = res.motion_plan_response.error_code
        if ec.val != MoveItErrorCodes.SUCCESS:
            return None, f"planning failed code={ec.val}"

        return res.motion_plan_response.trajectory, "ok"

    def _execute_trajectory(self, traj, label: str) -> tuple[bool, str]:
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = traj

        send_future = self.exec_client.send_goal_async(goal)
        try:
            goal_handle = self._wait_future(send_future, 5.0)
        except Exception as e:
            return False, f"send goal failed for {label}: {e}"

        if goal_handle is None or not goal_handle.accepted:
            return False, f"execute goal rejected for {label}"

        result_future = goal_handle.get_result_async()
        try:
            wrapped = self._wait_future(result_future, 120.0)
        except Exception as e:
            return False, f"execution wait failed for {label}: {e}"

        result = wrapped.result
        if result.error_code.val != MoveItErrorCodes.SUCCESS:
            return False, f"execution failed for {label}: code={result.error_code.val}"

        time.sleep(self.post_execute_sleep_s)
        return True, f"{label} done"

    def _plan_and_execute_pose(self, pose_msg: PoseStamped, label: str) -> tuple[bool, str]:
        self.get_logger().info(f"IK for {label}")
        joints, msg = self._compute_ik(pose_msg)
        if joints is None:
            return False, f"{label}: {msg}"

        self.get_logger().info(f"Planning {label}")
        traj, msg = self._plan_joint_goal(joints, label)
        if traj is None:
            return False, f"{label}: {msg}"

        self.get_logger().info(f"Executing {label}")
        return self._execute_trajectory(traj, label)

    def _run_stage(
        self,
        hole_name: str,
        target_key: str,
        label: str,
        latch_from_latest: bool,
        next_stage: str | None,
    ) -> tuple[bool, str]:
        if latch_from_latest:
            targets, msg = self._latch_targets_from_latest(hole_name)
        else:
            targets, msg = self._get_latched_targets(hole_name)

        if targets is None:
            return False, msg

        pose_msg = self._pose_from_target_key(targets, target_key)
        ok, msg = self._plan_and_execute_pose(pose_msg, label)
        if not ok:
            return False, msg

        if next_stage is not None:
            self._publish_next_intent(targets, next_stage, hole_name)

        return True, msg

    # -------------------------------------------------------------------------
    # hole 1 callbacks
    # -------------------------------------------------------------------------

    def _preinsert_cb(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        del request
        ok, msg = self._run_stage(
            hole_name="hole1",
            target_key="T_base_preinsert_ee",
            label="hole1_preinsert",
            latch_from_latest=True,
            next_stage="insert",
        )
        response.success = ok
        response.message = msg
        return response

    def _insert_cb(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        del request
        ok, msg = self._run_stage(
            hole_name="hole1",
            target_key="T_base_insert_ee",
            label="hole1_insert",
            latch_from_latest=False,
            next_stage="seat",
        )
        response.success = ok
        response.message = msg
        return response

    def _seat_cb(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        del request
        ok, msg = self._run_stage(
            hole_name="hole1",
            target_key="T_base_seat_ee",
            label="hole1_seat",
            latch_from_latest=False,
            next_stage="retreat",
        )
        response.success = ok
        response.message = msg
        return response

    def _retreat_cb(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        del request
        ok, msg = self._run_stage(
            hole_name="hole1",
            target_key="T_base_preinsert_ee",
            label="hole1_retreat",
            latch_from_latest=False,
            next_stage=None,
        )
        response.success = ok
        response.message = msg
        if ok:
            self.pub_next_stage_name.publish(String(data="hole1_done"))
            self.get_logger().info("Hole 1 sequence complete.")
        return response

    # -------------------------------------------------------------------------
    # hole 2 callbacks
    # -------------------------------------------------------------------------

    def _preinsert_hole2_cb(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        del request
        ok, msg = self._run_stage(
            hole_name="hole2",
            target_key="T_base_preinsert_ee",
            label="hole2_preinsert",
            latch_from_latest=True,
            next_stage="insert",
        )
        response.success = ok
        response.message = msg
        return response

    def _insert_hole2_cb(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        del request
        ok, msg = self._run_stage(
            hole_name="hole2",
            target_key="T_base_insert_ee",
            label="hole2_insert",
            latch_from_latest=False,
            next_stage="seat",
        )
        response.success = ok
        response.message = msg
        return response

    def _seat_hole2_cb(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        del request
        ok, msg = self._run_stage(
            hole_name="hole2",
            target_key="T_base_seat_ee",
            label="hole2_seat",
            latch_from_latest=False,
            next_stage="retreat",
        )
        response.success = ok
        response.message = msg
        return response

    def _retreat_hole2_cb(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        del request
        ok, msg = self._run_stage(
            hole_name="hole2",
            target_key="T_base_preinsert_ee",
            label="hole2_retreat",
            latch_from_latest=False,
            next_stage=None,
        )
        response.success = ok
        response.message = msg
        if ok:
            self.pub_next_stage_name.publish(String(data="hole2_done"))
            self.get_logger().info("Hole 2 sequence complete.")
        return response


def main() -> None:
    rclpy.init()
    node = MoveItInsertStages()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()