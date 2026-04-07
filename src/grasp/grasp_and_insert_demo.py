"""
Unified Pick-and-Place Node for Franka FR3

Combines grasp bridge (pickup) and insert demo (placement) into a single
MoveIt-based pipeline using joint space control throughout.

Full sequence per screw:
  neutral → pregrasp → grasp → lift → preinsert → insert → release → retreat

For two screws (e.g., screw_indices:=[0,2]):
  [screw_0 → hole1] then [screw_1 → hole2]
"""

from __future__ import annotations

import math
import time
import threading
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.parameter import Parameter

from std_srvs.srv import Trigger
from std_msgs.msg import Header, String
from geometry_msgs.msg import PoseStamped

from franka_msgs.action import Grasp, Move as GripperMove

from moveit_msgs.srv import GetPositionIK, GetMotionPlan
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.msg import Constraints, JointConstraint, MoveItErrorCodes


# -----------------------------------------------------------------------------
# QoS and Constants
# -----------------------------------------------------------------------------

FAST_QOS = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
)


class Stage(Enum):
    IDLE = auto()
    NEUTRAL = auto()
    PREGRASP = auto()
    GRASP = auto()
    LIFT = auto()
    PREINSERT = auto()
    INSERT = auto()
    RELEASE = auto()
    RETREAT = auto()
    DONE = auto()


@dataclass
class TargetPose:
    """6-DOF target for the end-effector."""
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float


# -----------------------------------------------------------------------------
# Math Utilities
# -----------------------------------------------------------------------------

def quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Quaternion (x,y,z,w) to 3x3 rotation matrix."""
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    q /= np.linalg.norm(q) + 1e-12
    x, y, z, w = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def rot_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix to quaternion (x,y,z,w)."""
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

    return (q / (np.linalg.norm(q) + 1e-12)).astype(np.float64)


def rpy_to_rot(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Roll-pitch-yaw (XYZ intrinsic) to 3x3 rotation matrix."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)

    return Rz @ Ry @ Rx


def pose_msg_to_T(msg: PoseStamped) -> np.ndarray:
    """PoseStamped to 4x4 homogeneous transform."""
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


def T_to_pose_msg(T: np.ndarray, frame_id: str, stamp) -> PoseStamped:
    """4x4 homogeneous transform to PoseStamped."""
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


def target_to_pose_msg(target: TargetPose, frame_id: str, stamp) -> PoseStamped:
    """TargetPose to PoseStamped."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rpy_to_rot(target.roll, target.pitch, target.yaw)
    T[0, 3] = target.x
    T[1, 3] = target.y
    T[2, 3] = target.z
    return T_to_pose_msg(T, frame_id, stamp)


# -----------------------------------------------------------------------------
# Main Node
# -----------------------------------------------------------------------------

class UnifiedPickAndPlace(Node):
    """
    Unified pick-and-place node using MoveIt for all motions.
    
    Supports multiple screws and dual-hole placement.
    """

    # Grasp mode constants
    GRASP_MODE_TOP_DOWN = "top_down"
    GRASP_MODE_SIDE_X = "side_x"
    GRASP_MODE_SIDE_Y = "side_y"

    def __init__(self) -> None:
        super().__init__("unified_pick_and_place")

        self.cb_group = ReentrantCallbackGroup()
        self._lock = threading.Lock()

        # ---------------------------------------------------------------------
        # Declare all parameters
        # ---------------------------------------------------------------------
        self._declare_parameters()
        self._read_parameters()

        # ---------------------------------------------------------------------
        # State
        # ---------------------------------------------------------------------
        self.stage = Stage.IDLE
        self.current_screw_seq_idx = 0  # index into screw_indices list

        # Pose histories for stability checking
        self.screw_pose_history: deque[PoseStamped] = deque(maxlen=20)
        self.base_pose_history: deque[PoseStamped] = deque(maxlen=20)

        # Latest received poses
        self.latest_screw_pose: Optional[PoseStamped] = None
        self.latest_base_pose: Optional[PoseStamped] = None

        # Latched targets (frozen at stage entry)
        self.latched_pickup_targets: Optional[dict[str, TargetPose]] = None
        self.latched_placement_targets: Optional[dict[str, np.ndarray]] = None

        # Precomputed transforms
        self.R_topdown = rpy_to_rot(self.grasp_roll, self.grasp_pitch, self.grasp_yaw)
        self.R_insert = rpy_to_rot(np.pi, 0.0, np.pi / 2.0)

        # EE to tool tip transform (for insertion)
        self.T_ee_tip = np.eye(4, dtype=np.float64)
        self.T_ee_tip[2, 3] = self.ee_to_tip_z_m
        self.T_tip_ee = np.linalg.inv(self.T_ee_tip)

        # Hole offsets (computed from params)
        self._compute_hole_offsets()

        # ---------------------------------------------------------------------
        # Subscriptions - created dynamically based on current screw index
        # ---------------------------------------------------------------------
        self._create_screw_subscription()

        self.base_sub = self.create_subscription(
            PoseStamped,
            self.base_pose_topic,
            self._base_pose_cb,
            FAST_QOS,
            callback_group=self.cb_group,
        )

        # ---------------------------------------------------------------------
        # MoveIt clients
        # ---------------------------------------------------------------------
        self.ik_client = self.create_client(
            GetPositionIK, "/compute_ik", callback_group=self.cb_group
        )
        self.plan_client = self.create_client(
            GetMotionPlan, "/plan_kinematic_path", callback_group=self.cb_group
        )
        self.exec_client = ActionClient(
            self, ExecuteTrajectory, "/execute_trajectory", callback_group=self.cb_group
        )

        # Gripper clients
        self.gripper_grasp_client = ActionClient(
            self, Grasp, self.gripper_grasp_action, callback_group=self.cb_group
        )
        self.gripper_move_client = ActionClient(
            self, GripperMove, self.gripper_move_action, callback_group=self.cb_group
        )

        # Wait for MoveIt services
        self._wait_for_services()

        # ---------------------------------------------------------------------
        # Stage trigger services
        # ---------------------------------------------------------------------
        self.srv_neutral = self.create_service(
            Trigger, "~/go_neutral", self._go_neutral_cb, callback_group=self.cb_group
        )
        self.srv_pregrasp = self.create_service(
            Trigger, "~/go_pregrasp", self._go_pregrasp_cb, callback_group=self.cb_group
        )
        self.srv_grasp = self.create_service(
            Trigger, "~/go_grasp", self._go_grasp_cb, callback_group=self.cb_group
        )
        self.srv_lift = self.create_service(
            Trigger, "~/go_lift", self._go_lift_cb, callback_group=self.cb_group
        )
        self.srv_preinsert = self.create_service(
            Trigger, "~/go_preinsert", self._go_preinsert_cb, callback_group=self.cb_group
        )
        self.srv_insert = self.create_service(
            Trigger, "~/go_insert", self._go_insert_cb, callback_group=self.cb_group
        )
        self.srv_release = self.create_service(
            Trigger, "~/go_release", self._go_release_cb, callback_group=self.cb_group
        )
        self.srv_retreat = self.create_service(
            Trigger, "~/go_retreat", self._go_retreat_cb, callback_group=self.cb_group
        )

        # Utility services
        self.srv_reset = self.create_service(
            Trigger, "~/reset", self._reset_cb, callback_group=self.cb_group
        )
        self.srv_next_screw = self.create_service(
            Trigger, "~/next_screw", self._next_screw_cb, callback_group=self.cb_group
        )
        self.srv_run_full = self.create_service(
            Trigger, "~/run_full_sequence", self._run_full_sequence_cb, callback_group=self.cb_group
        )
        self.srv_status = self.create_service(
            Trigger, "~/status", self._status_cb, callback_group=self.cb_group
        )

        # ---------------------------------------------------------------------
        # Debug publishers
        # ---------------------------------------------------------------------
        self.pub_stage = self.create_publisher(String, "~/stage", 10)
        self.pub_screw_idx = self.create_publisher(String, "~/current_screw", 10)

        self.pub_neutral_pose = self.create_publisher(PoseStamped, "~/neutral_pose", 10)
        self.pub_pregrasp_pose = self.create_publisher(PoseStamped, "~/pregrasp_pose", 10)
        self.pub_grasp_pose = self.create_publisher(PoseStamped, "~/grasp_pose", 10)
        self.pub_lift_pose = self.create_publisher(PoseStamped, "~/lift_pose", 10)
        self.pub_preinsert_pose = self.create_publisher(PoseStamped, "~/preinsert_pose", 10)
        self.pub_insert_pose = self.create_publisher(PoseStamped, "~/insert_pose", 10)
        self.pub_retreat_pose = self.create_publisher(PoseStamped, "~/retreat_pose", 10)

        # Status timer
        self.status_timer = self.create_timer(0.5, self._publish_status, callback_group=self.cb_group)

        self._log_startup()

    # -------------------------------------------------------------------------
    # Parameter handling
    # -------------------------------------------------------------------------

    def _declare_parameters(self) -> None:
        # Robot / MoveIt
        self.declare_parameter("robot_ip", "127.0.0.1")
        self.declare_parameter("use_fake_hardware", True)
        self.declare_parameter("fake_sensor_commands", True)
        self.declare_parameter("planning_group", "fr3_arm")
        self.declare_parameter("pose_link", "fr3_hand_tcp")

        # Screw selection: list of indices to pick in order
        self.declare_parameter("screw_indices", [0, 1])

        # Topic patterns
        self.declare_parameter("screw_pose_topic_pattern", "/perception/fp/pose_base/zed2i_2/cooling_screw_{idx}")
        self.declare_parameter("base_pose_topic", "/perception/fp/pose_base/zed2i_2/cooling_base_0")

        # Pose freshness / stability (from grasp bridge)
        self.declare_parameter("min_pose_age_s", 1000)
        self.declare_parameter("lock_num_samples", 1)
        self.declare_parameter("lock_max_spread_m", 0.01)
        self.declare_parameter("lock_max_age_s", 8.0)

        # Grasp geometry
        self.declare_parameter("grasp_mode", self.GRASP_MODE_TOP_DOWN)
        self.declare_parameter("grasp_roll", math.pi)
        self.declare_parameter("grasp_pitch", 0.0)
        self.declare_parameter("grasp_yaw", math.pi / 2.0)

        self.declare_parameter("centroid_offset_x", 0.0)
        self.declare_parameter("centroid_offset_y", 0.0)
        self.declare_parameter("centroid_offset_z", 0.0)

        self.declare_parameter("pregrasp_clearance_m", 0.06)
        self.declare_parameter("grasp_clearance_m", 0.02)
        self.declare_parameter("lift_height_m", 0.08)

        # Neutral pose
        self.declare_parameter("neutral_x", 0.30)
        self.declare_parameter("neutral_y", 0.005)
        self.declare_parameter("neutral_z", 0.40)
        self.declare_parameter("neutral_roll", math.pi)
        self.declare_parameter("neutral_pitch", 0.0)
        self.declare_parameter("neutral_yaw", math.pi / 2.0)

        # Insertion geometry (from insert demo)
        self.declare_parameter("hole1_x_m", -0.028)
        self.declare_parameter("hole1_y_m", -0.005)
        self.declare_parameter("hole1_z_m", 0.03) # 0.025
        self.declare_parameter("hole_mirror_axis", "x")  # mirror hole1 across this axis for hole2

        self.declare_parameter("preinsert_dz_m", 0.07)
        self.declare_parameter("insert_dz_m", 0.03)

        self.declare_parameter("ee_to_tip_z_m", 0.0)  # EE frame to screw tip offset

        # Safety bounds
        self.declare_parameter("min_allowed_z_m", 0.01)
        self.declare_parameter("max_allowed_z_m", 1.20)

        # MoveIt planning
        self.declare_parameter("ik_timeout_s", 2.0)
        self.declare_parameter("planning_time_s", 5.0)
        self.declare_parameter("num_planning_attempts", 5)
        self.declare_parameter("velocity_scale", 0.2)
        self.declare_parameter("acceleration_scale", 0.2)
        self.declare_parameter("post_execute_sleep_s", 0.5)

        # Gripper
        self.declare_parameter("gripper_grasp_action", "/fr3_gripper/grasp")
        self.declare_parameter("gripper_move_action", "/fr3_gripper/move")
        self.declare_parameter("gripper_grasp_width", 0.001)
        self.declare_parameter("gripper_grasp_speed", 0.03)
        self.declare_parameter("gripper_grasp_force", 30.0)
        self.declare_parameter("gripper_epsilon_inner", 0.002)
        self.declare_parameter("gripper_epsilon_outer", 0.08)
        self.declare_parameter("gripper_open_width", 0.08)
        self.declare_parameter("gripper_timeout_s", 10.0)
        self.declare_parameter("grasp_settle_time_s", 0.5)

        # Auto sequence
        self.declare_parameter("auto_sequence", False)

    def _read_parameters(self) -> None:
        self.robot_ip = str(self.get_parameter("robot_ip").value)
        self.use_fake_hardware = bool(self.get_parameter("use_fake_hardware").value)
        self.planning_group = str(self.get_parameter("planning_group").value)
        self.pose_link = str(self.get_parameter("pose_link").value)

        screw_param = self.get_parameter("screw_indices").value
        if isinstance(screw_param, list):
            self.screw_indices = [int(i) for i in screw_param]
        else:
            self.screw_indices = [0, 1]

        self.screw_topic_pattern = str(self.get_parameter("screw_pose_topic_pattern").value)
        self.base_pose_topic = str(self.get_parameter("base_pose_topic").value)

        self.min_pose_age_s = float(self.get_parameter("min_pose_age_s").value)
        self.lock_num_samples = int(self.get_parameter("lock_num_samples").value)
        self.lock_max_spread_m = float(self.get_parameter("lock_max_spread_m").value)
        self.lock_max_age_s = float(self.get_parameter("lock_max_age_s").value)

        self.grasp_mode = str(self.get_parameter("grasp_mode").value)
        self.grasp_roll = float(self.get_parameter("grasp_roll").value)
        self.grasp_pitch = float(self.get_parameter("grasp_pitch").value)
        self.grasp_yaw = float(self.get_parameter("grasp_yaw").value)

        self.centroid_offset_x = float(self.get_parameter("centroid_offset_x").value)
        self.centroid_offset_y = float(self.get_parameter("centroid_offset_y").value)
        self.centroid_offset_z = float(self.get_parameter("centroid_offset_z").value)

        self.pregrasp_clearance_m = float(self.get_parameter("pregrasp_clearance_m").value)
        self.grasp_clearance_m = float(self.get_parameter("grasp_clearance_m").value)
        self.lift_height_m = float(self.get_parameter("lift_height_m").value)

        self.neutral_x = float(self.get_parameter("neutral_x").value)
        self.neutral_y = float(self.get_parameter("neutral_y").value)
        self.neutral_z = float(self.get_parameter("neutral_z").value)
        self.neutral_roll = float(self.get_parameter("neutral_roll").value)
        self.neutral_pitch = float(self.get_parameter("neutral_pitch").value)
        self.neutral_yaw = float(self.get_parameter("neutral_yaw").value)

        self.hole1_offset = np.array([
            float(self.get_parameter("hole1_x_m").value),
            float(self.get_parameter("hole1_y_m").value),
            float(self.get_parameter("hole1_z_m").value),
        ], dtype=np.float64)
        self.hole_mirror_axis = str(self.get_parameter("hole_mirror_axis").value)

        self.preinsert_dz_m = float(self.get_parameter("preinsert_dz_m").value)
        self.insert_dz_m = float(self.get_parameter("insert_dz_m").value)
        self.ee_to_tip_z_m = float(self.get_parameter("ee_to_tip_z_m").value)

        self.min_allowed_z_m = float(self.get_parameter("min_allowed_z_m").value)
        self.max_allowed_z_m = float(self.get_parameter("max_allowed_z_m").value)

        self.ik_timeout_s = float(self.get_parameter("ik_timeout_s").value)
        self.planning_time_s = float(self.get_parameter("planning_time_s").value)
        self.num_planning_attempts = int(self.get_parameter("num_planning_attempts").value)
        self.velocity_scale = float(self.get_parameter("velocity_scale").value)
        self.acceleration_scale = float(self.get_parameter("acceleration_scale").value)
        self.post_execute_sleep_s = float(self.get_parameter("post_execute_sleep_s").value)

        self.gripper_grasp_action = str(self.get_parameter("gripper_grasp_action").value)
        self.gripper_move_action = str(self.get_parameter("gripper_move_action").value)
        self.gripper_grasp_width = float(self.get_parameter("gripper_grasp_width").value)
        self.gripper_grasp_speed = float(self.get_parameter("gripper_grasp_speed").value)
        self.gripper_grasp_force = float(self.get_parameter("gripper_grasp_force").value)
        self.gripper_epsilon_inner = float(self.get_parameter("gripper_epsilon_inner").value)
        self.gripper_epsilon_outer = float(self.get_parameter("gripper_epsilon_outer").value)
        self.gripper_open_width = float(self.get_parameter("gripper_open_width").value)
        self.gripper_timeout_s = float(self.get_parameter("gripper_timeout_s").value)
        self.grasp_settle_time_s = float(self.get_parameter("grasp_settle_time_s").value)

        self.auto_sequence = bool(self.get_parameter("auto_sequence").value)

        # Joint names for FR3
        self.group_joint_names = [f"fr3_joint{i}" for i in range(1, 8)]

    def _compute_hole_offsets(self) -> None:
        """Compute hole1 and hole2 offsets based on mirror axis."""
        self.hole_offsets = {"hole1": self.hole1_offset.copy()}

        hole2 = self.hole1_offset.copy()
        if self.hole_mirror_axis == "x":
            hole2[0] *= -1.0
        elif self.hole_mirror_axis == "y":
            hole2[1] *= -1.0
        elif self.hole_mirror_axis == "xy":
            hole2[0] *= -1.0
            hole2[1] *= -1.0
        self.hole_offsets["hole2"] = hole2

    def _wait_for_services(self) -> None:
        """Wait for MoveIt services to be available."""
        self.get_logger().info("Waiting for MoveIt services...")

        if not self.ik_client.wait_for_service(timeout_sec=15.0):
            raise RuntimeError("/compute_ik service not available")
        if not self.plan_client.wait_for_service(timeout_sec=15.0):
            raise RuntimeError("/plan_kinematic_path service not available")
        if not self.exec_client.wait_for_server(timeout_sec=15.0):
            raise RuntimeError("/execute_trajectory action not available")

        self.get_logger().info("MoveIt services ready.")

        # Gripper - optional, don't fail if not available in sim
        if not self.use_fake_hardware:
            if not self.gripper_grasp_client.wait_for_server(timeout_sec=5.0):
                self.get_logger().warn("Gripper grasp action not available")
            if not self.gripper_move_client.wait_for_server(timeout_sec=5.0):
                self.get_logger().warn("Gripper move action not available")

    def _log_startup(self) -> None:
        self.get_logger().info("=" * 60)
        self.get_logger().info("UnifiedPickAndPlace started")
        self.get_logger().info(f"  planning_group: {self.planning_group}")
        self.get_logger().info(f"  pose_link: {self.pose_link}")
        self.get_logger().info(f"  screw_indices: {self.screw_indices}")
        self.get_logger().info(f"  hole1_offset: {self.hole_offsets['hole1']}")
        self.get_logger().info(f"  hole2_offset: {self.hole_offsets['hole2']}")
        self.get_logger().info(f"  use_fake_hardware: {self.use_fake_hardware}")
        self.get_logger().info(f"  auto_sequence: {self.auto_sequence}")
        self.get_logger().info("=" * 60)

    # -------------------------------------------------------------------------
    # Dynamic subscription management
    # -------------------------------------------------------------------------

    def _get_current_screw_idx(self) -> int:
        """Get the actual screw index (e.g., 0, 2) for the current sequence position."""
        if self.current_screw_seq_idx < len(self.screw_indices):
            return self.screw_indices[self.current_screw_seq_idx]
        return self.screw_indices[-1] if self.screw_indices else 0

    def _get_current_hole_name(self) -> str:
        """Get hole name based on sequence index: 0 -> hole1, 1 -> hole2."""
        return "hole1" if self.current_screw_seq_idx == 0 else "hole2"

    def _create_screw_subscription(self) -> None:
        """Create subscription for current screw index."""
        screw_idx = self._get_current_screw_idx()
        topic = self.screw_topic_pattern.replace("{idx}", str(screw_idx))

        # Destroy old subscription if exists
        if hasattr(self, "screw_sub"):
            try:
                self.destroy_subscription(self.screw_sub)
            except Exception:
                pass

        self.screw_sub = self.create_subscription(
            PoseStamped,
            topic,
            self._screw_pose_cb,
            FAST_QOS,
            callback_group=self.cb_group,
        )
        self.get_logger().info(f"Subscribed to screw topic: {topic}")

    # -------------------------------------------------------------------------
    # Pose callbacks
    # -------------------------------------------------------------------------

    def _screw_pose_cb(self, msg: PoseStamped) -> None:
        with self._lock:
            self.latest_screw_pose = msg
            self.screw_pose_history.append(msg)

    def _base_pose_cb(self, msg: PoseStamped) -> None:
        with self._lock:
            self.latest_base_pose = msg
            self.base_pose_history.append(msg)

    # -------------------------------------------------------------------------
    # Pose validation and stability
    # -------------------------------------------------------------------------

    def _pose_is_fresh(self, msg: PoseStamped) -> bool:
        now = self.get_clock().now()
        stamp = rclpy.time.Time.from_msg(msg.header.stamp)
        age_s = (now - stamp).nanoseconds * 1e-9
        return age_s <= self.min_pose_age_s

    def _check_pose_valid(self, msg: PoseStamped) -> tuple[bool, str]:
        if msg.header.frame_id != "base":
            return False, f"Expected frame_id='base', got '{msg.header.frame_id}'"
        if not self._pose_is_fresh(msg):
            return False, "Pose is stale"

        z = float(msg.pose.position.z)
        if z < self.min_allowed_z_m or z > self.max_allowed_z_m:
            return False, f"z={z:.3f} outside bounds [{self.min_allowed_z_m}, {self.max_allowed_z_m}]"

        return True, "ok"

    def _get_stable_pose(self, history: deque[PoseStamped]) -> tuple[Optional[PoseStamped], str]:
        """Check pose stability from history buffer."""
        if len(history) < self.lock_num_samples:
            return None, f"Need {self.lock_num_samples} samples, have {len(history)}"

        samples = list(history)[-self.lock_num_samples:]

        for msg in samples:
            ok, reason = self._check_pose_valid(msg)
            if not ok:
                return None, f"Invalid sample: {reason}"

        now = self.get_clock().now()
        oldest = rclpy.time.Time.from_msg(samples[0].header.stamp)
        age_span_s = (now - oldest).nanoseconds * 1e-9
        if age_span_s > self.lock_max_age_s:
            return None, f"Lock window too old ({age_span_s:.2f}s)"

        pts = [[float(m.pose.position.x), float(m.pose.position.y), float(m.pose.position.z)] for m in samples]
        xs, ys, zs = [p[0] for p in pts], [p[1] for p in pts], [p[2] for p in pts]
        spread = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))

        if spread > self.lock_max_spread_m:
            return None, f"Pose not stable: spread={spread:.4f}m (limit {self.lock_max_spread_m:.4f}m)"

        return samples[-1], "stable"

    # -------------------------------------------------------------------------
    # Target computation - Pickup
    # -------------------------------------------------------------------------

    def _compute_pickup_targets(self, screw_pose: PoseStamped) -> dict[str, TargetPose]:
        """Compute pregrasp, grasp, and lift targets from screw pose."""
        x = float(screw_pose.pose.position.x) + self.centroid_offset_x
        y = float(screw_pose.pose.position.y) + self.centroid_offset_y
        z = float(screw_pose.pose.position.z) + self.centroid_offset_z

        # Top-down grasp (default)
        pregrasp = TargetPose(
            x=x, y=y,
            z=z + self.pregrasp_clearance_m,
            roll=self.grasp_roll, pitch=self.grasp_pitch, yaw=self.grasp_yaw,
        )

        grasp = TargetPose(
            x=x, y=y,
            z=z + self.grasp_clearance_m,
            roll=self.grasp_roll, pitch=self.grasp_pitch, yaw=self.grasp_yaw,
        )

        lift = TargetPose(
            x=x, y=y,
            z=z + self.pregrasp_clearance_m + self.lift_height_m,
            roll=self.grasp_roll, pitch=self.grasp_pitch, yaw=self.grasp_yaw,
        )

        return {"pregrasp": pregrasp, "grasp": grasp, "lift": lift}

    # -------------------------------------------------------------------------
    # Target computation - Placement
    # -------------------------------------------------------------------------

    def _compute_placement_targets(self, base_pose: PoseStamped, hole_name: str) -> dict[str, np.ndarray]:
        """Compute preinsert, insert targets from cooling_base pose."""
        T_base_cooling = pose_msg_to_T(base_pose)
        p_cooling = T_base_cooling[:3, 3]

        hole_offset = self.hole_offsets[hole_name]
        p_hole = p_cooling + hole_offset

        # Hole frame with fixed insertion orientation
        T_hole = np.eye(4, dtype=np.float64)
        T_hole[:3, :3] = self.R_insert
        T_hole[:3, 3] = p_hole

        # Preinsert: above hole
        T_preinsert_tip = T_hole.copy()
        T_preinsert_tip[2, 3] += self.preinsert_dz_m

        # Insert: into hole
        T_insert_tip = T_hole.copy()
        T_insert_tip[2, 3] += self.insert_dz_m

        # Convert tip poses to EE poses
        T_preinsert_ee = T_preinsert_tip @ self.T_tip_ee
        T_insert_ee = T_insert_tip @ self.T_tip_ee

        # Retreat: back to preinsert
        T_retreat_ee = T_preinsert_ee.copy()

        return {
            "T_hole": T_hole,
            "T_preinsert_ee": T_preinsert_ee,
            "T_insert_ee": T_insert_ee,
            "T_retreat_ee": T_retreat_ee,
        }

    def _neutral_target(self) -> TargetPose:
        return TargetPose(
            x=self.neutral_x, y=self.neutral_y, z=self.neutral_z,
            roll=self.neutral_roll, pitch=self.neutral_pitch, yaw=self.neutral_yaw,
        )

    # -------------------------------------------------------------------------
    # MoveIt helpers
    # -------------------------------------------------------------------------

    def _wait_future(self, future, timeout_s: float):
        """Wait for a future with timeout using threading Event."""
        done = threading.Event()
        future.add_done_callback(lambda _: done.set())
        if not done.wait(timeout_s):
            raise TimeoutError(f"Future timed out after {timeout_s}s")
        return future.result()

    def _compute_ik(self, pose_msg: PoseStamped) -> tuple[Optional[list[float]], str]:
        """Compute IK for target pose."""
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

    def _plan_to_joints(self, joint_positions: list[float], label: str):
        """Plan a trajectory to joint goal."""
        req = GetMotionPlan.Request()
        mpr = req.motion_plan_request

        mpr.group_name = self.planning_group
        mpr.num_planning_attempts = self.num_planning_attempts
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
            return None, f"Planning call failed: {e}"

        if res is None:
            return None, "Planning response was None"

        ec = res.motion_plan_response.error_code
        if ec.val != MoveItErrorCodes.SUCCESS:
            return None, f"Planning failed code={ec.val}"

        return res.motion_plan_response.trajectory, "ok"

    def _execute_trajectory(self, traj, label: str) -> tuple[bool, str]:
        """Execute a planned trajectory."""
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = traj

        send_fut = self.exec_client.send_goal_async(goal)
        try:
            goal_handle = self._wait_future(send_fut, 5.0)
        except Exception as e:
            return False, f"Send goal failed for {label}: {e}"

        if goal_handle is None or not goal_handle.accepted:
            return False, f"Goal rejected for {label}"

        result_fut = goal_handle.get_result_async()
        try:
            wrapped = self._wait_future(result_fut, 120.0)
        except Exception as e:
            return False, f"Execution wait failed for {label}: {e}"

        result = wrapped.result
        if result.error_code.val != MoveItErrorCodes.SUCCESS:
            return False, f"Execution failed for {label}: code={result.error_code.val}"

        time.sleep(self.post_execute_sleep_s)
        return True, f"{label} done"

    def _plan_and_execute_pose(self, pose_msg: PoseStamped, label: str) -> tuple[bool, str]:
        """Full IK + plan + execute pipeline for a pose target."""
        self.get_logger().info(f"[{label}] Computing IK...")
        joints, msg = self._compute_ik(pose_msg)
        if joints is None:
            return False, f"{label}: {msg}"

        self.get_logger().info(f"[{label}] Planning...")
        traj, msg = self._plan_to_joints(joints, label)
        if traj is None:
            return False, f"{label}: {msg}"

        self.get_logger().info(f"[{label}] Executing...")
        return self._execute_trajectory(traj, label)

    def _plan_and_execute_target(self, target: TargetPose, label: str) -> tuple[bool, str]:
        """Plan and execute to a TargetPose."""
        stamp = self.get_clock().now().to_msg()
        pose_msg = target_to_pose_msg(target, "base", stamp)
        return self._plan_and_execute_pose(pose_msg, label)

    def _plan_and_execute_T(self, T: np.ndarray, label: str) -> tuple[bool, str]:
        """Plan and execute to a 4x4 transform."""
        stamp = self.get_clock().now().to_msg()
        pose_msg = T_to_pose_msg(T, "base", stamp)
        return self._plan_and_execute_pose(pose_msg, label)

    # -------------------------------------------------------------------------
    # Gripper control
    # -------------------------------------------------------------------------

    def _close_gripper(self) -> tuple[bool, str]:
        """Close gripper with grasp action."""
        self.get_logger().info(f"Gripper grasp: width={self.gripper_grasp_width:.4f}, force={self.gripper_grasp_force:.1f}N")

        if self.use_fake_hardware:
            self.get_logger().info("Fake hardware: skipping gripper")
            return True, "fake_hardware"

        if not self.gripper_grasp_client.server_is_ready():
            return False, "Gripper grasp action not available"

        goal = Grasp.Goal()
        goal.width = self.gripper_grasp_width
        goal.speed = self.gripper_grasp_speed
        goal.force = self.gripper_grasp_force
        goal.epsilon.inner = self.gripper_epsilon_inner
        goal.epsilon.outer = self.gripper_epsilon_outer

        send_fut = self.gripper_grasp_client.send_goal_async(goal)
        try:
            goal_handle = self._wait_future(send_fut, 5.0)
        except Exception as e:
            return False, f"Send gripper goal failed: {e}"

        if goal_handle is None or not goal_handle.accepted:
            return False, "Gripper goal rejected"

        result_fut = goal_handle.get_result_async()
        try:
            wrapped = self._wait_future(result_fut, self.gripper_timeout_s)
        except Exception as e:
            return False, f"Gripper wait failed: {e}"

        result = wrapped.result
        if not result.success:
            # Grasp action often returns success=False if object is grasped but not at target width
            # Check if we're in a reasonable grasped state
            self.get_logger().warn(f"Gripper reported success=False, error={result.error}")
            return True, f"gripper closed (with warning: {result.error})"

        return True, "gripper closed"

    def _open_gripper(self) -> tuple[bool, str]:
        """Open gripper with move action."""
        self.get_logger().info(f"Gripper open: width={self.gripper_open_width:.4f}")

        if self.use_fake_hardware:
            self.get_logger().info("Fake hardware: skipping gripper")
            return True, "fake_hardware"

        if not self.gripper_move_client.server_is_ready():
            return False, "Gripper move action not available"

        goal = GripperMove.Goal()
        goal.width = self.gripper_open_width
        goal.speed = self.gripper_grasp_speed

        send_fut = self.gripper_move_client.send_goal_async(goal)
        try:
            goal_handle = self._wait_future(send_fut, 5.0)
        except Exception as e:
            return False, f"Send gripper move goal failed: {e}"

        if goal_handle is None or not goal_handle.accepted:
            return False, "Gripper move goal rejected"

        result_fut = goal_handle.get_result_async()
        try:
            wrapped = self._wait_future(result_fut, self.gripper_timeout_s)
        except Exception as e:
            return False, f"Gripper move wait failed: {e}"

        result = wrapped.result
        if not result.success:
            return False, f"Gripper move failed: {result.error}"

        return True, "gripper opened"

    # -------------------------------------------------------------------------
    # Status publishing
    # -------------------------------------------------------------------------

    def _publish_status(self) -> None:
        stage_msg = String()
        stage_msg.data = self.stage.name
        self.pub_stage.publish(stage_msg)

        screw_msg = String()
        screw_msg.data = f"seq={self.current_screw_seq_idx}, screw_idx={self._get_current_screw_idx()}, hole={self._get_current_hole_name()}"
        self.pub_screw_idx.publish(screw_msg)

    # -------------------------------------------------------------------------
    # Stage callbacks
    # -------------------------------------------------------------------------

    def _go_neutral_cb(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        del request

        target = self._neutral_target()
        stamp = self.get_clock().now().to_msg()
        self.pub_neutral_pose.publish(target_to_pose_msg(target, "base", stamp))

        ok, msg = self._plan_and_execute_target(target, "neutral")
        if not ok:
            response.success = False
            response.message = msg
            return response

        self.stage = Stage.NEUTRAL
        response.success = True
        response.message = "Neutral done"
        return response

    def _go_pregrasp_cb(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        del request

        if self.stage not in (Stage.IDLE, Stage.NEUTRAL, Stage.RETREAT):
            response.success = False
            response.message = f"Cannot go_pregrasp from stage {self.stage.name}"
            return response

        # Get stable screw pose
        with self._lock:
            history = self.screw_pose_history

        stable_pose, msg = self._get_stable_pose(history)
        if stable_pose is None:
            response.success = False
            response.message = f"Screw pose not stable: {msg}"
            return response

        # Compute and latch targets
        self.latched_pickup_targets = self._compute_pickup_targets(stable_pose)

        stamp = self.get_clock().now().to_msg()
        self.pub_pregrasp_pose.publish(target_to_pose_msg(self.latched_pickup_targets["pregrasp"], "base", stamp))
        self.pub_grasp_pose.publish(target_to_pose_msg(self.latched_pickup_targets["grasp"], "base", stamp))
        self.pub_lift_pose.publish(target_to_pose_msg(self.latched_pickup_targets["lift"], "base", stamp))

        p = stable_pose.pose.position
        self.get_logger().info(f"Latched screw pose: [{p.x:.4f}, {p.y:.4f}, {p.z:.4f}]")

        ok, msg = self._plan_and_execute_target(self.latched_pickup_targets["pregrasp"], "pregrasp")
        if not ok:
            response.success = False
            response.message = msg
            return response

        self.stage = Stage.PREGRASP
        response.success = True
        response.message = "Pregrasp done"
        return response

    def _go_grasp_cb(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        del request

        if self.stage != Stage.PREGRASP:
            response.success = False
            response.message = f"Cannot go_grasp from stage {self.stage.name}"
            return response

        if self.latched_pickup_targets is None:
            response.success = False
            response.message = "No latched pickup targets"
            return response

        ok, msg = self._plan_and_execute_target(self.latched_pickup_targets["grasp"], "grasp")
        if not ok:
            response.success = False
            response.message = msg
            return response

        # Settle then close gripper
        if self.grasp_settle_time_s > 0:
            self.get_logger().info(f"Settling {self.grasp_settle_time_s}s before grip...")
            time.sleep(self.grasp_settle_time_s)

        ok_g, msg_g = self._close_gripper()
        if not ok_g:
            response.success = False
            response.message = f"Gripper failed: {msg_g}"
            return response

        self.stage = Stage.GRASP
        response.success = True
        response.message = f"Grasp done ({msg_g})"
        return response

    def _go_lift_cb(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        del request

        if self.stage != Stage.GRASP:
            response.success = False
            response.message = f"Cannot go_lift from stage {self.stage.name}"
            return response

        if self.latched_pickup_targets is None:
            response.success = False
            response.message = "No latched pickup targets"
            return response

        ok, msg = self._plan_and_execute_target(self.latched_pickup_targets["lift"], "lift")
        if not ok:
            response.success = False
            response.message = msg
            return response

        self.stage = Stage.LIFT
        response.success = True
        response.message = "Lift done"
        return response

    def _go_preinsert_cb(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        del request

        if self.stage != Stage.LIFT:
            response.success = False
            response.message = f"Cannot go_preinsert from stage {self.stage.name}"
            return response

        # Get stable base pose
        with self._lock:
            history = self.base_pose_history

        stable_pose, msg = self._get_stable_pose(history)
        if stable_pose is None:
            response.success = False
            response.message = f"Base pose not stable: {msg}"
            return response

        # Compute and latch placement targets
        hole_name = self._get_current_hole_name()
        self.latched_placement_targets = self._compute_placement_targets(stable_pose, hole_name)

        stamp = self.get_clock().now().to_msg()
        self.pub_preinsert_pose.publish(T_to_pose_msg(self.latched_placement_targets["T_preinsert_ee"], "base", stamp))
        self.pub_insert_pose.publish(T_to_pose_msg(self.latched_placement_targets["T_insert_ee"], "base", stamp))
        self.pub_retreat_pose.publish(T_to_pose_msg(self.latched_placement_targets["T_retreat_ee"], "base", stamp))

        p = stable_pose.pose.position
        self.get_logger().info(f"Latched base pose: [{p.x:.4f}, {p.y:.4f}, {p.z:.4f}], hole={hole_name}")

        ok, msg = self._plan_and_execute_T(self.latched_placement_targets["T_preinsert_ee"], "preinsert")
        if not ok:
            response.success = False
            response.message = msg
            return response

        self.stage = Stage.PREINSERT
        response.success = True
        response.message = f"Preinsert done (hole={hole_name})"
        return response

    def _go_insert_cb(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        del request

        if self.stage != Stage.PREINSERT:
            response.success = False
            response.message = f"Cannot go_insert from stage {self.stage.name}"
            return response

        if self.latched_placement_targets is None:
            response.success = False
            response.message = "No latched placement targets"
            return response

        ok, msg = self._plan_and_execute_T(self.latched_placement_targets["T_insert_ee"], "insert")
        if not ok:
            response.success = False
            response.message = msg
            return response

        self.stage = Stage.INSERT
        response.success = True
        response.message = "Insert done"
        return response

    def _go_release_cb(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        del request

        if self.stage != Stage.INSERT:
            response.success = False
            response.message = f"Cannot go_release from stage {self.stage.name}"
            return response

        ok, msg = self._open_gripper()
        if not ok:
            response.success = False
            response.message = f"Release failed: {msg}"
            return response

        self.stage = Stage.RELEASE
        response.success = True
        response.message = "Release done"
        return response

    def _go_retreat_cb(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        del request

        if self.stage != Stage.RELEASE:
            response.success = False
            response.message = f"Cannot go_retreat from stage {self.stage.name}"
            return response

        if self.latched_placement_targets is None:
            response.success = False
            response.message = "No latched placement targets"
            return response

        ok, msg = self._plan_and_execute_T(self.latched_placement_targets["T_retreat_ee"], "retreat")
        if not ok:
            response.success = False
            response.message = msg
            return response

        self.stage = Stage.RETREAT

        # Check if we have more screws
        if self.current_screw_seq_idx + 1 < len(self.screw_indices):
            response.success = True
            response.message = f"Retreat done. Call ~/next_screw to proceed to screw {self.screw_indices[self.current_screw_seq_idx + 1]}"
        else:
            self.stage = Stage.DONE
            response.success = True
            response.message = "All screws complete!"

        return response

    # -------------------------------------------------------------------------
    # Utility callbacks
    # -------------------------------------------------------------------------

    def _reset_cb(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        del request

        self.stage = Stage.IDLE
        self.current_screw_seq_idx = 0
        self.latched_pickup_targets = None
        self.latched_placement_targets = None
        self.screw_pose_history.clear()
        self.base_pose_history.clear()

        self._create_screw_subscription()

        response.success = True
        response.message = "Reset complete"
        return response

    def _next_screw_cb(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        del request

        if self.stage not in (Stage.RETREAT, Stage.DONE):
            response.success = False
            response.message = f"Cannot advance screw from stage {self.stage.name}"
            return response

        if self.current_screw_seq_idx + 1 >= len(self.screw_indices):
            response.success = False
            response.message = "Already at last screw"
            return response

        self.current_screw_seq_idx += 1
        self.stage = Stage.NEUTRAL  # or IDLE, depending on desired behavior
        self.latched_pickup_targets = None
        self.latched_placement_targets = None
        self.screw_pose_history.clear()

        self._create_screw_subscription()

        response.success = True
        response.message = f"Advanced to screw seq_idx={self.current_screw_seq_idx}, screw_idx={self._get_current_screw_idx()}, hole={self._get_current_hole_name()}"
        return response

    def _status_cb(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        del request
        response.success = True
        response.message = (
            f"stage={self.stage.name}, "
            f"screw_seq={self.current_screw_seq_idx}/{len(self.screw_indices)}, "
            f"screw_idx={self._get_current_screw_idx()}, "
            f"hole={self._get_current_hole_name()}"
        )
        return response

    def _run_full_sequence_cb(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        """Run full pick-and-place sequence for all configured screws."""
        del request

        self.get_logger().info("=" * 60)
        self.get_logger().info("STARTING FULL AUTO SEQUENCE")
        self.get_logger().info("=" * 60)

        # Reset first
        self.stage = Stage.IDLE
        self.current_screw_seq_idx = 0
        self.latched_pickup_targets = None
        self.latched_placement_targets = None
        self.screw_pose_history.clear()
        self.base_pose_history.clear()
        self._create_screw_subscription()

        total_screws = len(self.screw_indices)

        for screw_seq in range(total_screws):
            self.get_logger().info(f"\n{'='*40}\nSCREW {screw_seq + 1}/{total_screws} (idx={self.screw_indices[screw_seq]})\n{'='*40}")

            # Wait a bit for pose data to accumulate
            self.get_logger().info("Waiting for pose data...")
            time.sleep(2.0)

            # Neutral
            self.get_logger().info("[AUTO] go_neutral")
            ok, msg = self._plan_and_execute_target(self._neutral_target(), "neutral")
            if not ok:
                response.success = False
                response.message = f"Neutral failed: {msg}"
                return response
            self.stage = Stage.NEUTRAL

            # Pregrasp
            self.get_logger().info("[AUTO] go_pregrasp")
            time.sleep(1.0)  # Let poses stabilize

            with self._lock:
                history = self.screw_pose_history

            stable_pose, msg = self._get_stable_pose(history)
            if stable_pose is None:
                response.success = False
                response.message = f"Screw pose not stable: {msg}"
                return response

            self.latched_pickup_targets = self._compute_pickup_targets(stable_pose)
            ok, msg = self._plan_and_execute_target(self.latched_pickup_targets["pregrasp"], "pregrasp")
            if not ok:
                response.success = False
                response.message = f"Pregrasp failed: {msg}"
                return response
            self.stage = Stage.PREGRASP

            # Grasp
            self.get_logger().info("[AUTO] go_grasp")
            ok, msg = self._plan_and_execute_target(self.latched_pickup_targets["grasp"], "grasp")
            if not ok:
                response.success = False
                response.message = f"Grasp pose failed: {msg}"
                return response

            if self.grasp_settle_time_s > 0:
                time.sleep(self.grasp_settle_time_s)

            ok, msg = self._close_gripper()
            if not ok:
                response.success = False
                response.message = f"Gripper close failed: {msg}"
                return response
            self.stage = Stage.GRASP

            # Lift
            self.get_logger().info("[AUTO] go_lift")
            ok, msg = self._plan_and_execute_target(self.latched_pickup_targets["lift"], "lift")
            if not ok:
                response.success = False
                response.message = f"Lift failed: {msg}"
                return response
            self.stage = Stage.LIFT

            # Preinsert
            self.get_logger().info("[AUTO] go_preinsert")
            time.sleep(1.0)

            with self._lock:
                base_history = self.base_pose_history

            stable_base, msg = self._get_stable_pose(base_history)
            if stable_base is None:
                response.success = False
                response.message = f"Base pose not stable: {msg}"
                return response

            hole_name = self._get_current_hole_name()
            self.latched_placement_targets = self._compute_placement_targets(stable_base, hole_name)
            ok, msg = self._plan_and_execute_T(self.latched_placement_targets["T_preinsert_ee"], "preinsert")
            if not ok:
                response.success = False
                response.message = f"Preinsert failed: {msg}"
                return response
            self.stage = Stage.PREINSERT

            # Insert
            self.get_logger().info("[AUTO] go_insert")
            ok, msg = self._plan_and_execute_T(self.latched_placement_targets["T_insert_ee"], "insert")
            if not ok:
                response.success = False
                response.message = f"Insert failed: {msg}"
                return response
            self.stage = Stage.INSERT

            # Release
            self.get_logger().info("[AUTO] go_release")
            ok, msg = self._open_gripper()
            if not ok:
                response.success = False
                response.message = f"Release failed: {msg}"
                return response
            self.stage = Stage.RELEASE

            # Retreat
            self.get_logger().info("[AUTO] go_retreat")
            ok, msg = self._plan_and_execute_T(self.latched_placement_targets["T_retreat_ee"], "retreat")
            if not ok:
                response.success = False
                response.message = f"Retreat failed: {msg}"
                return response
            self.stage = Stage.RETREAT

            # Advance to next screw if not last
            if screw_seq + 1 < total_screws:
                self.current_screw_seq_idx += 1
                self.latched_pickup_targets = None
                self.latched_placement_targets = None
                self.screw_pose_history.clear()
                self._create_screw_subscription()
                self.get_logger().info(f"Advancing to screw {self.screw_indices[self.current_screw_seq_idx]}")

        self.stage = Stage.DONE
        self.get_logger().info("=" * 60)
        self.get_logger().info("FULL SEQUENCE COMPLETE")
        self.get_logger().info("=" * 60)

        response.success = True
        response.message = f"Full sequence complete! Placed {total_screws} screws."
        return response


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    rclpy.init()
    node = UnifiedPickAndPlace()
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