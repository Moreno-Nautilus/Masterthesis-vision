from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml
import rclpy
import rclpy.time
from geometry_msgs.msg import PoseStamped
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from std_srvs.srv import Trigger

from franka_msgs.action import Grasp
from messages_fr3.srv import SetPose


FAST_QOS = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
)


@dataclass
class TargetPose:
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float


class FrankaGraspBridge(Node):
    """
    Generic grasp bridge supporting multiple objects via YAML configs.

    Features:
    - Loads object-specific grasp parameters from YAML
    - Dynamic object switching via ~/load_object service
    - Top-down and side grasp approaches via ~/set_grasp_mode service
    - Stability checking before freezing target
    - Manual trigger services for each stage
    - Debug pose publishers
    """

    # Grasp modes
    GRASP_MODE_TOP_DOWN = "top_down"
    GRASP_MODE_SIDE_X = "side_x"      # approach along +/- X axis
    GRASP_MODE_SIDE_Y = "side_y"      # approach along +/- Y axis
    GRASP_MODE_SIDE_AUTO = "side_auto"  # auto-select best side approach

    def __init__(self) -> None:
        super().__init__("franka_grasp_bridge")

        # ------------------------------------------------------------------
        # Declare all parameters with defaults
        # ------------------------------------------------------------------
        self._declare_all_parameters()

        # ------------------------------------------------------------------
        # Load object config if provided (overrides defaults)
        # ------------------------------------------------------------------
        object_config_path = str(self.get_parameter("object_config").value)
        if object_config_path:
            self._load_object_config(object_config_path)

        # ------------------------------------------------------------------
        # Read parameters into instance variables
        # ------------------------------------------------------------------
        self._read_parameters()

        # ------------------------------------------------------------------
        # Grasp mode state
        # ------------------------------------------------------------------
        self.grasp_mode = self.GRASP_MODE_TOP_DOWN
        self.side_approach_sign = -1.0  # -1 = approach from positive side, +1 = from negative

        # ------------------------------------------------------------------
        # State
        # ------------------------------------------------------------------
        self.last_pose_msg: Optional[PoseStamped] = None
        self.pose_history: deque[PoseStamped] = deque(maxlen=max(self.lock_num_samples * 2, 10))

        self.frozen_pose_msg: Optional[PoseStamped] = None
        self.pregrasp_target: Optional[TargetPose] = None
        self.grasp_target: Optional[TargetPose] = None
        self.lift_target: Optional[TargetPose] = None

        self.stage = "idle"
        self.current_config_path: Optional[str] = object_config_path if object_config_path else None

        # ------------------------------------------------------------------
        # ROS interfaces
        # ------------------------------------------------------------------
        self.pose_sub = self.create_subscription(
            PoseStamped,
            self.input_pose_topic,
            self._pose_cb,
            FAST_QOS,
        )

        self.pose_client = self.create_client(SetPose, self.service_name)
        self.grasp_client = ActionClient(self, Grasp, self.gripper_action_name)

        # Trigger services
        self.srv_base = self.create_service(Trigger, "~/go_base", self._go_neutral_cb)
        self.srv_pregrasp = self.create_service(Trigger, "~/go_pregrasp", self._go_pregrasp_cb)
        self.srv_grasp = self.create_service(Trigger, "~/go_grasp", self._go_grasp_cb)
        self.srv_lift = self.create_service(Trigger, "~/go_lift", self._go_lift_cb)
        self.srv_neutral = self.create_service(Trigger, "~/go_neutral", self._go_neutral_cb)
        self.srv_reset = self.create_service(Trigger, "~/reset_sequence", self._reset_sequence_cb)

        # Dynamic switching services
        self.srv_load_object = self.create_service(
            Trigger, "~/load_object", self._load_object_cb
        )
        self.srv_set_grasp_mode = self.create_service(
            Trigger, "~/set_grasp_mode", self._set_grasp_mode_cb
        )
        self.srv_list_modes = self.create_service(
            Trigger, "~/list_grasp_modes", self._list_grasp_modes_cb
        )

        # Debug publishers
        self.pub_locked_pose = self.create_publisher(PoseStamped, "~/locked_pose", 10)
        self.pub_pregrasp_pose = self.create_publisher(PoseStamped, "~/pregrasp_pose", 10)
        self.pub_grasp_pose = self.create_publisher(PoseStamped, "~/grasp_pose", 10)
        self.pub_lift_pose = self.create_publisher(PoseStamped, "~/lift_pose", 10)
        self.pub_neutral_pose = self.create_publisher(PoseStamped, "~/neutral_pose", 10)
        self.pub_stage = self.create_publisher(String, "~/stage", 10)
        self.pub_grasp_mode = self.create_publisher(String, "~/grasp_mode", 10)

        self.stage_timer = self.create_timer(0.5, self._publish_status)

        self._log_startup()

    # ------------------------------------------------------------------
    # Parameter declaration and loading
    # ------------------------------------------------------------------

    def _declare_all_parameters(self) -> None:
        # Object config file (optional, overrides other params)
        self.declare_parameter("object_config", "")
        self.declare_parameter("config_dir", "")  # base directory for configs

        # Input/output
        self.declare_parameter("input_pose_topic", "/perception/fp/pose_base/zed2i_2/object_0")
        self.declare_parameter("service_name", "set_pose")
        self.declare_parameter("dry_run", True)

        # Pose freshness
        self.declare_parameter("min_pose_age_s", 5.0)

        # Stability / target lock
        self.declare_parameter("lock_num_samples", 3)
        self.declare_parameter("lock_max_spread_m", 0.01)
        self.declare_parameter("lock_max_age_s", 8.0)

        # Object geometry
        self.declare_parameter("object_name", "object")
        self.declare_parameter("object_diameter_m", 0.05)
        self.declare_parameter("object_height_m", 0.05)  # needed for side grasps

        # Approach direction
        self.declare_parameter("approach_axis", "z")
        self.declare_parameter("approach_sign", -1.0)

        # Shift from CAD centroid to desired grasp center
        self.declare_parameter("centroid_offset_x", 0.0)
        self.declare_parameter("centroid_offset_y", 0.0)
        self.declare_parameter("centroid_offset_z", 0.0)

        # Surface clearances
        self.declare_parameter("pregrasp_clearance_m", 0.10)
        self.declare_parameter("grasp_clearance_m", 0.025)

        # Side grasp specific
        self.declare_parameter("side_grasp_height_offset_m", 0.0)  # how high above table for side grasp
        self.declare_parameter("side_pregrasp_standoff_m", 0.10)   # how far back to start

        # Extra offset for TCP / gripper geometry
        self.declare_parameter("tcp_extra_offset_m", 0.0)

        # Lift
        self.declare_parameter("lift_delta_z_m", 0.08)

        # Fixed EE orientation (for top-down)
        self.declare_parameter("roll", math.pi)
        self.declare_parameter("pitch", 0.0)
        self.declare_parameter("yaw", math.pi / 2.0)

        # Neutral/base pose
        self.declare_parameter("neutral_x", 0.30)
        self.declare_parameter("neutral_y", 0.00)
        self.declare_parameter("neutral_z", 0.35)
        self.declare_parameter("neutral_roll", math.pi)
        self.declare_parameter("neutral_pitch", 0.0)
        self.declare_parameter("neutral_yaw", math.pi / 2.0)

        # Safety bounds
        self.declare_parameter("min_allowed_z_m", 0.02)
        self.declare_parameter("max_allowed_z_m", 1.20)

        # Robot base position (for reachability checks)
        self.declare_parameter("robot_base_x", 0.0)
        self.declare_parameter("robot_base_y", 0.0)

        # Service timing
        self.declare_parameter("wait_for_service_timeout_s", 2.0)
        self.declare_parameter("set_pose_timeout_s", 30.0)
        self.declare_parameter("poll_interval_s", 0.05)

        # Gripper action settings
        self.declare_parameter("gripper_action_name", "/fr3_gripper/grasp")
        self.declare_parameter("gripper_grasp_width", 0.001)
        self.declare_parameter("gripper_grasp_speed", 0.03)
        self.declare_parameter("gripper_grasp_force", 20.0)
        self.declare_parameter("gripper_epsilon_inner", 0.001)
        self.declare_parameter("gripper_epsilon_outer", 0.08)
        self.declare_parameter("gripper_timeout_s", 10.0)
        self.declare_parameter("grasp_settle_time_s", 1.0)

    def _load_object_config(self, config_path: str) -> None:
        """Load object-specific parameters from YAML, overriding declared defaults."""
        path = Path(config_path)
        if not path.exists():
            self.get_logger().error(f"Object config not found: {config_path}")
            return

        with open(path, "r") as f:
            config = yaml.safe_load(f)

        if not config:
            self.get_logger().warn(f"Empty config file: {config_path}")
            return

        self.get_logger().info(f"Loading object config: {config_path}")

        # Map of YAML keys that can override parameters
        overridable = {
            "object_name", "object_diameter_m", "object_height_m",
            "input_pose_topic",
            "gripper_grasp_width", "gripper_grasp_force",
            "gripper_grasp_speed", "gripper_epsilon_inner", "gripper_epsilon_outer",
            "centroid_offset_x", "centroid_offset_y", "centroid_offset_z",
            "pregrasp_clearance_m", "grasp_clearance_m", "tcp_extra_offset_m",
            "side_grasp_height_offset_m", "side_pregrasp_standoff_m",
            "lift_delta_z_m",
            "roll", "pitch", "yaw",
            "approach_axis", "approach_sign",
        }

        for key, value in config.items():
            if key in overridable:
                param = self.get_parameter(key)
                self.set_parameters([Parameter(key, type_=param.type_, value=value)])
                self.get_logger().info(f"  {key} = {value}")

    def _read_parameters(self) -> None:
        """Read all parameters into instance variables."""
        self.config_dir = str(self.get_parameter("config_dir").value)
        self.input_pose_topic = str(self.get_parameter("input_pose_topic").value)
        self.service_name = str(self.get_parameter("service_name").value)
        self.dry_run = bool(self.get_parameter("dry_run").value)

        self.min_pose_age_s = float(self.get_parameter("min_pose_age_s").value)

        self.lock_num_samples = int(self.get_parameter("lock_num_samples").value)
        self.lock_max_spread_m = float(self.get_parameter("lock_max_spread_m").value)
        self.lock_max_age_s = float(self.get_parameter("lock_max_age_s").value)

        self.object_name = str(self.get_parameter("object_name").value)
        self.object_diameter_m = float(self.get_parameter("object_diameter_m").value)
        self.object_height_m = float(self.get_parameter("object_height_m").value)

        self.approach_axis = str(self.get_parameter("approach_axis").value).lower()
        self.approach_sign = float(self.get_parameter("approach_sign").value)

        self.centroid_offset_x = float(self.get_parameter("centroid_offset_x").value)
        self.centroid_offset_y = float(self.get_parameter("centroid_offset_y").value)
        self.centroid_offset_z = float(self.get_parameter("centroid_offset_z").value)

        self.pregrasp_clearance_m = float(self.get_parameter("pregrasp_clearance_m").value)
        self.grasp_clearance_m = float(self.get_parameter("grasp_clearance_m").value)
        self.tcp_extra_offset_m = float(self.get_parameter("tcp_extra_offset_m").value)

        self.side_grasp_height_offset_m = float(self.get_parameter("side_grasp_height_offset_m").value)
        self.side_pregrasp_standoff_m = float(self.get_parameter("side_pregrasp_standoff_m").value)

        self.lift_delta_z_m = float(self.get_parameter("lift_delta_z_m").value)

        self.roll = float(self.get_parameter("roll").value)
        self.pitch = float(self.get_parameter("pitch").value)
        self.yaw = float(self.get_parameter("yaw").value)

        self.neutral_x = float(self.get_parameter("neutral_x").value)
        self.neutral_y = float(self.get_parameter("neutral_y").value)
        self.neutral_z = float(self.get_parameter("neutral_z").value)
        self.neutral_roll = float(self.get_parameter("neutral_roll").value)
        self.neutral_pitch = float(self.get_parameter("neutral_pitch").value)
        self.neutral_yaw = float(self.get_parameter("neutral_yaw").value)

        self.min_allowed_z_m = float(self.get_parameter("min_allowed_z_m").value)
        self.max_allowed_z_m = float(self.get_parameter("max_allowed_z_m").value)

        self.robot_base_x = float(self.get_parameter("robot_base_x").value)
        self.robot_base_y = float(self.get_parameter("robot_base_y").value)

        self.wait_for_service_timeout_s = float(self.get_parameter("wait_for_service_timeout_s").value)
        self.set_pose_timeout_s = float(self.get_parameter("set_pose_timeout_s").value)
        self.poll_interval_s = float(self.get_parameter("poll_interval_s").value)

        self.gripper_action_name = str(self.get_parameter("gripper_action_name").value)
        self.gripper_grasp_width = float(self.get_parameter("gripper_grasp_width").value)
        self.gripper_grasp_speed = float(self.get_parameter("gripper_grasp_speed").value)
        self.gripper_grasp_force = float(self.get_parameter("gripper_grasp_force").value)
        self.gripper_epsilon_inner = float(self.get_parameter("gripper_epsilon_inner").value)
        self.gripper_epsilon_outer = float(self.get_parameter("gripper_epsilon_outer").value)
        self.gripper_timeout_s = float(self.get_parameter("gripper_timeout_s").value)
        self.grasp_settle_time_s = float(self.get_parameter("grasp_settle_time_s").value)

    def _log_startup(self) -> None:
        self.get_logger().info(
            f"FrankaGraspBridge started | object={self.object_name} "
            f"input={self.input_pose_topic} dry_run={self.dry_run}"
        )
        self.get_logger().info(
            f"  diameter={self.object_diameter_m:.4f}m "
            f"height={self.object_height_m:.4f}m"
        )
        self.get_logger().info(
            f"  grasp_mode={self.grasp_mode}"
        )
        self.get_logger().info(
            f"  neutral: [{self.neutral_x:.3f}, {self.neutral_y:.3f}, {self.neutral_z:.3f}]"
        )

    # ------------------------------------------------------------------
    # Dynamic object switching
    # ------------------------------------------------------------------

    def _switch_object(self, object_name: str) -> tuple[bool, str]:
        """Switch to a different object config."""
        # Determine config path
        if self.config_dir:
            config_path = Path(self.config_dir) / f"{object_name}.yaml"
        else:
            # Try to infer from current config path
            if self.current_config_path:
                config_path = Path(self.current_config_path).parent / f"{object_name}.yaml"
            else:
                return False, "No config_dir set and no current config to infer from"

        if not config_path.exists():
            return False, f"Config not found: {config_path}"

        # Destroy old subscription
        self.destroy_subscription(self.pose_sub)

        # Reset state
        self._reset_internal()

        # Load new config
        self._load_object_config(str(config_path))
        self._read_parameters()
        self.current_config_path = str(config_path)

        # Create new subscription
        self.pose_sub = self.create_subscription(
            PoseStamped,
            self.input_pose_topic,
            self._pose_cb,
            FAST_QOS,
        )

        self.get_logger().info(f"Switched to object: {self.object_name}")
        return True, f"Loaded {object_name} from {config_path}"

    def _load_object_cb(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        """
        Service to switch objects dynamically.
        Call with: ros2 service call ~/load_object std_srvs/srv/Trigger
        Then provide object name via parameter or extend to custom service.
        
        For now, uses a workaround: set object_name param before calling.
        """
        del request
        
        # Get object name from parameter (set it before calling this service)
        # e.g.: ros2 param set /franka_grasp_bridge object_name pb_screw
        #       ros2 service call ~/load_object std_srvs/srv/Trigger
        try:
            new_object = str(self.get_parameter("object_name").value)
        except Exception as e:
            response.success = False
            response.message = f"Failed to get object_name param: {e}"
            return response

        ok, msg = self._switch_object(new_object)
        response.success = ok
        response.message = msg
        return response

    # ------------------------------------------------------------------
    # Grasp mode switching
    # ------------------------------------------------------------------

    def _set_grasp_mode_cb(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        """
        Cycle through grasp modes or set via parameter.
        Set grasp_mode param before calling:
          ros2 param set /franka_grasp_bridge grasp_mode top_down
          ros2 service call ~/set_grasp_mode std_srvs/srv/Trigger
        """
        del request

        # Try to read from a temporary parameter
        try:
            self.declare_parameter("grasp_mode", self.grasp_mode)
        except:
            pass  # Already declared

        mode_str = str(self.get_parameter("grasp_mode").value)

        valid_modes = [
            self.GRASP_MODE_TOP_DOWN,
            self.GRASP_MODE_SIDE_X,
            self.GRASP_MODE_SIDE_Y,
            self.GRASP_MODE_SIDE_AUTO,
        ]

        if mode_str not in valid_modes:
            response.success = False
            response.message = f"Invalid mode '{mode_str}'. Valid: {valid_modes}"
            return response

        self.grasp_mode = mode_str
        self._reset_internal()

        response.success = True
        response.message = f"Grasp mode set to: {self.grasp_mode}"
        self.get_logger().info(response.message)
        return response

    def _list_grasp_modes_cb(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        del request
        response.success = True
        response.message = (
            f"Available modes: top_down, side_x, side_y, side_auto. "
            f"Current: {self.grasp_mode}"
        )
        return response

    # ------------------------------------------------------------------
    # Basic helpers
    # ------------------------------------------------------------------

    def _publish_status(self) -> None:
        msg = String()
        msg.data = self.stage
        self.pub_stage.publish(msg)

        mode_msg = String()
        mode_msg.data = self.grasp_mode
        self.pub_grasp_mode.publish(mode_msg)

    def _pose_cb(self, msg: PoseStamped) -> None:
        self.last_pose_msg = msg
        self.pose_history.append(msg)

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
            return False, f"z={z:.3f} outside allowed range"

        return True, "ok"

    def _target_to_pose_stamped(self, target: TargetPose, stamp=None) -> PoseStamped:
        msg = PoseStamped()
        msg.header.frame_id = "base"
        msg.header.stamp = self.get_clock().now().to_msg() if stamp is None else stamp
        msg.pose.position.x = float(target.x)
        msg.pose.position.y = float(target.y)
        msg.pose.position.z = float(target.z)
        msg.pose.orientation.w = 1.0
        return msg

    # ------------------------------------------------------------------
    # Target lock / stability
    # ------------------------------------------------------------------

    def _get_stable_pose(self) -> tuple[Optional[PoseStamped], str]:
        if len(self.pose_history) < self.lock_num_samples:
            return None, f"Need {self.lock_num_samples} samples, have {len(self.pose_history)}"

        samples = list(self.pose_history)[-self.lock_num_samples:]

        for msg in samples:
            ok, reason = self._check_pose_valid(msg)
            if not ok:
                return None, f"Unstable sample: {reason}"

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

    # ------------------------------------------------------------------
    # Target computation
    # ------------------------------------------------------------------

    def _compute_target_center(self, msg: PoseStamped) -> tuple[float, float, float]:
        x = float(msg.pose.position.x) + self.centroid_offset_x
        y = float(msg.pose.position.y) + self.centroid_offset_y
        z = float(msg.pose.position.z) + self.centroid_offset_z
        return x, y, z

    def _compute_pregrasp_and_grasp(self, msg: PoseStamped) -> tuple[TargetPose, TargetPose]:
        """Compute pregrasp and grasp poses based on current grasp mode."""
        x_c, y_c, z_c = self._compute_target_center(msg)

        if self.grasp_mode == self.GRASP_MODE_TOP_DOWN:
            return self._compute_top_down_grasp(x_c, y_c, z_c)
        elif self.grasp_mode == self.GRASP_MODE_SIDE_X:
            return self._compute_side_grasp(x_c, y_c, z_c, axis="x")
        elif self.grasp_mode == self.GRASP_MODE_SIDE_Y:
            return self._compute_side_grasp(x_c, y_c, z_c, axis="y")
        elif self.grasp_mode == self.GRASP_MODE_SIDE_AUTO:
            return self._compute_side_grasp_auto(x_c, y_c, z_c)
        else:
            # Default to top-down
            return self._compute_top_down_grasp(x_c, y_c, z_c)

    def _compute_top_down_grasp(
        self, x_c: float, y_c: float, z_c: float
    ) -> tuple[TargetPose, TargetPose]:
        """Top-down grasp: approach from above along -Z."""
        pregrasp = TargetPose(
            x=x_c,
            y=y_c,
            z=z_c + self.pregrasp_clearance_m + self.tcp_extra_offset_m,
            roll=self.roll,
            pitch=self.pitch,
            yaw=self.yaw,
        )

        grasp = TargetPose(
            x=x_c,
            y=y_c,
            z=z_c + self.grasp_clearance_m + self.tcp_extra_offset_m,
            roll=self.roll,
            pitch=self.pitch,
            yaw=self.yaw,
        )

        return pregrasp, grasp

    def _compute_side_grasp(
        self, x_c: float, y_c: float, z_c: float, axis: str
    ) -> tuple[TargetPose, TargetPose]:
        """
        Side grasp: approach horizontally along X or Y axis.
        
        Gripper orientation for side grasp:
        - roll=0: gripper "upright"
        - pitch=0: no tilt
        - yaw: determines approach direction
          - yaw=0: gripper points along +X (approach from -X)
          - yaw=π/2: gripper points along +Y (approach from -Y)
          - yaw=π: gripper points along -X (approach from +X)
          - yaw=-π/2: gripper points along -Y (approach from +Y)
        """
        # Grasp height: object center + optional offset
        grasp_z = z_c + self.side_grasp_height_offset_m

        # Determine approach direction based on object position relative to robot
        # Simple heuristic: approach from the side closer to robot base
        obj_rel_x = x_c - self.robot_base_x
        obj_rel_y = y_c - self.robot_base_y

        if axis == "x":
            # Approach along X axis
            if obj_rel_x > 0:
                # Object is in +X, approach from -X direction (robot side)
                approach_sign = -1.0
                yaw = 0.0  # gripper points +X
            else:
                # Object is in -X, approach from +X direction
                approach_sign = 1.0
                yaw = math.pi  # gripper points -X

            pregrasp = TargetPose(
                x=x_c + approach_sign * self.side_pregrasp_standoff_m,
                y=y_c,
                z=grasp_z,
                roll=0.0,
                pitch=0.0,
                yaw=yaw,
            )
            grasp = TargetPose(
                x=x_c + approach_sign * self.grasp_clearance_m,
                y=y_c,
                z=grasp_z,
                roll=0.0,
                pitch=0.0,
                yaw=yaw,
            )

        else:  # axis == "y"
            # Approach along Y axis
            if obj_rel_y > 0:
                # Object is in +Y, approach from -Y direction
                approach_sign = -1.0
                yaw = math.pi / 2.0  # gripper points +Y
            else:
                # Object is in -Y, approach from +Y direction
                approach_sign = 1.0
                yaw = -math.pi / 2.0  # gripper points -Y

            pregrasp = TargetPose(
                x=x_c,
                y=y_c + approach_sign * self.side_pregrasp_standoff_m,
                z=grasp_z,
                roll=0.0,
                pitch=0.0,
                yaw=yaw,
            )
            grasp = TargetPose(
                x=x_c,
                y=y_c + approach_sign * self.grasp_clearance_m,
                z=grasp_z,
                roll=0.0,
                pitch=0.0,
                yaw=yaw,
            )

        return pregrasp, grasp

    def _compute_side_grasp_auto(
        self, x_c: float, y_c: float, z_c: float
    ) -> tuple[TargetPose, TargetPose]:
        """
        Automatically select best side approach based on:
        1. Object position relative to robot (prefer approaching from robot side)
        2. Simple reachability heuristic (distance from base)
        
        Future: Could add collision checking, IK feasibility, etc.
        """
        obj_rel_x = x_c - self.robot_base_x
        obj_rel_y = y_c - self.robot_base_y

        # Score each approach direction (higher = better)
        # Prefer approaching from the robot's side (shorter reach)
        scores = {
            "x": abs(obj_rel_x),  # how far in X = how good X approach is
            "y": abs(obj_rel_y),  # how far in Y = how good Y approach is
        }

        # Also consider: don't approach from a direction that would put pregrasp
        # too close to robot base (collision) or too far (unreachable)
        for axis in ["x", "y"]:
            pregrasp, _ = self._compute_side_grasp(x_c, y_c, z_c, axis)
            dist_to_base = math.sqrt(
                (pregrasp.x - self.robot_base_x) ** 2 +
                (pregrasp.y - self.robot_base_y) ** 2
            )
            # Penalize if too close (<0.2m) or too far (>0.7m)
            if dist_to_base < 0.2:
                scores[axis] -= 1.0
            elif dist_to_base > 0.7:
                scores[axis] -= 0.5

        best_axis = max(scores, key=scores.get)
        self.get_logger().info(
            f"Auto side grasp: scores={scores}, selected={best_axis}"
        )

        return self._compute_side_grasp(x_c, y_c, z_c, best_axis)

    def _compute_lift(self, grasp: TargetPose) -> TargetPose:
        """Lift is always straight up regardless of grasp mode."""
        return TargetPose(
            x=grasp.x,
            y=grasp.y,
            z=grasp.z + self.lift_delta_z_m,
            roll=grasp.roll,
            pitch=grasp.pitch,
            yaw=grasp.yaw,
        )

    def _neutral_target(self) -> TargetPose:
        return TargetPose(
            x=self.neutral_x,
            y=self.neutral_y,
            z=self.neutral_z,
            roll=self.neutral_roll,
            pitch=self.neutral_pitch,
            yaw=self.neutral_yaw,
        )

    # ------------------------------------------------------------------
    # Blocking calls with polling (fixes the 15s timeout issue)
    # ------------------------------------------------------------------

    def _spin_until_done_or_timeout(self, future, timeout_s: float) -> bool:
        """
        Poll future completion with short spins.
        Returns True if future completed, False if timed out.
        """
        start = time.monotonic()
        while not future.done():
            elapsed = time.monotonic() - start
            if elapsed >= timeout_s:
                return False
            rclpy.spin_once(self, timeout_sec=self.poll_interval_s)
        return True

    def _send_pose_blocking(self, target: TargetPose) -> tuple[bool, str]:
        self.get_logger().info(
            f"Sending pose: [{target.x:.3f}, {target.y:.3f}, {target.z:.3f}] "
            f"rpy=[{target.roll:.3f}, {target.pitch:.3f}, {target.yaw:.3f}]"
        )

        if self.dry_run:
            return True, "dry_run=True, not sent"

        if not self.pose_client.wait_for_service(timeout_sec=self.wait_for_service_timeout_s):
            return False, f"Service '{self.service_name}' not available"

        req = SetPose.Request()
        req.x, req.y, req.z = float(target.x), float(target.y), float(target.z)
        req.roll, req.pitch, req.yaw = float(target.roll), float(target.pitch), float(target.yaw)

        future = self.pose_client.call_async(req)

        if not self._spin_until_done_or_timeout(future, self.set_pose_timeout_s):
            self.get_logger().warn("set_pose call timed out; assuming command was sent")
            return True, "set_pose ack timed out; assuming sent"

        try:
            resp = future.result()
        except Exception as e:
            return False, f"set_pose exception: {e}"

        if resp is None:
            return False, "set_pose returned None"
        if not bool(resp.success):
            return False, "set_pose success=False"

        return True, "ok"

    def _close_gripper_blocking(self) -> tuple[bool, str]:
        self.get_logger().info(
            f"Gripper grasp: width={self.gripper_grasp_width:.4f} "
            f"force={self.gripper_grasp_force:.1f}N"
        )

        if self.dry_run:
            return True, "dry_run=True, gripper not sent"

        if not self.grasp_client.wait_for_server(timeout_sec=2.0):
            return False, f"Gripper action '{self.gripper_action_name}' not available"

        goal = Grasp.Goal()
        goal.width = float(self.gripper_grasp_width)
        goal.speed = float(self.gripper_grasp_speed)
        goal.force = float(self.gripper_grasp_force)
        goal.epsilon.inner = float(self.gripper_epsilon_inner)
        goal.epsilon.outer = float(self.gripper_epsilon_outer)

        send_future = self.grasp_client.send_goal_async(goal)
        if not self._spin_until_done_or_timeout(send_future, 2.0):
            return False, "Sending gripper goal timed out"

        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            return False, "Gripper goal rejected"

        result_future = goal_handle.get_result_async()
        if not self._spin_until_done_or_timeout(result_future, self.gripper_timeout_s):
            return False, "Gripper action timed out"

        result_wrap = result_future.result()
        if result_wrap is None:
            return False, "Gripper returned no result"

        result = result_wrap.result
        if not bool(result.success):
            return False, f"Gripper failed: {result.error}"

        return True, f"ok, width={result.current_width:.4f}"

    # ------------------------------------------------------------------
    # Freeze / reset
    # ------------------------------------------------------------------

    def _freeze_current_pose_and_targets(self) -> tuple[bool, str]:
        stable_pose, msg = self._get_stable_pose()
        if stable_pose is None:
            return False, msg

        self.frozen_pose_msg = stable_pose
        self.pregrasp_target, self.grasp_target = self._compute_pregrasp_and_grasp(self.frozen_pose_msg)
        self.lift_target = self._compute_lift(self.grasp_target)

        # Publish debug poses
        self.pub_locked_pose.publish(self.frozen_pose_msg)
        self.pub_pregrasp_pose.publish(self._target_to_pose_stamped(self.pregrasp_target))
        self.pub_grasp_pose.publish(self._target_to_pose_stamped(self.grasp_target))
        self.pub_lift_pose.publish(self._target_to_pose_stamped(self.lift_target))

        p = self.frozen_pose_msg.pose.position
        self.get_logger().info(f"Frozen centroid: [{p.x:.3f}, {p.y:.3f}, {p.z:.3f}]")
        self.get_logger().info(f"Grasp mode: {self.grasp_mode}")
        self.get_logger().info(
            f"Targets -> pregrasp=[{self.pregrasp_target.x:.3f}, {self.pregrasp_target.y:.3f}, {self.pregrasp_target.z:.3f}] "
            f"grasp=[{self.grasp_target.x:.3f}, {self.grasp_target.y:.3f}, {self.grasp_target.z:.3f}]"
        )

        return True, "Pose frozen"

    def _reset_internal(self) -> None:
        self.frozen_pose_msg = None
        self.pregrasp_target = None
        self.grasp_target = None
        self.lift_target = None
        self.stage = "idle"
        self.pose_history.clear()

    # ------------------------------------------------------------------
    # Trigger callbacks
    # ------------------------------------------------------------------

    def _go_pregrasp_cb(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        del request

        if self.stage not in ("idle", "neutral_sent"):
            response.success = False
            response.message = f"Cannot go_pregrasp in stage '{self.stage}'"
            return response

        ok, msg = self._freeze_current_pose_and_targets()
        if not ok:
            response.success = False
            response.message = msg
            return response

        ok, msg = self._send_pose_blocking(self.pregrasp_target)
        if not ok:
            response.success = False
            response.message = f"Pregrasp failed: {msg}"
            return response

        self.stage = "pregrasp_sent"
        response.success = True
        response.message = "Pregrasp done"
        return response

    def _go_grasp_cb(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        del request

        if self.stage != "pregrasp_sent":
            response.success = False
            response.message = f"Cannot go_grasp in stage '{self.stage}'"
            return response

        if self.grasp_target is None:
            response.success = False
            response.message = "Missing grasp target"
            return response

        ok, msg = self._send_pose_blocking(self.grasp_target)
        if not ok:
            response.success = False
            response.message = f"Grasp pose failed: {msg}"
            return response

        if self.grasp_settle_time_s > 0.0:
            self.get_logger().info(f"Settling {self.grasp_settle_time_s:.2f}s before grip")
            time.sleep(self.grasp_settle_time_s)

        ok_g, msg_g = self._close_gripper_blocking()
        if not ok_g:
            response.success = False
            response.message = f"Gripper failed: {msg_g}"
            return response

        self.stage = "grasp_sent"
        response.success = True
        response.message = f"Grasp done ({msg_g})"
        return response

    def _go_lift_cb(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        del request

        if self.stage != "grasp_sent":
            response.success = False
            response.message = f"Cannot go_lift in stage '{self.stage}'"
            return response

        if self.lift_target is None:
            response.success = False
            response.message = "Missing lift target"
            return response

        ok, msg = self._send_pose_blocking(self.lift_target)
        if not ok:
            response.success = False
            response.message = f"Lift failed: {msg}"
            return response

        self.stage = "lift_sent"
        response.success = True
        response.message = "Lift done"
        return response

    def _go_neutral_cb(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        del request

        neutral = self._neutral_target()
        self.pub_neutral_pose.publish(self._target_to_pose_stamped(neutral))

        ok, msg = self._send_pose_blocking(neutral)
        if not ok:
            response.success = False
            response.message = f"Neutral failed: {msg}"
            return response

        self.stage = "neutral_sent"
        response.success = True
        response.message = "Neutral done"
        return response

    def _reset_sequence_cb(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        del request
        self._reset_internal()
        response.success = True
        response.message = "Sequence reset"
        return response


def main() -> None:
    rclpy.init()
    node = FrankaGraspBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()