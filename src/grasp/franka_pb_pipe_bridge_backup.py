from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Optional

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_srvs.srv import Trigger
from std_msgs.msg import String

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


class FrankaPbPipeBridge(Node):
    """
    Bridge for first pb_pipe grasp tests.

    Features:
    - subscribes to one raw base-frame pose topic
    - assumes incoming pose is the CAD centroid in frame 'base'
    - computes side pregrasp + grasp from centroid
    - supports dry-run mode
    - publishes debug target poses
    - checks target stability before freezing
    - manual triggers:
        ~/go_base
        ~/go_pregrasp
        ~/go_grasp
        ~/go_lift
        ~/go_neutral
        ~/reset_sequence
    """

    def __init__(self) -> None:
        super().__init__("franka_pb_pipe_bridge")

        self.declare_parameter(
            "input_pose_topic",
            "/perception/fp/pose_base/zed2i_2/pb_pipe_0",
        )
        self.declare_parameter("service_name", "set_pose")
        self.declare_parameter("dry_run", True)

        # Pose freshness
        self.declare_parameter("min_pose_age_s", 1.0)

        # Stability / target lock
        self.declare_parameter("lock_num_samples", 1)
        self.declare_parameter("lock_max_spread_m", 0.01)
        self.declare_parameter("lock_max_age_s", 1.5)

        # Object geometry
        self.declare_parameter("object_name", "pb_pipe")
        self.declare_parameter("object_diameter_m", 0.046)

        # Approach direction in base frame
        self.declare_parameter("approach_axis", "x")   # "x" or "y"
        self.declare_parameter("approach_sign", -1.0)  # -1.0 or +1.0

        # Shift from CAD centroid to desired grasp center
        self.declare_parameter("centroid_offset_x", 0.0)
        self.declare_parameter("centroid_offset_y", 0.0)
        self.declare_parameter("centroid_offset_z", 0.0)

        # Surface clearances
        self.declare_parameter("pregrasp_clearance_m", 0.080)
        self.declare_parameter("grasp_clearance_m", 0.020)

        # Extra offset for TCP / gripper geometry
        self.declare_parameter("tcp_extra_offset_m", 0.0)

        # Lift
        self.declare_parameter("lift_delta_z_m", 0.08)

        # Fixed EE orientation
        self.declare_parameter("roll", math.pi)
        self.declare_parameter("pitch", 0.0)
        self.declare_parameter("yaw", math.pi / 2.0)

        # Neutral pose
        self.declare_parameter("neutral_x", 0.30)
        self.declare_parameter("neutral_y", 0.00)
        self.declare_parameter("neutral_z", 0.35)
        self.declare_parameter("neutral_roll", math.pi)
        self.declare_parameter("neutral_pitch", 0.0)
        self.declare_parameter("neutral_yaw", -math.pi/2)

        # Safety bounds
        self.declare_parameter("min_allowed_z_m", 0.02)
        self.declare_parameter("max_allowed_z_m", 1.20)

        # Service timing
        self.declare_parameter("wait_for_service_timeout_s", 2.0)
        self.declare_parameter("set_pose_timeout_s", 15.0)

        # ------------------------------------------------------------------
        # Read parameters
        # ------------------------------------------------------------------
        self.input_pose_topic = str(self.get_parameter("input_pose_topic").value)
        self.service_name = str(self.get_parameter("service_name").value)
        self.dry_run = bool(self.get_parameter("dry_run").value)

        self.min_pose_age_s = float(self.get_parameter("min_pose_age_s").value)

        self.lock_num_samples = int(self.get_parameter("lock_num_samples").value)
        self.lock_max_spread_m = float(self.get_parameter("lock_max_spread_m").value)
        self.lock_max_age_s = float(self.get_parameter("lock_max_age_s").value)

        self.object_name = str(self.get_parameter("object_name").value)
        self.object_diameter_m = float(self.get_parameter("object_diameter_m").value)

        self.approach_axis = str(self.get_parameter("approach_axis").value).lower()
        self.approach_sign = float(self.get_parameter("approach_sign").value)

        self.centroid_offset_x = float(self.get_parameter("centroid_offset_x").value)
        self.centroid_offset_y = float(self.get_parameter("centroid_offset_y").value)
        self.centroid_offset_z = float(self.get_parameter("centroid_offset_z").value)

        self.pregrasp_clearance_m = float(self.get_parameter("pregrasp_clearance_m").value)
        self.grasp_clearance_m = float(self.get_parameter("grasp_clearance_m").value)
        self.tcp_extra_offset_m = float(self.get_parameter("tcp_extra_offset_m").value)

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

        self.wait_for_service_timeout_s = float(
            self.get_parameter("wait_for_service_timeout_s").value
        )
        self.set_pose_timeout_s = float(
            self.get_parameter("set_pose_timeout_s").value
        )

        if self.approach_axis not in ("x", "y"):
            raise ValueError("approach_axis must be 'x' or 'y'")
        if self.approach_sign not in (-1.0, 1.0):
            raise ValueError("approach_sign must be -1.0 or +1.0")

        # ------------------------------------------------------------------
        # State
        # ------------------------------------------------------------------
        self.last_pose_msg: Optional[PoseStamped] = None
        self.pose_history: deque[PoseStamped] = deque(maxlen=max(self.lock_num_samples * 2, 10))

        self.frozen_pose_msg: Optional[PoseStamped] = None
        self.pregrasp_target: Optional[TargetPose] = None
        self.grasp_target: Optional[TargetPose] = None
        self.lift_target: Optional[TargetPose] = None

        # idle | pregrasp_sent | grasp_sent | lift_sent | neutral_sent
        self.stage = "idle"

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

        self.srv_base = self.create_service(Trigger, "~/go_base", self._go_neutral_cb)
        self.srv_pregrasp = self.create_service(Trigger, "~/go_pregrasp", self._go_pregrasp_cb)
        self.srv_grasp = self.create_service(Trigger, "~/go_grasp", self._go_grasp_cb)
        self.srv_lift = self.create_service(Trigger, "~/go_lift", self._go_lift_cb)
        self.srv_neutral = self.create_service(Trigger, "~/go_neutral", self._go_neutral_cb)
        self.srv_reset = self.create_service(Trigger, "~/reset_sequence", self._reset_sequence_cb)

        # Debug publishers
        self.pub_locked_pose = self.create_publisher(PoseStamped, "~/locked_pose", 10)
        self.pub_pregrasp_pose = self.create_publisher(PoseStamped, "~/pregrasp_pose", 10)
        self.pub_grasp_pose = self.create_publisher(PoseStamped, "~/grasp_pose", 10)
        self.pub_lift_pose = self.create_publisher(PoseStamped, "~/lift_pose", 10)
        self.pub_neutral_pose = self.create_publisher(PoseStamped, "~/neutral_pose", 10)
        self.pub_stage = self.create_publisher(String, "~/stage", 10)

        self.stage_timer = self.create_timer(0.5, self._publish_stage)

        self.get_logger().info(
            f"FrankaPbPipeBridge started | input={self.input_pose_topic} "
            f"service={self.service_name} dry_run={self.dry_run}"
        )
        self.get_logger().info(
            f"object={self.object_name} diameter={self.object_diameter_m:.4f} "
            f"approach_axis={self.approach_axis} approach_sign={self.approach_sign:+.0f}"
        )
        self.get_logger().info(
            f"neutral/base pose: x={self.neutral_x:.3f}, y={self.neutral_y:.3f}, z={self.neutral_z:.3f}"
        )
        self.get_logger().info(
            f"timeouts: wait_for_service={self.wait_for_service_timeout_s:.1f}s "
            f"set_pose={self.set_pose_timeout_s:.1f}s"
        )

    # ------------------------------------------------------------------
    # Basic helpers
    # ------------------------------------------------------------------

    def _publish_stage(self) -> None:
        msg = String()
        msg.data = self.stage
        self.pub_stage.publish(msg)

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

        # Keep this simple for debug visualization: quaternion left identity.
        # The controller still receives full RPY through the service.
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

        pts = []
        for msg in samples:
            pts.append([
                float(msg.pose.position.x),
                float(msg.pose.position.y),
                float(msg.pose.position.z),
            ])

        # No numpy dependency here
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        zs = [p[2] for p in pts]

        spread_x = max(xs) - min(xs)
        spread_y = max(ys) - min(ys)
        spread_z = max(zs) - min(zs)
        spread = max(spread_x, spread_y, spread_z)

        if spread > self.lock_max_spread_m:
            return None, (
                f"Pose not stable enough: spread={spread:.4f}m "
                f"(limit {self.lock_max_spread_m:.4f}m)"
            )

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
        x_c, y_c, z_c = self._compute_target_center(msg)

        radius = 0.5 * self.object_diameter_m
        pre_offset = radius + self.pregrasp_clearance_m + self.tcp_extra_offset_m
        grasp_offset = radius + self.grasp_clearance_m + self.tcp_extra_offset_m

        if self.approach_axis == "x":
            x_pre = x_c + self.approach_sign * pre_offset
            y_pre = y_c
            z_pre = z_c

            x_grasp = x_c + self.approach_sign * grasp_offset
            y_grasp = y_c
            z_grasp = z_c
        else:
            x_pre = x_c
            y_pre = y_c + self.approach_sign * pre_offset
            z_pre = z_c

            x_grasp = x_c
            y_grasp = y_c + self.approach_sign * grasp_offset
            z_grasp = z_c

        pregrasp = TargetPose(
            x=x_pre, y=y_pre, z=z_pre,
            roll=self.roll, pitch=self.pitch, yaw=self.yaw,
        )
        grasp = TargetPose(
            x=x_grasp, y=y_grasp, z=z_grasp,
            roll=self.roll, pitch=self.pitch, yaw=self.yaw,
        )
        return pregrasp, grasp

    def _compute_lift(self, grasp: TargetPose) -> TargetPose:
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
    # Sending / logging
    # ------------------------------------------------------------------

    def _send_pose_blocking(self, target: TargetPose) -> tuple[bool, str]:
        self.get_logger().info(
            "Target pose: "
            f"x={target.x:.3f}, y={target.y:.3f}, z={target.z:.3f}, "
            f"rpy=[{target.roll:.3f}, {target.pitch:.3f}, {target.yaw:.3f}]"
        )

        if self.dry_run:
            return True, "dry_run=True, not sent"

        if not self.pose_client.wait_for_service(timeout_sec=self.wait_for_service_timeout_s):
            return False, f"Service '{self.service_name}' not available"

        req = SetPose.Request()
        req.x = float(target.x)
        req.y = float(target.y)
        req.z = float(target.z)
        req.roll = float(target.roll)
        req.pitch = float(target.pitch)
        req.yaw = float(target.yaw)

        future = self.pose_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=self.set_pose_timeout_s)

        if not future.done():
            return False, "set_pose call timed out"

        try:
            resp = future.result()
        except Exception as e:
            return False, f"set_pose call failed: {e}"

        if resp is None:
            return False, "set_pose returned no response"
        if not bool(resp.success):
            return False, "set_pose success=False"

        return True, "set_pose success=True"

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

        self.pub_locked_pose.publish(self.frozen_pose_msg)
        self.pub_pregrasp_pose.publish(self._target_to_pose_stamped(self.pregrasp_target))
        self.pub_grasp_pose.publish(self._target_to_pose_stamped(self.grasp_target))
        self.pub_lift_pose.publish(self._target_to_pose_stamped(self.lift_target))

        self.get_logger().info("Froze stable raw base pose")
        self.get_logger().info(
            f"Frozen centroid: x={self.frozen_pose_msg.pose.position.x:.3f}, "
            f"y={self.frozen_pose_msg.pose.position.y:.3f}, "
            f"z={self.frozen_pose_msg.pose.position.z:.3f}"
        )
        self.get_logger().info(
            f"Pregrasp: x={self.pregrasp_target.x:.3f}, "
            f"y={self.pregrasp_target.y:.3f}, z={self.pregrasp_target.z:.3f}"
        )
        self.get_logger().info(
            f"Grasp:    x={self.grasp_target.x:.3f}, "
            f"y={self.grasp_target.y:.3f}, z={self.grasp_target.z:.3f}"
        )
        self.get_logger().info(
            f"Lift:     x={self.lift_target.x:.3f}, "
            f"y={self.lift_target.y:.3f}, z={self.lift_target.z:.3f}"
        )

        return True, "Pose frozen"

    def _reset_internal(self) -> None:
        self.frozen_pose_msg = None
        self.pregrasp_target = None
        self.grasp_target = None
        self.lift_target = None
        self.stage = "idle"

    # ------------------------------------------------------------------
    # Triggers
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

        assert self.pregrasp_target is not None
        ok, msg = self._send_pose_blocking(self.pregrasp_target)
        if not ok:
            response.success = False
            response.message = f"Pregrasp failed: {msg}"
            return response

        self.stage = "pregrasp_sent"
        response.success = True
        response.message = f"Pregrasp done ({msg})"
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
            response.message = f"Grasp failed: {msg}"
            return response

        self.stage = "grasp_sent"
        response.success = True
        response.message = f"Grasp done ({msg})"
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
        response.message = f"Lift done ({msg})"
        return response

    def _go_neutral_cb(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        del request

        neutral = self._neutral_target()
        self.pub_neutral_pose.publish(self._target_to_pose_stamped(neutral))

        ok, msg = self._send_pose_blocking(neutral)
        if not ok:
            response.success = False
            response.message = f"Neutral/base failed: {msg}"
            return response

        self.stage = "neutral_sent"
        response.success = True
        response.message = f"Neutral/base done ({msg})"
        return response

    def _reset_sequence_cb(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        del request
        self._reset_internal()
        response.success = True
        response.message = "Sequence reset"
        return response


def main() -> None:
    rclpy.init()
    node = FrankaPbPipeBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()