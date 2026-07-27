#!/usr/bin/env python3

import math
import re
from dataclasses import dataclass

import numpy as np
import rclpy
from fp_debug_msgs.msg import DebugPoseItem
from geometry_msgs.msg import Point, Pose, PoseStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray


POSE_PREFIXES = (
    "/perception/fp/pose_base/",
    "/perception/fp/pose_base_init/",
    "/perception/fp/pose_base_track/",
)


@dataclass
class StoredPose:
    topic: str
    pose: Pose
    stamp_sec: float
    label: str = ""


def quat_xyzw_to_R(q):
    x, y, z, w = q
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)

    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s

    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=float,
    )


def R_to_rpy_deg(R):
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)

    if sy > 1e-6:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0.0

    return np.degrees([roll, pitch, yaw])


def short_name(topic: str) -> str:
    name = topic
    for p in POSE_PREFIXES:
        name = name.replace(p, "")
    return name


def safe_ns(topic: str) -> str:
    return re.sub(r"[^A-Za-z0-9_/]", "_", topic).strip("/")


def point_from_np(v):
    p = Point()
    p.x = float(v[0])
    p.y = float(v[1])
    p.z = float(v[2])
    return p


class BasePoseAxesDebug(Node):
    def __init__(self):
        super().__init__("base_pose_axes_debug")

        self.declare_parameter("base_frame", "base")
        self.declare_parameter("axis_length", 0.08)
        self.declare_parameter("publish_hz", 10.0)
        self.declare_parameter("max_age_sec", 2.0)
        self.declare_parameter("show_text", True)
        self.declare_parameter("only_track_topics", False)

        self.base_frame = str(self.get_parameter("base_frame").value)
        self.axis_length = float(self.get_parameter("axis_length").value)
        self.max_age_sec = float(self.get_parameter("max_age_sec").value)
        self.show_text = bool(self.get_parameter("show_text").value)
        self.only_track_topics = bool(self.get_parameter("only_track_topics").value)

        # Important:
        # Your perception pose publishers use BEST_EFFORT.
        # Default subscription QoS is RELIABLE, which causes incompatible QoS warnings.
        self.pose_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
        )

        self.marker_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
        )

        self.pub = self.create_publisher(
            MarkerArray,
            "/debug/base_pose_axes",
            self.marker_qos,
        )

        self.subscribed = set()
        self.subs = []
        self.latest: dict[str, StoredPose] = {}

        self.scan_timer = self.create_timer(1.0, self.scan_pose_topics)

        hz = float(self.get_parameter("publish_hz").value)
        self.pub_timer = self.create_timer(1.0 / hz, self.publish_markers)

        self.get_logger().info("Base pose axes debug started")
        self.get_logger().info("Publishing MarkerArray: /debug/base_pose_axes")
        self.get_logger().info(f"Marker frame_id: {self.base_frame}")
        self.get_logger().info("Subscribing to /perception/fp/pose_base* topics with BEST_EFFORT QoS")

    def topic_allowed(self, topic: str) -> bool:
        if self.only_track_topics:
            return topic.startswith("/perception/fp/pose_base_track/")
        return any(topic.startswith(prefix) for prefix in POSE_PREFIXES)

    def scan_pose_topics(self):
        for topic, types in self.get_topic_names_and_types():
            if topic in self.subscribed:
                continue

            if not self.topic_allowed(topic):
                continue

            if "geometry_msgs/msg/Pose" in types:
                self.subs.append(
                    self.create_subscription(
                        Pose,
                        topic,
                        lambda msg, topic=topic: self.on_pose(topic, msg),
                        self.pose_qos,
                    )
                )
                self.subscribed.add(topic)
                self.get_logger().info(f"Subscribed Pose: {topic}")

            elif "geometry_msgs/msg/PoseStamped" in types:
                self.subs.append(
                    self.create_subscription(
                        PoseStamped,
                        topic,
                        lambda msg, topic=topic: self.on_pose_stamped(topic, msg),
                        self.pose_qos,
                    )
                )
                self.subscribed.add(topic)
                self.get_logger().info(f"Subscribed PoseStamped: {topic}")

            elif "fp_debug_msgs/msg/DebugPoseItem" in types:
                # Fused/assembly topic: many parts share one topic, one
                # DebugPoseItem message per part, keyed by assembly/part_id.
                self.subs.append(
                    self.create_subscription(
                        DebugPoseItem,
                        topic,
                        lambda msg, topic=topic: self.on_debug_pose_item(topic, msg),
                        self.pose_qos,
                    )
                )
                self.subscribed.add(topic)
                self.get_logger().info(f"Subscribed DebugPoseItem: {topic}")

    def now_sec(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_pose(self, topic: str, msg: Pose):
        self.latest[topic] = StoredPose(
            topic=topic,
            pose=msg,
            stamp_sec=self.now_sec(),
        )

    def on_pose_stamped(self, topic: str, msg: PoseStamped):
        self.latest[topic] = StoredPose(
            topic=topic,
            pose=msg.pose,
            stamp_sec=self.now_sec(),
        )

    def on_debug_pose_item(self, topic: str, msg: DebugPoseItem):
        # Many parts share one topic (fused/assembly); key each part
        # separately so they don't overwrite each other in self.latest.
        label = f"{msg.assembly_name}/{msg.part_id}" if msg.assembly_name else str(msg.part_id)
        key = f"{topic}/{label}"
        self.latest[key] = StoredPose(
            topic=topic,
            pose=msg.pose_base.pose,
            stamp_sec=self.now_sec(),
            label=label,
        )

    def make_arrow(self, topic, axis_id, origin, end, color, stamp):
        m = Marker()
        m.header.frame_id = self.base_frame
        m.header.stamp = stamp

        m.ns = safe_ns(topic)
        m.id = axis_id
        m.type = Marker.ARROW
        m.action = Marker.ADD

        m.points = [point_from_np(origin), point_from_np(end)]

        # ARROW marker with points:
        # scale.x = shaft diameter
        # scale.y = head diameter
        # scale.z = head length
        m.scale.x = 0.008
        m.scale.y = 0.020
        m.scale.z = 0.030

        m.color = color
        return m

    def make_origin_sphere(self, topic, origin, stamp):
        m = Marker()
        m.header.frame_id = self.base_frame
        m.header.stamp = stamp

        m.ns = safe_ns(topic)
        m.id = 200
        m.type = Marker.SPHERE
        m.action = Marker.ADD

        m.pose.position.x = float(origin[0])
        m.pose.position.y = float(origin[1])
        m.pose.position.z = float(origin[2])
        m.pose.orientation.w = 1.0

        m.scale.x = 0.025
        m.scale.y = 0.025
        m.scale.z = 0.025

        m.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.8)
        return m

    def make_text(self, topic, origin, R, stamp, display_name=None):
        rpy = R_to_rpy_deg(R)

        m = Marker()
        m.header.frame_id = self.base_frame
        m.header.stamp = stamp

        m.ns = safe_ns(topic)
        m.id = 100
        m.type = Marker.TEXT_VIEW_FACING
        m.action = Marker.ADD

        m.pose.position.x = float(origin[0])
        m.pose.position.y = float(origin[1])
        m.pose.position.z = float(origin[2] + 0.10)
        m.pose.orientation.w = 1.0

        m.scale.z = 0.035
        m.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)

        m.text = (
            f"{display_name or short_name(topic)}\n"
            f"xyz [{origin[0]:+.3f}, {origin[1]:+.3f}, {origin[2]:+.3f}]\n"
            f"rpy [{rpy[0]:+.1f}, {rpy[1]:+.1f}, {rpy[2]:+.1f}] deg"
        )
        return m

    def publish_markers(self):
        now = self.now_sec()
        stamp = self.get_clock().now().to_msg()

        arr = MarkerArray()

        # Clear all previous markers each cycle.
        # This prevents old axes staying forever when tracking is lost.
        delete_all = Marker()
        delete_all.header.frame_id = self.base_frame
        delete_all.header.stamp = stamp
        delete_all.action = Marker.DELETEALL
        arr.markers.append(delete_all)

        colors = [
            ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0),  # x axis red
            ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0),  # y axis green
            ColorRGBA(r=0.0, g=0.2, b=1.0, a=1.0),  # z axis blue
        ]

        active_count = 0

        for topic, stored in list(self.latest.items()):
            age = now - stored.stamp_sec
            if age > self.max_age_sec:
                continue

            p = stored.pose.position
            q = stored.pose.orientation

            origin = np.array([p.x, p.y, p.z], dtype=float)
            R = quat_xyzw_to_R([q.x, q.y, q.z, q.w])

            # Object-frame axes expressed in base frame.
            x_end = origin + R[:, 0] * self.axis_length
            y_end = origin + R[:, 1] * self.axis_length
            z_end = origin + R[:, 2] * self.axis_length

            arr.markers.append(self.make_arrow(topic, 0, origin, x_end, colors[0], stamp))
            arr.markers.append(self.make_arrow(topic, 1, origin, y_end, colors[1], stamp))
            arr.markers.append(self.make_arrow(topic, 2, origin, z_end, colors[2], stamp))
            arr.markers.append(self.make_origin_sphere(topic, origin, stamp))

            if self.show_text:
                display_name = stored.label or short_name(stored.topic)
                arr.markers.append(self.make_text(topic, origin, R, stamp, display_name))

            active_count += 1

        self.pub.publish(arr)


def main():
    rclpy.init()
    node = BasePoseAxesDebug()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
