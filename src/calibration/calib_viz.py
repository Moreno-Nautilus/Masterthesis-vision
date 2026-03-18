from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped, Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
import yaml

from src.utils.se3 import SE3


EXTRINSICS_YAML = "config/camera_extrinsics_base.yaml"
BASE_BOARD_YAML = "config/base_board_pose.yaml"

BASE_FRAME = "base"
BOARD_FRAME = "checkerboard"

CHESS_COLS = 8
CHESS_ROWS = 11
SQUARE_SIZE_M = 0.03

MARKER_TOPIC = "/calibration_markers"
PUBLISH_PERIOD_S = 1.0


def _rpy_deg_to_R(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    r = np.deg2rad(roll_deg)
    p = np.deg2rad(pitch_deg)
    y = np.deg2rad(yaw_deg)

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(r), -np.sin(r)],
        [0, np.sin(r),  np.cos(r)],
    ], dtype=float)

    Ry = np.array([
        [ np.cos(p), 0, np.sin(p)],
        [0,          1, 0],
        [-np.sin(p), 0, np.cos(p)],
    ], dtype=float)

    Rz = np.array([
        [np.cos(y), -np.sin(y), 0],
        [np.sin(y),  np.cos(y), 0],
        [0,          0,         1],
    ], dtype=float)

    return Rz @ Ry @ Rx


def _rotmat_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    q = np.empty(4, dtype=float)
    trace = np.trace(R)

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        q[3] = 0.25 / s
        q[0] = (R[2, 1] - R[1, 2]) * s
        q[1] = (R[0, 2] - R[2, 0]) * s
        q[2] = (R[1, 0] - R[0, 1]) * s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            q[3] = (R[2, 1] - R[1, 2]) / s
            q[0] = 0.25 * s
            q[1] = (R[0, 1] + R[1, 0]) / s
            q[2] = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            q[3] = (R[0, 2] - R[2, 0]) / s
            q[0] = (R[0, 1] + R[1, 0]) / s
            q[1] = 0.25 * s
            q[2] = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            q[3] = (R[1, 0] - R[0, 1]) / s
            q[0] = (R[0, 2] + R[2, 0]) / s
            q[1] = (R[1, 2] + R[2, 1]) / s
            q[2] = 0.25 * s

    q /= np.linalg.norm(q) + 1e-12
    return q  # x, y, z, w


def _load_base_board(path: Path) -> SE3:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    bb = cfg["base_board"]
    t = np.array(bb["translation_xyz_m"], dtype=float)
    roll, pitch, yaw = bb["rotation_rpy_deg"]
    R = _rpy_deg_to_R(float(roll), float(pitch), float(yaw))
    return SE3(R, t)


def _parse_se3_from_entry(entry: dict) -> SE3:
    # Supports a few common YAML styles.
    if "translation_xyz_m" in entry and "rotation_matrix" in entry:
        t = np.array(entry["translation_xyz_m"], dtype=float)
        R = np.array(entry["rotation_matrix"], dtype=float).reshape(3, 3)
        return SE3(R, t)

    if "translation" in entry and "rotation_matrix" in entry:
        t = np.array(entry["translation"], dtype=float)
        R = np.array(entry["rotation_matrix"], dtype=float).reshape(3, 3)
        return SE3(R, t)

    if "t" in entry and "R" in entry:
        t = np.array(entry["t"], dtype=float).reshape(3)
        R = np.array(entry["R"], dtype=float).reshape(3, 3)
        return SE3(R, t)

    if "matrix" in entry:
        T = np.array(entry["matrix"], dtype=float).reshape(4, 4)
        return SE3(T[:3, :3], T[:3, 3])

    raise ValueError(f"Unsupported extrinsics entry format: {entry.keys()}")


def _load_extrinsics(path: Path) -> Dict[str, SE3]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    out: Dict[str, SE3] = {}
    for cam_name, entry in cfg.items():
        out[cam_name] = _parse_se3_from_entry(entry)
    return out


def _make_tf(parent: str, child: str, T_parent_child: SE3, stamp) -> TransformStamped:
    tf = TransformStamped()
    tf.header.stamp = stamp
    tf.header.frame_id = parent
    tf.child_frame_id = child
    tf.transform.translation.x = float(T_parent_child.t[0])
    tf.transform.translation.y = float(T_parent_child.t[1])
    tf.transform.translation.z = float(T_parent_child.t[2])

    q = _rotmat_to_quat_xyzw(T_parent_child.R)
    tf.transform.rotation.x = float(q[0])
    tf.transform.rotation.y = float(q[1])
    tf.transform.rotation.z = float(q[2])
    tf.transform.rotation.w = float(q[3])
    return tf


def _color(r, g, b, a=1.0) -> ColorRGBA:
    c = ColorRGBA()
    c.r = float(r)
    c.g = float(g)
    c.b = float(b)
    c.a = float(a)
    return c


def _axis_marker(frame_id: str, ns: str, mid: int, axis: str, length: float, radius: float) -> Marker:
    m = Marker()
    m.header.frame_id = frame_id
    m.ns = ns
    m.id = mid
    m.type = Marker.ARROW
    m.action = Marker.ADD

    p0 = Point(x=0.0, y=0.0, z=0.0)
    if axis == "x":
        p1 = Point(x=length, y=0.0, z=0.0)
        m.color = _color(1.0, 0.0, 0.0, 1.0)
    elif axis == "y":
        p1 = Point(x=0.0, y=length, z=0.0)
        m.color = _color(0.0, 1.0, 0.0, 1.0)
    else:
        p1 = Point(x=0.0, y=0.0, z=length)
        m.color = _color(0.0, 0.3, 1.0, 1.0)

    m.points = [p0, p1]
    m.scale.x = radius
    m.scale.y = radius * 2.0
    m.scale.z = radius * 2.0
    return m


def _board_marker(frame_id: str, board_x_m: float, board_y_m: float) -> Marker:
    m = Marker()
    m.header.frame_id = frame_id
    m.ns = "board"
    m.id = 100
    m.type = Marker.CUBE
    m.action = Marker.ADD

    # The board frame origin is one board corner.
    m.pose.position.x = board_x_m / 2.0
    m.pose.position.y = board_y_m / 2.0
    m.pose.position.z = 0.0
    m.pose.orientation.w = 1.0

    m.scale.x = board_x_m
    m.scale.y = board_y_m
    m.scale.z = 0.002
    m.color = _color(1.0, 1.0, 0.0, 0.35)
    return m


class CalibrationViz(Node):
    def __init__(self):
        super().__init__("calibration_visualizer")

        self.tf_pub = StaticTransformBroadcaster(self)
        self.marker_pub = self.create_publisher(MarkerArray, MARKER_TOPIC, 10)

        self.T_base_board = _load_base_board(Path(BASE_BOARD_YAML))
        self.extrinsics = _load_extrinsics(Path(EXTRINSICS_YAML))

        self.board_x_m = (CHESS_COLS - 1) * SQUARE_SIZE_M
        self.board_y_m = (CHESS_ROWS - 1) * SQUARE_SIZE_M

        self._publish_static_tfs()
        self.timer = self.create_timer(PUBLISH_PERIOD_S, self._publish_markers)

        self.get_logger().info(f"Loaded board pose from {BASE_BOARD_YAML}")
        self.get_logger().info(f"Loaded camera extrinsics from {EXTRINSICS_YAML}")
        for name, T in self.extrinsics.items():
            self.get_logger().info(f"{name}: t = {T.t}")

    def _publish_static_tfs(self):
        stamp = self.get_clock().now().to_msg()

        tfs = []
        tfs.append(_make_tf(BASE_FRAME, BOARD_FRAME, self.T_base_board, stamp))

        for cam_name, T_base_cam in self.extrinsics.items():
            tfs.append(_make_tf(BASE_FRAME, cam_name, T_base_cam, stamp))

        self.tf_pub.sendTransform(tfs)

    def _publish_markers(self):
        arr = MarkerArray()

        # Base axes
        arr.markers.append(_axis_marker(BASE_FRAME, "base_axes", 1, "x", 0.20, 0.01))
        arr.markers.append(_axis_marker(BASE_FRAME, "base_axes", 2, "y", 0.20, 0.01))
        arr.markers.append(_axis_marker(BASE_FRAME, "base_axes", 3, "z", 0.20, 0.01))

        # Board axes + board rectangle
        arr.markers.append(_axis_marker(BOARD_FRAME, "board_axes", 11, "x", 0.10, 0.006))
        arr.markers.append(_axis_marker(BOARD_FRAME, "board_axes", 12, "y", 0.10, 0.006))
        arr.markers.append(_axis_marker(BOARD_FRAME, "board_axes", 13, "z", 0.10, 0.006))
        arr.markers.append(_board_marker(BOARD_FRAME, self.board_x_m, self.board_y_m))

        # Camera axes
        base_id = 1000
        for i, cam_name in enumerate(self.extrinsics.keys()):
            arr.markers.append(_axis_marker(cam_name, f"{cam_name}_axes", base_id + 10*i + 1, "x", 0.08, 0.004))
            arr.markers.append(_axis_marker(cam_name, f"{cam_name}_axes", base_id + 10*i + 2, "y", 0.08, 0.004))
            arr.markers.append(_axis_marker(cam_name, f"{cam_name}_axes", base_id + 10*i + 3, "z", 0.08, 0.004))

        self.marker_pub.publish(arr)


def main():
    rclpy.init()
    node = CalibrationViz()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()