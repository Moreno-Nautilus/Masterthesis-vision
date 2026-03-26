#!/usr/bin/env python3

import json
import math
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.duration import Duration
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException


def quat_angle_deg(q1, q2) -> float:
    q1 = np.asarray(q1, dtype=float)
    q2 = np.asarray(q2, dtype=float)
    q1 /= np.linalg.norm(q1)
    q2 /= np.linalg.norm(q2)
    dot = abs(float(np.dot(q1, q2)))
    dot = min(1.0, max(-1.0, dot))
    return math.degrees(2.0 * math.acos(dot))


class TriggeredWristHandeyeRecorder(Node):
    def __init__(self) -> None:
        super().__init__("record_wrist_handeye_samples_triggered")

        self.declare_parameter("image_topic", "/zed_mini/zed_node/rgb/image_rect_color")
        self.declare_parameter("camera_info_topic", "/zed_mini/zed_node/rgb/camera_info")
        self.declare_parameter("base_frame", "base")
        self.declare_parameter("ee_frame", "fr3_hand")

        self.declare_parameter("board_cols", 9)   # inner corners
        self.declare_parameter("board_rows", 6)   # inner corners
        self.declare_parameter("square_size_m", 0.025)

        self.declare_parameter("output_dir", "./handeye_recording")
        self.declare_parameter("save_drawn_debug", True)

        self.declare_parameter("min_translation_delta_m", 0.015)
        self.declare_parameter("min_rotation_delta_deg", 8.0)
        self.declare_parameter("max_samples", 40)

        self.declare_parameter("sharpness_threshold", 80.0)
        self.declare_parameter("require_complete_board", True)
        self.declare_parameter("status_period_s", 1.0)

        self.image_topic = self.get_parameter("image_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value
        self.base_frame = self.get_parameter("base_frame").value
        self.ee_frame = self.get_parameter("ee_frame").value

        self.board_cols = int(self.get_parameter("board_cols").value)
        self.board_rows = int(self.get_parameter("board_rows").value)
        self.square_size_m = float(self.get_parameter("square_size_m").value)

        self.output_dir = Path(self.get_parameter("output_dir").value).expanduser().resolve()
        self.save_drawn_debug = bool(self.get_parameter("save_drawn_debug").value)

        self.min_translation_delta_m = float(self.get_parameter("min_translation_delta_m").value)
        self.min_rotation_delta_deg = float(self.get_parameter("min_rotation_delta_deg").value)
        self.max_samples = int(self.get_parameter("max_samples").value)

        self.sharpness_threshold = float(self.get_parameter("sharpness_threshold").value)
        self.require_complete_board = bool(self.get_parameter("require_complete_board").value)
        self.status_period_s = float(self.get_parameter("status_period_s").value)

        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.latest_camera_info: Optional[CameraInfo] = None
        self.last_image_stamp_ns: Optional[int] = None
        self.last_camera_info_stamp_ns: Optional[int] = None

        self.last_saved_t: Optional[np.ndarray] = None
        self.last_saved_q: Optional[np.ndarray] = None
        self.sample_idx = 0

        self.latest_valid = None
        self.latest_valid_lock = threading.Lock()

        self.latest_status = {
            "tf_ok": False,
            "camera_info_ok": False,
            "board_visible": False,
            "sharpness": None,
            "sharpness_ok": False,
            "latest_valid_available": False,
            "last_reason": "waiting for data",
        }
        self.status_lock = threading.Lock()

        self.pending_save_request = False
        self.pending_save_lock = threading.Lock()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "images").mkdir(exist_ok=True)
        if self.save_drawn_debug:
            (self.output_dir / "debug").mkdir(exist_ok=True)

        self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_cb, 10)
        self.create_subscription(Image, self.image_topic, self.image_cb, 10)

        self.status_timer = self.create_timer(self.status_period_s, self.print_status)

        self.input_thread = threading.Thread(target=self.keyboard_loop, daemon=True)
        self.input_thread.start()

        self.get_logger().info("=== Triggered wrist hand-eye recorder ===")
        self.get_logger().info(f"image_topic      : {self.image_topic}")
        self.get_logger().info(f"camera_info_topic: {self.camera_info_topic}")
        self.get_logger().info(f"base_frame       : {self.base_frame}")
        self.get_logger().info(f"ee_frame         : {self.ee_frame}")
        self.get_logger().info(f"board size       : {self.board_cols} x {self.board_rows} inner corners")
        self.get_logger().info(f"square size      : {self.square_size_m} m")
        self.get_logger().info(f"output_dir       : {self.output_dir}")
        self.get_logger().info("Move robot to pose, hold still, make checkerboard visible, then press Enter to save.")
        self.get_logger().info("Type 'q' + Enter to quit.")

    def camera_info_cb(self, msg: CameraInfo) -> None:
        self.latest_camera_info = msg
        self.last_camera_info_stamp_ns = self.get_clock().now().nanoseconds
        with self.status_lock:
            self.latest_status["camera_info_ok"] = True

    def get_base_to_ee(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.ee_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.2),
            )
            tr = tf.transform.translation
            ro = tf.transform.rotation
            t = np.array([tr.x, tr.y, tr.z], dtype=float)
            q = np.array([ro.x, ro.y, ro.z, ro.w], dtype=float)
            return t, q, tf
        except (LookupException, ConnectivityException, ExtrapolationException):
            return None, None, None

    def sharpness_score(self, gray: np.ndarray) -> float:
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def pose_is_diverse_enough(self, t: np.ndarray, q: np.ndarray) -> bool:
        if self.last_saved_t is None or self.last_saved_q is None:
            return True
        dt = float(np.linalg.norm(t - self.last_saved_t))
        da = quat_angle_deg(q, self.last_saved_q)
        return (dt >= self.min_translation_delta_m) or (da >= self.min_rotation_delta_deg)

    def keyboard_loop(self) -> None:
        while rclpy.ok():
            try:
                s = input()
            except EOFError:
                break

            if s.strip().lower() == "q":
                self.get_logger().info("Quit requested by user.")
                rclpy.shutdown()
                return

            with self.pending_save_lock:
                self.pending_save_request = True

    def print_status(self) -> None:
        with self.status_lock:
            st = dict(self.latest_status)

        sharpness_str = "n/a" if st["sharpness"] is None else f"{st['sharpness']:.1f}"
        self.get_logger().info(
            "[STATUS] "
            f"TF={'OK' if st['tf_ok'] else 'MISS'} | "
            f"CamInfo={'OK' if st['camera_info_ok'] else 'MISS'} | "
            f"Board={'YES' if st['board_visible'] else 'NO'} | "
            f"Sharp={sharpness_str} ({'OK' if st['sharpness_ok'] else 'LOW'}) | "
            f"Valid={'YES' if st['latest_valid_available'] else 'NO'} | "
            f"Saved={self.sample_idx}/{self.max_samples} | "
            f"Reason={st['last_reason']}"
        )

    def save_sample(self, valid_data: dict) -> None:
        sample_name = f"sample_{self.sample_idx:04d}"

        img_path = self.output_dir / "images" / f"{sample_name}.png"
        cv2.imwrite(str(img_path), valid_data["image_bgr"])

        if self.save_drawn_debug:
            dbg = valid_data["image_bgr"].copy()
            cv2.drawChessboardCorners(
                dbg,
                (self.board_cols, self.board_rows),
                valid_data["corners"],
                True,
            )
            dbg_path = self.output_dir / "debug" / f"{sample_name}_corners.png"
            cv2.imwrite(str(dbg_path), dbg)

        data = {
            "sample_name": sample_name,
            "image_path": str(img_path),
            "image_header": {
                "frame_id": valid_data["msg"].header.frame_id,
                "stamp_sec": int(valid_data["msg"].header.stamp.sec),
                "stamp_nanosec": int(valid_data["msg"].header.stamp.nanosec),
            },
            "board": {
                "cols_inner_corners": self.board_cols,
                "rows_inner_corners": self.board_rows,
                "square_size_m": self.square_size_m,
            },
            "camera_info": {
                "width": int(valid_data["camera_info"].width),
                "height": int(valid_data["camera_info"].height),
                "k": list(valid_data["camera_info"].k),
                "d": list(valid_data["camera_info"].d),
                "r": list(valid_data["camera_info"].r),
                "p": list(valid_data["camera_info"].p),
                "distortion_model": valid_data["camera_info"].distortion_model,
                "header_frame_id": valid_data["camera_info"].header.frame_id,
                "stamp_sec": int(valid_data["camera_info"].header.stamp.sec),
                "stamp_nanosec": int(valid_data["camera_info"].header.stamp.nanosec),
            },
            "detector": {
                "found_complete_board": True,
                "num_corners": int(len(valid_data["corners"])),
                "corners_px": valid_data["corners"].reshape(-1, 2).tolist(),
                "sharpness": float(valid_data["sharpness"]),
            },
            "robot_pose_base_to_ee": {
                "target_frame": self.base_frame,
                "source_frame": self.ee_frame,
                "translation_xyz": valid_data["t"].tolist(),
                "quaternion_xyzw": valid_data["q"].tolist(),
                "tf_header_frame_id": valid_data["tf_msg"].header.frame_id,
                "tf_child_frame_id": valid_data["tf_msg"].child_frame_id,
                "stamp_sec": int(valid_data["tf_msg"].header.stamp.sec),
                "stamp_nanosec": int(valid_data["tf_msg"].header.stamp.nanosec),
            },
        }

        json_path = self.output_dir / f"{sample_name}.json"
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)

        self.last_saved_t = valid_data["t"].copy()
        self.last_saved_q = valid_data["q"].copy()
        self.sample_idx += 1

        self.get_logger().info(
            f"[SAVED] {sample_name} | sharpness={valid_data['sharpness']:.1f} | total={self.sample_idx}/{self.max_samples}"
        )

        if self.sample_idx >= self.max_samples:
            self.get_logger().info(f"Reached max_samples={self.max_samples}. Stopping.")
            rclpy.shutdown()

    def image_cb(self, msg: Image) -> None:
        self.last_image_stamp_ns = self.get_clock().now().nanoseconds

        if self.latest_camera_info is None:
            with self.status_lock:
                self.latest_status["camera_info_ok"] = False
                self.latest_status["tf_ok"] = False
                self.latest_status["board_visible"] = False
                self.latest_status["sharpness"] = None
                self.latest_status["sharpness_ok"] = False
                self.latest_status["latest_valid_available"] = False
                self.latest_status["last_reason"] = "waiting for CameraInfo"
            self.handle_trigger_no_valid()
            return

        t, q, tf_msg = self.get_base_to_ee()
        if t is None:
            with self.status_lock:
                self.latest_status["tf_ok"] = False
                self.latest_status["board_visible"] = False
                self.latest_status["sharpness"] = None
                self.latest_status["sharpness_ok"] = False
                self.latest_status["latest_valid_available"] = False
                self.latest_status["last_reason"] = f"missing TF {self.base_frame}<-{self.ee_frame}"
            self.handle_trigger_no_valid()
            return

        with self.status_lock:
            self.latest_status["tf_ok"] = True
            self.latest_status["camera_info_ok"] = True

        try:
            image_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            with self.status_lock:
                self.latest_status["board_visible"] = False
                self.latest_status["sharpness"] = None
                self.latest_status["sharpness_ok"] = False
                self.latest_status["latest_valid_available"] = False
                self.latest_status["last_reason"] = f"cv_bridge failed: {e}"
            self.handle_trigger_no_valid()
            return

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        sharpness = self.sharpness_score(gray)
        sharpness_ok = sharpness >= self.sharpness_threshold

        pattern_size = (self.board_cols, self.board_rows)
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)

        with self.status_lock:
            self.latest_status["board_visible"] = bool(found)
            self.latest_status["sharpness"] = sharpness
            self.latest_status["sharpness_ok"] = bool(sharpness_ok)

        if not sharpness_ok:
            with self.latest_valid_lock:
                self.latest_valid = None
            with self.status_lock:
                self.latest_status["latest_valid_available"] = False
                self.latest_status["last_reason"] = "image too blurry"
            self.handle_trigger_no_valid()
            return

        if not found:
            with self.latest_valid_lock:
                self.latest_valid = None
            with self.status_lock:
                self.latest_status["latest_valid_available"] = False
                self.latest_status["last_reason"] = "checkerboard not visible"
            self.handle_trigger_no_valid()
            return

        corners = cv2.cornerSubPix(
            gray,
            corners,
            winSize=(11, 11),
            zeroZone=(-1, -1),
            criteria=(
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                30,
                0.001,
            ),
        )

        if self.require_complete_board and len(corners) != self.board_rows * self.board_cols:
            with self.latest_valid_lock:
                self.latest_valid = None
            with self.status_lock:
                self.latest_status["latest_valid_available"] = False
                self.latest_status["last_reason"] = "incomplete checkerboard"
            self.handle_trigger_no_valid()
            return

        valid_data = {
            "msg": msg,
            "camera_info": self.latest_camera_info,
            "image_bgr": image_bgr,
            "corners": corners,
            "sharpness": sharpness,
            "t": t,
            "q": q,
            "tf_msg": tf_msg,
        }

        with self.latest_valid_lock:
            self.latest_valid = valid_data

        with self.status_lock:
            self.latest_status["latest_valid_available"] = True
            self.latest_status["last_reason"] = "ready to save"

        should_save = False
        with self.pending_save_lock:
            if self.pending_save_request:
                should_save = True
                self.pending_save_request = False

        if not should_save:
            return

        if not self.pose_is_diverse_enough(valid_data["t"], valid_data["q"]):
            self.get_logger().warn("Triggered sample too similar to last saved pose. Move to a more different pose.")
            with self.status_lock:
                self.latest_status["last_reason"] = "pose too similar to last saved"
            return

        self.save_sample(valid_data)

    def handle_trigger_no_valid(self) -> None:
        should_save = False
        with self.pending_save_lock:
            if self.pending_save_request:
                should_save = True
                self.pending_save_request = False

        if should_save:
            with self.status_lock:
                reason = self.latest_status["last_reason"]
            self.get_logger().warn(f"No valid sample to save right now: {reason}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TriggeredWristHandeyeRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()