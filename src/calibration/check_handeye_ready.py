#!/usr/bin/env python3

import math
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from sensor_msgs.msg import Image, CameraInfo

from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException


class WristHandeyeReadinessChecker(Node):
    def __init__(self) -> None:
        super().__init__("check_wrist_handeye_ready")

        self.declare_parameter("image_topic", "/zed_mini/zed_node/rgb/image_rect_color")
        self.declare_parameter("camera_info_topic", "/zed_mini/zed_node/rgb/camera_info")
        self.declare_parameter("base_frame", "base")
        self.declare_parameter("ee_frame", "fr3_hand")
        self.declare_parameter("camera_frame", "zed_mini_left_camera_optical_frame")
        self.declare_parameter("wait_seconds", 8.0)
        self.declare_parameter("min_image_msgs", 5)
        self.declare_parameter("min_camera_info_msgs", 2)
        self.declare_parameter("freshness_warn_s", 1.0)

        self.image_topic = self.get_parameter("image_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value
        self.base_frame = self.get_parameter("base_frame").value
        self.ee_frame = self.get_parameter("ee_frame").value
        self.camera_frame = self.get_parameter("camera_frame").value
        self.wait_seconds = float(self.get_parameter("wait_seconds").value)
        self.min_image_msgs = int(self.get_parameter("min_image_msgs").value)
        self.min_camera_info_msgs = int(self.get_parameter("min_camera_info_msgs").value)
        self.freshness_warn_s = float(self.get_parameter("freshness_warn_s").value)

        self.image_count = 0
        self.camera_info_count = 0
        self.first_image_stamp: Optional[float] = None
        self.last_image_stamp: Optional[float] = None
        self.first_camera_info_stamp: Optional[float] = None
        self.last_camera_info_stamp: Optional[float] = None
        self.last_image_width: Optional[int] = None
        self.last_image_height: Optional[int] = None
        self.last_camera_info_width: Optional[int] = None
        self.last_camera_info_height: Optional[int] = None
        self.last_k: Optional[list] = None

        self.create_subscription(Image, self.image_topic, self.image_cb, 10)
        self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_cb, 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.start_time_ros = self.get_clock().now()
        self.done = False

        self.timer = self.create_timer(0.5, self.tick)

        self.get_logger().info("=== Wrist hand-eye readiness check (pre-hand-eye version) ===")
        self.get_logger().info(f"image_topic      : {self.image_topic}")
        self.get_logger().info(f"camera_info_topic: {self.camera_info_topic}")
        self.get_logger().info(f"base_frame       : {self.base_frame}")
        self.get_logger().info(f"ee_frame         : {self.ee_frame}")
        self.get_logger().info(f"camera_frame     : {self.camera_frame}")
        self.get_logger().info(
            "This version does NOT require ee<-camera to exist yet, because that transform is what hand-eye will estimate."
        )

    @staticmethod
    def stamp_to_sec(stamp) -> float:
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def image_cb(self, msg: Image) -> None:
        t = self.stamp_to_sec(msg.header.stamp)
        self.image_count += 1
        if self.first_image_stamp is None:
            self.first_image_stamp = t
        self.last_image_stamp = t
        self.last_image_width = msg.width
        self.last_image_height = msg.height

    def camera_info_cb(self, msg: CameraInfo) -> None:
        t = self.stamp_to_sec(msg.header.stamp)
        self.camera_info_count += 1
        if self.first_camera_info_stamp is None:
            self.first_camera_info_stamp = t
        self.last_camera_info_stamp = t
        self.last_camera_info_width = msg.width
        self.last_camera_info_height = msg.height
        self.last_k = list(msg.k)

    def _msg_rate(self, first_stamp: Optional[float], last_stamp: Optional[float], count: int) -> Optional[float]:
        if first_stamp is None or last_stamp is None or count < 2:
            return None
        dt = last_stamp - first_stamp
        if dt <= 0.0:
            return None
        return (count - 1) / dt

    def _freshness(self, last_stamp: Optional[float]) -> Optional[float]:
        if last_stamp is None:
            return None
        now = self.get_clock().now().nanoseconds * 1e-9
        return now - last_stamp

    def check_tf(self, target_frame: str, source_frame: str, timeout_s: float = 0.3) -> bool:
        try:
            self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=timeout_s),
            )
            return True
        except (LookupException, ConnectivityException, ExtrapolationException):
            return False

    def print_topic_summary(self) -> bool:
        ok = True

        print("\n--- Topic checks ---")

        if self.image_count >= self.min_image_msgs:
            image_rate = self._msg_rate(self.first_image_stamp, self.last_image_stamp, self.image_count)
            freshness = self._freshness(self.last_image_stamp)
            print(f"[OK]  Image messages received: {self.image_count}")
            print(
                f"      Last image size: {self.last_image_width}x{self.last_image_height}, "
                f"approx rate: {image_rate:.2f} Hz" if image_rate is not None
                else f"      Last image size: {self.last_image_width}x{self.last_image_height}, rate: n/a"
            )
            if freshness is not None:
                freshness_str = f"{freshness:.3f} s"
                if freshness > self.freshness_warn_s:
                    print(f"[WARN] Image timestamps are old by about {freshness_str}")
                    ok = False
                else:
                    print(f"[OK]  Image timestamps are fresh ({freshness_str} old)")
        else:
            print(
                f"[FAIL] Not enough image messages on {self.image_topic}: "
                f"got {self.image_count}, expected at least {self.min_image_msgs}"
            )
            ok = False

        if self.camera_info_count >= self.min_camera_info_msgs:
            ci_rate = self._msg_rate(
                self.first_camera_info_stamp,
                self.last_camera_info_stamp,
                self.camera_info_count,
            )
            freshness = self._freshness(self.last_camera_info_stamp)
            print(f"[OK]  CameraInfo messages received: {self.camera_info_count}")
            print(
                f"      Last CameraInfo size: {self.last_camera_info_width}x{self.last_camera_info_height}, "
                f"approx rate: {ci_rate:.2f} Hz" if ci_rate is not None
                else f"      Last CameraInfo size: {self.last_camera_info_width}x{self.last_camera_info_height}, rate: n/a"
            )
            if self.last_k is not None:
                fx = self.last_k[0]
                fy = self.last_k[4]
                cx = self.last_k[2]
                cy = self.last_k[5]
                print(f"      Intrinsics snapshot: fx={fx:.3f}, fy={fy:.3f}, cx={cx:.3f}, cy={cy:.3f}")
            if freshness is not None:
                freshness_str = f"{freshness:.3f} s"
                if freshness > self.freshness_warn_s:
                    print(f"[WARN] CameraInfo timestamps are old by about {freshness_str}")
                    ok = False
                else:
                    print(f"[OK]  CameraInfo timestamps are fresh ({freshness_str} old)")
        else:
            print(
                f"[FAIL] Not enough CameraInfo messages on {self.camera_info_topic}: "
                f"got {self.camera_info_count}, expected at least {self.min_camera_info_msgs}"
            )
            ok = False

        return ok

    def print_tf_summary(self) -> bool:
        ok = True

        print("\n--- TF checks ---")

        base_to_ee = self.check_tf(self.base_frame, self.ee_frame)
        if base_to_ee:
            print(f"[OK]  TF available: {self.base_frame} <- {self.ee_frame}")
        else:
            print(f"[FAIL] Missing TF: {self.base_frame} <- {self.ee_frame}")
            ok = False

        camera_frame_exists = False
        try:
            frames_yaml = self.tf_buffer.all_frames_as_yaml()
            if self.camera_frame in frames_yaml:
                camera_frame_exists = True
        except Exception:
            camera_frame_exists = False

        if camera_frame_exists:
            print(f"[OK]  Camera frame appears in TF tree: {self.camera_frame}")
        else:
            print(
                f"[WARN] Camera frame not found in current TF tree dump: {self.camera_frame}\n"
                "       This may still be okay if the camera driver is not publishing TF and you only use the frame_id from messages."
            )

        # Deliberately do not require ee<-camera yet.
        print(
            f"[INFO] Skipping check for {self.ee_frame} <- {self.camera_frame} "
            "because that fixed transform is the output of hand-eye calibration."
        )

        return ok

    def tick(self) -> None:
        elapsed = (self.get_clock().now() - self.start_time_ros).nanoseconds * 1e-9
        if self.done or elapsed < self.wait_seconds:
            return

        self.done = True

        topic_ok = self.print_topic_summary()
        tf_ok = self.print_tf_summary()

        print("\n--- Overall result ---")
        if topic_ok and tf_ok:
            print("[PASS] System looks ready for wrist-camera hand-eye data recording.")
        else:
            print("[FAIL] System is not fully ready yet. Fix the failed checks above first.")

        print("\nSuggested next lab command sequence:")
        print("1) Start robot + TF")
        print("2) Start wrist camera driver")
        print("3) Run this readiness checker")
        print("4) If PASS: start recording checkerboard data for hand-eye")

        rclpy.shutdown()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = WristHandeyeReadinessChecker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
    