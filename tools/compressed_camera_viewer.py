#!/usr/bin/env python3

import argparse

import cv2
import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge


class CameraViewer(Node):

    def __init__(self, camera):

        super().__init__("compressed_camera_viewer")

        self.camera = camera
        self.bridge = CvBridge()

        # ============================================================
        # Topics
        # ============================================================

        if camera in ["realsense_1", "realsense_2"]:

            # Already rectified RGB image
            self.rgb_topic = (
                f"/{camera}/camera/color/"
                "image_rect/compressed"
            )

            # Depth aligned to the RGB camera
            self.depth_topic = (
                f"/{camera}/camera/"
                "aligned_depth_to_color/"
                "image_raw/compressedDepth"
            )

        elif camera == "zed2i_1":

            self.rgb_topic = (
                "/zed2i_1/zed_node/"
                "rgb/color/rect/image/compressed"
            )

            self.depth_topic = (
                "/zed2i_1/zed_node/"
                "depth/depth_registered/compressedDepth"
            )

        else:
            raise ValueError(
                f"Unknown camera: {camera}"
            )

        self.get_logger().info(
            f"RGB topic:   {self.rgb_topic}"
        )

        self.get_logger().info(
            f"Depth topic: {self.depth_topic}"
        )

        # ============================================================
        # Latest images
        # ============================================================

        self.rgb = None
        self.depth = None

        # ============================================================
        # Subscribers
        # ============================================================

        self.rgb_sub = self.create_subscription(
            CompressedImage,
            self.rgb_topic,
            self.rgb_callback,
            10,
        )

        self.depth_sub = self.create_subscription(
            CompressedImage,
            self.depth_topic,
            self.depth_callback,
            10,
        )

        # Display at max 30 Hz
        self.timer = self.create_timer(
            1.0 / 30.0,
            self.display_callback,
        )

        self._printed_info = False

    # ================================================================
    # RGB callback
    # ================================================================

    def rgb_callback(self, msg):

        if not msg.data:
            self.get_logger().warning(
                "Received RGB CompressedImage with EMPTY data"
            )
            return

        self.get_logger().info(
            f"Received RGB compressed image: "
            f"{len(msg.data)} bytes, format='{msg.format}'"
        )

        try:
            self.rgb = self.bridge.compressed_imgmsg_to_cv2(
                msg,
                desired_encoding="bgr8",
            )

        except Exception as e:
            self.get_logger().error(
                f"RGB decode failed: {e}"
            )

        self.get_logger().info(
        f"decoded shape={self.rgb.shape}, "
        f"dtype={self.rgb.dtype}"
)

            
    # ================================================================
    # Depth callback
    # ================================================================

    def depth_callback(self, msg):

        try:

            if self.camera in ["realsense_1", "realsense_2"]:

                # cv_bridge.compressed_imgmsg_to_cv2() cannot decode
                # compressedDepth messages: compressed_depth_image_transport
                # prepends a 12-byte ConfigHeader (format enum + two
                # float32 quantization params) before the PNG payload,
                # which breaks cv2.imdecode if passed through untouched.
                #
                # For 16UC1 (RealSense) depth, no quantization is applied
                # on top of the PNG compression, so stripping the header
                # and decoding directly reproduces the raw depth exactly
                # (verified byte-for-byte against the uncompressed
                # aligned_depth_to_color/image_raw topic).
                raw = np.frombuffer(msg.data[12:], np.uint8)

                self.depth = cv2.imdecode(
                    raw,
                    cv2.IMREAD_UNCHANGED,
                )

            else:

                # compressedDepth -> original depth representation
                #
                # IMPORTANT:
                # Do NOT request bgr8 here.
                #
                # Usually this gives:
                #
                #   shape = (H, W)
                #   dtype = uint16
                #
                # for a 16UC1 depth image.
                #
                self.depth = self.bridge.compressed_imgmsg_to_cv2(
                    msg,
                    desired_encoding="passthrough",
                )

            self.get_logger().info(
                f"decoded depth shape={self.depth.shape}, "
                f"dtype={self.depth.dtype}"
            )

        except Exception as e:

            self.get_logger().error(
                f"Depth decode failed: {e}"
            )

    # ================================================================
    # Display
    # ================================================================

    def display_callback(self):

        if self.rgb is None or self.depth is None:
            return

        # ------------------------------------------------------------
        # Print actual decoded representations once
        # ------------------------------------------------------------

        if not self._printed_info:

            self.get_logger().info(
                f"RGB:   shape={self.rgb.shape}, "
                f"dtype={self.rgb.dtype}"
            )

            self.get_logger().info(
                f"Depth: shape={self.depth.shape}, "
                f"dtype={self.depth.dtype}"
            )

            self._printed_info = True

        # ------------------------------------------------------------
        # RGB
        # ------------------------------------------------------------

        rgb_display = self.rgb

        # ------------------------------------------------------------
        # Depth
        # ------------------------------------------------------------

        depth = self.depth

        depth_float = depth.astype(np.float32)

        # Valid depth pixels
        valid = (
            np.isfinite(depth_float)
            & (depth_float > 0)
        )

        if np.any(valid):

            # Robust range for visualization
            near = np.percentile(
                depth_float[valid],
                2,
            )

            far = np.percentile(
                depth_float[valid],
                98,
            )

            if far > near:

                depth_normalized = (
                    (depth_float - near)
                    / (far - near)
                    * 255.0
                )

                depth_normalized = np.clip(
                    depth_normalized,
                    0,
                    255,
                ).astype(np.uint8)

            else:

                depth_normalized = np.zeros(
                    depth.shape,
                    dtype=np.uint8,
                )

            # Invalid depth = black
            depth_normalized[~valid] = 0

            # Convert scalar depth into a visible color image
            depth_display = cv2.applyColorMap(
                depth_normalized,
                cv2.COLORMAP_JET,
            )

            depth_display[~valid] = 0

        else:

            depth_display = np.zeros(
                (
                    depth.shape[0],
                    depth.shape[1],
                    3,
                ),
                dtype=np.uint8,
            )

        # ------------------------------------------------------------
        # Match image sizes
        # ------------------------------------------------------------

        if (
            rgb_display.shape[:2]
            != depth_display.shape[:2]
        ):

            depth_display = cv2.resize(
                depth_display,
                (
                    rgb_display.shape[1],
                    rgb_display.shape[0],
                ),
                interpolation=cv2.INTER_NEAREST,
            )

        # ------------------------------------------------------------
        # Side-by-side display
        # ------------------------------------------------------------

        combined = np.hstack(
            [
                rgb_display,
                depth_display,
            ]
        )

        cv2.imshow(
            f"{self.camera} | RGB + aligned depth",
            combined,
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:

            rclpy.shutdown()

    # ================================================================

    def destroy_node(self):

        cv2.destroyAllWindows()

        super().destroy_node()


def main():

    parser = argparse.ArgumentParser(
        description="View compressed RGB and aligned depth"
    )

    parser.add_argument(
        "camera",
        choices=[
            "realsense_1",
            "realsense_2",
            "zed2i_1",
        ],
        help="Camera to view",
    )

    args = parser.parse_args()

    rclpy.init()

    node = CameraViewer(args.camera)

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:

        pass

    finally:

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()