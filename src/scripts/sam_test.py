from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Header

from src.calibration.io_extrinsics import load_extrinsics_yaml
from src.perception.ros.multicam_grabber import CameraTopics, MultiCamGrabber
from src.perception.learned.SAM.sam_segmentation import SAMSegmenter, SAMSegmenterConfig


CAMERAS = [
    CameraTopics(
        cam_id="zed2i_1",
        depth_topic="/zed2i_1/zed_node/depth/depth_registered",
        info_topic="/zed2i_1/zed_node/depth/depth_registered/camera_info",
        rgb_topic="/zed2i_1/zed_node/rgb/color/rect/image",
        rgb_info_topic="/zed2i_1/zed_node/rgb/color/rect/image/camera_info",
    ),
    CameraTopics(
        cam_id="zed2i_2",
        depth_topic="/zed2i_2/zed_node/depth/depth_registered",
        info_topic="/zed2i_2/zed_node/depth/depth_registered/camera_info",
        rgb_topic="/zed2i_2/zed_node/rgb/color/rect/image",
        rgb_info_topic="/zed2i_2/zed_node/rgb/color/rect/image/camera_info",
    ),
]

TIMER_PERIOD_S = 0.20
MIN_PROCESS_DT_S = 0.15
MAX_MASKS_DRAW = 8

FAST_QOS = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
)


def _rgb_numpy_to_imgmsg(rgb: np.ndarray, frame_id: str, stamp) -> Image:
    rgb = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8))
    msg = Image()
    msg.header = Header(frame_id=frame_id, stamp=stamp)
    msg.height = int(rgb.shape[0])
    msg.width = int(rgb.shape[1])
    msg.encoding = "rgb8"
    msg.is_bigendian = False
    msg.step = int(rgb.shape[1] * 3)
    msg.data = rgb.tobytes()
    return msg


def _try_get_view_stamp_ns(view: Any) -> Optional[int]:
    for attr in (
        "stamp_ns",
        "timestamp_ns",
        "depth_stamp_ns",
        "rgb_stamp_ns",
        "stamp",
        "depth_stamp",
        "rgb_stamp",
    ):
        value = getattr(view, attr, None)
        if value is None:
            continue
        if hasattr(value, "nanoseconds"):
            return int(value.nanoseconds)
        sec = getattr(value, "sec", None)
        nanosec = getattr(value, "nanosec", None)
        if sec is not None and nanosec is not None:
            return int(sec) * 1_000_000_000 + int(nanosec)
        if isinstance(value, (int, np.integer)):
            return int(value)
        if isinstance(value, float):
            return int(value * 1e9)
    return None


def _draw_mask_overlay(
    rgb: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int],
    alpha: float = 0.35,
) -> np.ndarray:
    out = rgb.copy()
    color_arr = np.array(color, dtype=np.uint8).reshape(1, 1, 3)
    mask3 = mask.astype(bool)[..., None]
    blended = ((1.0 - alpha) * out + alpha * color_arr).astype(np.uint8)
    out = np.where(mask3, blended, out)
    return out


class LiveSAMDebug(Node):
    def __init__(self, grabber: MultiCamGrabber):
        super().__init__("live_sam_debug")
        self.grabber = grabber

        self._busy = False
        self._frame = 0
        self._last_process_wall_t = 0.0
        self._last_views_signature: Optional[Tuple[Tuple[str, int], ...]] = None

        self.sam = SAMSegmenter(
            SAMSegmenterConfig(
                repo_root="external/sam2",
                checkpoint="external/sam2/checkpoints/sam2.1_hiera_base_plus.pt",
                model_cfg="configs/sam2.1/sam2.1_hiera_b+.yaml",
                max_image_side=1024,
                min_mask_area=1500,
                min_bbox_side_px=20,
                attach_rgb_crops=False,
            )
        )

        self.pub_overlay: Dict[str, Any] = {}
        self.pub_raw: Dict[str, Any] = {}
        for c in CAMERAS:
            self.pub_overlay[c.cam_id] = self.create_publisher(
                Image, f"/perception/debug/sam_overlay/{c.cam_id}", FAST_QOS
            )
            self.pub_raw[c.cam_id] = self.create_publisher(
                Image, f"/perception/debug/rgb_raw/{c.cam_id}", FAST_QOS
            )

        self.timer = self.create_timer(TIMER_PERIOD_S, self._tick)
        self.get_logger().info("LiveSAMDebug started")

    def _views_signature(self, views: Any) -> Optional[Tuple[Tuple[str, int], ...]]:
        signature = []
        for v in views:
            stamp_ns = _try_get_view_stamp_ns(v)
            if stamp_ns is None:
                return None
            signature.append((str(v.cam_id), int(stamp_ns)))
        signature.sort(key=lambda x: x[0])
        return tuple(signature)

    def _draw_candidates(self, rgb: np.ndarray, masks, max_masks: int = MAX_MASKS_DRAW) -> np.ndarray:
        vis = rgb.copy()
        palette = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
            (255, 128, 0),
            (128, 0, 255),
        ]

        for i, cand in enumerate(masks[:max_masks]):
            color = palette[i % len(palette)]
            vis = _draw_mask_overlay(vis, cand.mask, color, alpha=0.28)

            x0, y0, x1, y1 = cand.bbox_xyxy
            cv2.rectangle(vis, (x0, y0), (x1, y1), color, 2)

            label = f"{i}: s={cand.score:.2f} a={cand.area}"
            cv2.putText(
                vis,
                label,
                (x0, max(20, y0 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

        return vis

    def _tick(self) -> None:
        now = time.perf_counter()
        if self._busy:
            return
        if now - self._last_process_wall_t < MIN_PROCESS_DT_S:
            return

        views = self.grabber.get_latest_views()
        if views is None:
            return

        signature = self._views_signature(views)
        if signature is not None and signature == self._last_views_signature:
            return

        self._busy = True
        t0 = time.perf_counter()
        try:
            self._frame += 1
            stamp = self.get_clock().now().to_msg()

            for view in views:
                if getattr(view, "rgb", None) is None:
                    continue

                t_sam0 = time.perf_counter()
                masks = self.sam.generate_auto(view.rgb)
                t_sam1 = time.perf_counter()

                overlay = self._draw_candidates(view.rgb, masks)

                self.pub_raw[view.cam_id].publish(
                    _rgb_numpy_to_imgmsg(view.rgb, frame_id=view.cam_id, stamp=stamp)
                )
                self.pub_overlay[view.cam_id].publish(
                    _rgb_numpy_to_imgmsg(overlay, frame_id=view.cam_id, stamp=stamp)
                )

                self.get_logger().info(
                    f"[{view.cam_id}] masks={len(masks)} "
                    f"sam={(t_sam1 - t_sam0) * 1000:.1f} ms"
                )

            self._last_views_signature = signature
            self._last_process_wall_t = time.perf_counter()

            if self._frame % 10 == 0:
                dt_ms = (time.perf_counter() - t0) * 1000.0
                hz = 1000.0 / max(dt_ms, 1e-6)
                self.get_logger().info(
                    f"[frame {self._frame}] total={dt_ms:.1f} ms ({hz:.1f} Hz)"
                )

        finally:
            self._busy = False


def main() -> None:
    rclpy.init()

    T_map = load_extrinsics_yaml("config/camera_extrinsics.yaml")
    grabber = MultiCamGrabber(
        cameras=CAMERAS,
        sync_slop_s=0.10,
        use_best_effort_if_unsynced=True,
        static_extrinsics_base_cam=T_map,
        rgb_depth_max_dt_s=0.08,
    )

    node = LiveSAMDebug(grabber)

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(grabber)
    executor.add_node(node)

    try:
        executor.spin()
    finally:
        executor.remove_node(node)
        executor.remove_node(grabber)
        node.destroy_node()
        grabber.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()