from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Header

from src.calibration.io_extrinsics import load_extrinsics_yaml
from src.perception.learned.DINO.dino_identifier import DINOIdentifier, DINOIdentifierConfig
from src.perception.learned.SAM.sam_segmentation import SAMSegmenter, SAMSegmenterConfig
from src.perception.ros.multicam_grabber import CameraTopics, MultiCamGrabber


FAST_QOS = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
)

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


def _draw_mask_overlay(
    rgb: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int],
    alpha: float = 0.30,
) -> np.ndarray:
    out = rgb.copy()
    color_arr = np.array(color, dtype=np.uint8).reshape(1, 1, 3)
    mask3 = mask.astype(bool)[..., None]
    blended = ((1.0 - alpha) * out + alpha * color_arr).astype(np.uint8)
    return np.where(mask3, blended, out)


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


class DINODebugNode(Node):
    def __init__(
        self,
        mode: str,
        image_root: str,
        reference_dir: str,
        use_sam: bool,
        grabber: MultiCamGrabber | None = None,
    ):
        super().__init__("dino_debug")
        self.mode = mode
        self.image_root = Path(image_root)
        self.reference_dir = reference_dir
        self.use_sam = use_sam
        self.grabber = grabber

        self.pub_overlay: Dict[str, Any] = {}
        self.pub_raw: Dict[str, Any] = {}
        self.pub_sam_masks: Dict[str, Any] = {}

        for cam in CAMERAS:
            self.pub_overlay[cam.cam_id] = self.create_publisher(
                Image,
                f"/perception/debug/{cam.cam_id}/dino_overlay",
                FAST_QOS,
            )
            self.pub_raw[cam.cam_id] = self.create_publisher(
                Image,
                f"/perception/debug/{cam.cam_id}/dino_raw",
                FAST_QOS,
            )
            self.pub_sam_masks[cam.cam_id] = self.create_publisher(
                Image,
                f"/perception/debug/{cam.cam_id}/sam_masks",
                FAST_QOS,
            )

        self.palette = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
            (255, 128, 0),
            (128, 0, 255),
        ]

        self.dino = DINOIdentifier(
            DINOIdentifierConfig(
                model_name="dinov2_vitb14",
                reference_dir=self.reference_dir,
            )
        )
        self.get_logger().info("Building DINO reference bank...")
        self.dino.build_reference_bank_from_folder()
        self.get_logger().info(f"Reference bank size: {len(self.dino.reference_bank)}")

        self.sam = None
        if self.use_sam:
            self.sam = SAMSegmenter(
                SAMSegmenterConfig(
                    repo_root="external/sam2",
                    checkpoint="external/sam2/checkpoints/sam2.1_hiera_base_plus.pt",
                    model_cfg="configs/sam2.1/sam2.1_hiera_b+.yaml",
                    max_image_side=1024,
                    min_mask_area=300,
                    min_bbox_side_px=8,
                    attach_rgb_crops=False,
                )
            )
            self.get_logger().info("SAM enabled for proposal generation")

        self._last_views_signature = None
        self._busy = False
        self._index = 0

        if self.mode == "images":
            self.image_paths = self._collect_images(self.image_root)
            if not self.image_paths:
                raise RuntimeError(f"No images found under {self.image_root}")
            self.get_logger().info(f"Loaded {len(self.image_paths)} test images")
            self.timer = self.create_timer(1.0, self._tick_images)

        elif self.mode == "live":
            if self.grabber is None:
                raise ValueError("live mode requires a MultiCamGrabber")
            self.timer = self.create_timer(0.25, self._tick_live)

        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def _collect_images(self, root: Path) -> List[Path]:
        exts = {".png", ".jpg", ".jpeg"}
        paths = []
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.suffix.lower() in exts:
                if "reference_crops" in str(p):
                    continue
                paths.append(p)
        return paths

    def _classify_single_crop(self, rgb_crop: np.ndarray):
        res = self.dino.classify_crop(rgb_crop)

        scores = res.scores_by_object
        sorted_scores = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

        best_obj, best_score = sorted_scores[0]
        second_score = sorted_scores[1][1] if len(sorted_scores) > 1 else -1.0
        margin = best_score - second_score

        if best_score < 0.6:  # or margin < 0.05:
            label = "unknown"
        else:
            label = best_obj

        return label, best_score, scores

    def _draw_whole_image_label(self, rgb: np.ndarray, label: str, score: float) -> np.ndarray:
        vis = rgb.copy()
        txt = f"{label} ({score:.2f})"
        cv2.rectangle(vis, (20, 20), (420, 80), (0, 0, 0), -1)
        cv2.putText(
            vis,
            txt,
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return vis

    def _draw_sam_only(self, rgb: np.ndarray, masks) -> np.ndarray:
        vis = rgb.copy()
        for i, cand in enumerate(masks):
            color = self.palette[i % len(self.palette)]
            vis = _draw_mask_overlay(vis, cand.mask, color, alpha=0.28)

            x0, y0, x1, y1 = cand.bbox_xyxy
            cv2.rectangle(vis, (x0, y0), (x1, y1), color, 2)

            txt = f"sam_{i}"
            cv2.putText(
                vis,
                txt,
                (x0, max(20, y0 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )
        return vis

    def _draw_sam_dino(self, rgb: np.ndarray, masks) -> np.ndarray:
        vis = rgb.copy()
        for i, cand in enumerate(masks):
            color = self.palette[i % len(self.palette)]
            x0, y0, x1, y1 = cand.bbox_xyxy
            crop = rgb[y0:y1, x0:x1]

            if crop.size == 0:
                continue

            try:
                obj_id, score, _ = self._classify_single_crop(crop)
            except Exception as e:
                self.get_logger().warn(f"DINO classify failed on mask {i}: {e}")
                continue

            if obj_id == "unknown":
                continue

            vis = _draw_mask_overlay(vis, cand.mask, color, alpha=0.22)
            cv2.rectangle(vis, (x0, y0), (x1, y1), color, 2)
            txt = f"{obj_id} {score:.2f}"
            cv2.putText(
                vis,
                txt,
                (x0, max(20, y0 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )
        return vis

    def _publish_pair(
        self,
        cam_id: str,
        rgb: np.ndarray,
        overlay: np.ndarray,
        frame_id: Optional[str] = None,
    ) -> None:
        stamp = self.get_clock().now().to_msg()
        frame_id = frame_id or cam_id

        self.pub_raw[cam_id].publish(
            _rgb_numpy_to_imgmsg(rgb, frame_id=frame_id, stamp=stamp)
        )
        self.pub_overlay[cam_id].publish(
            _rgb_numpy_to_imgmsg(overlay, frame_id=frame_id, stamp=stamp)
        )

    def _publish_sam_masks(
        self,
        cam_id: str,
        sam_vis: np.ndarray,
        frame_id: Optional[str] = None,
    ) -> None:
        stamp = self.get_clock().now().to_msg()
        frame_id = frame_id or cam_id

        self.pub_sam_masks[cam_id].publish(
            _rgb_numpy_to_imgmsg(sam_vis, frame_id=frame_id, stamp=stamp)
        )

    def _tick_images(self) -> None:
        if self._busy:
            return
        self._busy = True
        try:
            img_path = self.image_paths[self._index]
            bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if bgr is None:
                self.get_logger().warn(f"Failed to read image: {img_path}")
                self._index = (self._index + 1) % len(self.image_paths)
                return

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            if self.use_sam and self.sam is not None:
                t0 = time.perf_counter()
                masks = self.sam.generate_auto(rgb)
                t1 = time.perf_counter()

                masks_vis = self._draw_sam_only(rgb, masks[:15])
                overlay = self._draw_sam_dino(rgb, masks[:15])

                self.get_logger().info(
                    f"[images] {img_path.name} masks={len(masks)} sam={(t1 - t0) * 1000:.1f} ms"
                )
            else:
                t0 = time.perf_counter()
                obj_id, score, scores_by_object = self._classify_single_crop(rgb)
                t1 = time.perf_counter()

                overlay = self._draw_whole_image_label(rgb, obj_id, score)
                masks_vis = rgb.copy()

                self.get_logger().info(
                    f"[images] {img_path.name} pred={obj_id} score={score:.3f} "
                    f"dino={(t1 - t0) * 1000:.1f} ms all={scores_by_object}"
                )

            self._publish_pair(
                cam_id="zed2i_1",
                rgb=rgb,
                overlay=overlay,
                frame_id="image_mode",
            )
            self._publish_sam_masks(
                cam_id="zed2i_1",
                sam_vis=masks_vis,
                frame_id="image_mode",
            )

            self._index = (self._index + 1) % len(self.image_paths)

        finally:
            self._busy = False

    def _tick_live(self) -> None:
        if self._busy:
            return
        self._busy = True

        try:
            views = self.grabber.get_latest_views()
            if views is None:
                return

            signature = []
            for v in views:
                stamp_ns = _try_get_view_stamp_ns(v)
                if stamp_ns is None:
                    signature = None
                    break
                signature.append((v.cam_id, stamp_ns))

            if signature is not None:
                signature = tuple(sorted(signature))
                if signature == self._last_views_signature:
                    return
                self._last_views_signature = signature

            for v in views:
                rgb = getattr(v, "rgb", None)
                cam_id = getattr(v, "cam_id", None)

                if rgb is None or cam_id is None:
                    continue

                frame_id = cam_id

                if self.use_sam and self.sam is not None:
                    t0 = time.perf_counter()
                    masks = self.sam.generate_auto(rgb)
                    t1 = time.perf_counter()

                    masks_vis = self._draw_sam_only(rgb, masks[:15])
                    overlay = self._draw_sam_dino(rgb, masks[:15])

                    self.get_logger().info(
                        f"[live:{cam_id}] masks={len(masks)} "
                        f"sam={(t1 - t0) * 1000:.1f} ms"
                    )
                else:
                    t0 = time.perf_counter()
                    obj_id, score, scores_by_object = self._classify_single_crop(rgb)
                    t1 = time.perf_counter()

                    overlay = self._draw_whole_image_label(rgb, obj_id, score)
                    masks_vis = rgb.copy()

                    self.get_logger().info(
                        f"[live:{cam_id}] pred={obj_id} score={score:.3f} "
                        f"dino={(t1 - t0) * 1000:.1f} ms all={scores_by_object}"
                    )

                self._publish_pair(
                    cam_id=cam_id,
                    rgb=rgb,
                    overlay=overlay,
                    frame_id=frame_id,
                )
                self._publish_sam_masks(
                    cam_id=cam_id,
                    sam_vis=masks_vis,
                    frame_id=frame_id,
                )

        finally:
            self._busy = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["images", "live"], default="images")
    p.add_argument("--image-root", default="/home/moreno/MasterThesis/Data/ZED_screens")
    p.add_argument("--reference-dir", default="Data/ZED_screens")
    p.add_argument("--use-sam", type=int, default=1, help="1=use SAM proposals, 0=classify whole image")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    rclpy.init()

    grabber = None
    executor = MultiThreadedExecutor(num_threads=4)

    if args.mode == "live":
        T_map = load_extrinsics_yaml("config/camera_extrinsics.yaml")
        grabber = MultiCamGrabber(
            cameras=CAMERAS,
            sync_slop_s=0.10,
            use_best_effort_if_unsynced=True,
            static_extrinsics_base_cam=T_map,
            rgb_depth_max_dt_s=0.08,
        )
        executor.add_node(grabber)

    node = DINODebugNode(
        mode=args.mode,
        image_root=args.image_root,
        reference_dir=args.reference_dir,
        use_sam=bool(args.use_sam),
        grabber=grabber,
    )
    executor.add_node(node)

    try:
        executor.spin()
    finally:
        executor.remove_node(node)
        node.destroy_node()
        if grabber is not None:
            executor.remove_node(grabber)
            grabber.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()