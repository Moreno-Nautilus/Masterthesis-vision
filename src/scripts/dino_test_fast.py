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


def _bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0 or len(ys) == 0:
        return (0, 0, 0, 0)
    x0 = int(xs.min())
    y0 = int(ys.min())
    x1 = int(xs.max()) + 1
    y1 = int(ys.max()) + 1
    return (x0, y0, x1, y1)


def _masked_tight_crop(
    rgb: np.ndarray,
    mask: np.ndarray,
    pad: int = 4,
    min_side_after_resize: int = 96,
) -> Optional[np.ndarray]:
    if mask is None or not np.any(mask):
        return None

    x0, y0, x1, y1 = _bbox_from_mask(mask)
    h_img, w_img = mask.shape

    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(w_img, x1 + pad)
    y1 = min(h_img, y1 + pad)

    crop = rgb[y0:y1, x0:x1].copy()
    crop_mask = mask[y0:y1, x0:x1]

    if crop.size == 0 or not np.any(crop_mask):
        return None

    out = np.zeros_like(crop)
    out[crop_mask] = crop[crop_mask]

    h, w = out.shape[:2]
    min_side = min(h, w)
    if min_side > 0 and min_side < min_side_after_resize:
        scale = float(min_side_after_resize) / float(min_side)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        out = cv2.resize(out, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    return np.ascontiguousarray(out)


def _reject_border_masks(
    masks,
    image_h,
    image_w,
    border_px: int = 8,
    max_border_fraction: float = 0.04,
):
    filtered = []

    for cand in masks:
        mask = cand.mask

        border = np.zeros_like(mask, dtype=bool)
        border[:border_px, :] = True
        border[-border_px:, :] = True
        border[:, :border_px] = True
        border[:, -border_px:] = True

        border_overlap = np.logical_and(mask, border).sum()
        mask_area = mask.sum()

        if mask_area == 0:
            continue

        frac = border_overlap / float(mask_area)
        if frac > max_border_fraction:
            continue

        filtered.append(cand)

    return filtered


def _reject_large_masks(
    masks,
    image_h,
    image_w,
    max_mask_area_ratio: float = 0.14,
    max_bbox_area_ratio: float = 0.18,
):
    image_area = float(image_h * image_w)
    filtered = []

    for cand in masks:
        x0, y0, x1, y1 = cand.bbox_xyxy
        bbox_area = float((x1 - x0) * (y1 - y0))
        mask_area = float(cand.area)

        if mask_area / image_area > max_mask_area_ratio:
            continue

        if bbox_area / image_area > max_bbox_area_ratio:
            continue

        filtered.append(cand)

    return filtered


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


def _resize_keep_aspect_by_width(
    rgb: np.ndarray,
    target_w: int,
) -> Tuple[np.ndarray, float]:
    h, w = rgb.shape[:2]
    if w <= 0:
        return rgb, 1.0
    if w == target_w:
        return rgb, 1.0

    scale = float(target_w) / float(w)
    new_h = max(1, int(round(h * scale)))
    rgb_small = cv2.resize(rgb, (target_w, new_h), interpolation=cv2.INTER_LINEAR)
    return rgb_small, scale


def _scale_mask_to_fullres(mask_small: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    mask_u8 = (mask_small.astype(np.uint8) * 255)
    mask_full = cv2.resize(mask_u8, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    return mask_full > 0


def _scale_bbox_xyxy(
    bbox_xyxy: Tuple[int, int, int, int],
    inv_scale: float,
) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox_xyxy
    return (
        int(round(x0 * inv_scale)),
        int(round(y0 * inv_scale)),
        int(round(x1 * inv_scale)),
        int(round(y1 * inv_scale)),
    )


def _crop_rgb_with_roi(rgb: np.ndarray, roi_xyxy: Tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = roi_xyxy
    return rgb[y0:y1, x0:x1].copy()


def _shift_bbox_xyxy(
    bbox_xyxy: Tuple[int, int, int, int],
    dx: int,
    dy: int,
) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox_xyxy
    return (x0 + dx, y0 + dy, x1 + dx, y1 + dy)


def _paste_roi_mask_into_full(
    roi_mask: np.ndarray,
    full_h: int,
    full_w: int,
    roi_xyxy: Tuple[int, int, int, int],
) -> np.ndarray:
    x0, y0, x1, y1 = roi_xyxy
    full = np.zeros((full_h, full_w), dtype=bool)
    full[y0:y1, x0:x1] = roi_mask
    return full


def _clip_polygon_to_image(poly: np.ndarray, image_h: int, image_w: int) -> np.ndarray:
    poly = poly.astype(np.int32).copy()
    poly[:, 0] = np.clip(poly[:, 0], 0, image_w - 1)
    poly[:, 1] = np.clip(poly[:, 1], 0, image_h - 1)
    return poly


def _bbox_from_polygon(poly: np.ndarray) -> Tuple[int, int, int, int]:
    x0 = int(np.min(poly[:, 0]))
    y0 = int(np.min(poly[:, 1]))
    x1 = int(np.max(poly[:, 0])) + 1
    y1 = int(np.max(poly[:, 1])) + 1
    return (x0, y0, x1, y1)


def _shift_polygon(poly: np.ndarray, dx: int, dy: int) -> np.ndarray:
    out = poly.copy()
    out[:, 0] += dx
    out[:, 1] += dy
    return out


def _make_polygon_mask(
    image_h: int,
    image_w: int,
    poly: np.ndarray,
) -> np.ndarray:
    mask = np.zeros((image_h, image_w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, poly.astype(np.int32), 255)
    return mask > 0


def _filter_masks_by_polygon_overlap(
    masks,
    polygon_mask_full: np.ndarray,
    min_inside_fraction: float = 0.65,
):
    filtered = []
    for cand in masks:
        mask = cand.mask
        area = float(mask.sum())
        if area <= 0:
            continue

        inside = float(np.logical_and(mask, polygon_mask_full).sum())
        frac_inside = inside / area

        if frac_inside >= min_inside_fraction:
            filtered.append(cand)

    return filtered


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

        # Speed knobs
        self.sam_input_width = 640
        self.max_masks_early = 10
        self.max_masks_for_dino = 4
        self.max_masks_for_sam_vis = 8
        self.sam_every_n_ticks = 4

        # Polygon workspace ROI per camera (FULL IMAGE PIXELS)
        # Tune these in Foxglove.
        self.workspace_poly_by_cam: Dict[str, np.ndarray] = {
            "zed2i_1": np.array(
                [
                    [10, 500],
                    [1500, 40],
                    [1900, 1100],
                    [250, 1000],
                ],
                dtype=np.int32,
            ),
            # to be edited
            "zed2i_2": np.array(
                [
                    [220, 210],
                    [1030, 170],
                    [1180, 860],
                    [250, 920],
                ],
                dtype=np.int32,
            ),
        }

        self._live_tick_by_cam: Dict[str, int] = {}
        self._cache_overlay_by_cam: Dict[str, np.ndarray] = {}
        self._cache_sam_vis_by_cam: Dict[str, np.ndarray] = {}

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
                    max_image_side=640,
                    min_mask_area=300,
                    min_bbox_side_px=8,
                    attach_rgb_crops=False,
                )
            )

            try:
                if hasattr(self.sam, "config") and hasattr(self.sam.config, "auto_points_per_side"):
                    self.sam.config.auto_points_per_side = 10
                if hasattr(self.sam, "auto_points_per_side"):
                    self.sam.auto_points_per_side = 10
            except Exception:
                pass

            self.get_logger().info("SAM enabled for proposal generation")
            self.get_logger().info(
                f"SAM config: sam_input_width={self.sam_input_width}, "
                f"max_masks_early={self.max_masks_early}, "
                f"max_masks_for_dino={self.max_masks_for_dino}, "
                f"sam_every_n_ticks={self.sam_every_n_ticks}"
            )

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

        if best_score < 0.6:
            label = "unknown"
        else:
            label = best_obj

        return label, best_score, scores

    def _get_workspace_polygon(self, rgb: np.ndarray, cam_id: str) -> np.ndarray:
        h, w = rgb.shape[:2]
        poly = self.workspace_poly_by_cam.get(cam_id, None)
        if poly is None:
            poly = np.array(
                [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
                dtype=np.int32,
            )
        return _clip_polygon_to_image(poly, h, w)

    def _draw_workspace_polygon(
        self,
        rgb: np.ndarray,
        poly: np.ndarray,
        cam_id: str,
        color: Tuple[int, int, int] = (255, 255, 255),
    ) -> np.ndarray:
        vis = rgb.copy()

        cv2.polylines(
            vis,
            [poly.reshape(-1, 1, 2)],
            isClosed=True,
            color=color,
            thickness=2,
        )

        x0, y0, _, _ = _bbox_from_polygon(poly)
        label = f"ROI {cam_id}"

        (tw, th), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            2,
        )
        box_y0 = max(0, y0 - th - baseline - 8)
        box_y1 = y0
        box_x1 = min(vis.shape[1], x0 + tw + 12)

        cv2.rectangle(vis, (x0, box_y0), (box_x1, box_y1), color, -1)
        cv2.putText(
            vis,
            label,
            (x0 + 6, box_y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        return vis

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

    def _draw_sam_dino_preds(self, rgb: np.ndarray, preds: List[Dict[str, Any]]) -> np.ndarray:
        vis = rgb.copy()
        for j, pred in enumerate(preds):
            if pred["obj_id"] == "unknown":
                continue

            color = self.palette[j % len(self.palette)]
            x0, y0, x1, y1 = pred["bbox_xyxy"]

            vis = _draw_mask_overlay(vis, pred["mask"], color, alpha=0.22)
            cv2.rectangle(vis, (x0, y0), (x1, y1), color, 2)

            txt = f'{pred["obj_id"]} {pred["score"]:.2f}'
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

    def _generate_sam_candidates_in_roi(
        self,
        rgb: np.ndarray,
        cam_id: str,
    ):
        if self.sam is None:
            return []

        h_full, w_full = rgb.shape[:2]

        poly_full = self._get_workspace_polygon(rgb, cam_id)
        x0_roi, y0_roi, x1_roi, y1_roi = _bbox_from_polygon(poly_full)
        roi_xyxy = (x0_roi, y0_roi, x1_roi, y1_roi)

        rgb_roi = _crop_rgb_with_roi(rgb, roi_xyxy)
        if rgb_roi.size == 0:
            return []

        h_roi, w_roi = rgb_roi.shape[:2]

        poly_roi = _shift_polygon(poly_full, -x0_roi, -y0_roi)
        polygon_mask_roi = _make_polygon_mask(h_roi, w_roi, poly_roi)
        polygon_mask_full = _paste_roi_mask_into_full(
            polygon_mask_roi,
            h_full,
            w_full,
            roi_xyxy,
        )

        rgb_small, scale = _resize_keep_aspect_by_width(rgb_roi, target_w=self.sam_input_width)
        inv_scale = 1.0 / scale

        masks_small = self.sam.generate_auto(rgb_small)
        masks_small = masks_small[: self.max_masks_early]

        masks = []
        for cand in masks_small:
            try:
                roi_mask = _scale_mask_to_fullres(cand.mask, h_roi, w_roi)

                # Keep only the part inside the workspace polygon
                roi_mask = np.logical_and(roi_mask, polygon_mask_roi)
                if roi_mask.sum() == 0:
                    continue

                full_mask = _paste_roi_mask_into_full(roi_mask, h_full, w_full, roi_xyxy)

                bbox_roi = _bbox_from_mask(roi_mask)
                bbox_full = _shift_bbox_xyxy(bbox_roi, x0_roi, y0_roi)

                cand.mask = full_mask
                cand.bbox_xyxy = bbox_full
                cand.area = int(full_mask.sum())
            except Exception:
                continue

            masks.append(cand)

        masks = _filter_masks_by_polygon_overlap(
            masks,
            polygon_mask_full=polygon_mask_full,
            min_inside_fraction=0.65,
        )
        masks = _reject_large_masks(masks, h_full, w_full)
        masks = _reject_border_masks(masks, h_full, w_full)

        return masks

    def _predict_mask_labels(
        self,
        rgb: np.ndarray,
        masks,
        max_masks: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        preds: List[Dict[str, Any]] = []
        max_masks = self.max_masks_for_dino if max_masks is None else max_masks

        for i, cand in enumerate(masks[:max_masks]):
            crop = _masked_tight_crop(rgb, cand.mask)
            if crop is None or crop.size == 0:
                continue

            try:
                obj_id, score, scores = self._classify_single_crop(crop)
            except Exception as e:
                self.get_logger().warn(f"DINO classify failed on mask {i}: {e}")
                continue

            preds.append(
                {
                    "index": i,
                    "mask": cand.mask,
                    "bbox_xyxy": cand.bbox_xyxy,
                    "obj_id": obj_id,
                    "score": score,
                    "scores": scores,
                }
            )

        return preds

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
            cam_id = "zed2i_1"
            poly = self._get_workspace_polygon(rgb, cam_id)

            if self.use_sam and self.sam is not None:
                t0 = time.perf_counter()
                masks = self._generate_sam_candidates_in_roi(rgb, cam_id=cam_id)
                t1 = time.perf_counter()

                preds = self._predict_mask_labels(rgb, masks, max_masks=self.max_masks_for_dino)
                t2 = time.perf_counter()

                masks_vis = self._draw_sam_only(rgb, masks[: self.max_masks_for_sam_vis])
                overlay = self._draw_sam_dino_preds(rgb, preds)

                masks_vis = self._draw_workspace_polygon(masks_vis, poly, cam_id)
                overlay = self._draw_workspace_polygon(overlay, poly, cam_id)

                self.get_logger().info(
                    f"[images] {img_path.name} masks={len(masks)} preds={len(preds)} "
                    f"sam={(t1 - t0) * 1000:.1f} ms "
                    f"dino={(t2 - t1) * 1000:.1f} ms "
                    f"total={(t2 - t0) * 1000:.1f} ms"
                )
            else:
                t0 = time.perf_counter()
                obj_id, score, scores_by_object = self._classify_single_crop(rgb)
                t1 = time.perf_counter()

                overlay = self._draw_whole_image_label(rgb, obj_id, score)
                masks_vis = rgb.copy()

                masks_vis = self._draw_workspace_polygon(masks_vis, poly, cam_id)
                overlay = self._draw_workspace_polygon(overlay, poly, cam_id)

                self.get_logger().info(
                    f"[images] {img_path.name} pred={obj_id} score={score:.3f} "
                    f"dino={(t1 - t0) * 1000:.1f} ms all={scores_by_object}"
                )

            self._publish_pair(
                cam_id=cam_id,
                rgb=rgb,
                overlay=overlay,
                frame_id="image_mode",
            )
            self._publish_sam_masks(
                cam_id=cam_id,
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
                poly = self._get_workspace_polygon(rgb, cam_id)

                if self.use_sam and self.sam is not None:
                    tick = self._live_tick_by_cam.get(cam_id, 0) + 1
                    self._live_tick_by_cam[cam_id] = tick

                    should_run_sam = (
                        cam_id not in self._cache_overlay_by_cam
                        or cam_id not in self._cache_sam_vis_by_cam
                        or (tick % self.sam_every_n_ticks == 0)
                    )

                    if should_run_sam:
                        t0 = time.perf_counter()
                        masks = self._generate_sam_candidates_in_roi(rgb, cam_id=cam_id)
                        t1 = time.perf_counter()

                        preds = self._predict_mask_labels(rgb, masks, max_masks=self.max_masks_for_dino)
                        t2 = time.perf_counter()

                        masks_vis = self._draw_sam_only(rgb, masks[: self.max_masks_for_sam_vis])
                        overlay = self._draw_sam_dino_preds(rgb, preds)

                        masks_vis = self._draw_workspace_polygon(masks_vis, poly, cam_id)
                        overlay = self._draw_workspace_polygon(overlay, poly, cam_id)

                        self._cache_overlay_by_cam[cam_id] = overlay.copy()
                        self._cache_sam_vis_by_cam[cam_id] = masks_vis.copy()

                        self.get_logger().info(
                            f"[live:{cam_id}] tick={tick} ran_sam=1 masks={len(masks)} preds={len(preds)} "
                            f"sam={(t1 - t0) * 1000:.1f} ms "
                            f"dino={(t2 - t1) * 1000:.1f} ms "
                            f"total={(t2 - t0) * 1000:.1f} ms"
                        )
                    else:
                        overlay = self._cache_overlay_by_cam[cam_id].copy()
                        masks_vis = self._cache_sam_vis_by_cam[cam_id].copy()

                        overlay = self._draw_workspace_polygon(overlay, poly, cam_id)
                        masks_vis = self._draw_workspace_polygon(masks_vis, poly, cam_id)

                        self.get_logger().info(
                            f"[live:{cam_id}] tick={tick} ran_sam=0 reused_cached=1"
                        )

                else:
                    t0 = time.perf_counter()
                    obj_id, score, scores_by_object = self._classify_single_crop(rgb)
                    t1 = time.perf_counter()

                    overlay = self._draw_whole_image_label(rgb, obj_id, score)
                    masks_vis = rgb.copy()

                    overlay = self._draw_workspace_polygon(overlay, poly, cam_id)
                    masks_vis = self._draw_workspace_polygon(masks_vis, poly, cam_id)

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