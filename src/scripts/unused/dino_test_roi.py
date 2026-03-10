from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import rclpy
import yaml
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Header

from src.calibration.io_extrinsics import load_extrinsics_yaml
from src.perception.learned.DINO.dino_identifier import DINOIdentifier, DINOIdentifierConfig
from src.perception.learned.SAM.sam_segmentation import SAMMaskCandidate, SAMSegmenter, SAMSegmenterConfig
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

TABLE_PLANE_YAML = "config/table_plane.yaml"
TABLE_ROI_YAML = "config/table_roi.yaml"

# Only used if table_roi.yaml is missing for a camera.
FALLBACK_TABLE_ROIS: Dict[str, Optional[Tuple[int, int, int, int]]] = {
    "zed2i_1": None,
    "zed2i_2": None,
}


def _as_contig_rgb(img: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(img, dtype=np.uint8).copy())


def _rgb_numpy_to_imgmsg(rgb: np.ndarray, frame_id: str, stamp) -> Image:
    rgb = _as_contig_rgb(rgb)
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
    out = _as_contig_rgb(rgb)
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
        "stamp_s",
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


def _bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0 or len(ys) == 0:
        return (0, 0, 0, 0)
    x0 = int(xs.min())
    y0 = int(ys.min())
    x1 = int(xs.max()) + 1
    y1 = int(ys.max()) + 1
    return (x0, y0, x1, y1)


def _clip_roi_to_image(
    roi: Optional[Tuple[int, int, int, int]],
    h: int,
    w: int,
) -> Optional[Tuple[int, int, int, int]]:
    if roi is None:
        return None
    x0, y0, x1, y1 = roi
    x0 = max(0, min(w, int(x0)))
    x1 = max(0, min(w, int(x1)))
    y0 = max(0, min(h, int(y0)))
    y1 = max(0, min(h, int(y1)))
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _shift_candidate_from_crop(
    cand: SAMMaskCandidate,
    x_off: int,
    y_off: int,
    full_h: int,
    full_w: int,
) -> SAMMaskCandidate:
    full_mask = np.zeros((full_h, full_w), dtype=bool)
    h, w = cand.mask.shape
    full_mask[y_off:y_off + h, x_off:x_off + w] = cand.mask

    x0, y0, x1, y1 = cand.bbox_xyxy
    bbox = (x0 + x_off, y0 + y_off, x1 + x_off, y1 + y_off)

    crop_rgb = None
    if cand.crop_rgb is not None:
        crop_rgb = _as_contig_rgb(cand.crop_rgb)

    return SAMMaskCandidate(
        mask=full_mask,
        score=float(cand.score),
        bbox_xyxy=bbox,
        area=int(cand.area),
        crop_rgb=crop_rgb,
    )


def _extract_transform_matrix(T_obj: Any) -> np.ndarray:
    if T_obj is None:
        return np.eye(4, dtype=np.float64)

    for attr in ("as_matrix", "matrix"):
        fn = getattr(T_obj, attr, None)
        if callable(fn):
            M = np.asarray(fn(), dtype=np.float64)
            if M.shape == (4, 4):
                return M

    for attr in ("T", "mat", "M", "_T"):
        M = getattr(T_obj, attr, None)
        if M is not None:
            M = np.asarray(M, dtype=np.float64)
            if M.shape == (4, 4):
                return M

    R = None
    t = None
    for r_attr in ("R", "rot", "rotation"):
        val = getattr(T_obj, r_attr, None)
        if val is not None:
            arr = np.asarray(val, dtype=np.float64)
            if arr.shape == (3, 3):
                R = arr
                break
    for t_attr in ("t", "trans", "translation"):
        val = getattr(T_obj, t_attr, None)
        if val is not None:
            arr = np.asarray(val, dtype=np.float64).reshape(-1)
            if arr.size == 3:
                t = arr
                break

    if R is not None and t is not None:
        M = np.eye(4, dtype=np.float64)
        M[:3, :3] = R
        M[:3, 3] = t
        return M

    return np.eye(4, dtype=np.float64)


def _transform_points(points: np.ndarray, T_obj: Any) -> np.ndarray:
    if points.size == 0:
        return points.copy()
    M = _extract_transform_matrix(T_obj)
    R = M[:3, :3]
    t = M[:3, 3]
    return points @ R.T + t[None, :]


def _depth_mask_to_points_cam(
    depth_m: np.ndarray,
    K: np.ndarray,
    mask: np.ndarray,
    zmin: float = 0.15,
    zmax: float = 2.0,
) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.empty((0, 3), dtype=np.float32)

    z = depth_m[ys, xs].astype(np.float32)
    valid = np.isfinite(z) & (z > zmin) & (z < zmax)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32)

    xs = xs[valid].astype(np.float32)
    ys = ys[valid].astype(np.float32)
    z = z[valid]

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy
    return np.stack([x, y, z], axis=1)


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = float(np.logical_and(a, b).sum())
    if inter <= 0.0:
        return 0.0
    union = float(np.logical_or(a, b).sum()) + 1e-9
    return inter / union


def _load_table_plane_yaml(path: str) -> Optional[np.ndarray]:
    p = Path(path)
    if not p.exists():
        return None
    with open(p, "r") as f:
        data = yaml.safe_load(f)

    if data is None:
        return None

    n = np.asarray(data["normal"], dtype=np.float64).reshape(3)
    d = float(data["d"])
    n = n / (np.linalg.norm(n) + 1e-12)

    if n[2] < 0.0:
        n = -n
        d = -d

    return np.array([n[0], n[1], n[2], d], dtype=np.float64)


def _load_table_rois_yaml(path: str) -> Dict[str, Tuple[int, int, int, int]]:
    p = Path(path)
    if not p.exists():
        return {}

    with open(p, "r") as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}

    out: Dict[str, Tuple[int, int, int, int]] = {}
    for cam_id, cfg in data.items():
        if cfg is None:
            continue
        out[cam_id] = (
            int(cfg["x0"]),
            int(cfg["y0"]),
            int(cfg["x1"]),
            int(cfg["y1"]),
        )
    return out


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

        self.table_plane = _load_table_plane_yaml(TABLE_PLANE_YAML)
        self.table_rois = _load_table_rois_yaml(TABLE_ROI_YAML)

        self.pub_overlay: Dict[str, Any] = {}
        self.pub_raw: Dict[str, Any] = {}
        self.pub_sam_masks: Dict[str, Any] = {}
        self.pub_table_mask: Dict[str, Any] = {}

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
            self.pub_table_mask[cam.cam_id] = self.create_publisher(
                Image,
                f"/perception/debug/{cam.cam_id}/table_mask",
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
            (180, 255, 0),
            (255, 180, 0),
            (100, 100, 255),
            (255, 100, 100),
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
                    max_image_side=896,
                    min_mask_area=120,
                    min_bbox_side_px=6,
                    max_mask_area_ratio=0.95,
                    attach_rgb_crops=False,
                    auto_points_per_side=16,
                    auto_pred_iou_thresh=0.86,
                    auto_stability_score_thresh=0.90,
                    auto_crop_n_layers=0,
                    auto_crop_n_points_downscale_factor=1,
                    auto_min_mask_region_area=60,
                )
            )
            self.get_logger().info("SAM enabled for proposal generation")

        if self.table_plane is not None:
            self.get_logger().info(
                f"Loaded table plane: n={self.table_plane[:3].tolist()} d={float(self.table_plane[3]):.4f}"
            )
        else:
            self.get_logger().warn("No table_plane.yaml found; plane filtering disabled")

        if len(self.table_rois) == 0:
            self.get_logger().warn("No table_roi.yaml found; falling back to full image or FALLBACK_TABLE_ROIS")

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
            self.timer = self.create_timer(0.35, self._tick_live)
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

    def _get_roi_for_cam(
        self,
        cam_id: str,
        h: int,
        w: int,
    ) -> Optional[Tuple[int, int, int, int]]:
        roi = self.table_rois.get(cam_id, FALLBACK_TABLE_ROIS.get(cam_id))
        return _clip_roi_to_image(roi, h, w)

    def _classify_single_crop(self, rgb_crop: np.ndarray):
        rgb_crop = _as_contig_rgb(rgb_crop)
        res = self.dino.classify_crop(rgb_crop)
        scores = res.scores_by_object
        sorted_scores = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

        best_obj, best_score = sorted_scores[0]
        _second_score = sorted_scores[1][1] if len(sorted_scores) > 1 else -1.0

        if best_score < 0.6:
            label = "unknown"
        else:
            label = best_obj

        return label, best_score, scores

    def _masked_tight_crop(
        self,
        rgb: np.ndarray,
        mask: np.ndarray,
        pad: int = 4,
    ) -> Optional[np.ndarray]:
        if mask is None or not np.any(mask):
            return None

        x0, y0, x1, y1 = _bbox_from_mask(mask)
        H, W = mask.shape
        x0 = max(0, x0 - pad)
        y0 = max(0, y0 - pad)
        x1 = min(W, x1 + pad)
        y1 = min(H, y1 + pad)

        crop = _as_contig_rgb(rgb[y0:y1, x0:x1])
        crop_mask = mask[y0:y1, x0:x1]

        if crop.size == 0 or not np.any(crop_mask):
            return None

        out = np.zeros_like(crop)
        out[crop_mask] = crop[crop_mask]

        h, w = out.shape[:2]
        if min(h, w) < 64:
            scale = max(1.0, 96.0 / max(1, min(h, w)))
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            out = cv2.resize(out, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        return _as_contig_rgb(out)

    def _draw_whole_image_label(self, rgb: np.ndarray, label: str, score: float) -> np.ndarray:
        vis = _as_contig_rgb(rgb)
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

    def _draw_table_mask(
        self,
        rgb: np.ndarray,
        roi: Optional[Tuple[int, int, int, int]],
    ) -> np.ndarray:
        vis = _as_contig_rgb(rgb)
        if roi is not None:
            x0, y0, x1, y1 = roi
            cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 255, 0), 2)
            cv2.putText(
                vis,
                "table_roi",
                (x0, max(20, y0 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )
        return vis

    def _draw_sam_only(self, rgb: np.ndarray, masks: List[SAMMaskCandidate]) -> np.ndarray:
        vis = _as_contig_rgb(rgb)
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

    def _draw_sam_dino(self, rgb: np.ndarray, masks: List[SAMMaskCandidate]) -> np.ndarray:
        vis = _as_contig_rgb(rgb)
        for i, cand in enumerate(masks):
            color = self.palette[i % len(self.palette)]

            crop = self._masked_tight_crop(rgb, cand.mask)
            if crop is None or crop.size == 0:
                continue

            try:
                obj_id, score, _ = self._classify_single_crop(crop)
            except Exception as e:
                self.get_logger().warn(f"DINO classify failed on mask {i}: {e}")
                continue

            if obj_id == "unknown":
                continue

            x0, y0, x1, y1 = cand.bbox_xyxy
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

    def _publish_table_mask(
        self,
        cam_id: str,
        table_vis: np.ndarray,
        frame_id: Optional[str] = None,
    ) -> None:
        stamp = self.get_clock().now().to_msg()
        frame_id = frame_id or cam_id
        self.pub_table_mask[cam_id].publish(
            _rgb_numpy_to_imgmsg(table_vis, frame_id=frame_id, stamp=stamp)
        )

    def _filter_masks_with_plane(
        self,
        masks: List[SAMMaskCandidate],
        depth_m: np.ndarray,
        K: np.ndarray,
        T_base_cam: Any,
        plane_model: Optional[np.ndarray],
    ) -> List[SAMMaskCandidate]:
        if plane_model is None:
            return masks

        out: List[SAMMaskCandidate] = []

        for cand in masks:
            pts_cam = _depth_mask_to_points_cam(depth_m, K, cand.mask, zmin=0.15, zmax=2.0)
            if len(pts_cam) < 30:
                continue

            pts_base = _transform_points(pts_cam, T_base_cam)
            d = pts_base @ plane_model[:3] + plane_model[3]
            d = d[np.isfinite(d)]
            if d.size < 30:
                continue

            plane_overlap = float(np.mean(np.abs(d) < 0.008))
            q80 = float(np.quantile(d, 0.80))
            q95 = float(np.quantile(d, 0.95))
            peak = float(np.max(d))

            x0, y0, x1, y1 = cand.bbox_xyxy
            bw = max(1, x1 - x0)
            bh = max(1, y1 - y0)
            aspect = float(max(bw, bh)) / float(min(bw, bh))

            keep = False

            if plane_overlap < 0.90 and q80 > 0.002:
                keep = True

            if aspect > 3.5 and q95 > 0.004:
                keep = True

            if peak > 0.006:
                keep = True

            if keep:
                out.append(cand)

        out.sort(key=lambda x: (x.score, x.area), reverse=True)

        dedup: List[SAMMaskCandidate] = []
        for cand in out:
            duplicate = False
            for kept in dedup:
                if _mask_iou(cand.mask, kept.mask) > 0.75:
                    duplicate = True
                    break
            if not duplicate:
                dedup.append(cand)

        return dedup

    def _run_sam_on_roi_and_plane(
        self,
        rgb: np.ndarray,
        depth_m: np.ndarray,
        K: np.ndarray,
        T_base_cam: Any,
        cam_id: str,
        max_masks: int = 10,
    ) -> Tuple[List[SAMMaskCandidate], Optional[Tuple[int, int, int, int]], bool]:
        rgb = _as_contig_rgb(rgb)
        H, W = rgb.shape[:2]
        roi = self._get_roi_for_cam(cam_id, H, W)

        plane_model_for_filter = self.table_plane
        plane_filter_enabled = True
        if depth_m.shape[:2] != rgb.shape[:2]:
            plane_model_for_filter = None
            plane_filter_enabled = False

        if roi is None:
            rgb_in = _as_contig_rgb(rgb)
            masks = self.sam.generate_auto(rgb_in)
            masks = masks[: max_masks * 2]
            masks = self._filter_masks_with_plane(masks, depth_m, K, T_base_cam, plane_model_for_filter)
            masks = masks[:max_masks]
            return masks, None, plane_filter_enabled

        x0, y0, x1, y1 = roi
        rgb_crop = _as_contig_rgb(rgb[y0:y1, x0:x1])
        if rgb_crop.size == 0:
            return [], roi, plane_filter_enabled

        crop_masks = self.sam.generate_auto(rgb_crop)
        crop_masks = crop_masks[: max_masks * 2]

        full_masks = [
            _shift_candidate_from_crop(c, x0, y0, H, W)
            for c in crop_masks
        ]

        full_masks = self._filter_masks_with_plane(
            full_masks, depth_m, K, T_base_cam, plane_model_for_filter
        )
        full_masks = full_masks[:max_masks]
        return full_masks, roi, plane_filter_enabled

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

            rgb = _as_contig_rgb(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

            if self.use_sam and self.sam is not None:
                t0 = time.perf_counter()
                masks = self.sam.generate_auto(_as_contig_rgb(rgb))
                masks = masks[:10]
                t1 = time.perf_counter()

                masks_vis = self._draw_sam_only(rgb, masks)
                overlay = self._draw_sam_dino(rgb, masks)
                table_vis = rgb.copy()

                self.get_logger().info(
                    f"[images] {img_path.name} masks={len(masks)} sam={(t1 - t0) * 1000:.1f} ms"
                )
            else:
                t0 = time.perf_counter()
                obj_id, score, scores_by_object = self._classify_single_crop(rgb)
                t1 = time.perf_counter()

                overlay = self._draw_whole_image_label(rgb, obj_id, score)
                masks_vis = rgb.copy()
                table_vis = rgb.copy()

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
            self._publish_table_mask(
                cam_id="zed2i_1",
                table_vis=table_vis,
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
                depth_m = getattr(v, "depth", None)
                K = getattr(v, "K", None)
                T_base_cam = getattr(v, "T_base_cam", None)
                cam_id = getattr(v, "cam_id", None)

                if rgb is None or depth_m is None or K is None or cam_id is None:
                    continue

                rgb = _as_contig_rgb(rgb)
                frame_id = cam_id

                if self.use_sam and self.sam is not None:
                    t0 = time.perf_counter()
                    masks, roi, plane_filter_enabled = self._run_sam_on_roi_and_plane(
                        rgb=rgb,
                        depth_m=depth_m,
                        K=K,
                        T_base_cam=T_base_cam,
                        cam_id=cam_id,
                        max_masks=10,
                    )
                    t1 = time.perf_counter()

                    masks_vis = self._draw_sam_only(rgb, masks)
                    overlay = self._draw_sam_dino(rgb, masks)
                    table_vis = self._draw_table_mask(rgb, roi)

                    self.get_logger().info(
                        f"[live:{cam_id}] masks={len(masks)} "
                        f"sam+plane={(t1 - t0) * 1000:.1f} ms "
                        f"roi={'full' if roi is None else roi} "
                        f"plane_filter={'on' if plane_filter_enabled else 'off_shape_mismatch'}"
                    )
                else:
                    t0 = time.perf_counter()
                    obj_id, score, scores_by_object = self._classify_single_crop(rgb)
                    t1 = time.perf_counter()

                    overlay = self._draw_whole_image_label(rgb, obj_id, score)
                    masks_vis = rgb.copy()
                    table_vis = rgb.copy()

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
                self._publish_table_mask(
                    cam_id=cam_id,
                    table_vis=table_vis,
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