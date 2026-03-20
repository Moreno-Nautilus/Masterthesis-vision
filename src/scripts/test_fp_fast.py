"""
Learned perception pipeline: SAM → DINO → FoundationPose
All outputs published as ROS topics for Foxglove visualization.
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import csv
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Header

from src.calibration.io_extrinsics import load_extrinsics_yaml
from src.perception.learned.DINO.dino_identifier import (
    DINOIdentifier,
    DINOIdentifierConfig,
    DINOResult,
)
from src.perception.learned.FP.pose_foundation_fast import (
    FoundationPoseConfig,
    FoundationPoseWrapper,
)
from src.perception.learned.SAM.sam_segmentation import (
    SAMMaskCandidate,
    SAMSegmenter,
    SAMSegmenterConfig,
)
from src.perception.ros.multicam_grabber import CameraTopics, MultiCamGrabber


FAST_QOS = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
)

CAMERAS = [
    CameraTopics(
        cam_id="zed2i_2",
        depth_topic="/zed2i_2/zed_node/depth/depth_registered",
        info_topic="/zed2i_2/zed_node/depth/depth_registered/camera_info",
        rgb_topic="/zed2i_2/zed_node/rgb/color/rect/image",
        rgb_info_topic="/zed2i_2/zed_node/rgb/color/rect/image/camera_info",
    ),
]


@dataclass
class CandidateSelection:
    object_id: str
    score: float
    scores_by_object: dict[str, float]
    candidate: SAMMaskCandidate


@dataclass
class ObjectTrackState:
    """Per-object tracking state (one per detected object per camera)."""
    object_id: str
    mesh_path: str
    fp_tracker: Optional[FoundationPoseWrapper] = None
    mode: str = "search"  # "search" | "track"
    T_object_camera: Optional[np.ndarray] = None
    dino_score: float = 0.0
    lost_count: int = 0
    last_mask_area: int = 0
    id_history: deque = field(default_factory=lambda: deque(maxlen=5))
    track_pose_convention: str = "raw"   # kept for debug compatibility
    recovery_mask: Optional[np.ndarray] = None

    # fast local tracking state
    last_bbox_xyxy: Optional[tuple[int, int, int, int]] = None
    last_depth_m: float = 0.0
    last_update_time_s: float = 0.0


def rgb_numpy_to_imgmsg(rgb: np.ndarray, frame_id: str, stamp) -> Image:
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


def rotation_matrix_to_quaternion_xyzw(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    q = np.empty(4, dtype=np.float64)
    trace = np.trace(R)
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        q[3] = 0.25 * s
        q[0] = (R[2, 1] - R[1, 2]) / s
        q[1] = (R[0, 2] - R[2, 0]) / s
        q[2] = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        q[3] = (R[2, 1] - R[1, 2]) / s
        q[0] = 0.25 * s
        q[1] = (R[0, 1] + R[1, 0]) / s
        q[2] = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        q[3] = (R[0, 2] - R[2, 0]) / s
        q[0] = (R[0, 1] + R[1, 0]) / s
        q[1] = 0.25 * s
        q[2] = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        q[3] = (R[1, 0] - R[0, 1]) / s
        q[0] = (R[0, 2] + R[2, 0]) / s
        q[1] = (R[1, 2] + R[2, 1]) / s
        q[2] = 0.25 * s
    q = q / (np.linalg.norm(q) + 1e-12)
    return q.astype(np.float32)


def T_to_pose_stamped(T: np.ndarray, frame_id: str, stamp) -> PoseStamped:
    T = np.asarray(T, dtype=np.float32).reshape(4, 4)
    msg = PoseStamped()
    msg.header = Header(frame_id=frame_id, stamp=stamp)
    t = T[:3, 3]
    q = rotation_matrix_to_quaternion_xyzw(T[:3, :3])
    msg.pose.position.x = float(t[0])
    msg.pose.position.y = float(t[1])
    msg.pose.position.z = float(t[2])
    msg.pose.orientation.x = float(q[0])
    msg.pose.orientation.y = float(q[1])
    msg.pose.orientation.z = float(q[2])
    msg.pose.orientation.w = float(q[3])
    return msg


def draw_mask_overlay(
    rgb: np.ndarray, mask: np.ndarray,
    color: tuple[int, int, int], alpha: float = 0.30,
) -> np.ndarray:
    out = rgb.copy()
    color_arr = np.array(color, dtype=np.uint8).reshape(1, 1, 3)
    mask3 = mask.astype(bool)[..., None]
    blended = ((1.0 - alpha) * out + alpha * color_arr).astype(np.uint8)
    return np.where(mask3, blended, out)


def draw_bbox_label(
    image: np.ndarray, bbox_xyxy: tuple[int, int, int, int],
    text: str, color: tuple[int, int, int], font_scale: float = 0.6,
) -> np.ndarray:
    out = image.copy()
    x0, y0, x1, y1 = [int(v) for v in bbox_xyxy]
    cv2.rectangle(out, (x0, y0), (x1, y1), color, 2)
    cv2.putText(
        out,
        text,
        (x0, max(20, y0 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        2,
        cv2.LINE_AA,
    )
    return out


def draw_pose_text(
    image: np.ndarray, object_id: str, dino_score: float,
    T_object_camera: np.ndarray, mode: str = "init", obj_idx: int = 0,
) -> np.ndarray:
    out = image.copy()
    t = T_object_camera[:3, 3]
    lines = [
        f"[{obj_idx}] {mode}: {object_id}",
        f"  dino: {dino_score:.3f}",
        f"  t=[{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}]",
    ]
    y = 32 + obj_idx * 100
    for line in lines:
        cv2.putText(
            out, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
            (255, 255, 255), 2, cv2.LINE_AA
        )
        cv2.putText(
            out, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
            (0, 0, 0), 1, cv2.LINE_AA
        )
        y += 26
    return out


def bbox_crop_with_local_mask(
    rgb: np.ndarray,
    mask: np.ndarray,
    bbox_xyxy: tuple[int, int, int, int],
    pad_frac: float = 0.15,
    min_pad_px: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = [int(v) for v in bbox_xyxy]

    bw = x1 - x0
    bh = y1 - y0

    pad_x = max(min_pad_px, int(round(bw * pad_frac)))
    pad_y = max(min_pad_px, int(round(bh * pad_frac)))

    x0p = max(0, x0 - pad_x)
    y0p = max(0, y0 - pad_y)
    x1p = min(w, x1 + pad_x)
    y1p = min(h, y1 + pad_y)

    return (
        rgb[y0p:y1p, x0p:x1p].copy(),
        mask[y0p:y1p, x0p:x1p].copy(),
    )


def reject_large_masks(
    masks: list[SAMMaskCandidate], h: int, w: int,
    max_mask_area_ratio: float = 0.25, max_bbox_area_ratio: float = 0.30,
) -> list[SAMMaskCandidate]:
    img_area = float(h * w)
    out = []
    for c in masks:
        x0, y0, x1, y1 = c.bbox_xyxy
        if float(c.area) / img_area > max_mask_area_ratio:
            continue
        if float((x1 - x0) * (y1 - y0)) / img_area > max_bbox_area_ratio:
            continue
        out.append(c)
    return out


def reject_outside_roi(
    masks: list[SAMMaskCandidate],
    x_min: int, x_max: int, y_min: int, y_max: int,
) -> list[SAMMaskCandidate]:
    kept = []
    for m in masks:
        x0, y0, x1, y1 = m.bbox_xyxy
        if x0 >= x_min and x1 <= x_max and y0 >= y_min and y1 <= y_max:
            kept.append(m)
    return kept


def reject_border_masks(
    masks: list[SAMMaskCandidate], h: int, w: int,
    border_px: int = 6, max_border_fraction: float = 0.0,
) -> list[SAMMaskCandidate]:
    out = []
    for c in masks:
        m = c.mask
        border_pixels = (
            m[:border_px, :].sum() + m[-border_px:, :].sum()
            + m[border_px:-border_px, :border_px].sum()
            + m[border_px:-border_px, -border_px:].sum()
        )
        if c.area == 0:
            continue
        if float(border_pixels) / float(c.area) > max_border_fraction:
            continue
        out.append(c)
    return out


def mask_depth_coverage(
    depth: np.ndarray, mask: np.ndarray,
    zmin: float = 0.05, zmax: float = 3.0,
) -> float:
    mask_bool = mask.astype(bool)
    n_mask = int(mask_bool.sum())
    if n_mask == 0:
        return 0.0
    d = depth[mask_bool]
    valid = np.isfinite(d) & (d > zmin) & (d < zmax)
    return float(valid.sum()) / float(n_mask)


def vote_object_id(history: deque) -> str:
    if not history:
        return "unknown"
    counts: dict[str, int] = defaultdict(int)
    for obj_id in history:
        counts[obj_id] += 1
    return max(counts, key=counts.get)


def batch_dino_classify(
    dino: DINOIdentifier,
    crops_rgb: list[np.ndarray],
    crops_mask: list[np.ndarray | None],
    debug_dir: str | None = None,
) -> list[DINOResult]:
    if not crops_rgb:
        return []

    tensors = []
    from pathlib import Path
    import cv2
    debug_root = Path(debug_dir) if debug_dir is not None else None
    if debug_root is not None:
        debug_root.mkdir(parents=True, exist_ok=True)

    for i, (rgb, mask) in enumerate(zip(crops_rgb, crops_mask)):
        rgb_proc = dino._ensure_rgb(rgb)
        rgb_masked = dino._apply_mask(rgb_proc, mask)
        t = dino._preprocess(rgb_masked)
        tensors.append(t)

        if debug_root is not None:
            crop_dir = debug_root / f"crop_{i:02d}"
            crop_dir.mkdir(parents=True, exist_ok=True)

            cv2.imwrite(str(crop_dir / "rgb.png"),
                        cv2.cvtColor(rgb_proc, cv2.COLOR_RGB2BGR))

            if mask is not None:
                cv2.imwrite(str(crop_dir / "mask.png"),
                            (mask.astype(np.uint8) * 255))

            cv2.imwrite(str(crop_dir / "masked.png"),
                        cv2.cvtColor(rgb_masked, cv2.COLOR_RGB2BGR))

    batch = torch.cat(tensors, dim=0)

    with torch.inference_mode():
        feats = dino.model(batch)

    if isinstance(feats, dict):
        if "x_norm_clstoken" in feats:
            feats = feats["x_norm_clstoken"]
        else:
            raise RuntimeError(f"Unexpected DINO output keys: {list(feats.keys())}")

    feats = feats.reshape(batch.shape[0], -1)
    if dino.cfg.normalize_embeddings:
        feats = F.normalize(feats, dim=1)

    embeddings = feats.detach().cpu().numpy()

    results = []
    for i, emb in enumerate(embeddings):
        res = dino.classify_embedding(emb)
        results.append(res)

        if debug_root is not None:
            crop_dir = debug_root / f"crop_{i:02d}"
            with open(crop_dir / "scores.txt", "w") as f:
                f.write(f"pred: {res.object_id}\n")
                f.write(f"score: {res.score:.4f}\n\n")
                for k, v in sorted(res.scores_by_object.items(),
                                   key=lambda x: x[1], reverse=True):
                    f.write(f"{k}: {float(v):.4f}\n")
    return results


class ProjectedMaskProvider:
    def get_mask(self, view: Any, object_id_hint: str | None = None) -> np.ndarray:
        raise NotImplementedError("Projected mask mode is not wired yet.")


class FoundationPoseTrackerNode(Node):
    def __init__(self, args: argparse.Namespace, grabber: MultiCamGrabber, T_base_cam_map):
        super().__init__("foundationpose_tracker")
        self.args = args
        self.grabber = grabber
        self.T_base_cam_map = T_base_cam_map

        self.pub_pose_base: dict[str, Any] = {}
        self.pub_pose_base_init: dict[str, Any] = {}
        self.pub_pose_base_track: dict[str, Any] = {}

        self.palette = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255),
        ]

        self.last_signature: Optional[tuple[tuple[str, int], ...]] = None
        self.busy = False
        self.frame_counter = 0
        self.csv_log_path = str(Path(args.output_root) / "fp_pose_debug.csv")
        Path(self.csv_log_path).parent.mkdir(parents=True, exist_ok=True)

        with open(self.csv_log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "frame",
                "cam",
                "object",
                "raw_x", "raw_y", "raw_z",
                "inv_x", "inv_y", "inv_z",
                "base_raw_z",
                "base_inv_z",
            ])

        self.mesh_map = self._build_mesh_map(args.cad_dir)

        self.track_states: dict[str, list[ObjectTrackState]] = {
            c.cam_id: [] for c in CAMERAS
        }

        self.dino = DINOIdentifier(
            DINOIdentifierConfig(
                model_name=args.dino_model_name,
                device=args.device,
                reference_dir=args.reference_dir,
                use_masked_background=False,
            )
        )
        self.get_logger().info("Building DINO reference bank...")
        self.dino.build_reference_bank_from_folder()
        self.get_logger().info(
            f"DINO reference bank: {len(self.dino.reference_bank)} images, "
            f"objects: {sorted(set(r.object_id for r in self.dino.reference_bank))}"
        )

        self.sam: SAMSegmenter | None = None
        if args.mask_source == "sam":
            self.sam = SAMSegmenter(
                SAMSegmenterConfig(
                    repo_root=args.sam_repo_root,
                    checkpoint=args.sam_checkpoint,
                    model_cfg=args.sam_model_cfg,
                    device=args.device,
                    max_image_side=args.sam_max_image_side,
                    min_mask_area=args.sam_min_mask_area,
                    min_bbox_side_px=args.sam_min_bbox_side_px,
                    attach_rgb_crops=False,
                )
            )

        self.projected_provider = ProjectedMaskProvider()

        self.fp_cfg = FoundationPoseConfig(
            repo_root=args.fp_repo_root,
            weights_dir=args.fp_weights_dir,
            debug_dir=str(Path(args.output_root).resolve() / "fp_internal_debug"),
            debug=args.fp_debug,
            est_refine_iter=args.est_refine_iter,
            mesh_scale=args.mesh_scale,
        )

        self.pub_raw: dict[str, Any] = {}
        self.pub_sam_overlay: dict[str, Any] = {}
        self.pub_dino_overlay: dict[str, Any] = {}
        self.pub_pose_overlay: dict[str, Any] = {}

        self.pub_pose: dict[str, Any] = {}
        self.pub_pose_init: dict[str, Any] = {}
        self.pub_pose_track: dict[str, Any] = {}

        self.last_sam_overlay: dict[str, np.ndarray] = {}
        self.last_dino_overlay: dict[str, np.ndarray] = {}

        for c in CAMERAS:
            cid = c.cam_id
            self.pub_raw[cid] = self.create_publisher(
                Image, f"/perception/fp/rgb_raw/{cid}", FAST_QOS
            )
            self.pub_sam_overlay[cid] = self.create_publisher(
                Image, f"/perception/fp/sam_overlay/{cid}", FAST_QOS
            )
            self.pub_dino_overlay[cid] = self.create_publisher(
                Image, f"/perception/fp/dino_overlay/{cid}", FAST_QOS
            )
            self.pub_pose_overlay[cid] = self.create_publisher(
                Image, f"/perception/fp/pose_overlay/{cid}", FAST_QOS
            )

        self.timer = self.create_timer(args.timer_period_s, self._tick)
        self.get_logger().info(
            f"FoundationPoseTrackerNode started | run_mode={self.args.run_mode}"
        )

    @staticmethod
    def _build_mesh_map(cad_dir: str) -> dict[str, str]:
        cad_root = Path(cad_dir)
        mesh_map: dict[str, str] = {}

        if cad_root.is_dir():
            for ext in ("*.obj", "*.stl"):
                for mesh_file in cad_root.glob(ext):
                    name = mesh_file.stem
                    mesh_map[name] = str(mesh_file)
                    mesh_map[name.lower()] = str(mesh_file)
                    mesh_map[name.capitalize()] = str(mesh_file)

        return mesh_map

    def _clip_bbox(self, bbox_xyxy: tuple[int, int, int, int], h: int, w: int) -> tuple[int, int, int, int]:
        x0, y0, x1, y1 = bbox_xyxy
        x0 = max(0, min(w - 1, int(x0)))
        y0 = max(0, min(h - 1, int(y0)))
        x1 = max(x0 + 1, min(w, int(x1)))
        y1 = max(y0 + 1, min(h, int(y1)))
        return x0, y0, x1, y1

    def _expand_bbox(
        self,
        bbox_xyxy: tuple[int, int, int, int],
        h: int,
        w: int,
        pad_px: int,
    ) -> tuple[int, int, int, int]:
        x0, y0, x1, y1 = bbox_xyxy
        return self._clip_bbox((x0 - pad_px, y0 - pad_px, x1 + pad_px, y1 + pad_px), h, w)

    def _crop_K(self, K: np.ndarray, x0: int, y0: int) -> np.ndarray:
        Kc = np.asarray(K, dtype=np.float32).copy()
        Kc[0, 2] -= float(x0)
        Kc[1, 2] -= float(y0)
        return Kc

    def _make_global_mask_from_local(
        self,
        local_mask: np.ndarray,
        x0: int,
        y0: int,
        h: int,
        w: int,
    ) -> np.ndarray:
        out = np.zeros((h, w), dtype=bool)
        hh, ww = local_mask.shape[:2]
        x1 = min(w, x0 + ww)
        y1 = min(h, y0 + hh)
        out[y0:y1, x0:x1] = local_mask[:y1 - y0, :x1 - x0].astype(bool)
        return out

    def _largest_component(self, mask: np.ndarray) -> np.ndarray:
        mask_u8 = (mask.astype(np.uint8) * 255)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
        if n <= 1:
            return mask.astype(bool)

        best_idx = -1
        best_area = 0
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area > best_area:
                best_area = area
                best_idx = i

        if best_idx < 0:
            return mask.astype(bool)

        return (labels == best_idx)

    def _bbox_from_mask(self, mask: np.ndarray) -> Optional[tuple[int, int, int, int]]:
        ys, xs = np.where(mask.astype(bool))
        if xs.size == 0 or ys.size == 0:
            return None
        x0 = int(xs.min())
        x1 = int(xs.max()) + 1
        y0 = int(ys.min())
        y1 = int(ys.max()) + 1
        return (x0, y0, x1, y1)

    def _build_local_track_observation(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        K: np.ndarray,
        state: ObjectTrackState,
    ) -> tuple[
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[tuple[int, int, int, int]],
        Optional[float],
        Optional[tuple[int, int, int, int]],
    ]:
        """
        Fast tracking observation:
        - crop a small ROI around last bbox
        - build a cheap depth-gated local mask
        """
        if state.last_bbox_xyxy is None or state.T_object_camera is None:
            return None, None, None, None, None, None, None

        h, w = rgb.shape[:2]
        roi_bbox = self._expand_bbox(
            state.last_bbox_xyxy, h, w, pad_px=self.args.track_roi_pad_px
        )
        x0, y0, x1, y1 = roi_bbox

        rgb_crop = rgb[y0:y1, x0:x1].copy()
        depth_crop = depth[y0:y1, x0:x1].copy()
        if rgb_crop.size == 0 or depth_crop.size == 0:
            return None, None, None, None, None, None, None

        K_crop = self._crop_K(K, x0, y0)

        z_prev = float(state.last_depth_m if state.last_depth_m > 0 else state.T_object_camera[2, 3])
        z_lo = max(self.args.min_valid_z_m, z_prev - self.args.track_depth_band_m)
        z_hi = min(self.args.max_valid_z_m, z_prev + self.args.track_depth_band_m)

        valid = np.isfinite(depth_crop) & (depth_crop > z_lo) & (depth_crop < z_hi)

        valid_u8 = (valid.astype(np.uint8) * 255)
        kernel = np.ones((5, 5), np.uint8)
        valid_u8 = cv2.morphologyEx(valid_u8, cv2.MORPH_OPEN, kernel)
        valid_u8 = cv2.morphologyEx(valid_u8, cv2.MORPH_CLOSE, kernel)
        local_mask = valid_u8 > 0

        if int(local_mask.sum()) < self.args.track_min_local_mask_px:
            z_lo = max(self.args.min_valid_z_m, z_prev - 2.0 * self.args.track_depth_band_m)
            z_hi = min(self.args.max_valid_z_m, z_prev + 2.0 * self.args.track_depth_band_m)
            valid = np.isfinite(depth_crop) & (depth_crop > z_lo) & (depth_crop < z_hi)
            local_mask = valid

        if int(local_mask.sum()) < self.args.track_min_local_mask_px:
            return None, None, None, None, None, None, None

        local_mask = self._largest_component(local_mask)

        if int(local_mask.sum()) < self.args.track_min_local_mask_px:
            return None, None, None, None, None, None, None

        local_bbox = self._bbox_from_mask(local_mask)
        if local_bbox is None:
            return None, None, None, None, None, None, None

        lx0, ly0, lx1, ly1 = local_bbox
        global_bbox = self._clip_bbox((x0 + lx0, y0 + ly0, x0 + lx1, y0 + ly1), h, w)

        d = depth_crop[local_mask]
        d = d[np.isfinite(d) & (d > self.args.min_valid_z_m) & (d < self.args.max_valid_z_m)]
        if d.size == 0:
            return None, None, None, None, None, None, None

        z_med = float(np.median(d))
        return rgb_crop, depth_crop, K_crop, local_mask.astype(bool), global_bbox, z_med, roi_bbox

    def _get_or_create_pose_base_pub(self, cam_id: str, object_id: str, idx: int) -> Any:
        key = f"{cam_id}/{object_id}_{idx}"
        if key not in self.pub_pose_base:
            self.pub_pose_base[key] = self.create_publisher(
                PoseStamped, f"/perception/fp/pose_base/{key}", FAST_QOS
            )
        return self.pub_pose_base[key]

    def _get_or_create_pose_base_init_pub(self, cam_id: str, object_id: str, idx: int) -> Any:
        key = f"{cam_id}/{object_id}_{idx}"
        if key not in self.pub_pose_base_init:
            self.pub_pose_base_init[key] = self.create_publisher(
                PoseStamped, f"/perception/fp/pose_base_init/{key}", FAST_QOS
            )
        return self.pub_pose_base_init[key]

    def _get_or_create_pose_base_track_pub(self, cam_id: str, object_id: str, idx: int) -> Any:
        key = f"{cam_id}/{object_id}_{idx}"
        if key not in self.pub_pose_base_track:
            self.pub_pose_base_track[key] = self.create_publisher(
                PoseStamped, f"/perception/fp/pose_base_track/{key}", FAST_QOS
            )
        return self.pub_pose_base_track[key]

    def _publish_pose_base(
        self, cam_id: str, object_id: str, idx: int, T_object_camera: np.ndarray, stamp
    ) -> None:
        T_object_camera = np.asarray(T_object_camera, dtype=np.float32).reshape(4, 4)

        T_base_object_raw = self._to_base_pose(cam_id, T_object_camera)
        pub_raw = self._get_or_create_pose_base_pub(cam_id, object_id + "_raw", idx)
        pub_raw.publish(T_to_pose_stamped(T_base_object_raw, frame_id="base", stamp=stamp))

        try:
            T_object_camera_inv = np.linalg.inv(T_object_camera)
            T_base_object_inv = self._to_base_pose(cam_id, T_object_camera_inv)
            pub_inv = self._get_or_create_pose_base_pub(cam_id, object_id + "_inv", idx)
            pub_inv.publish(T_to_pose_stamped(T_base_object_inv, frame_id="base", stamp=stamp))
        except np.linalg.LinAlgError:
            pass

    def _publish_pose_base_init(
        self, cam_id: str, object_id: str, idx: int, T_object_camera: np.ndarray, stamp
    ) -> None:
        T_object_camera = np.asarray(T_object_camera, dtype=np.float32).reshape(4, 4)

        T_base_object_raw = self._to_base_pose(cam_id, T_object_camera)
        pub_raw = self._get_or_create_pose_base_init_pub(cam_id, object_id + "_raw", idx)
        pub_raw.publish(T_to_pose_stamped(T_base_object_raw, frame_id="base", stamp=stamp))

        try:
            T_object_camera_inv = np.linalg.inv(T_object_camera)
            T_base_object_inv = self._to_base_pose(cam_id, T_object_camera_inv)
            pub_inv = self._get_or_create_pose_base_init_pub(cam_id, object_id + "_inv", idx)
            pub_inv.publish(T_to_pose_stamped(T_base_object_inv, frame_id="base", stamp=stamp))
        except np.linalg.LinAlgError:
            pass

    def _publish_pose_base_track(
        self, cam_id: str, object_id: str, idx: int, T_object_camera: np.ndarray, stamp
    ) -> None:
        T_object_camera = np.asarray(T_object_camera, dtype=np.float32).reshape(4, 4)

        T_base_object_raw = self._to_base_pose(cam_id, T_object_camera)
        pub_raw = self._get_or_create_pose_base_track_pub(cam_id, object_id + "_raw", idx)
        pub_raw.publish(T_to_pose_stamped(T_base_object_raw, frame_id="base", stamp=stamp))

        try:
            T_object_camera_inv = np.linalg.inv(T_object_camera)
            T_base_object_inv = self._to_base_pose(cam_id, T_object_camera_inv)
            pub_inv = self._get_or_create_pose_base_track_pub(cam_id, object_id + "_inv", idx)
            pub_inv.publish(T_to_pose_stamped(T_base_object_inv, frame_id="base", stamp=stamp))
        except np.linalg.LinAlgError:
            pass

    def _resolve_mesh_path(self, object_id: str) -> str:
        if object_id in self.mesh_map:
            return self.mesh_map[object_id]

        for ext in (".obj", ".stl"):
            direct = Path(self.args.cad_dir) / f"{object_id}{ext}"
            if direct.exists():
                return str(direct)

        raise FileNotFoundError(f"No CAD mesh for object_id='{object_id}'")

    def _get_or_create_pose_pub(self, cam_id: str, object_id: str, idx: int) -> Any:
        key = f"{cam_id}/{object_id}_{idx}"
        if key not in self.pub_pose:
            self.pub_pose[key] = self.create_publisher(
                PoseStamped, f"/perception/fp/pose/{key}", FAST_QOS
            )
        return self.pub_pose[key]

    def _get_or_create_pose_init_pub(self, cam_id: str, object_id: str, idx: int) -> Any:
        key = f"{cam_id}/{object_id}_{idx}"
        if key not in self.pub_pose_init:
            self.pub_pose_init[key] = self.create_publisher(
                PoseStamped, f"/perception/fp/pose_init/{key}", FAST_QOS
            )
        return self.pub_pose_init[key]

    def _get_or_create_pose_track_pub(self, cam_id: str, object_id: str, idx: int) -> Any:
        key = f"{cam_id}/{object_id}_{idx}"
        if key not in self.pub_pose_track:
            self.pub_pose_track[key] = self.create_publisher(
                PoseStamped, f"/perception/fp/pose_track/{key}", FAST_QOS
            )
        return self.pub_pose_track[key]

    def _views_signature(self, views: list[Any]) -> tuple[tuple[str, int], ...]:
        items = []
        for v in views:
            stamp_ns = int(float(v.stamp_s) * 1e9)
            items.append((str(v.cam_id), stamp_ns))
        items.sort(key=lambda x: x[0])
        return tuple(items)

    def _to_base_pose(self, cam_id: str, T_object_camera: np.ndarray) -> np.ndarray:
        T_object_camera = np.asarray(T_object_camera, dtype=np.float32).reshape(4, 4)

        if cam_id not in self.T_base_cam_map:
            raise KeyError(f"No base extrinsic for cam_id={cam_id}")

        T_base_cam = self.T_base_cam_map[cam_id]
        if hasattr(T_base_cam, "as_matrix"):
            T_base_cam = T_base_cam.as_matrix()
        elif hasattr(T_base_cam, "matrix"):
            T_base_cam = T_base_cam.matrix
        else:
            T_base_cam = np.asarray(T_base_cam, dtype=np.float32).reshape(4, 4)

        return T_base_cam @ T_object_camera

    def _generate_and_filter_masks(self, rgb: np.ndarray, cam_id: str) -> list[SAMMaskCandidate]:
        assert self.sam is not None
        masks = self.sam.generate_auto(rgb)
        if not masks:
            self.get_logger().warn(f"[{cam_id}] SAM returned 0 masks")
            return []

        h, w = rgb.shape[:2]
        masks = reject_large_masks(
            masks, h, w,
            max_mask_area_ratio=self.args.sam_max_mask_area_ratio,
            max_bbox_area_ratio=self.args.sam_max_bbox_area_ratio,
        )
        masks = reject_border_masks(
            masks, h, w,
            border_px=self.args.sam_border_px,
            max_border_fraction=self.args.sam_max_border_fraction,
        )
        x_min = 250
        x_max = w - 250
        y_min = 250
        y_max = h - 10
        masks = reject_outside_roi(masks, x_min, x_max, y_min, y_max)
        return masks

    def _classify_masks_batched(
        self, rgb: np.ndarray, masks: list[SAMMaskCandidate],
    ) -> list[CandidateSelection]:
        if not masks:
            return []

        crops_rgb: list[np.ndarray] = []
        crops_mask: list[np.ndarray | None] = []
        valid_indices: list[int] = []

        for i, cand in enumerate(masks):
            crop_rgb, crop_mask = bbox_crop_with_local_mask(rgb, cand.mask, cand.bbox_xyxy)
            if crop_rgb.size == 0 or int(crop_mask.sum()) == 0:
                continue
            crops_rgb.append(crop_rgb)
            crops_mask.append(crop_mask)
            valid_indices.append(i)

        if not crops_rgb:
            return []

        try:
            debug_dir = str(
                Path(self.args.output_root) / "dino_debug" / f"frame_{self.frame_counter:06d}"
            )

            dino_results = batch_dino_classify(
                self.dino,
                crops_rgb,
                crops_mask,
                debug_dir=debug_dir,
            )
        except Exception as e:
            self.get_logger().warn(f"Batched DINO failed: {e}")
            return []

        out: list[CandidateSelection] = []
        for j, res in enumerate(dino_results):
            mask_idx = valid_indices[j]
            cand = masks[mask_idx]

            best_score = float(res.score)
            sorted_scores = sorted(
                res.scores_by_object.items(), key=lambda kv: kv[1], reverse=True
            )
            second_score = float(sorted_scores[1][1]) if len(sorted_scores) > 1 else -1.0
            margin = best_score - second_score

            object_id = res.object_id
            if best_score < self.args.dino_min_score:
                object_id = "unknown"
            if self.args.dino_min_margin > 0.0 and margin < self.args.dino_min_margin:
                object_id = "unknown"

            out.append(
                CandidateSelection(
                    object_id=object_id,
                    score=best_score,
                    scores_by_object={k: float(v) for k, v in res.scores_by_object.items()},
                    candidate=cand,
                )
            )

        out.sort(key=lambda x: x.score, reverse=True)
        return out

    def _select_top_candidates(
        self, ranked: list[CandidateSelection], depth: np.ndarray,
    ) -> list[CandidateSelection]:
        selected: list[CandidateSelection] = []
        used_pixels = np.zeros(depth.shape[:2], dtype=bool)

        for sel in ranked:
            if sel.object_id == "unknown":
                continue
            if len(selected) >= self.args.max_objects:
                break

            mask = sel.candidate.mask.astype(bool)

            overlap = np.logical_and(mask, used_pixels).sum()
            if mask.sum() > 0 and float(overlap) / float(mask.sum()) > 0.3:
                continue

            coverage = mask_depth_coverage(
                depth,
                mask,
                zmin=self.args.min_valid_z_m,
                zmax=self.args.max_valid_z_m,
            )
            if coverage < self.args.min_depth_coverage:
                self.get_logger().info(
                    f"  skip {sel.object_id}: depth coverage {coverage:.2f} "
                    f"< {self.args.min_depth_coverage:.2f}"
                )
                continue

            selected.append(sel)
            used_pixels |= mask

        return selected

    def _pose_reason(self, T: np.ndarray) -> tuple[bool, str]:
        R = T[:3, :3]
        t = T[:3, 3]

        trace = np.trace(R)
        if trace < -1.5:
            return False, f"flipped_orientation trace={trace:.3f}"

        z = t[2]
        if z < 0.3 or z > 0.8:
            return False, f"bad_z z={z:.3f} (expected 0.3-0.8)"

        t_mag = np.linalg.norm(t)
        if t_mag < 0.4 or t_mag > 0.9:
            return False, f"bad_distance mag={t_mag:.3f}"

        return True, "ok"

    def _draw_track_box(
        self,
        image: np.ndarray,
        state: ObjectTrackState,
        obj_idx: int,
        mode: str,
    ) -> np.ndarray:
        out = image.copy()
        color = self.palette[obj_idx % len(self.palette)]

        if state.last_bbox_xyxy is not None:
            out = draw_bbox_label(
                out,
                state.last_bbox_xyxy,
                f"{state.object_id} {mode}",
                color,
                font_scale=0.6,
            )

        if state.recovery_mask is not None:
            out = draw_mask_overlay(out, state.recovery_mask, color=color, alpha=0.18)

        if state.T_object_camera is not None:
            out = draw_pose_text(
                out,
                state.object_id,
                state.dino_score,
                state.T_object_camera,
                mode=mode,
                obj_idx=obj_idx,
            )
        return out

    def _draw_sam_overlay(self, rgb: np.ndarray, masks: list[SAMMaskCandidate]) -> np.ndarray:
        vis = rgb.copy()
        for i, cand in enumerate(masks[:self.args.max_candidate_draw]):
            color = self.palette[i % len(self.palette)]
            vis = draw_mask_overlay(vis, cand.mask, color=color, alpha=0.22)
            txt = f"{i}: sam={cand.score:.2f} area={cand.area}"
            vis = draw_bbox_label(vis, cand.bbox_xyxy, txt, color, font_scale=0.5)
        return vis

    def _draw_dino_overlay(self, rgb: np.ndarray, ranked: list[CandidateSelection]) -> np.ndarray:
        vis = rgb.copy()
        for i, sel in enumerate(ranked[:self.args.max_candidate_draw]):
            color = self.palette[i % len(self.palette)]
            vis = draw_mask_overlay(vis, sel.candidate.mask, color=color, alpha=0.22)
            txt = f"{sel.object_id} {sel.score:.2f}"
            vis = draw_bbox_label(vis, sel.candidate.bbox_xyxy, txt, color, font_scale=0.55)
        return vis

    def _publish_overlays(
        self, cam_id: str, stamp,
        rgb: np.ndarray,
        sam_overlay: np.ndarray,
        dino_overlay: np.ndarray,
        pose_overlay: np.ndarray,
    ) -> None:
        self.pub_raw[cam_id].publish(rgb_numpy_to_imgmsg(rgb, frame_id=cam_id, stamp=stamp))
        self.pub_sam_overlay[cam_id].publish(
            rgb_numpy_to_imgmsg(sam_overlay, frame_id=cam_id, stamp=stamp)
        )
        self.pub_dino_overlay[cam_id].publish(
            rgb_numpy_to_imgmsg(dino_overlay, frame_id=cam_id, stamp=stamp)
        )
        self.pub_pose_overlay[cam_id].publish(
            rgb_numpy_to_imgmsg(pose_overlay, frame_id=cam_id, stamp=stamp)
        )

    def _publish_pose(self, cam_id: str, object_id: str, idx: int, T: np.ndarray, stamp) -> None:
        pub = self._get_or_create_pose_pub(cam_id, object_id, idx)
        pub.publish(T_to_pose_stamped(T, frame_id=cam_id, stamp=stamp))

    def _publish_pose_init(self, cam_id: str, object_id: str, idx: int, T: np.ndarray, stamp) -> None:
        pub = self._get_or_create_pose_init_pub(cam_id, object_id, idx)
        pub.publish(T_to_pose_stamped(T, frame_id=cam_id, stamp=stamp))

    def _publish_pose_track(self, cam_id: str, object_id: str, idx: int, T: np.ndarray, stamp) -> None:
        pub = self._get_or_create_pose_track_pub(cam_id, object_id, idx)
        pub.publish(T_to_pose_stamped(T, frame_id=cam_id, stamp=stamp))

    def _log_pose(self, prefix: str, cam_id: str, object_id: str, idx: int, T: np.ndarray, score: float) -> None:
        q = rotation_matrix_to_quaternion_xyzw(T[:3, :3])
        self.get_logger().info(
            f"[{cam_id}] {prefix} [{idx}] {object_id} "
            f"t=[{T[0,3]:.3f}, {T[1,3]:.3f}, {T[2,3]:.3f}] "
            f"q=[{q[0]:.3f}, {q[1]:.3f}, {q[2]:.3f}, {q[3]:.3f}] "
            f"score={score:.3f}"
        )

    def _try_local_recover(
        self,
        view: Any,
        state: ObjectTrackState,
    ) -> tuple[Optional[np.ndarray], Optional[str]]:
        """
        Slow recovery path:
        rerun local FP estimate on a cropped ROI, then continue.
        """
        if state.fp_tracker is None:
            return None, None

        rgb = view.rgb
        depth = view.depth
        K = np.asarray(view.K, dtype=np.float32).reshape(3, 3)

        obs = self._build_local_track_observation(rgb, depth, K, state)
        rgb_crop, depth_crop, K_crop, mask_crop, new_bbox_xyxy, z_med, roi_bbox = obs

        if rgb_crop is None:
            return None, None

        try:
            result = state.fp_tracker.estimate_pose(
                object_id=state.object_id,
                mesh_path=state.mesh_path,
                rgb=rgb_crop,
                depth=depth_crop,
                K=K_crop,
                mask=mask_crop,
            )
        except Exception as e:
            self.get_logger().warn(
                f"[{view.cam_id}] RECOVER {state.object_id} failed: {e}"
            )
            return None, None

        T_rec = np.asarray(result.T_object_camera, dtype=np.float32).reshape(4, 4)

        ok_pose, pose_msg = self._pose_reason(T_rec)
        if not ok_pose:
            self.get_logger().warn(
                f"[{view.cam_id}] RECOVER {state.object_id} unreasonable: {pose_msg}"
            )
            return None, None

        h, w = rgb.shape[:2]
        if roi_bbox is not None:
            rx0, ry0, _, _ = roi_bbox
            state.recovery_mask = self._make_global_mask_from_local(mask_crop, rx0, ry0, h, w)
        state.last_bbox_xyxy = new_bbox_xyxy
        state.last_depth_m = float(z_med) if z_med is not None else float(T_rec[2, 3])
        state.last_update_time_s = time.time()

        return T_rec, "raw"

    def _initialize_objects(
        self, view: Any, selections: list[CandidateSelection], stamp,
    ) -> tuple[list[ObjectTrackState], np.ndarray]:
        rgb = view.rgb
        depth = view.depth
        K = np.asarray(view.K, dtype=np.float32).reshape(3, 3)
        cam_id = view.cam_id

        new_states: list[ObjectTrackState] = []
        pose_overlay = rgb.copy()

        for i, sel in enumerate(selections):
            try:
                mesh_path = self._resolve_mesh_path(sel.object_id)
            except FileNotFoundError as e:
                self.get_logger().warn(f"[{cam_id}] {e}")
                continue

            tracker = FoundationPoseWrapper(
                FoundationPoseConfig(
                    repo_root=self.fp_cfg.repo_root,
                    weights_dir=self.fp_cfg.weights_dir,
                    debug_dir=str(
                        Path(self.fp_cfg.debug_dir).resolve() / f"{cam_id}_{sel.object_id}_{i}"
                    ),
                    debug=self.fp_cfg.debug,
                    est_refine_iter=self.fp_cfg.est_refine_iter,
                    mesh_scale=self.fp_cfg.mesh_scale,
                )
            )

            mask = sel.candidate.mask.astype(bool)
            n_mask = int(mask.sum())
            depth_vals = depth[mask]
            valid = np.isfinite(depth_vals) & (depth_vals > self.args.min_valid_z_m) & (depth_vals < self.args.max_valid_z_m)

            self.get_logger().info(
                f"[{cam_id}] INIT INPUT {sel.object_id} "
                f"mask_px={n_mask} "
                f"valid_depth={int(valid.sum())}/{len(depth_vals)} "
                f"bbox={sel.candidate.bbox_xyxy}"
            )

            t0 = time.perf_counter()
            debug_frame_dir = Path(self.args.output_root) / "debug_frames" / f"frame_{self.frame_counter:06d}"
            debug_frame_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(debug_frame_dir / "rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            np.save(str(debug_frame_dir / "depth.npy"), depth)
            cv2.imwrite(str(debug_frame_dir / "mask.png"), (sel.candidate.mask * 255).astype(np.uint8))
            np.save(str(debug_frame_dir / "K.npy"), K)
            try:
                mask_depth = depth[mask]
                valid_depth = mask_depth[np.isfinite(mask_depth) & (mask_depth > 0)]
                print(f"[DEPTH CHECK] median={np.median(valid_depth):.3f} std={np.std(valid_depth):.3f}")
                print(f"[DEPTH CHECK] Expecting object at ~0.58m, mesh is 0.12m wide")
                print(f"[DEPTH CHECK] If mesh center is at depth={np.median(valid_depth):.3f}, mesh extends {np.median(valid_depth)-0.06:.3f} to {np.median(valid_depth)+0.06:.3f}")
                self.get_logger().info(
                    f"[FP INPUT DEBUG] {sel.object_id} "
                    f"depth: min={np.min(valid_depth):.3f} median={np.median(valid_depth):.3f} max={np.max(valid_depth):.3f} "
                    f"mask_area={int(mask.sum())} "
                    f"K_fx={K[0,0]:.1f} K_fy={K[1,1]:.1f}"
                )
                result = tracker.estimate_pose(
                    object_id=sel.object_id,
                    mesh_path=mesh_path,
                    rgb=rgb,
                    depth=depth,
                    K=K,
                    mask=sel.candidate.mask,
                )
                T = result.T_object_camera
                print(f"[FP RESULT] t={T[:3,3]} det={np.linalg.det(T):.6f}")
                print(f"[FP RESULT] R_trace={np.trace(T[:3,:3]):.6f}")

                T = result.T_object_camera
                R = T[:3, :3]
                t = T[:3, 3]

                self.get_logger().info(
                    f"[FP OUTPUT DEBUG] {sel.object_id} "
                    f"t={t} "
                    f"trace={np.trace(R):.3f} "
                    f"det={np.linalg.det(R):.6f}"
                )

                T_inv = np.linalg.inv(T)
                t_inv = T_inv[:3, 3]
                self.get_logger().info(
                    f"[FP OUTPUT DEBUG INV] {sel.object_id} "
                    f"t_inv={t_inv}"
                )

                R = T[:3, :3]
                print(f"[FP RESULT] R@R.T (should be I):\n{R @ R.T}")
                trace = np.trace(R)

                self.get_logger().info(
                    f"[FP CONVENTION CHECK] trace={trace:.3f} t_mag={np.linalg.norm(t):.3f} "
                    f"depth_median={np.median(valid_depth):.3f}"
                )
            except Exception as e:
                self.get_logger().warn(
                    f"[{cam_id}] INIT [{i}] {sel.object_id} FP exception: {e}"
                )
                continue
            dt_fp = (time.perf_counter() - t0) * 1000.0

            T = np.asarray(result.T_object_camera, dtype=np.float32).reshape(4, 4)

            T_inv = None
            T_base_raw = None
            T_base_inv = None

            try:
                T_inv = np.linalg.inv(T)
            except np.linalg.LinAlgError:
                pass

            try:
                T_base_raw = self._to_base_pose(cam_id, T)
            except Exception as e:
                self.get_logger().warn(f"[{cam_id}] INIT base raw failed: {e}")

            if T_inv is not None:
                try:
                    T_base_inv = self._to_base_pose(cam_id, T_inv)
                except Exception as e:
                    self.get_logger().warn(f"[{cam_id}] INIT base inv failed: {e}")

            self.get_logger().info(
                f"[{cam_id}] DEBUG INIT RAW {sel.object_id} "
                f"t_cam=[{T[0,3]:.3f}, {T[1,3]:.3f}, {T[2,3]:.3f}]"
            )

            if T_inv is not None:
                self.get_logger().info(
                    f"[{cam_id}] DEBUG INIT INV {sel.object_id} "
                    f"t_cam_inv=[{T_inv[0,3]:.3f}, {T_inv[1,3]:.3f}, {T_inv[2,3]:.3f}]"
                )

            if T_base_raw is not None:
                self.get_logger().info(
                    f"[{cam_id}] DEBUG INIT BASE_RAW {sel.object_id} "
                    f"t_base_raw=[{T_base_raw[0,3]:.3f}, {T_base_raw[1,3]:.3f}, {T_base_raw[2,3]:.3f}]"
                )

            if T_base_inv is not None:
                self.get_logger().info(
                    f"[{cam_id}] DEBUG INIT BASE_INV {sel.object_id} "
                    f"t_base_inv=[{T_base_inv[0,3]:.3f}, {T_base_inv[1,3]:.3f}, {T_base_inv[2,3]:.3f}]"
                )

            def xyz(Tm):
                if Tm is None:
                    return [None, None, None]
                return [float(Tm[0,3]), float(Tm[1,3]), float(Tm[2,3])]

            with open(self.csv_log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.frame_counter,
                    cam_id,
                    sel.object_id,
                    *xyz(T),
                    *xyz(T_inv),
                    T_base_raw[2,3] if T_base_raw is not None else None,
                    T_base_inv[2,3] if T_base_inv is not None else None,
                ])

            ok_pose, pose_msg = self._pose_reason(T)
            if not ok_pose:
                self.get_logger().warn(
                    f"[{cam_id}] INIT {sel.object_id}: pose unreasonable: {pose_msg} "
                    f"t=[{T[0,3]:.3f}, {T[1,3]:.3f}, {T[2,3]:.3f}]"
                )
                continue

            self._log_pose("INIT POSE", cam_id, sel.object_id, i, T, float(sel.score))
            self._publish_pose_init(cam_id, sel.object_id, i, T, stamp)
            self._publish_pose_base_init(cam_id, sel.object_id, i, T, stamp)

            track_pose_convention = "raw"
            try:
                t_same = time.perf_counter()
                same_result = tracker.track_pose(
                    object_id=sel.object_id,
                    mesh_path=mesh_path,
                    rgb=rgb,
                    depth=depth,
                    K=K,
                    T_object_camera_init=T,
                )
                dt_same = (time.perf_counter() - t_same) * 1000.0

                T_same_raw = np.asarray(same_result.T_object_camera, dtype=np.float32).reshape(4, 4)
                jump_same_raw = float(np.linalg.norm(T_same_raw[:3, 3] - T[:3, 3]))

                try:
                    T_same_inv = np.linalg.inv(T_same_raw)
                    jump_same_inv = float(np.linalg.norm(T_same_inv[:3, 3] - T[:3, 3]))
                except np.linalg.LinAlgError:
                    jump_same_inv = float("inf")

                track_pose_convention = "inv" if jump_same_inv < jump_same_raw else "raw"

                self.get_logger().warn(
                    f"[{cam_id}] SAME-FRAME CHECK [{i}] {sel.object_id} "
                    f"track_same={dt_same:.1f}ms "
                    f"jump_same_raw={jump_same_raw:.3f}m "
                    f"jump_same_inv={jump_same_inv:.3f}m "
                    f"track_convention={track_pose_convention}"
                )
            except Exception as e:
                self.get_logger().warn(
                    f"[{cam_id}] SAME-FRAME CHECK [{i}] {sel.object_id} failed: {e}"
                )

            state = ObjectTrackState(
                object_id=sel.object_id,
                mesh_path=mesh_path,
                fp_tracker=tracker,
                mode="track",
                T_object_camera=T.copy(),
                dino_score=float(sel.score),
                lost_count=0,
                last_mask_area=int(sel.candidate.mask.sum()),
                track_pose_convention=track_pose_convention,
                recovery_mask=sel.candidate.mask.copy(),
                last_bbox_xyxy=tuple(int(v) for v in sel.candidate.bbox_xyxy),
                last_depth_m=float(np.median(valid_depth)) if len(valid_depth) > 0 else float(T[2, 3]),
                last_update_time_s=time.time(),
            )
            state.id_history.append(sel.object_id)
            new_states.append(state)

            color = self.palette[i % len(self.palette)]
            pose_overlay = draw_mask_overlay(
                pose_overlay, sel.candidate.mask, color=color, alpha=0.25
            )
            pose_overlay = draw_bbox_label(
                pose_overlay,
                sel.candidate.bbox_xyxy,
                f"{sel.object_id} {sel.score:.2f}",
                color,
                font_scale=0.6,
            )
            pose_overlay = draw_pose_text(
                pose_overlay, sel.object_id, sel.score, T, mode="init", obj_idx=i
            )

            self.get_logger().info(
                f"[{cam_id}] INIT [{i}] obj={sel.object_id} "
                f"dino={sel.score:.3f} mask_area={state.last_mask_area} "
                f"fp={dt_fp:.1f}ms"
            )

        return new_states, pose_overlay

    def _track_objects(
        self, view: Any, states: list[ObjectTrackState], stamp,
    ) -> tuple[list[ObjectTrackState], np.ndarray]:
        """
        Fast local tracking mode:
        - NO full-image SAM
        - NO DINO
        - local ROI around previous bbox
        - cheap depth-gated mask inside ROI
        - local FP register on cropped inputs
        """
        rgb = view.rgb
        depth = view.depth
        K = np.asarray(view.K, dtype=np.float32).reshape(3, 3)
        cam_id = view.cam_id

        surviving: list[ObjectTrackState] = []
        pose_overlay = rgb.copy()

        for i, state in enumerate(states):
            if state.T_object_camera is None or state.fp_tracker is None or state.last_bbox_xyxy is None:
                state.lost_count += 1
                self.get_logger().warn(
                    f"[{cam_id}] TRACK [{i}] {state.object_id} missing tracking state "
                    f"(lost={state.lost_count})"
                )
                if state.lost_count < self.args.max_lost_count:
                    surviving.append(state)
                continue

            obs = self._build_local_track_observation(rgb, depth, K, state)
            rgb_crop, depth_crop, K_crop, mask_crop, new_bbox_xyxy, z_med, roi_bbox = obs

            if rgb_crop is None:
                state.lost_count += 1
                self.get_logger().warn(
                    f"[{cam_id}] TRACK [{i}] {state.object_id} local observation failed "
                    f"(lost={state.lost_count})"
                )
                if state.lost_count < self.args.max_lost_count:
                    surviving.append(state)
                continue

            t0 = time.perf_counter()
            try:
                result = state.fp_tracker.estimate_pose(
                    object_id=state.object_id,
                    mesh_path=state.mesh_path,
                    rgb=rgb_crop,
                    depth=depth_crop,
                    K=K_crop,
                    mask=mask_crop,
                )
            except Exception as e:
                state.lost_count += 1
                self.get_logger().warn(
                    f"[{cam_id}] TRACK [{i}] {state.object_id} local FP exception: {e} "
                    f"(lost={state.lost_count})"
                )
                if state.lost_count < self.args.max_lost_count:
                    surviving.append(state)
                continue

            dt_fp = (time.perf_counter() - t0) * 1000.0

            T_new = np.asarray(result.T_object_camera, dtype=np.float32).reshape(4, 4)
            prev = np.asarray(state.T_object_camera, dtype=np.float32).reshape(4, 4)
            jump_m = float(np.linalg.norm(T_new[:3, 3] - prev[:3, 3]))

            ok_pose, pose_msg = self._pose_reason(T_new)
            if not ok_pose:
                state.lost_count += 1
                self.get_logger().warn(
                    f"[{cam_id}] TRACK [{i}] {state.object_id} bad local pose: {pose_msg} "
                    f"(lost={state.lost_count})"
                )
                if state.lost_count < self.args.max_lost_count:
                    surviving.append(state)
                continue

            if jump_m > self.args.max_translation_jump_m:
                state.lost_count += 1
                self.get_logger().warn(
                    f"[{cam_id}] TRACK [{i}] {state.object_id} jump suspicious: "
                    f"{jump_m:.3f} m > {self.args.max_translation_jump_m:.3f} m "
                    f"(lost={state.lost_count})"
                )
                if state.lost_count < self.args.max_lost_count:
                    surviving.append(state)
                continue

            state.last_bbox_xyxy = new_bbox_xyxy
            state.last_depth_m = float(z_med) if z_med is not None else float(T_new[2, 3])
            state.last_update_time_s = time.time()
            state.last_mask_area = int(mask_crop.sum())

            h, w = rgb.shape[:2]
            if roi_bbox is not None:
                rx0, ry0, _, _ = roi_bbox
                state.recovery_mask = self._make_global_mask_from_local(mask_crop, rx0, ry0, h, w)

            state.T_object_camera = T_new.copy()
            state.lost_count = 0
            state.mode = "track"
            surviving.append(state)

            smoothed_id = vote_object_id(state.id_history)

            pose_overlay = self._draw_track_box(
                pose_overlay, state, i, mode="track_fast"
            )

            self.get_logger().info(
                f"[{cam_id}] TRACK [{i}] {state.object_id} "
                f"fp={dt_fp:.1f}ms jump={jump_m:.3f}m "
                f"bbox={state.last_bbox_xyxy} mask_px={int(mask_crop.sum())}"
            )

            self._log_pose("TRACK POSE", cam_id, smoothed_id, i, T_new, state.dino_score)
            self._publish_pose_track(cam_id, state.object_id, i, T_new, stamp)
            self._publish_pose_base_track(cam_id, state.object_id, i, T_new, stamp)
            self._publish_pose(cam_id, state.object_id, i, T_new, stamp)
            self._publish_pose_base(cam_id, state.object_id, i, T_new, stamp)

        return surviving, pose_overlay

    def _process_single_view(self, view: Any) -> None:
        cam_id = view.cam_id

        if view.rgb is None or view.depth is None:
            self.get_logger().warn(f"[{cam_id}] missing rgb/depth, skip")
            return
        if view.rgb.shape[:2] != view.depth.shape[:2]:
            self.get_logger().warn(f"[{cam_id}] shape mismatch, skip")
            return

        stamp = self.get_clock().now().to_msg()
        rgb = view.rgb
        depth = view.depth
        states = self.track_states[cam_id]

        if self.args.run_mode == "track" and states and all(s.mode == "track" for s in states):
            t0 = time.perf_counter()
            surviving, pose_overlay = self._track_objects(view, states, stamp)
            dt = (time.perf_counter() - t0) * 1000.0

            if surviving:
                self.track_states[cam_id] = surviving

                sam_overlay_cached = self.last_sam_overlay.get(cam_id, rgb)
                dino_overlay_cached = self.last_dino_overlay.get(cam_id, rgb)

                self._publish_overlays(
                    cam_id, stamp, rgb,
                    sam_overlay=sam_overlay_cached,
                    dino_overlay=dino_overlay_cached,
                    pose_overlay=pose_overlay,
                )
                self.get_logger().info(
                    f"[{cam_id}] TRACK total={dt:.1f}ms objects={len(surviving)}"
                )
                return
            else:
                self.get_logger().warn(
                    f"[{cam_id}] all tracks lost, falling back to global re-init"
                )

        t_total = time.perf_counter()

        if self.args.mask_source != "sam":
            self.get_logger().warn(
                f"[{cam_id}] only 'sam' mask source supported for multi-object"
            )
            return

        t0 = time.perf_counter()
        masks = self._generate_and_filter_masks(rgb, cam_id)
        dt_sam = (time.perf_counter() - t0) * 1000.0

        if not masks:
            sam_overlay_cached = self.last_sam_overlay.get(cam_id, rgb)
            dino_overlay_cached = self.last_dino_overlay.get(cam_id, rgb)

            self._publish_overlays(
                cam_id,
                stamp,
                rgb,
                sam_overlay=sam_overlay_cached,
                dino_overlay=dino_overlay_cached,
                pose_overlay=rgb,
            )
            return

        sam_overlay = self._draw_sam_overlay(rgb, masks)

        t0 = time.perf_counter()
        ranked = self._classify_masks_batched(rgb, masks)
        dt_dino = (time.perf_counter() - t0) * 1000.0

        dino_overlay = self._draw_dino_overlay(rgb, ranked) if ranked else rgb.copy()

        self.last_sam_overlay[cam_id] = sam_overlay.copy()
        self.last_dino_overlay[cam_id] = dino_overlay.copy()

        selected = self._select_top_candidates(ranked, depth)

        if not selected:
            self.get_logger().info(f"[{cam_id}] no valid candidates after filtering")
            self.track_states[cam_id] = []
            self._publish_overlays(cam_id, stamp, rgb, sam_overlay, dino_overlay, rgb)
            return

        t0 = time.perf_counter()
        new_states, pose_overlay = self._initialize_objects(view, selected, stamp)
        dt_fp = (time.perf_counter() - t0) * 1000.0

        if self.args.run_mode == "track":
            self.track_states[cam_id] = new_states
        else:
            self.track_states[cam_id] = []

        self._publish_overlays(cam_id, stamp, rgb, sam_overlay, dino_overlay, pose_overlay)

        for i, s in enumerate(new_states):
            if s.T_object_camera is not None:
                self._publish_pose(cam_id, s.object_id, i, s.T_object_camera, stamp)
                self._publish_pose_base(cam_id, s.object_id, i, s.T_object_camera, stamp)

        dt_total = (time.perf_counter() - t_total) * 1000.0
        self.get_logger().info(
            f"[{cam_id}] INIT total={dt_total:.1f}ms "
            f"(sam={dt_sam:.1f} dino={dt_dino:.1f} fp={dt_fp:.1f}) "
            f"masks={len(masks)} ranked={len(ranked)} "
            f"selected={len(selected)} initialized={len(new_states)} "
            f"run_mode={self.args.run_mode}"
        )

    def _tick(self) -> None:
        if self.busy:
            return

        views = self.grabber.get_latest_views()
        if views is None:
            return

        signature = self._views_signature(views)
        if signature == self.last_signature:
            return
        self.last_signature = signature

        self.busy = True
        try:
            self.frame_counter += 1
            for view in views:
                try:
                    self._process_single_view(view)
                except Exception as e:
                    self.get_logger().warn(f"[{view.cam_id}] processing failed: {e}")
                    self.track_states[view.cam_id] = []
        finally:
            self.busy = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--device", default="cuda")
    p.add_argument("--mask-source", choices=["sam", "projected"], default="sam")
    p.add_argument("--target-object", default=None)
    p.add_argument("--run-mode", choices=["track", "init_only"], default="track")

    p.add_argument("--reference-dir", default="Data/ZED_screens")
    p.add_argument("--cad-dir", default="Data/CAD_Models")
    p.add_argument("--output-root", default="outputs/foundationpose")

    p.add_argument("--dino-model-name", default="dinov2_vitb14")
    p.add_argument("--dino-min-score", type=float, default=0.68)
    p.add_argument("--dino-min-margin", type=float, default=0.005)

    p.add_argument("--sam-repo-root", default="external/sam2")
    p.add_argument(
        "--sam-checkpoint",
        default="external/sam2/checkpoints/sam2.1_hiera_base_plus.pt",
    )
    p.add_argument("--sam-model-cfg", default="configs/sam2.1/sam2.1_hiera_b+.yaml")
    p.add_argument("--sam-max-image-side", type=int, default=1024)
    p.add_argument("--sam-min-mask-area", type=int, default=600)
    p.add_argument("--sam-min-bbox-side-px", type=int, default=10)
    p.add_argument("--sam-max-mask-area-ratio", type=float, default=0.10)
    p.add_argument("--sam-max-bbox-area-ratio", type=float, default=0.12)
    p.add_argument("--sam-border-px", type=int, default=6)
    p.add_argument("--sam-max-border-fraction", type=float, default=0.00)

    p.add_argument("--fp-repo-root", default="external/FoundationPose")
    p.add_argument("--fp-weights-dir", default="external/FoundationPose/weights")
    p.add_argument("--fp-debug", type=int, default=2)
    p.add_argument("--est-refine-iter", type=int, default=0)
    p.add_argument("--mesh-scale", type=float, default=0.01)

    p.add_argument("--timer-period-s", type=float, default=0.25)
    p.add_argument("--max-candidate-draw", type=int, default=25)

    p.add_argument("--min-valid-z-m", type=float, default=0.05)
    p.add_argument("--max-valid-z-m", type=float, default=10.00)
    p.add_argument("--max-translation-jump-m", type=float, default=0.80)

    p.add_argument("--max-objects", type=int, default=4)
    p.add_argument("--max-lost-count", type=int, default=8)

    p.add_argument("--min-depth-coverage", type=float, default=0.30)

    # fast local tracking
    p.add_argument("--track-roi-pad-px", type=int, default=70)
    p.add_argument("--track-depth-band-m", type=float, default=0.08)
    p.add_argument("--track-min-local-mask-px", type=int, default=500)
    p.add_argument("--track-bbox-pad-after-update-px", type=int, default=20)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    rclpy.init()

    T_map = load_extrinsics_yaml("config/camera_extrinsics_base.yaml")
    grabber = MultiCamGrabber(
        cameras=CAMERAS,
        sync_slop_s=0.10,
        use_best_effort_if_unsynced=True,
        static_extrinsics_base_cam=T_map,
        rgb_depth_max_dt_s=0.08,
    )

    node = FoundationPoseTrackerNode(args=args, grabber=grabber, T_base_cam_map=T_map)

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