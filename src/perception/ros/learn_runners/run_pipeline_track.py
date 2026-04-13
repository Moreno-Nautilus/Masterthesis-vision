from __future__ import annotations

import argparse
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import rclpy
from geometry_msgs.msg import Pose, PoseStamped
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Header

from fp_debug_msgs.msg import DebugCandidate, DebugFrame, DebugMaskCrop, DebugPoseItem

from src.calibration.io_extrinsics import load_extrinsics_yaml
from src.perception.learned.DINO.dino_identifier import (
    DINOIdentifier,
    DINOIdentifierConfig,
    DINOResult,
)
from src.perception.learned.FP.pose_foundation import (
    FoundationPoseConfig,
    FoundationPoseWrapper,
)
from src.perception.learned.SAM.sam_segmentation import (
    SAMMaskCandidate,
    SAMSegmenter,
    SAMSegmenterConfig,
)
from src.perception.ros.multicam_grabber import CameraTopics, MultiCamGrabber
from src.perception.tracking.realtime_tracker import RealtimeTracker, RealtimeTrackerConfig
from src.perception.tracking.cutie_tracker import CutieConfig
from src.perception.tracking.icp_refiner import ICPConfig, ICPVariant

FAST_QOS = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
)

CAMERAS = [
    # CameraTopics(
    #     cam_id="zed2i_1",
    #     depth_topic="/zed2i_1/zed_node/depth/depth_registered",
    #     info_topic="/zed2i_1/zed_node/depth/depth_registered/camera_info",
    #     rgb_topic="/zed2i_1/zed_node/rgb/color/rect/image",
    #     rgb_info_topic="/zed2i_1/zed_node/rgb/color/rect/image/camera_info",
    # ),
    CameraTopics(
        cam_id="zed2i_2",
        depth_topic="/zed2i_2/zed_node/depth/depth_registered",
        info_topic="/zed2i_2/zed_node/depth/depth_registered/camera_info",
        rgb_topic="/zed2i_2/zed_node/rgb/color/rect/image",
        rgb_info_topic="/zed2i_2/zed_node/rgb/color/rect/image/camera_info",
    ),
]


@dataclass
class ObjectTrackState:
    object_id: str
    mesh_path: str
    mode: str = "search"
    T_object_camera: Optional[np.ndarray] = None
    dino_score: float = 0.0
    lost_count: int = 0
    last_mask_area: int = 0
    id_history: deque = field(default_factory=lambda: deque(maxlen=5))
    track_pose_convention: str = "raw"
    recovery_mask: Optional[np.ndarray] = None

    last_logged_T_base: Optional[np.ndarray] = None
    last_logged_convention: Optional[str] = None


@dataclass
class CandidateSelection:
    object_id: str
    score: float
    scores_by_object: dict[str, float]
    candidate: SAMMaskCandidate


@dataclass
class CameraSAMParams:
    min_mask_area: int
    min_bbox_side_px: int
    max_mask_area_ratio: float
    max_bbox_area_ratio: float
    border_px: int
    max_border_fraction: float
    roi_polygon: np.ndarray


def rotation_angle_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    R_rel = R1.T @ R2
    c = (np.trace(R_rel) - 1.0) * 0.5
    c = float(np.clip(c, -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def should_log_track_update(
    T_base_new: np.ndarray,
    T_base_last: Optional[np.ndarray],
    convention_new: str,
    convention_last: Optional[str],
    trans_thresh_m: float = 0.005,
    rot_thresh_deg: float = 4.0,
) -> bool:
    if T_base_last is None:
        return True
    if convention_last != convention_new:
        return True

    dt = float(np.linalg.norm(T_base_new[:3, 3] - T_base_last[:3, 3]))
    drot = rotation_angle_deg(T_base_last[:3, :3], T_base_new[:3, :3])
    return (dt > trans_thresh_m) or (drot > rot_thresh_deg)


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


def quaternion_xyzw_to_pose_msg(t_xyz: np.ndarray, q_xyzw: np.ndarray) -> Pose:
    msg = Pose()
    msg.position.x = float(t_xyz[0])
    msg.position.y = float(t_xyz[1])
    msg.position.z = float(t_xyz[2])
    msg.orientation.x = float(q_xyzw[0])
    msg.orientation.y = float(q_xyzw[1])
    msg.orientation.z = float(q_xyzw[2])
    msg.orientation.w = float(q_xyzw[3])
    return msg


def T_to_pose_msg(T: np.ndarray) -> Pose:
    T = np.asarray(T, dtype=np.float32).reshape(4, 4)
    t = T[:3, 3]
    q = rotation_matrix_to_quaternion_xyzw(T[:3, :3])
    return quaternion_xyzw_to_pose_msg(t, q)


def T_to_pose_stamped(T: np.ndarray, frame_id: str, stamp) -> PoseStamped:
    T = np.asarray(T, dtype=np.float32).reshape(4, 4)
    msg = PoseStamped()
    msg.header = Header(frame_id=frame_id, stamp=stamp)
    msg.pose = T_to_pose_msg(T)
    return msg


def parse_polygon_string(s: str) -> np.ndarray:
    vals = [int(v.strip()) for v in s.split(",")]
    if len(vals) % 2 != 0:
        raise ValueError(f"Polygon string must have even number of values: {s}")
    return np.array(vals, dtype=np.int32).reshape(-1, 2)


def parse_xyxy_string(s: str) -> tuple[int, int, int, int]:
    vals = [int(v.strip()) for v in s.split(",")]
    if len(vals) != 4:
        raise ValueError(f"ROI box string must have 4 ints: {s}")
    return vals[0], vals[1], vals[2], vals[3]


def bbox_size_xyxy(b: tuple[int, int, int, int]) -> tuple[int, int]:
    x0, y0, x1, y1 = b
    return x1 - x0, y1 - y0


def nms_by_position(
    states: list[ObjectTrackState],
    position_threshold: float = 0.03,
) -> list[ObjectTrackState]:
    if len(states) <= 1:
        return states

    by_class: dict[str, list[ObjectTrackState]] = {}
    for s in states:
        by_class.setdefault(s.object_id, []).append(s)

    kept: list[ObjectTrackState] = []
    for _, obj_states in by_class.items():
        if len(obj_states) == 1:
            kept.extend(obj_states)
            continue

        obj_states = sorted(obj_states, key=lambda x: x.dino_score, reverse=True)
        keep_mask = [True] * len(obj_states)

        for i in range(len(obj_states)):
            if not keep_mask[i]:
                continue
            if obj_states[i].T_object_camera is None:
                continue
            pos_i = obj_states[i].T_object_camera[:3, 3]

            for j in range(i + 1, len(obj_states)):
                if not keep_mask[j]:
                    continue
                if obj_states[j].T_object_camera is None:
                    continue
                pos_j = obj_states[j].T_object_camera[:3, 3]
                dist = np.linalg.norm(pos_i - pos_j)
                if dist < position_threshold:
                    keep_mask[j] = False

        kept.extend([s for s, k in zip(obj_states, keep_mask) if k])

    return kept


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
    masks: list[SAMMaskCandidate],
    h: int,
    w: int,
    max_mask_area_ratio: float,
    max_bbox_area_ratio: float,
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


def reject_outside_roi_polygon(
    masks: list[SAMMaskCandidate],
    polygon: np.ndarray,
) -> list[SAMMaskCandidate]:
    kept = []
    for m in masks:
        x0, y0, x1, y1 = m.bbox_xyxy
        cx = (x0 + x1) // 2
        cy = (y0 + y1) // 2
        if cv2.pointPolygonTest(polygon, (float(cx), float(cy)), False) >= 0:
            kept.append(m)
    return kept


def reject_border_masks(
    masks: list[SAMMaskCandidate],
    border_px: int,
    max_border_fraction: float,
) -> list[SAMMaskCandidate]:
    out = []
    for c in masks:
        m = c.mask
        h, w = m.shape[:2]

        bp = min(border_px, h // 2, w // 2)
        if bp <= 0:
            out.append(c)
            continue

        border_pixels = (
            m[:bp, :].sum() + m[-bp:, :].sum()
            + m[bp:-bp, :bp].sum()
            + m[bp:-bp, -bp:].sum()
        )
        if c.area == 0:
            continue
        if float(border_pixels) / float(c.area) > max_border_fraction:
            continue
        out.append(c)
    return out


def mask_depth_coverage(
    depth: np.ndarray,
    mask: np.ndarray,
    zmin: float = 0.05,
    zmax: float = 3.0,
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


def bbox_containment_ratio(
    inner: tuple[int, int, int, int],
    outer: tuple[int, int, int, int],
) -> float:
    ix0, iy0, ix1, iy1 = inner
    ox0, oy0, ox1, oy1 = outer

    ax0 = max(ix0, ox0)
    ay0 = max(iy0, oy0)
    ax1 = min(ix1, ox1)
    ay1 = min(iy1, oy1)

    iw = max(0, ax1 - ax0)
    ih = max(0, ay1 - ay0)
    inter = iw * ih
    inner_area = max(1, (ix1 - ix0) * (iy1 - iy0))

    return float(inter) / float(inner_area)


def bbox_iou_xyxy(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    area_a = max(1, (ax1 - ax0) * (ay1 - ay0))
    area_b = max(1, (bx1 - bx0) * (by1 - by0))
    union = area_a + area_b - inter
    return float(inter) / float(union) if union > 0 else 0.0


def dedup_masks_by_bbox_iou(
    masks: list[SAMMaskCandidate],
    iou_thresh: float = 0.7,
    containment_thresh: float = 0.9,
) -> list[SAMMaskCandidate]:
    out = []
    masks_sorted = sorted(masks, key=lambda m: m.area, reverse=True)

    for m in masks_sorted:
        keep = True
        for k in out:
            if bbox_iou_xyxy(m.bbox_xyxy, k.bbox_xyxy) > iou_thresh:
                keep = False
                break
            if bbox_containment_ratio(m.bbox_xyxy, k.bbox_xyxy) > containment_thresh:
                keep = False
                break
        if keep:
            out.append(m)

    return out


def crop_rgb_to_polygon_bbox(
    rgb: np.ndarray,
    polygon: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    h, w = rgb.shape[:2]
    xs = polygon[:, 0]
    ys = polygon[:, 1]

    x0 = max(0, int(xs.min()))
    y0 = max(0, int(ys.min()))
    x1 = min(w, int(xs.max()) + 1)
    y1 = min(h, int(ys.max()) + 1)

    rgb_crop = rgb[y0:y1, x0:x1].copy()
    polygon_crop = polygon.copy()
    polygon_crop[:, 0] -= x0
    polygon_crop[:, 1] -= y0

    return rgb_crop, polygon_crop.astype(np.int32), x0, y0


def lift_crop_masks_to_full_image(
    crop_masks: list[SAMMaskCandidate],
    full_h: int,
    full_w: int,
    x0: int,
    y0: int,
) -> list[SAMMaskCandidate]:
    lifted = []
    for c in crop_masks:
        full_mask = np.zeros((full_h, full_w), dtype=c.mask.dtype)
        h, w = c.mask.shape[:2]
        full_mask[y0:y0 + h, x0:x0 + w] = c.mask

        bx0, by0, bx1, by1 = c.bbox_xyxy
        lifted.append(
            SAMMaskCandidate(
                mask=full_mask,
                bbox_xyxy=(bx0 + x0, by0 + y0, bx1 + x0, by1 + y0),
                area=int(c.area),
                score=float(c.score),
            )
        )
    return lifted


def find_blue_blob_masks(
    rgb: np.ndarray,
    min_area: int = 20,
    max_area: int = 3000,
) -> list[np.ndarray]:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    lower = np.array([95, 90, 70], dtype=np.uint8)
    upper = np.array([140, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    out = []
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        comp = (labels == i).astype(np.uint8)
        out.append(comp)
    return out


def blob_masks_to_candidates(blob_masks: list[np.ndarray]) -> list[SAMMaskCandidate]:
    out = []
    for m in blob_masks:
        ys, xs = np.where(m > 0)
        if xs.size == 0 or ys.size == 0:
            continue
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        out.append(
            SAMMaskCandidate(
                mask=m.astype(bool),
                bbox_xyxy=(x0, y0, x1, y1),
                area=int(m.sum()),
                score=1.0,
            )
        )
    return out


def batch_dino_classify(
    dino: DINOIdentifier,
    crops_rgb: list[np.ndarray],
    crops_mask: list[np.ndarray | None],
) -> list[DINOResult]:
    if not crops_rgb:
        return []

    tensors = []
    for rgb, mask in zip(crops_rgb, crops_mask):
        rgb_proc = dino._ensure_rgb(rgb)
        rgb_masked = dino._apply_mask(rgb_proc, mask)
        t = dino._preprocess(rgb_masked)
        tensors.append(t)

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
    results = [dino.classify_embedding(emb) for emb in embeddings]
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

        self.palette = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255),
        ]

        self.last_signature: Optional[tuple[tuple[str, int], ...]] = None
        self.busy = False
        self.frame_counter = 0

        self.mesh_map = self._build_mesh_map(args.cad_dir)

        self.track_states: dict[str, list[ObjectTrackState]] = {
            c.cam_id: [] for c in CAMERAS
        }

        self.cam_sam_params: dict[str, CameraSAMParams] = {
            "zed2i_1": CameraSAMParams(
                min_mask_area=args.cam1_sam_min_mask_area,
                min_bbox_side_px=args.cam1_sam_min_bbox_side_px,
                max_mask_area_ratio=args.cam1_sam_max_mask_area_ratio,
                max_bbox_area_ratio=args.cam1_sam_max_bbox_area_ratio,
                border_px=args.cam1_sam_border_px,
                max_border_fraction=args.cam1_sam_max_border_fraction,
                roi_polygon=parse_polygon_string(args.cam1_roi_polygon),
            ),
            "zed2i_2": CameraSAMParams(
                min_mask_area=args.cam2_sam_min_mask_area,
                min_bbox_side_px=args.cam2_sam_min_bbox_side_px,
                max_mask_area_ratio=args.cam2_sam_max_mask_area_ratio,
                max_bbox_area_ratio=args.cam2_sam_max_bbox_area_ratio,
                border_px=args.cam2_sam_border_px,
                max_border_fraction=args.cam2_sam_max_border_fraction,
                roi_polygon=parse_polygon_string(args.cam2_roi_polygon),
            ),
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
            f"DINO ready | objects={sorted(set(r.object_id for r in self.dino.reference_bank))}"
        )

        self.sam_by_cam: dict[str, SAMSegmenter] = {}
        self.sam_tiny_by_cam: dict[str, SAMSegmenter] = {}

        if args.mask_source == "sam":
            for cam in CAMERAS:
                cam_params = self.cam_sam_params[cam.cam_id]
                self.sam_by_cam[cam.cam_id] = SAMSegmenter(
                    SAMSegmenterConfig(
                        repo_root=args.sam_repo_root,
                        checkpoint=args.sam_checkpoint,
                        model_cfg=args.sam_model_cfg,
                        device=args.device,
                        max_image_side=args.sam_max_image_side,
                        min_mask_area=cam_params.min_mask_area,
                        min_bbox_side_px=cam_params.min_bbox_side_px,
                        attach_rgb_crops=False,
                    )
                )

            if args.tiny_objects_enabled:
                self.sam_tiny_by_cam["zed2i_2"] = SAMSegmenter(
                    SAMSegmenterConfig(
                        repo_root=args.sam_repo_root,
                        checkpoint=args.sam_checkpoint,
                        model_cfg=args.sam_model_cfg,
                        device=args.device,
                        max_image_side=max(args.sam_max_image_side, args.tiny_sam_max_image_side),
                        min_mask_area=args.tiny_sam_min_mask_area,
                        min_bbox_side_px=args.tiny_sam_min_bbox_side_px,
                        attach_rgb_crops=False,
                    )
                )

        self.projected_provider = ProjectedMaskProvider()

        self.fp_tracker_by_cam: dict[str, FoundationPoseWrapper] = {}
        for cam in CAMERAS:
            self.fp_tracker_by_cam[cam.cam_id] = FoundationPoseWrapper(
                FoundationPoseConfig(
                    repo_root=args.fp_repo_root,
                    weights_dir=args.fp_weights_dir,
                    debug_dir=str(Path(args.output_root).resolve() / f"fp_debug_{cam.cam_id}"),
                    debug=args.fp_debug,
                    est_refine_iter=args.est_refine_iter,
                    mesh_scale=args.mesh_scale,
                )
            )
        
        # Real-time tracker (CuteVOS + ICP) — one per camera
        self.realtime_trackers: dict[str, RealtimeTracker] = {}
        self.rt_active: dict[str, bool] = {c.cam_id: False for c in CAMERAS}

        self.pub_pose_base: dict[str, Any] = {}
        self.pub_pose_base_init: dict[str, Any] = {}
        self.pub_pose_base_track: dict[str, Any] = {}
        self.pub_debug_frame: dict[str, Any] = {}

        for c in CAMERAS:
            cid = c.cam_id
            self.pub_debug_frame[cid] = self.create_publisher(
                DebugFrame, f"/perception/fp/debug_frame/{cid}", FAST_QOS
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

    def _safe_to_base_pose(self, cam_id: str, T_object_camera: np.ndarray) -> np.ndarray:
        try:
            return self._to_base_pose(cam_id, T_object_camera)
        except Exception:
            return np.asarray(T_object_camera, dtype=np.float32).reshape(4, 4)

    def _base_pose_string(self, cam_id: str, T_object_camera: np.ndarray) -> str:
        T_base = self._safe_to_base_pose(cam_id, T_object_camera)
        t = T_base[:3, 3]
        q = rotation_matrix_to_quaternion_xyzw(T_base[:3, :3])
        return (
            f"t_base=[{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}] "
            f"q_base=[{q[0]:.3f}, {q[1]:.3f}, {q[2]:.3f}, {q[3]:.3f}]"
        )

    def _log_base_pose(
        self,
        stage: str,
        cam_id: str,
        object_id: str,
        idx: int,
        T_object_camera: np.ndarray,
        extra: str = "",
    ) -> None:
        pose_str = self._base_pose_string(cam_id, T_object_camera)
        suffix = f" | {extra}" if extra else ""
        self.get_logger().info(
            f"[{cam_id}] {stage} [{idx}] {object_id} | {pose_str}{suffix}"
        )

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
        self,
        cam_id: str,
        object_id: str,
        idx: int,
        T_object_camera: np.ndarray,
        stamp,
    ) -> None:
        T_base_object = self._to_base_pose(cam_id, T_object_camera)
        pub = self._get_or_create_pose_base_pub(cam_id, object_id, idx)
        pub.publish(T_to_pose_stamped(T_base_object, frame_id="base", stamp=stamp))

    def _publish_pose_base_init(
        self,
        cam_id: str,
        object_id: str,
        idx: int,
        T_object_camera: np.ndarray,
        stamp,
    ) -> None:
        T_base_object = self._to_base_pose(cam_id, T_object_camera)
        pub = self._get_or_create_pose_base_init_pub(cam_id, object_id, idx)
        pub.publish(T_to_pose_stamped(T_base_object, frame_id="base", stamp=stamp))

    def _publish_pose_base_track(
        self,
        cam_id: str,
        object_id: str,
        idx: int,
        T_object_camera: np.ndarray,
        stamp,
    ) -> None:
        T_base_object = self._to_base_pose(cam_id, T_object_camera)
        pub = self._get_or_create_pose_base_track_pub(cam_id, object_id, idx)
        pub.publish(T_to_pose_stamped(T_base_object, frame_id="base", stamp=stamp))

    def _make_mask_crop_msg(
        self,
        mask: np.ndarray,
        bbox_xyxy: tuple[int, int, int, int],
        max_side: int = 96,
    ) -> tuple[bool, DebugMaskCrop]:
        msg = DebugMaskCrop()
        x0, y0, x1, y1 = [int(v) for v in bbox_xyxy]

        if x1 <= x0 or y1 <= y0:
            return False, msg

        crop = mask[y0:y1, x0:x1].astype(np.uint8) * 255
        if crop.size == 0 or int(crop.sum()) == 0:
            return False, msg

        h, w = crop.shape[:2]
        if max(h, w) > max_side:
            scale = float(max_side) / float(max(h, w))
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        msg.width = int(crop.shape[1])
        msg.height = int(crop.shape[0])
        msg.data = crop.reshape(-1).tolist()
        return True, msg

    def _build_debug_frame(
        self,
        cam_id: str,
        stamp,
        update_sam: bool,
        update_dino: bool,
        sam_candidates: list[DebugCandidate],
        dino_candidates: list[DebugCandidate],
        pose_items: list[DebugPoseItem],
    ) -> DebugFrame:
        frame = DebugFrame()
        frame.stamp = stamp
        frame.cam_id = cam_id
        frame.max_candidate_draw = int(self.args.max_candidate_draw)
        frame.show_axes = True

        roi = self.cam_sam_params[cam_id].roi_polygon.reshape(-1)
        frame.roi_polygon_xy_flat = [int(v) for v in roi.tolist()]

        if cam_id == "zed2i_2" and self.args.tiny_objects_enabled:
            x0, y0, x1, y1 = parse_xyxy_string(self.args.cam2_tiny_roi)
            frame.has_tiny_roi = True
            frame.tiny_roi_xyxy = [int(x0), int(y0), int(x1), int(y1)]
        else:
            frame.has_tiny_roi = False
            frame.tiny_roi_xyxy = [0, 0, 0, 0]

        frame.update_sam = bool(update_sam)
        frame.update_dino = bool(update_dino)
        frame.sam_candidates = sam_candidates
        frame.dino_ranked_candidates = dino_candidates
        frame.pose_items = pose_items
        return frame

    def _sam_candidates_to_msgs(
        self,
        masks: list[SAMMaskCandidate],
    ) -> list[DebugCandidate]:
        out: list[DebugCandidate] = []
        for m in masks:
            msg = DebugCandidate()
            msg.object_id = ""
            msg.score = float(m.score)
            msg.bbox_xyxy = [int(v) for v in m.bbox_xyxy]
            ok_mask, mask_msg = self._make_mask_crop_msg(m.mask, m.bbox_xyxy)
            msg.has_mask = bool(ok_mask)
            msg.mask = mask_msg
            out.append(msg)
        return out

    def _dino_ranked_to_msgs(
        self,
        ranked: list[CandidateSelection],
    ) -> list[DebugCandidate]:
        out: list[DebugCandidate] = []
        for r in ranked:
            msg = DebugCandidate()
            msg.object_id = str(r.object_id)
            msg.score = float(r.score)
            msg.bbox_xyxy = [int(v) for v in r.candidate.bbox_xyxy]
            ok_mask, mask_msg = self._make_mask_crop_msg(r.candidate.mask, r.candidate.bbox_xyxy)
            msg.has_mask = bool(ok_mask)
            msg.mask = mask_msg
            out.append(msg)
        return out

    def _states_to_pose_item_msgs(
        self,
        cam_id: str,
        states: list[ObjectTrackState],
        include_masks: bool,
    ) -> list[DebugPoseItem]:
        out: list[DebugPoseItem] = []
        for s in states:
            if s.T_object_camera is None:
                continue

            msg = DebugPoseItem()
            msg.object_id = str(s.object_id)
            msg.mode = str(s.mode)
            msg.score = float(s.dino_score)
            msg.pose_camera = T_to_pose_msg(s.T_object_camera)
            msg.pose_base = T_to_pose_msg(self._safe_to_base_pose(cam_id, s.T_object_camera))
            msg.axis_len_m = 0.03

            msg.has_bbox = False
            msg.bbox_xyxy = [0, 0, 0, 0]
            msg.has_mask = False
            msg.mask = DebugMaskCrop()

            if include_masks and s.recovery_mask is not None:
                ys, xs = np.where(s.recovery_mask.astype(bool))
                if xs.size > 0 and ys.size > 0:
                    bbox = (
                        int(xs.min()),
                        int(ys.min()),
                        int(xs.max()) + 1,
                        int(ys.max()) + 1,
                    )
                    msg.has_bbox = True
                    msg.bbox_xyxy = [bbox[0], bbox[1], bbox[2], bbox[3]]
                    ok_mask, mask_msg = self._make_mask_crop_msg(s.recovery_mask, bbox)
                    msg.has_mask = bool(ok_mask)
                    msg.mask = mask_msg

            out.append(msg)
        return out

    def _resolve_mesh_path(self, object_id: str) -> str:
        if object_id in self.mesh_map:
            return self.mesh_map[object_id]

        for ext in (".obj", ".stl"):
            direct = Path(self.args.cad_dir) / f"{object_id}{ext}"
            if direct.exists():
                return str(direct)

        raise FileNotFoundError(f"No CAD mesh for object_id='{object_id}'")

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

    def _generate_tiny_object_masks_cam2(self, rgb: np.ndarray) -> list[SAMMaskCandidate]:
        if "zed2i_2" not in self.sam_tiny_by_cam:
            return []

        x0, y0, x1, y1 = parse_xyxy_string(self.args.cam2_tiny_roi)
        h, w = rgb.shape[:2]
        x0 = max(0, min(x0, w - 1))
        x1 = max(x0 + 1, min(x1, w))
        y0 = max(0, min(y0, h - 1))
        y1 = max(y0 + 1, min(y1, h))

        crop = rgb[y0:y1, x0:x1].copy()
        if crop.size == 0:
            return []

        sam = self.sam_tiny_by_cam["zed2i_2"]
        masks = sam.generate_auto(crop)
        self.get_logger().info(f"[zed2i_2] Tiny-object SAM raw masks: {len(masks)}")

        if not masks:
            return []

        ch, cw = crop.shape[:2]
        masks = reject_large_masks(
            masks,
            ch,
            cw,
            max_mask_area_ratio=self.args.tiny_max_mask_area_ratio,
            max_bbox_area_ratio=self.args.tiny_max_bbox_area_ratio,
        )

        masks = [
            m for m in masks
            if m.area >= self.args.tiny_sam_min_mask_area
            and (m.bbox_xyxy[2] - m.bbox_xyxy[0]) >= self.args.tiny_sam_min_bbox_side_px
            and (m.bbox_xyxy[3] - m.bbox_xyxy[1]) >= self.args.tiny_sam_min_bbox_side_px
        ]

        lifted = lift_crop_masks_to_full_image(masks, h, w, x0, y0)
        self.get_logger().info(f"[zed2i_2] Tiny-object SAM kept masks: {len(lifted)}")
        return lifted

    def _generate_and_filter_masks(self, rgb: np.ndarray, cam_id: str) -> list[SAMMaskCandidate]:
        if cam_id not in self.sam_by_cam:
            return []

        sam = self.sam_by_cam[cam_id]
        cam_params = self.cam_sam_params[cam_id]
        full_h, full_w = rgb.shape[:2]
        polygon_full = cam_params.roi_polygon

        rgb_crop, polygon_crop, crop_x0, crop_y0 = crop_rgb_to_polygon_bbox(rgb, polygon_full)
        crop_h, crop_w = rgb_crop.shape[:2]

        roi_mask_crop = np.zeros((crop_h, crop_w), dtype=np.uint8)
        cv2.fillPoly(roi_mask_crop, [polygon_crop], 255)
        rgb_crop_masked = rgb_crop.copy()
        rgb_crop_masked[roi_mask_crop == 0] = 0

        masks_crop = sam.generate_auto(rgb_crop_masked)
        self.get_logger().info(
            f"[{cam_id}] SAM generated {len(masks_crop)} masks on ROI crop "
            f"({crop_w}x{crop_h}) from full image ({full_w}x{full_h})"
        )

        if not masks_crop:
            masks = []
        else:
            masks_crop = reject_large_masks(
                masks_crop,
                crop_h,
                crop_w,
                cam_params.max_mask_area_ratio,
                cam_params.max_bbox_area_ratio,
            )
            self.get_logger().info(f"[{cam_id}] After size filter: {len(masks_crop)}")

            masks_crop = reject_border_masks(
                masks_crop,
                cam_params.border_px,
                cam_params.max_border_fraction
            )
            self.get_logger().info(f"[{cam_id}] After border filter: {len(masks_crop)}")

            masks_crop = reject_outside_roi_polygon(masks_crop, polygon_crop)
            self.get_logger().info(f"[{cam_id}] After ROI filter: {len(masks_crop)}")

            masks = lift_crop_masks_to_full_image(
                masks_crop, full_h, full_w, crop_x0, crop_y0
            )

        masks = sorted(masks, key=lambda m: m.area)

        if cam_id == "zed2i_2" and self.args.tiny_objects_enabled:
            tiny_masks = self._generate_tiny_object_masks_cam2(rgb)
            self.get_logger().info(f"[{cam_id}] Tiny-object SAM produced {len(tiny_masks)} masks")
            masks.extend(tiny_masks)

        if cam_id == "zed2i_2" and self.args.blue_blob_proposals_enabled:
            blue_blobs = find_blue_blob_masks(
                rgb,
                min_area=self.args.blue_blob_min_area,
                max_area=self.args.blue_blob_max_area,
            )
            blue_candidates = blob_masks_to_candidates(blue_blobs)
            self.get_logger().info(f"[{cam_id}] Blue blob proposals: {len(blue_candidates)}")
            masks.extend(blue_candidates)

        masks = reject_outside_roi_polygon(masks, cam_params.roi_polygon)
        self.get_logger().info(f"[{cam_id}] After ROI filter: {len(masks)}")

        masks = dedup_masks_by_bbox_iou(masks, iou_thresh=self.args.mask_dedup_iou)
        self.get_logger().info(f"[{cam_id}] After dedup: {len(masks)}")

        return masks

    def _classify_masks_batched(
        self,
        rgb: np.ndarray,
        masks: list[SAMMaskCandidate],
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
            dino_results = batch_dino_classify(self.dino, crops_rgb, crops_mask)
        except Exception as e:
            self.get_logger().warn(f"Batched DINO failed: {e}")
            return []

        out: list[CandidateSelection] = []
        img_area = float(rgb.shape[0] * rgb.shape[1])

        for j, res in enumerate(dino_results):
            mask_idx = valid_indices[j]
            cand = masks[mask_idx]

            sorted_scores = sorted(
                res.scores_by_object.items(), key=lambda kv: kv[1], reverse=True
            )

            top1_name, top1_score = sorted_scores[0]
            top2_name, top2_score = sorted_scores[1] if len(sorted_scores) > 1 else ("", -1.0)

            raw_best_score = float(top1_score)
            object_id = top1_name

            top_dbg = ", ".join([f"{k}:{v:.3f}" for k, v in sorted_scores[:4]])
            if object_id in {"cooling_f", "cooling_screw"} or raw_best_score < 0.80:
                self.get_logger().info(f"DINO cand {j} | top scores: {top_dbg}")

            pair_is_cf_cs = {top1_name, top2_name} == {"cooling_f", "cooling_screw"}
            pair_gap = float(top1_score - top2_score)
            geometric_resolved = False

            if pair_is_cf_cs and pair_gap < 0.15:
                bw_tmp, bh_tmp = bbox_size_xyxy(cand.bbox_xyxy)
                aspect = max(bw_tmp, bh_tmp) / (min(bw_tmp, bh_tmp) + 1e-6)
                if aspect > 2.2:
                    object_id = "cooling_f"
                    raw_best_score = float(res.scores_by_object["cooling_f"])
                else:
                    object_id = "cooling_screw"
                    raw_best_score = float(res.scores_by_object["cooling_screw"])
                geometric_resolved = True

            second_score = float(top2_score)
            margin = float(top1_score - second_score)

            bw, bh = bbox_size_xyxy(cand.bbox_xyxy)
            bbox_area = bw * bh
            is_small_object = bbox_area < 5000

            if is_small_object:
                min_score_for_small = 0.40
                min_margin_for_small = 0.02
                if raw_best_score < min_score_for_small:
                    object_id = "unknown"
                elif margin < min_margin_for_small and not geometric_resolved:
                    object_id = "unknown"

                if object_id != "unknown":
                    self.get_logger().info(
                        f"DINO small obj ACCEPT: {object_id} score={raw_best_score:.3f} margin={margin:.3f} bbox={bw}x{bh}"
                    )
            else:
                if raw_best_score < self.args.dino_min_score:
                    object_id = "unknown"
                if self.args.dino_min_margin > 0.0 and margin < self.args.dino_min_margin:
                    object_id = "unknown"

            bbox_max_side = max(bw, bh)
            if object_id == "cooling_base" and bbox_max_side < self.args.cooling_base_min_bbox_side_px:
                object_id = "unknown"

            area_ratio = float(cand.area) / img_area
            x0, y0, x1, y1 = cand.bbox_xyxy
            bbox_area = max(1, (x1 - x0) * (y1 - y0))
            fill_ratio = float(cand.area) / float(bbox_area)

            if object_id == "unknown":
                self.get_logger().info(
                    f"UNKNOWN: score={raw_best_score:.3f}, margin={margin:.3f}, bbox={bw}x{bh}"
                )

            final_score = (
                raw_best_score
                - self.args.area_penalty_weight * area_ratio
                + self.args.fill_ratio_weight * fill_ratio
            )

            out.append(
                CandidateSelection(
                    object_id=object_id,
                    score=final_score,
                    scores_by_object={k: float(v) for k, v in res.scores_by_object.items()},
                    candidate=cand,
                )
            )

        out.sort(key=lambda x: x.score, reverse=True)
        return out

    def _select_top_candidates(
        self,
        ranked: list[CandidateSelection],
        depth: np.ndarray,
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
            if mask.sum() > 0 and float(overlap) / float(mask.sum()) > 0.15:
                continue

            coverage = mask_depth_coverage(
                depth,
                mask,
                zmin=self.args.min_valid_z_m,
                zmax=self.args.max_valid_z_m,
            )
            if coverage < self.args.min_depth_coverage:
                continue

            selected.append(sel)
            used_pixels |= mask

        return selected

    def _pose_reason(self, T_camera: np.ndarray, cam_id: str) -> tuple[bool, str]:
        R = T_camera[:3, :3]
        t_cam = T_camera[:3, 3]

        trace = np.trace(R)
        if trace < -1.5:
            return False, f"flipped_orientation trace={trace:.3f}"

        t_mag = np.linalg.norm(t_cam)
        if t_mag < 0.4 or t_mag > 0.9:
            return False, f"bad_distance mag={t_mag:.3f}"

        try:
            T_base = self._to_base_pose(cam_id, T_camera)
            z_base = T_base[2, 3]
            if z_base < 0.0 or z_base > 0.5:
                return False, f"bad_z_base z={z_base:.3f}"
        except Exception:
            pass

        return True, "ok"

    def _try_local_recover(
        self,
        view: Any,
        state: ObjectTrackState,
        tracker: FoundationPoseWrapper,
    ) -> tuple[Optional[np.ndarray], Optional[str]]:
        if state.recovery_mask is None:
            return None, None

        rgb = view.rgb
        depth = view.depth
        K = np.asarray(view.K, dtype=np.float32).reshape(3, 3)

        try:
            result = tracker.estimate_pose(
                object_id=state.object_id,
                mesh_path=state.mesh_path,
                rgb=rgb,
                depth=depth,
                K=K,
                mask=state.recovery_mask,
            )
        except Exception:
            return None, None

        T_rec = np.asarray(result.T_object_camera, dtype=np.float32).reshape(4, 4)

        ok_pose, _ = self._pose_reason(T_rec, view.cam_id)
        if not ok_pose:
            return None, None

        try:
            same_result = tracker.track_pose(
                object_id=state.object_id,
                mesh_path=state.mesh_path,
                rgb=rgb,
                depth=depth,
                K=K,
                T_object_camera_init=T_rec,
            )
            T_same_raw = np.asarray(same_result.T_object_camera, dtype=np.float32).reshape(4, 4)
            jump_same_raw = float(np.linalg.norm(T_same_raw[:3, 3] - T_rec[:3, 3]))

            try:
                T_same_inv = np.linalg.inv(T_same_raw)
                jump_same_inv = float(np.linalg.norm(T_same_inv[:3, 3] - T_rec[:3, 3]))
            except np.linalg.LinAlgError:
                jump_same_inv = float("inf")

            recovered_convention = "inv" if jump_same_inv < jump_same_raw else "raw"
        except Exception:
            recovered_convention = state.track_pose_convention

        return T_rec, recovered_convention

    def _initialize_objects(
        self,
        view: Any,
        selections: list[CandidateSelection],
        stamp,
    ) -> list[ObjectTrackState]:
        rgb = view.rgb
        depth = view.depth
        K = np.asarray(view.K, dtype=np.float32).reshape(3, 3)
        cam_id = view.cam_id

        tracker = self.fp_tracker_by_cam[cam_id]
        new_states: list[ObjectTrackState] = []

        for i, sel in enumerate(selections):
            try:
                mesh_path = self._resolve_mesh_path(sel.object_id)
            except FileNotFoundError:
                continue

            try:
                result = tracker.estimate_pose(
                    object_id=sel.object_id,
                    mesh_path=mesh_path,
                    rgb=rgb,
                    depth=depth,
                    K=K,
                    mask=sel.candidate.mask,
                )
            except Exception as e:
                self.get_logger().warn(f"[{cam_id}] INIT [{i}] {sel.object_id} estimate_pose failed: {e}")
                continue

            T = np.asarray(result.T_object_camera, dtype=np.float32).reshape(4, 4)
            ok_pose, reason = self._pose_reason(T, cam_id)
            if not ok_pose:
                self.get_logger().info(f"[{cam_id}] INIT reject [{i}] {sel.object_id} | {reason}")
                continue

            track_pose_convention = "raw"
            try:
                same_result = tracker.track_pose(
                    object_id=sel.object_id,
                    mesh_path=mesh_path,
                    rgb=rgb,
                    depth=depth,
                    K=K,
                    T_object_camera_init=T,
                )
                T_same_raw = np.asarray(same_result.T_object_camera, dtype=np.float32).reshape(4, 4)
                jump_same_raw = float(np.linalg.norm(T_same_raw[:3, 3] - T[:3, 3]))

                try:
                    T_same_inv = np.linalg.inv(T_same_raw)
                    jump_same_inv = float(np.linalg.norm(T_same_inv[:3, 3] - T[:3, 3]))
                except np.linalg.LinAlgError:
                    jump_same_inv = float("inf")

                track_pose_convention = "inv" if jump_same_inv < jump_same_raw else "raw"
            except Exception:
                pass

            state = ObjectTrackState(
                object_id=sel.object_id,
                mesh_path=mesh_path,
                mode="track",
                T_object_camera=T.copy(),
                dino_score=float(sel.score),
                lost_count=0,
                last_mask_area=int(sel.candidate.mask.sum()),
                track_pose_convention=track_pose_convention,
                recovery_mask=sel.candidate.mask.copy(),
            )
            state.id_history.append(sel.object_id)
            new_states.append(state)

            self._publish_pose_base_init(cam_id, sel.object_id, i, T, stamp)
            self._publish_pose_base(cam_id, sel.object_id, i, T, stamp)
            self._log_base_pose(
                "INIT",
                cam_id,
                sel.object_id,
                i,
                T,
                extra=f"dino={sel.score:.3f} convention={track_pose_convention}",
            )
            state.last_logged_T_base = None
            state.last_logged_convention = None

        return new_states

    def _track_objects(
        self,
        view: Any,
        states: list[ObjectTrackState],
        stamp,
    ) -> list[ObjectTrackState]:
        """Track objects using CuteVOS + Colored ICP."""
        cam_id = view.cam_id
        rgb = view.rgb
        depth = view.depth
        K = np.asarray(view.K, dtype=np.float32).reshape(3, 3)

        surviving: list[ObjectTrackState] = []

        for i, state in enumerate(states):
            tracker_key = f"{cam_id}_{state.object_id}_{i}"

            # Initialize real-time tracker if not active
            if tracker_key not in self.realtime_trackers:
                try:
                    cfg = RealtimeTrackerConfig(
                        cutie_cfg=CutieConfig(max_internal_size=480),
                        icp_cfg=ICPConfig(
                            variant=ICPVariant.GENERALIZED,
                            max_correspondence_distance=0.10,
                            voxel_size=0.002,
                        ),
                        min_icp_fitness=0.20,
                        max_translation_per_frame=0.05,
                        lost_frames_before_reinit=5,
                        verbose=False,
                    )
                    rt = RealtimeTracker(cfg)

                    # Get mask from last known pose projection or recovery mask
                    init_mask = state.recovery_mask
                    if init_mask is None or init_mask.sum() < 100:
                        # No mask — can't init RT tracker, fall back to re-init
                        self.get_logger().warn(f"[{cam_id}] No mask for RT init, requesting re-init")
                        continue
                    
                    init_mask = np.asarray(init_mask).astype(bool)
                    if init_mask.ndim != 2:
                        self.get_logger().warn(f"[{cam_id}] Mask has wrong dims: {init_mask.shape}")
                        continue

                    print(f"[DEBUG] init_mask shape={init_mask.shape}, rgb shape={rgb.shape}")

                    rt.initialize(
                        rgb=rgb,
                        depth=depth,
                        mask=init_mask,
                        T_init=state.T_object_camera,
                        K=K,
                        mesh_path=state.mesh_path,
                    )
                    self.realtime_trackers[tracker_key] = rt
                    self.get_logger().info(f"[{cam_id}] RT tracker initialized for {state.object_id}")
                except Exception as e:
                    self.get_logger().warn(f"[{cam_id}] RT init failed: {e}")
                    continue

            # Run tracking
            rt = self.realtime_trackers[tracker_key]
            t0 = time.time()

            try:
                result = rt.track(rgb, depth, K)
            except Exception as e:
                self.get_logger().warn(f"[{cam_id}] RT track failed: {e}")
                del self.realtime_trackers[tracker_key]
                continue

            elapsed_ms = (time.time() - t0) * 1000
            self.get_logger().info(
                f"[{cam_id}] RT track {state.object_id}: {elapsed_ms:.1f}ms "
                f"fitness={result.icp_fitness:.2f}"
            )

            if not result.valid:
                self.get_logger().warn(f"[{cam_id}] RT lost {state.object_id}: {result.message}")
                del self.realtime_trackers[tracker_key]
                # Don't add to surviving — will trigger re-init
                continue

            # Update state
            T_new = result.T_object_camera
            state.T_object_camera = T_new.copy()
            state.recovery_mask = result.mask  # Update mask for next frame
            state.lost_count = 0
            state.mode = "track/rt"
            surviving.append(state)

            # Publish
            self._publish_pose_base_track(cam_id, state.object_id, i, T_new, stamp)
            self._publish_pose_base(cam_id, state.object_id, i, T_new, stamp)

        return surviving    

    def _process_single_view(self, view: Any) -> None:
        t_start = time.time()
        cam_id = view.cam_id

        if view.rgb is None or view.depth is None:
            return
        if view.rgb.shape[:2] != view.depth.shape[:2]:
            return

        stamp = self.get_clock().now().to_msg()

        rgb = view.rgb
        depth = view.depth
        states = self.track_states[cam_id]

        if self.args.run_mode == "track" and states and all(s.mode.startswith("track") or s.mode.startswith("recover") for s in states):
            t1 = time.time()
            surviving = self._track_objects(view, states, stamp)
            print(f"[DEBUG] _track_objects: {(time.time()-t1)*1000:.1f}ms")

            if surviving:
                self.track_states[cam_id] = surviving

                pose_items = self._states_to_pose_item_msgs(
                    cam_id,
                    surviving,
                    include_masks=False,
                )
                frame = self._build_debug_frame(
                    cam_id=cam_id,
                    stamp=stamp,
                    update_sam=False,
                    update_dino=False,
                    sam_candidates=[],
                    dino_candidates=[],
                    pose_items=pose_items,
                )
                self.pub_debug_frame[cam_id].publish(frame)
                print(f"[TRACK PATH] total time: {(time.time() - t_start)*1000:.1f}ms")
                return
            else:
                self.get_logger().info(f"[{cam_id}] TRACK -> REINIT")

        if self.args.mask_source != "sam":
            return

        masks = self._generate_and_filter_masks(rgb, cam_id)
        self.get_logger().info(f"[{cam_id}] SAM raw masks after filtering: {len(masks)}")

        if not masks:
            self.track_states[cam_id] = []
            frame = self._build_debug_frame(
                cam_id=cam_id,
                stamp=stamp,
                update_sam=True,
                update_dino=True,
                sam_candidates=[],
                dino_candidates=[],
                pose_items=[],
            )
            self.pub_debug_frame[cam_id].publish(frame)
            print(f"[DEBUG] Published debug frame for {cam_id}")

            return

        ranked = self._classify_masks_batched(rgb, masks)
        selected = self._select_top_candidates(ranked, depth)

        sam_msgs = self._sam_candidates_to_msgs(masks)
        dino_msgs = self._dino_ranked_to_msgs(ranked)

        if not selected:
            self.track_states[cam_id] = []
            frame = self._build_debug_frame(
                cam_id=cam_id,
                stamp=stamp,
                update_sam=True,
                update_dino=True,
                sam_candidates=sam_msgs,
                dino_candidates=dino_msgs,
                pose_items=[],
            )
            self.pub_debug_frame[cam_id].publish(frame)
            print(f"[DEBUG] Published debug frame for {cam_id}")

            return

        new_states = self._initialize_objects(view, selected, stamp)
        new_states = nms_by_position(new_states, position_threshold=0.03)

        if self.args.run_mode == "track":
            self.track_states[cam_id] = new_states
        else:
            self.track_states[cam_id] = []

        pose_items = self._states_to_pose_item_msgs(
            cam_id,
            new_states,
            include_masks=True,
        )

        frame = self._build_debug_frame(
            cam_id=cam_id,
            stamp=stamp,
            update_sam=True,
            update_dino=True,
            sam_candidates=sam_msgs,
            dino_candidates=dino_msgs,
            pose_items=pose_items,
        )
        self.pub_debug_frame[cam_id].publish(frame)
        print(f"[DEBUG] Published debug frame for {cam_id}")


        if new_states:
            self.get_logger().info(
                f"[{cam_id}] INIT done | masks={len(masks)} ranked={len(ranked)} "
                f"selected={len(selected)} initialized={len(new_states)}"
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

        print(f"[TICK] New frame at {time.time():.3f}")
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
    p.add_argument("--dino-min-score", type=float, default=0.55)
    p.add_argument("--dino-min-margin", type=float, default=0.0)
    p.add_argument("--area-penalty-weight", type=float, default=1.5)
    p.add_argument("--fill-ratio-weight", type=float, default=0.15)
    p.add_argument("--cooling-base-min-bbox-side-px", type=int, default=140)
    p.add_argument("--cooling-screw-max-bbox-side-px", type=int, default=90)
    p.add_argument("--cooling-f-min-bbox-side-px", type=int, default=60)

    p.add_argument("--sam-repo-root", default="external/sam2")
    p.add_argument(
        "--sam-checkpoint",
        default="external/sam2/checkpoints/sam2.1_hiera_base_plus.pt",
    )
    p.add_argument("--sam-model-cfg", default="configs/sam2.1/sam2.1_hiera_b+.yaml")
    p.add_argument("--sam-max-image-side", type=int, default=1536)

    p.add_argument("--cam1-sam-min-mask-area", type=int, default=150)
    p.add_argument("--cam1-sam-min-bbox-side-px", type=int, default=10)
    p.add_argument("--cam1-sam-max-mask-area-ratio", type=float, default=0.007)
    p.add_argument("--cam1-sam-max-bbox-area-ratio", type=float, default=0.007)
    p.add_argument("--cam1-sam-border-px", type=int, default=6)
    p.add_argument("--cam1-sam-max-border-fraction", type=float, default=0.00)

    p.add_argument("--cam2-sam-min-mask-area", type=int, default=20)
    p.add_argument("--cam2-sam-min-bbox-side-px", type=int, default=3)
    p.add_argument("--cam2-sam-max-mask-area-ratio", type=float, default=0.06)
    p.add_argument("--cam2-sam-max-bbox-area-ratio", type=float, default=0.06)
    p.add_argument("--cam2-sam-border-px", type=int, default=6)
    p.add_argument("--cam2-sam-max-border-fraction", type=float, default=0.00)

    p.add_argument("--cam1-roi-polygon", type=str,
        default="950,104,210,530,735,1080,1160,1080,1250,560,1630,320")
    p.add_argument("--cam2-roi-polygon", type=str,
        default="430,410,1190,160,1920,500,1920,780,1720,1080,940,1080")

    p.add_argument("--tiny-objects-enabled", action="store_true")
    p.add_argument("--cam2-tiny-roi", type=str, default="700,500,1350,1080")
    p.add_argument("--tiny-sam-max-image-side", type=int, default=1920)
    p.add_argument("--tiny-sam-min-mask-area", type=int, default=8)
    p.add_argument("--tiny-sam-min-bbox-side-px", type=int, default=2)
    p.add_argument("--tiny-max-mask-area-ratio", type=float, default=0.01)
    p.add_argument("--tiny-max-bbox-area-ratio", type=float, default=0.02)

    p.add_argument("--blue-blob-proposals-enabled", action="store_true")
    p.add_argument("--blue-blob-min-area", type=int, default=20)
    p.add_argument("--blue-blob-max-area", type=int, default=3000)
    p.add_argument("--mask-dedup-iou", type=float, default=0.6)

    p.add_argument("--fp-repo-root", default="external/FoundationPose")
    p.add_argument("--fp-weights-dir", default="external/FoundationPose/weights")
    p.add_argument("--fp-debug", type=int, default=0)
    p.add_argument("--est-refine-iter", type=int, default=0)
    p.add_argument("--mesh-scale", type=float, default=0.01)

    p.add_argument("--timer-period-s", type=float, default=0.25)
    p.add_argument("--max-candidate-draw", type=int, default=25)

    p.add_argument("--min-valid-z-m", type=float, default=0.05)
    p.add_argument("--max-valid_z_m", dest="max_valid_z_m", type=float, default=10.00)
    p.add_argument("--max-translation-jump-m", type=float, default=0.80)

    p.add_argument("--max-objects", type=int, default=15)
    p.add_argument("--max-lost-count", type=int, default=10)

    p.add_argument("--min-depth-coverage", type=float, default=0.50)
    p.add_argument("--track-log-trans-thresh-m", type=float, default=0.005)
    p.add_argument("--track-log-rot-thresh-deg", type=float, default=4.0)

    return p.parse_args()


def main() -> None:
    torch.cuda.empty_cache()
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