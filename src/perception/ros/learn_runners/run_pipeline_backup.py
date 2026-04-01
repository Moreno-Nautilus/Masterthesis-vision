from __future__ import annotations

import argparse
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import os
import time

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
    # fp_tracker: Optional[FoundationPoseWrapper] = None  # REMOVED - now shared per camera
    mode: str = "search"  # "search" | "track"
    T_object_camera: Optional[np.ndarray] = None
    dino_score: float = 0.0
    lost_count: int = 0
    last_mask_area: int = 0
    id_history: deque = field(default_factory=lambda: deque(maxlen=5))
    track_pose_convention: str = "raw"   # "raw" or "inv"
    recovery_mask: Optional[np.ndarray] = None

    last_logged_T_base: Optional[np.ndarray] = None
    last_logged_convention: Optional[str] = None

@dataclass
class CandidateSelection:
    object_id: str
    score: float
    scores_by_object: dict[str, float]
    candidate: SAMMaskCandidate


# @dataclass
# class ObjectTrackState:
#     object_id: str
#     mesh_path: str
#     fp_tracker: Optional[FoundationPoseWrapper] = None
#     mode: str = "search"  # "search" | "track"
#     T_object_camera: Optional[np.ndarray] = None
#     dino_score: float = 0.0
#     lost_count: int = 0
#     last_mask_area: int = 0
#     id_history: deque = field(default_factory=lambda: deque(maxlen=5))
#     track_pose_convention: str = "raw"   # "raw" or "inv"
#     recovery_mask: Optional[np.ndarray] = None

#     last_logged_T_base: Optional[np.ndarray] = None
#     last_logged_convention: Optional[str] = None


@dataclass
class CameraSAMParams:
    min_mask_area: int
    min_bbox_side_px: int
    max_mask_area_ratio: float
    max_bbox_area_ratio: float
    border_px: int
    max_border_fraction: float
    roi_polygon: np.ndarray  # Nx2 polygon points
    # roi_left_px: int
    # roi_right_px: int
    # roi_top_px: int
    # roi_bottom_px: int


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


def draw_roi_polygon(
    image: np.ndarray,
    polygon: np.ndarray,
    color: tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
    label: str = "ROI",
) -> np.ndarray:
    out = image.copy()
    cv2.polylines(out, [polygon], isClosed=True, color=color, thickness=thickness)
    # Put label near first point
    cv2.putText(
        out, label, (polygon[0, 0] + 8, polygon[0, 1] - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA,
    )
    return out


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

def parse_polygon_string(s: str) -> np.ndarray:
    """Parse 'x1,y1,x2,y2,...' string to Nx2 polygon array."""
    vals = [int(v.strip()) for v in s.split(",")]
    if len(vals) % 2 != 0:
        raise ValueError(f"Polygon string must have even number of values: {s}")
    return np.array(vals, dtype=np.int32).reshape(-1, 2)

def draw_mask_overlay(
    rgb: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: float = 0.30,
) -> np.ndarray:
    out = rgb.copy()
    color_arr = np.array(color, dtype=np.uint8).reshape(1, 1, 3)
    mask3 = mask.astype(bool)[..., None]
    blended = ((1.0 - alpha) * out + alpha * color_arr).astype(np.uint8)
    return np.where(mask3, blended, out)


def draw_bbox_label(
    image: np.ndarray,
    bbox_xyxy: tuple[int, int, int, int],
    text: str,
    color: tuple[int, int, int],
    font_scale: float = 0.6,
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
    image: np.ndarray,
    object_id: str,
    dino_score: float,
    T_display: np.ndarray,
    mode: str = "init",
    obj_idx: int = 0,
) -> np.ndarray:
    out = image.copy()
    t = T_display[:3, 3]
    lines = [
        f"[{obj_idx}] {mode}: {object_id}",
        f"  dino: {dino_score:.3f}",
        f"  t_base=[{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}]",
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
    polygon: np.ndarray,  # Nx2 array of points
) -> list[SAMMaskCandidate]:
    """Keep only masks whose bbox center is inside the polygon."""
    import cv2
    kept = []
    for m in masks:
        x0, y0, x1, y1 = m.bbox_xyxy
        cx = (x0 + x1) // 2
        cy = (y0 + y1) // 2
        # pointPolygonTest returns >0 if inside, 0 on edge, <0 outside
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


def batch_dino_classify(
    dino: DINOIdentifier,
    crops_rgb: list[np.ndarray],
    crops_mask: list[np.ndarray | None],
) -> list[DINOResult]:
    if not crops_rgb:
        return []
    
    debug_dir = "/workspace/MasterThesis/outputs/DINODEBUG"
    os.makedirs(f"{debug_dir}/query_crops", exist_ok=True)
    os.makedirs(f"{debug_dir}/matches", exist_ok=True)  # ADD THIS
    ts = int(time.time() * 1000)
    
    print(f"[DINO DEBUG] Saving {len(crops_rgb)} crops to {debug_dir}/query_crops/")

    tensors = []
    for i, (rgb, mask) in enumerate(zip(crops_rgb, crops_mask)):
        rgb_proc = dino._ensure_rgb(rgb)
        rgb_masked = dino._apply_mask(rgb_proc, mask)
        
        # DEBUG: Save the masked crop before preprocessing
        save_path = f"{debug_dir}/query_crops/crop_{ts}_{i}.png"
        cv2.imwrite(save_path, cv2.cvtColor(rgb_masked, cv2.COLOR_RGB2BGR))
        
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

    results = []
    for i, emb in enumerate(embeddings):
        res = dino.classify_embedding(emb)
        results.append(res)
        
        # DEBUG: Save side-by-side comparison with best matching reference
        best_ref_path = None
        best_sim = -1.0
        for ref in dino.reference_bank:
            if ref.object_id == res.object_id:
                sim = float(np.dot(emb, ref.embedding))
                if sim > best_sim:
                    best_sim = sim
                    best_ref_path = ref.image_path
        
        if best_ref_path and os.path.exists(best_ref_path):
            # Load reference image
            ref_bgr = cv2.imread(best_ref_path)
            if ref_bgr is not None:
                # Load query crop we just saved
                query_bgr = cv2.imread(f"{debug_dir}/query_crops/crop_{ts}_{i}.png")
                if query_bgr is not None:
                    # Resize both to same height
                    h = 150
                    q_h, q_w = query_bgr.shape[:2]
                    r_h, r_w = ref_bgr.shape[:2]
                    query_resized = cv2.resize(query_bgr, (int(q_w * h / q_h), h))
                    ref_resized = cv2.resize(ref_bgr, (int(r_w * h / r_h), h))
                    
                    # Pad to same width if needed
                    max_w = max(query_resized.shape[1], ref_resized.shape[1])
                    if query_resized.shape[1] < max_w:
                        pad = np.zeros((h, max_w - query_resized.shape[1], 3), dtype=np.uint8)
                        query_resized = np.hstack([query_resized, pad])
                    if ref_resized.shape[1] < max_w:
                        pad = np.zeros((h, max_w - ref_resized.shape[1], 3), dtype=np.uint8)
                        ref_resized = np.hstack([ref_resized, pad])
                    
                    # Stack vertically: query on top, reference below
                    combined = np.vstack([query_resized, ref_resized])
                    
                    # Add label
                    cv2.putText(combined, f"Q: crop_{i}", (5, 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(combined, f"R: {res.object_id} ({res.score:.3f})", (5, h + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    
                    cv2.imwrite(f"{debug_dir}/matches/match_{ts}_{i}_{res.object_id}_{res.score:.2f}.png", combined)

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
                # roi_left_px=args.cam1_roi_left_px,
                # roi_right_px=args.cam1_roi_right_px,
                # roi_top_px=args.cam1_roi_top_px,
                # roi_bottom_px=args.cam1_roi_bottom_px,
            ),
            "zed2i_2": CameraSAMParams(
                min_mask_area=args.cam2_sam_min_mask_area,
                min_bbox_side_px=args.cam2_sam_min_bbox_side_px,
                max_mask_area_ratio=args.cam2_sam_max_mask_area_ratio,
                max_bbox_area_ratio=args.cam2_sam_max_bbox_area_ratio,
                border_px=args.cam2_sam_border_px,
                max_border_fraction=args.cam2_sam_max_border_fraction,
                roi_polygon=parse_polygon_string(args.cam2_roi_polygon),
                # roi_left_px=args.cam2_roi_left_px,
                # roi_right_px=args.cam2_roi_right_px,
                # roi_top_px=args.cam2_roi_top_px,
                # roi_bottom_px=args.cam2_roi_bottom_px,
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

        self.projected_provider = ProjectedMaskProvider()

        self.fp_cfg = FoundationPoseConfig(
            repo_root=args.fp_repo_root,
            weights_dir=args.fp_weights_dir,
            debug_dir=str(Path(args.output_root).resolve() / "fp_internal_debug"),
            debug=args.fp_debug,
            est_refine_iter=args.est_refine_iter,
            mesh_scale=args.mesh_scale,
        )

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

        self.pub_raw: dict[str, Any] = {}
        self.pub_sam_overlay: dict[str, Any] = {}
        self.pub_dino_overlay: dict[str, Any] = {}
        self.pub_pose_overlay: dict[str, Any] = {}

        self.pub_pose_base: dict[str, Any] = {}
        self.pub_pose_base_init: dict[str, Any] = {}
        self.pub_pose_base_track: dict[str, Any] = {}

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
        T_object_camera = np.asarray(T_object_camera, dtype=np.float32).reshape(4, 4)
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
        T_object_camera = np.asarray(T_object_camera, dtype=np.float32).reshape(4, 4)
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
        T_object_camera = np.asarray(T_object_camera, dtype=np.float32).reshape(4, 4)
        T_base_object = self._to_base_pose(cam_id, T_object_camera)
        pub = self._get_or_create_pose_base_track_pub(cam_id, object_id, idx)
        pub.publish(T_to_pose_stamped(T_base_object, frame_id="base", stamp=stamp))

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

    # def _generate_and_filter_masks(self, rgb: np.ndarray, cam_id: str) -> list[SAMMaskCandidate]:
    #     if cam_id not in self.sam_by_cam:
    #         return []

    #     sam = self.sam_by_cam[cam_id]
    #     cam_params = self.cam_sam_params[cam_id]

    #     masks = sam.generate_auto(rgb)

    #     self.get_logger().info(f"[{cam_id}] SAM generated {len(masks)} raw masks")

    #     # DEBUG: Save all raw masks visualization
    #     debug_dir = "/workspace/MasterThesis/outputs/SAMDEBUG"
    #     os.makedirs(debug_dir, exist_ok=True)
    #     vis = rgb.copy()
    #     for i, m in enumerate(masks):
    #         color = self.palette[i % len(self.palette)]
    #         vis = draw_mask_overlay(vis, m.mask, color=color, alpha=0.3)
    #         x0, y0, x1, y1 = m.bbox_xyxy
    #         cv2.rectangle(vis, (x0, y0), (x1, y1), color, 1)
    #         cv2.putText(vis, f"{i}:{m.area}", (x0, max(10, y0-5)), 
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
    #     # Draw ROI polygon too
    #     cv2.polylines(vis, [cam_params.roi_polygon], True, (255, 255, 255), 2)
    #     ts = int(time.time() * 1000)
    #     cv2.imwrite(f"{debug_dir}/raw_masks_{cam_id}_{ts}.png", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    #     self.get_logger().info(f"[{cam_id}] Saved raw masks to {debug_dir}/raw_masks_{cam_id}_{ts}.png")

    #     if not masks:
    #         return []

    #     h, w = rgb.shape[:2]

    #     sorted_by_area = sorted(masks, key=lambda m: m.area)
    #     self.get_logger().info(f"[{cam_id}] Smallest 5 masks BEFORE filtering:")
    #     for m in sorted_by_area[:5]:
    #         self.get_logger().info(f"[{cam_id}]   area={m.area} bbox={m.bbox_xyxy}")


    #     masks_after_size = reject_large_masks(
    #         masks,
    #         h,
    #         w,
    #         max_mask_area_ratio=cam_params.max_mask_area_ratio,
    #         max_bbox_area_ratio=cam_params.max_bbox_area_ratio,
    #     )

    #     self.get_logger().info(f"[{cam_id}] After size filter: {len(masks_after_size)}")

    #     masks_after_border = reject_border_masks(
    #         masks_after_size,
    #         border_px=cam_params.border_px,
    #         max_border_fraction=cam_params.max_border_fraction,
    #     )
    #     self.get_logger().info(f"[{cam_id}] After border filter: {len(masks_after_border)}")


    #     # x_min = cam_params.roi_left_px
    #     # x_max = w - cam_params.roi_right_px
    #     # y_min = cam_params.roi_top_px
    #     # y_max = h - cam_params.roi_bottom_px

    #     masks_after_roi = reject_outside_roi_polygon(masks_after_border, cam_params.roi_polygon)
    #     self.get_logger().info(f"[{cam_id}] After ROI filter: {len(masks_after_roi)}")

    #     for i, m in enumerate(masks_after_roi[:10]):  # first 10
    #         self.get_logger().info(f"[{cam_id}]   mask[{i}] area={m.area} bbox={m.bbox_xyxy}")

    #     return masks_after_roi
    def _generate_and_filter_masks(self, rgb: np.ndarray, cam_id: str) -> list[SAMMaskCandidate]:
        if cam_id not in self.sam_by_cam:
            return []

        sam = self.sam_by_cam[cam_id]
        cam_params = self.cam_sam_params[cam_id]
        h, w = rgb.shape[:2]
        polygon = cam_params.roi_polygon

        # Black out everything outside polygon
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(roi_mask, [polygon], 255)
        rgb_masked = rgb.copy()
        rgb_masked[roi_mask == 0] = 0

        # Run SAM on masked image
        masks = sam.generate_auto(rgb_masked)
        self.get_logger().info(f"[{cam_id}] SAM generated {len(masks)} masks (ROI-masked input)")

        # DEBUG: Save all raw masks visualization
        debug_dir = "/workspace/MasterThesis/outputs/SAMDEBUG"
        os.makedirs(debug_dir, exist_ok=True)
        vis = rgb.copy()  # Use original RGB for visualization
        for i, m in enumerate(masks):
            color = self.palette[i % len(self.palette)]
            vis = draw_mask_overlay(vis, m.mask, color=color, alpha=0.3)
            x0, y0, x1, y1 = m.bbox_xyxy
            cv2.rectangle(vis, (x0, y0), (x1, y1), color, 1)
            cv2.putText(vis, f"{i}:{m.area}", (x0, max(10, y0-5)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        cv2.polylines(vis, [polygon], True, (255, 255, 255), 2)
        ts = int(time.time() * 1000)
        cv2.imwrite(f"{debug_dir}/raw_masks_{cam_id}_{ts}.png", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        self.get_logger().info(f"[{cam_id}] Saved raw masks to {debug_dir}/raw_masks_{cam_id}_{ts}.png")

        if not masks:
            return []

        # Filter by size
        masks = reject_large_masks(masks, h, w,
            cam_params.max_mask_area_ratio, cam_params.max_bbox_area_ratio)
        self.get_logger().info(f"[{cam_id}] After size filter: {len(masks)}")

        # Filter by border
        masks = reject_border_masks(masks, cam_params.border_px, cam_params.max_border_fraction)
        self.get_logger().info(f"[{cam_id}] After border filter: {len(masks)}")

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
            if mask.sum() > 0 and float(overlap) / float(mask.sum()) > 0.3:
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

    def _pose_reason(self, T: np.ndarray) -> tuple[bool, str]:
        R = T[:3, :3]
        t = T[:3, 3]

        trace = np.trace(R)
        if trace < -1.5:
            return False, f"flipped_orientation trace={trace:.3f}"

        z = t[2]
        if z < 0.01 or z > 1.5:
            return False, f"bad_z z={z:.3f}"

        t_mag = np.linalg.norm(t)
        if t_mag < 0.4 or t_mag > 0.9:
            return False, f"bad_distance mag={t_mag:.3f}"

        return True, "ok"

    def _draw_track_box(
        self,
        image: np.ndarray,
        cam_id: str,
        state: ObjectTrackState,
        obj_idx: int,
        mode: str,
    ) -> np.ndarray:
        out = image.copy()
        color = self.palette[obj_idx % len(self.palette)]

        if state.recovery_mask is not None:
            ys, xs = np.where(state.recovery_mask.astype(bool))
            if xs.size > 0 and ys.size > 0:
                x0, x1 = int(xs.min()), int(xs.max())
                y0, y1 = int(ys.min()), int(ys.max())
                out = draw_bbox_label(
                    out,
                    (x0, y0, x1, y1),
                    f"{state.object_id} {mode}",
                    color,
                    font_scale=0.6,
                )
                out = draw_mask_overlay(out, state.recovery_mask, color=color, alpha=0.18)

        if state.T_object_camera is not None:
            T_display = self._safe_to_base_pose(cam_id, state.T_object_camera)
            out = draw_pose_text(
                out,
                state.object_id,
                state.dino_score,
                T_display,
                mode=mode,
                obj_idx=obj_idx,
            )
        return out

    def _draw_sam_overlay(self, cam_id: str, rgb: np.ndarray, masks: list[SAMMaskCandidate]) -> np.ndarray:
        vis = rgb.copy()

        cam_params = self.cam_sam_params[cam_id]
        h, w = rgb.shape[:2]
        # x_min = cam_params.roi_left_px
        # x_max = w - cam_params.roi_right_px
        # y_min = cam_params.roi_top_px
        # y_max = h - cam_params.roi_bottom_px

        vis = draw_roi_polygon(vis, cam_params.roi_polygon, label=f"ROI {cam_id}")

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
        self,
        cam_id: str,
        stamp,
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

        ok_pose, _ = self._pose_reason(T_rec)
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
    ) -> tuple[list[ObjectTrackState], np.ndarray]:
        rgb = view.rgb
        depth = view.depth
        K = np.asarray(view.K, dtype=np.float32).reshape(3, 3)
        cam_id = view.cam_id

        # Use shared tracker for this camera
        tracker = self.fp_tracker_by_cam[cam_id]

        new_states: list[ObjectTrackState] = []
        pose_overlay = rgb.copy()

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
                torch.cuda.empty_cache()
                continue

            T = np.asarray(result.T_object_camera, dtype=np.float32).reshape(4, 4)

            ok_pose, reason = self._pose_reason(T)
            if not ok_pose:
                self.get_logger().info(
                    f"[{cam_id}] INIT reject [{i}] {sel.object_id} | {reason}"
                )
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
                # fp_tracker removed - using shared tracker
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
            T_display = self._safe_to_base_pose(cam_id, T)
            pose_overlay = draw_pose_text(
                pose_overlay, sel.object_id, sel.score, T_display, mode="init", obj_idx=i
            )

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
            torch.cuda.empty_cache()

        return new_states, pose_overlay

    def _track_objects(
        self,
        view: Any,
        states: list[ObjectTrackState],
        stamp,
    ) -> tuple[list[ObjectTrackState], np.ndarray]:
        rgb = view.rgb
        depth = view.depth
        K = np.asarray(view.K, dtype=np.float32).reshape(3, 3)
        cam_id = view.cam_id

        # Use shared tracker for this camera
        tracker = self.fp_tracker_by_cam[cam_id]

        surviving: list[ObjectTrackState] = []
        pose_overlay = rgb.copy()

        for i, state in enumerate(states):
            if state.T_object_camera is None:
                state.lost_count += 1
                if state.lost_count < self.args.max_lost_count:
                    surviving.append(state)
                else:
                    self.get_logger().info(
                        f"[{cam_id}] LOST [{i}] {state.object_id}"
                    )
                continue

            try:
                result = tracker.track_pose(
                    object_id=state.object_id,
                    mesh_path=state.mesh_path,
                    rgb=rgb,
                    depth=depth,
                    K=K,
                    T_object_camera_init=state.T_object_camera,
                )
            except Exception:
                state.lost_count += 1
                if state.lost_count < self.args.max_lost_count:
                    surviving.append(state)
                else:
                    self.get_logger().info(
                        f"[{cam_id}] LOST [{i}] {state.object_id}"
                    )
                continue

            prev = np.asarray(state.T_object_camera, dtype=np.float32).reshape(4, 4)
            T_raw = np.asarray(result.T_object_camera, dtype=np.float32).reshape(4, 4)

            try:
                T_inv = np.linalg.inv(T_raw)
                inv_valid = True
            except np.linalg.LinAlgError:
                T_inv = None
                inv_valid = False

            convention = getattr(state, "track_pose_convention", "raw")

            if convention == "inv":
                if not inv_valid:
                    state.lost_count += 1
                    if state.lost_count < self.args.max_lost_count:
                        surviving.append(state)
                    else:
                        self.get_logger().info(
                            f"[{cam_id}] LOST [{i}] {state.object_id}"
                        )
                    continue
                T_new = T_inv
            else:
                T_new = T_raw

            jump_m = float(np.linalg.norm(T_new[:3, 3] - prev[:3, 3]))
            ok_pose, reason = self._pose_reason(T_new)

            need_recover = False
            recover_reason = ""
            if not ok_pose:
                need_recover = True
                recover_reason = reason
            elif jump_m > self.args.max_translation_jump_m:
                need_recover = True
                recover_reason = f"jump={jump_m:.3f}m"

            if need_recover:
                T_rec, recovered_convention = self._try_local_recover(view, state, tracker)
                if T_rec is not None:
                    state.T_object_camera = T_rec.copy()
                    if recovered_convention is not None:
                        state.track_pose_convention = recovered_convention
                    state.lost_count = 0
                    state.mode = "track"
                    surviving.append(state)

                    smoothed_id = vote_object_id(state.id_history)
                    pose_overlay = self._draw_track_box(
                        pose_overlay,
                        cam_id,
                        state,
                        i,
                        mode=f"recover/{state.track_pose_convention}",
                    )

                    self._publish_pose_base_track(cam_id, state.object_id, i, T_rec, stamp)
                    self._publish_pose_base(cam_id, state.object_id, i, T_rec, stamp)
                    self._log_base_pose(
                        "RECOVER",
                        cam_id,
                        smoothed_id,
                        i,
                        T_rec,
                        extra=f"convention={state.track_pose_convention}",
                    )
                    state.last_logged_T_base = self._safe_to_base_pose(cam_id, T_rec).copy()
                    state.last_logged_convention = state.track_pose_convention
                    continue

                state.lost_count += 1
                if state.lost_count < self.args.max_lost_count:
                    surviving.append(state)
                else:
                    self.get_logger().info(
                        f"[{cam_id}] LOST [{i}] {state.object_id} | {recover_reason}"
                    )
                continue

            state.T_object_camera = T_new.copy()
            state.lost_count = 0
            state.mode = "track"
            surviving.append(state)

            smoothed_id = vote_object_id(state.id_history)
            pose_overlay = self._draw_track_box(
                pose_overlay,
                cam_id,
                state,
                i,
                mode=f"track/{convention}",
            )

            self._publish_pose_base_track(cam_id, state.object_id, i, T_new, stamp)
            self._publish_pose_base(cam_id, state.object_id, i, T_new, stamp)
            T_base_new = self._safe_to_base_pose(cam_id, T_new)

            if should_log_track_update(
                T_base_new=T_base_new,
                T_base_last=state.last_logged_T_base,
                convention_new=convention,
                convention_last=state.last_logged_convention,
                trans_thresh_m=self.args.track_log_trans_thresh_m,
                rot_thresh_deg=self.args.track_log_rot_thresh_deg,
            ):
                self._log_base_pose(
                    "TRACK",
                    cam_id,
                    smoothed_id,
                    i,
                    T_new,
                    extra=f"convention={convention}",
                )
                state.last_logged_T_base = T_base_new.copy()
                state.last_logged_convention = convention

        return surviving, pose_overlay


    def _process_single_view(self, view: Any) -> None:
        cam_id = view.cam_id

        if view.rgb is None or view.depth is None:
            return
        if view.rgb.shape[:2] != view.depth.shape[:2]:
            return

        stamp = self.get_clock().now().to_msg()
        rgb = view.rgb
        depth = view.depth
        states = self.track_states[cam_id]

        if self.args.run_mode == "track" and states and all(s.mode == "track" for s in states):
            surviving, pose_overlay = self._track_objects(view, states, stamp)

            if surviving:
                self.track_states[cam_id] = surviving

                sam_overlay_cached = self.last_sam_overlay.get(cam_id, rgb)
                dino_overlay_cached = self.last_dino_overlay.get(cam_id, rgb)

                self._publish_overlays(
                    cam_id,
                    stamp,
                    rgb,
                    sam_overlay=sam_overlay_cached,
                    dino_overlay=dino_overlay_cached,
                    pose_overlay=pose_overlay,
                )
                return
            else:
                self.get_logger().info(f"[{cam_id}] TRACK -> REINIT")

        if self.args.mask_source != "sam":
            return

        masks = self._generate_and_filter_masks(rgb, cam_id)

        self.get_logger().info(f"[{cam_id}] SAM raw masks after filtering: {len(masks)}")


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

        sam_overlay = self._draw_sam_overlay(cam_id, rgb, masks)

        ranked = self._classify_masks_batched(rgb, masks)
        dino_overlay = self._draw_dino_overlay(rgb, ranked) if ranked else rgb.copy()

        self.last_sam_overlay[cam_id] = sam_overlay.copy()
        self.last_dino_overlay[cam_id] = dino_overlay.copy()

        selected = self._select_top_candidates(ranked, depth)

        if not selected:
            self.track_states[cam_id] = []
            self._publish_overlays(cam_id, stamp, rgb, sam_overlay, dino_overlay, rgb)
            return

        new_states, pose_overlay = self._initialize_objects(view, selected, stamp)

        if self.args.run_mode == "track":
            self.track_states[cam_id] = new_states
        else:
            self.track_states[cam_id] = []

        self._publish_overlays(cam_id, stamp, rgb, sam_overlay, dino_overlay, pose_overlay)

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
    p.add_argument("--dino-min-score", type=float, default=0.65)
    p.add_argument("--dino-min-margin", type=float, default=0.005)

    p.add_argument("--sam-repo-root", default="external/sam2")
    p.add_argument(
        "--sam-checkpoint",
        default="external/sam2/checkpoints/sam2.1_hiera_base_plus.pt",
    )
    p.add_argument("--sam-model-cfg", default="configs/sam2.1/sam2.1_hiera_b+.yaml")
    p.add_argument("--sam-max-image-side", type=int, default=1024)#1024

    # Camera 1 SAM/filter params
    p.add_argument("--cam1-sam-min-mask-area", type=int, default=150)
    p.add_argument("--cam1-sam-min-bbox-side-px", type=int, default=10)
    p.add_argument("--cam1-sam-max-mask-area-ratio", type=float, default=0.007)
    p.add_argument("--cam1-sam-max-bbox-area-ratio", type=float, default=0.007)
    p.add_argument("--cam1-sam-border-px", type=int, default=6)
    p.add_argument("--cam1-sam-max-border-fraction", type=float, default=0.00)
    # p.add_argument("--cam1-roi-left-px", type=int, default=200)
    # p.add_argument("--cam1-roi-right-px", type=int, default=500)
    # p.add_argument("--cam1-roi-top-px", type=int, default=100)
    # p.add_argument("--cam1-roi-bottom-px", type=int, default=50)

    # Camera 2 SAM/filter params = current values
    p.add_argument("--cam2-sam-min-mask-area", type=int, default=150)
    p.add_argument("--cam2-sam-min-bbox-side-px", type=int, default=10)
    p.add_argument("--cam2-sam-max-mask-area-ratio", type=float, default=0.06)
    p.add_argument("--cam2-sam-max-bbox-area-ratio", type=float, default=0.06)
    p.add_argument("--cam2-sam-border-px", type=int, default=6)
    p.add_argument("--cam2-sam-max-border-fraction", type=float, default=0.00)
    # Camera 1 polygon ROI (list of x,y points defining table corners)
    p.add_argument("--cam1-roi-polygon", type=str, 
        default="950,104,210,530,735,1080,1160,1080,1250,560,1630,320",  # placeholder - you'll calibrate this
        help="Comma-separated x1,y1,x2,y2,... polygon points for cam1 ROI")

    # Camera 2 polygon ROI
    p.add_argument("--cam2-roi-polygon", type=str,
        default="300,530,1120,185,1813,480,1480,1080,850,1080",  # placeholder
        help="Comma-separated x1,y1,x2,y2,... polygon points for cam2 ROI")

    # p.add_argument("--cam2-roi-left-px", type=int, default=250)
    # p.add_argument("--cam2-roi-right-px", type=int, default=250)
    # p.add_argument("--cam2-roi-top-px", type=int, default=120)
    # p.add_argument("--cam2-roi-bottom-px", type=int, default=10)

    p.add_argument("--fp-repo-root", default="external/FoundationPose")
    p.add_argument("--fp-weights-dir", default="external/FoundationPose/weights")
    p.add_argument("--fp-debug", type=int, default=0)
    p.add_argument("--est-refine-iter", type=int, default=0)
    p.add_argument("--mesh-scale", type=float, default=0.01)

    p.add_argument("--timer-period-s", type=float, default=0.25)
    p.add_argument("--max-candidate-draw", type=int, default=25)

    p.add_argument("--min-valid-z-m", type=float, default=0.05)
    p.add_argument("--max-valid-z-m", type=float, default=10.00)
    p.add_argument("--max-translation-jump-m", type=float, default=0.80)

    p.add_argument("--max-objects", type=int, default=8)
    p.add_argument("--max-lost-count", type=int, default=8)

    p.add_argument("--min-depth-coverage", type=float, default=0.30)
    p.add_argument("--track-log-trans-thresh-m", type=float, default=0.005)
    p.add_argument("--track-log-rot-thresh-deg", type=float, default=4.0)

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