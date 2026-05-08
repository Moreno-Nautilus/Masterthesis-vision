from __future__ import annotations

import argparse
import array
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import rclpy
from geometry_msgs.msg import Pose, PoseStamped
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import Header
from geometry_msgs.msg import Vector3Stamped
import trimesh
from fp_debug_msgs.msg import DebugCandidate, DebugFrame, DebugMaskCrop, DebugPoseItem
from concurrent.futures import ThreadPoolExecutor, as_completed

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
from src.perception.tracking.cutie_tracker import CutieConfig, CutieTracker
from src.perception.tracking.icp_refiner import ICPConfig, ICPVariant
from dataclasses import dataclass, field
from typing import Optional
from src.perception.multicam_fusion import (
    FusionConfig,
    FusedDetection,
    run_multicam_fusion,
)
from typing import Any, Optional
import open3d as o3d
from src.perception.fused_multicam_helpers import (
    lift_masked_depth_to_base,
    merge_point_clouds,
    merge_point_clouds_weighted,
    mesh_to_pcd_cached,
    run_icp_in_base_frame,
    chamfer_distance_one_way,
    weighted_average_poses,
    MedianPoseBuffer,
    apply_x_bias_correction,
    fill_depth_holes_in_mask,
)


FAST_QOS = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
)
LATCHED_QOS = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
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

class PoseKalmanFilter:
    """
    Simple Kalman filter for 6DoF pose prediction.
    Tracks position and velocity, predicts next position.
    """
    
    def __init__(self, process_noise: float = 0.01, measurement_noise: float = 0.002):
        """
        Args:
            process_noise: How much we expect velocity to change (m/frame)
            measurement_noise: How noisy our pose measurements are (m)
        """
        # State: [x, y, z, vx, vy, vz]
        self.state = np.zeros(6, dtype=np.float64)
        
        # Covariance matrix
        self.P = np.eye(6, dtype=np.float64) * 0.1
        
        # Process noise (velocity can change)
        self.Q = np.eye(6, dtype=np.float64)
        self.Q[:3, :3] *= process_noise ** 2  # Position process noise
        self.Q[3:, 3:] *= (process_noise * 2) ** 2  # Velocity process noise (higher)
        
        # Measurement noise (position only)
        self.R = np.eye(3, dtype=np.float64) * measurement_noise ** 2
        
        # State transition matrix (constant velocity model)
        self.F = np.eye(6, dtype=np.float64)
        self.F[0, 3] = 1.0  # x += vx
        self.F[1, 4] = 1.0  # y += vy
        self.F[2, 5] = 1.0  # z += vz
        
        # Measurement matrix (we only measure position)
        self.H = np.zeros((3, 6), dtype=np.float64)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        
        self._initialized = False
        self._frame_count = 0
       
    
    def initialize(self, position: np.ndarray) -> None:
        """Initialize filter with first position."""
        self.state[:3] = position
        self.state[3:] = 0.0  # Zero initial velocity
        self.P = np.eye(6, dtype=np.float64) * 0.1
        self._initialized = True
        self._frame_count = 1
    
    def predict(self) -> np.ndarray:
        """
        Predict next position based on current state.
        Returns predicted [x, y, z].
        """
        if not self._initialized:
            return np.zeros(3)
        
        # Predict state
        state_pred = self.F @ self.state
        
        # Predict covariance
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return state_pred[:3].copy()
    
    def update(self, position: np.ndarray) -> None:
        """
        Update filter with new measured position.
        """
        if not self._initialized:
            self.initialize(position)
            return
        
        # Measurement residual
        y = position - self.H @ self.state
        
        # Residual covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state
        self.state = self.state + K @ y
        
        # Update covariance
        I = np.eye(6)
        self.P = (I - K @ self.H) @ self.P
        
        self._frame_count += 1

    
    def get_speed(self) -> float:
        """Get current speed in m/frame."""
        return float(np.linalg.norm(self.state[3:]))
    
    def get_predicted_position(self, frames_ahead: int = 1) -> np.ndarray:
        """Predict position N frames ahead."""
        pos = self.state[:3].copy()
        vel = self.state[3:].copy()
        return pos + vel * frames_ahead
    
    def reset(self) -> None:
        """Reset filter state."""
        self.state = np.zeros(6, dtype=np.float64)
        self.P = np.eye(6, dtype=np.float64) * 0.1
        self._initialized = False
        self._frame_count = 0
    
    @property
    def is_initialized(self) -> bool:
        return self._initialized

@dataclass
class ObjectTrackState:
    object_id: str
    mesh_path: str
    mode: str = "search"
    T_object_camera: Optional[np.ndarray] = None
    dino_score: float = 0.0
    lost_count: int = 0
    last_mask_area: int = 0
    degraded_count: int = 0
    id_history: deque = field(default_factory=lambda: deque(maxlen=5))
    track_pose_convention: str = "raw"
    recovery_mask: Optional[np.ndarray] = None
    last_good_mask: Optional[np.ndarray] = None
    last_good_T: Optional[np.ndarray] = None
    kalman: Optional[PoseKalmanFilter] = None  

    last_logged_T_base: Optional[np.ndarray] = None
    last_logged_convention: Optional[str] = None


    def __post_init__(self):
        if self.kalman is None:
            self.kalman = PoseKalmanFilter()


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

def save_init_pose_render(
    T_base: np.ndarray,
    model_pcd,
    obj_id: str,
    save_path: str,
    accepted: bool,
    gt_pos=None,
) -> None:
    """Render a 3D scatter of the init pose estimate using matplotlib (headless-safe)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    try:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        # Transform model cloud to base frame
        pts = np.asarray(model_pcd.points)
        R, t = T_base[:3, :3], T_base[:3, 3]
        pts_base = (R @ pts.T).T + t

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        color = '#3366CC' if accepted else '#CC3333'
        ax.scatter(pts_base[::3, 0], pts_base[::3, 1], pts_base[::3, 2],
                   s=1, c=color, alpha=0.4)

        # Draw pose axes (50mm)
        for axis_i, axis_color in enumerate(['r', 'g', 'b']):
            axis_end = t + R[:, axis_i] * 0.05
            ax.plot([t[0], axis_end[0]], [t[1], axis_end[1]], [t[2], axis_end[2]],
                    color=axis_color, linewidth=2)

        if gt_pos is not None:
            ax.scatter(*gt_pos, s=80, c='lime', edgecolors='black', zorder=5)

        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_title(f'{obj_id} | {"ACCEPT" if accepted else "REJECT"}')
        ax.view_init(elev=30, azim=135)

        # Equal aspect ratio
        all_pts = pts_base
        mid = all_pts.mean(axis=0)
        max_range = (all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2 * 1.2
        ax.set_xlim(mid[0]-max_range, mid[0]+max_range)
        ax.set_ylim(mid[1]-max_range, mid[1]+max_range)
        ax.set_zlim(mid[2]-max_range, mid[2]+max_range)

        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f"  [WARN] save_init_pose_render failed: {e}")


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


def apply_clahe_rgb(
    rgb: np.ndarray,
    clip_limit: float = 2.0,
    grid_size: int = 8,
) -> np.ndarray:
    """D1: CLAHE on the L channel of LAB. Boosts local contrast on the
    matte 3D-printed parts under flat lab lighting without colour shifts.
    Returns a new (H, W, 3) uint8 RGB array.
    """
    if rgb.size == 0:
        return rgb
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=(int(grid_size), int(grid_size)),
    )
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def upscale_crop_if_small(
    rgb: np.ndarray,
    mask: np.ndarray,
    min_side: int,
) -> tuple[np.ndarray, np.ndarray]:
    """A8: bicubic-upscale a small crop so DINOv2's input-size downsample
    doesn't throw away detail. RGB uses bicubic, mask uses nearest neighbour
    (so the boolean stays clean). No-op when min_side <= 0 or the crop is
    already large enough.
    """
    if min_side <= 0:
        return rgb, mask
    h, w = rgb.shape[:2]
    short = min(h, w)
    if short == 0 or short >= min_side:
        return rgb, mask
    scale = float(min_side) / float(short)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    rgb_up = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    mask_up = cv2.resize(
        mask.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST
    ).astype(mask.dtype)
    return rgb_up, mask_up


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

def pad_mask_for_fp(mask: np.ndarray, pad_px: int = 5) -> np.ndarray:
    """Dilate mask by pad_px pixels to give FP more context around the object."""
    if pad_px <= 0:
        return mask
    kernel = np.ones((2 * pad_px + 1, 2 * pad_px + 1), np.uint8)
    dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
    return dilated.astype(mask.dtype)

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

        # B6: pick reference source. "real" uses --reference-dir (existing
        # behaviour), "renders" uses --reference-renders-dir, "both" merges them.
        ref_source = args.reference_source
        primary_ref = (
            args.reference_renders_dir if ref_source == "renders"
            else args.reference_dir
        )
        extra_refs: list[str] = []
        if ref_source == "both":
            extra_refs.append(args.reference_renders_dir)

        self.dino = DINOIdentifier(
            DINOIdentifierConfig(
                model_name=args.dino_model_name,
                device=args.device,
                reference_dir=primary_ref,
                use_masked_background=False,
                embedding_mode=args.dino_embedding_mode,
                gem_p=float(args.dino_gem_p),
                similarity=args.dino_similarity,
                joint_score_alpha=float(args.dino_joint_score_alpha),
            )
        )
        self.get_logger().info(
            f"Building DINO reference bank | source={ref_source} primary={primary_ref}"
            + (f" extra={extra_refs}" if extra_refs else "")
        )
        self.dino.build_reference_bank_from_folder(extra_dirs=extra_refs)
        self.get_logger().info(
            f"DINO ready | objects={sorted(set(r.object_id for r in self.dino.reference_bank))}"
        )

        self.sam_by_cam: dict[str, SAMSegmenter] = {}
        self.sam_tiny_by_cam: dict[str, SAMSegmenter] = {}

        # gdino_sam reuses the same SAM instance — only the proposal stage
        # differs (boxes from Grounding DINO instead of automatic mask gen).
        sam_modes = {"sam", "gdino_sam"}
        if args.mask_source in sam_modes:
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

        # Grounding-DINO proposer (lazy-loaded) for the gdino_sam path.
        self.gdino_proposer = None
        if args.mask_source == "gdino_sam":
            from src.perception.learned.GDINO.grounding_dino_proposal import (
                GDINOConfig, GroundingDINOProposer,
            )
            text_prompts = [p.strip() for p in args.gdino_text_prompts.split(",") if p.strip()]
            self.gdino_proposer = GroundingDINOProposer(
                GDINOConfig(
                    model_id=args.gdino_model_id,
                    device=args.device,
                    box_threshold=float(args.gdino_box_threshold),
                    text_threshold=float(args.gdino_text_threshold),
                    max_boxes_per_image=int(args.gdino_max_boxes),
                    text_prompts=text_prompts,
                )
            )
            self.get_logger().info(
                f"GDINO proposer ready | model={args.gdino_model_id} "
                f"prompts={text_prompts}"
            )

            if args.tiny_objects_enabled:
                # A3/A4/A7: dedicated tiny-pass SAM config. Lighter checkpoint
                # (if provided), fewer prompt points, lower acceptance thresholds.
                # Defaults still produce a working pipeline if the user has not
                # downloaded a smaller checkpoint — falls back to main ckpt.
                # Bundle 11: build the tiny segmenter for every camera so the
                # tiny-pass runs on both cams (was previously zed2i_2-only).
                tiny_ckpt = args.tiny_sam_checkpoint or args.sam_checkpoint
                tiny_cfg = args.tiny_sam_model_cfg or args.sam_model_cfg
                tiny_sam_cfg = SAMSegmenterConfig(
                    repo_root=args.sam_repo_root,
                    checkpoint=tiny_ckpt,
                    model_cfg=tiny_cfg,
                    device=args.device,
                    max_image_side=max(args.sam_max_image_side, args.tiny_sam_max_image_side),
                    min_mask_area=args.tiny_sam_min_mask_area,
                    min_bbox_side_px=args.tiny_sam_min_bbox_side_px,
                    attach_rgb_crops=False,
                    auto_points_per_side=int(args.tiny_sam_points_per_side),
                    auto_pred_iou_thresh=float(args.tiny_sam_pred_iou_thresh),
                    auto_stability_score_thresh=float(args.tiny_sam_stability_score_thresh),
                    max_aspect_ratio=float(args.tiny_sam_max_aspect_ratio),
                )
                # Share weights across cams: build the model once and reuse.
                tiny_segmenter = SAMSegmenter(tiny_sam_cfg)
                for cam in CAMERAS:
                    self.sam_tiny_by_cam[cam.cam_id] = tiny_segmenter


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

        
        # Pre-cache meshes for faster first init
        self.get_logger().info("Pre-caching meshes for FoundationPose...")
        for obj_id, mesh_path in self.mesh_map.items():
            for cam in CAMERAS:
                self.fp_tracker_by_cam[cam.cam_id].preload_mesh(mesh_path=mesh_path, object_id=obj_id)
                break  # Only need to cache once per object (mesh is shared)
        self.get_logger().info(f"Pre-cached {len(self.mesh_map)} meshes")
        
        # Real-time tracker (CuteVOS + ICP) — one per camera
        self.realtime_trackers: dict[str, RealtimeTracker] = {}
        self.rt_active: dict[str, bool] = {c.cam_id: False for c in CAMERAS}

        self._fused_track_memory: dict[str, dict[str, Any]] = {}
        self._fused_icp_metrics: dict[str, dict[str, Any]] = {}
        self._fused_translation_kalman: dict[str, PoseKalmanFilter] = {}

        self.pub_pose_base: dict[str, Any] = {}
        self.pub_pose_base_track: dict[str, Any] = {}
        self.pub_debug_frame: dict[str, Any] = {}


        # Pre-warm Cutie to avoid 2.5s delay on first track
        self.get_logger().info("Pre-warming Cutie model...")
        self._cutie_prewarmer = CutieTracker(CutieConfig())
        self._cutie_prewarmer._lazy_load()
        self.get_logger().info("Cutie pre-warmed")

        for c in CAMERAS:
            cid = c.cam_id
            self.pub_debug_frame[cid] = self.create_publisher(
                DebugFrame, f"/perception/fp/debug_frame/{cid}", FAST_QOS
            )

        self.timer = self.create_timer(args.timer_period_s, self._tick)
        self.get_logger().info(
            f"FoundationPoseTrackerNode started | run_mode={self.args.run_mode}"
        )
        self._depth_bias_by_cam = {
            "zed2i_1": float(self.args.cam1_depth_bias_m),
            "zed2i_2": float(self.args.cam2_depth_bias_m),
            }
        self._fused_warmup_count = {}       # obj_id -> frames since init
        self._median_pose_buffers = {}      # obj_id -> MedianPoseBuffer
        self._T_cam_base_cache: dict[str, np.ndarray] = {}


        self._consecutive_chamfer_fails: dict[str, int] = {}
        self._last_reinit_time: dict[str, float] = {}
        self._init_chamfer_history: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=10)
        )
        # Bundle 11: (cam_id, object_id) -> {frame_idx, mask, occlusion_score,
        # T_object_camera}. Populated by _track_multicam_fused and consumed
        # by _process_multicam_init when --skip-dino-when-tracker-healthy.
        self._recent_tracker_health: dict[tuple[str, str], dict] = {}

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
    

    @staticmethod
    def _octahedral_rotations() -> list[np.ndarray]:
        """24 elements of the chiral octahedral group (cube symmetries)."""
        from scipy.spatial.transform import Rotation as SciRot
        return [
            np.eye(3),
            SciRot.from_euler('x',  90, degrees=True).as_matrix(),
            SciRot.from_euler('x', 180, degrees=True).as_matrix(),
            SciRot.from_euler('x', 270, degrees=True).as_matrix(),
            SciRot.from_euler('y',  90, degrees=True).as_matrix(),
            SciRot.from_euler('y', 180, degrees=True).as_matrix(),
            SciRot.from_euler('y', 270, degrees=True).as_matrix(),
            SciRot.from_euler('z',  90, degrees=True).as_matrix(),
            SciRot.from_euler('z', 180, degrees=True).as_matrix(),
            SciRot.from_euler('z', 270, degrees=True).as_matrix(),
            SciRot.from_euler('xz', [ 90,  90], degrees=True).as_matrix(),
            SciRot.from_euler('xz', [ 90, 270], degrees=True).as_matrix(),
            SciRot.from_euler('xz', [270,  90], degrees=True).as_matrix(),
            SciRot.from_euler('xz', [270, 270], degrees=True).as_matrix(),
            SciRot.from_euler('xy', [ 90,  90], degrees=True).as_matrix(),
            SciRot.from_euler('xy', [ 90, 270], degrees=True).as_matrix(),
            SciRot.from_euler('xy', [270,  90], degrees=True).as_matrix(),
            SciRot.from_euler('xy', [270, 270], degrees=True).as_matrix(),
            SciRot.from_euler('xz', [180,  90], degrees=True).as_matrix(),
            SciRot.from_euler('xz', [180, 270], degrees=True).as_matrix(),
            SciRot.from_euler('yz', [180,  90], degrees=True).as_matrix(),
            SciRot.from_euler('yz', [180, 270], degrees=True).as_matrix(),
            SciRot.from_rotvec(
                np.array([1, 1, 1], dtype=float) / np.sqrt(3) * np.radians(120)
            ).as_matrix(),
            SciRot.from_rotvec(
                np.array([1, 1, 1], dtype=float) / np.sqrt(3) * np.radians(240)
            ).as_matrix(),
        ]

    @staticmethod
    def _fibonacci_rotations(n: int) -> list[np.ndarray]:
        """n uniformly-distributed SO(3) rotations (deterministic seed). Identity first."""
        from scipy.spatial.transform import Rotation as SciRot
        if n <= 1:
            return [np.eye(3)]
        mats = [np.eye(3)]
        mats.extend(list(SciRot.random(num=n - 1, random_state=42).as_matrix()))
        return mats

    @staticmethod
    def _jitter_rotations(R_center: np.ndarray, n: int, max_deg: float, seed: int = 0) -> list[np.ndarray]:
        """n random rotations with axis-angle magnitude ≤ max_deg, applied to R_center."""
        from scipy.spatial.transform import Rotation as SciRot
        rng = np.random.default_rng(seed)
        axes = rng.normal(size=(n, 3))
        axes /= (np.linalg.norm(axes, axis=1, keepdims=True) + 1e-12)
        angles = rng.uniform(0.0, np.radians(max_deg), size=n)
        R_jit = SciRot.from_rotvec(axes * angles[:, None]).as_matrix()
        return [R_center @ R_j for R_j in R_jit]

    def _grid_chamfer(
        self,
        model_pcd,
        fused_cloud,
        per_cam_clouds,
        T: np.ndarray,
    ) -> float:
        """Chamfer score for grid evaluation. Optionally averages over per-cam clouds."""
        if (getattr(self.args, "icp_grid_cross_cam_chamfer", False)
                and per_cam_clouds is not None and len(per_cam_clouds) >= 2):
            return float(np.mean([
                chamfer_distance_one_way(model_pcd, c, T) for c in per_cam_clouds
            ]))
        return chamfer_distance_one_way(model_pcd, fused_cloud, T)

    def _icp_rotation_grid(
            self,
            t_base: np.ndarray,
            model_pcd,
            fused_cloud,
            per_cam_clouds=None,
        ) -> tuple:
            """
            Search rotation seeds around translation `t_base`, refine each by ICP,
            score by Chamfer (lowest = best). Translation stays fixed at the
            FP estimate throughout the seed search; ICP is free to translate.

            Default seed set: 24 elements of the chiral octahedral group.
            With --icp-grid-fibonacci, switches to N uniformly-sampled SO(3) seeds.
            With --icp-grid-prescreen, skips ICP for seeds whose raw-Chamfer
            already exceeds tau (typical seeds are far enough off that ICP
            won't recover — checking the seed first saves the ICP cost).
            With --icp-grid-second-pass, re-refines around the top-K winners
            with smaller angular jitter.
            With --icp-grid-cross-cam-chamfer, scores by mean Chamfer across
            the per-camera clouds (more discriminative for symmetric objects
            where one camera sees a featureless face).
            With --icp-grid-tie-by-inliers, breaks Chamfer ties by ICP fitness.

            Returns (best_T, best_chamfer).
            """
            use_fib = bool(getattr(self.args, "icp_grid_fibonacci", False))
            n_rot = int(getattr(self.args, "icp_grid_n_rot", 60))
            prescreen = bool(getattr(self.args, "icp_grid_prescreen", False))
            prescreen_tau = float(getattr(self.args, "icp_grid_prescreen_tau", 0.04))
            second_pass = bool(getattr(self.args, "icp_grid_second_pass", False))
            second_k = int(getattr(self.args, "icp_grid_second_pass_k", 3))
            second_n = int(getattr(self.args, "icp_grid_second_pass_n", 8))
            second_jitter = float(getattr(self.args, "icp_grid_second_pass_jitter_deg", 15.0))
            tie_by_inliers = bool(getattr(self.args, "icp_grid_tie_by_inliers", False))
            tie_chamfer_eps = float(getattr(self.args, "icp_grid_tie_chamfer_eps_m", 0.0005))

            CANDIDATES = (self._fibonacci_rotations(n_rot) if use_fib
                          else self._octahedral_rotations())

            # Each entry: (chamfer, T_refined, fitness, R_seed)
            scored: list[tuple[float, np.ndarray, float, np.ndarray]] = []

            for R_candidate in CANDIDATES:
                T_seed = np.eye(4, dtype=np.float32)
                T_seed[:3, :3] = R_candidate.astype(np.float32)
                T_seed[:3, 3] = t_base

                if prescreen:
                    raw_ch = chamfer_distance_one_way(model_pcd, fused_cloud, T_seed)
                    if raw_ch > prescreen_tau:
                        continue

                T_ref, fit, _ = run_icp_in_base_frame(
                    fused_cloud, model_pcd, T_seed, max_iteration=30,
                    variant=self.args.icp_variant,
                )
                if fit < 0.10:
                    continue
                ch = self._grid_chamfer(model_pcd, fused_cloud, per_cam_clouds, T_ref)
                scored.append((ch, T_ref, fit, R_candidate.astype(np.float32)))

            if not scored:
                return None, float('inf')

            scored.sort(key=lambda x: x[0])

            if second_pass and second_k > 0 and second_n > 0:
                for k_idx, (_, _, _, R_top) in enumerate(scored[:second_k]):
                    for R_jit in self._jitter_rotations(R_top, second_n, second_jitter, seed=42 + k_idx):
                        T_seed = np.eye(4, dtype=np.float32)
                        T_seed[:3, :3] = R_jit.astype(np.float32)
                        T_seed[:3, 3] = t_base

                        if prescreen:
                            raw_ch = chamfer_distance_one_way(model_pcd, fused_cloud, T_seed)
                            if raw_ch > prescreen_tau:
                                continue

                        T_ref, fit, _ = run_icp_in_base_frame(
                            fused_cloud, model_pcd, T_seed, max_iteration=30,
                            variant=self.args.icp_variant,
                        )
                        if fit < 0.10:
                            continue
                        ch = self._grid_chamfer(model_pcd, fused_cloud, per_cam_clouds, T_ref)
                        scored.append((ch, T_ref, fit, R_jit.astype(np.float32)))

                scored.sort(key=lambda x: x[0])

            best = scored[0]
            if tie_by_inliers and len(scored) >= 2:
                # Among entries within tie_chamfer_eps of the best, pick highest fitness.
                ties = [s for s in scored if s[0] <= best[0] + tie_chamfer_eps]
                best = max(ties, key=lambda s: s[2])

            return best[1], float(best[0])

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

    def _resolve_T_base_cam(self, cam_id: str) -> np.ndarray:
        """Get T_base_cam as a plain float64 4x4 array, cached."""
        if not hasattr(self, '_T_base_cam_cache'):
            self._T_base_cam_cache: dict[str, np.ndarray] = {}
        if cam_id not in self._T_base_cam_cache:
            T = self.T_base_cam_map[cam_id]
            if hasattr(T, "as_matrix"):
                T = T.as_matrix()
            elif hasattr(T, "matrix"):
                T = T.matrix
            self._T_base_cam_cache[cam_id] = np.asarray(T, dtype=np.float64).reshape(4, 4)
        return self._T_base_cam_cache[cam_id]

    def _resolve_T_cam_base(self, cam_id: str) -> np.ndarray:
        """Get T_cam_base (inverse of extrinsic), cached."""
        if cam_id not in self._T_cam_base_cache:
            T_bc = self._resolve_T_base_cam(cam_id)
            self._T_cam_base_cache[cam_id] = np.linalg.inv(T_bc).astype(np.float32)
        return self._T_cam_base_cache[cam_id]

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
        # array.array('B', bytes) is ~50x faster than tolist() for uint8[] fields
        # because rclpy serializes the buffer directly without per-element Python ops.
        msg.data = array.array('B', np.ascontiguousarray(crop).tobytes())
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
        track_debug: Optional[dict] = None,
        ) -> DebugFrame:
        frame = DebugFrame()
        frame.stamp = stamp
        frame.cam_id = cam_id
        frame.max_candidate_draw = int(self.args.max_candidate_draw)
        frame.show_axes = True

        roi = self.cam_sam_params[cam_id].roi_polygon.reshape(-1)
        frame.roi_polygon_xy_flat = [int(v) for v in roi.tolist()]

        roi = (
            self._tiny_roi_for_cam(cam_id)
            if self.args.tiny_objects_enabled else None
        )
        if roi is not None:
            x0, y0, x1, y1 = roi
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

        # Populate tracking mask if available
        if track_debug is not None:
            frame.has_track_mask = True
            frame.track_mask_bbox_xyxy = [int(v) for v in track_debug["bbox_xyxy"]]
            frame.track_object_id = str(track_debug["object_id"])
            frame.track_icp_fitness = float(track_debug["icp_fitness"])
            frame.track_icp_rmse_mm = float(track_debug["icp_rmse_mm"])
            
            # Create mask crop message
            ok_mask, mask_msg = self._make_mask_crop_msg(
                track_debug["mask"], 
                track_debug["bbox_xyxy"],
                max_side=128,  # Slightly larger for tracking visualization
            )
            if ok_mask:
                frame.track_mask = mask_msg
            else:
                frame.has_track_mask = False
        else:
            frame.has_track_mask = False
            frame.track_mask_bbox_xyxy = [0, 0, 0, 0]
            frame.track_object_id = ""
            frame.track_icp_fitness = 0.0
            frame.track_icp_rmse_mm = 0.0

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

            # 1) Convert raw camera-frame CAD pose to base frame.
            T_base_dbg = self._safe_to_base_pose(cam_id, s.T_object_camera)

            # 3) Store corrected base pose.
            msg.pose_base = T_to_pose_msg(T_base_dbg)

            # 4) IMPORTANT:
            # Recompute corrected camera-frame pose from corrected base pose.
            # Do NOT use raw s.T_object_camera here.
            T_cam_base = self._resolve_T_cam_base(cam_id)
            T_cam_dbg = (T_cam_base @ T_base_dbg).astype(np.float32)

            msg.pose_camera = T_to_pose_msg(T_cam_dbg)

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
        if cam_id not in self.T_base_cam_map:
            raise KeyError(f"No base extrinsic for cam_id={cam_id}")
        T_object_camera = np.asarray(T_object_camera, dtype=np.float32).reshape(4, 4)
        # Hits the _T_base_cam_cache populated lazily on first use; avoids the
        # as_matrix/asarray fallback chain on the per-state per-tick hot path.
        T_base_cam = self._resolve_T_base_cam(cam_id)
        return (T_base_cam @ T_object_camera.astype(np.float64)).astype(np.float32)


    def _inherit_from_tracker_health(
        self, cam_id: str, masks: list[SAMMaskCandidate],
    ) -> tuple[list[CandidateSelection], list[SAMMaskCandidate]]:
        """Bundle 11: assign masks to objects whose tracker was healthy on a
        recent frame, by IoU with the tracker's last good mask. Returns
        (inherited_selections, remaining_masks_for_dino).

        Skipped masks bypass DINO classification entirely — saving DINO
        compute proportional to the number of healthy tracker objects.
        """
        if not masks:
            return [], masks
        stale_frames = int(getattr(self.args, "tracker_health_stale_frames", 5))
        iou_thr = float(getattr(self.args, "tracker_health_iou_threshold", 0.5))
        max_occ = float(getattr(self.args, "tracker_health_max_occlusion", 0.4))

        # Collect healthy entries for this cam.
        healthy: list[tuple[str, np.ndarray]] = []  # (object_id, mask)
        for (cid, obj_id), info in self._recent_tracker_health.items():
            if cid != cam_id:
                continue
            if (self.frame_counter - int(info.get("frame_idx", 0))) > stale_frames:
                continue
            if float(info.get("occlusion_score", 0.0)) > max_occ:
                continue
            mask = info.get("mask")
            if mask is None or int(np.asarray(mask).sum()) <= 0:
                continue
            healthy.append((obj_id, np.asarray(mask, dtype=bool)))

        if not healthy:
            return [], masks

        inherited: list[CandidateSelection] = []
        remaining: list[SAMMaskCandidate] = []
        used_obj_ids: set[str] = set()
        for cand in masks:
            cm = np.asarray(cand.mask, dtype=bool)
            cm_sum = int(cm.sum())
            if cm_sum == 0:
                remaining.append(cand)
                continue
            best_iou = 0.0
            best_obj: Optional[str] = None
            for obj_id, hmask in healthy:
                if obj_id in used_obj_ids:
                    continue
                if hmask.shape != cm.shape:
                    continue
                inter = int(np.logical_and(cm, hmask).sum())
                if inter == 0:
                    continue
                union = cm_sum + int(hmask.sum()) - inter
                if union <= 0:
                    continue
                iou = float(inter) / float(union)
                if iou > best_iou:
                    best_iou = iou
                    best_obj = obj_id
            if best_obj is not None and best_iou >= iou_thr:
                inherited.append(CandidateSelection(
                    object_id=best_obj,
                    score=1.0,
                    scores_by_object={best_obj: 1.0},
                    candidate=cand,
                ))
                used_obj_ids.add(best_obj)
            else:
                remaining.append(cand)
        return inherited, remaining

    def _tiny_roi_for_cam(self, cam_id: str) -> Optional[tuple[int, int, int, int]]:
        """Look up the tiny-object ROI for a camera. Returns None if no ROI
        is configured (which means: skip the tiny pass on this cam)."""
        roi_str = None
        if cam_id == "zed2i_1":
            roi_str = getattr(self.args, "cam1_tiny_roi", "") or ""
        elif cam_id == "zed2i_2":
            roi_str = getattr(self.args, "cam2_tiny_roi", "") or ""
        if not roi_str.strip():
            return None
        return parse_xyxy_string(roi_str)

    def _generate_tiny_object_masks(self, rgb: np.ndarray, cam_id: str) -> list[SAMMaskCandidate]:
        if cam_id not in self.sam_tiny_by_cam:
            return []

        roi = self._tiny_roi_for_cam(cam_id)
        if roi is None:
            return []
        x0, y0, x1, y1 = roi
        h, w = rgb.shape[:2]
        x0 = max(0, min(x0, w - 1))
        x1 = max(x0 + 1, min(x1, w))
        y0 = max(0, min(y0, h - 1))
        y1 = max(y0 + 1, min(y1, h))

        crop = rgb[y0:y1, x0:x1].copy()
        if crop.size == 0:
            return []

        sam = self.sam_tiny_by_cam[cam_id]
        masks = sam.generate_auto(crop)
        self.get_logger().info(f"[{cam_id}] Tiny-object SAM raw masks: {len(masks)}")

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
        self.get_logger().info(f"[{cam_id}] Tiny-object SAM kept masks: {len(lifted)}")
        return lifted

   
    def _generate_and_filter_masks(self, rgb: np.ndarray, cam_id: str) -> list[SAMMaskCandidate]:
        if cam_id not in self.sam_by_cam:
            return []

        sam = self.sam_by_cam[cam_id]
        cam_params = self.cam_sam_params[cam_id]
        full_h, full_w = rgb.shape[:2]
        polygon_full = cam_params.roi_polygon

        # --- Crop to ROI ---
        t0 = time.time()
        rgb_crop, polygon_crop, crop_x0, crop_y0 = crop_rgb_to_polygon_bbox(rgb, polygon_full)
        crop_h, crop_w = rgb_crop.shape[:2]

        roi_mask_crop = np.zeros((crop_h, crop_w), dtype=np.uint8)
        cv2.fillPoly(roi_mask_crop, [polygon_crop], 255)
        rgb_crop_masked = rgb_crop.copy()
        rgb_crop_masked[roi_mask_crop == 0] = 0

        # D1: CLAHE on the ROI before SAM. Off by default.
        if bool(getattr(self.args, "clahe_enabled", False)):
            rgb_crop_masked = apply_clahe_rgb(
                rgb_crop_masked,
                clip_limit=float(self.args.clahe_clip_limit),
                grid_size=int(self.args.clahe_grid_size),
            )

        print(f"[TIMING]   ROI crop prep: {(time.time() - t0)*1000:.0f}ms")

        # --- Main proposal stage ---
        # Default: SAM automatic mask generator. With --mask-source gdino_sam,
        # boxes come from Grounding DINO and SAM is run as a box-prompted
        # segmenter (one mask per box).
        t1 = time.time()
        if (self.args.mask_source == "gdino_sam"
                and self.gdino_proposer is not None):
            proposals = self.gdino_proposer.propose(rgb_crop_masked)
            if proposals:
                boxes = np.array([p.bbox_xyxy for p in proposals], dtype=np.float32)
                masks_crop = sam.generate_from_boxes(rgb_crop_masked, boxes)
            else:
                masks_crop = []
            print(
                f"[TIMING]   GDINO+SAM (main): {(time.time() - t1)*1000:.0f}ms "
                f"-> {len(proposals)} boxes -> {len(masks_crop)} masks"
            )
        else:
            masks_crop = sam.generate_auto(rgb_crop_masked)
            print(f"[TIMING]   SAM generate_auto (main): {(time.time() - t1)*1000:.0f}ms -> {len(masks_crop)} raw masks")

        if not masks_crop:
            masks = []
        else:
            # --- Filtering ---
            t2 = time.time()
            masks_crop = reject_large_masks(
                masks_crop, crop_h, crop_w,
                cam_params.max_mask_area_ratio,
                cam_params.max_bbox_area_ratio,
            )
            masks_crop = reject_border_masks(
                masks_crop,
                cam_params.border_px,
                cam_params.max_border_fraction
            )
            masks_crop = reject_outside_roi_polygon(masks_crop, polygon_crop)
            masks = lift_crop_masks_to_full_image(
                masks_crop, full_h, full_w, crop_x0, crop_y0
            )
            print(f"[TIMING]   Mask filtering: {(time.time() - t2)*1000:.0f}ms -> {len(masks)} after filter")

        masks = sorted(masks, key=lambda m: m.area)

        # --- Tiny objects SAM (if enabled) ---
        # Bundle 11: tiny-pass now runs on every cam that has a configured
        # tiny ROI, not just zed2i_2.
        if self.args.tiny_objects_enabled and cam_id in self.sam_tiny_by_cam:
            t3 = time.time()
            tiny_masks = self._generate_tiny_object_masks(rgb, cam_id)
            print(f"[TIMING]   SAM tiny objects [{cam_id}]: {(time.time() - t3)*1000:.0f}ms -> {len(tiny_masks)} masks")
            masks.extend(tiny_masks)

        # --- Blue blob proposals (if enabled) ---
        if cam_id == "zed2i_2" and self.args.blue_blob_proposals_enabled:
            t4 = time.time()
            blue_blobs = find_blue_blob_masks(
                rgb,
                min_area=self.args.blue_blob_min_area,
                max_area=self.args.blue_blob_max_area,
            )
            blue_candidates = blob_masks_to_candidates(blue_blobs)
            print(f"[TIMING]   Blue blob detection: {(time.time() - t4)*1000:.0f}ms -> {len(blue_candidates)} blobs")
            masks.extend(blue_candidates)

        # --- Final filtering ---
        t5 = time.time()
        masks = reject_outside_roi_polygon(masks, cam_params.roi_polygon)
        masks = dedup_masks_by_bbox_iou(masks, iou_thresh=self.args.mask_dedup_iou)
        print(f"[TIMING]   Final filter + dedup: {(time.time() - t5)*1000:.0f}ms -> {len(masks)} final")

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

        dino_min_crop = int(getattr(self.args, "dino_min_crop_side", 0))
        for i, cand in enumerate(masks):
            crop_rgb, crop_mask = bbox_crop_with_local_mask(rgb, cand.mask, cand.bbox_xyxy)
            if crop_rgb.size == 0 or int(crop_mask.sum()) == 0:
                continue
            # A8: upscale tiny crops so DINOv2's resize-to-input_size doesn't
            # throw away detail on small objects (cooling_screw, cooling_f).
            crop_rgb, crop_mask = upscale_crop_if_small(crop_rgb, crop_mask, dino_min_crop)
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

            # B2: hardcoded cooling_f vs cooling_screw aspect rule. Disable
            # this when MUSE-style features are good enough to discriminate
            # on their own (--dino-embedding-mode concat is the typical case
            # where this rule becomes redundant or actively wrong on
            # generalised objects).
            if (
                self.args.use_aspect_cf_cs_rule
                and pair_is_cf_cs and pair_gap < 0.15
            ):
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

            # B4: softmax-entropy gate. Independent of raw cosine scale — flags
            # cases where multiple classes look near-equally similar even if the
            # raw score is high.
            if (
                self.args.dino_entropy_threshold > 0.0
                and object_id != "unknown"
                and not geometric_resolved
                and len(res.scores_by_object) >= 2
            ):
                tau = max(1e-3, float(self.args.dino_entropy_tau))
                logits = np.array(list(res.scores_by_object.values()), dtype=np.float64) / tau
                logits -= logits.max()
                probs = np.exp(logits)
                probs /= probs.sum() + 1e-12
                entropy = float(-(probs * np.log(probs + 1e-12)).sum())
                if entropy > self.args.dino_entropy_threshold:
                    self.get_logger().info(
                        f"DINO entropy reject {object_id}: H={entropy:.3f} > "
                        f"{self.args.dino_entropy_threshold:.3f}"
                    )
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
        if t_mag < 0.4 or t_mag > 1.5:
            return False, f"bad_distance mag={t_mag:.3f}"

        # D2: table-plane physical check. Objects sit on a flat table near
        # z_base ~= table_z. Reject anything outside [z_min, z_max]; defaults
        # match the historical [0.0, 0.5] m gate but are now configurable.
        try:
            T_base = self._to_base_pose(cam_id, T_camera)
            z_base = float(T_base[2, 3])
            z_lo = float(getattr(self.args, "table_plane_z_min", 0.0))
            z_hi = float(getattr(self.args, "table_plane_z_max", 0.5))
            if z_base < z_lo or z_base > z_hi:
                return False, f"bad_z_base z={z_base:.3f} (table window [{z_lo:.3f}, {z_hi:.3f}])"
        except Exception:
            pass

        return True, "ok"
    
    def _get_previous_object_base_pose(
        self,
        obj_id: str,
        per_object_entries: list[dict],
     ) -> Optional[np.ndarray]:
        mem = self._fused_track_memory.get(obj_id, {})
        T_mem = mem.get("T_base")
        if T_mem is not None:
            return np.asarray(T_mem, dtype=np.float32).reshape(4, 4).copy()

        for cr in per_object_entries:
            state = cr["state"]
            cam_id = cr["cam_id"]
            T_local = state.last_good_T if state.last_good_T is not None else state.T_object_camera
            if T_local is None:
                continue
            try:
                return self._to_base_pose(cam_id, T_local)
            except Exception:
                continue
        return None

    @staticmethod
    def _pose_delta_from_base(
        T_base_prev: Optional[np.ndarray],
        T_base_new: Optional[np.ndarray],
        ) -> tuple[float, float]:
        if T_base_prev is None or T_base_new is None:
            return 0.0, 0.0
        dt = float(np.linalg.norm(T_base_new[:3, 3] - T_base_prev[:3, 3]))
        drot = rotation_angle_deg(T_base_prev[:3, :3], T_base_new[:3, :3])
        return dt, drot

    @staticmethod
    def _stamp_to_seconds(stamp) -> Optional[float]:
        if stamp is None:
            return None
        try:
            return float(stamp.sec) + float(stamp.nanosec) * 1e-9
        except Exception:
            return None

    def _compute_fused_motion_dt(self, obj_id: str, stamp) -> tuple[float, Optional[float]]:
        nominal_dt = float(getattr(self.args, 'fused_track_nominal_dt_s', 0.15))
        dt_min = float(getattr(self.args, 'fused_track_min_dt_s', 0.10))
        dt_max = float(getattr(self.args, 'fused_track_max_dt_s', 0.30))
        t_now = self._stamp_to_seconds(stamp)
        t_prev = self._fused_track_memory.get(obj_id, {}).get('stamp_s')
        dt_raw: Optional[float] = None
        if t_now is not None and t_prev is not None:
            dt_raw = max(0.0, float(t_now) - float(t_prev))
        dt_eff = nominal_dt if dt_raw is None else dt_raw
        dt_eff = float(np.clip(dt_eff, dt_min, dt_max))
        return dt_eff, dt_raw

    def _compute_fused_motion_thresholds(self, dt_eff: float) -> tuple[float, float]:
        v_max = float(getattr(self.args, 'fused_track_max_translation_speed_mps', 0.13333333333333333))
        w_max = float(getattr(self.args, 'fused_track_max_rotation_speed_degps', 66.66666666666667))
        min_trans = float(getattr(self.args, 'fused_track_min_translation_jump_m', 0.008))
        min_rot = float(getattr(self.args, 'fused_track_min_rotation_jump_deg', 4.0))
        return max(min_trans, v_max * float(dt_eff)), max(min_rot, w_max * float(dt_eff))

    def _get_or_create_fused_translation_kalman(self, obj_id: str) -> PoseKalmanFilter:
        kf = self._fused_translation_kalman.get(obj_id)
        if kf is None:
            kf = PoseKalmanFilter()
            self._fused_translation_kalman[obj_id] = kf
        return kf

    def _get_fused_kalman_predicted_position(self, obj_id: str) -> Optional[np.ndarray]:
        kf = self._fused_translation_kalman.get(obj_id)
        if kf is None or not kf.is_initialized:
            return None
        return kf.get_predicted_position(1).astype(np.float32)


    def _evaluate_fused_camera_candidate(
        self,
        cr: dict,
        prev_base_pose: Optional[np.ndarray],
        is_warmup: bool = False,
        ) -> dict:
        """
        FIXED: During warmup (first N frames after init), override rt_invalid
        rejection and use relaxed gating thresholds. This lets the tracker
        stabilize after init without dropping into hold_previous immediately.
        """
        cam_id = cr["cam_id"]
        state = cr["state"]
        view = cr["view"]
        mask = np.asarray(cr["mask"]).astype(bool)
        rt_result = cr.get("rt_result")
    
        metrics = {
            "cam_id": cam_id,
            "state_idx": cr.get("state_idx", -1),
            "mask": mask,
            "bbox_xyxy": cr.get("bbox_xyxy"),
            "mask_area": int(mask.sum()),
            "last_good_mask_area": int(state.last_good_mask.sum()) if state.last_good_mask is not None else 0,
            "mask_area_ratio": None,
            "depth_coverage": 0.0,
            "cloud_points": 0,
            "cloud_centroid_dist_m": None,
            "per_cam_icp_fitness": float(getattr(rt_result, "icp_fitness", 0.0)) if rt_result is not None else 0.0,
            "per_cam_icp_rmse_m": float(getattr(rt_result, "icp_rmse", np.inf)) if rt_result is not None else float("inf"),
            "pcd": None,
            "accepted": False,
            "reason": "unknown",
        }
    
        # ── RT validity check (with warmup override) ──
        if rt_result is None:
            metrics["reason"] = "no_rt_result"
            if not (is_warmup and metrics["mask_area"] > 100):
                return metrics
        elif not bool(getattr(rt_result, "valid", False)):
            metrics["reason"] = "rt_invalid"
            if not is_warmup:
                return metrics
            if metrics["mask_area"] < 50:
                return metrics
            if getattr(self.args, "debug_verbose_logs", False):
                self.get_logger().info(
                    f"  [{cam_id}] Warmup override: rt_invalid but mask={metrics['mask_area']}"
                )
    
        # ── Mask size ──
        min_mask_area = max(20, int(getattr(self.args, "fused_gate_min_mask_area", 50)))
        if metrics["mask_area"] < min_mask_area:
            metrics["reason"] = f"mask_too_small({metrics['mask_area']})"
            return metrics
    
        # ── Mask ratio (relaxed during warmup) ──
        last_good_area = metrics["last_good_mask_area"]
        if last_good_area > 0:
            ratio = float(metrics["mask_area"]) / float(max(1, last_good_area))
            metrics["mask_area_ratio"] = ratio
            min_ratio = float(self.args.fused_gate_min_mask_area_ratio)
            max_ratio = float(self.args.fused_gate_max_mask_area_ratio)
            if is_warmup:
                min_ratio *= 0.3
                max_ratio *= 2.0
            if ratio < min_ratio or ratio > max_ratio:
                metrics["reason"] = f"mask_ratio_out_of_range({ratio:.2f})"
                return metrics
    
        # ── Depth coverage (relaxed during warmup) ──
        depth_coverage = mask_depth_coverage(
            view.depth, mask,
            zmin=self.args.min_valid_z_m,
            zmax=self.args.max_valid_z_m,
        )
        metrics["depth_coverage"] = depth_coverage
        min_cov = float(self.args.fused_gate_min_depth_coverage)
        if is_warmup:
            min_cov *= 0.5
        if depth_coverage < min_cov:
            metrics["reason"] = f"depth_coverage_low({depth_coverage:.2f})"
            return metrics
    
        # ── Build point cloud (WITH depth bias correction) ──
        T_bc = self._resolve_T_base_cam(cam_id)
        depth_bias = self._depth_bias_by_cam.get(cam_id, 0.0)
        pcd = lift_masked_depth_to_base(
            depth=view.depth,
            mask=mask,
            K=cr["K"],
            T_base_cam=T_bc,
            z_min=self.args.min_valid_z_m,
            z_max=self.args.max_valid_z_m,
            voxel_size=0.002,
            depth_bias_m=depth_bias,
            mask_morph_close_kernel=int(getattr(self.args, "icp_mask_close_kernel", 0)),
            mask_interior_erosion=int(getattr(self.args, "icp_mask_interior_erosion", 0)),
        )
        metrics["pcd"] = pcd
        metrics["cloud_points"] = int(len(pcd.points)) if pcd is not None else 0
        if pcd is None or metrics["cloud_points"] < int(self.args.fused_gate_min_cloud_points):
            metrics["reason"] = f"cloud_too_small({metrics['cloud_points']})"
            return metrics
    
        # ── Centroid distance (skip during warmup — init→track shift is normal) ──
        if prev_base_pose is not None and not is_warmup:
            pts = np.asarray(pcd.points)
            if pts.size > 0:
                centroid = pts.mean(axis=0)
                centroid_dist = float(np.linalg.norm(centroid - prev_base_pose[:3, 3]))
                metrics["cloud_centroid_dist_m"] = centroid_dist
                if centroid_dist > float(self.args.fused_gate_max_centroid_dist_m):
                    metrics["reason"] = f"centroid_far({centroid_dist:.3f}m)"
                    return metrics
    
        # ── Per-cam ICP gates (skipped when per-cam ICP is disabled, or
        # during warmup). With skip_per_cam_icp_tracking=True the per-cam
        # fitness/rmse fields are zero by construction, so these gates would
        # always reject; the fused-cloud ICP downstream is what gates pose
        # quality instead.
        per_cam_icp_disabled = bool(getattr(self.args, "skip_per_cam_icp_tracking", True))
        if not is_warmup and not per_cam_icp_disabled:
            if metrics["per_cam_icp_fitness"] < float(self.args.fused_gate_min_per_cam_icp_fitness):
                metrics["reason"] = f"per_cam_fitness_low({metrics['per_cam_icp_fitness']:.3f})"
                return metrics
            rmse = metrics["per_cam_icp_rmse_m"]
            if np.isfinite(rmse) and rmse > float(self.args.fused_gate_max_per_cam_icp_rmse_m):
                metrics["reason"] = f"per_cam_rmse_high({rmse*1000:.1f}mm)"
                return metrics
    
        metrics["accepted"] = True
        metrics["reason"] = "warmup_ok" if is_warmup else "ok"
        return metrics


    def _track_multicam_fused(self, views: list, stamp) -> None:
        """
        Fused multi-camera tracking — FIXED version.
        See docstring header for change list.
        """
        t_start = time.time()
    
        per_cam_results = {}
        tracker_key_to_state = {}
    
        # Collect work items
        camera_work = []
        for view in views:
            cam_id = view.cam_id
            states = self.track_states.get(cam_id, [])
            if not states:
                continue
            camera_work.append((view, states))
    
        def _run_one_camera(view, states):
            """Run Cutie+ICP for all objects on one camera. Thread-safe."""
            cam_id = view.cam_id
            rgb = view.rgb
            depth = view.depth
            K = np.asarray(view.K, dtype=np.float32).reshape(3, 3)
            results = {}
    
            for idx, state in enumerate(states):
                tracker_key = f"{cam_id}_{state.object_id}_{idx}"
    
                if tracker_key not in self.realtime_trackers:
                    try:
                        from src.perception.tracking.realtime_tracker import RealtimeTracker, RealtimeTrackerConfig
                        from src.perception.tracking.cutie_tracker import CutieConfig
                        from src.perception.tracking.icp_refiner import ICPConfig, ICPVariant
                        from src.perception.tracking.sam2_video_tracker import SAM2VideoConfig

                        video_kind = str(getattr(self.args, "video_tracker", "cutie")).lower()
                        sam2_cfg = SAM2VideoConfig(
                            max_internal_size=480,
                            memory_crop_padding_px=int(getattr(self.args, "track_memory_crop_padding", 0)),
                        )
                        cfg = RealtimeTrackerConfig(
                            cutie_cfg=CutieConfig(max_internal_size=480),
                            icp_cfg=ICPConfig(
                                variant=ICPVariant.POINT_TO_POINT,
                                max_correspondence_distance=0.05,
                                voxel_size=0.002,
                                mask_morph_close_kernel=int(getattr(self.args, "icp_mask_close_kernel", 0)),
                                mask_interior_erosion=int(getattr(self.args, "icp_mask_interior_erosion", 0)),
                            ),
                            min_icp_fitness=0.20,
                            max_translation_per_frame=0.08,
                            lost_frames_before_reinit=999,
                            verbose=False,
                            video_tracker_kind=video_kind,
                            sam2_cfg=sam2_cfg,
                        )
                        rt = RealtimeTracker(cfg)
                        init_mask = state.recovery_mask
                        if init_mask is None or init_mask.sum() < 100:
                            continue
                        init_mask = np.asarray(init_mask).astype(bool)
                        if init_mask.ndim != 2:
                            continue
    
                        rt.initialize(
                            rgb=rgb, depth=depth, mask=init_mask,
                            T_init=state.T_object_camera, K=K,
                            mesh_path=state.mesh_path,
                        )
                        self.realtime_trackers[tracker_key] = rt
                        state.last_good_mask = init_mask.copy()
                        state.last_good_T = state.T_object_camera.copy()
                    except Exception as e:
                        continue
    
                rt = self.realtime_trackers[tracker_key]
                try:
                    result = rt.track(
                        rgb, depth, K,
                        skip_icp=bool(getattr(self.args, "skip_per_cam_icp_tracking", True)),
                    )
                except Exception:
                    result = None
    
                if result is not None and result.mask_area > 20:
                    results[tracker_key] = {
                        "tracker_key": tracker_key,
                        "cam_id": cam_id,
                        "mask": result.mask,
                        "bbox_xyxy": result.bbox_xyxy,
                        "rt_result": result,
                        "state": state,
                        "state_idx": idx,
                        "view": view,
                        "K": K,
                    }
                else:
                    state.lost_count += 1
                    state.mode = "degraded"
    
            return results
    
        # Run cameras in parallel (or serially if only 1)

        if len(camera_work) >= 2:
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(_run_one_camera, v, s) for v, s in camera_work]
                for f in futures:
                    per_cam_results.update(f.result())
        else:
            for view, states in camera_work:
                per_cam_results.update(_run_one_camera(view, states))
    
        if not per_cam_results:
            self.get_logger().warn("Fused tracking: no camera produced a usable mask")
            return
    
        # Group by object
        by_object = {}
        for cr in per_cam_results.values():
            obj_id = cr["state"].object_id
            by_object.setdefault(obj_id, []).append(cr)
    
        object_decisions = {}
        t_fuse = time.time()
    
        for obj_id, entries in by_object.items():
            prev_base_pose = self._get_previous_object_base_pose(obj_id, entries)
    
            # ── FIX B: Warmup detection ──
            warmup_count = self._fused_warmup_count.get(obj_id, 0)
            warmup_frames = int(getattr(self.args, 'fused_track_warmup_frames', 5))
            is_warmup = warmup_count < warmup_frames
    
            gate_metrics = [
                self._evaluate_fused_camera_candidate(cr, prev_base_pose, is_warmup=is_warmup)
                for cr in entries
            ]
            survivors = [m for m in gate_metrics if m["accepted"]]
            survived_cam_ids = [m["cam_id"] for m in survivors]
    
            decision = {
                "object_id": obj_id,
                "mode": "hold_previous",
                "accepted": False,
                "reason": "no_survivors",
                "T_base": None,
                "fitness": 0.0,
                "rmse_m": float("inf"),
                "trans_jump_m": 0.0,
                "rot_jump_deg": 0.0,
                "survived_cam_ids": survived_cam_ids,
                "gate_metrics": gate_metrics,
                "used_tracker_key": None,
                "used_bbox_xyxy": None,
            }
    
            warmup_tag = f" [WARMUP {warmup_count}/{warmup_frames}]" if is_warmup else ""
            if getattr(self.args, "debug_verbose_logs", False):
                gate_log = " | ".join(
                    (
                        f"{m['cam_id']}:mask={m['mask_area']},"
                        f"ratio={'na' if m['mask_area_ratio'] is None else format(m['mask_area_ratio'], '.2f')},"
                        f"cov={m['depth_coverage']:.2f},"
                        f"pts={m['cloud_points']},"
                        f"reason={m['reason']}"
                    )
                    for m in gate_metrics
                )
                self.get_logger().info(
                    f"FUSED GATE {obj_id}{warmup_tag}: survivors={survived_cam_ids or 'none'} | {gate_log}"
                )
    
            if not survivors:
                object_decisions[obj_id] = decision
                self._fused_icp_metrics[obj_id] = {
                    "fitness": 0.0, "rmse_mm": 0.0,
                    "mode": "hold_previous", "accepted": False,
                    "reason": "no_survivors",
                }
                # Still increment warmup counter even on failure
                self._fused_warmup_count[obj_id] = warmup_count + 1
                continue
    
            # ── Single-camera fallback ──
            if len(survivors) == 1:
                m = survivors[0]
                cr = next(cr for cr in entries if cr["cam_id"] == m["cam_id"] and cr["state_idx"] == m["state_idx"])
                T_bc = self._resolve_T_base_cam(m["cam_id"])
    
                if cr.get("rt_result") is not None and hasattr(cr["rt_result"], "T_object_camera"):
                    T_base_candidate = (T_bc @ cr["rt_result"].T_object_camera.astype(np.float64)).astype(np.float32)
                else:
                    # During warmup with rt_invalid, use last good pose
                    T_base_candidate = (T_bc @ cr["state"].T_object_camera.astype(np.float64)).astype(np.float32)
    
                dt, drot = self._pose_delta_from_base(prev_base_pose, T_base_candidate)
                motion_dt_s, motion_dt_raw_s = self._compute_fused_motion_dt(obj_id, stamp)
                max_trans_jump_m, max_rot_jump_deg = self._compute_fused_motion_thresholds(motion_dt_s)
    
                kalman_pred_pos = self._get_fused_kalman_predicted_position(obj_id)
                kalman_residual_m = (
                    float(np.linalg.norm(T_base_candidate[:3, 3] - kalman_pred_pos))
                    if kalman_pred_pos is not None else None
                )
    
                if is_warmup:
                    # During warmup: only reject truly insane results
                    WARMUP_MAX_TRANS_M = 0.20
                    WARMUP_MAX_ROT_DEG = 90.0
                    motion_ok = (
                        (prev_base_pose is None)
                        or (dt <= WARMUP_MAX_TRANS_M and drot <= WARMUP_MAX_ROT_DEG)
                    )
                    kalman_soft_reject = False
                else:
                    motion_ok = dt <= max_trans_jump_m and drot <= max_rot_jump_deg
                    kalman_soft_reject = (
                        kalman_residual_m is not None
                        and kalman_residual_m > float(self.args.fused_track_kalman_soft_translation_residual_m)
                        and float(m["per_cam_icp_fitness"]) < float(self.args.fused_track_kalman_soft_max_icp_fitness)
                    )
    
                if motion_ok and not kalman_soft_reject:
                    T_base_publish = T_base_candidate
                    buf_size = int(getattr(self.args, 'median_pose_buffer_size', 3))
                    if buf_size > 0:
                        buf = self._median_pose_buffers.setdefault(obj_id, MedianPoseBuffer(buf_size))
                        buf.push(T_base_candidate)
                        if buf.is_ready():
                            T_base_publish = buf.get_median()
    
                    x_bias = float(getattr(self.args, 'x_bias_correction_m', 0.0))
                    if abs(x_bias) > 1e-6:
                        T_base_publish = apply_x_bias_correction(T_base_publish, x_bias)
    
                    decision.update({
                        "mode": "single_cam_fallback",
                        "accepted": True,
                        "reason": "single_cam_ok",
                        "T_base": T_base_publish,
                        "fitness": float(m["per_cam_icp_fitness"]),
                        "rmse_m": float(m["per_cam_icp_rmse_m"]),
                        "trans_jump_m": dt,
                        "rot_jump_deg": drot,
                    })
                    self._fused_track_memory[obj_id] = {
                        "T_base": T_base_publish.copy(),
                        "mode": "single_cam_fallback",
                        "stamp_s": self._stamp_to_seconds(stamp),
                    }
                else:
                    decision.update({
                        "mode": "hold_previous",
                        "accepted": False,
                        "reason": f"single_cam_motion_reject(dt={dt:.3f},rot={drot:.1f})",
                    })
    
                self._fused_icp_metrics[obj_id] = {
                    "fitness": decision.get("fitness", 0.0),
                    "rmse_mm": 0.0,
                    "mode": decision["mode"],
                    "accepted": decision["accepted"],
                    "reason": decision["reason"],
                }
                object_decisions[obj_id] = decision
                self._fused_warmup_count[obj_id] = warmup_count + 1
                continue
    
            # ── Multi-camera fusion path ──
            state0 = entries[0]["state"]
            # Tracking uses a lighter model PCD than init — cache key includes
            # num_points so init (5000) and tracking (e.g. 2000) coexist.
            track_num_points = int(getattr(self.args, "track_icp_num_points", 2000))
            model_pcd = mesh_to_pcd_cached(
                state0.mesh_path, float(self.args.mesh_scale),
                num_points=track_num_points,
            )
    
            survivor_pcds = [m["pcd"] for m in survivors if m["pcd"] is not None]
            use_weighted = bool(getattr(self.args, 'use_weighted_cloud_merge', False))
    
            if use_weighted and len(survivor_pcds) >= 2:
                cam_positions = [
                    self._resolve_T_base_cam(m["cam_id"])[:3, 3]
                    for m in survivors if m["pcd"] is not None
                ]
                # Weighted merge needs the voxel grid to do the weight-averaging,
                # so we keep the voxel size here.
                fused_cloud = merge_point_clouds_weighted(
                    survivor_pcds, cam_positions,
                    voxel_size=0.002,
                    distance_exponent=float(getattr(self.args, 'cloud_merge_distance_exponent', 2.0)),
                )
            else:
                # Each per-cam pcd was already voxel-downsampled at 2mm in
                # lift_masked_depth_to_base. A second 2mm voxel pass only dedups
                # boundary overlap and burns ~ms per tick; skip it.
                fused_cloud = merge_point_clouds(survivor_pcds, voxel_size=0.0)
    
            if fused_cloud is None or len(fused_cloud.points) < int(self.args.fused_gate_min_cloud_points):
                decision["reason"] = f"fused_cloud_too_small({0 if fused_cloud is None else len(fused_cloud.points)})"
                self._fused_icp_metrics[obj_id] = {
                    "fitness": 0.0, "rmse_mm": 0.0,
                    "mode": "hold_previous", "accepted": False,
                    "reason": decision["reason"],
                }
                object_decisions[obj_id] = decision
                self._fused_warmup_count[obj_id] = warmup_count + 1
                continue
    
            if prev_base_pose is not None:
                T_base_init = prev_base_pose.astype(np.float32)
            else:
                first_survivor = survivors[0]
                first_cr = next(cr for cr in entries if cr["cam_id"] == first_survivor["cam_id"])
                T_bc_0 = self._resolve_T_base_cam(first_survivor["cam_id"])
                T_base_init = (T_bc_0 @ first_cr["state"].T_object_camera.astype(np.float64)).astype(np.float32)
    
            # ── FIX G: Reduced ICP iterations ──
            icp_iters = int(getattr(self.args, 'fused_track_icp_max_iteration', 30))
            T_base_fused, fitness, rmse = run_icp_in_base_frame(
                scene_pcd=fused_cloud,
                model_pcd=model_pcd,
                T_base_object_init=T_base_init,
                max_correspondence_dist=float(self.args.fused_track_icp_max_correspondence_dist_m),
                max_iteration=icp_iters,
                variant=self.args.icp_variant,
            )
    
            dt, drot = self._pose_delta_from_base(prev_base_pose, T_base_fused)
            motion_dt_s, motion_dt_raw_s = self._compute_fused_motion_dt(obj_id, stamp)
            max_trans_jump_m, max_rot_jump_deg = self._compute_fused_motion_thresholds(motion_dt_s)
    
            kalman_pred_pos = self._get_fused_kalman_predicted_position(obj_id)
            kalman_residual_m = (
                float(np.linalg.norm(T_base_fused[:3, 3] - kalman_pred_pos))
                if kalman_pred_pos is not None else None
            )
    
            # ── FIX D: Chamfer distance gate (lazy) ──
            # Chamfer is the most expensive metric (NN search model↔fused
            # cloud). Compute it only when we actually need it: during
            # non-warmup frames where fitness/rmse/motion are all clean we
            # can skip the call entirely. The gate logic below uses
            # `chamfer is None` to mean "not computed → not used to reject".
            chamfer_skip_fitness_min = float(getattr(self.args, "chamfer_skip_fitness_min", 0.30))
            chamfer_skip_rmse_max = float(getattr(self.args, "chamfer_skip_rmse_max_m", 0.005))
            chamfer_skip_motion_max = float(getattr(self.args, "chamfer_skip_motion_max_m", 0.010))
            quality_clean = (
                fitness >= chamfer_skip_fitness_min
                and rmse <= chamfer_skip_rmse_max
                and (prev_base_pose is None or dt <= chamfer_skip_motion_max)
            )
            need_chamfer = (not is_warmup) and (not quality_clean)
            chamfer: Optional[float] = (
                chamfer_distance_one_way(model_pcd, fused_cloud, T_base_fused)
                if need_chamfer else None
            )

            accept = True
            reject_reasons = []

            if is_warmup:
                # ── WARMUP: skip ALL tight checks. Only reject truly insane results. ──
                # The tracker is still settling from init — rotation/translation jumps,
                # low fitness, high RMSE are all expected in the first few frames.
                # We only reject if the pose is completely nonsensical.
                WARMUP_MAX_TRANS_M = 0.20      # 200mm — way beyond normal motion
                WARMUP_MAX_ROT_DEG = 90.0      # quarter turn — clearly wrong
                WARMUP_MIN_FITNESS = 0.03      # basically "did ICP find anything at all"
    
                if fitness < WARMUP_MIN_FITNESS:
                    accept = False
                    reject_reasons.append(f"warmup_fitness_garbage({fitness:.3f})")
                if prev_base_pose is not None and dt > WARMUP_MAX_TRANS_M:
                    accept = False
                    reject_reasons.append(f"warmup_teleport({dt*1000:.0f}mm)")
                if prev_base_pose is not None and drot > WARMUP_MAX_ROT_DEG:
                    accept = False
                    reject_reasons.append(f"warmup_spin({drot:.0f}deg)")
            else:
                # ── NORMAL: full gating ──
                # Track which gates failed and WHY, so we can rescue jump-only rejects
                fitness_ok = fitness >= float(self.args.fused_track_min_fused_icp_fitness)
                rmse_ok = rmse <= float(self.args.fused_track_max_fused_icp_rmse_m)
                jump_t_ok = dt <= max_trans_jump_m
                jump_r_ok = drot <= max_rot_jump_deg

                if not fitness_ok:
                    accept = False
                    reject_reasons.append(f"fitness_low({fitness:.3f})")
                if not rmse_ok:
                    accept = False
                    reject_reasons.append(f"rmse_high({rmse*1000:.1f}mm)")
                if not jump_t_ok:
                    accept = False
                    reject_reasons.append(f"jump_t({dt*1000:.1f}mm>{max_trans_jump_m*1000:.1f}mm)")
                if not jump_r_ok:
                    accept = False
                    reject_reasons.append(f"jump_r({drot:.1f}>{max_rot_jump_deg:.1f})")

                # Chamfer gate. If we skipped the chamfer compute above
                # (quality already clean), treat the gate as passed.
                max_chamfer = float(getattr(self.args, 'fused_track_max_chamfer_m', 0.015))
                chamfer_ok = (chamfer is None) or (chamfer <= max_chamfer)
                if not chamfer_ok:
                    accept = False
                    reject_reasons.append(f"chamfer({chamfer*1000:.1f}mm>{max_chamfer*1000:.0f}mm)")

                # Axis-dominant jump check (from original)
                axis_jump = np.abs(T_base_fused[:3, 3] - prev_base_pose[:3, 3]) if prev_base_pose is not None else np.zeros(3)
                dominant_frac = float(axis_jump.max() / (np.linalg.norm(axis_jump) + 1e-9)) if np.linalg.norm(axis_jump) > 1e-9 else 0.0
                weak_icp = fitness < float(self.args.fused_track_weak_icp_fitness)
                axis_dominant_ok = True
                if (prev_base_pose is not None
                    and dominant_frac > float(self.args.fused_track_axis_dominant_fraction)
                    and dt > float(self.args.fused_track_axis_dominant_min_translation_m)
                    and weak_icp):
                    accept = False
                    axis_dominant_ok = False
                    reject_reasons.append("axis_dominant_jump")

                # Kalman soft reject
                kalman_soft_reject = (
                    kalman_residual_m is not None
                    and kalman_residual_m > float(self.args.fused_track_kalman_soft_translation_residual_m)
                    and fitness < float(self.args.fused_track_kalman_soft_max_icp_fitness)
                )
                if kalman_soft_reject:
                    accept = False
                    reject_reasons.append(f"kalman_soft(res={kalman_residual_m*1000:.1f}mm)")

                # ── CHAMFER RESCUE: if rejected ONLY because of jump limits,
                # but the new pose actually aligns well with the cloud, accept it.
                # This handles: real fast motion, or tracker snapping back to
                # correct pose after brief drift.
                if not accept:
                    # Check: was the rejection purely motion-based?
                    quality_ok = fitness_ok and rmse_ok and chamfer_ok and axis_dominant_ok and not kalman_soft_reject
                    jump_rejected = not jump_t_ok or not jump_r_ok

                    if quality_ok and jump_rejected:
                        # The pose FITS the cloud well (good fitness, rmse, chamfer)
                        # but moved too fast. This might be real motion or drift recovery.
                        # Use a tighter chamfer threshold as extra validation.
                        # Compute chamfer on demand if it was skipped above.
                        if chamfer is None:
                            chamfer = chamfer_distance_one_way(
                                model_pcd, fused_cloud, T_base_fused,
                            )
                        rescue_chamfer_thresh = max_chamfer * 0.75  # tighter than normal
                        if chamfer <= rescue_chamfer_thresh:
                            accept = True
                            reject_reasons.clear()
                            reject_reasons.append(
                                f"RESCUED:jump_dt={dt*1000:.1f}mm,drot={drot:.1f}deg,"
                                f"chamfer={chamfer*1000:.1f}mm<={rescue_chamfer_thresh*1000:.1f}mm"
                            )
                            self.get_logger().warn(
                                f"  ⚠ JUMP RESCUED {obj_id}: large motion "
                                f"dt={dt*1000:.1f}mm drot={drot:.1f}deg "
                                f"but chamfer={chamfer*1000:.1f}mm is good — accepting with flag"
                            )

            if accept:
                # ── Median pose buffer ──
                T_base_publish = T_base_fused
                buf_size = int(getattr(self.args, 'median_pose_buffer_size', 3))
                if buf_size > 0:
                    buf = self._median_pose_buffers.setdefault(obj_id, MedianPoseBuffer(buf_size))
                    buf.push(T_base_fused)
                    if buf.is_ready():
                        T_base_publish = buf.get_median()

                # ── X-bias correction ──
                x_bias = float(getattr(self.args, 'x_bias_correction_m', 0.0))
                if abs(x_bias) > 1e-6:
                    T_base_publish = apply_x_bias_correction(T_base_publish, x_bias)

                # Determine mode — flag rescued jumps distinctly
                was_rescued = any("RESCUED" in r for r in reject_reasons)
                if was_rescued:
                    mode_str = "fusion_rescued"
                elif len(survivors) >= 2:
                    mode_str = "fusion_2cam"
                else:
                    mode_str = "fusion"

                decision.update({
                    "mode": mode_str,
                    "accepted": True,
                    "reason": reject_reasons[0] if reject_reasons else "ok",
                    "T_base": T_base_publish,
                    "fitness": float(fitness),
                    "rmse_m": float(rmse),
                    "trans_jump_m": dt,
                    "rot_jump_deg": drot,
                    "chamfer_m": float(chamfer) if chamfer is not None else -1.0,
                })
                self._fused_track_memory[obj_id] = {
                    "T_base": T_base_publish.copy(),
                    "mode": decision["mode"],
                    "stamp_s": self._stamp_to_seconds(stamp),
                }
            else:
                decision.update({
                    "mode": "hold_previous",
                    "accepted": False,
                    "reason": ",".join(reject_reasons),
                    "fitness": float(fitness),
                    "rmse_m": float(rmse),
                    "trans_jump_m": dt,
                    "rot_jump_deg": drot,
                    "chamfer_m": float(chamfer) if chamfer is not None else -1.0,
                })
    
            self._fused_icp_metrics[obj_id] = {
                "fitness": float(fitness),
                "rmse_mm": float(rmse) * 1000.0,
                "mode": decision["mode"],
                "accepted": decision["accepted"],
                "reason": decision["reason"],
            }
    
            if getattr(self.args, "debug_verbose_logs", False):
                chamfer_str = (
                    f"{chamfer*1000:.1f}mm" if chamfer is not None else "skipped"
                )
                self.get_logger().info(
                    f"FUSED DECISION {obj_id}{warmup_tag}: mode={decision['mode']} accepted={decision['accepted']} "
                    f"cams={survived_cam_ids} fitness={fitness:.3f} rmse={rmse*1000:.1f}mm "
                    f"chamfer={chamfer_str} dt={dt*1000:.1f}mm drot={drot:.1f}deg "
                    f"reason={decision['reason']}"
                )
            object_decisions[obj_id] = decision
            self._fused_warmup_count[obj_id] = warmup_count + 1
    
        if getattr(self.args, "debug_verbose_logs", False):
            print(f"[TIMING] Fused ICP all objects: {(time.time()-t_fuse)*1000:.0f}ms")
    
        # ── Kalman update (same as original) ──
        for obj_id, decision in object_decisions.items():
            T_base_acc = decision.get("T_base")
            if not (bool(decision.get("accepted", False)) and T_base_acc is not None):
                continue
            kf = self._get_or_create_fused_translation_kalman(obj_id)
            pos = np.asarray(T_base_acc[:3, 3], dtype=np.float64).reshape(3)
            if not kf.is_initialized:
                kf.initialize(pos)
            else:
                kf.update(pos)
    
        # ── Back-project and publish (same structure as original) ──
        track_debug_by_cam = {}
    
        for obj_id, entries in by_object.items():
            decision = object_decisions.get(obj_id, {
                "mode": "hold_previous", "accepted": False,
                "reason": "missing", "T_base": None,
                "used_tracker_key": None, "used_bbox_xyxy": None,
            })
            T_base = decision.get("T_base")
            accepted = bool(decision.get("accepted", False)) and T_base is not None
    
            for cr in entries:
                cam_id = cr["cam_id"]
                state = cr["state"]
                idx = cr["state_idx"]
                rt_result = cr["rt_result"]
                current_mask = np.asarray(cr["mask"]).astype(bool)

                if accepted:
                    T_cam_base = self._resolve_T_cam_base(cam_id)
                    T_local = (T_cam_base @ T_base).astype(np.float32)
                    state.T_object_camera = T_local
                    state.last_good_T = T_local.copy()
                    state.last_good_mask = current_mask
                    state.lost_count = 0
                    state.degraded_count = 0
                    state.mode = "track"
                    if state.kalman is not None:
                        state.kalman.update(T_local[:3, 3])
                    # Bundle 11: remember healthy tracker masks so the next
                    # init (if it fires) can skip DINO for these objects.
                    occ = float(getattr(rt_result, "occlusion_score", 0.0)) if rt_result is not None else 0.0
                    self._recent_tracker_health[(cam_id, state.object_id)] = {
                        "frame_idx": self.frame_counter,
                        "mask": current_mask.copy(),
                        "occlusion_score": occ,
                        "T_object_camera": T_local.copy(),
                    }
                else:
                    if state.last_good_T is not None:
                        state.T_object_camera = state.last_good_T.copy()
                    state.degraded_count += 1
                    state.mode = "degraded"

                state.recovery_mask = current_mask
                state.last_mask_area = int(current_mask.sum())

                # Per-cam debug publish (off by default — fused canonical is the
                # production output). Single publish per object per cam.
                if getattr(self.args, "debug_per_cam_pose_publish", False):
                    if accepted:
                        pose_msg = T_to_pose_stamped(T_base, frame_id="base", stamp=stamp)
                        self._get_or_create_pose_base_pub(
                            cam_id, state.object_id, idx
                        ).publish(pose_msg)
                    else:
                        self._publish_pose_base(
                            cam_id, state.object_id, idx,
                            state.T_object_camera, stamp,
                        )
    
                if cam_id not in track_debug_by_cam:
                    fused_metrics = self._fused_icp_metrics.get(obj_id, {})
                    track_debug_by_cam[cam_id] = {
                        "mask": current_mask,
                        "bbox_xyxy": cr["bbox_xyxy"],
                        "object_id": state.object_id,
                        "icp_fitness": fused_metrics.get("fitness", 0.0),
                        "icp_rmse_mm": fused_metrics.get("rmse_mm", 0.0),
                    }
    
        # ── Debug frame publishing ──
        for view in views:
            cam_id = view.cam_id
            states = self.track_states.get(cam_id, [])
            if not states:
                continue
            pose_items = self._states_to_pose_item_msgs(cam_id, states, include_masks=False)
            track_debug = track_debug_by_cam.get(cam_id)
            frame = self._build_debug_frame(
                cam_id=cam_id, stamp=stamp,
                update_sam=False, update_dino=False,
                sam_candidates=[], dino_candidates=[],
                pose_items=pose_items, track_debug=track_debug,
            )
            if cam_id in self.pub_debug_frame:
                self.pub_debug_frame[cam_id].publish(frame)
    
        # ── Publish fused canonical poses ──
        # Cache the publisher reference on the decision dict so subsequent
        # frames skip the dict lookup. The publisher dict is keyed per object.
        if not hasattr(self, "_pub_fused_pose"):
            self._pub_fused_pose = {}
        if not hasattr(self, "_fused_pose_pub_by_obj"):
            self._fused_pose_pub_by_obj: dict[str, Any] = {}

        for obj_id, decision in object_decisions.items():
            if not decision.get("accepted") or decision.get("T_base") is None:
                continue
            pub = self._fused_pose_pub_by_obj.get(obj_id)
            if pub is None:
                fused_key = f"fused/{obj_id}_0"
                pub = self._pub_fused_pose.get(fused_key)
                if pub is None:
                    pub = self.create_publisher(
                        PoseStamped, f"/perception/fp/pose_base/{fused_key}",
                        FAST_QOS,
                    )
                    self._pub_fused_pose[fused_key] = pub
                self._fused_pose_pub_by_obj[obj_id] = pub
            pub.publish(
                T_to_pose_stamped(decision["T_base"], frame_id="base", stamp=stamp)
            )

        if getattr(self.args, "debug_verbose_logs", False):
            t_total = (time.time() - t_start) * 1000
            accepted_count = sum(1 for d in object_decisions.values() if d.get("accepted"))
            self.get_logger().info(
                f"FUSED TRACK total: {t_total:.0f}ms | {accepted_count} objects updated"
            )

   
    def _check_chamfer_drift(self) -> None:
        """
        Compare mean Chamfer across cameras over recent inits.
        If one camera is consistently 2x+ worse than the other,
        warn about likely extrinsic drift.
        Requires at least 3 samples per camera to trigger.
        """
        cam_ids = list(self._init_chamfer_history.keys())
        if len(cam_ids) < 2:
            return

        means = {}
        for cid in cam_ids:
            history = self._init_chamfer_history[cid]
            if len(history) < 3:
                return  # not enough data yet
            means[cid] = float(np.mean(list(history)))

        best_cam = min(means, key=means.get)
        best_mean = means[best_cam]

        for cid, mean_ch in means.items():
            if cid == best_cam:
                continue
            ratio = mean_ch / (best_mean + 1e-9)
            if ratio > 2.0:
                self.get_logger().warn(
                    f"  ⚠ CHAMFER DRIFT: {cid} mean={mean_ch*1000:.1f}mm vs "
                    f"{best_cam} mean={best_mean*1000:.1f}mm (ratio={ratio:.1f}x) — "
                    f"extrinsic for {cid} may have drifted"
                )
            else:
                self.get_logger().info(
                    f"  Chamfer balance: {cid}={mean_ch*1000:.1f}mm vs "
                    f"{best_cam}={best_mean*1000:.1f}mm (ratio={ratio:.1f}x) — OK"
                )
    


    def _process_multicam_init(self, views: list, stamp) -> None:
        """
        Multi-camera fusion init with single-FP + symmetry-grid refinement.

        Flow for each fused detection:
        1. Run FP estimate_pose ONCE on each contributing camera
        2. Convert each FP result to base frame
        3. Lift all masked depths to base frame (1mm voxels), merge into fused cloud
        4. ICP-refine each base-frame pose against the fused cloud
        5. If chamfer > 8mm, run 24-seed symmetry grid (absolute orientations)
        6. Hard-reject cameras with chamfer > 2x best camera's chamfer
        7. Weighted average by ICP fitness → single canonical base-frame pose
        8. Polishing ICP of canonical pose against fused cloud
        9. Back-project to all contributing cameras
        """
        t_start = time.time()
        print(f"\n[TIMING] ========== MULTICAM INIT START ==========")

        # ── Phase 1: SAM + DINO per camera (unchanged) ──
        selections_by_cam: dict[str, list[CandidateSelection]] = {}
        views_by_cam: dict[str, Any] = {}

        for view in views:
            cam_id = view.cam_id
            if view.rgb is None or view.depth is None:
                continue
            if view.rgb.shape[:2] != view.depth.shape[:2]:
                continue
            views_by_cam[cam_id] = view

            t_sam = time.time()
            masks = self._generate_and_filter_masks(view.rgb, cam_id)
            print(f"[TIMING] SAM {cam_id}: {(time.time()-t_sam)*1000:.0f}ms -> {len(masks)} masks")

            if not masks:
                selections_by_cam[cam_id] = []
                continue

            # Bundle 11: skip DINO for masks that match a recently-healthy
            # tracker mask via IoU. Inherit the tracker's object_id directly.
            inherited: list[CandidateSelection] = []
            remaining_masks = masks
            if bool(getattr(self.args, "skip_dino_when_tracker_healthy", False)):
                inherited, remaining_masks = self._inherit_from_tracker_health(
                    cam_id=cam_id, masks=masks,
                )
                if inherited:
                    print(f"[TIMING] DINO-skip {cam_id}: inherited {len(inherited)} masks from tracker (skipped DINO)")

            t_dino = time.time()
            ranked = self._classify_masks_batched(view.rgb, remaining_masks) if remaining_masks else []
            selected = self._select_top_candidates(inherited + ranked, view.depth)
            print(f"[TIMING] DINO+select {cam_id}: {(time.time()-t_dino)*1000:.0f}ms -> {len(selected)} selected")
            selections_by_cam[cam_id] = selected

        if sum(len(v) for v in selections_by_cam.values()) == 0:
            for view in views:
                self.track_states[view.cam_id] = []
            print(f"[TIMING] MULTICAM INIT: no detections")
            return

        # ── Phase 2: Fusion matching (unchanged) ──
        t_fusion = time.time()
        fusion_cfg = FusionConfig(
            debug_enabled=False,
        )
        fused_detections = run_multicam_fusion(
            selections_by_cam=selections_by_cam,
            views_by_cam=views_by_cam,
            T_base_cam_map=self.T_base_cam_map,
            cfg=fusion_cfg,
        )
        print(f"[TIMING] Fusion matching: {(time.time()-t_fusion)*1000:.0f}ms -> {len(fused_detections)} fused objects")

        # ── Phase 3: Single-FP + ICP + symmetry grid + weighted average ──
        t_fp_all = time.time()
        new_states_by_cam: dict[str, list[ObjectTrackState]] = {
            v.cam_id: [] for v in views
        }

        if not hasattr(self, "_pub_fused_pose"):
            self._pub_fused_pose: dict[str, Any] = {}

        for i, fused in enumerate(fused_detections):
            try:
                mesh_path = self._resolve_mesh_path(fused.object_id)
            except FileNotFoundError:
                continue

            # ─── Step 3a: Lift all masked depths to base frame, merge ───
            # [CHANGE] 1mm voxels for init (was 2mm) — denser cloud gives
            # ICP more surface detail to lock onto, especially for small parts.
            INIT_VOXEL_SIZE = 0.001

            t_lift = time.time()
            per_cam_clouds: list[o3d.geometry.PointCloud] = []

            for det in fused.detections:
                cam_id = det.cam_id
                if cam_id not in views_by_cam:
                    continue
                T_bc = self._resolve_T_base_cam(cam_id)
                K_cam = np.asarray(views_by_cam[cam_id].K, dtype=np.float32).reshape(3, 3)

                # C5: optionally fill ZED depth holes inside the mask before
                # lifting. Off by default (kernel=0). Costs one medianBlur per
                # camera per init.
                depth_for_lift = views_by_cam[cam_id].depth
                hole_k = int(getattr(self.args, "depth_fill_holes_kernel", 0))
                if hole_k > 0:
                    depth_for_lift = fill_depth_holes_in_mask(
                        depth_for_lift, det.mask, kernel=hole_k,
                    )

                pcd = lift_masked_depth_to_base(
                    depth=depth_for_lift,
                    mask=det.mask,
                    K=K_cam,
                    T_base_cam=T_bc,
                    voxel_size=INIT_VOXEL_SIZE,
                    mask_morph_close_kernel=int(getattr(self.args, "icp_mask_close_kernel", 0)),
                    mask_interior_erosion=int(getattr(self.args, "icp_mask_interior_erosion", 0)),
                )
                if pcd is not None:
                    per_cam_clouds.append(pcd)
                    print(f"  [{cam_id}] Lifted {len(pcd.points)} pts for {fused.object_id}")

            # ─── Cloud overlap diagnostic ───
            # Two NN-distance passes over the full per-cam clouds — pure
            # diagnostic for extrinsic drift. Gate behind verbose-logs since
            # extrinsics rarely shift between init runs.
            if len(per_cam_clouds) >= 2 and getattr(self.args, "debug_verbose_logs", False):
                d01 = per_cam_clouds[0].compute_point_cloud_distance(per_cam_clouds[1])
                d10 = per_cam_clouds[1].compute_point_cloud_distance(per_cam_clouds[0])
                mean_overlap_dist = (float(np.mean(d01)) + float(np.mean(d10))) / 2.0
                cam_ids_str = [det.cam_id for det in fused.detections if det.cam_id in views_by_cam]
                print(
                    f"  CLOUD OVERLAP {fused.object_id}: "
                    f"{cam_ids_str[0]}<->{cam_ids_str[1]} "
                    f"mean_dist={mean_overlap_dist*1000:.1f}mm"
                )
                if mean_overlap_dist > 0.008:
                    self.get_logger().warn(
                        f"  ⚠ CLOUD MISALIGNMENT {fused.object_id}: "
                        f"{mean_overlap_dist*1000:.1f}mm > 8mm — "
                        f"extrinsic calibration may need re-running"
                    )

            # Unweighted merge at init
            fused_cloud = merge_point_clouds(per_cam_clouds, voxel_size=INIT_VOXEL_SIZE)

            if fused_cloud is None or len(fused_cloud.points) < 50:
                print(f"  Fused cloud too small for {fused.object_id}, skipping")
                continue
            print(f"  Fused cloud: {len(fused_cloud.points)} pts ({(time.time()-t_lift)*1000:.0f}ms)")

            # Get mesh point cloud for ICP
            model_pcd = mesh_to_pcd_cached(mesh_path, float(self.args.mesh_scale), num_points=5000)

            # ─── Step 3b: Run FP ONCE on each contributing camera ───
            # C7: grid-skip threshold is configurable; 0.008m kept the historical
            # default. Below this, FP+ICP is already in the good regime so the
            # rotation-grid sweep is skipped — saves ~100ms when init is clean.
            SYMMETRY_GRID_CHAMFER_M = float(getattr(self.args, "icp_grid_skip_chamfer_m", 0.008))
            CHAMFER_REJECT_M = 0.012 if len(fused.detections) == 1 else 0.015

            candidate_poses: list[np.ndarray] = []
            candidate_weights: list[float] = []
            candidate_chamfers: list[float] = []   # [CHANGE] store for hard-reject gate
            candidate_cam_ids: list[str] = []
            candidate_det_indices: list[int] = []

            is_single_cam = len(fused.detections) == 1
            if is_single_cam:
                print(f"  ⚠ SINGLE-CAM init for {fused.object_id}")

            for det_idx, det in enumerate(fused.detections):
                cam_id = det.cam_id
                if cam_id not in views_by_cam:
                    continue

                view = views_by_cam[cam_id]
                K_cam = np.asarray(view.K, dtype=np.float32).reshape(3, 3)
                tracker = self.fp_tracker_by_cam[cam_id]

                # ─── Single FP call ───
                try:
                    t_fp = time.time()
                    result = tracker.estimate_pose(
                        object_id=fused.object_id,
                        mesh_path=mesh_path,
                        rgb=view.rgb,
                        depth=view.depth,
                        K=K_cam,
                        mask=pad_mask_for_fp(det.mask, pad_px=5),
                    )
                    fp_ms = (time.time() - t_fp) * 1000
                except Exception as e:
                    self.get_logger().warn(
                        f"  FP failed [{cam_id}] {fused.object_id}: {e}"
                    )
                    continue

                T_cam = np.asarray(result.T_object_camera, dtype=np.float32).reshape(4, 4)
                ok_pose, reason = self._pose_reason(T_cam, cam_id)
                if not ok_pose:
                    self.get_logger().info(
                        f"  FP pose reject {cam_id} {fused.object_id}: {reason}"
                    )
                    continue

                # Convert to base frame
                T_bc = self._resolve_T_base_cam(cam_id)
                T_base = (T_bc @ T_cam.astype(np.float64)).astype(np.float32)

                # ICP-refine against fused cloud
                t_icp = time.time()
                T_refined, fitness, rmse = run_icp_in_base_frame(
                    scene_pcd=fused_cloud,
                    model_pcd=model_pcd,
                    T_base_object_init=T_base,
                    max_correspondence_dist=0.05,
                    max_iteration=30,
                    variant=self.args.icp_variant,
                )
                icp_ms = (time.time() - t_icp) * 1000

                if fitness < 0.10:
                    self.get_logger().info(
                        f"  ICP fitness too low {cam_id} {fused.object_id}: {fitness:.3f}"
                    )
                    continue

                chamfer = chamfer_distance_one_way(model_pcd, fused_cloud, T_refined)

                print(
                    f"  FP+ICP [{cam_id}] {fused.object_id}: "
                    f"fp={fp_ms:.0f}ms icp={icp_ms:.0f}ms "
                    f"fitness={fitness:.3f} chamfer={chamfer*1000:.2f}mm"
                )

                best_T = T_refined
                best_chamfer = chamfer
                best_fitness = fitness
                best_rmse = rmse

                # ─── Single symmetry grid call at threshold ───
                grid_ran = False
                if best_chamfer > SYMMETRY_GRID_CHAMFER_M:
                    grid_ran = True
                    t_grid = time.time()
                    grid_T, grid_chamfer = self._icp_rotation_grid(
                        best_T[:3, 3], model_pcd, fused_cloud,
                        per_cam_clouds=per_cam_clouds,
                    )
                    grid_ms = (time.time() - t_grid) * 1000

                    if grid_T is not None and grid_chamfer < best_chamfer:
                        print(
                            f"    Symmetry grid improved [{cam_id}]: "
                            f"{best_chamfer*1000:.2f}mm → {grid_chamfer*1000:.2f}mm "
                            f"({grid_ms:.0f}ms)"
                        )
                        best_T = grid_T
                        best_chamfer = grid_chamfer
                    else:
                        print(f"    Symmetry grid no improvement [{cam_id}] ({grid_ms:.0f}ms)")

                # ─── C8: optional FP refiner pass after the grid ───
                # Lock in the grid-chosen rotation with FP's learned refiner.
                # Off by default. Only fires when the grid actually ran (the
                # whole point is to clean up after a rotation flip), and only
                # if the grid produced a usable pose.
                if (grid_ran
                        and bool(getattr(self.args, "fp_refine_after_grid", False))
                        and best_T is not None):
                    try:
                        t_fpref = time.time()
                        T_bc = self._resolve_T_base_cam(cam_id)
                        T_cb = np.linalg.inv(T_bc)
                        T_cam_obj = (T_cb @ best_T.astype(np.float64)).astype(np.float32)
                        n_iter = int(getattr(self.args, "fp_refine_iterations", 1))
                        ref_res = tracker.refine_pose(
                            object_id=fused.object_id,
                            mesh_path=mesh_path,
                            rgb=view.rgb,
                            depth=view.depth,
                            K=K_cam,
                            T_object_camera_init=T_cam_obj,
                            iterations=n_iter,
                        )
                        T_cam_ref = np.asarray(
                            ref_res.T_object_camera, dtype=np.float32
                        ).reshape(4, 4)
                        T_base_ref = (T_bc @ T_cam_ref.astype(np.float64)).astype(np.float32)
                        ref_chamfer = chamfer_distance_one_way(
                            model_pcd, fused_cloud, T_base_ref
                        )
                        fpref_ms = (time.time() - t_fpref) * 1000
                        if ref_chamfer < best_chamfer:
                            print(
                                f"    FP-refine improved [{cam_id}]: "
                                f"{best_chamfer*1000:.2f}mm → {ref_chamfer*1000:.2f}mm "
                                f"({fpref_ms:.0f}ms)"
                            )
                            best_T = T_base_ref
                            best_chamfer = ref_chamfer
                        else:
                            print(
                                f"    FP-refine no improvement [{cam_id}] "
                                f"({fpref_ms:.0f}ms, chamfer {ref_chamfer*1000:.2f}mm)"
                            )
                    except Exception as e:
                        self.get_logger().warn(
                            f"  FP-refine after grid failed [{cam_id}]: {e}"
                        )

                # ─── Optional CSV + PNG logging ───
                if getattr(self.args, 'log_init_poses', False):
                    accepted_attempt = (best_chamfer <= CHAMFER_REJECT_M)
                    try:
                        from scipy.spatial.transform import Rotation as SciRot
                        roll, pitch, yaw = SciRot.from_matrix(
                            best_T[:3, :3].astype(np.float64)
                        ).as_euler('xyz', degrees=True)
                        log_path = Path('init_pose_log.csv')
                        write_header = not log_path.exists()
                        with open(log_path, 'a') as f:
                            if write_header:
                                f.write('timestamp,obj_id,cam_id,'
                                        'tx,ty,tz,roll,pitch,yaw,'
                                        'chamfer_mm,accepted\n')
                            f.write(
                                f'{time.time():.3f},{fused.object_id},{cam_id},'
                                f'{best_T[0,3]:.4f},{best_T[1,3]:.4f},'
                                f'{best_T[2,3]:.4f},'
                                f'{roll:.2f},{pitch:.2f},{yaw:.2f},'
                                f'{best_chamfer*1000:.2f},{accepted_attempt}\n'
                            )
                    except Exception as e:
                        print(f"  [WARN] init pose CSV log failed: {e}")

                    save_init_pose_render(
                        best_T, model_pcd, fused.object_id,
                        f'init_renders/{fused.object_id}_{cam_id}.png',
                        accepted=accepted_attempt,
                    )

                print(
                    f"  FINAL [{cam_id}] {fused.object_id}: "
                    f"fitness={best_fitness:.3f} rmse={best_rmse*1000:.1f}mm "
                    f"chamfer={best_chamfer*1000:.2f}mm"
                )

                # ─── [CHANGE] Per-camera Chamfer history tracking ───
                self._init_chamfer_history[cam_id].append(best_chamfer)

                # ─── Chamfer reject handling (unchanged logic) ───
                init_quality = 'good'
                if best_chamfer > CHAMFER_REJECT_M:
                    init_quality = 'uncertain'
                    self.get_logger().warn(
                        f"  ⚠ CHAMFER HIGH {cam_id} {fused.object_id}: "
                        f"{best_chamfer*1000:.1f}mm > {CHAMFER_REJECT_M*1000:.0f}mm — "
                        f"accepting with init_quality='uncertain'"
                    )

                    fail_key = fused.object_id
                    self._consecutive_chamfer_fails[fail_key] = \
                        self._consecutive_chamfer_fails.get(fail_key, 0) + 1

                    n_fails = self._consecutive_chamfer_fails[fail_key]
                    last_reinit = self._last_reinit_time.get(fail_key, 0.0)
                    if n_fails >= 3 and (time.time() - last_reinit) > 10.0:
                        self.get_logger().warn(
                            f"  ✗ {n_fails} consecutive Chamfer failures for "
                            f"{fused.object_id} — full SAM+DINO reinit on next cycle"
                        )
                        self._last_reinit_time[fail_key] = time.time()
                        self._consecutive_chamfer_fails[fail_key] = 0
                        for cid in self.track_states:
                            self.track_states[cid] = [
                                s for s in self.track_states[cid]
                                if s.object_id != fused.object_id
                            ]
                else:
                    self._consecutive_chamfer_fails[fused.object_id] = 0

                candidate_poses.append(best_T)
                weight = best_fitness / (best_chamfer + 1e-6)
                candidate_weights.append(weight)
                candidate_chamfers.append(best_chamfer)
                candidate_cam_ids.append(cam_id)
                candidate_det_indices.append(det_idx)

            # ─── End of per-camera loop ───

            if not candidate_poses:
                self.get_logger().info(f"  No valid FP results for {fused.object_id}")
                continue

            # ─── [CHANGE] Hard-reject cameras with chamfer > 2x best ───
            if len(candidate_poses) >= 2:
                best_chamfer_val = min(candidate_chamfers)
                rejection_threshold = best_chamfer_val * 2.0

                filtered_indices = [
                    j for j, ch in enumerate(candidate_chamfers)
                    if ch <= rejection_threshold
                ]

                if len(filtered_indices) < len(candidate_poses):
                    rejected_cams = [
                        f"{candidate_cam_ids[j]}({candidate_chamfers[j]*1000:.1f}mm)"
                        for j in range(len(candidate_poses))
                        if j not in filtered_indices
                    ]
                    self.get_logger().info(
                        f"  CHAMFER HARD-REJECT {fused.object_id}: "
                        f"best={best_chamfer_val*1000:.1f}mm, "
                        f"threshold={rejection_threshold*1000:.1f}mm, "
                        f"rejected=[{', '.join(rejected_cams)}]"
                    )

                    # Only apply if at least one candidate survives
                    if filtered_indices:
                        candidate_poses = [candidate_poses[j] for j in filtered_indices]
                        candidate_weights = [candidate_weights[j] for j in filtered_indices]
                        candidate_chamfers = [candidate_chamfers[j] for j in filtered_indices]
                        candidate_cam_ids = [candidate_cam_ids[j] for j in filtered_indices]
                        candidate_det_indices = [candidate_det_indices[j] for j in filtered_indices]

            # ─── Step 3d: Weighted average → canonical base-frame pose ───
            T_base_canonical = weighted_average_poses(candidate_poses, candidate_weights)

            # ─── [CHANGE] Polishing ICP: snap the averaged pose back onto the cloud ───
            t_polish = time.time()
            T_base_canonical, polish_fitness, polish_rmse = run_icp_in_base_frame(
                scene_pcd=fused_cloud,
                model_pcd=model_pcd,
                T_base_object_init=T_base_canonical,
                max_correspondence_dist=0.03,   # tighter than initial — already close
                max_iteration=20,
                variant=self.args.icp_variant,
            )
            polish_chamfer = chamfer_distance_one_way(model_pcd, fused_cloud, T_base_canonical)
            polish_ms = (time.time() - t_polish) * 1000
            print(
                f"  POLISH ICP {fused.object_id}: "
                f"fitness={polish_fitness:.3f} rmse={polish_rmse*1000:.1f}mm "
                f"chamfer={polish_chamfer*1000:.2f}mm ({polish_ms:.0f}ms)"
            )

            T_publish = T_base_canonical.copy()

            # Log the fusion result
            t_canon = T_publish[:3, 3]
            weights_str = ", ".join(
                f"{cid}:{w:.2f}" for cid, w in zip(candidate_cam_ids, candidate_weights)
            )
            print(
                f"  CANONICAL {fused.object_id}: "
                f"t=[{t_canon[0]:.4f}, {t_canon[1]:.4f}, {t_canon[2]:.4f}] "
                f"weights=[{weights_str}]"
                f"{' [SINGLE-CAM]' if is_single_cam else ''}"
            )

            # C9: distance/mask-area-based confidence warning. Cheap, log-only.
            if self.args.distance_confidence_warn:
                dist_m = float(np.linalg.norm(T_publish[:3, 3]))
                min_mask_area = min(
                    int(d.mask.sum()) for d in fused.detections
                ) if fused.detections else 0
                if (
                    dist_m > self.args.distance_confidence_max_m
                    or min_mask_area < self.args.distance_confidence_min_mask_area
                ):
                    print(
                        f"  LOW-CONFIDENCE {fused.object_id}: "
                        f"dist={dist_m:.2f}m min_mask_area={min_mask_area}"
                    )

            # ─── Step 3e: Back-project to all contributing cameras ───
            for det_idx, det in enumerate(fused.detections):
                cam_id = det.cam_id
                if cam_id not in views_by_cam:
                    continue

                T_bc = self._resolve_T_base_cam(cam_id)
                T_cam_base = self._resolve_T_cam_base(cam_id)
                T_local = T_cam_base @ T_base_canonical

                ok_local, reason_local = self._pose_reason(T_local, cam_id)
                if not ok_local:
                    self.get_logger().info(
                        f"  Back-proj reject {fused.object_id} for {cam_id}: {reason_local}"
                    )
                    continue

                state = ObjectTrackState(
                    object_id=fused.object_id,
                    mesh_path=mesh_path,
                    mode="track",
                    T_object_camera=T_local.copy(),
                    dino_score=float(det.dino_score),
                    lost_count=0,
                    last_mask_area=int(det.mask.sum()),
                    track_pose_convention="raw",
                    recovery_mask=det.mask.copy(),
                )
                state.id_history.append(fused.object_id)
                state.last_good_mask = det.mask.copy()
                state.last_good_T = T_local.copy()
                new_states_by_cam[cam_id].append(state)

                self._log_base_pose(
                    f"INIT", cam_id, fused.object_id, i, T_local,
                    extra=f"dino={det.dino_score:.3f} cams={len(candidate_poses)}",
                )

            # Publish canonical fused pose
            fused_key = f"fused/{fused.object_id}_{i}"
            if fused_key not in self._pub_fused_pose:
                self._pub_fused_pose[fused_key] = self.create_publisher(
                    PoseStamped, f"/perception/fp/pose_base/{fused_key}", FAST_QOS,
                )
            self._pub_fused_pose[fused_key].publish(
                T_to_pose_stamped(T_publish, frame_id="base", stamp=stamp)
            )

        print(f"[TIMING] FP all objects: {(time.time()-t_fp_all)*1000:.0f}ms")

        # ─── [CHANGE] Per-camera Chamfer drift detection ───
        self._check_chamfer_drift()

        # ── Publish debug frames ──
        for cam_id, states in new_states_by_cam.items():
            if not states:
                continue
            pose_items = self._states_to_pose_item_msgs(cam_id, states, include_masks=False)
            frame = self._build_debug_frame(
                cam_id=cam_id, stamp=stamp,
                update_sam=False, update_dino=False,
                sam_candidates=[], dino_candidates=[],
                pose_items=pose_items,
            )
            if cam_id in self.pub_debug_frame:
                self.pub_debug_frame[cam_id].publish(frame)

        # ── Store states with NMS ──
        for cam_id, states in new_states_by_cam.items():
            states = nms_by_position(states, position_threshold=0.03)
            if self.args.run_mode == "track":
                self.track_states[cam_id] = states
            else:
                self.track_states[cam_id] = []

        self._reset_tracking_state_for_reinit(fused_detections)

        torch.cuda.empty_cache()
        t_total = (time.time() - t_start) * 1000
        total_inited = sum(len(s) for s in new_states_by_cam.values())
        print(f"[TIMING] ========== MULTICAM INIT TOTAL: {t_total:.0f}ms | {total_inited} objects ==========\n")

    def _reset_tracking_state_for_reinit(self, fused_detections):
        """
        Call this in _process_multicam_init after creating new states.
        Resets warmup counters, median buffers, and Kalman filters for
        all re-initialized objects.
        """
        for fused in fused_detections:
            obj_id = fused.object_id
            self._fused_warmup_count[obj_id] = 0
            if obj_id in self._median_pose_buffers:
                self._median_pose_buffers[obj_id].reset()
            if obj_id in self._fused_translation_kalman:
                self._fused_translation_kalman[obj_id].reset()
            if obj_id in self._fused_track_memory:
                del self._fused_track_memory[obj_id]

    def _tick(self) -> None:
        """
        FIXED: If ANY camera has tracked objects, run fused tracking.
        Only fall to init if NO camera has states at all.
        """
        if self.busy:
            return
        views = self.grabber.get_latest_views()
        if views is None:
            if getattr(self.args, "debug_verbose_logs", False):
                print("[TICK] No views yet...")
            return
    
        self.busy = True
        try:
            self.frame_counter += 1
            stamp = self.get_clock().now().to_msg()
    
            # Check which cameras have active tracking
            any_tracking = any(
                bool(self.track_states.get(v.cam_id))
                and all(
                    s.mode in ("track", "track/rt", "degraded", "fast_recovery")
                    for s in self.track_states[v.cam_id]
                )
                for v in views
            )
    
            no_states = all(
                not bool(self.track_states.get(v.cam_id))
                for v in views
            )
    
            if any_tracking and not no_states:
                # ─── FUSED TRACKING ───
                # Cameras without states simply don't contribute to fusion
                self._track_multicam_fused(views, stamp)
            else:
                # ─── MULTICAM INIT ───
                torch.cuda.empty_cache()
                try:
                    self._process_multicam_init(views, stamp)
                except Exception as e:
                    self.get_logger().warn(f"Multicam init failed: {e}")
                    import traceback
                    traceback.print_exc()
                    for view in views:
                        self.track_states[view.cam_id] = []
        finally:
            self.busy = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--device", default="cuda")
    p.add_argument("--mask-source",
                   choices=["sam", "projected", "gdino_sam"],
                   default="sam")
    # B7/B8/A9: Grounding DINO + SAM (MUSE-style) proposal stage. Only used
    # when --mask-source gdino_sam.
    p.add_argument("--gdino-model-id", default="IDEA-Research/grounding-dino-base")
    p.add_argument("--gdino-box-threshold", type=float, default=0.30)
    p.add_argument("--gdino-text-threshold", type=float, default=0.25)
    p.add_argument("--gdino-max-boxes", type=int, default=30)
    # Comma-separated text prompts. Default covers the current object set;
    # override via --gdino-text-prompts "a,b,c" to extend.
    p.add_argument(
        "--gdino-text-prompts",
        default="cooling base,cooling f,cooling screw,blue cube,green cube,red cube,screwdriver,pb base,pb pipe,pb screw,pb top",
    )
    p.add_argument("--target-object", default=None)
    p.add_argument("--run-mode", choices=["track", "init_only"], default="track")

    p.add_argument("--reference-dir", default="Data/ZED_screens")
    p.add_argument("--cad-dir", default="Data/CAD_Models_centered")
    p.add_argument("--output-root", default="outputs/foundationpose")

    # B6: synthetic-render reference bank. Generated by
    # tools/generate_dino_reference_renders.py. Selects which sources
    # contribute to the DINO reference bank.
    p.add_argument("--reference-source", choices=["real", "renders", "both"],
                   default="real")
    p.add_argument("--reference-renders-dir", default="Data/reference_renders")

    # DINO backbone. dinov2_vitb14 = current default (faster). Pass
    # dinov2_vitl14 for higher fidelity per DINOv2 paper Fig. 2/5.
    p.add_argument("--dino-model-name", default="dinov2_vitb14")
    p.add_argument("--dino-min-score", type=float, default=0.55)
    p.add_argument("--dino-min-margin", type=float, default=0.0)
    # B4: softmax-entropy gate over per-object scores. Higher = less confident.
    # Disabled at 0.0 (default); typical effective range ~0.6-0.9 with tau=0.05.
    p.add_argument("--dino-entropy-threshold", type=float, default=0.0)
    p.add_argument("--dino-entropy-tau", type=float, default=0.05)
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

    p.add_argument("--cam1-sam-min-mask-area", type=int, default=20)
    p.add_argument("--cam1-sam-min-bbox-side-px", type=int, default=3)
    p.add_argument("--cam1-sam-max-mask-area-ratio", type=float, default=0.06)
    p.add_argument("--cam1-sam-max-bbox-area-ratio", type=float, default=0.06)
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
    # Bundle 11: per-cam tiny ROIs. Empty string disables the tiny pass on
    # that cam. cam1 default is empty so the runner stays backwards
    # compatible — pass an explicit ROI to enable cam1's tiny pass.
    p.add_argument("--cam1-tiny-roi", type=str, default="")
    p.add_argument("--cam2-tiny-roi", type=str, default="700,500,1350,1080")
    p.add_argument("--tiny-sam-max-image-side", type=int, default=1920)
    p.add_argument("--tiny-sam-min-mask-area", type=int, default=8)
    p.add_argument("--tiny-sam-min-bbox-side-px", type=int, default=2)
    p.add_argument("--tiny-max-mask-area-ratio", type=float, default=0.01)
    p.add_argument("--tiny-max-bbox-area-ratio", type=float, default=0.02)

    # A3/A4/A7: lighter tiny-pass SAM config. Empty checkpoint => fall back to
    # main checkpoint. Pass e.g.
    #   --tiny-sam-checkpoint external/sam2/checkpoints/sam2.1_hiera_small.pt
    #   --tiny-sam-model-cfg configs/sam2.1/sam2.1_hiera_s.yaml
    # to use Hiera Small once the checkpoint is downloaded.
    p.add_argument("--tiny-sam-checkpoint", type=str, default="")
    p.add_argument("--tiny-sam-model-cfg", type=str, default="")
    p.add_argument("--tiny-sam-points-per-side", type=int, default=32)
    p.add_argument("--tiny-sam-pred-iou-thresh", type=float, default=0.55)
    p.add_argument("--tiny-sam-stability-score-thresh", type=float, default=0.55)
    p.add_argument("--tiny-sam-max-aspect-ratio", type=float, default=4.5)

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

    # Fused tracking robustness
    p.add_argument("--fused-gate-min-mask-area", type=int, default=50)
    p.add_argument("--fused-gate-min-mask-area-ratio", type=float, default=0.40)
    p.add_argument("--fused-gate-max-mask-area-ratio", type=float, default=2.50)
    p.add_argument("--fused-gate-min-depth-coverage", type=float, default=0.30)
    p.add_argument("--fused-gate-min-cloud-points", type=int, default=40)
    p.add_argument("--fused-gate-max-centroid-dist-m", type=float, default=0.08)
    p.add_argument("--fused-gate-min-per-cam-icp-fitness", type=float, default=0.18)
    p.add_argument("--fused-gate-max-per-cam-icp-rmse-m", type=float, default=0.015)

    p.add_argument("--fused-track-min-fused-icp-fitness", type=float, default=0.12)
    p.add_argument("--fused-track-max-fused-icp-rmse-m", type=float, default=0.012)
    p.add_argument("--fused-track-max-translation-jump-m", type=float, default=0.05)
    p.add_argument("--fused-track-max-rotation-jump-deg", type=float, default=180.0)
    p.add_argument("--fused-track-nominal-dt-s", type=float, default=0.15)
    p.add_argument("--fused-track-min-dt-s", type=float, default=0.10)
    p.add_argument("--fused-track-max-dt-s", type=float, default=0.30)
    p.add_argument("--fused-track-max-translation-speed-mps", type=float, default=0.13333333333333333)
    p.add_argument("--fused-track-max-rotation-speed-degps", type=float, default=66.66666666666667)
    p.add_argument("--fused-track-min-translation-jump-m", type=float, default=0.008)
    p.add_argument("--fused-track-min-rotation-jump-deg", type=float, default=4.0)
    p.add_argument("--fused-track-kalman-soft-translation-residual-m", type=float, default=0.025)
    p.add_argument("--fused-track-kalman-soft-max-icp-fitness", type=float, default=0.22)
    p.add_argument("--fused-track-weak-icp-fitness", type=float, default=0.18)
    p.add_argument("--fused-track-axis-dominant-fraction", type=float, default=0.80)
    p.add_argument("--fused-track-axis-dominant-min-translation-m", type=float, default=0.012)
    p.add_argument("--fused-track-icp-max-correspondence-dist-m", type=float, default=0.05)
    # Tracking uses a tight init from the previous frame, so 30 iters is overkill;
    # 15 lands within ~0.5mm of the converged pose for our object scale.
    p.add_argument("--fused-track-icp-max-iteration", type=int, default=15)

    # Positive = sensor reads too short, negative = too long
    # Calibrate: measure known distance, compare to depth reading
    p.add_argument("--cam1-depth-bias-m", type=float, default=0.0)
    p.add_argument("--cam2-depth-bias-m", type=float, default=0.0)
 
    # Distance-weighted cloud merging (mitigates depth bias at distance)
    p.add_argument("--use-weighted-cloud-merge", action="store_true")
    p.add_argument("--cloud-merge-distance-exponent", type=float, default=2.0)
 
    # X-axis bias correction (if x consistently underestimated, set > 0)
    p.add_argument("--x-bias-correction-m", type=float, default=0.0)
 
    # Warmup frames after init (relaxed gating to let tracker stabilize)
    p.add_argument("--fused-track-warmup-frames", type=int, default=5)
 
    # Chamfer distance gate for outlier rejection during tracking
    p.add_argument("--fused-track-max-chamfer-m", type=float, default=0.015)
 
    # Median pose buffer (temporal outlier filter, 0 to disable)
    p.add_argument("--median-pose-buffer-size", type=int, default=3)

    # [POST-RUN3 Item 4] Init rotation logging + 3D PNG render (off by default)
    p.add_argument("--log-init-poses", action="store_true",
                   help="Log CSV of init pose RPY + render 3D PNGs per attempt")

    # ── Latency / debug flags ──
    # When true, also publish one pose-per-cam topic for debug viz; otherwise
    # only the fused canonical pose is published.
    p.add_argument("--debug-per-cam-pose-publish", action="store_true")
    # When true, emit per-frame INFO logs and [TIMING] prints. Off by default.
    p.add_argument("--debug-verbose-logs", action="store_true")

    # Tracking ICP can be skipped entirely (per-cam ICP is redundant because the
    # fused-cloud ICP refines the pose anyway).
    p.add_argument("--skip-per-cam-icp-tracking", action="store_true", default=True)

    # Tracking-time model PCD point count (init keeps the original 5000 via a
    # separate cache entry). For iter count see --fused-track-icp-max-iteration.
    p.add_argument("--track-icp-num-points", type=int, default=2000)

    # Conditional chamfer thresholds. Skip chamfer when both fitness/rmse are
    # already clean and motion is small.
    p.add_argument("--chamfer-skip-fitness-min", type=float, default=0.30)
    p.add_argument("--chamfer-skip-rmse-max-m", type=float, default=0.005)
    p.add_argument("--chamfer-skip-motion-max-m", type=float, default=0.010)

    # C1: ICP variant for run_icp_in_base_frame. point_to_plane penalises
    # normal-direction error and converges better on flat faces / cylinders;
    # point_to_point is the historical default and our current behaviour.
    p.add_argument("--icp-variant", choices=["point_to_point", "point_to_plane"],
                   default="point_to_point")

    # C9: warn-tag poses that are likely lower confidence (far away / tiny mask).
    # Default off so behaviour matches current logs.
    p.add_argument("--distance-confidence-warn", action="store_true")
    p.add_argument("--distance-confidence-max-m", type=float, default=1.5)
    p.add_argument("--distance-confidence-min-mask-area", type=int, default=2000)

    # C2/C3/C4/C6: rotation-grid search overhaul. All defaults preserve the
    # current 24-element octahedral grid behaviour. Enable individually:
    #   --icp-grid-fibonacci          : sample N uniform SO(3) seeds instead of 24 cube seeds
    #   --icp-grid-prescreen          : skip ICP for seeds whose raw-Chamfer > tau (cheap)
    #   --icp-grid-second-pass        : refine top-K winners with small angular jitter
    #   --icp-grid-cross-cam-chamfer  : score by mean Chamfer across per-cam clouds
    #   --icp-grid-tie-by-inliers     : break Chamfer ties by ICP fitness (≈ inlier count)
    p.add_argument("--icp-grid-fibonacci", action="store_true")
    p.add_argument("--icp-grid-n-rot", type=int, default=60)
    p.add_argument("--icp-grid-prescreen", action="store_true")
    p.add_argument("--icp-grid-prescreen-tau", type=float, default=0.04)
    p.add_argument("--icp-grid-second-pass", action="store_true")
    p.add_argument("--icp-grid-second-pass-k", type=int, default=3)
    p.add_argument("--icp-grid-second-pass-n", type=int, default=8)
    p.add_argument("--icp-grid-second-pass-jitter-deg", type=float, default=15.0)
    p.add_argument("--icp-grid-cross-cam-chamfer", action="store_true")
    p.add_argument("--icp-grid-tie-by-inliers", action="store_true")
    p.add_argument("--icp-grid-tie-chamfer-eps-m", type=float, default=0.0005)

    # C5: fill ZED depth holes inside the mask before lifting to a cloud.
    # 0 = off (current behaviour). 3 or 5 = cv2.medianBlur kernel.
    p.add_argument("--depth-fill-holes-kernel", type=int, default=0)

    # C7: threshold above which the rotation grid runs. Below this, FP+ICP
    # is already in the good regime and the grid sweep is skipped.
    p.add_argument("--icp-grid-skip-chamfer-m", type=float, default=0.008)

    # C8: run FP's neural refiner after the rotation grid to lock in the
    # geometry-best seed. Off by default — only useful when the grid runs.
    p.add_argument("--fp-refine-after-grid", action="store_true")
    p.add_argument("--fp-refine-iterations", type=int, default=1)

    # B2 / Bundle 5: MUSE-style DINO upgrades. Defaults preserve original
    # CLS-token + cosine behaviour.
    #   --dino-embedding-mode concat enables MUSE-style CLS+GeM-patch
    #   --dino-similarity tanimoto switches off cosine
    #   --dino-joint-score-alpha > 0 blends raw scores with relative ranking
    p.add_argument("--dino-embedding-mode", choices=["cls", "patch_gem", "concat"],
                   default="cls")
    p.add_argument("--dino-gem-p", type=float, default=3.0)
    p.add_argument("--dino-similarity", choices=["cosine", "tanimoto"], default="cosine")
    p.add_argument("--dino-joint-score-alpha", type=float, default=0.0)
    # B2: hardcoded cooling_f vs cooling_screw aspect rule. On for back-compat;
    # disable once MUSE features make the rule redundant (and dangerous if
    # extended to new objects).
    p.add_argument("--use-aspect-cf-cs-rule", action="store_true", default=True)
    p.add_argument("--no-aspect-cf-cs-rule", dest="use_aspect_cf_cs_rule",
                   action="store_false")

    # A8: bicubic-upscale DINO crops whose short side is below this many
    # pixels (after the bbox+mask crop). 0 = off (current behaviour).
    # 224 is a safe default since DINOv2 ultimately resizes to 518.
    p.add_argument("--dino-min-crop-side", type=int, default=0)

    # D1: CLAHE preprocessing on the L channel of LAB before SAM. Off by
    # default. Helps on low-contrast matte parts under flat lighting.
    p.add_argument("--clahe-enabled", action="store_true")
    p.add_argument("--clahe-clip-limit", type=float, default=2.0)
    p.add_argument("--clahe-grid-size", type=int, default=8)

    # D2: table-plane physical check. _pose_reason rejects poses with
    # z_base outside this window. Defaults preserve current [0, 0.5] gate.
    p.add_argument("--table-plane-z-min", type=float, default=0.0)
    p.add_argument("--table-plane-z-max", type=float, default=0.5)

    # Bundle 11: skip DINO classification for masks that match a recently
    # healthy tracker mask (by IoU). Off by default — flip on when the
    # video tracker is reliable enough to authoritatively name objects.
    p.add_argument("--skip-dino-when-tracker-healthy", action="store_true")
    p.add_argument("--tracker-health-iou-threshold", type=float, default=0.5)
    p.add_argument("--tracker-health-stale-frames", type=int, default=5)
    p.add_argument("--tracker-health-max-occlusion", type=float, default=0.4)

    # Bundle 11: mask preprocessing before depth lift / ICP. Both 0 by
    # default to preserve current behaviour. Sane starting values would be
    # close-kernel=5 (fills 5px specular holes) and erosion=3 (drops
    # boundary pixels with the noisiest ZED depth).
    p.add_argument("--icp-mask-close-kernel", type=int, default=0)
    p.add_argument("--icp-mask-interior-erosion", type=int, default=0)

    # Bundle 10: A5/A6/B5 — alternative video tracker.
    # A5: choose backend; default keeps current Cutie behaviour. SAM2 falls
    # back to Cutie automatically if it fails to load or fails mid-stream.
    p.add_argument("--video-tracker", choices=["cutie", "sam2"], default="cutie")
    # B5: pad (in internal-resolution px) of the memory crop used to re-prompt
    # SAM2 each frame. 0 disables the crop and feeds the full image, matching
    # current behaviour. Only consulted when --video-tracker sam2.
    p.add_argument("--track-memory-crop-padding", type=int, default=0)

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