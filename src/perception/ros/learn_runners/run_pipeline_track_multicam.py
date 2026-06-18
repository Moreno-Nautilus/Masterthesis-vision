"""Main multi-camera pose pipeline node: detect → fuse → FoundationPose+ICP init, then Cutie+ICP tracking."""
from __future__ import annotations

import argparse
import array
import csv
import os
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
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Header
from fp_debug_msgs.msg import DebugCandidate, DebugFrame, DebugMaskCrop, DebugPoseItem
from concurrent.futures import ThreadPoolExecutor

from src.calibration.io_extrinsics import load_extrinsics_yaml
from src.perception.learned.DINO.dino_identifier import (
    DINOIdentifier,
    DINOIdentifierConfig,
    DINOResult,
    MUSE_EMBEDDING_MODE,
    MUSE_JOINT_SCORE_ALPHA,
    MUSE_OBJECTNESS_PRIOR_GAMMA,
    MUSE_RELATIVE_SCORE_MODE,
    MUSE_SIMILARITY,
    MUSE_STREAM_ALPHA,
    MUSE_TAU,
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
from src.perception.multicam_fusion import (
    FusionConfig,
    run_multicam_fusion,
)
from typing import Any, Iterable, Optional
import open3d as o3d
from src.perception.fused_multicam_helpers import (
    lift_masked_depth_to_base,
    merge_point_clouds,
    merge_point_clouds_weighted,
    mesh_to_pcd_cached,
    run_icp_in_base_frame,
    evaluate_icp_in_base_frame,
    chamfer_distance_one_way,
    weighted_average_poses,
    MedianPoseBuffer,
    fill_depth_holes_in_mask,
)


_DEBUG_LOGGING = False


FAST_CUTIE_PROFILE_OVERRIDES = {
    "track_require_pose_origin_in_mask": True,
    "track_pose_mask_margin_px": 8,
    "track_icp_num_points": 800,
    "fused_track_icp_max_iteration": 6,
    "fused_track_icp_relative_fitness": 1e-3,
    "fused_track_icp_relative_rmse": 1e-3,
    "median_pose_buffer_size": 1,
    "fused_track_max_chamfer_m": 0.018,
    "fused_track_max_translation_speed_mps": 0.25,
    "fused_track_max_rotation_speed_degps": 180.0,
    "fused_track_min_translation_jump_m": 0.02,
    "fused_track_min_rotation_jump_deg": 10.0,
    "memory_crop_enable": False,
    "skip_per_cam_icp_tracking": True,
    "debug_frame_publish": False,
    "debug_per_cam_pose_publish": False,
    "debug_verbose_logs": False,
    "debug_logging": False,
    "disable_fused_kalman": True,
    "disable_axis_jump_gate": True,
    "chamfer_every_n_frames": 3,
}


def _apply_fast_cutie_profile(namespace: argparse.Namespace) -> None:
    # Mutate argparse's namespace exactly as if each fast-tracking flag was passed.
    for key, value in FAST_CUTIE_PROFILE_OVERRIDES.items():
        setattr(namespace, key, value)
    setattr(namespace, "fast_cutie", True)
    setattr(namespace, "tracking_profile", "fast_cutie")


def _apply_default_tracking_profile(
    parser: argparse.ArgumentParser,
    namespace: argparse.Namespace,
) -> None:
    # Reset profile-owned fields back to parser defaults when --tracking-profile default wins.
    for key in FAST_CUTIE_PROFILE_OVERRIDES:
        setattr(namespace, key, parser.get_default(key))
    setattr(namespace, "fast_cutie", False)
    setattr(namespace, "tracking_profile", "default")


class _FastCutieAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=0, **kwargs):
        super().__init__(option_strings, dest, nargs=0, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        _apply_fast_cutie_profile(namespace)


class _TrackingProfileAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if values == "fast_cutie":
            _apply_fast_cutie_profile(namespace)
        else:
            _apply_default_tracking_profile(parser, namespace)


def _dprint(*args, **kwargs) -> None:
    # Cheap global gate for timing/debug prints outside ROS logging.
    if _DEBUG_LOGGING:
        # flush so this interleaves deterministically with _UnifiedLogger output
        # (both are on stdout now) when the run is redirected to a single file.
        kwargs.setdefault("flush", True)
        print(*args, **kwargs)


class _UnifiedLogger:
    """Single-stream replacement for the ROS node logger.

    The ROS get_logger() writes to stderr while the pipeline's own diagnostics
    use print/_dprint on stdout; two independently-buffered streams interleave
    non-deterministically when redirected to one log file. Routing every
    get_logger() call through here puts ALL output on stdout with flush, so a
    redirected log is ordered and consistently prefixed.

    Level gating: warn/error/fatal always print (no longer silently dropped when
    debug is off); info/debug print only when verbose.
    """

    def __init__(self, verbose: bool) -> None:
        self._verbose = bool(verbose)

    @staticmethod
    def _emit(level: str, msg) -> None:
        t = time.time()
        ts = time.strftime("%H:%M:%S", time.localtime(t)) + f".{int((t % 1) * 1000):03d}"
        print(f"[{ts}] [{level}] {msg}", flush=True)

    def info(self, msg="", *a, **kw):
        if self._verbose:
            self._emit("INFO", msg)

    def debug(self, msg="", *a, **kw):
        if self._verbose:
            self._emit("DEBUG", msg)

    def warn(self, msg="", *a, **kw):
        self._emit("WARN", msg)

    def warning(self, msg="", *a, **kw):
        self._emit("WARN", msg)

    def error(self, msg="", *a, **kw):
        self._emit("ERROR", msg)

    def fatal(self, msg="", *a, **kw):
        self._emit("FATAL", msg)


FAST_QOS = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
)


# All known cameras and their ROS topics; --num-cameras selects the leading subset.
ALL_CAMERAS = [
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
    CameraTopics(
        cam_id="zed2i_3",
        depth_topic="/zed2i_3/zed_node/depth/depth_registered",
        info_topic="/zed2i_3/zed_node/depth/depth_registered/camera_info",
        rgb_topic="/zed2i_3/zed_node/rgb/color/rect/image",
        rgb_info_topic="/zed2i_3/zed_node/rgb/color/rect/image/camera_info",
    ),
]


def select_cameras(num_cameras: int) -> list[CameraTopics]:
    """Active camera set for this run (the first `num_cameras` of ALL_CAMERAS)."""
    if num_cameras < 1 or num_cameras > len(ALL_CAMERAS):
        raise ValueError(
            f"num_cameras must be in 1..{len(ALL_CAMERAS)}, got {num_cameras}"
        )
    return ALL_CAMERAS[:num_cameras]

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
        
        # Process noise 
        self.Q = np.eye(6, dtype=np.float64)
        self.Q[:3, :3] *= process_noise ** 2  # Position 
        self.Q[3:, 3:] *= (process_noise * 2) ** 2  # Velocity
        
        # Measurement noise (position only)
        self.R = np.eye(3, dtype=np.float64) * measurement_noise ** 2
        
        # State transition matrix (constant velocity model)
        self.F = np.eye(6, dtype=np.float64)
        self.F[0, 3] = 1.0  # x += vx
        self.F[1, 4] = 1.0  # y += vy
        self.F[2, 5] = 1.0  # z += vz
        
        # Measurement matrix (only position)
        self.H = np.zeros((3, 6), dtype=np.float64)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        
        self._initialized = False
        self._frame_count = 0
       
    
    def initialize(self, position: np.ndarray) -> None:
        self.state[:3] = position
        self.state[3:] = 0.0  # Zero initial velocity
        self.P = np.eye(6, dtype=np.float64) * 0.1
        self._initialized = True
        self._frame_count = 1
    
    def predict(self) -> np.ndarray:
        """
        Advance state one step using the constant-velocity model and
        propagate covariance.
        """
        if not self._initialized:
            return np.zeros(3)

        # x ← Fx, P ← FPFᵀ + Q.
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q

        return self.state[:3].copy()
    
    def update(self, position: np.ndarray) -> None:
        """
        Update filter with new measured position.
        """
        if not self._initialized:
            self.initialize(position)
            return

        # Standard Kalman correction step (residual → gain → state/covariance update).
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

    def reset(self) -> None:
        """Reset filter state."""
        self.state = np.zeros(6, dtype=np.float64)
        self.P = np.eye(6, dtype=np.float64) * 0.1
        self._initialized = False
        self._frame_count = 0
    
    @property
    def is_initialized(self) -> bool:
        return self._initialized

# Per-camera, per-object tracking state carried between ticks (pose, masks, mode, Kalman).
@dataclass
class ObjectTrackState:
    object_id: str
    mesh_path: str
    track_id: str = ""
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
class MemoryCrop:
    """A clean appearance snapshot of a tracked object, captured when ICP
    fitness was good. Used at reinit to re-rank SAM/DINO candidates so the
    pipeline picks the same instance it was tracking instead of a same-class
    look-alike."""
    object_id: str
    cam_id: str
    frame_idx: int
    fitness: float
    embedding: np.ndarray            # L2-normalised DINO embedding
    track_id: str = ""
    rgb_crop: Optional[np.ndarray] = None   # uint8 HxWx3;
    mask_crop: Optional[np.ndarray] = None  # bool HxW

# One accepted detection for an object on a camera (chosen mask + its DINO scores).
@dataclass
class CandidateSelection:
    object_id: str
    score: float
    scores_by_object: dict[str, float]
    candidate: SAMMaskCandidate
    base_scores_by_object: dict[str, float] = field(default_factory=dict)
    score_source: str = "dino_base"
    track_id_hint: str = ""
    dino_class_score: float = 0.0
    dino_margin: float = 0.0


# Per-camera SAM mask-filtering parameters + ROI polygon.
@dataclass
class CameraSAMParams:
    min_mask_area: int
    min_bbox_side_px: int
    max_mask_area_ratio: float
    max_bbox_area_ratio: float
    border_px: int
    max_border_fraction: float
    roi_polygon: np.ndarray


# Geodesic angle (deg) between two rotation matrices.
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
    # Save a 3D scatter PNG of the model cloud at the estimated pose (init debug viz).
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

        color = '#3366CC' if accepted else "#AF2323"
        ax.scatter(pts_base[::3, 0], pts_base[::3, 1], pts_base[::3, 2],
                   s=1, c=color, alpha=0.4)

        # Draw pose axes
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
        _dprint(f"  [WARN] save_init_pose_render failed: {e}")


# Rotation matrix → quaternion [x,y,z,w] (branch on the largest diagonal term).
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


# Build a ROS Pose from a translation + quaternion.
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
    # Empty/whitespace => no ROI; resolved to the full frame at use time.
    if not s or not s.strip():
        return np.zeros((0, 2), dtype=np.int32)
    vals = [int(v.strip()) for v in s.split(",")]
    if len(vals) % 2 != 0:
        raise ValueError(f"Polygon string must have even number of values: {s}")
    return np.array(vals, dtype=np.int32).reshape(-1, 2)


def bbox_size_xyxy(b: tuple[int, int, int, int]) -> tuple[int, int]:
    x0, y0, x1, y1 = b
    return x1 - x0, y1 - y0


# Remove duplicate same-class states that occupy nearly the same camera-frame position.
def nms_by_position(
    states: list[ObjectTrackState],
    position_threshold: float = 0.05,
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
            tid_i = getattr(obj_states[i], "track_id", "") or ""

            for j in range(i + 1, len(obj_states)):
                if not keep_mask[j]:
                    continue
                if obj_states[j].T_object_camera is None:
                    continue
                tid_j = getattr(obj_states[j], "track_id", "") or ""
                if tid_i and tid_j and tid_i != tid_j:
                    continue
                pos_j = obj_states[j].T_object_camera[:3, 3]
                dist = np.linalg.norm(pos_i - pos_j)
                if dist < position_threshold:
                    keep_mask[j] = False

        kept.extend([s for s, k in zip(obj_states, keep_mask) if k])

    return kept


# Crop RGB and mask around a bbox, keeping a little context for DINO/memory crops.
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


def upscale_crop_if_small(
    rgb: np.ndarray,
    mask: np.ndarray,
    min_side: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Bicubic-upscale a small crop so DINOv2's input-size downsample
    doesn't throw away detail.
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


# Filter obviously over-large SAM masks before expensive DINO scoring.
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


# Keep masks whose bbox center lies inside the camera ROI polygon.
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


# Drop masks that mostly touch the image border; these are usually background spills.
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
    """Fraction of mask pixels with finite depth inside the valid range."""
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
    """How much of `inner` is covered by `outer`."""
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
    """Greedy bbox-level dedup, keeping larger masks first."""
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
    """Crop an RGB frame to the ROI bbox and shift polygon coords into crop space."""
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
    """Map masks produced on the ROI crop back into full-image coordinates."""
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
                prompt_score=c.prompt_score,
            )
        )
    return lifted


def batch_dino_classify(
    dino: DINOIdentifier,
    crops_rgb: list[np.ndarray],
    crops_mask: list[np.ndarray | None],
    objectness_priors: list[float | None] | None = None,
) -> list[DINOResult]:
    """Embed `crops_rgb` in one forward pass, then classify each via the ref bank."""

    if not crops_rgb:
        return []
    if objectness_priors is not None and len(objectness_priors) != len(crops_rgb):
        raise ValueError(
            f"objectness_priors len {len(objectness_priors)} != crops {len(crops_rgb)}"
        )

    tensors = []
    for rgb, mask in zip(crops_rgb, crops_mask):
        rgb_proc = dino._ensure_rgb(rgb)
        rgb_masked = dino._apply_mask(rgb_proc, mask)
        t = dino._preprocess(rgb_masked)
        tensors.append(t)

    batch = torch.cat(tensors, dim=0)

    with torch.inference_mode():
        out = dino.model.forward_features(batch)
    if not isinstance(out, dict):
        raise RuntimeError(
            f"forward_features returned {type(out)}; expected dict"
        )
    cls_tok = out["x_norm_clstoken"].reshape(batch.shape[0], -1)
    patch_toks = out["x_norm_patchtokens"]  # (B, N, D)
    gem = dino._gem_pool(patch_toks, p=float(dino.cfg.gem_p)).reshape(
        batch.shape[0], -1
    )
    cls_n = F.normalize(cls_tok, dim=1)
    gem_n = F.normalize(gem, dim=1)
    embeddings = torch.stack([cls_n, gem_n], dim=1).detach().cpu().numpy()

    results: list[DINOResult] = []
    for i, emb in enumerate(embeddings):
        prior = None
        if objectness_priors is not None:
            p = objectness_priors[i]
            if p is not None:
                prior = float(p)
        results.append(
            dino.classify_embedding(
                emb,
                objectness_prior=prior,
            )
        )
    return results



class FoundationPoseTrackerNode(Node):
    """ROS2 node that alternates between multicam initialization and fused tracking."""

    def __init__(self, args: argparse.Namespace, grabber: MultiCamGrabber, T_base_cam_map):
        super().__init__("foundationpose_tracker")
        self.args = args
        self.grabber = grabber
        self.T_base_cam_map = T_base_cam_map

        # Active cameras and stdout-backed logger are fixed once at startup.
        self.cameras = select_cameras(int(getattr(args, "num_cameras", 2)))

        self._unified_logger = _UnifiedLogger(
            verbose=bool(getattr(args, "debug_logging", False))
        )
        self.get_logger = lambda: self._unified_logger

        self.palette = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255),
        ]

        self.busy = False
        self.frame_counter = 0

        self.mesh_map = self._build_mesh_map(args.cad_dir)

        # Per-camera runtime state; each camera owns one ObjectTrackState per visible track.
        self.track_states: dict[str, list[ObjectTrackState]] = {
            c.cam_id: [] for c in self.cameras
        }

        # Per-camera SAM params, built from the cam{N}-* CLI args for each
        # active camera (cam 1 -> --cam1-*, cam 2 -> --cam2-*, ...).
        self.cam_sam_params: dict[str, CameraSAMParams] = {
            cam.cam_id: CameraSAMParams(
                min_mask_area=getattr(args, f"cam{i}_sam_min_mask_area"),
                min_bbox_side_px=getattr(args, f"cam{i}_sam_min_bbox_side_px"),
                max_mask_area_ratio=getattr(args, f"cam{i}_sam_max_mask_area_ratio"),
                max_bbox_area_ratio=getattr(args, f"cam{i}_sam_max_bbox_area_ratio"),
                border_px=getattr(args, f"cam{i}_sam_border_px"),
                max_border_fraction=getattr(args, f"cam{i}_sam_max_border_fraction"),
                roi_polygon=parse_polygon_string(getattr(args, f"cam{i}_roi_polygon")),
            )
            for i, cam in enumerate(self.cameras, start=1)
        }

        ref_source = args.reference_source
        primary_ref = (
            args.reference_renders_dir if ref_source == "renders"
            else args.reference_dir
        )
        extra_refs: list[str] = []
        if ref_source == "both":
            extra_refs.append(args.reference_renders_dir)

        # DINO owns object identity: build one reference bank shared by all cameras.
        self.dino = DINOIdentifier(
            DINOIdentifierConfig(
                model_name=args.dino_model_name,
                device=args.dino_device or args.device,
                reference_dir=primary_ref,
                use_masked_background=False,
                gem_p=float(args.dino_gem_p),
                verbose=bool(getattr(args, "debug_logging", False)),
            )
        )
        self.get_logger().info(
            f"Building DINO reference bank | source={ref_source} primary={primary_ref}"
            + (f" extra={extra_refs}" if extra_refs else "")
        )
        self.get_logger().info(
            f"DINO cfg | mode={MUSE_EMBEDDING_MODE} sim={MUSE_SIMILARITY} "
            f"gem_p={args.dino_gem_p} stream_alpha={MUSE_STREAM_ALPHA} "
            f"beta(joint)={MUSE_JOINT_SCORE_ALPHA} rel={MUSE_RELATIVE_SCORE_MODE} "
            f"tau={MUSE_TAU} gamma(prior)={MUSE_OBJECTNESS_PRIOR_GAMMA}"
        )
        self.dino.build_reference_bank_from_folder(extra_dirs=extra_refs)
        self.get_logger().info(
            f"DINO ready | objects={sorted(set(r.object_id for r in self.dino.reference_bank))}"
        )

        self.sam: Optional[SAMSegmenter] = None

        # SAM is heavy, so the node shares one segmenter across all camera streams.
        sam_modes = {"sam", "gdino_sam"}
        if args.mask_source in sam_modes:
            use_bf16 = not bool(getattr(args, "sam_fp32", False))
            print(
                f"[SAM-PRECISION] use_bfloat16={use_bf16} "
                f"({'bf16 autocast' if use_bf16 else 'fp32'}) "
                f"| pass --sam-fp32 to force fp32",
                flush=True,
            )
            # One shared SAM model for all cameras.
            min_area = min(p.min_mask_area for p in self.cam_sam_params.values())
            min_bbox = min(p.min_bbox_side_px for p in self.cam_sam_params.values())
            self.get_logger().info("Building shared SAM model for all cameras...")
            self.sam = SAMSegmenter(
                SAMSegmenterConfig(
                    repo_root=args.sam_repo_root,
                    checkpoint=args.sam_checkpoint,
                    model_cfg=args.sam_model_cfg,
                    device=args.device,
                    max_image_side=args.sam_max_image_side,
                    min_mask_area=min_area,
                    min_bbox_side_px=min_bbox,
                    attach_rgb_crops=False,
                    use_bfloat16=use_bf16,
                )
            )
            if not self.sam.warmup_or_rebuild():
                self.get_logger().warn(
                    "Shared SAM never warmed after rebuilds — masks may be unreliable"
                )
            torch.cuda.empty_cache()

        # Optional Grounding-DINO front-end proposes boxes before SAM masks them.
        self.gdino_proposer = None
        if args.mask_source == "gdino_sam":
            from src.perception.learned.GDINO.grounding_dino_proposal import (
                GDINOConfig, GroundingDINOProposer,
            )
            if bool(getattr(args, "gdino_use_items_prompt", False)):
                # MUSE-style class-agnostic prompt.
                text_prompts = ["items"]
            else:
                text_prompts = [p.strip() for p in args.gdino_text_prompts.split(",") if p.strip()]
            self.gdino_proposer = GroundingDINOProposer(
                GDINOConfig(
                    model_id=args.gdino_model_id,
                    device=args.gdino_device or args.device,
                    box_threshold=float(args.gdino_box_threshold),
                    text_threshold=float(args.gdino_text_threshold),
                    max_boxes_per_image=int(args.gdino_max_boxes),
                    text_prompts=text_prompts,
                )
            )
            try:
                self.gdino_proposer._lazy_load()
            except Exception as e:
                self.get_logger().warn(f"GDINO eager preload failed (will lazy-load): {e}")
            self.get_logger().info(
                f"GDINO proposer ready | model={args.gdino_model_id} "
                f"prompts={text_prompts}"
            )

        # One FoundationPose wrapper handles all cameras and object meshes.
        self.fp_tracker = FoundationPoseWrapper(
            FoundationPoseConfig(
                repo_root=args.fp_repo_root,
                weights_dir=args.fp_weights_dir,
                debug_dir=str(Path(args.output_root).resolve() / "fp_debug"),
                debug=args.fp_debug,
                mesh_scale=args.mesh_scale,
            )
        )

        # Pre-cache at least one mesh so the first init pays less model setup cost.
        self.get_logger().info("Pre-caching meshes for FoundationPose...")
        for obj_id, mesh_path in self.mesh_map.items():
            self.fp_tracker.preload_mesh(mesh_path=mesh_path, object_id=obj_id)
            break
        self.get_logger().info(f"Pre-cached {len(self.mesh_map)} meshes")
        
        # Real-time tracker (CuteVOS + ICP)
        self.realtime_trackers: dict[str, RealtimeTracker] = {}
        # One multi-object Cutie session per CAMERA (keyed by cam_id)
        self.cutie_sessions: dict[str, CutieTracker] = {}
        # Per-mesh shaft axis cache for optional PCA rotation correction.
        self._shaft_axis_cache: dict[str, Optional[np.ndarray]] = {}
        # Fused per-track memory: last base pose, metrics, filters, and publishers.
        self._fused_track_memory: dict[str, dict[str, Any]] = {}
        self._fused_icp_metrics: dict[str, dict[str, Any]] = {}
        self._fused_translation_kalman: dict[str, PoseKalmanFilter] = {}

        self.pub_pose_base: dict[str, Any] = {}
        self.pub_debug_frame: dict[str, Any] = {}


        # Load Cutie once before the first tracking tick to avoid a hot-loop stall.
        self._cutie_prewarmer = None
        run_mode = str(getattr(args, "run_mode", "track")).lower()
        if run_mode == "init_only":
            self.get_logger().info("Skipping Cutie pre-warm (run_mode=init_only)")
        else:
            self.get_logger().info("Pre-warming Cutie model...")
            self._cutie_prewarmer = CutieTracker(CutieConfig())
            self._cutie_prewarmer._lazy_load()
            self.get_logger().info("Cutie pre-warmed")

        if self._debug_frame_publish_enabled():
            for c in self.cameras:
                cid = c.cam_id
                self.pub_debug_frame[cid] = self.create_publisher(
                    DebugFrame, f"/perception/fp/debug_frame/{cid}", FAST_QOS
                )

        # Timer drives the init/track state machine; the grabber runs in the same executor.
        self.timer = self.create_timer(args.timer_period_s, self._tick)
        self.get_logger().info(
            f"FoundationPoseTrackerNode started | run_mode={self.args.run_mode} "
            f"profile={getattr(self.args, 'tracking_profile', 'default')}"
        )
        self._fused_warmup_count = {}       # track_id -> frames since init
        self._median_pose_buffers = {}      # track_id -> MedianPoseBuffer
        self._T_cam_base_cache: dict[str, np.ndarray] = {}

        # Per-track hold/lost tracking. lost_count is consecutive frames
        # without an accepted fused pose. pose_status flips fresh→held→
        # stale→lost
        self._fused_lost_count: dict[str, int] = {}
        self._fused_pose_status: dict[str, str] = {}
        self._force_reinit_tracks: set[str] = set()
        self._fused_last_drop_reasons: dict[str, dict[str, Any]] = {}

        # Per-class monotonic counter feeding _allocate_track_id.
        self._next_track_id_counter: dict[str, int] = {}
        # Last accepted base-frame centroid per track_id.
        self._last_known_track_centroids: dict[str, np.ndarray] = {}
        # track_id -> object_id reverse map.
        self._track_id_to_object_id: dict[str, str] = {}
        self._claimed_track_ids_this_init: set[str] = set()

        self._track_pose_log_path: Optional[Path] = None
        if bool(getattr(args, "log_track_poses", False)):
            # Overwrite pose logs at startup so each run has a clean CSV.
            self._track_pose_log_path = Path(str(args.track_pose_log_path))
            self._track_pose_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._track_pose_log_path.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "stamp_s",
                    "frame",
                    "object_id",
                    "track_id",
                    "pose_source",
                    "accepted",
                    "pose_status",
                    "lost_count",
                    "mode",
                    "pose_mode",
                    "cams",
                    "tx",
                    "ty",
                    "tz",
                    "fitness",
                    "rmse_mm",
                    "chamfer_mm",
                    "projected_overlap",
                    "dt_mm",
                    "drot_deg",
                    "reason",
                    "qx",
                    "qy",
                    "qz",
                    "qw",
                ])

        self._consecutive_chamfer_fails: dict[str, int] = {}
        self._last_reinit_time: dict[str, float] = {}
        self._init_chamfer_history: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=10)
        )
        # Recent healthy masks are a shortcut for reinit: match by IoU/centroid before DINO.
        self._recent_tracker_health: dict[tuple[str, str], dict] = {}

        # Per-object appearance memory bank.
        max_crops = int(getattr(args, "memory_crop_max_per_object", 8))
        self._memory_crops_by_object: dict[str, deque[MemoryCrop]] = defaultdict(
            lambda: deque(maxlen=max(1, max_crops))
        )
        self._memory_crops_by_track_id: dict[str, deque[MemoryCrop]] = defaultdict(
            lambda: deque(maxlen=max(1, max_crops))
        )

    # ── track_id allocation / cross-init identity preservation ──────────
    def _allocate_track_id(self, object_id: str) -> str:
        n = self._next_track_id_counter.get(object_id, 0)
        self._next_track_id_counter[object_id] = n + 1
        tid = f"{object_id}_inst{n}"
        self._track_id_to_object_id[tid] = object_id
        return tid

    def _resolve_track_id_for_new_detection(
        self,
        object_id: str,
        base_centroid: np.ndarray,
        match_radius_m: float = 0.08,
    ) -> str:
        """At init, try to reuse an existing track_id whose last accepted
        base-frame centroid is within match_radius_m of the new detection's
        centroid. Falls back to allocating a fresh id."""
        c = np.asarray(base_centroid, dtype=np.float64).reshape(3)
        best_tid: Optional[str] = None
        best_dist = float("inf")
        for tid, last_c in self._last_known_track_centroids.items():
            if self._track_id_to_object_id.get(tid) != object_id:
                continue
            if tid in self._claimed_track_ids_this_init:
                continue
            d = float(np.linalg.norm(np.asarray(last_c, dtype=np.float64).reshape(3) - c))
            if d < best_dist:
                best_dist = d
                best_tid = tid
        if best_tid is not None and best_dist <= match_radius_m:
            self._claimed_track_ids_this_init.add(best_tid)
            return best_tid
        tid = self._allocate_track_id(object_id)
        self._claimed_track_ids_this_init.add(tid)
        return tid

    @staticmethod
    def _build_mesh_map(cad_dir: str) -> dict[str, str]:
        """Map common object_id spellings to CAD mesh paths."""
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
        """Best-effort conversion for debug logging; fall back to the input pose."""
        try:
            return self._to_base_pose(cam_id, T_object_camera)
        except Exception:
            return np.asarray(T_object_camera, dtype=np.float32).reshape(4, 4)

    def _base_pose_string(self, cam_id: str, T_object_camera: np.ndarray) -> str:
        """Compact base-frame pose string for logs."""
        T_base = self._safe_to_base_pose(cam_id, T_object_camera)
        t = T_base[:3, 3]
        q = rotation_matrix_to_quaternion_xyzw(T_base[:3, :3])
        return (
            f"t_base=[{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}] "
            f"q_base=[{q[0]:.3f}, {q[1]:.3f}, {q[2]:.3f}, {q[3]:.3f}]"
        )
    

    @staticmethod
    def _fibonacci_rotations(n: int) -> list[np.ndarray]:
        """n uniformly-distributed SO(3) rotations (deterministic seed). Identity first."""
        from scipy.spatial.transform import Rotation as SciRot
        if n <= 1:
            return [np.eye(3)]
        mats = [np.eye(3)]
        mats.extend(list(SciRot.random(num=n - 1, random_state=42).as_matrix()))
        return mats

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
            n_rot: Optional[int] = None,
            icp_max_iter: int = 30,
        ) -> tuple:
            """
            Search rotation seeds around translation `t_base`, refine each by ICP,
            score by Chamfer (lowest = best)

            n_rot/icp_max_iter default to the init-grid behavior; the tracking
            rotation re-seed passes smaller values for a cheaper search.
            """
            if n_rot is None:
                n_rot = int(getattr(self.args, "icp_grid_n_rot", 100))
            n_rot = int(n_rot)
            prescreen = bool(getattr(self.args, "icp_grid_prescreen", False))
            prescreen_tau = float(getattr(self.args, "icp_grid_prescreen_tau", 0.06))

            CANDIDATES = self._fibonacci_rotations(n_rot)

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
                    fused_cloud, model_pcd, T_seed, max_iteration=int(icp_max_iter),
                    variant=self.args.icp_variant,
                )
                if fit < 0.10:
                    continue
                ch = self._grid_chamfer(model_pcd, fused_cloud, per_cam_clouds, T_ref)
                scored.append((ch, T_ref, fit, R_candidate.astype(np.float32)))

            if not scored:
                return None, float('inf')

            scored.sort(key=lambda x: x[0])

            best = scored[0]
            return best[1], float(best[0])

    # ── PCA shaft-axis correction helpers (used only by --fused-track-pca-axis) ──
    @staticmethod
    def _principal_axis(pts: np.ndarray) -> tuple:
        """PCA of a point set. Returns (unit principal axis, elongation ratio
        lambda1/lambda2). For an elongated object the principal axis is the
        shaft direction — a global estimate with no ICP local-minimum issue."""
        c = pts.mean(axis=0)
        X = pts - c
        cov = (X.T @ X) / max(len(pts) - 1, 1)
        evals, evecs = np.linalg.eigh(cov)  # ascending eigenvalues
        axis = evecs[:, -1]
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        l1 = float(evals[-1])
        l2 = float(evals[-2])
        return axis, (l1 / max(l2, 1e-12))

    @staticmethod
    def _axis_angle_to_R(axis: np.ndarray, angle: float) -> np.ndarray:
        """Rodrigues: rotation matrix for `angle` rad about unit `axis`."""
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        x, y, z = axis
        c = np.cos(angle)
        s = np.sin(angle)
        C = 1.0 - c
        return np.array([
            [c + x*x*C,   x*y*C - z*s, x*z*C + y*s],
            [y*x*C + z*s, c + y*y*C,   y*z*C - x*s],
            [z*x*C - y*s, z*y*C + x*s, c + z*z*C],
        ], dtype=np.float64)

    def _rotation_between(self, a: np.ndarray, b: np.ndarray, blend: float = 1.0) -> np.ndarray:
        """Minimal rotation taking unit vector `a` onto `b`, scaled by `blend`
        in [0,1] (1.0 = full snap). Handles the parallel/anti-parallel cases."""
        a = a / (np.linalg.norm(a) + 1e-12)
        b = b / (np.linalg.norm(b) + 1e-12)
        v = np.cross(a, b)
        s = float(np.linalg.norm(v))
        cdot = float(np.clip(np.dot(a, b), -1.0, 1.0))
        if s < 1e-9:
            if cdot > 0.0:
                return np.eye(3)
            # 180 deg: rotate about any axis perpendicular to a
            perp = np.array([0.0, 1.0, 0.0]) if abs(a[0]) > 0.9 else np.array([1.0, 0.0, 0.0])
            axis = np.cross(a, perp)
            full_angle = np.pi
        else:
            axis = v / s
            full_angle = float(np.arccos(cdot))
        return self._axis_angle_to_R(axis, full_angle * float(np.clip(blend, 0.0, 1.0)))

    def _object_shaft_axis(self, mesh_path: str, model_pcd) -> Optional[np.ndarray]:
        """Cached principal (shaft) axis of the object model, in object frame."""
        if mesh_path in self._shaft_axis_cache:
            return self._shaft_axis_cache[mesh_path]
        axis = None
        try:
            pts = np.asarray(model_pcd.points)
            if len(pts) >= 3:
                axis, _ = self._principal_axis(pts)
        except Exception:
            axis = None
        self._shaft_axis_cache[mesh_path] = axis
        return axis

    def _log_base_pose(
        self,
        stage: str,
        cam_id: str,
        instance_key: str,
        T_object_camera: np.ndarray,
        extra: str = "",
        ) -> None:
        pose_str = self._base_pose_string(cam_id, T_object_camera)
        suffix = f" | {extra}" if extra else ""
        self.get_logger().info(
            f"[{cam_id}] {stage} {instance_key} | {pose_str}{suffix}"
        )

    def _get_or_create_pose_base_pub(self, cam_id: str, instance_key: str) -> Any:
        """Lazy-create per-camera pose publishers keyed by track id."""
        key = f"{cam_id}/{instance_key}"
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

    def _debug_frame_publish_enabled(self) -> bool:
        """Central switch for all debug frame construction and publishing."""
        return bool(getattr(self.args, "debug_frame_publish", True))

    def _publish_pose_base(
        self,
        cam_id: str,
        instance_key: str,
        T_object_camera: np.ndarray,
        stamp,
        ) -> None:
        """Publish a camera-frame object pose after converting it to base frame."""
        T_base_object = self._to_base_pose(cam_id, T_object_camera)
        pub = self._get_or_create_pose_base_pub(cam_id, instance_key)
        pub.publish(T_to_pose_stamped(T_base_object, frame_id="base", stamp=stamp))

    def _make_mask_crop_msg(
        self,
        mask: np.ndarray,
        bbox_xyxy: tuple[int, int, int, int],
        max_side: int = 96,
        ) -> tuple[bool, DebugMaskCrop]:
        """Pack a small mask crop into the custom debug message."""
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
        """Assemble one debug frame message from candidates, poses, and tracker mask."""
        frame = DebugFrame()
        frame.stamp = stamp
        frame.cam_id = cam_id
        frame.max_candidate_draw = int(self.args.max_candidate_draw)
        frame.show_axes = True

        roi = self.cam_sam_params[cam_id].roi_polygon.reshape(-1)
        frame.roi_polygon_xy_flat = [int(v) for v in roi.tolist()]

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
        """Convert raw SAM candidates into compact debug messages."""
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
        """Convert DINO-ranked selections into debug candidate messages."""
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
        """Convert tracked state into debug pose overlays."""
        out: list[DebugPoseItem] = []

        for s in states:
            if s.T_object_camera is None:
                continue

            msg = DebugPoseItem()
            msg.object_id = str(s.object_id)
            msg.mode = str(s.mode)
            msg.score = float(s.dino_score)

            # Convert raw camera-frame CAD pose to base frame.
            T_base_dbg = self._safe_to_base_pose(cam_id, s.T_object_camera)

            # Store corrected base pose.
            msg.pose_base = T_to_pose_msg(T_base_dbg)

            # Recompute corrected camera-frame pose from corrected base pose.
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
        """Resolve object_id to a CAD file, accepting direct fallback paths."""
        if object_id in self.mesh_map:
            return self.mesh_map[object_id]

        for ext in (".obj", ".stl"):
            direct = Path(self.args.cad_dir) / f"{object_id}{ext}"
            if direct.exists():
                return str(direct)

        raise FileNotFoundError(f"No CAD mesh for object_id='{object_id}'")

    def _to_base_pose(self, cam_id: str, T_object_camera: np.ndarray) -> np.ndarray:
        """Convert object pose from camera coordinates into the shared base frame."""
        if cam_id not in self.T_base_cam_map:
            raise KeyError(f"No base extrinsic for cam_id={cam_id}")
        T_object_camera = np.asarray(T_object_camera, dtype=np.float32).reshape(4, 4)
        T_base_cam = self._resolve_T_base_cam(cam_id)
        return (T_base_cam @ T_object_camera.astype(np.float64)).astype(np.float32)

    def _mask_base_centroid(
        self,
        cam_id: str,
        depth: np.ndarray | None,
        K: np.ndarray | None,
        mask: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Lift a mask to base-frame depth points and return its centroid."""
        if depth is None or K is None:
            return None
        try:
            pcd = lift_masked_depth_to_base(
                depth=np.asarray(depth),
                mask=np.asarray(mask).astype(bool),
                K=np.asarray(K, dtype=np.float32).reshape(3, 3),
                T_base_cam=self._resolve_T_base_cam(cam_id),
                z_min=float(getattr(self.args, "min_valid_z_m", 0.05)),
                z_max=float(getattr(self.args, "max_valid_z_m", 3.0)),
                voxel_size=0.004,
            )
            if pcd is None or len(pcd.points) < 10:
                return None
            pts = np.asarray(pcd.points, dtype=np.float64)
            return pts.mean(axis=0).reshape(3)
        except Exception as e:
            if bool(getattr(self.args, "debug_verbose_logs", False)):
                self.get_logger().info(
                    f"identity centroid failed cam={cam_id}: {e}"
                )
            return None

    def _track_matches_by_object(
        self,
        base_centroid: Optional[np.ndarray],
        object_ids: Iterable[str],
    ) -> tuple[dict[str, str], set[str], dict[str, float]]:
        """Find nearby live tracks for candidate memory/DINO disambiguation."""
        matched: dict[str, str] = {}
        ambiguous: set[str] = set()
        nearest_dist: dict[str, float] = {}
        if base_centroid is None:
            return matched, ambiguous, nearest_dist

        c = np.asarray(base_centroid, dtype=np.float64).reshape(3)
        max_dist = float(getattr(self.args, "identity_shortcut_max_centroid_dist_m", 0.05))
        ambiguity_radius = float(getattr(self.args, "identity_shortcut_ambiguity_radius_m", 0.05))

        for obj_id in {str(o) for o in object_ids if str(o)}:
            candidates: list[tuple[float, str]] = []
            for tid, last_c in self._last_known_track_centroids.items():
                if self._track_id_to_object_id.get(tid) != obj_id:
                    continue
                d = float(np.linalg.norm(np.asarray(last_c, dtype=np.float64).reshape(3) - c))
                candidates.append((d, tid))
            if not candidates:
                continue
            candidates.sort(key=lambda x: x[0])
            nearest_dist[obj_id] = candidates[0][0]
            if len(candidates) >= 2 and candidates[1][0] <= ambiguity_radius:
                ambiguous.add(obj_id)
                continue
            if candidates[0][0] <= max_dist:
                matched[obj_id] = candidates[0][1]
        return matched, ambiguous, nearest_dist


    def _inherit_from_tracker_health(
        self,
        cam_id: str,
        masks: list[SAMMaskCandidate],
        depth: np.ndarray | None = None,
        K: np.ndarray | None = None,
        ) -> tuple[list[CandidateSelection], list[SAMMaskCandidate]]:
        """Assign masks to recently healthy tracks, avoiding DINO when identity is clear."""
        if not masks:
            return [], masks
        stale_frames = int(getattr(self.args, "tracker_health_stale_frames", 5))
        iou_thr = float(getattr(self.args, "tracker_health_iou_threshold", 0.5))
        max_occ = float(getattr(self.args, "tracker_health_max_occlusion", 0.4))

        max_centroid_dist = float(getattr(self.args, "identity_shortcut_max_centroid_dist_m", 0.05))
        ambiguity_radius = float(getattr(self.args, "identity_shortcut_ambiguity_radius_m", 0.05))

        # Collect healthy entries for this cam, per-instance (track_id-keyed).
        healthy: list[tuple[str, str, np.ndarray, Optional[np.ndarray]]] = []
        for (cid, track_id), info in self._recent_tracker_health.items():
            if cid != cam_id:
                continue
            if (self.frame_counter - int(info.get("frame_idx", 0))) > stale_frames:
                continue
            if float(info.get("occlusion_score", 0.0)) > max_occ:
                continue
            mask = info.get("mask")
            if mask is None or int(np.asarray(mask).sum()) <= 0:
                continue
            obj_id = str(info.get("object_id", ""))
            if not obj_id or not track_id:
                continue
            centroid = info.get("centroid_base")
            if centroid is None and info.get("T_object_camera") is not None:
                try:
                    centroid = self._safe_to_base_pose(
                        cam_id, info["T_object_camera"]
                    )[:3, 3]
                except Exception:
                    centroid = None
            centroid_arr = (
                np.asarray(centroid, dtype=np.float64).reshape(3)
                if centroid is not None else None
            )
            healthy.append((track_id, obj_id, np.asarray(mask, dtype=bool), centroid_arr))

        if not healthy:
            return [], masks

        inherited: list[CandidateSelection] = []
        remaining: list[SAMMaskCandidate] = []
        used_track_ids: set[str] = set()
        for cand in masks:
            cm = np.asarray(cand.mask, dtype=bool)
            cm_sum = int(cm.sum())
            if cm_sum == 0:
                remaining.append(cand)
                continue
            best_iou = 0.0
            best_obj: Optional[str] = None
            best_tid: Optional[str] = None
            best_centroid_dist: Optional[float] = None
            cand_centroid = self._mask_base_centroid(cam_id, depth, K, cm)
            for tid, obj_id, hmask, hcentroid in healthy:
                if tid in used_track_ids:
                    continue
                if hmask.shape != cm.shape:
                    continue
                if cand_centroid is None or hcentroid is None:
                    continue
                centroid_dist = float(np.linalg.norm(cand_centroid - hcentroid))
                if centroid_dist > max_centroid_dist:
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
                    best_tid = tid
                    best_centroid_dist = centroid_dist
            if best_obj is not None and best_tid is not None and best_iou >= iou_thr:
                ambiguous = False
                if cand_centroid is not None:
                    for tid, obj_id, _hmask, hcentroid in healthy:
                        if tid == best_tid or obj_id != best_obj or hcentroid is None:
                            continue
                        if float(np.linalg.norm(cand_centroid - hcentroid)) <= ambiguity_radius:
                            ambiguous = True
                            break
                if ambiguous:
                    remaining.append(cand)
                    self.get_logger().info(
                        f"[identity] DINO-skip blocked cam={cam_id} track={best_tid} "
                        f"obj={best_obj}: nearby same-class track"
                    )
                    continue
                inherited.append(CandidateSelection(
                    object_id=best_obj,
                    score=1.0,
                    scores_by_object={best_obj: 1.0},
                    candidate=cand,
                    base_scores_by_object={best_obj: 1.0},
                    score_source="tracker_health",
                    track_id_hint=best_tid,
                    dino_class_score=1.0,
                    dino_margin=1.0,
                ))
                used_track_ids.add(best_tid)
                dist_txt = (
                    f"{best_centroid_dist:.3f}m"
                    if best_centroid_dist is not None else "unknown"
                )
                self.get_logger().info(
                    f"[identity] inherited cam={cam_id} track={best_tid} obj={best_obj} "
                    f"iou={best_iou:.3f} centroid_dist={dist_txt}"
                )
            else:
                remaining.append(cand)
        return inherited, remaining

    def _generate_and_filter_masks(self, rgb: np.ndarray, cam_id: str) -> list[SAMMaskCandidate]:
        """Run SAM/GDINO inside the camera ROI, then apply cheap geometry filters."""
        if self.sam is None:
            return []

        sam = self.sam
        cam_params = self.cam_sam_params[cam_id]

        # [VRAM] free-memory at each SAM forward.
        try:
            _free, _total = torch.cuda.mem_get_info()
            _dprint(f"[VRAM] {cam_id}: free={_free/1e9:.2f}GB / {_total/1e9:.1f}GB")
        except Exception:
            pass
        full_h, full_w = rgb.shape[:2]
        polygon_full = cam_params.roi_polygon
        if polygon_full.shape[0] < 3:
            polygon_full = np.array(
                [[0, 0], [full_w, 0], [full_w, full_h], [0, full_h]],
                dtype=np.int32,
            )

        # --- Crop to ROI ---
        t0 = time.time()
        rgb_crop, polygon_crop, crop_x0, crop_y0 = crop_rgb_to_polygon_bbox(rgb, polygon_full)
        crop_h, crop_w = rgb_crop.shape[:2]

        roi_mask_crop = np.zeros((crop_h, crop_w), dtype=np.uint8)
        cv2.fillPoly(roi_mask_crop, [polygon_crop], 255)
        rgb_crop_masked = rgb_crop.copy()
        rgb_crop_masked[roi_mask_crop == 0] = 0

        _dprint(f"[TIMING]   ROI crop prep: {(time.time() - t0)*1000:.0f}ms")

        # --- Main proposal stage ---
        t1 = time.time()
        if (self.args.mask_source == "gdino_sam"
                and self.gdino_proposer is not None):
            proposals = self.gdino_proposer.propose(rgb_crop_masked)
            # Record how many box proposals SAM was handed this call
            self._last_sam_n_boxes = len(proposals)
            if proposals:
                boxes = np.array([p.bbox_xyxy for p in proposals], dtype=np.float32)
                box_scores = np.array([p.score for p in proposals], dtype=np.float32)
                # [IMG-DBG] diagnostic: image + box sanity for the cam being processed
                _img = rgb_crop_masked
                _nan = int(np.isnan(_img).sum()) if np.issubdtype(_img.dtype, np.floating) else 0
                _bw = boxes[:, 2] - boxes[:, 0]
                _bh = boxes[:, 3] - boxes[:, 1]
                _dprint(
                    f"[IMG-DBG] {cam_id}: img shape={_img.shape} dtype={_img.dtype} "
                    f"min={float(_img.min()):.1f} max={float(_img.max()):.1f} "
                    f"mean={float(_img.mean()):.1f} nan={_nan} | "
                    f"boxes n={len(boxes)} x=[{float(boxes[:,0].min()):.0f},{float(boxes[:,2].max()):.0f}] "
                    f"y=[{float(boxes[:,1].min()):.0f},{float(boxes[:,3].max()):.0f}] "
                    f"degenerate={int(((_bw<=0)|(_bh<=0)).sum())} "
                    f"box_score[min,max]=[{float(box_scores.min()):.2f},{float(box_scores.max()):.2f}] "
                    f"box_score_nan={int((~np.isfinite(box_scores)).sum())}"
                )
                masks_crop = sam.generate_from_boxes(
                    rgb_crop_masked, boxes, box_scores=box_scores,
                )
            else:
                masks_crop = []
            _dprint(
                f"[TIMING]   GDINO+SAM (main): {(time.time() - t1)*1000:.0f}ms "
                f"-> {len(proposals)} boxes -> {len(masks_crop)} masks"
            )
        else:
            masks_crop = sam.generate_auto(rgb_crop_masked)
            _dprint(f"[TIMING]   SAM generate_auto (main): {(time.time() - t1)*1000:.0f}ms -> {len(masks_crop)} raw masks")

        if not masks_crop:
            masks = []
        else:
            # --- Filtering ---
            t2 = time.time()
            # Per-camera min area / min bbox side.
            masks_crop = [
                m for m in masks_crop
                if m.area >= cam_params.min_mask_area
                and min(m.bbox_xyxy[2] - m.bbox_xyxy[0],
                        m.bbox_xyxy[3] - m.bbox_xyxy[1]) >= cam_params.min_bbox_side_px
            ]
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
            _dprint(f"[TIMING]   Mask filtering: {(time.time() - t2)*1000:.0f}ms -> {len(masks)} after filter")

        masks = sorted(masks, key=lambda m: m.area)

        # --- Final filtering ---
        t5 = time.time()
        masks = reject_outside_roi_polygon(masks, polygon_full)
        masks = dedup_masks_by_bbox_iou(masks, iou_thresh=self.args.mask_dedup_iou)
        _dprint(f"[TIMING]   Final filter + dedup: {(time.time() - t5)*1000:.0f}ms -> {len(masks)} final")

        return masks

    # ── per-object appearance memory ─────────────────────────────────────
    def _maybe_save_memory_crop(
        self,
        object_id: str,
        track_id: str,
        cam_id: str,
        rgb_full: np.ndarray,
        mask_full: np.ndarray,
        bbox_xyxy: Optional[tuple[int, int, int, int]],
        fitness: float,
        ) -> bool:
        """Snapshot a clean RGB crop into the per-object memory bank when the
        fused ICP fitness is good. Returns True if a crop was saved."""
        if not getattr(self.args, "memory_crop_enable", False):
            return False
        if self.dino is None:
            return False
        threshold = float(self.args.memory_crop_fitness_threshold)
        if fitness < threshold:
            return False

        bank = self._memory_crops_by_object[object_id]
        track_bank = self._memory_crops_by_track_id[track_id] if track_id else None
        gap = int(self.args.memory_crop_min_frame_gap)
        if gap > 0 and bank:
            for entry in reversed(bank):
                if entry.cam_id == cam_id and (not track_id or entry.track_id == track_id):
                    if self.frame_counter - entry.frame_idx < gap:
                        return False
                    break

        if bbox_xyxy is None:
            ys, xs = np.where(mask_full.astype(bool))
            if xs.size == 0:
                return False
            bbox_xyxy = (int(xs.min()), int(ys.min()),
                         int(xs.max()) + 1, int(ys.max()) + 1)

        try:
            crop_rgb, crop_mask = bbox_crop_with_local_mask(
                rgb_full, mask_full.astype(bool), bbox_xyxy,
            )
        except Exception as e:
            self.get_logger().warn(f"memory-crop bbox crop failed: {e}")
            return False
        if crop_rgb.size == 0 or int(crop_mask.sum()) == 0:
            return False

        try:
            embedding = self.dino.embed_image(crop_rgb, crop_mask)
        except Exception as e:
            self.get_logger().warn(f"memory-crop DINO embed failed: {e}")
            return False

        keep_rgb = bool(getattr(self.args, "memory_crop_keep_rgb", False))
        entry = MemoryCrop(
            object_id=object_id,
            cam_id=cam_id,
            frame_idx=self.frame_counter,
            fitness=float(fitness),
            embedding=np.asarray(embedding, dtype=np.float32),
            track_id=str(track_id or ""),
            rgb_crop=crop_rgb if keep_rgb else None,
            mask_crop=crop_mask if keep_rgb else None,
        )
        bank.append(entry)
        if track_bank is not None:
            track_bank.append(entry)

        save_dir = str(self.args.memory_crop_save_dir or "").strip()
        if save_dir:
            try:
                out_dir = Path(save_dir) / object_id
                out_dir.mkdir(parents=True, exist_ok=True)
                fname = f"f{self.frame_counter:06d}_{cam_id}_fit{fitness:.2f}.png"
                bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(out_dir / fname), bgr)
            except Exception as e:
                self.get_logger().warn(f"memory-crop save_dir write failed: {e}")

        self.get_logger().info(
            f"[memcrop] saved {object_id}[{track_id}] from {cam_id} fit={fitness:.3f} "
            f"bank={len(bank)}/{bank.maxlen}"
        )
        return True

    def _memory_similarity_by_object(
        self,
        query_embedding: np.ndarray,
        matched_track_ids_by_object: Optional[dict[str, str]] = None,
        ambiguous_objects: Optional[set[str]] = None,
        ) -> dict[str, float]:
        """Return max cosine similarity between the query embedding and each
        object's memory bank. Assumes all embeddings are L2-normalised."""
        out: dict[str, float] = {}
        if query_embedding is None:
            return out
        q = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        if q.size == 0:
            return out
        matched_track_ids_by_object = matched_track_ids_by_object or {}
        ambiguous_objects = ambiguous_objects or set()
        require_consistency = bool(
            getattr(self.args, "memory_rerank_require_track_consistency", True)
        )
        for obj_id, bank in self._memory_crops_by_object.items():
            if obj_id in ambiguous_objects:
                continue
            track_id = matched_track_ids_by_object.get(obj_id, "")
            track_bank = self._memory_crops_by_track_id.get(track_id) if track_id else None
            selected_bank = track_bank if track_bank else None
            if selected_bank is None:
                if require_consistency and not track_id:
                    continue
                selected_bank = bank
            if not selected_bank:
                continue
            mat = np.stack([e.embedding.reshape(-1) for e in selected_bank], axis=0)
            if mat.shape[1] != q.shape[0]:
                continue
            sims = mat @ q
            out[obj_id] = float(np.max(sims))
        return out

    def _classify_masks_batched(
        self,
        rgb: np.ndarray,
        masks: list[SAMMaskCandidate],
        cam_id: str = "",
        depth: np.ndarray | None = None,
        K: np.ndarray | None = None,
        ) -> list[CandidateSelection]:
        """Batch-classify SAM masks with DINO and optionally blend appearance memory."""
        if not masks:
            return []

        crops_rgb: list[np.ndarray] = []
        crops_mask: list[np.ndarray | None] = []
        crops_prior: list[float | None] = []
        valid_indices: list[int] = []

        dino_min_crop = int(getattr(self.args, "dino_min_crop_side", 0))
        for i, cand in enumerate(masks):
            crop_rgb, crop_mask = bbox_crop_with_local_mask(rgb, cand.mask, cand.bbox_xyxy)
            if crop_rgb.size == 0 or int(crop_mask.sum()) == 0:
                continue
            # Upscale tiny crops so DINOv2's resize-to-input_size keeps object detail.
            crop_rgb, crop_mask = upscale_crop_if_small(crop_rgb, crop_mask, dino_min_crop)
            crops_rgb.append(crop_rgb)
            crops_mask.append(crop_mask)
            crops_prior.append(getattr(cand, "prompt_score", None))
            valid_indices.append(i)

        if not crops_rgb:
            return []

        try:
            dino_results = batch_dino_classify(
                self.dino, crops_rgb, crops_mask, objectness_priors=crops_prior,
            )
        except Exception as e:
            self.get_logger().warn(f"Batched DINO failed: {e}")
            if (
                getattr(self.dino, "device", None) is not None
                and self.dino.device.type == "cuda"
                and "illegal memory access" in str(e).lower()
            ):
                self.get_logger().fatal(
                    "DINOv2 CUDA hit an illegal memory access; exiting before "
                    "the poisoned CUDA context crashes later stages. Re-run with "
                    "--dino-device cpu."
                )
                os._exit(42)
            return []

        out: list[CandidateSelection] = []
        img_area = float(rgb.shape[0] * rgb.shape[1])

        mem_weight = float(getattr(self.args, "memory_crop_weight", 0.0))
        mem_floor = float(getattr(self.args, "memory_crop_min_score_floor", 0.0))
        mem_enabled = (
            bool(getattr(self.args, "memory_crop_enable", False))
            and mem_weight > 0.0
            and (
                any(len(b) > 0 for b in self._memory_crops_by_object.values())
                or any(len(b) > 0 for b in self._memory_crops_by_track_id.values())
            )
        )

        for j, res in enumerate(dino_results):
            mask_idx = valid_indices[j]
            cand = masks[mask_idx]

            base_scores = {
                k: float(v)
                for k, v in res.scores_by_object.items()
                if np.isfinite(float(v))
            }
            if not base_scores:
                self.get_logger().warn(
                    f"DINO cand {j} skipped: no finite class scores"
                )
                continue
            decision_scores = dict(base_scores)
            score_source = "dino_base"

            mem_sims: dict[str, float] = {}
            matched_tracks: dict[str, str] = {}
            ambiguous_objects: set[str] = set()
            nearest_dists: dict[str, float] = {}
            if mem_enabled and base_scores:
                base_top = max(base_scores.values())
                if base_top >= mem_floor:
                    # Only let memory influence classes whose nearby track identity is unambiguous.
                    cand_centroid = self._mask_base_centroid(cam_id, depth, K, cand.mask)
                    matched_tracks, ambiguous_objects, nearest_dists = (
                        self._track_matches_by_object(cand_centroid, base_scores.keys())
                    )
                    mem_sims = self._memory_similarity_by_object(
                        res.embedding,
                        matched_track_ids_by_object=matched_tracks,
                        ambiguous_objects=ambiguous_objects,
                    )
                    if mem_sims:
                        mem_sims = {
                            k: float(v)
                            for k, v in mem_sims.items()
                            if np.isfinite(float(v))
                        }
                        blended: dict[str, float] = {}
                        for k, v in base_scores.items():
                            if k in mem_sims:
                                blended[k] = (1.0 - mem_weight) * v + mem_weight * mem_sims[k]
                            else:
                                blended[k] = v
                        base_top1 = max(base_scores, key=base_scores.get)
                        new_top1 = max(blended, key=blended.get)
                        if new_top1 != base_top1:
                            tid = matched_tracks.get(new_top1, "")
                            dist = nearest_dists.get(new_top1)
                            dist_txt = f"{dist:.3f}m" if dist is not None else "unknown"
                            self.get_logger().info(
                                f"[memcrop] cand {j} flipped top1 {base_top1}->{new_top1} "
                                f"track={tid or 'none'} dist={dist_txt} "
                                f"(base={base_scores[base_top1]:.3f}->{base_scores.get(new_top1, 0.0):.3f}, "
                                f"mem={mem_sims.get(base_top1, 0.0):.3f}->{mem_sims.get(new_top1, 0.0):.3f})"
                            )
                        decision_scores = blended
                        score_source = "memory_blended"
                    elif ambiguous_objects and bool(getattr(self.args, "debug_verbose_logs", False)):
                        self.get_logger().info(
                            f"[memcrop] cand {j} skipped: ambiguous nearby tracks "
                            f"{sorted(ambiguous_objects)}"
                        )

            decision_scores = {
                k: float(v)
                for k, v in decision_scores.items()
                if np.isfinite(float(v))
            }
            sorted_scores = sorted(
                decision_scores.items(), key=lambda kv: kv[1], reverse=True
            )
            if not sorted_scores:
                self.get_logger().warn(
                    f"DINO cand {j} skipped: decision scores became non-finite"
                )
                continue

            top1_name, top1_score = sorted_scores[0]
            top2_name, top2_score = sorted_scores[1] if len(sorted_scores) > 1 else ("", -1.0)

            decision_best_score = float(top1_score)
            object_id = top1_name

            top_dbg = ", ".join([f"{k}:{v:.3f}" for k, v in sorted_scores[:4]])
            if decision_best_score < 0.80:
                self.get_logger().info(
                    f"DINO cand {j} | source={score_source} top scores: {top_dbg}"
                )

            second_score = float(top2_score)
            margin = float(top1_score - second_score)
            if not np.isfinite(decision_best_score) or not np.isfinite(margin):
                self.get_logger().warn(
                    f"DINO cand {j} rejected: non-finite score/margin "
                    f"score={decision_best_score} margin={margin}"
                )
                object_id = "unknown"

            bw, bh = bbox_size_xyxy(cand.bbox_xyxy)
            bbox_area = bw * bh
            is_small_object = bbox_area < 5000

            if is_small_object:
                # Small masks get lower absolute-score tolerance but still need a margin.
                min_score_for_small = 0.40
                min_margin_for_small = 0.025
                if decision_best_score < min_score_for_small:
                    object_id = "unknown"
                elif margin < min_margin_for_small:
                    object_id = "unknown"

                if object_id != "unknown":
                    self.get_logger().info(
                        f"DINO small obj ACCEPT: {object_id} score={decision_best_score:.3f} margin={margin:.3f} bbox={bw}x{bh}"
                    )
            else:
                if decision_best_score < self.args.dino_min_score:
                    object_id = "unknown"
                if self.args.dino_min_margin > 0.0 and margin < self.args.dino_min_margin:
                    object_id = "unknown"

            area_ratio = float(cand.area) / img_area
            x0, y0, x1, y1 = cand.bbox_xyxy
            bbox_area = max(1, (x1 - x0) * (y1 - y0))
            fill_ratio = float(cand.area) / float(bbox_area)

            if object_id == "unknown":
                self.get_logger().info(
                    f"UNKNOWN: score={decision_best_score:.3f}, margin={margin:.3f}, bbox={bw}x{bh}"
                )

            final_score = (
                decision_best_score
                - self.args.area_penalty_weight * area_ratio
                + self.args.fill_ratio_weight * fill_ratio
            )

            out.append(
                CandidateSelection(
                    object_id=object_id,
                    score=final_score,
                    scores_by_object={k: float(v) for k, v in decision_scores.items()},
                    candidate=cand,
                    base_scores_by_object={k: float(v) for k, v in base_scores.items()},
                    score_source=score_source,
                    track_id_hint=matched_tracks.get(top1_name, ""),
                    dino_class_score=decision_best_score,
                    dino_margin=margin,
                )
            )

        out.sort(key=lambda x: x.score, reverse=True)
        return out

    def _select_top_candidates(
        self,
        ranked: list[CandidateSelection],
        depth: np.ndarray,
        ) -> list[CandidateSelection]:
        """Keep high-confidence, non-overlapping candidates with enough usable depth."""
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
        """Coarse sanity checks for FoundationPose camera-frame estimates."""
        R = T_camera[:3, :3]
        t_cam = T_camera[:3, 3]

        trace = np.trace(R)
        if trace < -1.5:
            return False, f"flipped_orientation trace={trace:.3f}"

        t_mag = np.linalg.norm(t_cam)
        if t_mag < 0.4 or t_mag > 1.5:
            return False, f"bad_distance mag={t_mag:.3f}"

        try:
            T_base = self._to_base_pose(cam_id, T_camera)
            z_base = float(T_base[2, 3])
            z_lo = float(getattr(self.args, "table_plane_z_min", 0.0))
            z_hi = float(getattr(self.args, "table_plane_z_max", 0.9))
            if z_base < z_lo or z_base > z_hi:
                return False, f"bad_z_base z={z_base:.3f} (table window [{z_lo:.3f}, {z_hi:.3f}])"
        except Exception:
            pass

        return True, "ok"
    
    def _get_previous_object_base_pose(
        self,
        track_id: str,
        per_object_entries: list[dict],
     ) -> Optional[np.ndarray]:
        """Prefer fused pose memory, then fall back to any per-camera last-good pose."""
        mem = self._fused_track_memory.get(track_id, {})
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
        """Translation/rotation delta between two base-frame poses."""
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

    @staticmethod
    def _csv_float(value: Any, digits: int = 6) -> str:
        try:
            value_f = float(value)
        except Exception:
            return ""
        if not np.isfinite(value_f):
            return ""
        return f"{value_f:.{digits}f}"

    def _log_track_pose_csv(self, object_decisions: dict[str, dict], stamp) -> None:
        """Append one row per fused track with pose, gates, and quality metrics."""
        if self._track_pose_log_path is None:
            return

        stamp_s = self._stamp_to_seconds(stamp)
        rows: list[list[Any]] = []
        for track_id, decision in sorted(object_decisions.items()):
            pose_source = "decision"
            T_base = decision.get("T_base")
            pose_status = str(decision.get("pose_status", ""))
            if T_base is None and pose_status == "held":
                mem = self._fused_track_memory.get(track_id)
                if mem is not None:
                    T_base = mem.get("T_base")
                    pose_source = "held_memory"

            if T_base is not None:
                t = np.asarray(T_base[:3, 3], dtype=np.float64).reshape(3)
                tx, ty, tz = (self._csv_float(v) for v in t)
                q = rotation_matrix_to_quaternion_xyzw(
                    np.asarray(T_base[:3, :3], dtype=np.float64)
                )
                qx, qy, qz, qw = (self._csv_float(v, digits=6) for v in q)
            else:
                pose_source = "none"
                tx = ty = tz = ""
                qx = qy = qz = qw = ""

            chamfer_m = decision.get("chamfer_m", float("nan"))
            chamfer_mm = (
                ""
                if not np.isfinite(float(chamfer_m))
                or float(chamfer_m) < 0.0
                else self._csv_float(float(chamfer_m) * 1000.0, digits=3)
            )
            cams = "|".join(str(c) for c in decision.get("survived_cam_ids", []))
            rows.append([
                self._csv_float(stamp_s),
                int(self.frame_counter),
                decision.get("object_id", ""),
                track_id,
                pose_source,
                bool(decision.get("accepted", False)),
                pose_status,
                int(decision.get("lost_count", 0)),
                decision.get("mode", ""),
                decision.get("pose_mode", ""),
                cams,
                tx,
                ty,
                tz,
                self._csv_float(decision.get("fitness", float("nan")), digits=4),
                self._csv_float(float(decision.get("rmse_m", float("nan"))) * 1000.0, digits=3),
                chamfer_mm,
                self._csv_float(decision.get("projected_overlap", float("nan")), digits=4),
                self._csv_float(float(decision.get("trans_jump_m", 0.0)) * 1000.0, digits=3),
                self._csv_float(decision.get("rot_jump_deg", 0.0), digits=3),
                decision.get("reason", ""),
                qx,
                qy,
                qz,
                qw,
            ])

        try:
            with self._track_pose_log_path.open("a", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(rows)
        except Exception as e:
            self.get_logger().warn(f"track pose CSV log failed: {e}")

    def _compute_fused_motion_dt(self, track_id: str, stamp) -> tuple[float, Optional[float]]:
        """Clamp frame dt so motion gates scale with time without exploding on jitter."""
        nominal_dt = float(getattr(self.args, 'fused_track_nominal_dt_s', 0.15))
        dt_min = float(getattr(self.args, 'fused_track_min_dt_s', 0.10))
        dt_max = float(getattr(self.args, 'fused_track_max_dt_s', 0.30))
        t_now = self._stamp_to_seconds(stamp)
        t_prev = self._fused_track_memory.get(track_id, {}).get('stamp_s')
        dt_raw: Optional[float] = None
        if t_now is not None and t_prev is not None:
            dt_raw = max(0.0, float(t_now) - float(t_prev))
        dt_eff = nominal_dt if dt_raw is None else dt_raw
        dt_eff = float(np.clip(dt_eff, dt_min, dt_max))
        return dt_eff, dt_raw

    def _compute_fused_motion_thresholds(self, dt_eff: float) -> tuple[float, float]:
        """Turn velocity limits into per-frame translation/rotation jump gates."""
        v_max = float(getattr(self.args, 'fused_track_max_translation_speed_mps', 0.13333333333333333))
        w_max = float(getattr(self.args, 'fused_track_max_rotation_speed_degps', 66.66666666666667))
        min_trans = float(getattr(self.args, 'fused_track_min_translation_jump_m', 0.008))
        min_rot = float(getattr(self.args, 'fused_track_min_rotation_jump_deg', 4.0))
        return max(min_trans, v_max * float(dt_eff)), max(min_rot, w_max * float(dt_eff))

    def _remember_fused_drop_reason(
        self,
        track_id: str,
        decision: dict[str, Any],
        *,
        accepted: bool,
        status: str,
    ) -> None:
        """Keep the last rejected decision so forced reinit logs explain what failed."""
        if accepted:
            self._fused_last_drop_reasons.pop(track_id, None)
            return

        reason = str(decision.get("reason", "") or "unknown")
        self._fused_last_drop_reasons[track_id] = {
            "object_id": str(
                decision.get("object_id", "")
                or self._track_id_to_object_id.get(track_id, "")
            ),
            "reason": reason,
            "mode": str(decision.get("mode", "")),
            "pose_mode": str(decision.get("pose_mode", "")),
            "pose_status": status,
            "lost_count": int(self._fused_lost_count.get(track_id, 0)),
            "gate_reasons": self._summarize_fused_gate_metrics(
                decision.get("gate_metrics", [])
            ),
        }

    @staticmethod
    def _summarize_fused_gate_metrics(gate_metrics: Any) -> str:
        if not gate_metrics:
            return ""
        parts = []
        for m in gate_metrics:
            try:
                cam = str(m.get("cam_id", "?"))
                reason = str(m.get("reason", "unknown") or "unknown")
                mask_area = int(m.get("mask_area", 0) or 0)
                ratio = m.get("mask_area_ratio")
                ratio_s = "na" if ratio is None else f"{float(ratio):.2f}"
                cov = float(m.get("depth_coverage", 0.0) or 0.0)
                pts = int(m.get("cloud_points", 0) or 0)
                centroid = m.get("cloud_centroid_dist_m")
                centroid_s = "na" if centroid is None else f"{float(centroid):.3f}m"
                parts.append(
                    f"{cam}:{reason},mask={mask_area},ratio={ratio_s},"
                    f"cov={cov:.2f},pts={pts},centroid={centroid_s}"
                )
            except Exception:
                continue
        return " | ".join(parts)

    def _format_fused_drop_reasons(self, track_ids: set[str]) -> str:
        """Human-readable lost-track reasons for FORCE REINIT warnings."""
        parts = []
        for tid in sorted(track_ids):
            info = self._fused_last_drop_reasons.get(tid, {})
            obj_id = str(info.get("object_id", "") or self._track_id_to_object_id.get(tid, ""))
            label = tid if not obj_id or obj_id in tid else f"{tid}/{obj_id}"
            lost_count = int(info.get("lost_count", self._fused_lost_count.get(tid, 0)))
            reason = str(info.get("reason", "unknown") or "unknown")
            mode = str(info.get("mode", "") or "")
            mode_suffix = f", mode={mode}" if mode else ""
            gate_reasons = str(info.get("gate_reasons", "") or "")
            gate_suffix = f", gates=[{gate_reasons}]" if gate_reasons else ""
            parts.append(
                f"{label}: max_lost_exceeded(lost_count={lost_count}, "
                f"last_reason={reason}{mode_suffix}{gate_suffix})"
            )
        return "[" + "; ".join(parts) + "]"

    def _get_or_create_fused_translation_kalman(self, track_id: str) -> PoseKalmanFilter:
        """Lazy-create the base-frame translation Kalman filter for a fused track."""
        kf = self._fused_translation_kalman.get(track_id)
        if kf is None:
            kf = PoseKalmanFilter()
            self._fused_translation_kalman[track_id] = kf
        return kf

    def _get_fused_kalman_predicted_position(self, track_id: str) -> Optional[np.ndarray]:
        """Current Kalman position prediction, if this track has an initialized filter."""
        kf = self._fused_translation_kalman.get(track_id)
        if kf is None or not kf.is_initialized:
            return None
        return kf.state[:3].astype(np.float32).copy()

    def _pose_origin_inside_tracker_masks(
        self,
        T_base_object: np.ndarray,
        mask_entries: list[dict],
    ) -> tuple[bool, str]:
        """Check whether the projected object-frame origin lands inside tracker masks."""
        if not bool(getattr(self.args, "track_require_pose_origin_in_mask", False)):
            return True, "disabled"
        margin_px = max(0, int(getattr(self.args, "track_pose_mask_margin_px", 8)))
        if not mask_entries:
            return False, "pose_origin_no_masks"

        p_base = np.asarray(T_base_object[:3, 3], dtype=np.float64).reshape(3)
        p_h = np.array([p_base[0], p_base[1], p_base[2], 1.0], dtype=np.float64)
        failures = []

        for entry in mask_entries:
            cam_id = entry["cam_id"]
            mask = np.asarray(entry["mask"], dtype=bool)
            K = np.asarray(entry["K"], dtype=np.float64).reshape(3, 3)
            h, w = mask.shape[:2]
            T_cam_base = self._resolve_T_cam_base(cam_id)
            p_cam = (T_cam_base @ p_h)[:3]
            z = float(p_cam[2])
            if z <= 1e-6:
                failures.append(f"{cam_id}:behind")
                continue
            u = int(round(float(K[0, 0] * p_cam[0] / z + K[0, 2])))
            v = int(round(float(K[1, 1] * p_cam[1] / z + K[1, 2])))
            if u < 0 or u >= w or v < 0 or v >= h:
                failures.append(f"{cam_id}:outside_image({u},{v})")
                continue

            if margin_px > 0:
                x0 = max(0, u - margin_px)
                x1 = min(w, u + margin_px + 1)
                y0 = max(0, v - margin_px)
                y1 = min(h, v + margin_px + 1)
                inside = bool(mask[y0:y1, x0:x1].any())
            else:
                inside = bool(mask[v, u])
            if not inside:
                failures.append(f"{cam_id}:outside_mask({u},{v})")

        if failures:
            return False, "pose_origin_outside_mask[" + ";".join(failures) + "]"
        return True, "pose_origin_inside_mask"

    def _evaluate_fused_camera_candidate(
        self,
        cr: dict,
        prev_base_pose: Optional[np.ndarray],
        is_warmup: bool = False,
        ) -> dict:
        """
        During warmup (first N frames after init), override rt_invalid
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
    
        # ── Build point cloud ──
        T_bc = self._resolve_T_base_cam(cam_id)
        pcd = lift_masked_depth_to_base(
            depth=view.depth,
            mask=mask,
            K=cr["K"],
            T_base_cam=T_bc,
            z_min=self.args.min_valid_z_m,
            z_max=self.args.max_valid_z_m,
            voxel_size=0.002,
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

        # ── Per-cam ICP gates
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


    def _filter_consistent_survivors(
        self,
        survivors: list[dict],
        anchor_pos: Optional[np.ndarray],
        max_disagreement_m: float,
        obj_id: str,
        track_id: str,
        ) -> list[dict]:
        """Drop survivor cameras whose cloud centroid disagrees with the others
        before fusing, guarding against one camera's tracker drifting onto a
        nearby object"""
        if max_disagreement_m <= 0.0 or len(survivors) < 2:
            return survivors

        cents: list[np.ndarray] = []
        valid: list[dict] = []
        for m in survivors:
            pcd = m.get("pcd")
            if pcd is None:
                continue
            pts = np.asarray(pcd.points)
            if pts.size == 0:
                continue
            cents.append(pts.mean(axis=0))
            valid.append(m)
        if len(valid) < 2:
            return survivors

        if len(valid) >= 3:
            consensus = np.median(np.stack(cents), axis=0)
            kept, dropped = [], []
            for m, c in zip(valid, cents):
                if float(np.linalg.norm(c - consensus)) <= max_disagreement_m:
                    kept.append(m)
                else:
                    dropped.append(m)
            if dropped and kept:
                self.get_logger().warn(
                    f"CONSISTENCY {obj_id}[{track_id}]: dropped "
                    f"{[m['cam_id'] for m in dropped]} as centroid outliers "
                    f"(> {max_disagreement_m*1000:.0f}mm from median)"
                )
                return kept
            return survivors

        
        d = float(np.linalg.norm(cents[0] - cents[1]))
        if d <= max_disagreement_m or anchor_pos is None:
            return survivors
        d0 = float(np.linalg.norm(cents[0] - anchor_pos))
        d1 = float(np.linalg.norm(cents[1] - anchor_pos))
        keep_idx = 0 if d0 <= d1 else 1
        self.get_logger().warn(
            f"CONSISTENCY {obj_id}[{track_id}]: 2-cam disagree by {d*1000:.0f}mm "
            f"(> {max_disagreement_m*1000:.0f}mm); keeping "
            f"{valid[keep_idx]['cam_id']} (closer to previous pose)"
        )
        return [valid[keep_idx]]

    def _build_centroid_recovery_seed(
        self,
        *,
        track_id: str,
        obj_id: str,
        gate_metrics: list[dict],
        normal_survivors: list[dict],
        prev_base_pose: Optional[np.ndarray],
    ) -> tuple[list[dict], Optional[np.ndarray], str]:
        """Build a translated ICP seed from agreeing centroid-far camera clouds."""
        if prev_base_pose is None:
            return [], None, ""

        candidates = []
        centroids = []
        for m in gate_metrics:
            reason = str(m.get("reason", ""))
            pcd = m.get("pcd")
            if not reason.startswith("centroid_far(") or pcd is None:
                continue
            if int(m.get("cloud_points", 0) or 0) < int(self.args.fused_gate_min_cloud_points):
                continue
            pts = np.asarray(pcd.points)
            if pts.size == 0:
                continue
            candidates.append(m)
            centroids.append(pts.mean(axis=0).astype(np.float64))

        if not candidates:
            return [], None, ""

        cluster_dist = float(getattr(self.args, "fused_track_centroid_recovery_cluster_dist_m", 0.12))
        min_cams = max(1, int(getattr(self.args, "fused_track_centroid_recovery_min_cameras", 1)))

        best_idxs: list[int] = []
        for i, c0 in enumerate(centroids):
            idxs = [
                j for j, c in enumerate(centroids)
                if float(np.linalg.norm(c - c0)) <= cluster_dist
            ]
            if len(idxs) > len(best_idxs):
                best_idxs = idxs

        if len(best_idxs) < min_cams:
            return [], None, ""

        # If a normal survivor cluster has equal-or-better support, stay on the
        # ordinary path. Recovery is meant for obvious jumps, not tie-breaking.
        if normal_survivors and len(best_idxs) <= len(normal_survivors):
            return [], None, ""

        recovery_survivors = [candidates[i] for i in best_idxs]
        cluster_centroid = np.mean(np.stack([centroids[i] for i in best_idxs]), axis=0)

        mem = self._fused_track_memory.get(track_id, {})
        prev_cloud_centroid = mem.get("cloud_centroid_base")
        if prev_cloud_centroid is None:
            prev_cloud_centroid = prev_base_pose[:3, 3]
        prev_cloud_centroid = np.asarray(prev_cloud_centroid, dtype=np.float64).reshape(3)

        delta = cluster_centroid - prev_cloud_centroid
        max_seed_jump = float(getattr(self.args, "fused_track_centroid_recovery_max_seed_jump_m", 0.75))
        jump_norm = float(np.linalg.norm(delta))
        if jump_norm > max_seed_jump:
            return [], None, ""

        T_seed = prev_base_pose.astype(np.float32).copy()
        T_seed[:3, 3] = (prev_base_pose[:3, 3].astype(np.float64) + delta).astype(np.float32)

        cam_ids = [str(m.get("cam_id", "?")) for m in recovery_survivors]
        reason = (
            f"centroid_recovery_seed(cams={cam_ids},"
            f"jump={jump_norm*1000:.1f}mm)"
        )
        if getattr(self.args, "debug_verbose_logs", False):
            self.get_logger().info(f"CENTROID RECOVERY {obj_id}[{track_id}]: {reason}")
        return recovery_survivors, T_seed, reason


    def _track_multicam_fused(self, views: list, stamp) -> None:
        """Fused multi-camera tracking: Cutie masks per camera, one ICP decision per track."""
        t_start = time.time()
    
        per_cam_results = {}
        cam_timing: dict = {}  # cam_id -> (n_cutie_calls, cutie_ms), for latency profiling
    
        # Collect work items
        camera_work = []
        for view in views:
            cam_id = view.cam_id
            states = self.track_states.get(cam_id, [])
            if not states:
                continue
            camera_work.append((view, states))
    
        def _run_one_camera(view, states):
            """Run multi-object Cutie + per-object ICP for one camera.

            One Cutie session per camera tracks ALL objects in a single
            forward (instead of one session per (camera, object)), then each
            object's mask is fed into its own RealtimeTracker for the existing
            ICP/fusion path. Thread-safe: each camera owns its own session and
            its own RealtimeTrackers (keyed by cam_id).
            """
            cam_id = view.cam_id
            rgb = view.rgb
            depth = view.depth
            K = np.asarray(view.K, dtype=np.float32).reshape(3, 3)
            results = {}
            skip_icp = bool(getattr(self.args, "skip_per_cam_icp_tracking", True))

            # One multi-object Cutie session per camera.
            session = self.cutie_sessions.get(cam_id)
            if session is None:
                session = CutieTracker(CutieConfig(
                    max_internal_size=int(getattr(self.args, "cutie_max_internal_size", 480)),
                ))
                self.cutie_sessions[cam_id] = session

            # ── Pass 1: make sure every state has a RealtimeTracker (pose/ICP
            # state) and is registered in the shared Cutie session. New objects
            # are memorized incrementally; existing ones keep their memory. ──
            active = []  # (state_idx, state, tracker_key)
            for idx, state in enumerate(states):
                # Per-instance key (track_id) so multi-instance same-class
                # scenes stay distinct in the Cutie session.
                tracker_key = f"{cam_id}_{state.track_id}"

                if tracker_key not in self.realtime_trackers:
                    try:
                        init_mask = state.recovery_mask
                        if init_mask is None or init_mask.sum() < 100:
                            continue
                        init_mask = np.asarray(init_mask).astype(bool)
                        if init_mask.ndim != 2:
                            continue

                        cfg = RealtimeTrackerConfig(
                            cutie_cfg=CutieConfig(
                                max_internal_size=int(getattr(self.args, "cutie_max_internal_size", 480)),
                            ),
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
                        )
                        rt = RealtimeTracker(cfg)
                        # Register the object in the shared per-camera session;
                        # the RealtimeTracker's own Cutie is not used here.
                        session.add_object(state.track_id, rgb, init_mask)
                        rt.initialize(
                            rgb=rgb, depth=depth, mask=init_mask,
                            T_init=state.T_object_camera, K=K,
                            mesh_path=state.mesh_path,
                            init_cutie=False,
                        )
                        self.realtime_trackers[tracker_key] = rt
                        state.last_good_mask = init_mask.copy()
                        state.last_good_T = state.T_object_camera.copy()
                    except Exception:
                        continue
                elif not session.has_object(state.track_id):
                    # Tracker exists but the session lost this object (e.g.
                    # after a partial drop). Re-register from the last good mask.
                    seed = state.last_good_mask
                    if seed is None:
                        seed = state.recovery_mask
                    try:
                        if seed is not None:
                            seed = np.asarray(seed).astype(bool)
                            if seed.ndim == 2 and seed.sum() >= 100:
                                session.add_object(state.track_id, rgb, seed)
                    except Exception:
                        pass

                active.append((idx, state, tracker_key))

            # Drop session objects whose track is no longer present on this cam,
            # so Cutie stops segmenting stale objects.
            current_track_ids = {s.track_id for s in states}
            for tid in list(session.tracked_object_ids):
                if tid not in current_track_ids:
                    session.remove_object(tid)

            # ── ONE Cutie forward for all of this camera's objects. ──
            _t_cutie = time.time()
            try:
                masks_by_track = session.track_multi(rgb)
            except Exception:
                masks_by_track = {}
            _cutie_ms = (time.time() - _t_cutie) * 1000.0
            _cutie_n = 1 if session.tracked_object_ids else 0

            # ── Pass 2: feed each object's mask into its RealtimeTracker. ──
            for idx, state, tracker_key in active:
                rt = self.realtime_trackers.get(tracker_key)
                if rt is None:
                    continue

                cutie_result = masks_by_track.get(state.track_id)
                if cutie_result is None:
                    state.lost_count += 1
                    state.mode = "degraded"
                    continue

                try:
                    result = rt.track_with_mask(
                        cutie_result, depth, K=K, skip_icp=skip_icp, rgb=rgb,
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

            cam_timing[cam_id] = (_cutie_n, _cutie_ms)
            return results
    
        # Run camera trackers in parallel when there is more than one active stream.
        if len(camera_work) >= 2:
            # One worker per camera so a 3rd cam overlaps instead of serializing
            # after the first two (which would ~double tracking-loop latency).
            with ThreadPoolExecutor(max_workers=len(camera_work)) as pool:
                futures = [pool.submit(_run_one_camera, v, s) for v, s in camera_work]
                for f in futures:
                    per_cam_results.update(f.result())
        else:
            for view, states in camera_work:
                per_cam_results.update(_run_one_camera(view, states))
    
        # All currently-tracked track_ids across every cam. We process them
        # via the pose_status state machine even if no camera produced a
        # usable mask this tick
        all_tracked_track_ids: set[str] = set()
        track_id_to_object_id_local: dict[str, str] = {}
        for states in self.track_states.values():
            for s in states:
                all_tracked_track_ids.add(s.track_id)
                track_id_to_object_id_local[s.track_id] = s.object_id

        if not per_cam_results:
            self.get_logger().warn("Fused tracking: no camera produced a usable mask")

        # Group by track_id (per-instance), not object_id (class). Without
        # this, N same-class instances collapse into one fused decision.
        by_track: dict[str, list[dict]] = {}
        for cr in per_cam_results.values():
            tid = cr["state"].track_id
            by_track.setdefault(tid, []).append(cr)

        object_decisions: dict[str, dict] = {}
        t_fuse = time.time()
        fused_kalman_disabled = bool(getattr(self.args, "disable_fused_kalman", False))
        axis_jump_gate_disabled = bool(getattr(self.args, "disable_axis_jump_gate", False))

        for track_id, entries in by_track.items():
            obj_id = entries[0]["state"].object_id
            prev_base_pose = self._get_previous_object_base_pose(track_id, entries)
            kf_obj = None if fused_kalman_disabled else self._get_or_create_fused_translation_kalman(track_id)
            if kf_obj is not None and kf_obj.is_initialized:
                kf_obj.predict()

            # ── Warmup detection ──
            warmup_count = self._fused_warmup_count.get(track_id, 0)
            warmup_frames = int(getattr(self.args, 'fused_track_warmup_frames', 5))
            is_warmup = warmup_count < warmup_frames

            gate_metrics = [
                self._evaluate_fused_camera_candidate(cr, prev_base_pose, is_warmup=is_warmup)
                for cr in entries
            ]
            survivors = [m for m in gate_metrics if m["accepted"]]
            centroid_recovery_seed: Optional[np.ndarray] = None
            centroid_recovery_reason = ""

            # Cross-camera consistency: drop a camera that drifted onto a nearby
            # object before it can corrupt the fused cloud. Skipped during warmup
            # (init->track shift is expected) and a no-op unless enabled.
            if not is_warmup:
                survivors = self._filter_consistent_survivors(
                    survivors,
                    None if prev_base_pose is None else prev_base_pose[:3, 3],
                    float(getattr(self.args, "fused_consistency_max_disagreement_m", 0.0)),
                    obj_id, track_id,
                )
                if bool(getattr(self.args, "fused_track_centroid_recovery", False)):
                    recovery_survivors, recovery_seed, recovery_reason = (
                        self._build_centroid_recovery_seed(
                            track_id=track_id,
                            obj_id=obj_id,
                            gate_metrics=gate_metrics,
                            normal_survivors=survivors,
                            prev_base_pose=prev_base_pose,
                        )
                    )
                    if recovery_seed is not None and recovery_survivors:
                        survivors = recovery_survivors
                        centroid_recovery_seed = recovery_seed
                        centroid_recovery_reason = recovery_reason
            survived_cam_ids = [m["cam_id"] for m in survivors]
            survived_keys = [(m["cam_id"], m["state_idx"]) for m in survivors]

            decision = {
                "object_id": obj_id,
                "track_id": track_id,
                "mode": "hold_previous",
                "pose_mode": "hold_previous",
                "accepted": False,
                "reason": "no_survivors",
                "T_base": None,
                "fitness": float("nan"),
                "rmse_m": float("nan"),
                "trans_jump_m": 0.0,
                "rot_jump_deg": 0.0,
                "survived_cam_ids": survived_cam_ids,
                "survived_keys": survived_keys,
                "gate_metrics": gate_metrics,
                "used_tracker_key": None,
                "used_bbox_xyxy": None,
                "is_warmup": is_warmup,
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
                    f"FUSED GATE {obj_id}[{track_id}]{warmup_tag}: survivors={survived_cam_ids or 'none'} | {gate_log}"
                )

            if not survivors:
                object_decisions[track_id] = decision
                self._fused_icp_metrics[track_id] = {
                    "fitness": float("nan"), "rmse_mm": float("nan"),
                    "mode": "hold_previous", "pose_mode": "hold_previous",
                    "accepted": False,
                    "reason": "no_survivors",
                }
                # Still increment warmup counter even on failure
                self._fused_warmup_count[track_id] = warmup_count + 1
                continue
    
            # ── Single-camera path ──
            unique_survivor_cams = {m["cam_id"] for m in survivors}
            if len(unique_survivor_cams) == 1:
                m = survivors[0]
                cr = next(cr for cr in entries if cr["cam_id"] == m["cam_id"] and cr["state_idx"] == m["state_idx"])
                T_bc = self._resolve_T_base_cam(m["cam_id"])

                if cr.get("rt_result") is not None and hasattr(cr["rt_result"], "T_object_camera"):
                    T_base_seed = (T_bc @ cr["rt_result"].T_object_camera.astype(np.float64)).astype(np.float32)
                else:
                    # During warmup with rt_invalid, use last good pose
                    T_base_seed = (T_bc @ cr["state"].T_object_camera.astype(np.float64)).astype(np.float32)

                # Single-camera base-frame ICP
                T_base_candidate = T_base_seed
                single_cam_icp_fitness = float("nan")
                single_cam_icp_rmse_m = float("nan")
                single_cam_icp_ran = False
                single_cam_icp_failed = False
                pcd_ok = (
                    m.get("pcd") is not None
                    and len(m["pcd"].points) >= int(self.args.fused_gate_min_cloud_points)
                )
                if pcd_ok:
                    state0_sc = cr["state"]
                    track_num_points_sc = int(getattr(self.args, "track_icp_num_points", 2000))
                    model_pcd_sc = mesh_to_pcd_cached(
                        state0_sc.mesh_path, float(self.args.mesh_scale),
                        num_points=track_num_points_sc,
                    )
                    T_init_sc = (
                        centroid_recovery_seed.astype(np.float32)
                        if centroid_recovery_seed is not None
                        else prev_base_pose.astype(np.float32)
                        if prev_base_pose is not None
                        else T_base_seed
                    )
                    try:
                        T_refined, fit_sc, rmse_sc = run_icp_in_base_frame(
                            scene_pcd=m["pcd"],
                            model_pcd=model_pcd_sc,
                            T_base_object_init=T_init_sc,
                            max_correspondence_dist=float(self.args.fused_track_icp_max_correspondence_dist_m),
                            max_iteration=int(getattr(self.args, 'fused_track_icp_max_iteration', 30)),
                            variant=self.args.icp_variant,
                            relative_fitness=float(getattr(self.args, 'fused_track_icp_relative_fitness', 1e-4)),
                            relative_rmse=float(getattr(self.args, 'fused_track_icp_relative_rmse', 1e-4)),
                        )
                        T_base_candidate = T_refined.astype(np.float32)
                        single_cam_icp_fitness = float(fit_sc)
                        single_cam_icp_rmse_m = float(rmse_sc)
                        single_cam_icp_ran = True
                    except Exception as e:
                        # ICP exception against a real cloud
                        single_cam_icp_failed = True
                        self.get_logger().warn(
                            f"single-cam ICP failed for {obj_id}[{track_id}]: {e} — holding previous"
                        )

                dt, drot = self._pose_delta_from_base(prev_base_pose, T_base_candidate)
                motion_dt_s, motion_dt_raw_s = self._compute_fused_motion_dt(track_id, stamp)
                max_trans_jump_m, max_rot_jump_deg = self._compute_fused_motion_thresholds(motion_dt_s)

                kalman_pred_pos = (
                    None if fused_kalman_disabled
                    else self._get_fused_kalman_predicted_position(track_id)
                )
                kalman_residual_m = (
                    float(np.linalg.norm(T_base_candidate[:3, 3] - kalman_pred_pos))
                    if kalman_pred_pos is not None else None
                )
    
                # Prefer single-cam base-frame ICP metrics when we ran it.
                # Otherwise fall back to the per-cam ICP fitness/rmse 
                if single_cam_icp_ran:
                    fitness_val = single_cam_icp_fitness
                    rmse_val = single_cam_icp_rmse_m
                    single_cam_pose_mode = "single_cam_icp"
                else:
                    per_cam_icp_ran = not bool(getattr(self.args, "skip_per_cam_icp_tracking", True))
                    fitness_val = float(m["per_cam_icp_fitness"]) if per_cam_icp_ran else float("nan")
                    rmse_val = float(m["per_cam_icp_rmse_m"]) if per_cam_icp_ran else float("nan")
                    single_cam_pose_mode = "single_cam_icp" if per_cam_icp_ran else "mask_only"

                # Fitness used by kalman_soft_reject AND the quality gate below.
                gate_fitness = single_cam_icp_fitness if single_cam_icp_ran else float(m["per_cam_icp_fitness"])

                quality_reject_reason: Optional[str] = None
                pose_mask_ok, pose_mask_reason = self._pose_origin_inside_tracker_masks(
                    T_base_candidate,
                    [{
                        "cam_id": m["cam_id"],
                        "mask": m["mask"],
                        "K": cr["K"],
                    }],
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
                    fitness_ok_sc = True
                    rmse_ok_sc = True
                else:
                    motion_ok = dt <= max_trans_jump_m and drot <= max_rot_jump_deg
                    kalman_soft_reject = (
                        not fused_kalman_disabled
                        and kalman_residual_m is not None
                        and kalman_residual_m > float(self.args.fused_track_kalman_soft_translation_residual_m)
                        and gate_fitness < float(self.args.fused_track_kalman_soft_max_icp_fitness)
                    )
                    if single_cam_icp_ran:
                        min_fit = float(self.args.fused_track_min_fused_icp_fitness)
                        max_rmse = float(self.args.fused_track_max_fused_icp_rmse_m)
                        fitness_ok_sc = single_cam_icp_fitness >= min_fit
                        rmse_ok_sc = single_cam_icp_rmse_m <= max_rmse
                        if not fitness_ok_sc:
                            quality_reject_reason = f"single_cam_fitness_low({single_cam_icp_fitness:.3f})"
                        elif not rmse_ok_sc:
                            quality_reject_reason = f"single_cam_rmse_high({single_cam_icp_rmse_m*1000:.1f}mm)"
                    else:
                        # No base-frame ICP available
                        fitness_ok_sc = True
                        rmse_ok_sc = True

                if (
                    motion_ok and not kalman_soft_reject and fitness_ok_sc and rmse_ok_sc
                    and pose_mask_ok and not single_cam_icp_failed
                ):
                    T_base_publish = T_base_candidate
                    buf_size = int(getattr(self.args, 'median_pose_buffer_size', 3))
                    if buf_size > 0:
                        buf = self._median_pose_buffers.setdefault(track_id, MedianPoseBuffer(buf_size))
                        buf.push(T_base_candidate)
                        if buf.is_ready():
                            T_base_publish = buf.get_median()

                    mode_str = (
                        "single_cam_centroid_recovery"
                        if centroid_recovery_seed is not None
                        else "single_cam_fallback"
                    )
                    decision.update({
                        "mode": mode_str,
                        "pose_mode": single_cam_pose_mode,
                        "accepted": True,
                        "reason": (
                            centroid_recovery_reason or "centroid_recovery_ok"
                            if centroid_recovery_seed is not None
                            else "single_cam_ok"
                        ),
                        "T_base": T_base_publish,
                        "fitness": fitness_val,
                        "rmse_m": rmse_val,
                        "trans_jump_m": dt,
                        "rot_jump_deg": drot,
                    })
                    memory_update = {
                        "T_base": T_base_publish.copy(),
                        "mode": mode_str,
                        "stamp_s": self._stamp_to_seconds(stamp),
                    }
                    if bool(getattr(self.args, "fused_track_centroid_recovery", False)):
                        pcd = m.get("pcd")
                        if pcd is not None and len(pcd.points) > 0:
                            memory_update["cloud_centroid_base"] = (
                                np.asarray(pcd.points).mean(axis=0).astype(np.float32)
                            )
                    self._fused_track_memory[track_id] = memory_update
                    self._last_known_track_centroids[track_id] = (
                        T_base_publish[:3, 3].astype(np.float64).copy()
                    )
                else:
                    if single_cam_icp_failed:
                        reject_reason = "single_cam_icp_exception"
                    elif quality_reject_reason is not None:
                        reject_reason = quality_reject_reason
                    elif not pose_mask_ok:
                        reject_reason = pose_mask_reason
                    elif kalman_soft_reject:
                        reject_reason = f"single_cam_kalman_soft(res={kalman_residual_m*1000:.1f}mm)"
                    else:
                        reject_reason = f"single_cam_motion_reject(dt={dt:.3f},rot={drot:.1f})"
                    decision.update({
                        "mode": "hold_previous",
                        "pose_mode": "hold_previous",
                        "accepted": False,
                        "reason": reject_reason,
                        "fitness": fitness_val,
                        "rmse_m": rmse_val,
                    })

                self._fused_icp_metrics[track_id] = {
                    "fitness": fitness_val,
                    "rmse_mm": rmse_val * 1000.0 if np.isfinite(rmse_val) else float("nan"),
                    "mode": decision["mode"],
                    "pose_mode": decision["pose_mode"],
                    "accepted": decision["accepted"],
                    "reason": decision["reason"],
                }
                object_decisions[track_id] = decision
                self._fused_warmup_count[track_id] = warmup_count + 1
                continue

            # ── Multi-camera fusion path ──
            state0 = entries[0]["state"]
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
                fused_cloud = merge_point_clouds_weighted(
                    survivor_pcds, cam_positions,
                    voxel_size=0.002,
                    distance_exponent=float(getattr(self.args, 'cloud_merge_distance_exponent', 2.0)),
                )
            else:
                fused_cloud = merge_point_clouds(survivor_pcds, voxel_size=0.0)

            if fused_cloud is None or len(fused_cloud.points) < int(self.args.fused_gate_min_cloud_points):
                decision["reason"] = f"fused_cloud_too_small({0 if fused_cloud is None else len(fused_cloud.points)})"
                decision["pose_mode"] = "hold_previous"
                self._fused_icp_metrics[track_id] = {
                    "fitness": float("nan"), "rmse_mm": float("nan"),
                    "mode": "hold_previous", "pose_mode": "hold_previous",
                    "accepted": False,
                    "reason": decision["reason"],
                }
                object_decisions[track_id] = decision
                self._fused_warmup_count[track_id] = warmup_count + 1
                continue

            if centroid_recovery_seed is not None:
                T_base_init = centroid_recovery_seed.astype(np.float32)
            elif prev_base_pose is not None:
                T_base_init = prev_base_pose.astype(np.float32)
            else:
                first_survivor = survivors[0]
                first_cr = next(cr for cr in entries if cr["cam_id"] == first_survivor["cam_id"])
                T_bc_0 = self._resolve_T_base_cam(first_survivor["cam_id"])
                T_base_init = (T_bc_0 @ first_cr["state"].T_object_camera.astype(np.float64)).astype(np.float32)
    
            # Refine the merged cloud from the previous pose or centroid recovery seed.
            icp_iters = int(getattr(self.args, 'fused_track_icp_max_iteration', 30))
            T_base_fused, fitness, rmse = run_icp_in_base_frame(
                scene_pcd=fused_cloud,
                model_pcd=model_pcd,
                T_base_object_init=T_base_init,
                max_correspondence_dist=float(self.args.fused_track_icp_max_correspondence_dist_m),
                max_iteration=icp_iters,
                variant=self.args.icp_variant,
                relative_fitness=float(getattr(self.args, 'fused_track_icp_relative_fitness', 1e-4)),
                relative_rmse=float(getattr(self.args, 'fused_track_icp_relative_rmse', 1e-4)),
            )

            # ── Optional rotation re-seed (opt-in: --fused-track-rot-reseed) ──
            if getattr(self.args, "fused_track_rot_reseed", False) and not is_warmup:
                reseed_lo = float(getattr(self.args, "fused_track_rot_reseed_chamfer_m", 0.010))
                reseed_hi = float(getattr(self.args, "fused_track_rot_reseed_max_chamfer_m", 0.080))
                ch_now = self._grid_chamfer(model_pcd, fused_cloud, survivor_pcds, T_base_fused)
                if reseed_lo <= ch_now <= reseed_hi:
                    T_reseed, ch_reseed = self._icp_rotation_grid(
                        t_base=T_base_fused[:3, 3],
                        model_pcd=model_pcd,
                        fused_cloud=fused_cloud,
                        per_cam_clouds=survivor_pcds,
                        n_rot=int(getattr(self.args, "fused_track_rot_reseed_n_rot", 24)),
                        icp_max_iter=int(getattr(self.args, "fused_track_rot_reseed_icp_iters", 10)),
                    )
                    if T_reseed is not None and ch_reseed < ch_now:
                        T_base_fused = T_reseed.astype(np.float32)
                        fitness, rmse = evaluate_icp_in_base_frame(
                            fused_cloud, model_pcd, T_base_fused,
                            float(self.args.fused_track_icp_max_correspondence_dist_m),
                        )
                        if getattr(self.args, "debug_verbose_logs", False):
                            self.get_logger().info(
                                f"ROT-RESEED {obj_id}[{track_id}]: chamfer "
                                f"{ch_now*1000:.1f}->{ch_reseed*1000:.1f}mm fitness={fitness:.3f}"
                            )

            # ── Optional PCA shaft-axis correction (opt-in: --fused-track-pca-axis) ──
            if (getattr(self.args, "fused_track_pca_axis", False)
                    and not is_warmup and prev_base_pose is not None):
                min_pts = int(getattr(self.args, "fused_track_pca_axis_min_points", 50))
                scene_pts = np.asarray(fused_cloud.points)
                model_axis_obj = self._object_shaft_axis(state0.mesh_path, model_pcd)
                if model_axis_obj is not None and len(scene_pts) >= min_pts:
                    scene_axis, elong = self._principal_axis(scene_pts)
                    min_elong = float(getattr(self.args, "fused_track_pca_axis_min_elongation", 3.0))
                    if elong >= min_elong:
                        R = T_base_fused[:3, :3].astype(np.float64)
                        icp_axis = R @ model_axis_obj
                        icp_axis = icp_axis / (np.linalg.norm(icp_axis) + 1e-12)
                        # Sign: keep continuity with the current (ICP) direction.
                        if float(np.dot(scene_axis, icp_axis)) < 0.0:
                            scene_axis = -scene_axis
                        cosang = float(np.clip(np.dot(icp_axis, scene_axis), -1.0, 1.0))
                        ang_deg = float(np.degrees(np.arccos(cosang)))
                        min_deg = float(getattr(self.args, "fused_track_pca_axis_min_deg", 10.0))
                        max_deg = float(getattr(self.args, "fused_track_pca_axis_max_deg", 60.0))
                        if min_deg <= ang_deg <= max_deg:
                            blend = float(getattr(self.args, "fused_track_pca_axis_blend", 1.0))
                            R_corr = self._rotation_between(icp_axis, scene_axis, blend)
                            T_base_fused = T_base_fused.copy()
                            T_base_fused[:3, :3] = (R_corr @ R).astype(np.float32)
                            fitness, rmse = evaluate_icp_in_base_frame(
                                fused_cloud, model_pcd, T_base_fused,
                                float(self.args.fused_track_icp_max_correspondence_dist_m),
                            )
                            if getattr(self.args, "debug_verbose_logs", False):
                                self.get_logger().info(
                                    f"PCA-AXIS {obj_id}[{track_id}]: shaft off "
                                f"{ang_deg:.1f}deg -> corrected (elong={elong:.1f})"
                            )

            # Optional rotation damping: clamp or smooth orientation changes without touching translation.
            slew_limit_deg = float(getattr(self.args, "fused_track_rot_slew_limit_deg", 0.0))
            rot_lowpass = float(getattr(self.args, "fused_track_rot_lowpass", 0.0))
            if ((slew_limit_deg > 0.0 or rot_lowpass > 0.0)
                    and not is_warmup and prev_base_pose is not None):
                from scipy.spatial.transform import Rotation as SciRot
                R_prev = prev_base_pose[:3, :3].astype(np.float64)
                R_new = T_base_fused[:3, :3].astype(np.float64)
                # rotvec from prev->new; its norm is the geodesic angle (rad).
                rotvec = SciRot.from_matrix(R_prev.T @ R_new).as_rotvec()
                ang_deg = float(np.degrees(np.linalg.norm(rotvec)))
                if ang_deg > 1e-6:
                    # Fraction of the prev->new turn to keep (1.0 = full update).
                    frac = 1.0
                    if slew_limit_deg > 0.0 and ang_deg > slew_limit_deg:
                        frac = slew_limit_deg / ang_deg
                    if rot_lowpass > 0.0:
                        # Blend the slew-limited rotation toward prev by lowpass;
                        # along the same geodesic this scales the kept fraction.
                        frac *= (1.0 - float(np.clip(rot_lowpass, 0.0, 1.0)))
                    if frac < 1.0:
                        R_damped = R_prev @ SciRot.from_rotvec(frac * rotvec).as_matrix()
                        T_base_fused = T_base_fused.copy()
                        T_base_fused[:3, :3] = R_damped.astype(np.float32)
                        if getattr(self.args, "debug_verbose_logs", False):
                            self.get_logger().info(
                                f"ROT-DAMP {obj_id}[{track_id}]: turn {ang_deg:.1f}deg "
                                f"-> {ang_deg * frac:.1f}deg (frac={frac:.2f})"
                            )

            pose_mask_entries = []
            for m in survivors:
                cr_match = next(
                    cr for cr in entries
                    if cr["cam_id"] == m["cam_id"] and cr["state_idx"] == m["state_idx"]
                )
                pose_mask_entries.append({
                    "cam_id": m["cam_id"],
                    "mask": m["mask"],
                    "K": cr_match["K"],
                })
            pose_mask_ok, pose_mask_reason = self._pose_origin_inside_tracker_masks(
                T_base_fused,
                pose_mask_entries,
            )
    
            dt, drot = self._pose_delta_from_base(prev_base_pose, T_base_fused)
            motion_dt_s, motion_dt_raw_s = self._compute_fused_motion_dt(track_id, stamp)
            max_trans_jump_m, max_rot_jump_deg = self._compute_fused_motion_thresholds(motion_dt_s)

            kalman_pred_pos = (
                None if fused_kalman_disabled
                else self._get_fused_kalman_predicted_position(track_id)
            )
            kalman_residual_m = (
                float(np.linalg.norm(T_base_fused[:3, 3] - kalman_pred_pos))
                if kalman_pred_pos is not None else None
            )
    
            # Chamfer is expensive, so skip it when ICP quality and motion are already clean.
            chamfer_skip_fitness_min = float(getattr(self.args, "chamfer_skip_fitness_min", 0.30))
            chamfer_skip_rmse_max = float(getattr(self.args, "chamfer_skip_rmse_max_m", 0.005))
            chamfer_skip_motion_max = float(getattr(self.args, "chamfer_skip_motion_max_m", 0.010))
            quality_clean = (
                fitness >= chamfer_skip_fitness_min
                and rmse <= chamfer_skip_rmse_max
                and (prev_base_pose is None or dt <= chamfer_skip_motion_max)
            )
            chamfer_every_n = max(1, int(getattr(self.args, "chamfer_every_n_frames", 1)))
            chamfer_due = chamfer_every_n <= 1 or (self.frame_counter % chamfer_every_n == 0)
            need_chamfer = (not is_warmup) and (not quality_clean) and chamfer_due
            chamfer: Optional[float] = (
                chamfer_distance_one_way(model_pcd, fused_cloud, T_base_fused)
                if need_chamfer else None
            )

            accept = True
            reject_reasons = []

            if is_warmup:
                # ── WARMUP: skip ALL tight checks. Only reject truly insane results. ──
                WARMUP_MAX_TRANS_M = 0.20      # 200mm 
                WARMUP_MAX_ROT_DEG = 90.0      # quarter turn 
                WARMUP_MIN_FITNESS = 0.03 
    
                if fitness < WARMUP_MIN_FITNESS:
                    accept = False
                    reject_reasons.append(f"warmup_fitness_garbage({fitness:.3f})")
                if prev_base_pose is not None and dt > WARMUP_MAX_TRANS_M:
                    accept = False
                    reject_reasons.append(f"warmup_teleport({dt*1000:.0f}mm)")
                if prev_base_pose is not None and drot > WARMUP_MAX_ROT_DEG:
                    accept = False
                    reject_reasons.append(f"warmup_spin({drot:.0f}deg)")
                if not pose_mask_ok:
                    accept = False
                    reject_reasons.append(pose_mask_reason)
            else:
                # Track which gates failed and why, so jump-only rejects can be rescued.
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

                # Chamfer catches visually wrong fits that still pass ICP fitness/RMSE.
                max_chamfer = float(getattr(self.args, 'fused_track_max_chamfer_m', 0.015))
                chamfer_ok = (chamfer is None) or (chamfer <= max_chamfer)
                if not chamfer_ok:
                    accept = False
                    reject_reasons.append(f"chamfer({chamfer*1000:.1f}mm>{max_chamfer*1000:.0f}mm)")

                # Axis-dominant weak-ICP jumps are often depth artifacts, not real motion.
                axis_jump = np.abs(T_base_fused[:3, 3] - prev_base_pose[:3, 3]) if prev_base_pose is not None else np.zeros(3)
                dominant_frac = float(axis_jump.max() / (np.linalg.norm(axis_jump) + 1e-9)) if np.linalg.norm(axis_jump) > 1e-9 else 0.0
                weak_icp = fitness < float(self.args.fused_track_weak_icp_fitness)
                axis_dominant_ok = True
                if (not axis_jump_gate_disabled
                    and prev_base_pose is not None
                    and dominant_frac > float(self.args.fused_track_axis_dominant_fraction)
                    and dt > float(self.args.fused_track_axis_dominant_min_translation_m)
                    and weak_icp):
                    accept = False
                    axis_dominant_ok = False
                    reject_reasons.append("axis_dominant_jump")

                # Kalman soft reject only fires when both prediction residual and ICP quality look bad.
                kalman_soft_reject = (
                    not fused_kalman_disabled
                    and kalman_residual_m is not None
                    and kalman_residual_m > float(self.args.fused_track_kalman_soft_translation_residual_m)
                    and fitness < float(self.args.fused_track_kalman_soft_max_icp_fitness)
                )
                if kalman_soft_reject:
                    accept = False
                    reject_reasons.append(f"kalman_soft(res={kalman_residual_m*1000:.1f}mm)")
                if not pose_mask_ok:
                    accept = False
                    reject_reasons.append(pose_mask_reason)

            
                if not accept:
                    # If only the motion gate failed, a strong chamfer can rescue fast real motion.
                    quality_ok = (
                        fitness_ok and rmse_ok and chamfer_ok and axis_dominant_ok
                        and not kalman_soft_reject and pose_mask_ok
                    )
                    jump_rejected = not jump_t_ok or not jump_r_ok

                    if quality_ok and jump_rejected and chamfer_due:
                        # The pose FITS the cloud well (good fitness, rmse, chamfer)
                        # but moved too fast.
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
                                f"  ⚠ JUMP RESCUED {obj_id}[{track_id}]: large motion "
                                f"dt={dt*1000:.1f}mm drot={drot:.1f}deg "
                                f"but chamfer={chamfer*1000:.1f}mm is good — accepting with flag"
                            )

            if accept:
                # ── Median pose buffer ──
                T_base_publish = T_base_fused
                buf_size = int(getattr(self.args, 'median_pose_buffer_size', 3))
                if buf_size > 0:
                    buf = self._median_pose_buffers.setdefault(track_id, MedianPoseBuffer(buf_size))
                    buf.push(T_base_fused)
                    if buf.is_ready():
                        T_base_publish = buf.get_median()

                # Determine mode — flag rescued jumps distinctly
                was_rescued = any("RESCUED" in r for r in reject_reasons)
                unique_cams_used = len({m["cam_id"] for m in survivors})
                if centroid_recovery_seed is not None:
                    mode_str = "fusion_centroid_recovery"
                elif was_rescued:
                    mode_str = "fusion_rescued"
                elif unique_cams_used >= 2:
                    mode_str = "fusion_2cam"
                else:
                    mode_str = "fusion"

                decision.update({
                    "mode": mode_str,
                    "pose_mode": "fused_icp",
                    "accepted": True,
                    "reason": (
                        centroid_recovery_reason or "centroid_recovery_ok"
                        if centroid_recovery_seed is not None and not reject_reasons
                        else reject_reasons[0] if reject_reasons else "ok"
                    ),
                    "T_base": T_base_publish,
                    "fitness": float(fitness),
                    "rmse_m": float(rmse),
                    "trans_jump_m": dt,
                    "rot_jump_deg": drot,
                    "chamfer_m": float(chamfer) if chamfer is not None else -1.0,
                })
                memory_update = {
                    "T_base": T_base_publish.copy(),
                    "mode": decision["mode"],
                    "stamp_s": self._stamp_to_seconds(stamp),
                }
                if bool(getattr(self.args, "fused_track_centroid_recovery", False)):
                    pts_mem = np.asarray(fused_cloud.points)
                    if pts_mem.size > 0:
                        memory_update["cloud_centroid_base"] = (
                            pts_mem.mean(axis=0).astype(np.float32)
                        )
                self._fused_track_memory[track_id] = memory_update
                self._last_known_track_centroids[track_id] = (
                    T_base_publish[:3, 3].astype(np.float64).copy()
                )
            else:
                decision.update({
                    "mode": "hold_previous",
                    "pose_mode": "hold_previous",
                    "accepted": False,
                    "reason": ",".join(reject_reasons),
                    "fitness": float(fitness),
                    "rmse_m": float(rmse),
                    "trans_jump_m": dt,
                    "rot_jump_deg": drot,
                    "chamfer_m": float(chamfer) if chamfer is not None else -1.0,
                })
    
            self._fused_icp_metrics[track_id] = {
                "fitness": float(fitness),
                "rmse_mm": float(rmse) * 1000.0,
                "mode": decision["mode"],
                "pose_mode": decision["pose_mode"],
                "accepted": decision["accepted"],
                "reason": decision["reason"],
            }
    
            if getattr(self.args, "debug_verbose_logs", False):
                chamfer_str = (
                    f"{chamfer*1000:.1f}mm" if chamfer is not None else "skipped"
                )
                self.get_logger().info(
                    f"FUSED DECISION {obj_id}[{track_id}]{warmup_tag}: mode={decision['mode']} accepted={decision['accepted']} "
                    f"cams={survived_cam_ids} fitness={fitness:.3f} rmse={rmse*1000:.1f}mm "
                    f"chamfer={chamfer_str} dt={dt*1000:.1f}mm drot={drot:.1f}deg "
                    f"reason={decision['reason']}"
                )
            object_decisions[track_id] = decision
            self._fused_warmup_count[track_id] = warmup_count + 1

        t_icp_end = time.time()
        if getattr(self.args, "debug_verbose_logs", False):
            _dprint(f"[TIMING] Fused ICP all objects: {(t_icp_end-t_fuse)*1000:.0f}ms")

        # ── Kalman update ──
        if not fused_kalman_disabled:
            for tid, decision in object_decisions.items():
                T_base_acc = decision.get("T_base")
                if not (bool(decision.get("accepted", False)) and T_base_acc is not None):
                    continue
                kf = self._get_or_create_fused_translation_kalman(tid)
                pos = np.asarray(T_base_acc[:3, 3], dtype=np.float64).reshape(3)
                if not kf.is_initialized:
                    kf.initialize(pos)
                else:
                    kf.update(pos)


        warmup_frames_default = int(getattr(self.args, 'fused_track_warmup_frames', 5))
        for tid in all_tracked_track_ids:
            if tid in object_decisions:
                continue
            warmup_count_missing = self._fused_warmup_count.get(tid, 0)
            object_decisions[tid] = {
                "object_id": track_id_to_object_id_local.get(tid, self._track_id_to_object_id.get(tid, "")),
                "track_id": tid,
                "mode": "hold_previous",
                "pose_mode": "hold_previous",
                "accepted": False,
                "reason": "no_per_cam_results",
                "T_base": None,
                "fitness": float("nan"),
                "rmse_m": float("nan"),
                "trans_jump_m": 0.0,
                "rot_jump_deg": 0.0,
                "survived_cam_ids": [],
                "survived_keys": [],
                "gate_metrics": [],
                "used_tracker_key": None,
                "used_bbox_xyxy": None,
                "is_warmup": warmup_count_missing < warmup_frames_default,
            }
            self._fused_icp_metrics[tid] = {
                "fitness": float("nan"), "rmse_mm": float("nan"),
                "mode": "hold_previous", "pose_mode": "hold_previous",
                "accepted": False,
                "reason": "no_per_cam_results",
            }
            self._fused_warmup_count[tid] = warmup_count_missing + 1

        # ── Per-track hold/lost state machine ──
        hold_window = int(getattr(self.args, "fused_track_hold_window_frames", 5))
        max_lost = int(getattr(self.args, "fused_track_max_lost_frames", 20))
        for tid, decision in object_decisions.items():
            accepted = bool(decision.get("accepted", False)) and decision.get("T_base") is not None
            if accepted:
                self._fused_lost_count[tid] = 0
                status = "fresh"
            else:
                prev = self._fused_lost_count.get(tid, 0)
                lost = prev + 1
                self._fused_lost_count[tid] = lost
                if lost <= hold_window:
                    status = "held"
                elif lost <= max_lost:
                    status = "stale"
                else:
                    status = "lost"
                    self._force_reinit_tracks.add(tid)
            self._fused_pose_status[tid] = status
            decision["pose_status"] = status
            decision["lost_count"] = int(self._fused_lost_count.get(tid, 0))
            self._remember_fused_drop_reason(
                tid,
                decision,
                accepted=accepted,
                status=status,
            )
            if tid in self._fused_icp_metrics:
                self._fused_icp_metrics[tid]["pose_status"] = status
                self._fused_icp_metrics[tid]["lost_count"] = decision["lost_count"]

        self._log_track_pose_csv(object_decisions, stamp)
    
        # ── Back-project and publish (same structure as original) ──
        track_debug_by_cam = {}

        for track_id, entries in by_track.items():
            decision = object_decisions.get(track_id, {
                "mode": "hold_previous", "accepted": False,
                "reason": "missing", "T_base": None,
                "used_tracker_key": None, "used_bbox_xyxy": None,
            })
            obj_id = entries[0]["state"].object_id
            T_base = decision.get("T_base")
            accepted = bool(decision.get("accepted", False)) and T_base is not None

            survived_keys_set = set(decision.get("survived_keys", []))

            for cr in entries:
                cam_id = cr["cam_id"]
                state = cr["state"]
                idx = cr["state_idx"]
                rt_result = cr["rt_result"]
                current_mask = np.asarray(cr["mask"]).astype(bool)

                is_survivor = accepted and (cam_id, idx) in survived_keys_set

                if is_survivor:
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

                    # Push fused pose back into the per-cam RealtimeTracker so
                    # its internal pose stays in sync with the fused result.
                    tracker_key = cr.get("tracker_key")
                    rt = self.realtime_trackers.get(tracker_key) if tracker_key else None
                    if rt is not None:
                        try:
                            rt.force_pose_update(T_local)
                        except Exception:
                            pass

                    # remember healthy tracker masks so the next
                    # init (if it fires) can skip DINO for these objects.
                    occ = float(getattr(rt_result, "occlusion_score", 0.0)) if rt_result is not None else 0.0
                    self._recent_tracker_health[(cam_id, state.track_id)] = {
                        "object_id": state.object_id,
                        "frame_idx": self.frame_counter,
                        "mask": current_mask.copy(),
                        "occlusion_score": occ,
                        "T_object_camera": T_local.copy(),
                        "centroid_base": np.asarray(T_base[:3, 3], dtype=np.float64).copy(),
                    }
                    # Per-object appearance memory: stash a clean RGB crop
                    # while ICP fitness is high so the next reinit can
                    # disambiguate the instance from look-alikes.
                    cr_view = cr.get("view")
                    is_warmup_obj = bool(decision.get("is_warmup", False))
                    if (
                        not is_warmup_obj
                        and cr_view is not None
                        and getattr(cr_view, "rgb", None) is not None
                    ):
                        fused_metrics_for_save = self._fused_icp_metrics.get(track_id, {})
                        fitness_for_save = float(fused_metrics_for_save.get("fitness", 0.0))
                        self._maybe_save_memory_crop(
                            object_id=state.object_id,
                            track_id=track_id,
                            cam_id=cam_id,
                            rgb_full=cr_view.rgb,
                            mask_full=current_mask,
                            bbox_xyxy=cr.get("bbox_xyxy"),
                            fitness=fitness_for_save,
                        )

                    # recovery_mask seeds the next reinit's RealtimeTracker.
                    state.recovery_mask = current_mask
                else:
                    if state.last_good_T is not None:
                        state.T_object_camera = state.last_good_T.copy()
                    state.degraded_count += 1
                    state.mode = "degraded"

                state.last_mask_area = int(current_mask.sum())

                # Per-cam debug publish
                if getattr(self.args, "debug_per_cam_pose_publish", False):
                    if accepted:
                        pose_msg = T_to_pose_stamped(T_base, frame_id="base", stamp=stamp)
                        self._get_or_create_pose_base_pub(
                            cam_id, state.track_id
                        ).publish(pose_msg)
                    else:
                        self._publish_pose_base(
                            cam_id, state.track_id,
                            state.T_object_camera, stamp,
                        )

                if cam_id not in track_debug_by_cam:
                    fused_metrics = self._fused_icp_metrics.get(track_id, {})
                    track_debug_by_cam[cam_id] = {
                        "mask": current_mask,
                        "bbox_xyxy": cr["bbox_xyxy"],
                        "object_id": state.object_id,
                        "icp_fitness": fused_metrics.get("fitness", 0.0),
                        "icp_rmse_mm": fused_metrics.get("rmse_mm", 0.0),
                    }
    
        # ── Debug frame publishing ──
        if self._debug_frame_publish_enabled():
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
        if not hasattr(self, "_pub_fused_pose"):
            self._pub_fused_pose = {}
        if not hasattr(self, "_fused_pose_pub_by_obj"):
            self._fused_pose_pub_by_obj: dict[str, Any] = {}

        for tid, decision in object_decisions.items():
            accepted_pub = decision.get("accepted") and decision.get("T_base") is not None
            if accepted_pub:
                T_pub = decision["T_base"]
            elif decision.get("pose_status") == "held":
                mem = self._fused_track_memory.get(tid)
                T_pub = mem["T_base"] if mem is not None else None
            else:
                T_pub = None
            if T_pub is None:
                continue
            pub = self._fused_pose_pub_by_obj.get(tid)
            if pub is None:
                fused_key = f"fused/{tid}"
                pub = self._pub_fused_pose.get(fused_key)
                if pub is None:
                    pub = self.create_publisher(
                        PoseStamped, f"/perception/fp/pose_base/{fused_key}",
                        FAST_QOS,
                    )
                    self._pub_fused_pose[fused_key] = pub
                self._fused_pose_pub_by_obj[tid] = pub
            pub.publish(
                T_to_pose_stamped(T_pub, frame_id="base", stamp=stamp)
            )

        t_end = time.time()
        t_total = (t_end - t_start) * 1000
        accepted_count = sum(1 for d in object_decisions.values() if d.get("accepted"))
        if getattr(self.args, "debug_verbose_logs", False):
            self.get_logger().info(
                f"FUSED TRACK total: {t_total:.0f}ms | {accepted_count} objects updated"
            )
        # Per-stage latency breakdown
        if self.frame_counter % 20 == 0:
            percam_ms = (t_fuse - t_start) * 1000.0
            icp_ms = (t_icp_end - t_fuse) * 1000.0
            post_ms = (t_end - t_icp_end) * 1000.0
            cutie_n = sum(v[0] for v in cam_timing.values())
            cutie_sum = sum(v[1] for v in cam_timing.values())
            print(
                f"[STAGE] percam={percam_ms:.0f}ms (cutie {cutie_n} calls, "
                f"{cutie_sum:.0f}ms sum)  icp+lift={icp_ms:.0f}ms  "
                f"post={post_ms:.0f}ms  total={t_total:.0f}ms",
                flush=True,
            )

   
    def _check_chamfer_drift(self) -> None:
        """
        Compare mean Chamfer across cameras over recent inits.
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
        """
        t_start = time.time()
        _dprint("\n[TIMING] ========== MULTICAM INIT START ==========")

        # Reset the per-init-pass claim set so each fused detection picks
        # an unused (or newly allocated) track_id.
        self._claimed_track_ids_this_init = set()

        # ── Phase 1: SAM + DINO per camera ──
        selections_by_cam: dict[str, list[CandidateSelection]] = {}
        views_by_cam: dict[str, Any] = {}

        cycle_total_masks = 0
        cycle_total_boxes = 0

        for view in views:
            cam_id = view.cam_id
            if view.rgb is None or view.depth is None:
                continue
            if view.rgb.shape[:2] != view.depth.shape[:2]:
                continue
            views_by_cam[cam_id] = view

            t_sam = time.time()
            self._last_sam_n_boxes = 0
            masks = self._generate_and_filter_masks(view.rgb, cam_id)
            cycle_total_masks += len(masks)
            cycle_total_boxes += int(getattr(self, "_last_sam_n_boxes", 0))
            _dprint(f"[TIMING] SAM {cam_id}: {(time.time()-t_sam)*1000:.0f}ms -> {len(masks)} masks")

            if not masks:
                selections_by_cam[cam_id] = []
                if self._debug_frame_publish_enabled() and cam_id in self.pub_debug_frame:
                    frame = self._build_debug_frame(
                        cam_id=cam_id, stamp=stamp,
                        update_sam=True, update_dino=True,
                        sam_candidates=[], dino_candidates=[],
                        pose_items=[],
                    )
                    self.pub_debug_frame[cam_id].publish(frame)
                continue

            inherited: list[CandidateSelection] = []
            remaining_masks = masks
            if bool(getattr(self.args, "skip_dino_when_tracker_healthy", False)):
                inherited, remaining_masks = self._inherit_from_tracker_health(
                    cam_id=cam_id, masks=masks, depth=view.depth, K=view.K,
                )
                if inherited:
                    _dprint(f"[TIMING] DINO-skip {cam_id}: inherited {len(inherited)} masks from tracker (skipped DINO)")

            t_dino = time.time()
            ranked = (
                self._classify_masks_batched(
                    view.rgb, remaining_masks,
                    cam_id=cam_id, depth=view.depth, K=view.K,
                )
                if remaining_masks else []
            )
            selected = self._select_top_candidates(inherited + ranked, view.depth)
            _dprint(f"[TIMING] DINO+select {cam_id}: {(time.time()-t_dino)*1000:.0f}ms -> {len(selected)} selected")
            selections_by_cam[cam_id] = selected

            if self._debug_frame_publish_enabled() and cam_id in self.pub_debug_frame:
                frame = self._build_debug_frame(
                    cam_id=cam_id, stamp=stamp,
                    update_sam=True, update_dino=True,
                    sam_candidates=self._sam_candidates_to_msgs(masks),
                    dino_candidates=self._dino_ranked_to_msgs(inherited + ranked),
                    pose_items=[],
                )
                self.pub_debug_frame[cam_id].publish(frame)

    
        if bool(getattr(self.args, "restart_on_dead_init", True)):
            # GDINO found boxes but SAM returned no masks across cameras: treat as a bad SAM session.
            min_boxes = int(getattr(self.args, "dead_init_min_boxes", 3))
            if cycle_total_boxes >= min_boxes and cycle_total_masks == 0:
                self._consecutive_dead_init_cycles = (
                    int(getattr(self, "_consecutive_dead_init_cycles", 0)) + 1
                )
                need = int(getattr(self.args, "dead_init_cycles", 2))
                self.get_logger().warn(
                    f"DEAD-INIT: all cameras returned 0 masks for "
                    f"{cycle_total_boxes} boxes "
                    f"({self._consecutive_dead_init_cycles}/{need} consecutive) "
                    f"— bad SAM session, restart-cured"
                )
                if self._consecutive_dead_init_cycles >= need:
                    print(
                        "[DEAD-INIT] bad SAM session confirmed (all cams 0 masks "
                        f"with boxes present, {need}x). Exiting code 42 for "
                        "supervised relaunch.",
                        flush=True,
                    )
                    os._exit(42)
            else:
                self._consecutive_dead_init_cycles = 0

        if sum(len(v) for v in selections_by_cam.values()) == 0:
            for view in views:
                self.track_states[view.cam_id] = []
            _dprint("[TIMING] MULTICAM INIT: no detections")
            return

        # ── Phase 2: Fusion matching ──
        t_fusion = time.time()
        fusion_cfg = FusionConfig(
            max_centroid_distance=float(self.args.fusion_match_max_centroid_dist_m),
            match_ambiguity_margin=float(self.args.fusion_match_ambiguity_margin_m),
            label_match_penalty_weight=float(self.args.fusion_match_label_penalty_m),
        )
        fused_detections = run_multicam_fusion(
            selections_by_cam=selections_by_cam,
            views_by_cam=views_by_cam,
            T_base_cam_map=self.T_base_cam_map,
            cfg=fusion_cfg,
        )
        _dprint(f"[TIMING] Fusion matching: {(time.time()-t_fusion)*1000:.0f}ms -> {len(fused_detections)} fused objects")

        # ── Phase 3: Single-FP + ICP + symmetry grid + weighted average ──
        t_fp_all = time.time()
        new_states_by_cam: dict[str, list[ObjectTrackState]] = {
            v.cam_id: [] for v in views
        }
        fp_skip_cameras = {
            c.strip()
            for c in str(getattr(self.args, "fp_skip_cameras", "")).split(",")
            if c.strip()
        }

        if not hasattr(self, "_pub_fused_pose"):
            self._pub_fused_pose: dict[str, Any] = {}

        for i, fused in enumerate(fused_detections):
            try:
                mesh_path = self._resolve_mesh_path(fused.object_id)
            except FileNotFoundError:
                continue

            # Lift all contributing masks to base-frame clouds before pose estimation.
            INIT_VOXEL_SIZE = 0.001

            t_lift = time.time()
            per_cam_clouds: list[o3d.geometry.PointCloud] = []

            for det in fused.detections:
                cam_id = det.cam_id
                if cam_id not in views_by_cam:
                    continue
                T_bc = self._resolve_T_base_cam(cam_id)
                K_cam = np.asarray(views_by_cam[cam_id].K, dtype=np.float32).reshape(3, 3)

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
                    _dprint(f"  [{cam_id}] Lifted {len(pcd.points)} pts for {fused.object_id}")

            # ─── Cloud overlap diagnostic ───
            if len(per_cam_clouds) >= 2 and getattr(self.args, "debug_verbose_logs", False):
                d01 = per_cam_clouds[0].compute_point_cloud_distance(per_cam_clouds[1])
                d10 = per_cam_clouds[1].compute_point_cloud_distance(per_cam_clouds[0])
                mean_overlap_dist = (float(np.mean(d01)) + float(np.mean(d10))) / 2.0
                cam_ids_str = [det.cam_id for det in fused.detections if det.cam_id in views_by_cam]
                _dprint(
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

            # Init uses an unweighted high-resolution cloud; tracking can choose weighted merge.
            fused_cloud = merge_point_clouds(per_cam_clouds, voxel_size=INIT_VOXEL_SIZE)

            if fused_cloud is None or len(fused_cloud.points) < 50:
                _dprint(f"  Fused cloud too small for {fused.object_id}, skipping")
                continue
            _dprint(f"  Fused cloud: {len(fused_cloud.points)} pts ({(time.time()-t_lift)*1000:.0f}ms)")

            # Mesh sample used by both initial ICP and optional rotation grid.
            model_pcd = mesh_to_pcd_cached(mesh_path, float(self.args.mesh_scale), num_points=5000)

            # Run FoundationPose once per contributing camera, then score/refine in base frame.
            SYMMETRY_GRID_CHAMFER_M = float(getattr(self.args, "icp_grid_skip_chamfer_m", 0.008))
            CHAMFER_REJECT_M = 0.012 if len(fused.detections) == 1 else 0.015

            candidate_poses: list[np.ndarray] = []
            candidate_weights: list[float] = []
            candidate_chamfers: list[float] = []
            candidate_cam_ids: list[str] = []
            candidate_det_indices: list[int] = []

            is_single_cam = len(fused.detections) == 1
            if is_single_cam:
                _dprint(f"  ⚠ SINGLE-CAM init for {fused.object_id}")

            for det_idx, det in enumerate(fused.detections):
                cam_id = det.cam_id
                if cam_id not in views_by_cam:
                    continue
                if cam_id in fp_skip_cameras:
                    _dprint(
                        f"  FP skip [{cam_id}] {fused.object_id}: --fp-skip-cameras"
                    )
                    continue

                view = views_by_cam[cam_id]
                K_cam = np.asarray(view.K, dtype=np.float32).reshape(3, 3)
                tracker = self.fp_tracker

                # FoundationPose sees one camera crop/mask; ICP scores it against the fused cloud.
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

                # Convert candidate pose into the shared base frame for cross-camera comparison.
                T_bc = self._resolve_T_base_cam(cam_id)
                T_base = (T_bc @ T_cam.astype(np.float64)).astype(np.float32)

                # Refine against all contributing depth, not just the camera that produced FP.
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

                _dprint(
                    f"  FP+ICP [{cam_id}] {fused.object_id}: "
                    f"fp={fp_ms:.0f}ms icp={icp_ms:.0f}ms "
                    f"fitness={fitness:.3f} chamfer={chamfer*1000:.2f}mm"
                )

                best_T = T_refined
                best_chamfer = chamfer
                best_fitness = fitness
                best_rmse = rmse

                # If Chamfer is suspicious, try alternative rotations for symmetric objects.
                if best_chamfer > SYMMETRY_GRID_CHAMFER_M:
                    t_grid = time.time()
                    grid_T, grid_chamfer = self._icp_rotation_grid(
                        best_T[:3, 3], model_pcd, fused_cloud,
                        per_cam_clouds=per_cam_clouds,
                    )
                    grid_ms = (time.time() - t_grid) * 1000

                    if grid_T is not None and grid_chamfer < best_chamfer:
                        _dprint(
                            f"    Symmetry grid improved [{cam_id}]: "
                            f"{best_chamfer*1000:.2f}mm → {grid_chamfer*1000:.2f}mm "
                            f"({grid_ms:.0f}ms)"
                        )
                        best_T = grid_T
                        best_chamfer = grid_chamfer
                        # Recompute fitness/rmse against the new pose
                        best_fitness, best_rmse = evaluate_icp_in_base_frame(
                            scene_pcd=fused_cloud,
                            model_pcd=model_pcd,
                            T_base_object=best_T,
                        )
                    else:
                        _dprint(f"    Symmetry grid no improvement [{cam_id}] ({grid_ms:.0f}ms)")

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
                        _dprint(f"  [WARN] init pose CSV log failed: {e}")

                    save_init_pose_render(
                        best_T, model_pcd, fused.object_id,
                        f'init_renders/{fused.object_id}_{cam_id}.png',
                        accepted=accepted_attempt,
                    )

                _dprint(
                    f"  FINAL [{cam_id}] {fused.object_id}: "
                    f"fitness={best_fitness:.3f} rmse={best_rmse*1000:.1f}mm "
                    f"chamfer={best_chamfer*1000:.2f}mm"
                )

                # Track per-camera Chamfer to catch extrinsic drift over repeated inits.
                self._init_chamfer_history[cam_id].append(best_chamfer)

                # High Chamfer is accepted but counted; repeated failures force a fresh init cycle.
                if best_chamfer > CHAMFER_REJECT_M:
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

            # End of per-camera FP attempts for this fused object.

            if not candidate_poses:
                self.get_logger().info(f"  No valid FP results for {fused.object_id}")
                continue

            # Reject camera candidates that are much worse than the best pose before averaging.
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

                    # Only apply if at least one candidate survives.
                    if filtered_indices:
                        candidate_poses = [candidate_poses[j] for j in filtered_indices]
                        candidate_weights = [candidate_weights[j] for j in filtered_indices]
                        candidate_chamfers = [candidate_chamfers[j] for j in filtered_indices]
                        candidate_cam_ids = [candidate_cam_ids[j] for j in filtered_indices]
                        candidate_det_indices = [candidate_det_indices[j] for j in filtered_indices]

            # Weighted average gives one canonical base-frame pose for this fused detection.
            T_base_canonical = weighted_average_poses(candidate_poses, candidate_weights)

            # ─── Polishing ICP: snap the averaged pose back onto the cloud ───
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
            _dprint(
                f"  POLISH ICP {fused.object_id}: "
                f"fitness={polish_fitness:.3f} rmse={polish_rmse*1000:.1f}mm "
                f"chamfer={polish_chamfer*1000:.2f}mm ({polish_ms:.0f}ms)"
            )

            T_publish = T_base_canonical.copy()

            # Log the canonical fused estimate before it is split back to cameras.
            t_canon = T_publish[:3, 3]
            weights_str = ", ".join(
                f"{cid}:{w:.2f}" for cid, w in zip(candidate_cam_ids, candidate_weights)
            )
            _dprint(
                f"  CANONICAL {fused.object_id}: "
                f"t=[{t_canon[0]:.4f}, {t_canon[1]:.4f}, {t_canon[2]:.4f}] "
                f"weights=[{weights_str}]"
                f"{' [SINGLE-CAM]' if is_single_cam else ''}"
            )

            if self.args.distance_confidence_warn:
                dist_m = float(np.linalg.norm(T_publish[:3, 3]))
                min_mask_area = min(
                    int(d.mask.sum()) for d in fused.detections
                ) if fused.detections else 0
                if (
                    dist_m > self.args.distance_confidence_max_m
                    or min_mask_area < self.args.distance_confidence_min_mask_area
                ):
                    _dprint(
                        f"  LOW-CONFIDENCE {fused.object_id}: "
                        f"dist={dist_m:.2f}m min_mask_area={min_mask_area}"
                    )

            # Allocate (or reuse) one track_id per fused detection so all
            # per-cam ObjectTrackStates from this fused detection share
            # identity.
            base_centroid = np.asarray(T_publish[:3, 3], dtype=np.float64).reshape(3)
            track_id_for_fused = self._resolve_track_id_for_new_detection(
                fused.object_id, base_centroid,
            )

            # Back-project the canonical pose into each camera's local tracker state.
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
                    track_id=track_id_for_fused,
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
                    "INIT", cam_id, track_id_for_fused, T_local,
                    extra=f"dino={det.dino_score:.3f} cams={len(candidate_poses)}",
                )

            # Publish canonical fused pose under the track_id key so multi-
            # instance same-class scenes get distinct topics.
            self._last_known_track_centroids[track_id_for_fused] = base_centroid.copy()
            fused_key = f"fused/{track_id_for_fused}"
            if fused_key not in self._pub_fused_pose:
                self._pub_fused_pose[fused_key] = self.create_publisher(
                    PoseStamped, f"/perception/fp/pose_base/{fused_key}", FAST_QOS,
                )
            self._pub_fused_pose[fused_key].publish(
                T_to_pose_stamped(T_publish, frame_id="base", stamp=stamp)
            )

        _dprint(f"[TIMING] FP all objects: {(time.time()-t_fp_all)*1000:.0f}ms")

        # Compare recent init Chamfer by camera to catch slow calibration drift.
        self._check_chamfer_drift()

        # ── Publish debug frames ──
        if self._debug_frame_publish_enabled():
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
        self.get_logger().info(
            f"[TIMING] ========== MULTICAM INIT TOTAL: {t_total:.0f}ms | {total_inited} objects =========="
        )

    def _reset_tracking_state_for_reinit(self, fused_detections):
        """
        Resets warmup counters, median buffers, Kalman filters, and
        per-camera RealtimeTracker instances so the next tick rebuilds
        them against the freshly initialised poses instead of reusing
        Cutie memory tied to the previous instance.
        """
        fresh_track_ids: set[str] = set()
        for states in self.track_states.values():
            for s in states:
                if getattr(s, "track_id", ""):
                    fresh_track_ids.add(s.track_id)

        for tid in fresh_track_ids:
            self._fused_warmup_count[tid] = 0
            if tid in self._median_pose_buffers:
                self._median_pose_buffers[tid].reset()
            if tid in self._fused_translation_kalman:
                self._fused_translation_kalman[tid].reset()
            if tid in self._fused_track_memory:
                del self._fused_track_memory[tid]
            self._fused_lost_count[tid] = 0
            self._fused_pose_status[tid] = "fresh"
            self._fused_last_drop_reasons.pop(tid, None)
            self._force_reinit_tracks.discard(tid)

        for key, rt in list(self.realtime_trackers.items()):
            try:
                rt.reset()
            except Exception:
                pass
            del self.realtime_trackers[key]

        # Reset the per-camera Cutie sessions too (clears Cutie memory and
        # object identities) but keep the loaded network weights so the next
        # tick re-registers objects without reloading the model.
        for session in self.cutie_sessions.values():
            try:
                session.reset()
            except Exception:
                pass

    def _drop_realtime_trackers_for_track_ids(self, track_ids: set[str]) -> None:
        """Release per-camera video trackers for tracks that are about to reinit."""
        if not track_ids:
            return
        for key, rt in list(self.realtime_trackers.items()):
            if not any(key.endswith(f"_{tid}") for tid in track_ids):
                continue
            try:
                rt.reset()
            except Exception:
                pass
            del self.realtime_trackers[key]

        # Remove the dropped objects from every camera's shared Cutie session.
        for session in self.cutie_sessions.values():
            for tid in track_ids:
                try:
                    session.remove_object(tid)
                except Exception:
                    pass
        try:
            import gc
            gc.collect()
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _tick(self) -> None:
        """Timer callback: choose tracking vs reinit for the latest synchronized views."""
        if self.busy:
            return
        views = self.grabber.get_latest_views()
        if views is None:
            if getattr(self.args, "debug_verbose_logs", False):
                _dprint("[TICK] No views yet...")
            return
    
        self.busy = True
        try:
            self.frame_counter += 1
            stamp = self.get_clock().now().to_msg()
    
            if self._force_reinit_tracks:
                # Drop only tracks that exceeded the lost window; keep survivors unless configured otherwise.
                lost_track_ids = set(self._force_reinit_tracks)
                drop_reasons = self._format_fused_drop_reasons(lost_track_ids)
                self.get_logger().warn(
                    f"FORCE REINIT: pose_status=lost for {sorted(lost_track_ids)} "
                    f"reasons={drop_reasons} "
                    f"— dropping their states, re-detecting on this tick"
                )
                for cid in list(self.track_states.keys()):
                    self.track_states[cid] = [
                        s for s in self.track_states[cid]
                        if s.track_id not in lost_track_ids
                    ]
                for tid in lost_track_ids:
                    self._fused_lost_count[tid] = 0
                    self._fused_pose_status[tid] = "fresh"
                    self._fused_last_drop_reasons.pop(tid, None)
                self._drop_realtime_trackers_for_track_ids(lost_track_ids)
                self._force_reinit_tracks.clear()
                remaining_track_ids = sorted({
                    s.track_id
                    for states in self.track_states.values()
                    for s in states
                    if getattr(s, "track_id", "")
                })
                allow_partial_reinit = bool(getattr(self.args, "reinit_lost_tracks_while_tracking", False))
                if remaining_track_ids and not allow_partial_reinit:
                    self.get_logger().warn(
                        f"PARTIAL TRACK LOSS: dropped {sorted(lost_track_ids)}; "
                        f"reasons={drop_reasons}; continuing {remaining_track_ids} "
                        f"without global reinit"
                    )
                    force_init_this_tick = False
                else:
                    force_init_this_tick = True
            else:
                force_init_this_tick = False

            # If any active camera has usable states, stay in tracking; otherwise run full init.
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

            if any_tracking and not no_states and not force_init_this_tick:
                # Fused tracking is the hot loop: Cutie masks + fused ICP.
                self._track_multicam_fused(views, stamp)
            else:
                # Multicam init is expensive but gives fresh identities and poses.
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
    """Command-line knobs for model loading, init quality gates, and tracking behavior."""
    p = argparse.ArgumentParser()

    # Runtime devices and top-level mode selection.
    p.add_argument("--device", default="cuda")
    p.add_argument("--dino-device", choices=["", "cpu", "cuda"], default="",
                   help="Device for DINOv2 image embeddings. Empty reuses --device.")
    p.add_argument("--mask-source",
                   choices=["sam", "projected", "gdino_sam"],
                   default="gdino_sam")
    # Grounding DINO + SAM (MUSE-style) proposal stage
    p.add_argument("--gdino-model-id", default="IDEA-Research/grounding-dino-base")
    p.add_argument("--gdino-box-threshold", type=float, default=0.20)
    p.add_argument("--gdino-text-threshold", type=float, default=0.25)
    p.add_argument("--gdino-max-boxes", type=int, default=40)
    p.add_argument("--gdino-device", choices=["", "cpu", "cuda"], default="",
                   help="Device for Grounding DINO. Empty reuses --device.")
    # Comma-separated text prompts. Default covers the current object set;
    # override via --gdino-text-prompts "a,b,c" to extend.
    p.add_argument(
        "--gdino-text-prompts",
        default="cooling base,cooling f,cooling screw,pb base,pb pipe,pb screw,pb top",
    )
    p.add_argument("--run-mode", choices=["track", "init_only"], default="track")
    p.add_argument(
        "--tracking-profile",
        choices=["default", "fast_cutie"],
        default="default",
        action=_TrackingProfileAction,
        help="Named tracking profile. fast_cutie keeps Cutie in the hot loop with tuned tracking gates.",
    )
    p.add_argument(
        "--fast-cutie",
        action=_FastCutieAction,
        default=False,
        help="Shortcut for --tracking-profile fast_cutie.",
    )

    # Object reference images, CAD meshes, and output folders.
    p.add_argument("--reference-dir", default="Data/ZED_screens")
    p.add_argument("--cad-dir", default="Data/CAD_Models_centered")
    p.add_argument("--output-root", default="outputs/foundationpose")

    # synthetic-render reference bank
    p.add_argument("--reference-source", choices=["real", "renders", "both"],
                   default="real")
    p.add_argument("--reference-renders-dir", default="Data/reference_renders")

    p.add_argument("--dino-model-name", default="dinov2_vitg14")
    p.add_argument("--dino-min-score", type=float, default=0.50)
    p.add_argument("--dino-min-margin", type=float, default=0.05)
    p.add_argument("--area-penalty-weight", type=float, default=1.5)
    p.add_argument("--fill-ratio-weight", type=float, default=0.15)
    # Per-object appearance memory: snapshot a clean RGB crop whenever the
    # fused ICP fitness is good, then use those embeddings to stabilize reinit.

    p.add_argument("--memory-crop-enable", action="store_true", default=True,
                   help="Save high-fitness crops and use them at reinit.")
    p.add_argument("--no-memory-crop", dest="memory_crop_enable",
                   action="store_false",
                   help="Disable the appearance memory bank.")
    p.add_argument("--memory-crop-fitness-threshold", type=float, default=0.5,
                   help="Minimum fused ICP fitness required to save a crop.")
    p.add_argument("--memory-crop-min-frame-gap", type=int, default=15,
                   help="Frames to wait between consecutive crop saves per object/cam.")
    p.add_argument("--memory-crop-max-per-object", type=int, default=8,
                   help="Ring-buffer capacity per object.")
    p.add_argument("--memory-crop-weight", type=float, default=0.35,
                   help="Blend weight for memory similarity vs raw DINO score at reinit. 0 disables re-ranking.")
    p.add_argument("--memory-crop-min-score-floor", type=float, default=0.0,

                   help="Skip memory re-ranking when raw DINO top1 < this (avoids boosting garbage).")
    p.add_argument("--memory-crop-keep-rgb", action="store_true", default=False,
                   help="Keep the raw RGB+mask alongside the embedding (debug only).")
    p.add_argument("--memory-crop-save-dir", default="",
                   help="If set, dump saved crops to this directory for inspection.")

    # SAM2 and dead-session recovery.
    p.add_argument("--sam-repo-root", default="external/sam2")
    p.add_argument(
        "--sam-checkpoint",
        default="external/sam2/checkpoints/sam2.1_hiera_base_plus.pt",
    )
    p.add_argument("--sam-model-cfg", default="configs/sam2.1/sam2.1_hiera_b+.yaml")
    p.add_argument("--sam-max-image-side", type=int, default=1536)
    p.add_argument("--sam-fp32", action="store_true")

    p.add_argument("--restart-on-dead-init", action="store_true", default=True)
    p.add_argument("--no-restart-on-dead-init", dest="restart_on_dead_init",
                   action="store_false")
    p.add_argument("--dead-init-cycles", type=int, default=2,
                   help="consecutive all-cam-0-mask cycles before exiting for restart")
    p.add_argument("--dead-init-min-boxes", type=int, default=3,
                   help="min GDINO boxes in a cycle to count it as a real (non-empty) scene")
    

    # Number of cameras to use. 2 = zed2i_1+zed2i_2 (default), 3 adds zed2i_3.
    p.add_argument("--num-cameras", type=int, default=3, choices=[2, 3])
    p.add_argument(
        "--fp-skip-cameras",
        type=str,
        default="",
        help="Comma-separated camera ids to exclude from FoundationPose init calls.",
    )

    # Per-camera SAM/ROI filtering thresholds.
    p.add_argument("--cam1-sam-min-mask-area", type=int, default=10)
    p.add_argument("--cam1-sam-min-bbox-side-px", type=int, default=2)
    p.add_argument("--cam1-sam-max-mask-area-ratio", type=float, default=0.06)
    p.add_argument("--cam1-sam-max-bbox-area-ratio", type=float, default=0.06)
    p.add_argument("--cam1-sam-border-px", type=int, default=6)
    p.add_argument("--cam1-sam-max-border-fraction", type=float, default=0.00)

    p.add_argument("--cam2-sam-min-mask-area", type=int, default=10)
    p.add_argument("--cam2-sam-min-bbox-side-px", type=int, default=2)
    p.add_argument("--cam2-sam-max-mask-area-ratio", type=float, default=0.06)
    p.add_argument("--cam2-sam-max-bbox-area-ratio", type=float, default=0.06)
    p.add_argument("--cam2-sam-border-px", type=int, default=6)
    p.add_argument("--cam2-sam-max-border-fraction", type=float, default=0.00)

    # cam3 SAM params (only used when --num-cameras 3). Defaults mirror cam2.
    p.add_argument("--cam3-sam-min-mask-area", type=int, default=10)
    p.add_argument("--cam3-sam-min-bbox-side-px", type=int, default=2)
    p.add_argument("--cam3-sam-max-mask-area-ratio", type=float, default=0.06)
    p.add_argument("--cam3-sam-max-bbox-area-ratio", type=float, default=0.06)
    p.add_argument("--cam3-sam-border-px", type=int, default=6)
    p.add_argument("--cam3-sam-max-border-fraction", type=float, default=0.00)

    p.add_argument("--cam1-roi-polygon", type=str,
        default="" )
    p.add_argument("--cam2-roi-polygon", type=str,
        default="")
    # Empty string => no ROI mask for cam3 until tuned.
    p.add_argument("--cam3-roi-polygon", type=str, default="")

    p.add_argument("--mask-dedup-iou", type=float, default=0.6)

    p.add_argument("--fp-repo-root", default="external/FoundationPose")
    p.add_argument("--fp-weights-dir", default="external/FoundationPose/weights")
    p.add_argument("--fp-debug", type=int, default=0)
    p.add_argument("--mesh-scale", type=float, default=0.01)

    p.add_argument("--timer-period-s", type=float, default=0.05)
    p.add_argument("--max-candidate-draw", type=int, default=25)

    p.add_argument("--min-valid-z-m", type=float, default=0.05)
    p.add_argument("--max-valid_z_m", dest="max_valid_z_m", type=float, default=10.00)

    p.add_argument("--max-objects", type=int, default=15)

    p.add_argument("--min-depth-coverage", type=float, default=0.50)

    # Fused tracking candidate gates before cloud merge.
    p.add_argument("--fused-gate-min-mask-area", type=int, default=50)
    p.add_argument("--fused-gate-min-mask-area-ratio", type=float, default=0.40)
    p.add_argument("--fused-gate-max-mask-area-ratio", type=float, default=2.50)
    p.add_argument("--fused-gate-min-depth-coverage", type=float, default=0.30)
    p.add_argument("--fused-gate-min-cloud-points", type=int, default=40)
    p.add_argument("--fused-gate-max-centroid-dist-m", type=float, default=0.08)
    # Cross-camera consistency check at merge
    p.add_argument("--fused-consistency-max-disagreement-m", type=float, default=0.0)

    # Cross-camera fusion MATCHING gate (init-only; distinct from the tracking
    # gate above). Two per-cam detections fuse into one object iff their base-
    # frame centroids are within this distance.
    p.add_argument("--fusion-match-max-centroid-dist-m", type=float, default=0.07)
    # Geometric ambiguity guard
    p.add_argument("--fusion-match-ambiguity-margin-m", type=float, default=0.0)
    # Soft DINO label penalty (meters of extra matching cost for a full label
    # disagreement)
    p.add_argument("--fusion-match-label-penalty-m", type=float, default=0.0)
    p.add_argument("--fused-gate-min-per-cam-icp-fitness", type=float, default=0.18)
    p.add_argument("--fused-gate-max-per-cam-icp-rmse-m", type=float, default=0.015)

    # Fused pose acceptance gates after ICP.
    p.add_argument("--fused-track-min-fused-icp-fitness", type=float, default=0.12)
    p.add_argument("--fused-track-max-fused-icp-rmse-m", type=float, default=0.012)
    p.add_argument("--fused-track-nominal-dt-s", type=float, default=0.15)
    p.add_argument("--fused-track-min-dt-s", type=float, default=0.10)
    p.add_argument("--fused-track-max-dt-s", type=float, default=0.30)
    p.add_argument("--fused-track-max-translation-speed-mps", type=float, default=0.13333333333333333)
    p.add_argument("--fused-track-max-rotation-speed-degps", type=float, default=66.66666666666667)
    p.add_argument("--fused-track-min-translation-jump-m", type=float, default=0.008)
    p.add_argument("--fused-track-min-rotation-jump-deg", type=float, default=4.0)
    p.add_argument("--fused-track-kalman-soft-translation-residual-m", type=float, default=0.05)
    p.add_argument("--fused-track-kalman-soft-max-icp-fitness", type=float, default=0.22)
    p.add_argument("--fused-track-weak-icp-fitness", type=float, default=0.18)
    p.add_argument("--fused-track-axis-dominant-fraction", type=float, default=0.80)
    p.add_argument("--fused-track-axis-dominant-min-translation-m", type=float, default=0.012)
    p.add_argument("--fused-track-icp-max-correspondence-dist-m", type=float, default=0.05)
    # Tracking uses a tight init from the previous frame
    p.add_argument("--fused-track-icp-max-iteration", type=int, default=15)
    # Adaptive early-stop tolerances for tracking ICP
    p.add_argument("--fused-track-icp-relative-fitness", type=float, default=1e-4)
    p.add_argument("--fused-track-icp-relative-rmse", type=float, default=1e-4)

    # Optional fast-motion recovery: translate the ICP seed from agreeing cloud centroids.
    p.add_argument("--fused-track-centroid-recovery", action="store_true",
                   help="Enable centroid-seeded ICP recovery for large mask-cloud jumps.")
    p.add_argument("--fused-track-centroid-recovery-cluster-dist-m", type=float, default=0.12,
                   help="Max distance between camera cloud centroids to form a recovery cluster.")
    p.add_argument("--fused-track-centroid-recovery-min-cameras", type=int, default=1,
                   help="Minimum agreeing centroid-far cameras required for recovery.")
    p.add_argument("--fused-track-centroid-recovery-max-seed-jump-m", type=float, default=0.75,
                   help="Reject centroid recovery if the proposed seed jump exceeds this distance.")

    # Optional rotation re-seed during tracking, useful for symmetric/elongated meshes.
    p.add_argument("--fused-track-rot-reseed", action="store_true",
                   help="Enable chamfer-triggered rotation re-seed during tracking.")
    p.add_argument("--fused-track-rot-reseed-chamfer-m", type=float, default=0.010,
                   help="Trigger: re-seed only when grid-chamfer exceeds this (m).")
    p.add_argument("--fused-track-rot-reseed-max-chamfer-m", type=float, default=0.080,
                   help="Ceiling: above this the object is lost (not mis-rotated); skip the grid.")
    p.add_argument("--fused-track-rot-reseed-n-rot", type=int, default=24,
                   help="Rotation candidates for the tracking re-seed grid (init uses --icp-grid-n-rot).")
    p.add_argument("--fused-track-rot-reseed-icp-iters", type=int, default=10,
                   help="ICP iterations per candidate in the tracking re-seed grid.")

    # Optional PCA shaft-axis correction for shaft-like objects.
    p.add_argument("--fused-track-pca-axis", action="store_true",
                   help="Enable PCA shaft-axis correction during tracking.")
    p.add_argument("--fused-track-pca-axis-min-deg", type=float, default=10.0,
                   help="Only correct when ICP shaft-axis disagrees with PCA axis by more than this (deg).")
    p.add_argument("--fused-track-pca-axis-max-deg", type=float, default=60.0,
                   help="Ceiling: above this the PCA axis is likely unreliable; skip correction (deg).")
    p.add_argument("--fused-track-pca-axis-min-elongation", type=float, default=3.0,
                   help="Only apply when the cloud is shaft-like (lambda1/lambda2 >= this).")
    p.add_argument("--fused-track-pca-axis-min-points", type=int, default=50,
                   help="Minimum fused-cloud points required for a stable PCA axis.")
    p.add_argument("--fused-track-pca-axis-blend", type=float, default=1.0,
                   help="Correction strength in [0,1]; 1.0 = full snap, <1.0 = partial (less jitter).")

    # Optional rotation damping during tracking.
    p.add_argument("--fused-track-rot-slew-limit-deg", type=float, default=0.0,
                   help="Cap the per-frame rotation change (deg) vs the previous pose; "
                        "SLERP from prev toward the ICP update so the change == limit. 0.0 = off.")
    p.add_argument("--fused-track-rot-lowpass", type=float, default=0.0,
                   help="Low-pass the (slew-limited) rotation toward the previous pose by "
                        "this factor in [0,1] to smooth jitter. 0.0 = off.")

    # Distance-weighted cloud merging (mitigates depth bias at distance)
    p.add_argument("--use-weighted-cloud-merge", action="store_true")
    p.add_argument("--cloud-merge-distance-exponent", type=float, default=2.0)

    # Warmup frames after init relax gates while Cutie/ICP settle.
    p.add_argument("--fused-track-warmup-frames", type=int, default=5)

    # Hold/stale/lost windows for publishing the last good pose before reinit.
    p.add_argument("--fused-track-hold-window-frames", type=int, default=5)
    p.add_argument("--fused-track-max-lost-frames", type=int, default=20)
    p.add_argument(
        "--reinit-lost-tracks-while-tracking",
        action="store_true",
        help=(
            "When one track is lost but other tracks still exist, run global "
            "multicam init immediately. By default partial loss only drops the "
            "lost track and keeps tracking the survivors."
        ),
    )

    # Chamfer and mask-origin gates for outlier rejection during tracking.
    p.add_argument("--fused-track-max-chamfer-m", type=float, default=0.015)
    p.add_argument(
        "--track-require-pose-origin-in-mask",
        action="store_true",
        help=(
            "Reject tracking poses whose projected object-frame origin is not "
            "inside the current Cutie tracker mask."
        ),
    )
    p.add_argument(
        "--track-pose-mask-margin-px",
        type=int,
        default=8,
        help="Pixel tolerance around the projected pose origin for --track-require-pose-origin-in-mask.",
    )

    # Median pose buffer (temporal outlier filter, 0 to disable).
    p.add_argument("--median-pose-buffer-size", type=int, default=3)

    p.add_argument("--log-init-poses", action="store_true",
                   help="Log CSV of init pose RPY + render 3D PNGs per attempt")
    p.add_argument("--log-track-poses", action="store_true",
                   help="Log compact per-tick fused tracking poses and metrics to CSV.")
    p.add_argument("--track-pose-log-path",
                   default="outputs/logs/track_pose_log.csv",
                   help="CSV path for --log-track-poses. Overwritten at node startup.")

    # ── Latency / debug flags ──
    p.add_argument("--debug-per-cam-pose-publish", action="store_true")
    p.add_argument("--debug-frame-publish", dest="debug_frame_publish",
                   action="store_true", default=True,
                   help="Publish fp_debug_msgs/DebugFrame messages.")
    p.add_argument("--no-debug-frame-publish", dest="debug_frame_publish",
                   action="store_false",
                   help="Skip DebugFrame construction and publication.")
    # When true, emit per-frame INFO logs and [TIMING] prints. Off by default.
    p.add_argument("--debug-verbose-logs", action="store_true")
    # Master switch for all logging/print statements across the pipeline.
    p.add_argument("--debug-logging", action="store_true",
                   help="Enable all pipeline logging (prints + ROS logger info/warn).")

    # Tracking ICP can be skipped entirely (per-cam ICP is redundant because the
    # fused-cloud ICP refines the pose anyway).
    p.add_argument("--skip-per-cam-icp-tracking", action="store_true", default=True)

    p.add_argument("--cutie-max-internal-size", type=int, default=480,
                   help="Max image side for Cutie tracking resize. <=0 disables Cutie downscale.")

    # Tracking-time model PCD point count (init keeps the original 5000 via a
    # separate cache entry).
    p.add_argument("--track-icp-num-points", type=int, default=2000)

    # Conditional chamfer thresholds. Skip chamfer when both fitness/rmse are
    # already clean and motion is small.
    p.add_argument("--chamfer-skip-fitness-min", type=float, default=0.30)
    p.add_argument("--chamfer-skip-rmse-max-m", type=float, default=0.005)
    p.add_argument("--chamfer-skip-motion-max-m", type=float, default=0.010)
    p.add_argument("--chamfer-every-n-frames", type=int, default=1,
                   help="Run non-clean tracking Chamfer gate every N frames. 1 checks every eligible frame.")
    p.add_argument("--disable-fused-kalman", action="store_true", default=False,
                   help="Skip fused tracking Kalman prediction/update and soft reject gate.")
    p.add_argument("--enable-fused-kalman", dest="disable_fused_kalman",
                   action="store_false",
                   help="Re-enable fused tracking Kalman when a profile disabled it.")
    p.add_argument("--disable-axis-jump-gate", action="store_true", default=False,
                   help="Skip axis-dominant weak-ICP jump rejection.")
    p.add_argument("--enable-axis-jump-gate", dest="disable_axis_jump_gate",
                   action="store_false",
                   help="Re-enable axis-dominant jump rejection when a profile disabled it.")

    # ICP variant for run_icp_in_base_frame.
    p.add_argument("--icp-variant", choices=["point_to_point", "point_to_plane"],
                   default="point_to_point")

    # warn-tag poses that are likely lower confidence (far away / tiny mask).
    p.add_argument("--distance-confidence-warn", action="store_true")
    p.add_argument("--distance-confidence-max-m", type=float, default=1.5)
    p.add_argument("--distance-confidence-min-mask-area", type=int, default=2000)


    # Rotation-grid rescue for init and optional tracking re-seed.
    #   --icp-grid-n-rot             : number of uniform SO(3) seed rotations
    #   --icp-grid-prescreen         : skip ICP for seeds whose raw-Chamfer > tau
    #   --icp-grid-cross-cam-chamfer : score by mean Chamfer across per-cam clouds
    p.add_argument("--icp-grid-n-rot", type=int, default=45)
    p.add_argument("--icp-grid-prescreen", action="store_true")
    p.add_argument("--icp-grid-prescreen-tau", type=float, default=0.04)
    p.add_argument("--icp-grid-cross-cam-chamfer", action="store_true")


    p.add_argument("--depth-fill-holes-kernel", type=int, default=0)

    # threshold above which the rotation grid runs.
    p.add_argument("--icp-grid-skip-chamfer-m", type=float, default=0.004)

    p.add_argument("--dino-gem-p", type=float, default=1.5,
                   help="GeM exponent for the MUSE two-stream patch stream.")

    p.add_argument("--gdino-use-items-prompt", dest="gdino_use_items_prompt",
                   action="store_true", default=True,
                   help="Use the MUSE class-agnostic literal 'items' prompt.")
    p.add_argument("--no-gdino-use-items-prompt", dest="gdino_use_items_prompt",
                   action="store_false",
                   help="Use --gdino-text-prompts instead of the MUSE 'items' prompt.")

    # bicubic-upscale DINO crops whose short side is below this many pixels
    p.add_argument("--dino-min-crop-side", type=int, default=0)

    p.add_argument("--table-plane-z-min", type=float, default=0.0)
    p.add_argument("--table-plane-z-max", type=float, default=0.9)


    p.add_argument("--skip-dino-when-tracker-healthy", action="store_true")
    p.add_argument("--tracker-health-iou-threshold", type=float, default=0.5)
    p.add_argument("--tracker-health-stale-frames", type=int, default=5)
    p.add_argument("--tracker-health-max-occlusion", type=float, default=0.4)
    p.add_argument("--identity-shortcut-max-centroid-dist-m", type=float, default=0.05)
    p.add_argument("--identity-shortcut-ambiguity-radius-m", type=float, default=0.05)
    p.add_argument("--memory-rerank-require-track-consistency", action="store_true", default=True)
    p.add_argument("--no-memory-rerank-require-track-consistency",
                   dest="memory_rerank_require_track_consistency",
                   action="store_false")

    p.add_argument("--icp-mask-close-kernel", type=int, default=0)
    p.add_argument("--icp-mask-interior-erosion", type=int, default=0)

    return p.parse_args()


def main() -> None:
    """Initialize ROS, wire grabber + tracker nodes, and spin until shutdown."""
    torch.cuda.empty_cache()
    args = parse_args()
    global _DEBUG_LOGGING
    _DEBUG_LOGGING = bool(args.debug_logging)
    rclpy.init()

    # Extrinsics and camera topics must agree: grabber timestamps frames, tracker uses poses.
    T_map = load_extrinsics_yaml("config/camera_extrinsics_base.yaml")
    cameras = select_cameras(args.num_cameras)
    print(f"[PIPELINE] Using {len(cameras)} cameras: {[c.cam_id for c in cameras]}")
    grabber = MultiCamGrabber(
        cameras=cameras,
        sync_slop_s=0.10,
        use_best_effort_if_unsynced=True,
        static_extrinsics_base_cam=T_map,
        rgb_depth_max_dt_s=0.08,
    )

    node = FoundationPoseTrackerNode(args=args, grabber=grabber, T_base_cam_map=T_map)

    # MultiThreadedExecutor lets the grabber callbacks and tracker timer overlap.
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(grabber)
    executor.add_node(node)

    try:
        executor.spin()
    finally:
        # Explicit teardown keeps ROS shutdown clean after Ctrl-C or supervised exits.
        executor.remove_node(node)
        executor.remove_node(grabber)
        node.destroy_node()
        grabber.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
