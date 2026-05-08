"""
Real-time 6DoF object tracker combining:
- CuteVOS for mask tracking (handles motion, occlusion)
- Colored ICP for pose refinement (6DoF from depth)

Target: 10Hz tracking for robot manipulation feedback.

Usage:
    tracker = RealtimeTracker(RealtimeTrackerConfig())
    
    # Initialize with first detection
    tracker.initialize(
        rgb=first_frame,
        depth=first_depth,
        mask=initial_mask,
        T_init=initial_pose,  # From FoundationPose
        K=camera_intrinsics,
        mesh_path=object_mesh,
    )
    
    # Track loop
    for rgb, depth in camera_stream:
        result = tracker.track(rgb, depth, K)
        if result.valid:
            robot.update_target(result.T_object_camera)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np

from .cutie_tracker import CutieConfig, CutieTracker, CutieResult
from .icp_refiner import ICPConfig, ICPRefiner, ICPResult, ICPVariant


class TrackingState(Enum):
    UNINITIALIZED = "uninitialized"
    TRACKING = "tracking"
    LOST = "lost"
    NEEDS_REINIT = "needs_reinit"


@dataclass
class RealtimeTrackerConfig:
    """Configuration for the combined tracker."""
    
    # Sub-module configs
    cutie_cfg: CutieConfig = field(default_factory=CutieConfig)
    icp_cfg: ICPConfig = field(default_factory=ICPConfig)
    
    # Quality thresholds for tracking
    min_mask_area: int = 100  # Minimum pixels to consider valid
    min_icp_fitness: float = 0.25  # Below this = lost
    max_icp_rmse: float = 0.008  # 8mm, above this = degraded
    
    # Motion sanity checks
    max_translation_per_frame: float = 0.10  # 5cm max motion between frames
    max_rotation_per_frame_deg: float = 90.0  # 30° max rotation between frames
    
    # Recovery
    lost_frames_before_reinit: int = 5  # Request re-init after this many lost frames
    
    # Timing budget (for adaptive quality)
    target_hz: float = 10.0
    
    # Debug
    verbose: bool = False


@dataclass
class TrackingResult:
    """Result from single tracking iteration."""
    
    # State
    valid: bool  # True if pose is reliable
    state: TrackingState
    
    # Pose (object in camera frame)
    T_object_camera: np.ndarray  # (4, 4) 
    
    # Mask
    mask: np.ndarray  # (H, W) bool
    bbox_xyxy: tuple[int, int, int, int]
    
    # Quality metrics
    mask_area: int
    icp_fitness: float
    icp_rmse: float
    
    # Timing
    cutie_ms: float
    icp_ms: float
    total_ms: float
    
    # Debug info
    message: str = ""


class RealtimeTracker:
    """
    Combined CuteVOS + ICP tracker for real-time 6DoF pose tracking.
    """
    
    def __init__(self, cfg: Optional[RealtimeTrackerConfig] = None):
        self.cfg = cfg or RealtimeTrackerConfig()
        
        # Sub-modules
        self._cutie = CutieTracker(self.cfg.cutie_cfg)
        self._icp = ICPRefiner(self.cfg.icp_cfg)
        
        # State
        self._state = TrackingState.UNINITIALIZED
        self._T_current: Optional[np.ndarray] = None
        self._K: Optional[np.ndarray] = None
        self._lost_count = 0
        self._frame_count = 0
        
        # For motion sanity checks
        self._T_prev: Optional[np.ndarray] = None
        
    @property
    def state(self) -> TrackingState:
        return self._state
        
    @property
    def is_tracking(self) -> bool:
        return self._state == TrackingState.TRACKING
        
    @property
    def needs_reinit(self) -> bool:
        return self._state == TrackingState.NEEDS_REINIT
    
    def initialize(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        mask: np.ndarray,
        T_init: np.ndarray,
        K: np.ndarray,
        mesh_path: Optional[str] = None,
        model_vertices: Optional[np.ndarray] = None,
        model_colors: Optional[np.ndarray] = None,
    ) -> None:
        """
        Initialize tracker with first detection.
        
        Must provide either mesh_path OR (model_vertices, model_colors).
        
        Args:
            rgb: (H, W, 3) uint8 RGB image
            depth: (H, W) float32 depth in meters
            mask: (H, W) bool mask of object
            T_init: (4, 4) initial pose from FoundationPose
            K: (3, 3) camera intrinsics
            mesh_path: Path to object mesh file
            model_vertices: (N, 3) model point cloud (alternative to mesh)
            model_colors: (N, 3) model colors [0, 1] (optional)
        """
        # Initialize Cutie with first frame + mask
        self._cutie.initialize(rgb, mask.astype(np.uint8))
        
        # Initialize ICP with model
        if mesh_path is not None:
            self._icp.set_model_from_mesh(mesh_path,  scale=0.01)
        elif model_vertices is not None:
            self._icp.set_model_cloud(model_vertices, model_colors)
        else:
            raise ValueError("Must provide mesh_path or model_vertices")
        
        # Store state
        self._T_current = T_init.copy()
        self._T_prev = T_init.copy()
        self._K = K.copy()
        self._state = TrackingState.TRACKING
        self._lost_count = 0
        self._frame_count = 1
        
        if self.cfg.verbose:
            print(f"[RealtimeTracker] Initialized with T={T_init[:3, 3]}")
    
    def track(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        K: Optional[np.ndarray] = None,
        skip_icp: bool = False,
    ) -> TrackingResult:
        """
        Track object in new frame.

        Args:
            rgb: (H, W, 3) uint8 RGB image
            depth: (H, W) float32 depth in meters
            K: (3, 3) camera intrinsics (uses stored K if not provided)
            skip_icp: when True, only Cutie runs and the pose is held at
                the current internal value. Used by the multicam-fused pipeline
                where a single fused-cloud ICP refines the pose downstream.

        Returns:
            TrackingResult with pose and quality metrics
        """
        if self._state == TrackingState.UNINITIALIZED:
            return self._make_invalid_result("Not initialized")

        if self._state == TrackingState.NEEDS_REINIT:
            return self._make_invalid_result("Needs re-initialization")

        K = K if K is not None else self._K
        t0 = time.time()

        # =====================================================================
        # Stage 1: Mask tracking with CuteVOS
        # =====================================================================
        try:
            cutie_result = self._cutie.track(rgb)
        except Exception as e:
            self._handle_lost(f"Cutie failed: {e}")
            return self._make_invalid_result(f"Cutie failed: {e}")

        cutie_ms = cutie_result.elapsed_ms

        # Check mask validity
        if cutie_result.area < self.cfg.min_mask_area:
            self._handle_lost(f"Mask too small: {cutie_result.area} < {self.cfg.min_mask_area}")
            return self._make_invalid_result(
                f"Mask too small: {cutie_result.area}",
                mask=cutie_result.mask,
                bbox=cutie_result.bbox_xyxy,
                cutie_ms=cutie_ms,
            )

        # =====================================================================
        # Mask-only fast path: skip ICP, return pose unchanged. The fused-cloud
        # ICP downstream is the authoritative pose refinement.
        # =====================================================================
        if skip_icp:
            self._lost_count = 0
            self._frame_count += 1
            self._state = TrackingState.TRACKING
            total_ms = (time.time() - t0) * 1000
            return TrackingResult(
                valid=True,
                state=TrackingState.TRACKING,
                T_object_camera=self._T_current.copy(),
                mask=cutie_result.mask,
                bbox_xyxy=cutie_result.bbox_xyxy,
                mask_area=cutie_result.area,
                icp_fitness=0.0,
                icp_rmse=0.0,
                cutie_ms=cutie_ms,
                icp_ms=0.0,
                total_ms=total_ms,
                message="OK (mask-only)",
            )
        
        # =====================================================================
        # Stage 2: Pose refinement with ICP
        # =====================================================================
        t1 = time.time()
        
        try:
            icp_result = self._icp.refine(
                depth=depth,
                rgb=rgb,
                mask=cutie_result.mask,
                K=K,
                T_init=self._T_current,
            )
        except Exception as e:
            self._handle_lost(f"ICP failed: {e}")
            return self._make_invalid_result(
                f"ICP failed: {e}",
                mask=cutie_result.mask,
                bbox=cutie_result.bbox_xyxy,
                cutie_ms=cutie_ms,
            )
        
        icp_ms = icp_result.elapsed_ms
        
        # =====================================================================
        # Stage 3: Quality checks
        # =====================================================================
        
        # Check ICP quality
        if icp_result.fitness < self.cfg.min_icp_fitness:
            self._handle_lost(f"ICP fitness too low: {icp_result.fitness:.3f}")
            return self._make_invalid_result(
                f"ICP fitness too low: {icp_result.fitness:.3f}",
                mask=cutie_result.mask,
                bbox=cutie_result.bbox_xyxy,
                cutie_ms=cutie_ms,
                icp_ms=icp_ms,
                icp_fitness=icp_result.fitness,
                icp_rmse=icp_result.inlier_rmse,
            )
        
        # Check motion sanity
        T_new = icp_result.T_refined
        if self._frame_count == 1:
            pass
        elif not self._check_motion_sanity(T_new):
            self._handle_lost("Motion sanity check failed")
            return self._make_invalid_result(
                "Motion too large between frames",
                mask=cutie_result.mask,
                bbox=cutie_result.bbox_xyxy,
                cutie_ms=cutie_ms,
                icp_ms=icp_ms,
                icp_fitness=icp_result.fitness,
                icp_rmse=icp_result.inlier_rmse,
            )
        
        # =====================================================================
        # Success: Update state
        # =====================================================================
        self._T_prev = self._T_current.copy()
        self._T_current = T_new.copy()
        self._lost_count = 0
        self._frame_count += 1
        self._state = TrackingState.TRACKING
        
        total_ms = (time.time() - t0) * 1000
        
        if self.cfg.verbose:
            print(f"[Track #{self._frame_count}] "
                  f"cutie={cutie_ms:.1f}ms icp={icp_ms:.1f}ms total={total_ms:.1f}ms "
                  f"fitness={icp_result.fitness:.3f} rmse={icp_result.inlier_rmse*1000:.2f}mm")
        
        return TrackingResult(
            valid=True,
            state=TrackingState.TRACKING,
            T_object_camera=T_new,
            mask=cutie_result.mask,
            bbox_xyxy=cutie_result.bbox_xyxy,
            mask_area=cutie_result.area,
            icp_fitness=icp_result.fitness,
            icp_rmse=icp_result.inlier_rmse,
            cutie_ms=cutie_ms,
            icp_ms=icp_ms,
            total_ms=total_ms,
            message="OK",
        )
    
    def _check_motion_sanity(self, T_new: np.ndarray) -> bool:
        if self._T_prev is None:
            return True
            
        t_prev = self._T_prev[:3, 3]
        t_new = T_new[:3, 3]
        translation = np.linalg.norm(t_new - t_prev)
        
        # print(f"[Motion DEBUG] t_prev={t_prev}, t_new={t_new}")
        #print(f"[Motion DEBUG] delta={translation*1000:.1f}mm, threshold={self.cfg.max_translation_per_frame*1000:.1f}mm")
        
        if translation > self.cfg.max_translation_per_frame:
            return False
        
        # Rotation check
        R_prev = self._T_prev[:3, :3]
        R_new = T_new[:3, :3]
        R_rel = R_prev.T @ R_new
        
        cos_angle = (np.trace(R_rel) - 1) / 2
        cos_angle = np.clip(cos_angle, -1, 1)
        angle_deg = np.degrees(np.arccos(cos_angle))
        
        # print(f"[Motion DEBUG] rotation delta={angle_deg:.1f}°, threshold={self.cfg.max_rotation_per_frame_deg:.1f}°")
        
        if angle_deg > self.cfg.max_rotation_per_frame_deg:
            return False
        
        return True
    
    def _handle_lost(self, reason: str) -> None:
        """Handle lost tracking."""
        self._lost_count += 1
        
        if self.cfg.verbose:
            print(f"[Lost #{self._lost_count}] {reason}")
        
        if self._lost_count >= self.cfg.lost_frames_before_reinit:
            self._state = TrackingState.NEEDS_REINIT
            if self.cfg.verbose:
                print(f"[RealtimeTracker] Lost for {self._lost_count} frames, requesting re-init")
        else:
            self._state = TrackingState.LOST
    
    def _make_invalid_result(
        self,
        message: str,
        mask: Optional[np.ndarray] = None,
        bbox: Optional[tuple] = None,
        cutie_ms: float = 0.0,
        icp_ms: float = 0.0,
        icp_fitness: float = 0.0,
        icp_rmse: float = float('inf'),
    ) -> TrackingResult:
        """Create an invalid tracking result."""
        return TrackingResult(
            valid=False,
            state=self._state,
            T_object_camera=self._T_current if self._T_current is not None else np.eye(4),
            mask=mask if mask is not None else np.zeros((1, 1), dtype=bool),
            bbox_xyxy=bbox if bbox is not None else (0, 0, 0, 0),
            mask_area=int(mask.sum()) if mask is not None else 0,
            icp_fitness=icp_fitness,
            icp_rmse=icp_rmse,
            cutie_ms=cutie_ms,
            icp_ms=icp_ms,
            total_ms=cutie_ms + icp_ms,
            message=message,
        )
    
    def reset(self) -> None:
        """Reset tracker state."""
        self._cutie.reset()
        self._state = TrackingState.UNINITIALIZED
        self._T_current = None
        self._T_prev = None
        self._lost_count = 0
        self._frame_count = 0
    
    def get_last_pose(self) -> Optional[np.ndarray]:
        """Get last known pose (even if tracking is lost)."""
        return self._T_current.copy() if self._T_current is not None else None
    
    def force_pose_update(self, T_new: np.ndarray) -> None:
        """
        Force update pose from external source (e.g., after re-init).
        Also updates Cutie memory if needed.
        """
        self._T_prev = self._T_current.copy() if self._T_current is not None else T_new.copy()
        self._T_current = T_new.copy()
        self._lost_count = 0
        self._state = TrackingState.TRACKING