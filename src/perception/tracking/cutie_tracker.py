"""
CuteVOS (Cutie) wrapper for real-time video object segmentation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import torch


@dataclass
class CutieConfig:
    variant: str = "base"
    device: str = "cuda"
    max_memory_frames: int = 5
    top_k: int = 30
    mem_every: int = 3
    max_internal_size: Optional[int] = 480


@dataclass  
class CutieResult:
    mask: np.ndarray
    prob: np.ndarray
    bbox_xyxy: tuple[int, int, int, int]
    area: int
    elapsed_ms: float


class CutieTracker:
    def __init__(self, cfg: Optional[CutieConfig] = None):
        self.cfg = cfg or CutieConfig()
        self.device = torch.device(self.cfg.device)
        self._model = None
        self._initialized = False
        self._frame_count = 0

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
            
        print("[CutieTracker] Loading model...")
        t0 = time.time()
        
        try:
            from hydra.core.global_hydra import GlobalHydra
            if GlobalHydra.instance().is_initialized():
                GlobalHydra.instance().clear()
        except:
            pass
        
        from cutie.inference.inference_core import InferenceCore
        from cutie.utils.get_default_model import get_default_model
        
        cutie_model = get_default_model()
        self._model = InferenceCore(cutie_model, cfg=cutie_model.cfg)
        self._model.max_internal_size = -1  # We handle resizing ourselves
        self._model.top_k = self.cfg.top_k
        self._model.mem_every = self.cfg.mem_every
        print(f"[CutieTracker] Model loaded in {(time.time() - t0)*1000:.0f}ms")
                
    def initialize(self, rgb: np.ndarray, mask: np.ndarray, object_id: int = 1) -> None:
        self._lazy_load()
        
        self._orig_h, self._orig_w = rgb.shape[:2]
        
        # Resize for memory efficiency
        max_side = 480
        scale = max_side / max(self._orig_h, self._orig_w)
        if scale < 1.0:
            self._new_h, self._new_w = int(self._orig_h * scale), int(self._orig_w * scale)
            rgb_small = cv2.resize(rgb, (self._new_w, self._new_h), interpolation=cv2.INTER_LINEAR)
            mask_small = cv2.resize(mask.astype(np.uint8), (self._new_w, self._new_h), interpolation=cv2.INTER_NEAREST)
        else:
            self._new_h, self._new_w = self._orig_h, self._orig_w
            rgb_small = rgb
            mask_small = mask.astype(np.uint8)
            scale = 1.0
        
        self._scale = scale
        print(f"[CutieTracker] Resized {self._orig_h}x{self._orig_w} -> {self._new_h}x{self._new_w}")
        
        # Cutie expects: image (3, H, W), mask (H, W) - NO batch dimension
        image = torch.from_numpy(np.ascontiguousarray(rgb_small)).permute(2, 0, 1).float() / 255.0
        image = image.to(self.device)  # (3, H, W)
        
        mask_labels = np.where(mask_small > 0, object_id, 0).astype(np.int64)
        msk = torch.from_numpy(np.ascontiguousarray(mask_labels)).to(self.device)  # (H, W)
        
        print(f"[CutieTracker DEBUG] image: {image.shape}, mask: {msk.shape}")
        
        self._model.clear_memory()
        
        # Key: pass objects_in_mask to tell Cutie which object IDs are present
        with torch.inference_mode():
            self._model.step(image, msk, objects=[object_id])
        
        self._initialized = True
        self._frame_count = 1
        self._object_id = object_id
        print(f"[CutieTracker] Initialized OK, mask area={int(mask.sum())}")

    def track(self, rgb: np.ndarray) -> CutieResult:
        if not self._initialized:
            raise RuntimeError("Call initialize() before track()")
        
        t0 = time.time()
        
        # Resize
        if self._scale < 1.0:
            rgb_small = cv2.resize(rgb, (self._new_w, self._new_h), interpolation=cv2.INTER_LINEAR)
        else:
            rgb_small = rgb
        
        # (3, H, W) - no batch dim
        image = torch.from_numpy(np.ascontiguousarray(rgb_small)).permute(2, 0, 1).float() / 255.0
        image = image.to(self.device)
        
        with torch.inference_mode():
            prob = self._model.step(image)
        
        # prob shape: (num_objects+1, H, W) where 0 is background
        prob_obj = prob[self._object_id].cpu().numpy()
        mask_small = (prob_obj > 0.5).astype(np.uint8)
        
        # Resize back to original
        if self._scale < 1.0:
            mask_full = cv2.resize(mask_small, (self._orig_w, self._orig_h), interpolation=cv2.INTER_NEAREST)
            prob_full = cv2.resize(prob_obj, (self._orig_w, self._orig_h), interpolation=cv2.INTER_LINEAR)
        else:
            mask_full = mask_small
            prob_full = prob_obj
        
        # Bbox
        ys, xs = np.where(mask_full > 0)
        if len(xs) > 0:
            bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        else:
            bbox = (0, 0, 0, 0)
        
        self._frame_count += 1
        elapsed_ms = (time.time() - t0) * 1000
        
        return CutieResult(
            mask=mask_full.astype(bool),
            prob=prob_full.astype(np.float32),
            bbox_xyxy=bbox,
            area=int(mask_full.sum()),
            elapsed_ms=elapsed_ms,
        )
        
    def reset(self) -> None:
        if self._model is not None:
            self._model.clear_memory()
        self._initialized = False
        self._frame_count = 0
        
    @property
    def is_initialized(self) -> bool:
        return self._initialized
    
    @property
    def frame_count(self) -> int:
        return self._frame_count

# =============================================================================
# RAFT Optical Flow fallback (if Cutie is too slow)
# =============================================================================

@dataclass
class RAFTConfig:
    """Configuration for RAFT optical flow."""
    model: str = "raft-things"  # or "raft-sintel", "raft-small"
    device: str = "cuda"
    iters: int = 12  # More iters = more accurate but slower


@dataclass
class FlowResult:
    """Result from optical flow tracking."""
    bbox_xyxy: tuple[int, int, int, int]
    center_displacement: tuple[float, float]  # (dx, dy) in pixels
    elapsed_ms: float


class RAFTTracker:
    """
    Optical flow based tracking using RAFT.
    Faster than Cutie (~35ms) but less robust to occlusion.
    
    Install: pip install torchvision  (RAFT is included)
    """
    
    def __init__(self, cfg: Optional[RAFTConfig] = None):
        self.cfg = cfg or RAFTConfig()
        self.device = torch.device(self.cfg.device)
        self._model = None
        self._prev_rgb = None
        self._prev_bbox = None
        
    def _lazy_load(self) -> None:
        if self._model is not None:
            return
            
        print("[RAFTTracker] Loading model...")
        from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
        
        self._model = raft_large(weights=Raft_Large_Weights.DEFAULT)
        self._model = self._model.to(self.device).eval()
        self._transforms = Raft_Large_Weights.DEFAULT.transforms()
        
    def initialize(self, rgb: np.ndarray, bbox_xyxy: tuple[int, int, int, int]) -> None:
        """Initialize with first frame and bounding box."""
        self._lazy_load()
        self._prev_rgb = rgb.copy()
        self._prev_bbox = bbox_xyxy
        
    def track(self, rgb: np.ndarray) -> FlowResult:
        """Track object to new frame using optical flow."""
        if self._prev_rgb is None:
            raise RuntimeError("Call initialize() first")
            
        t0 = time.time()
        
        # Convert to torch
        prev_t = torch.from_numpy(self._prev_rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)
        curr_t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        # Apply transforms
        prev_t, curr_t = self._transforms(prev_t, curr_t)
        
        # Compute flow
        with torch.inference_mode():
            flow = self._model(prev_t, curr_t, num_flow_updates=self.cfg.iters)[-1]
        
        flow_np = flow[0].permute(1, 2, 0).cpu().numpy()  # (H, W, 2)
        
        # Get average flow in bbox region
        x0, y0, x1, y1 = self._prev_bbox
        roi_flow = flow_np[y0:y1, x0:x1]
        mean_flow = roi_flow.mean(axis=(0, 1))  # (dx, dy)
        
        # Update bbox
        dx, dy = mean_flow
        new_bbox = (
            int(x0 + dx), int(y0 + dy),
            int(x1 + dx), int(y1 + dy)
        )
        
        # Update state
        self._prev_rgb = rgb.copy()
        self._prev_bbox = new_bbox
        
        elapsed_ms = (time.time() - t0) * 1000
        
        return FlowResult(
            bbox_xyxy=new_bbox,
            center_displacement=(float(dx), float(dy)),
            elapsed_ms=elapsed_ms,
        )
    
    def reset(self) -> None:
        self._prev_rgb = None
        self._prev_bbox = None