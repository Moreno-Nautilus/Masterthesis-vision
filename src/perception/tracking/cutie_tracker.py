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
        # Multi-object session state (one CutieTracker per CAMERA tracking all
        # objects in a single forward). Maps track_id (str) -> Cutie integer
        # object id. Empty when used in single-object mode.
        self._objects: dict[str, int] = {}
        self._next_cutie_id = 1

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
        
        # Resize for memory efficiency.
        max_side = self.cfg.max_internal_size
        scale = (
            float(max_side) / float(max(self._orig_h, self._orig_w))
            if max_side is not None and max_side > 0 else 1.0
        )
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

    # ------------------------------------------------------------------
    # Multi-object API: one Cutie session per camera, tracking ALL objects
    # in a single forward. Cutie supports this natively (one object channel
    # per registered mask). add_object() memorizes a new object incrementally
    # (existing objects keep their memory); track_multi() runs ONE forward and
    # splits the multi-object output back into per-object masks keyed by
    # track_id. This replaces N separate (camera, object) sessions, so the
    # per-tick Cutie call count drops from N_objects to 1 per camera.
    # ------------------------------------------------------------------

    def _setup_scale(self, rgb: np.ndarray) -> None:
        """Compute the internal working resolution from the first frame."""
        self._orig_h, self._orig_w = rgb.shape[:2]
        max_side = self.cfg.max_internal_size
        scale = (
            float(max_side) / float(max(self._orig_h, self._orig_w))
            if max_side is not None and max_side > 0 else 1.0
        )
        if scale < 1.0:
            self._new_h, self._new_w = int(self._orig_h * scale), int(self._orig_w * scale)
        else:
            self._new_h, self._new_w = self._orig_h, self._orig_w
            scale = 1.0
        self._scale = scale

    def _resize_rgb(self, rgb: np.ndarray) -> np.ndarray:
        if self._scale < 1.0:
            return cv2.resize(rgb, (self._new_w, self._new_h), interpolation=cv2.INTER_LINEAR)
        return rgb

    def has_object(self, track_id: str) -> bool:
        return track_id in self._objects

    @property
    def tracked_object_ids(self) -> set[str]:
        return set(self._objects.keys())

    def add_object(self, track_id: str, rgb: np.ndarray, mask: np.ndarray) -> int:
        """
        Register and memorize a new object in this multi-object session.

        The first object sets the working resolution and clears memory; later
        objects are added incrementally (one Cutie step), so already-tracked
        objects are segmented from memory and keep tracking uninterrupted.
        Returns the Cutie integer object id assigned to this track_id.
        """
        self._lazy_load()

        if not self._objects:
            # First object: establish working resolution and a clean memory.
            self._setup_scale(rgb)
            self._model.clear_memory()
            self._frame_count = 0

        if track_id in self._objects:
            return self._objects[track_id]

        cutie_id = self._next_cutie_id
        self._next_cutie_id += 1
        self._objects[track_id] = cutie_id

        rgb_small = self._resize_rgb(rgb)
        if self._scale < 1.0:
            mask_small = cv2.resize(
                mask.astype(np.uint8), (self._new_w, self._new_h),
                interpolation=cv2.INTER_NEAREST,
            )
        else:
            mask_small = mask.astype(np.uint8)

        image = torch.from_numpy(np.ascontiguousarray(rgb_small)).permute(2, 0, 1).float() / 255.0
        image = image.to(self.device)

        mask_labels = np.where(mask_small > 0, cutie_id, 0).astype(np.int64)
        msk = torch.from_numpy(np.ascontiguousarray(mask_labels)).to(self.device)

        with torch.inference_mode():
            self._model.step(image, msk, objects=[cutie_id])

        self._initialized = True
        self._frame_count += 1
        return cutie_id

    def remove_object(self, track_id: str) -> None:
        """Drop an object from this session (e.g. when its track is lost)."""
        cutie_id = self._objects.pop(track_id, None)
        if cutie_id is None:
            return
        if self._model is not None:
            try:
                self._model.delete_objects([cutie_id])
            except Exception:
                pass
        if not self._objects:
            self._initialized = False

    def track_multi(self, rgb: np.ndarray) -> dict[str, CutieResult]:
        """
        Run ONE Cutie forward for all registered objects and split the
        multi-object output into per-object masks keyed by track_id.
        """
        if not self._objects or not self._initialized:
            return {}

        t0 = time.time()

        rgb_small = self._resize_rgb(rgb)
        image = torch.from_numpy(np.ascontiguousarray(rgb_small)).permute(2, 0, 1).float() / 255.0
        image = image.to(self.device)

        with torch.inference_mode():
            prob = self._model.step(image)

        # prob: (num_objects+1, H, W); channel 0 is background, channels 1..N
        # are ordered by Cutie tmp_id. One device->host copy, then index.
        prob_np = prob.detach().to("cpu").numpy()
        self._frame_count += 1
        elapsed_ms = (time.time() - t0) * 1000.0

        om = self._model.object_manager
        out: dict[str, CutieResult] = {}
        for track_id, cutie_id in self._objects.items():
            try:
                tmp_id = om.find_tmp_by_id(cutie_id)
            except Exception:
                continue
            if tmp_id < 1 or tmp_id >= prob_np.shape[0]:
                continue

            prob_obj = prob_np[tmp_id]
            mask_small = (prob_obj > 0.5).astype(np.uint8)

            if self._scale < 1.0:
                mask_full = cv2.resize(
                    mask_small, (self._orig_w, self._orig_h),
                    interpolation=cv2.INTER_NEAREST,
                )
                prob_full = cv2.resize(
                    prob_obj, (self._orig_w, self._orig_h),
                    interpolation=cv2.INTER_LINEAR,
                )
            else:
                mask_full = mask_small
                prob_full = prob_obj

            ys, xs = np.where(mask_full > 0)
            if len(xs) > 0:
                bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
            else:
                bbox = (0, 0, 0, 0)

            out[track_id] = CutieResult(
                mask=mask_full.astype(bool),
                prob=prob_full.astype(np.float32),
                bbox_xyxy=bbox,
                area=int(mask_full.sum()),
                # Shared forward time; per-object split overhead is negligible.
                elapsed_ms=elapsed_ms,
            )
        return out

    def reset(self) -> None:
        # Full reset: clear memory AND object identities so a reused session
        # (e.g. after a re-init) does not carry stale Cutie object ids. The
        # network weights stay loaded.
        if self._model is not None:
            try:
                from cutie.inference.object_manager import ObjectManager
                from cutie.inference.memory_manager import MemoryManager
                self._model.object_manager = ObjectManager()
                self._model.memory = MemoryManager(
                    cfg=self._model.cfg, object_manager=self._model.object_manager
                )
                self._model.curr_ti = -1
                self._model.last_mem_ti = 0
                self._model.last_mask = None
            except Exception:
                self._model.clear_memory()
        self._initialized = False
        self._frame_count = 0
        self._objects = {}
        self._next_cutie_id = 1

    @property
    def is_initialized(self) -> bool:
        return self._initialized
    
    @property
    def frame_count(self) -> int:
        return self._frame_count
