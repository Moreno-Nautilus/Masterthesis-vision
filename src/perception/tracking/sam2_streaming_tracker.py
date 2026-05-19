"""True-streaming wrapper around SAM2VideoPredictor.

The official SAM2VideoPredictor.init_state expects an offline clip (a folder
of JPEGs or an .mp4) loaded upfront. This wrapper bypasses init_state and
drives the predictor one frame at a time, keeping the full memory bank
across calls. Compared to the SAM2ImagePredictor-with-mask-prompt path in
sam2_video_tracker.py, this exposes the real memory-attention behaviour
(N prior frames' encoded features, not a length-1 mask prior).

Public API mirrors CutieTracker / SAM2VideoTracker:
    tr.initialize(rgb, mask, object_id=1)
    res = tr.track(rgb)
    tr.reset()
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from .sam2_video_tracker import SAM2VideoResult


@dataclass
class SAM2StreamingConfig:
    repo_root: str = "external/sam2"
    checkpoint: str = "external/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
    model_cfg: str = "configs/sam2.1/sam2.1_hiera_b+.yaml"
    device: str = "cuda"
    use_bfloat16: bool = True
    # occlusion-score band on the soft probability map (matches image-predictor path)
    occ_band_low: float = 0.30
    occ_band_high: float = 0.70
    # cap on how many past per-object outputs we keep around. The model's
    # memory attention only attends to the most recent few entries anyway
    # (sam2 default num_maskmem=7); keeping a small sliding window avoids
    # unbounded host/GPU memory growth on long streams.
    max_memory_frames: int = 16


class SAM2StreamingTracker:
    def __init__(self, cfg: Optional[SAM2StreamingConfig] = None) -> None:
        self.cfg = cfg or SAM2StreamingConfig()
        self.device = torch.device(
            self.cfg.device if (self.cfg.device == "cuda" and torch.cuda.is_available()) else "cpu"
        )
        self._predictor = None
        self._state: Optional[dict] = None
        self._frame_count = 0
        self._initialized = False
        self._object_id = 1
        self._orig_h = 0
        self._orig_w = 0
        self._img_mean: Optional[torch.Tensor] = None
        self._img_std: Optional[torch.Tensor] = None

    def _ensure_repo_on_path(self) -> None:
        import sys
        repo_root = Path(self.cfg.repo_root).resolve()
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

    def _lazy_load(self) -> None:
        if self._predictor is not None:
            return
        self._ensure_repo_on_path()
        from sam2.build_sam import build_sam2_video_predictor

        self._predictor = build_sam2_video_predictor(
            config_file=self.cfg.model_cfg,
            ckpt_path=str(Path(self.cfg.checkpoint).resolve()),
            device=self.device.type,
        )
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=self.device)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=self.device)
        self._img_mean = mean[:, None, None]
        self._img_std = std[:, None, None]

    def _autocast_ctx(self):
        import contextlib
        if self.device.type == "cuda" and self.cfg.use_bfloat16:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def _preprocess(self, rgb: np.ndarray) -> torch.Tensor:
        """RGB uint8 (H,W,3) -> normalised float tensor (3, S, S) on device."""
        size = int(self._predictor.image_size)
        resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
        t = torch.from_numpy(resized).to(self.device).permute(2, 0, 1).float() / 255.0
        t = (t - self._img_mean) / self._img_std
        return t

    def _build_state(self, rgb_t: torch.Tensor) -> dict:
        return {
            "images": [rgb_t],
            "num_frames": 1,
            "offload_video_to_cpu": False,
            "offload_state_to_cpu": False,
            "video_height": self._orig_h,
            "video_width": self._orig_w,
            "device": self.device,
            "storage_device": self.device,
            "point_inputs_per_obj": {},
            "mask_inputs_per_obj": {},
            "cached_features": {},
            "constants": {},
            "obj_id_to_idx": OrderedDict(),
            "obj_idx_to_id": OrderedDict(),
            "obj_ids": [],
            "output_dict_per_obj": {},
            "temp_output_dict_per_obj": {},
            "frames_tracked_per_obj": {},
        }

    def initialize(self, rgb: np.ndarray, mask: np.ndarray, object_id: int = 1) -> None:
        self._lazy_load()
        self._orig_h, self._orig_w = rgb.shape[:2]
        self._object_id = int(object_id)

        rgb_t = self._preprocess(rgb)
        state = self._build_state(rgb_t)
        self._state = state

        with torch.inference_mode(), self._autocast_ctx():
            # Mask is at original resolution; add_new_mask resizes internally.
            self._predictor.add_new_mask(
                inference_state=state,
                frame_idx=0,
                obj_id=self._object_id,
                mask=torch.from_numpy(mask.astype(bool)),
            )
            # Encode the seed mask into memory (would otherwise happen lazily
            # at the start of propagate_in_video).
            self._predictor.propagate_in_video_preflight(state)

        self._frame_count = 1
        self._initialized = True

    def _track_one_frame(self, frame_idx: int) -> torch.Tensor:
        """Run one forward step using the model's memory bank. Returns logits
        at original video resolution, shape (B, 1, H, W)."""
        state = self._state
        batch_size = self._predictor._get_obj_num(state)
        pred_masks_per_obj = [None] * batch_size
        for obj_idx in range(batch_size):
            obj_output_dict = state["output_dict_per_obj"][obj_idx]
            current_out, pred_masks = self._predictor._run_single_frame_inference(
                inference_state=state,
                output_dict=obj_output_dict,
                frame_idx=frame_idx,
                batch_size=1,
                is_init_cond_frame=False,
                point_inputs=None,
                mask_inputs=None,
                reverse=False,
                run_mem_encoder=True,
            )
            obj_output_dict["non_cond_frame_outputs"][frame_idx] = current_out
            state["frames_tracked_per_obj"][obj_idx][frame_idx] = {"reverse": False}
            pred_masks_per_obj[obj_idx] = pred_masks

        all_pred_masks = (
            pred_masks_per_obj[0] if batch_size == 1 else torch.cat(pred_masks_per_obj, dim=0)
        )
        _, video_res_masks = self._predictor._get_orig_video_res_output(state, all_pred_masks)
        return video_res_masks

    def _prune_state(self, current_frame_idx: int) -> None:
        keep = int(max(1, self.cfg.max_memory_frames))
        cutoff = current_frame_idx - keep
        if cutoff < 1:
            return
        state = self._state
        for obj_idx, obj_output_dict in state["output_dict_per_obj"].items():
            ncf = obj_output_dict["non_cond_frame_outputs"]
            for k in [k for k in ncf.keys() if k < cutoff]:
                del ncf[k]
            ft = state["frames_tracked_per_obj"].get(obj_idx, {})
            for k in [k for k in ft.keys() if k < cutoff and k != 0]:
                del ft[k]
        # Drop old image tensors we no longer need.
        for i in range(cutoff):
            if i < len(state["images"]) and state["images"][i] is not None:
                state["images"][i] = None

    def track(self, rgb: np.ndarray) -> SAM2VideoResult:
        if not self._initialized:
            raise RuntimeError("Call initialize() before track()")
        t0 = time.time()

        rgb_t = self._preprocess(rgb)
        state = self._state
        frame_idx = self._frame_count
        # Grow the images list to index frame_idx.
        while len(state["images"]) <= frame_idx:
            state["images"].append(None)
        state["images"][frame_idx] = rgb_t
        state["num_frames"] = frame_idx + 1

        with torch.inference_mode(), self._autocast_ctx():
            video_res_logits = self._track_one_frame(frame_idx)

        # video_res_logits: (B, 1, H_orig, W_orig). One object → B=1.
        logits = video_res_logits[0, 0].detach().float().cpu().numpy()
        prob = 1.0 / (1.0 + np.exp(-logits))
        mask = prob > 0.5
        ys, xs = np.where(mask)
        if len(xs) == 0:
            bbox = (0, 0, 0, 0)
            area = 0
        else:
            bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
            area = int(mask.sum())

        band = (prob > self.cfg.occ_band_low) & (prob < self.cfg.occ_band_high)
        denom = max(int(mask.sum() + band.sum()), 1)
        occlusion = float(band.sum()) / float(denom)

        self._frame_count = frame_idx + 1
        self._prune_state(frame_idx)

        return SAM2VideoResult(
            mask=mask,
            prob=prob.astype(np.float32),
            bbox_xyxy=bbox,
            area=area,
            elapsed_ms=(time.time() - t0) * 1000.0,
            occlusion_score=occlusion,
        )

    def reset(self) -> None:
        if self._predictor is not None and self._state is not None:
            try:
                self._predictor.reset_state(self._state)
            except Exception:
                pass
        self._state = None
        self._frame_count = 0
        self._initialized = False
