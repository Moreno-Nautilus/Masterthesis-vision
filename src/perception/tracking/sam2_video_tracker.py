"""SAM2-based streaming video tracker.

Mirrors the CutieTracker API (`initialize` / `track` / `reset`) so the
runner can swap between Cutie and SAM2 via a flag. Internally this uses
SAM2's image predictor with the previous-frame low-resolution mask as a
mask prompt — that is the streaming-friendly path through SAM2 (the
proper SAM2VideoPredictor wants an offline frame folder, which doesn't
fit our ROS streaming pipeline).

The track result carries an `occlusion_score` derived from the mid-prob
band of the soft mask (A6): the larger the band of pixels with
0.3 < p < 0.7, the more uncertain the segmentation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch


@dataclass
class SAM2VideoConfig:
    repo_root: str = "external/sam2"
    checkpoint: str = "external/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
    model_cfg: str = "configs/sam2.1/sam2.1_hiera_b+.yaml"
    device: str = "cuda"
    use_bfloat16: bool = True
    # streaming behaviour
    max_internal_size: Optional[int] = 480
    # how often to reseed the mask prompt with a tighter crop around the
    # last good mask (B5). Set <= 0 to disable.
    memory_crop_padding_px: int = 0
    # occlusion-score band on the soft probability map
    occ_band_low: float = 0.30
    occ_band_high: float = 0.70


@dataclass
class SAM2VideoResult:
    mask: np.ndarray            # (H, W) bool
    prob: np.ndarray            # (H, W) float32 in [0,1]
    bbox_xyxy: tuple[int, int, int, int]
    area: int
    elapsed_ms: float
    occlusion_score: float      # fraction of pixels in mid-prob band


class SAM2VideoTracker:
    def __init__(self, cfg: Optional[SAM2VideoConfig] = None) -> None:
        self.cfg = cfg or SAM2VideoConfig()
        self.device = torch.device(
            self.cfg.device if (self.cfg.device == "cuda" and torch.cuda.is_available()) else "cpu"
        )
        self._predictor = None
        self._initialized = False
        self._frame_count = 0
        self._last_low_res_mask: Optional[np.ndarray] = None
        self._last_full_mask: Optional[np.ndarray] = None
        self._scale = 1.0
        self._orig_h = 0
        self._orig_w = 0
        self._new_h = 0
        self._new_w = 0
        self._object_id = 1

    def _ensure_repo_on_path(self) -> None:
        import sys
        repo_root = Path(self.cfg.repo_root).resolve()
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

    def _lazy_load(self) -> None:
        if self._predictor is not None:
            return
        self._ensure_repo_on_path()
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        model = build_sam2(
            config_file=self.cfg.model_cfg,
            ckpt_path=str(Path(self.cfg.checkpoint).resolve()),
            device=self.device.type,
        )
        self._predictor = SAM2ImagePredictor(model)

    def _resize_for_streaming(self, rgb: np.ndarray) -> np.ndarray:
        if self._scale >= 1.0:
            return rgb
        return cv2.resize(rgb, (self._new_w, self._new_h), interpolation=cv2.INTER_LINEAR)

    def _autocast_ctx(self):
        import contextlib
        if self.device.type == "cuda" and self.cfg.use_bfloat16:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def initialize(self, rgb: np.ndarray, mask: np.ndarray, object_id: int = 1) -> None:
        self._lazy_load()

        self._orig_h, self._orig_w = rgb.shape[:2]
        max_side = self.cfg.max_internal_size or 0
        if max_side > 0:
            scale = max_side / max(self._orig_h, self._orig_w)
        else:
            scale = 1.0
        if scale < 1.0:
            self._new_h = int(self._orig_h * scale)
            self._new_w = int(self._orig_w * scale)
            rgb_small = cv2.resize(rgb, (self._new_w, self._new_h), interpolation=cv2.INTER_LINEAR)
            mask_small = cv2.resize(mask.astype(np.uint8), (self._new_w, self._new_h),
                                    interpolation=cv2.INTER_NEAREST).astype(bool)
        else:
            self._new_h, self._new_w = self._orig_h, self._orig_w
            rgb_small = rgb
            mask_small = mask.astype(bool)
            scale = 1.0
        self._scale = scale
        self._object_id = int(object_id)

        with torch.inference_mode(), self._autocast_ctx():
            self._predictor.set_image(rgb_small)
            # bbox prompt from the seed mask; SAM2 returns a low-res logits
            # map we keep as the "memory" for next frame.
            ys, xs = np.where(mask_small)
            if len(xs) == 0:
                raise ValueError("SAM2VideoTracker.initialize received an empty mask")
            box = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)
            masks, scores, low_res = self._predictor.predict(
                box=box[None, :], multimask_output=False, return_logits=True,
            )
        self._last_low_res_mask = np.asarray(low_res[0], dtype=np.float32)  # (1, h, w) or (h, w)
        if self._last_low_res_mask.ndim == 2:
            self._last_low_res_mask = self._last_low_res_mask[None, :, :]
        self._last_full_mask = mask.astype(bool)

        self._initialized = True
        self._frame_count = 1

    def _maybe_memory_crop(self, rgb_small: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int] | None]:
        pad = int(self.cfg.memory_crop_padding_px)
        if pad <= 0 or self._last_full_mask is None:
            return rgb_small, None
        # The last_full_mask is in original resolution; map it to the
        # internal resolution.
        if self._scale < 1.0:
            mask_small = cv2.resize(self._last_full_mask.astype(np.uint8),
                                    (self._new_w, self._new_h),
                                    interpolation=cv2.INTER_NEAREST).astype(bool)
        else:
            mask_small = self._last_full_mask
        ys, xs = np.where(mask_small)
        if len(xs) == 0:
            return rgb_small, None
        x0 = max(0, int(xs.min()) - pad)
        y0 = max(0, int(ys.min()) - pad)
        x1 = min(self._new_w, int(xs.max()) + 1 + pad)
        y1 = min(self._new_h, int(ys.max()) + 1 + pad)
        if x1 - x0 < 16 or y1 - y0 < 16:
            return rgb_small, None
        return rgb_small[y0:y1, x0:x1], (x0, y0, x1, y1)

    def _build_mask_prior(
        self,
        crop_box: tuple[int, int, int, int] | None,
        low_res_hw: tuple[int, int],
    ) -> np.ndarray:
        """Rebuild the SAM2 mask_input prior in the *current* image frame.

        The previous frame's low-res logits are tied to the previous crop
        coordinates; when crop_box changes frame-to-frame, reusing them
        feeds SAM2 a prior in mismatched coordinates. We instead resample
        ``self._last_full_mask`` into the current crop (or full rgb_small
        when crop_box is None), then convert to a logit-style prior at
        SAM2's low-res input shape.
        """
        if self._scale < 1.0:
            mask_small = cv2.resize(
                self._last_full_mask.astype(np.uint8),
                (self._new_w, self._new_h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        else:
            mask_small = self._last_full_mask
        if crop_box is None:
            region = mask_small
        else:
            x0, y0, x1, y1 = crop_box
            region = mask_small[y0:y1, x0:x1]
        h_low, w_low = low_res_hw
        region_low = cv2.resize(
            region.astype(np.uint8), (w_low, h_low),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        # ±10 maps through sigmoid to ~{0,1}, matching SAM/SAM2 prior conventions.
        logits = np.where(region_low, 10.0, -10.0).astype(np.float32)
        return logits[None, :, :]

    def track(self, rgb: np.ndarray) -> SAM2VideoResult:
        if not self._initialized:
            raise RuntimeError("Call initialize() before track()")
        t0 = time.time()

        rgb_small = self._resize_for_streaming(rgb)
        crop, crop_box = self._maybe_memory_crop(rgb_small)

        with torch.inference_mode(), self._autocast_ctx():
            self._predictor.set_image(crop)
            # Re-prompt SAM2 with a prior aligned to the current crop frame.
            # (Reusing self._last_low_res_mask directly would break when
            # crop_box changes between frames — see _build_mask_prior.)
            low_res_hw = (
                self._last_low_res_mask.shape[-2],
                self._last_low_res_mask.shape[-1],
            )
            mask_input = self._build_mask_prior(crop_box, low_res_hw)
            masks, scores, low_res = self._predictor.predict(
                mask_input=mask_input,
                multimask_output=False,
                return_logits=True,
            )
        # masks is (1, H_crop, W_crop) bool; logits is (1, h, w) float
        crop_mask = np.asarray(masks[0]).astype(bool)
        crop_logits = np.asarray(low_res[0], dtype=np.float32)
        if crop_logits.ndim == 2:
            crop_logits = crop_logits[None, :, :]
        # Convert logits -> prob via sigmoid for the occlusion-score band.
        # SAM2 returns logits centered around 0; sigmoid is cheap.
        crop_prob = 1.0 / (1.0 + np.exp(-crop_logits[0]))
        # Resize crop prob back up to full crop H,W (predictor returns
        # logits at the model's internal resolution).
        crop_prob_full = cv2.resize(
            crop_prob, (crop.shape[1], crop.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

        # Reassemble into the streaming-resolution canvas.
        small_mask = np.zeros((self._new_h, self._new_w), dtype=bool)
        small_prob = np.zeros((self._new_h, self._new_w), dtype=np.float32)
        if crop_box is None:
            small_mask = crop_mask
            small_prob = crop_prob_full
        else:
            x0, y0, x1, y1 = crop_box
            small_mask[y0:y1, x0:x1] = crop_mask
            small_prob[y0:y1, x0:x1] = crop_prob_full

        # Resize back to original.
        if self._scale < 1.0:
            mask_full = cv2.resize(small_mask.astype(np.uint8),
                                   (self._orig_w, self._orig_h),
                                   interpolation=cv2.INTER_NEAREST).astype(bool)
            prob_full = cv2.resize(small_prob, (self._orig_w, self._orig_h),
                                   interpolation=cv2.INTER_LINEAR)
        else:
            mask_full = small_mask
            prob_full = small_prob

        ys, xs = np.where(mask_full)
        if len(xs) > 0:
            bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        else:
            bbox = (0, 0, 0, 0)

        # A6: occlusion score = fraction of pixels in [low, high] band.
        lo = float(self.cfg.occ_band_low)
        hi = float(self.cfg.occ_band_high)
        in_band = (prob_full >= lo) & (prob_full <= hi)
        # Normalize over the bbox area (or fall back to total pixels if empty)
        denom = max(1, int(mask_full.sum()) if int(mask_full.sum()) > 0 else prob_full.size)
        occ = float(in_band.sum()) / float(denom)

        # Update memory: keep the latest low-res logits (in crop frame).
        self._last_low_res_mask = crop_logits
        self._last_full_mask = mask_full

        self._frame_count += 1
        elapsed_ms = (time.time() - t0) * 1000.0

        return SAM2VideoResult(
            mask=mask_full,
            prob=prob_full,
            bbox_xyxy=bbox,
            area=int(mask_full.sum()),
            elapsed_ms=elapsed_ms,
            occlusion_score=occ,
        )

    def reset(self) -> None:
        self._initialized = False
        self._frame_count = 0
        self._last_low_res_mask = None
        self._last_full_mask = None

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def frame_count(self) -> int:
        return self._frame_count
