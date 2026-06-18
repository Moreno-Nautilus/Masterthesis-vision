from __future__ import annotations

import logging
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import gc

import numpy as np
import torch


@contextmanager
def _force_math_sdp_attention():
    """Force PyTorch's *math* scaled-dot-product-attention backend for the
    enclosed forward passes, disabling the flash / mem-efficient kernels.

    SAM2's hiera image encoder and mask decoder call F.scaled_dot_product_attention
    with no backend guard, so PyTorch auto-selects flash/mem-efficient kernels.
    On some GPU/driver/toolkit combinations (here: RTX 4090 with a CUDA
    Toolkit-newer-than-Driver skew) those fused kernels intermittently emit
    garbage — a freshly built model then produces empty/near-empty masks on every
    forward (the 'cursed build'), randomly ~half the time and independent of
    bf16/fp32. The math backend is numerically plain and deterministic; it is a
    bit slower but SAM only runs at init, so the cost is irrelevant. Falls back
    gracefully across torch SDPA API versions."""
    try:
        # torch >= 2.1 preferred API
        from torch.nn.attention import sdpa_kernel, SDPBackend
        ctx = sdpa_kernel([SDPBackend.MATH])
    except Exception:
        try:
            # older torch fallback
            ctx = torch.backends.cuda.sdp_kernel(
                enable_flash=False,
                enable_mem_efficient=False,
                enable_math=True,
            )
        except Exception:
            # SDPA backend control unavailable: run unguarded rather than break.
            ctx = nullcontext()

    with ctx:
        yield


@contextmanager
def _suppress_sam_info_logs():
    """Silence SAM2's per-image `[set_image()]` logging.info spam (3 lines per
    forward) while keeping warnings/errors. SAM2 logs through the root logger;
    we raise its effective level to WARNING for the duration of set_image and
    restore it afterward. The pipeline's own output uses print / the ROS logger,
    not the `logging` module, so this does not touch it."""
    root = logging.getLogger()
    prev_level = root.level
    if prev_level < logging.WARNING:
        root.setLevel(logging.WARNING)
    try:
        yield
    finally:
        root.setLevel(prev_level)


@dataclass
class SAMMaskCandidate:
    mask: np.ndarray  # (H, W) bool
    score: float  # SAM IoU / stability score
    bbox_xyxy: tuple[int, int, int, int]
    area: int
    crop_rgb: np.ndarray | None = None
    # Box-proposer confidence for prompted masks.
    prompt_score: float | None = None


@dataclass
class SAMSegmenterConfig:
    repo_root: str = "external/sam2"
    checkpoint: str = "external/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
    model_cfg: str = "configs/sam2.1/sam2.1_hiera_b+.yaml"

    device: str = "cuda"
    use_bfloat16: bool = True

    # proposal filtering
    min_mask_area: int = 800
    max_mask_area_ratio: float = 0.06
    min_bbox_side_px: int = 20

    # image resizing before mask generation
    max_image_side: int | None = 1024

    # whether to keep crop RGB for downstream DINO
    attach_rgb_crops: bool = True

    # max_side / min_side
    max_aspect_ratio: float = 4.5

    # HSV shadow rejection
    shadow_filter_enabled: bool = False #TRUE
    min_saturation: int = 60
    min_value: int = 80
    max_value_for_low_sat: int = 220


class SAMSegmenter:
    """SAM2 wrapper for box-prompted mask proposals."""

    def __init__(self, cfg: SAMSegmenterConfig | None = None) -> None:
        self.cfg = cfg or SAMSegmenterConfig()
        self.device = torch.device(
            self.cfg.device if self.cfg.device == "cuda" and torch.cuda.is_available() else "cpu"
        )

        # Build the model plus the box-prompt predictor.
        model = self._build_model()
        self._model = model
        self._predictor = self._build_predictor(model)
        self._log_param_health("after-build")

    def _rebuild_model(self) -> bool:
        """Recreate the SAM model stack for this camera after a persistent
        collapse. Returns True on a successful swap, False if the rebuild could
        not complete (in which case the existing model is left untouched).

        CRASH-SAFETY: the new model/predictor are built into locals FIRST and
        only swapped onto self once all three succeed. The previous version
        deleted self._predictor before rebuilding, so any failure during
        build_sam2() left the segmenter with NO _predictor — permanently bricking
        this camera (it is cached in sam_by_cam) and crashing every later init
        tick at set_image(). build_sam2() in particular fails at runtime because
        Hydra has a single global config search path: Cutie's pre-warm repoints
        it to external/Cutie/cutie/config after our initial build, so a later
        build_sam2() raises MissingConfigException for the sam2 configs. Keeping
        the old (still-usable) predictor on failure means the worst case is one
        skipped frame, not a dead camera."""
        print("[SAM-REBUILD] rebuilding SAM model after persistent mask collapse")
        # Build the replacement into locals first; only commit if all three succeed.
        try:
            new_model = self._build_model()
            new_predictor = self._build_predictor(new_model)
        except Exception as e:
            print(f"[SAM-REBUILD] rebuild failed, keeping existing model: {e}")
            return False

        # Swap the new stack in and free the old one.
        old_model = getattr(self, "_model", None)
        old_predictor = getattr(self, "_predictor", None)
        self._model = new_model
        self._predictor = new_predictor
        del old_model, old_predictor
        gc.collect()
        if self.device.type == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"[SAM-REBUILD] empty_cache failed during rebuild: {e}")
        self._health_logged = False
        self._log_param_health("after-rebuild")
        return True

    def _log_param_health(self, tag: str) -> None:
        """[SAM-WEIGHTS] diagnostic: count parameters AND buffers with NaN/Inf."""
        try:
            p_nan = p_inf = p_tot = 0
            for p in self._model.parameters():
                p_tot += 1
                if torch.isnan(p).any():
                    p_nan += 1
                if torch.isinf(p).any():
                    p_inf += 1
            b_nan = b_inf = b_tot = 0
            for b in self._model.buffers():
                b_tot += 1
                if b.is_floating_point():
                    if torch.isnan(b).any():
                        b_nan += 1
                    if torch.isinf(b).any():
                        b_inf += 1
            print(
                f"[SAM-WEIGHTS:{tag}] params total={p_tot} nan={p_nan} inf={p_inf} | "
                f"buffers total={b_tot} nan={b_nan} inf={b_inf}"
            )
        except Exception as e:
            print(f"[SAM-WEIGHTS:{tag}] check failed: {e}")

    def warmup(self, max_iters: int = 6) -> bool:
        # Run a synthetic red-square-on-grey box prompt until SAM returns a valid mask.
        dummy = np.full((512, 512, 3), 30, dtype=np.uint8)
        dummy[208:304, 208:304] = (220, 40, 40)
        box = np.array([[208, 208, 304, 304]], dtype=np.float32)
        for i in range(max_iters):
            try:
                cands = self.generate_from_boxes(dummy, box)
            except Exception as e:  # pragma: no cover - defensive
                print(f"[SAM-WARMUP] iter {i} raised {e!r}")
                cands = []
            if cands and np.isfinite(cands[0].score):
                print(f"[SAM-WARMUP] warm after {i + 1} forward(s)")
                return True
        print(f"[SAM-WARMUP] still cold after {max_iters} forward(s) — proceeding anyway")
        return False

    def warmup_or_rebuild(self, max_rebuilds: int = 4, warmup_iters: int = 6) -> bool:
        """Warm the model; if it stays persistently cold, rebuild a fresh model
        and retry, up to max_rebuilds times. Returns True once warm.

        Some builds are 'cursed': the model emits empty/NaN masks no matter how
        many forward passes it gets (warmup() alone can't fix them — observed as
        a whole camera that never produces a usable mask and never recovers). A
        throwaway build only shields the FIRST build, but in practice the 2nd
        build can be cursed too. Rebuilding until warm is self-correcting for
        however many leading builds are bad; later builds warm on the first try,
        so this normally costs at most one extra ~1s build. Meant to run at
        startup, where rebuilds are cheap and Hydra still resolves the sam2
        config (and _build_model re-asserts it regardless)."""
        # Warm first; only rebuild fresh models if it stays cold ("cursed" builds).
        if self.warmup(max_iters=warmup_iters):
            return True
        for attempt in range(1, max_rebuilds + 1):
            print(f"[SAM-WARMUP] cold model — rebuilding "
                  f"(attempt {attempt}/{max_rebuilds})")
            if not self._rebuild_model():
                print("[SAM-WARMUP] rebuild unavailable; cannot recover cold model")
                return False
            if self.warmup(max_iters=warmup_iters):
                return True
        print(f"[SAM-WARMUP] still cold after {max_rebuilds} rebuild(s) "
              f"— proceeding anyway")
        return False

    def _ensure_repo_on_path(self) -> None:
        repo_root = Path(self.cfg.repo_root).resolve()
        if not repo_root.exists():
            raise FileNotFoundError(f"SAM2 repo_root does not exist: {repo_root}")

        import sys
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

    def _build_model(self) -> Any:
        # Build the SAM2 model from its config + checkpoint (Hydra config re-pointed first).
        ckpt = Path(self.cfg.checkpoint).resolve()
        if not ckpt.exists():
            raise FileNotFoundError(f"SAM2 checkpoint does not exist: {ckpt}")

        self._ensure_repo_on_path()

        # delayed import so the rest of the repo remains importable
        from sam2.build_sam import build_sam2

        self._ensure_sam2_hydra_config()

        model = build_sam2(
            config_file=self.cfg.model_cfg,
            ckpt_path=str(ckpt),
            device=self.device.type,
        )
        return model

    @staticmethod
    def _ensure_sam2_hydra_config() -> None:
        """Point Hydra's global config search path at the sam2 config module,
        clearing whatever another package (e.g. Cutie) may have initialized it
        to. Safe to call repeatedly; sam2 only composes (never runs a Hydra job),
        and Cutie reloads its own config via context-managed initialize() at its
        own load time, so re-pointing here does not disturb an already-loaded
        Cutie model."""
        from hydra import initialize_config_module
        from hydra.core.global_hydra import GlobalHydra

        gh = GlobalHydra.instance()
        if gh.is_initialized():
            gh.clear()
        initialize_config_module("sam2", version_base="1.2")

    def _build_predictor(self, model: Any) -> Any:
        self._ensure_repo_on_path()
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        predictor = SAM2ImagePredictor(model)
        return predictor

    @staticmethod
    def _resize_if_needed(rgb: np.ndarray, max_side: int | None) -> tuple[np.ndarray, float]:
        # Downscale so the longest side ≤ max_side; return the image and the scale used.
        if max_side is None:
            return rgb, 1.0

        h, w = rgb.shape[:2]
        scale = min(max_side / max(h, w), 1.0)
        if scale == 1.0:
            return rgb, 1.0

        import cv2
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        rgb_small = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return rgb_small, scale

    @staticmethod
    def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
        # Tight xyxy bounding box around the mask's set pixels.
        ys, xs = np.nonzero(mask)
        if len(xs) == 0 or len(ys) == 0:
            return (0, 0, 0, 0)
        x0 = int(xs.min())
        y0 = int(ys.min())
        x1 = int(xs.max()) + 1
        y1 = int(ys.max()) + 1
        return (x0, y0, x1, y1)

    @staticmethod
    def _crop_rgb(rgb: np.ndarray, bbox_xyxy: tuple[int, int, int, int]) -> np.ndarray:
        x0, y0, x1, y1 = bbox_xyxy
        return rgb[y0:y1, x0:x1].copy()

    def _filter_mask(
        self,
        mask: np.ndarray,
        score: float,
        image_shape: tuple[int, int],
        rgb: np.ndarray | None = None,
    ) -> str:
        """Return "" if the mask passes, else a short reason code for why it was
        rejected (used by the [SAM-REJECT] diagnostic in generate_*)."""
        # Area must be within [min, ratio·image] — reject specks and whole-frame masks.
        h, w = image_shape
        area = int(mask.sum())
        if area < self.cfg.min_mask_area:
            return "area_small"

        if area > int(self.cfg.max_mask_area_ratio * h * w):
            return "area_large"

        # Reject too-thin boxes and non-finite scores.
        x0, y0, x1, y1 = self._bbox_from_mask(mask)
        bw = x1 - x0
        bh = y1 - y0
        if bw < self.cfg.min_bbox_side_px or bh < self.cfg.min_bbox_side_px:
            return "bbox_small"

        if not np.isfinite(score):
            return "bad_score"

        # Reject extreme aspect ratios (sliver masks).
        aspect_ratio = max(bw, bh) / (min(bw, bh) + 1e-6)
        if aspect_ratio > self.cfg.max_aspect_ratio:
            return "aspect"

        # Reject masks that barely fill their bounding box.
        bbox_area = bw * bh
        if bbox_area > 0:
            fill_ratio = area / bbox_area
            if fill_ratio < 0.20:
                return "fill_ratio"

        # Optional HSV check: drop low-saturation mid-value blobs (shadows).
        if self.cfg.shadow_filter_enabled and rgb is not None:
            import cv2
            mask_bool = mask.astype(bool)
            masked_pixels = rgb[mask_bool]

            if len(masked_pixels) > 10:
                pixels_rgb = masked_pixels.reshape(1, -1, 3).astype(np.uint8)
                pixels_hsv = cv2.cvtColor(pixels_rgb, cv2.COLOR_RGB2HSV)
                pixels_hsv = pixels_hsv.reshape(-1, 3)

                mean_saturation = pixels_hsv[:, 1].mean()
                mean_value = pixels_hsv[:, 2].mean()

                if mean_saturation < self.cfg.min_saturation:
                    if mean_value > self.cfg.min_value and mean_value < self.cfg.max_value_for_low_sat:
                        return "shadow"

        # For large masks, reject if they split into multiple blobs (now or after erosion).
        if area > 5000 and rgb is not None:
            import cv2
            from scipy import ndimage

            mask_uint8 = mask.astype(np.uint8)
            _, num_features = ndimage.label(mask_uint8)
            if num_features > 1:
                return "fragmented"

            eroded = cv2.erode(mask_uint8, np.ones((5, 5), np.uint8), iterations=3)
            _, num_after_erode = ndimage.label(eroded)
            if num_after_erode > 1:
                return "fragmented_erode"

        return ""

    def generate_from_boxes(
        self,
        rgb: np.ndarray,
        boxes_xyxy: np.ndarray,
        box_scores: np.ndarray | None = None,
    ) -> list[SAMMaskCandidate]:
        """Segment one mask per box prompt."""
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"rgb must be (H, W, 3), got {rgb.shape}")
        boxes = np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4)
        if boxes.shape[0] == 0:
            return []
        if box_scores is not None:
            box_scores = np.asarray(box_scores, dtype=np.float32).reshape(-1)
            if box_scores.shape[0] != boxes.shape[0]:
                raise ValueError(
                    f"box_scores length {box_scores.shape[0]} != boxes {boxes.shape[0]}"
                )

        if not getattr(self, "_health_logged", False):
            self._health_logged = True
            self._log_param_health("first-forward")

        # Resize the image and scale the boxes to match.
        rgb_small, scale = self._resize_if_needed(rgb, self.cfg.max_image_side)
        boxes_scaled = boxes * scale if scale != 1.0 else boxes
        n_boxes = len(boxes)

        # First attempt in the configured precision (bf16 by default).
        out, _reject = self._predict_boxes_once(
            rgb, rgb_small, boxes_scaled, scale, box_scores,
            use_bf16=self.cfg.use_bfloat16,
        )

        # SAM-collapse recovery: every box's mask falling below min_mask_area at
        # once is the signature of a degenerate set_image() embedding 
        if self._is_mask_collapse(out, n_boxes, _reject):
            print(f"[SAM-RETRY] from_boxes: {n_boxes} boxes collapsed "
                  f"(area_small={_reject.get('area_small', 0)}); re-encoding (bf16)")
            out, _reject = self._predict_boxes_once(
                rgb, rgb_small, boxes_scaled, scale, box_scores,
                use_bf16=self.cfg.use_bfloat16,
            )
            if self._is_mask_collapse(out, n_boxes, _reject):
                print("[SAM-RETRY] from_boxes: still collapsed; falling back to fp32")
                out, _reject = self._predict_boxes_once(
                    rgb, rgb_small, boxes_scaled, scale, box_scores,
                    use_bf16=False,
                )
            if self._is_mask_collapse(out, n_boxes, _reject):
                print("[SAM-RETRY] from_boxes: still collapsed after fp32; "
                      "skipping this camera for this cycle (recovers next cycle)")

        if _reject:
            print(f"[SAM-REJECT] from_boxes: {n_boxes} boxes -> {len(out)} kept "
                  f"| rejected: {dict(sorted(_reject.items(), key=lambda kv: -kv[1]))}")
        # Best (score, area) first.
        out.sort(key=lambda x: (x.score, x.area), reverse=True)
        return out

    @staticmethod
    def _is_mask_collapse(
        out: list[SAMMaskCandidate],
        n_boxes: int,
        reject: dict[str, int],
    ) -> bool:
        """True when the masks collapsed: nothing kept and the vast majority of
        boxes were rejected as area_small. Requires >=3 boxes so genuinely empty
        scenes (few proposals) and shadow/area_large rejections don't trigger a
        retry."""
        return (
            len(out) == 0
            and n_boxes >= 3
            and reject.get("area_small", 0) >= 0.8 * n_boxes
        )

    def _predict_boxes_once(
        self,
        rgb: np.ndarray,
        rgb_small: np.ndarray,
        boxes_scaled: np.ndarray,
        scale: float,
        box_scores: np.ndarray | None,
        use_bf16: bool,
    ) -> tuple[list[SAMMaskCandidate], dict[str, int]]:
        """One full set_image() + per-box predict + filter pass. set_image() is
        called inside so each retry re-encodes the image (a fresh embedding is the
        actual fix for a collapse)."""
        import cv2

        # Precision context (bf16 autocast on CUDA, else plain) + math-SDPA guard.
        with torch.inference_mode():
            if self.device.type == "cuda" and use_bf16:
                ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            else:
                from contextlib import nullcontext
                ctx = nullcontext()
            with ctx, _force_math_sdp_attention():
                # Encode the image once for all box prompts.
                with _suppress_sam_info_logs():
                    self._predictor.set_image(rgb_small)
                out: list[SAMMaskCandidate] = []
                _reject: dict[str, int] = {}  # [SAM-REJECT] diagnostic
                h0, w0 = rgb.shape[:2]
                # One mask per box; upscale to full res, filter, and keep the survivors.
                for idx, box in enumerate(boxes_scaled):
                    masks, scores, _ = self._predictor.predict(
                        box=box, multimask_output=False,
                    )
                    if masks is None or len(masks) == 0:
                        _reject["no_mask"] = _reject.get("no_mask", 0) + 1
                        continue
                    mask_small = np.asarray(masks[0], dtype=np.uint8)
                    if scale != 1.0:
                        mask = cv2.resize(mask_small, (w0, h0), interpolation=cv2.INTER_NEAREST) > 0
                    else:
                        mask = mask_small > 0
                    score = float(scores[0]) if scores is not None and len(scores) else 0.0
                    reason = self._filter_mask(mask, score, (h0, w0), rgb=rgb)
                    if reason:
                        _reject[reason] = _reject.get(reason, 0) + 1
                        continue
                    bbox = self._bbox_from_mask(mask)
                    area = int(mask.sum())
                    crop_rgb = self._crop_rgb(rgb, bbox) if self.cfg.attach_rgb_crops else None
                    prompt_score = (
                        float(box_scores[idx]) if box_scores is not None else None
                    )
                    out.append(SAMMaskCandidate(
                        mask=mask, score=score, bbox_xyxy=bbox, area=area,
                        crop_rgb=crop_rgb, prompt_score=prompt_score,
                    ))
        return out, _reject
