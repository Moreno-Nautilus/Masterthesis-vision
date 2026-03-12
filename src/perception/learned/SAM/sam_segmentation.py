from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass
class SAMMaskCandidate:
    mask: np.ndarray  # (H, W) bool
    score: float
    bbox_xyxy: tuple[int, int, int, int]
    area: int
    crop_rgb: np.ndarray | None = None


@dataclass
class SAMSegmenterConfig:
    repo_root: str = "external/sam2"
    checkpoint: str = "external/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
    model_cfg: str = "configs/sam2.1/sam2.1_hiera_b+.yaml"

    device: str = "cuda"
    use_bfloat16: bool = True

    # proposal filtering
    min_mask_area: int = 1500
    max_mask_area_ratio: float = 0.85
    min_bbox_side_px: int = 20

    # image resizing before mask generation
    max_image_side: int | None = 1024

    # whether to keep crop RGB for downstream DINO
    attach_rgb_crops: bool = True

    # automatic mask generation
    auto_points_per_side: int = 24
    auto_pred_iou_thresh: float = 0.88
    auto_stability_score_thresh: float = 0.92
    auto_crop_n_layers: int = 0
    auto_crop_n_points_downscale_factor: int = 1
    auto_min_mask_region_area: int = 200


class SAMSegmenter:
    """
    Thin SAM2 image-segmentation wrapper.

    Current design goal:
    - take one RGB image
    - return a list of candidate masks
    - keep everything simple and deterministic for later DINO integration

    Notes:
    - This wrapper intentionally does NOT do semantic classification.
    - It is only a mask proposal generator.
    """

    def __init__(self, cfg: SAMSegmenterConfig | None = None) -> None:
        self.cfg = cfg or SAMSegmenterConfig()
        self.device = torch.device(
            self.cfg.device if self.cfg.device == "cuda" and torch.cuda.is_available() else "cpu"
        )

        model = self._build_model()
        self._predictor = self._build_predictor(model)
        self._auto_mask_generator = self._build_auto_mask_generator(model)

    def _ensure_repo_on_path(self) -> None:
        repo_root = Path(self.cfg.repo_root).resolve()
        if not repo_root.exists():
            raise FileNotFoundError(f"SAM2 repo_root does not exist: {repo_root}")

        import sys
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

    def _build_model(self) -> Any:
        ckpt = Path(self.cfg.checkpoint).resolve()
        if not ckpt.exists():
            raise FileNotFoundError(f"SAM2 checkpoint does not exist: {ckpt}")

        self._ensure_repo_on_path()

        # delayed import so the rest of the repo remains importable
        from sam2.build_sam import build_sam2

        model = build_sam2(
            config_file=self.cfg.model_cfg,
            ckpt_path=str(ckpt),
            device=self.device.type,
        )
        return model

    def _build_predictor(self, model: Any) -> Any:
        self._ensure_repo_on_path()
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        predictor = SAM2ImagePredictor(model)
        return predictor

    def _build_auto_mask_generator(self, model: Any) -> Any:
        self._ensure_repo_on_path()
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

        amg = SAM2AutomaticMaskGenerator(
            model=model,
            points_per_side=self.cfg.auto_points_per_side,
            pred_iou_thresh=self.cfg.auto_pred_iou_thresh,
            stability_score_thresh=self.cfg.auto_stability_score_thresh,
            crop_n_layers=self.cfg.auto_crop_n_layers,
            crop_n_points_downscale_factor=self.cfg.auto_crop_n_points_downscale_factor,
            min_mask_region_area=self.cfg.auto_min_mask_region_area,
        )
        return amg

    @staticmethod
    def _resize_if_needed(rgb: np.ndarray, max_side: int | None) -> tuple[np.ndarray, float]:
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
    ) -> bool:
        h, w = image_shape
        area = int(mask.sum())
        if area < self.cfg.min_mask_area:
            return False

        if area > int(self.cfg.max_mask_area_ratio * h * w):
            return False

        x0, y0, x1, y1 = self._bbox_from_mask(mask)
        bw = x1 - x0
        bh = y1 - y0
        if bw < self.cfg.min_bbox_side_px or bh < self.cfg.min_bbox_side_px:
            return False

        if not np.isfinite(score):
            return False

        return True

    def generate_from_points(
        self,
        rgb: np.ndarray,
        prompt_points_xy: np.ndarray,
        prompt_labels: np.ndarray | None = None,
        multimask_output: bool = True,
    ) -> list[SAMMaskCandidate]:
        """
        Generate masks from explicit point prompts.

        Parameters
        ----------
        rgb : np.ndarray
            RGB image, uint8, shape (H, W, 3)
        prompt_points_xy : np.ndarray
            Shape (N, 2), image coordinates in pixels
        prompt_labels : np.ndarray | None
            Shape (N,), 1 for foreground, 0 for background.
            If None, all prompts are treated as foreground.
        """
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"rgb must be (H, W, 3), got {rgb.shape}")

        rgb_small, scale = self._resize_if_needed(rgb, self.cfg.max_image_side)

        pts = np.asarray(prompt_points_xy, dtype=np.float32).reshape(-1, 2)
        if prompt_labels is None:
            labels = np.ones((pts.shape[0],), dtype=np.int32)
        else:
            labels = np.asarray(prompt_labels, dtype=np.int32).reshape(-1)

        if scale != 1.0:
            pts = pts * scale

        with torch.inference_mode():
            if self.device.type == "cuda" and self.cfg.use_bfloat16:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    self._predictor.set_image(rgb_small)
                    masks, scores, _ = self._predictor.predict(
                        point_coords=pts,
                        point_labels=labels,
                        multimask_output=multimask_output,
                    )
            else:
                self._predictor.set_image(rgb_small)
                masks, scores, _ = self._predictor.predict(
                    point_coords=pts,
                    point_labels=labels,
                    multimask_output=multimask_output,
                )

        return self._postprocess_masks(
            rgb_original=rgb,
            masks_pred=masks,
            scores_pred=scores,
            scale=scale,
        )

    def generate_auto(self, rgb: np.ndarray) -> list[SAMMaskCandidate]:
        """
        Automatic mask proposals from a single RGB image.
        """
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"rgb must be (H, W, 3), got {rgb.shape}")

        rgb_small, scale = self._resize_if_needed(rgb, self.cfg.max_image_side)
        h0, w0 = rgb.shape[:2]

        with torch.inference_mode():
            if self.device.type == "cuda" and self.cfg.use_bfloat16:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    raw_masks = self._auto_mask_generator.generate(rgb_small)
            else:
                raw_masks = self._auto_mask_generator.generate(rgb_small)

        import cv2

        out: list[SAMMaskCandidate] = []
        for item in raw_masks:
            mask_small = np.asarray(item["segmentation"], dtype=np.uint8)

            if scale != 1.0:
                mask = cv2.resize(mask_small, (w0, h0), interpolation=cv2.INTER_NEAREST) > 0
            else:
                mask = mask_small > 0

            score = float(item.get("predicted_iou", 0.0))

            if not self._filter_mask(mask, score, (h0, w0)):
                continue

            bbox = self._bbox_from_mask(mask)
            area = int(mask.sum())
            crop_rgb = self._crop_rgb(rgb, bbox) if self.cfg.attach_rgb_crops else None

            out.append(
                SAMMaskCandidate(
                    mask=mask,
                    score=score,
                    bbox_xyxy=bbox,
                    area=area,
                    crop_rgb=crop_rgb,
                )
            )

        out.sort(key=lambda x: (x.score, x.area), reverse=True)
        return out

    def _postprocess_masks(
        self,
        rgb_original: np.ndarray,
        masks_pred: np.ndarray,
        scores_pred: np.ndarray,
        scale: float,
    ) -> list[SAMMaskCandidate]:
        import cv2

        h0, w0 = rgb_original.shape[:2]
        out: list[SAMMaskCandidate] = []

        for mask_small, score in zip(masks_pred, scores_pred):
            mask_small = np.asarray(mask_small, dtype=np.uint8)

            if scale != 1.0:
                mask = cv2.resize(mask_small, (w0, h0), interpolation=cv2.INTER_NEAREST) > 0
            else:
                mask = mask_small > 0

            if not self._filter_mask(mask, float(score), (h0, w0)):
                continue

            bbox = self._bbox_from_mask(mask)
            area = int(mask.sum())
            crop_rgb = self._crop_rgb(rgb_original, bbox) if self.cfg.attach_rgb_crops else None

            out.append(
                SAMMaskCandidate(
                    mask=mask,
                    score=float(score),
                    bbox_xyxy=bbox,
                    area=area,
                    crop_rgb=crop_rgb,
                )
            )

        out.sort(key=lambda x: x.score, reverse=True)
        return out