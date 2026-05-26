"""Grounding DINO text-prompted box proposer.

Wraps the IDEA-Research Grounding DINO model so the pipeline can produce
class-conditioned box proposals before SAM segmentation (MUSE-style).
The boxes are then handed to SAM as box prompts, replacing the default
automatic-mask-generator stage.

The model is loaded lazily on the first call so importing this file is
free even when --mask-source != gdino_sam.

Default checkpoint: HuggingFace 'IDEA-Research/grounding-dino-base' via
the transformers library. To use the larger variant, pass
--gdino-model-id IDEA-Research/grounding-dino-large.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


@dataclass
class GDINOConfig:
    model_id: str = "IDEA-Research/grounding-dino-base"
    device: str = "cuda"
    box_threshold: float = 0.30
    text_threshold: float = 0.25
    max_boxes_per_image: int = 30
    # One prompt string per object class. Multiple words inside a string
    # are concatenated with a period as Grounding DINO expects.
    text_prompts: list[str] = field(default_factory=list)


@dataclass
class GDINOProposal:
    bbox_xyxy: tuple[int, int, int, int]
    score: float
    label: str  # class name from text_prompts, lowercase


class GroundingDINOProposer:
    """Thin wrapper around HuggingFace transformers' AutoModelForZeroShotObjectDetection.

    Usage:
        cfg = GDINOConfig(text_prompts=["cooling base", "cooling f", "cooling screw"])
        proposer = GroundingDINOProposer(cfg)
        proposals = proposer.propose(rgb)  # list[GDINOProposal]
    """

    def __init__(self, cfg: GDINOConfig | None = None) -> None:
        self.cfg = cfg or GDINOConfig()
        self._processor = None
        self._model = None
        self._device = None
        self._prompt_string = self._build_prompt_string(self.cfg.text_prompts)

    @staticmethod
    def _build_prompt_string(text_prompts: Iterable[str]) -> str:
        cleaned = [p.strip().lower() for p in text_prompts if p and p.strip()]
        if not cleaned:
            return ""
        # Grounding DINO expects period-separated phrases ending with a period.
        return ". ".join(cleaned) + "."

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import (
                AutoProcessor,
                AutoModelForZeroShotObjectDetection,
            )
        except ImportError as e:
            raise ImportError(
                "Grounding DINO requires 'transformers'. Install with:\n"
                "  pip install 'transformers>=4.40' "
            ) from e

        self._device = torch.device(
            self.cfg.device if (self.cfg.device == "cuda" and torch.cuda.is_available()) else "cpu"
        )
        self._processor = AutoProcessor.from_pretrained(self.cfg.model_id)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.cfg.model_id
        ).to(self._device).eval()

    def set_text_prompts(self, text_prompts: list[str]) -> None:
        self.cfg.text_prompts = list(text_prompts)
        self._prompt_string = self._build_prompt_string(self.cfg.text_prompts)

    def propose(self, rgb: np.ndarray) -> list[GDINOProposal]:
        if not self._prompt_string:
            return []
        self._lazy_load()

        import torch
        from PIL import Image

        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"rgb must be (H, W, 3), got {rgb.shape}")

        pil = Image.fromarray(rgb.astype(np.uint8))
        inputs = self._processor(
            images=pil, text=self._prompt_string, return_tensors="pt"
        ).to(self._device)

        with torch.inference_mode():
            outputs = self._model(**inputs)

        try:
            results = self._processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=self.cfg.box_threshold,
                text_threshold=self.cfg.text_threshold,
                target_sizes=[pil.size[::-1]],  # (H, W)
            )
        except TypeError as e:
            if "box_threshold" not in str(e):
                raise
            # Transformers versions differ here: some expect the box score
            # cutoff as "threshold" instead of "box_threshold".
            results = self._processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=self.cfg.box_threshold,
                text_threshold=self.cfg.text_threshold,
                target_sizes=[pil.size[::-1]],  # (H, W)
            )
        if not results:
            return []
        res = results[0]

        boxes = res.get("boxes")
        scores = res.get("scores")
        labels = res.get("labels")
        if boxes is None or len(boxes) == 0:
            return []

        proposals: list[GDINOProposal] = []
        for box, score, label in zip(boxes, scores, labels):
            x0, y0, x1, y1 = [int(round(float(v))) for v in box.tolist()]
            proposals.append(
                GDINOProposal(
                    bbox_xyxy=(x0, y0, x1, y1),
                    score=float(score),
                    label=str(label).strip().lower(),
                )
            )

        proposals.sort(key=lambda p: p.score, reverse=True)
        return proposals[: self.cfg.max_boxes_per_image]
