"""YOLO class-agnostic box proposer.

Counter module to the Grounding-DINO text-prompted proposer: instead of
asking for boxes that match a textual class list, we run a generic YOLO
detector (default: yolov8x.pt, COCO-pretrained) and return *every* box
it finds. Class labels are kept around for logging but are not used as
a filter — the downstream DINO classifier decides what each region is.

This is useful when:
  - You don't want a hand-maintained list of text prompts.
  - You want "interesting regions" rather than "regions matching X".
  - You want a fast, single-pass box stage feeding SAM.

The boxes are then handed to SAM as box prompts, exactly like the
gdino_sam path: `sam.generate_from_boxes(rgb, boxes)`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class YOLOConfig:
    # Any ultralytics-compatible checkpoint. Bigger = slower but better
    # recall on small / partially-visible objects.
    model_name: str = "yolov8x.pt"
    device: str = "cuda"
    conf_threshold: float = 0.10
    iou_threshold: float = 0.45
    image_size: int = 640
    max_boxes_per_image: int = 50
    # Optional whitelist of class names to keep. Empty = class-agnostic
    # (keep everything). Use this only if you want to bias proposals
    # without going through Grounding-DINO.
    class_filter: list[str] = field(default_factory=list)
    # If True, ignore YOLO's class id entirely and treat detections as
    # pure "objectness" boxes — labels become "object". Default True so
    # the downstream classifier owns the label decision.
    class_agnostic: bool = True


@dataclass
class YOLOProposal:
    bbox_xyxy: tuple[int, int, int, int]
    score: float
    label: str  # class name from YOLO, lowercase; "object" if class_agnostic


class YOLOProposer:
    """Thin wrapper around ultralytics YOLO for class-agnostic box proposals.

    Usage:
        cfg = YOLOConfig(model_name="yolov8x.pt", conf_threshold=0.1)
        proposer = YOLOProposer(cfg)
        proposals = proposer.propose(rgb)  # list[YOLOProposal]
    """

    def __init__(self, cfg: Optional[YOLOConfig] = None) -> None:
        self.cfg = cfg or YOLOConfig()
        self._model = None
        self._device_str: str = "cpu"
        self._names: dict[int, str] = {}

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "YOLO proposer requires 'ultralytics'. Install with:\n"
                "  pip install ultralytics"
            ) from e

        if self.cfg.device == "cuda" and torch.cuda.is_available():
            self._device_str = "cuda"
        else:
            self._device_str = "cpu"

        self._model = YOLO(self.cfg.model_name)
        try:
            self._model.to(self._device_str)
        except Exception:
            # Some ultralytics builds infer device at predict time; ignore.
            pass

        raw_names = getattr(self._model, "names", {}) or {}
        if isinstance(raw_names, dict):
            self._names = {int(k): str(v) for k, v in raw_names.items()}
        else:
            self._names = {i: str(v) for i, v in enumerate(raw_names)}

    def propose(self, rgb: np.ndarray) -> list[YOLOProposal]:
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"rgb must be (H, W, 3), got {rgb.shape}")
        self._lazy_load()

        # Ultralytics accepts numpy arrays directly; expects BGR for
        # parity with cv2.imread. Convert from our RGB convention.
        import cv2
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        results = self._model.predict(
            bgr,
            conf=float(self.cfg.conf_threshold),
            iou=float(self.cfg.iou_threshold),
            imgsz=int(self.cfg.image_size),
            device=self._device_str,
            verbose=False,
        )
        if not results:
            return []
        res = results[0]
        boxes = getattr(res, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        whitelist = {c.strip().lower() for c in self.cfg.class_filter if c.strip()}

        xyxy = boxes.xyxy.detach().cpu().numpy()
        confs = boxes.conf.detach().cpu().numpy()
        cls_ids = boxes.cls.detach().cpu().numpy().astype(int)

        proposals: list[YOLOProposal] = []
        for box, conf, cls_id in zip(xyxy, confs, cls_ids):
            label = self._names.get(int(cls_id), str(int(cls_id))).strip().lower()
            if whitelist and label not in whitelist:
                continue
            x0, y0, x1, y1 = [int(round(float(v))) for v in box.tolist()]
            proposals.append(
                YOLOProposal(
                    bbox_xyxy=(x0, y0, x1, y1),
                    score=float(conf),
                    label="object" if self.cfg.class_agnostic else label,
                )
            )

        proposals.sort(key=lambda p: p.score, reverse=True)
        return proposals[: self.cfg.max_boxes_per_image]
