"""Shared helpers for the refbank-building tools.

Both tools/refbank_crop_screenshots.py (manual crop) and
tools/refbank_autocrop_masks.py (mask-driven auto-crop) write into the same
Data/ZED_screens/<assembly>/<object_id>/*.png layout consumed by
src/perception/learned/DINO/dino_identifier.py (see README.md section 2.3).
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REFBANK_DIR = REPO_ROOT / "Data" / "ZED_screens"
DEFAULT_ASSEMBLY_MAP = REPO_ROOT / "Data" / "assembly_part_ids.json"


def load_assembly_map(path: Path) -> dict[str, str]:
    """Invert Data/assembly_part_ids.json (assembly -> [parts]) into part -> assembly."""
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    part_to_assembly: dict[str, str] = {}
    for assembly, parts in raw.items():
        for part in parts:
            part_to_assembly[part] = assembly
    return part_to_assembly


def resolve_object_dir(refbank_dir: Path, object_id: str, assembly_map: dict[str, str]) -> Path:
    """Objects with a known assembly live at <refbank>/<assembly>/<object_id>/,
    objects with none (cubes, screwdrivers, ...) sit directly under <refbank>/<object_id>/."""
    assembly = assembly_map.get(object_id)
    object_dir = refbank_dir / assembly / object_id if assembly else refbank_dir / object_id
    return object_dir


def save_crop(object_dir: Path, stem: str, image_bgr: np.ndarray, dry_run: bool) -> Path | None:
    """Write a crop as <object_dir>/<stem>.png. Skips (returns None) if that
    exact target already exists, so re-running a tool is a resumable no-op."""
    target = object_dir / f"{stem}.png"
    if target.exists():
        return None
    if dry_run:
        return target
    object_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(target), image_bgr)
    return target


def downsample_to_hd720(image: np.ndarray, interpolation: int) -> np.ndarray:
    """Scale down (never up) so the frame fits within 1280x720, preserving aspect ratio."""
    h, w = image.shape[:2]
    scale = min(1280 / w, 720 / h, 1.0)
    if scale >= 1.0:
        return image
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    return cv2.resize(image, (new_w, new_h), interpolation=interpolation)
