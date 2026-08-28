#!/usr/bin/env python3
"""Threshold assembly-part screenshots and recolor the masked (object) pixels.

Input layout matches Data/ZED_screens/<assembly>/<part>/*.png (the same
layout consumed by src/perception/learned/DINO/dino_identifier.py): each
image shows a single part centered on a plain table, so the part is
segmented from the background with an HSV threshold (the parts are molded
from the same dark plastic, so one brown/near-black range covers all of
them) and the masked pixels are recolored per-part.

Usage:
    python tools/recolor_assembly_parts.py --assembly plumbers_block
    python tools/recolor_assembly_parts.py --assembly plumbers_block --in-place
    python tools/recolor_assembly_parts.py --assembly plumbers_block --parts pb_base pb_top --save-masks
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "config" / "assembly_recolor.json"
DEFAULT_DATA_ROOT = REPO_ROOT / "Data" / "ZED_screens"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "Data" / "ZED_screens_recolored"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


@dataclass
class PartSpec:
    name: str
    hsv_lower: tuple[int, int, int]
    hsv_upper: tuple[int, int, int]
    recolor_rgb: tuple[int, int, int] | None  # None means "leave as-is"


def load_part_specs(config_path: Path, assembly: str, only_parts: list[str] | None) -> list[PartSpec]:
    config = json.loads(config_path.read_text())
    if assembly not in config:
        raise SystemExit(f"assembly '{assembly}' not found in {config_path} (known: {sorted(config)})")
    assembly_cfg = config[assembly]
    default_threshold = assembly_cfg.get("threshold", {})
    specs = []
    for part_name, part_cfg in assembly_cfg.get("parts", {}).items():
        if only_parts and part_name not in only_parts:
            continue
        threshold = part_cfg.get("threshold", default_threshold)
        recolor = part_cfg.get("recolor_rgb")
        specs.append(
            PartSpec(
                name=part_name,
                hsv_lower=tuple(threshold["hsv_lower"]),
                hsv_upper=tuple(threshold["hsv_upper"]),
                recolor_rgb=tuple(recolor) if recolor is not None else None,
            )
        )
    if not specs:
        raise SystemExit(f"no parts resolved for assembly '{assembly}' (--parts filter: {only_parts})")
    return specs


def threshold_mask(img_bgr: np.ndarray, hsv_lower, hsv_upper, largest_component_only: bool) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_lower, dtype=np.uint8), np.array(hsv_upper, dtype=np.uint8))

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    if largest_component_only:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels > 1:
            largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)

    return mask


def recolor_masked(img_bgr: np.ndarray, mask: np.ndarray, recolor_rgb) -> np.ndarray:
    out = img_bgr.copy()
    recolor_bgr = np.array(recolor_rgb[::-1], dtype=np.uint8)
    out[mask == 255] = recolor_bgr
    return out


def backup_original(path: Path, output_root: Path, data_root: Path) -> None:
    backup_path = output_root / "_backups" / path.relative_to(data_root)
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    if not backup_path.exists():
        shutil.copy2(path, backup_path)


def process_part(
    part_dir: Path,
    spec: PartSpec,
    data_root: Path,
    output_root: Path,
    in_place: bool,
    save_masks: bool,
    largest_component_only: bool,
    dry_run: bool,
) -> tuple[int, int]:
    images = sorted(p for p in part_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    n_written = 0
    for img_path in images:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"  [skip] could not read {img_path}", file=sys.stderr)
            continue

        mask = threshold_mask(img_bgr, spec.hsv_lower, spec.hsv_upper, largest_component_only)

        if spec.recolor_rgb is not None:
            out_img = recolor_masked(img_bgr, mask, spec.recolor_rgb)
        else:
            out_img = img_bgr

        if dry_run:
            n_written += 1
            continue

        if save_masks:
            mask_path = output_root / "_masks" / img_path.relative_to(data_root)
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(mask_path), mask)

        if in_place:
            backup_original(img_path, output_root, data_root)
            cv2.imwrite(str(img_path), out_img)
        else:
            out_path = output_root / img_path.relative_to(data_root)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), out_img)

        n_written += 1

    return len(images), n_written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--assembly", default="plumbers_block", help="Assembly name / subfolder under --data-root (default: plumbers_block)")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help=f"Root containing <assembly>/<part>/*.png (default: {DEFAULT_DATA_ROOT})")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help=f"JSON config with thresholds + recolor targets (default: {DEFAULT_CONFIG})")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help=f"Where recolored copies are written when not --in-place (default: {DEFAULT_OUTPUT_ROOT})")
    parser.add_argument("--parts", nargs="+", default=None, help="Only process these part folder names (default: all parts listed for the assembly in --config)")
    parser.add_argument("--in-place", action="store_true", help="Overwrite the source images instead of writing to --output-root. Originals are backed up to <output-root>/_backups/ first.")
    parser.add_argument("--save-masks", action="store_true", help="Also write the binary threshold mask for each image to <output-root>/_masks/")
    parser.add_argument("--no-largest-component", action="store_true", help="Disable keeping only the largest connected component of the mask (default: on)")
    parser.add_argument("--dry-run", action="store_true", help="Report what would be processed without writing any files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    assembly_dir = args.data_root / args.assembly
    if not assembly_dir.is_dir():
        raise SystemExit(f"assembly directory not found: {assembly_dir}")

    specs = load_part_specs(args.config, args.assembly, args.parts)

    if not args.dry_run and not args.in_place:
        args.output_root.mkdir(parents=True, exist_ok=True)

    total_images = 0
    total_written = 0
    for spec in specs:
        part_dir = assembly_dir / spec.name
        if not part_dir.is_dir():
            print(f"  [skip] no folder for part '{spec.name}' under {assembly_dir}", file=sys.stderr)
            continue

        recolor_desc = f"-> RGB{spec.recolor_rgb}" if spec.recolor_rgb is not None else "(left unchanged)"
        print(f"{spec.name}: HSV[{spec.hsv_lower} .. {spec.hsv_upper}] {recolor_desc}")

        n_images, n_written = process_part(
            part_dir=part_dir,
            spec=spec,
            data_root=args.data_root,
            output_root=args.output_root,
            in_place=args.in_place,
            save_masks=args.save_masks,
            largest_component_only=not args.no_largest_component,
            dry_run=args.dry_run,
        )
        total_images += n_images
        total_written += n_written
        print(f"  {n_written}/{n_images} images processed")

    mode = "dry-run" if args.dry_run else ("in-place" if args.in_place else f"-> {args.output_root}")
    print(f"\nDone ({mode}): {total_written}/{total_images} images across {len(specs)} part(s)")


if __name__ == "__main__":
    main()
