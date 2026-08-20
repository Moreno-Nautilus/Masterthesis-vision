#!/usr/bin/env python3
"""Auto-crop imgpy render sessions (image + segmentation mask) into the real
DINO reference bank, using the mask to find each part's bounding box instead
of a human dragging one.

Expected input layout (what `Data/raw_synthetic/` already is -- a copy of
imgpy's `workdir/`, so this tool only reads from it, never writes there):

    <input-dir>/<session>/render/<idx>_image.png
    <input-dir>/<session>/render/<idx>_mask.exr   (or _mask.png)
    <input-dir>/<job_name>.json                   (imgpy render job config)

`<session>` is named `<timestamp>-render-<job_name>`. For each session, the
job config (protagonist + scene.clutter) is read to build a
`mask_value -> object_id` table: imgpy encodes each object's `class_id` into
the mask as `class_id + 1` (0 is background), and each object's Blender name
(`<assembly>_<index>`) is resolved to a part id via
Data/assembly_part_ids.json[<assembly>][<index>]. Two instances of the same
part (e.g. two screws) share a class_id, so they crop as one bounding box.

If a session's job config is missing/unusable, pass `--object-id-map` (a
JSON file of `{"<session-dir-name>": "<object_id>"}`) to manually say "every
nonzero mask pixel in this session is <object_id>" instead.

Each frame is downsampled to fit within 1280x720 (image: area averaging,
mask: nearest-neighbor to keep integer labels) before the bounding box is
computed, then cropped (with padding) and saved into
Data/ZED_screens/<assembly>/<object_id>/. Already-cropped frames are skipped
on re-run.

Usage:
    python tools/refbank_autocrop_masks.py --input-dir Data/raw_synthetic --dry-run
    python tools/refbank_autocrop_masks.py --input-dir Data/raw_synthetic
    python tools/refbank_autocrop_masks.py --input-dir Data/raw_synthetic --sessions 20260817-144737-render-plumbers_block_base_only
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _refbank_common import (
    DEFAULT_ASSEMBLY_MAP,
    DEFAULT_REFBANK_DIR,
    downsample_to_hd720,
    load_assembly_map,
    resolve_object_dir,
    save_crop,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
OBJECT_NAME_RE = re.compile(r"^(?P<prefix>.+)_(?P<idx>\d+)$")


@dataclass
class SessionPlan:
    session_dir: Path
    job_name: str
    class_map: dict[int, str] | None  # mask_value -> object_id, or None = "any nonzero" fallback
    fallback_object_id: str | None


def load_parts_lists(assembly_map_path: Path) -> dict[str, list[str]]:
    if not assembly_map_path.exists():
        return {}
    return json.loads(assembly_map_path.read_text())


def resolve_object_id(name: str, parts_lists: dict[str, list[str]]) -> str | None:
    """'plumbers_block_2' -> parts_lists['plumbers_block'][2], honoring the
    longest matching assembly prefix (assembly names can contain '_')."""
    m = OBJECT_NAME_RE.match(name)
    if not m:
        return None
    prefix, idx = m.group("prefix"), int(m.group("idx"))
    parts = parts_lists.get(prefix)
    if parts is None or idx >= len(parts):
        return None
    return parts[idx]


def build_class_map(config_path: Path, parts_lists: dict[str, list[str]]) -> dict[int, str]:
    config = json.loads(config_path.read_text())
    entries = [config.get("protagonist", {})]
    entries.extend(config.get("scene", {}).get("clutter", []))

    class_map: dict[int, str] = {}
    for entry in entries:
        name = entry.get("name") or entry.get("object")
        class_id = entry.get("class_id")
        if name is None or class_id is None:
            continue
        object_id = resolve_object_id(name, parts_lists)
        if object_id is None:
            continue
        mask_value = int(class_id) + 1
        existing = class_map.get(mask_value)
        if existing is not None and existing != object_id:
            print(f"  [warn] mask value {mask_value} maps to both {existing!r} and {object_id!r} in {config_path.name}", file=sys.stderr)
            continue
        class_map[mask_value] = object_id
    return class_map


def plan_session(
    session_dir: Path,
    configs_dir: Path,
    parts_lists: dict[str, list[str]],
    object_id_map: dict[str, str],
) -> SessionPlan | None:
    m = re.search(r"-render-(?P<job>.+)$", session_dir.name)
    job_name = m.group("job") if m else session_dir.name

    config_path = configs_dir / f"{job_name}.json"
    class_map: dict[int, str] = {}
    if config_path.exists():
        try:
            class_map = build_class_map(config_path, parts_lists)
        except Exception as e:
            print(f"  [warn] failed to parse {config_path}: {e}", file=sys.stderr)

    if class_map:
        return SessionPlan(session_dir, job_name, class_map, None)

    fallback = object_id_map.get(session_dir.name)
    if fallback:
        return SessionPlan(session_dir, job_name, None, fallback)

    print(
        f"[skip session] {session_dir.name}: no usable class_id mapping "
        f"(missing/empty {config_path.name}) and no --object-id-map entry",
        file=sys.stderr,
    )
    return None


def read_mask(path: Path) -> np.ndarray:
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(path)
    if raw.ndim == 3:
        raw = raw[:, :, 0]
    return np.rint(raw.astype(np.float32)).astype(np.int32)


def find_frames(render_dir: Path) -> list[tuple[str, Path, Path]]:
    """Return (idx, image_path, mask_path) triples for frames that have both files."""
    frames = []
    for img_path in sorted(render_dir.glob("*_image.png")):
        idx = img_path.stem.removesuffix("_image")
        mask_path = render_dir / f"{idx}_mask.exr"
        if not mask_path.exists():
            mask_path = render_dir / f"{idx}_mask.png"
        if not mask_path.exists():
            print(f"  [skip frame] no mask for {img_path.name}", file=sys.stderr)
            continue
        frames.append((idx, img_path, mask_path))
    return frames


def crop_regions(
    image_ds: np.ndarray,
    mask_ds: np.ndarray,
    class_map: dict[int, str] | None,
    padding: int,
    min_area: int,
    max_fill_fraction: float,
) -> tuple[list[tuple[str, np.ndarray]], int]:
    h, w = mask_ds.shape[:2]
    frame_area = h * w
    out = []

    if class_map is None:
        regions = [(None, mask_ds > 0)]
    else:
        regions = [(object_id, mask_ds == value) for value, object_id in class_map.items()]

    n_skipped = 0
    for object_id, region in regions:
        ys, xs = np.where(region)
        fill_fraction = ys.size / frame_area
        if ys.size < min_area or fill_fraction > max_fill_fraction:
            # Too small (barely visible) or, like a domain-randomized render
            # camera sometimes produces, an extreme close-up filling the
            # frame with no recognizable object shape left.
            n_skipped += 1
            continue
        y0, y1 = max(0, ys.min() - padding), min(h, ys.max() + 1 + padding)
        x0, x1 = max(0, xs.min() - padding), min(w, xs.max() + 1 + padding)
        out.append((object_id, image_ds[y0:y1, x0:x1]))
    return out, n_skipped


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", type=Path, required=True, help="Root containing <session>/render/ folders (e.g. Data/raw_synthetic)")
    p.add_argument("--configs-dir", type=Path, default=None, help="Where <job_name>.json configs live (default: --input-dir)")
    p.add_argument("--sessions", nargs="+", default=None, help="Only process these session folder names (default: all with a render/ subfolder)")
    p.add_argument("--object-id-map", type=Path, default=None, help='JSON {"<session-dir-name>": "<object_id>"} fallback for sessions with no usable job config')
    p.add_argument("--refbank-dir", type=Path, default=DEFAULT_REFBANK_DIR, help=f"Reference-bank root to write into (default: {DEFAULT_REFBANK_DIR})")
    p.add_argument("--assembly-map", type=Path, default=DEFAULT_ASSEMBLY_MAP, help=f"assembly_part_ids.json, also used as the class_id name->part-index table (default: {DEFAULT_ASSEMBLY_MAP})")
    p.add_argument("--padding", type=int, default=12, help="Pixels of context added around each mask bounding box (default: 12)")
    p.add_argument("--min-area", type=int, default=400, help="Skip a detected region smaller than this many mask pixels (default: 400)")
    p.add_argument("--max-fill-fraction", type=float, default=0.7, help="Skip a detected region covering more than this fraction of the downsampled frame -- catches frame-filling close-ups with no recognizable object shape (default: 0.7)")
    p.add_argument("--dry-run", action="store_true", help="Report what would be cropped without reading pixel data or writing files")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_dir.is_dir():
        raise SystemExit(f"input dir not found: {args.input_dir}")
    configs_dir = args.configs_dir or args.input_dir

    assembly_map = load_assembly_map(args.assembly_map)
    parts_lists = load_parts_lists(args.assembly_map)
    object_id_map = json.loads(args.object_id_map.read_text()) if args.object_id_map else {}

    session_dirs = sorted(
        d for d in args.input_dir.iterdir()
        if d.is_dir() and (d / "render").is_dir() and (not args.sessions or d.name in args.sessions)
    )
    if not session_dirs:
        raise SystemExit(f"no session folders with a render/ subfolder under {args.input_dir}")

    total_frames = total_saved = total_already_done = total_regions_skipped = 0

    for session_dir in session_dirs:
        plan = plan_session(session_dir, configs_dir, parts_lists, object_id_map)
        if plan is None:
            continue

        frames = find_frames(session_dir / "render")
        mapping_desc = plan.class_map if plan.class_map is not None else f"any-nonzero -> {plan.fallback_object_id}"
        print(f"\n=== {session_dir.name} (job={plan.job_name}) {mapping_desc} -- {len(frames)} frame(s) ===")

        for idx, img_path, mask_path in frames:
            total_frames += 1
            stem = f"{session_dir.name}_{idx}"

            if args.dry_run:
                targets = plan.class_map.values() if plan.class_map is not None else [plan.fallback_object_id]
                for object_id in targets:
                    object_dir = resolve_object_dir(args.refbank_dir, object_id, assembly_map)
                    tag = "would skip (exists)" if (object_dir / f"{stem}.png").exists() else "would save"
                    print(f"  [{idx}] {tag} -> {object_dir / f'{stem}.png'}")
                continue

            image = cv2.imread(str(img_path))
            if image is None:
                print(f"  [skip] could not read {img_path}", file=sys.stderr)
                continue
            mask = read_mask(mask_path)

            image_ds = downsample_to_hd720(image, cv2.INTER_AREA)
            mask_ds = downsample_to_hd720(mask.astype(np.float32), cv2.INTER_NEAREST)
            mask_ds = np.rint(mask_ds).astype(np.int32)

            regions, n_skipped = crop_regions(
                image_ds, mask_ds, plan.class_map, args.padding, args.min_area, args.max_fill_fraction
            )
            total_regions_skipped += n_skipped
            for object_id, crop in regions:
                object_dir = resolve_object_dir(args.refbank_dir, object_id, assembly_map)
                saved = save_crop(object_dir, stem, crop, dry_run=False)
                if saved is None:
                    total_already_done += 1
                else:
                    total_saved += 1
                    print(f"  [{idx}] saved {saved} ({crop.shape[1]}x{crop.shape[0]})")

    print(
        f"\nDone: {total_saved} saved, {total_already_done} already had crops, "
        f"{total_regions_skipped} region(s) skipped (too small/too close-up), "
        f"{total_frames} frame(s) seen across {len(session_dirs)} session(s)."
    )


if __name__ == "__main__":
    main()
