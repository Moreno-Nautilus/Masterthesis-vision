#!/usr/bin/env python3
"""Interactively crop raw screenshots into the real DINO reference bank.

Two input layouts are supported:

- Structured: --screenshots-dir has one subfolder per part (matching the
  target refbank layout, e.g. `pb_base/*.png`, `pb_top/*.png`). The
  subfolder name is used as the object_id directly.
- Flat/mixed: --screenshots-dir has images directly in it, of whatever
  parts, in no particular order. Before each crop you're prompted in the
  terminal for that image's object_id (Enter repeats the last one you typed).

Every time an image is processed, an ROI-selection window opens. Existing
crops are NEVER treated as "already done": rerunning this script will show
the images again and create additional uniquely named crops. This allows
multiple screws/instances to be cropped from the same screenshot across
multiple runs.

Controls (per image, via cv2.selectROI):
    drag a box, then ENTER/SPACE to confirm the crop
    c                 redraw the box
    ENTER/SPACE with no box drawn (or ESC) -> image is skipped, nothing saved

The displayed image is automatically scaled down to fit a 1200x800 window
while crop coordinates are mapped back to the original full-resolution image.

Usage:
    python tools/refbank_crop_screenshots.py --screenshots-dir /path/to/screenshots
    python tools/refbank_crop_screenshots.py --screenshots-dir /path/to/pb_base_only --part pb_base
    python tools/refbank_crop_screenshots.py --screenshots-dir /path/to/screenshots --parts pb_base pb_top
    python tools/refbank_crop_screenshots.py --screenshots-dir Data/raw_screenshots
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _refbank_common import DEFAULT_ASSEMBLY_MAP, DEFAULT_REFBANK_DIR, load_assembly_map, resolve_object_dir, save_crop

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
WINDOW = "crop (drag box, ENTER/SPACE=save, c=redo, ESC=skip)"
MAX_DISPLAY_W = 1200
MAX_DISPLAY_H = 800


def iter_part_dirs(
    screenshots_dir: Path,
    single_part: str | None,
    only_parts: list[str] | None,
):
    if single_part:
        yield single_part, screenshots_dir
        return

    for entry in sorted(screenshots_dir.iterdir()):
        if not entry.is_dir():
            continue
        if only_parts and entry.name not in only_parts:
            continue
        yield entry.name, entry


def prompt_part_id(img_name: str, last: str | None) -> str | None:
    """Ask which part an image in a flat/mixed folder belongs to.

    Enter repeats `last`; 'q' aborts the run; blank with no `last` reprompts.
    """
    suffix = f" [{last}]" if last else ""

    while True:
        try:
            answer = input(
                f"  {img_name} -- object_id{suffix} (q=quit): "
            ).strip()
        except EOFError:
            return None

        if answer.lower() == "q":
            return None

        if answer:
            return answer

        if last:
            return last

        print("    no previous object_id to repeat -- type one.")


def get_unique_crop_path(object_dir: Path, source_stem: str) -> Path:
    """Return a unique filename for a new crop without overwriting anything."""
    object_dir.mkdir(parents=True, exist_ok=True)

    i = 1
    while True:
        candidate = object_dir / f"{source_stem}_crop_{i:03d}.png"
        if not candidate.exists():
            return candidate
        i += 1


def crop_one(
    img_path: Path,
) -> tuple[object, tuple[int, int, int, int]] | None:
    """Display an image at a manageable size and return full-resolution ROI."""
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  [skip] could not read {img_path}", file=sys.stderr)
        return None

    h, w = img.shape[:2]

    # Scale the displayed image down only if necessary.
    scale = min(
        MAX_DISPLAY_W / w,
        MAX_DISPLAY_H / h,
        1.0,
    )

    if scale < 1.0:
        display_w = int(round(w * scale))
        display_h = int(round(h * scale))
        display_img = cv2.resize(
            img,
            (display_w, display_h),
            interpolation=cv2.INTER_AREA,
        )
    else:
        display_img = img

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(
        WINDOW,
        display_img.shape[1],
        display_img.shape[0],
    )
    cv2.setWindowTitle(
        WINDOW,
        f"{img_path.name} -- {WINDOW}",
    )

    # ROI is selected on the displayed/scaled image.
    x, y, roi_w, roi_h = cv2.selectROI(
        WINDOW,
        display_img,
        showCrosshair=True,
        fromCenter=False,
    )

    if roi_w <= 0 or roi_h <= 0:
        return None

    # Convert ROI coordinates back to the original image coordinates.
    if scale < 1.0:
        x = int(round(x / scale))
        y = int(round(y / scale))
        roi_w = int(round(roi_w / scale))
        roi_h = int(round(roi_h / scale))

    # Clamp coordinates to the original image boundaries.
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    roi_w = min(roi_w, w - x)
    roi_h = min(roi_h, h - y)

    if roi_w <= 0 or roi_h <= 0:
        return None

    return img, (x, y, roi_w, roi_h)


def save_new_crop(
    object_dir: Path,
    source_stem: str,
    crop,
    dry_run: bool = False,
) -> Path:
    """Save a crop using a unique filename."""
    output_path = get_unique_crop_path(object_dir, source_stem)

    if dry_run:
        return output_path

    object_dir.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(output_path), crop):
        raise RuntimeError(f"failed to write crop to {output_path}")

    return output_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--screenshots-dir",
        type=Path,
        required=True,
        help=(
            "Root with one subfolder per part (or a single part's folder, "
            "combined with --part)"
        ),
    )
    p.add_argument(
        "--part",
        default=None,
        help=(
            "Treat --screenshots-dir as a single part's folder with this "
            "object_id, instead of iterating subfolders"
        ),
    )
    p.add_argument(
        "--parts",
        nargs="+",
        default=None,
        help=(
            "Only process these subfolder/object_id names "
            "(default: all subfolders)"
        ),
    )
    p.add_argument(
        "--refbank-dir",
        type=Path,
        default=DEFAULT_REFBANK_DIR,
        help=(
            "Reference-bank root to write into "
            f"(default: {DEFAULT_REFBANK_DIR})"
        ),
    )
    p.add_argument(
        "--assembly-map",
        type=Path,
        default=DEFAULT_ASSEMBLY_MAP,
        help=(
            "assembly_part_ids.json used to place each part under "
            f"<assembly>/ (default: {DEFAULT_ASSEMBLY_MAP})"
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Walk images and report what would happen without opening "
            "windows or writing files"
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.screenshots_dir.is_dir():
        raise SystemExit(
            f"screenshots dir not found: {args.screenshots_dir}"
        )

    assembly_map = load_assembly_map(args.assembly_map)

    is_flat = (
        not args.part
        and not any(
            p.is_dir() for p in args.screenshots_dir.iterdir()
        )
    )

    total_seen = 0
    total_saved = 0
    total_skipped = 0

    try:
        if is_flat:
            images = sorted(
                p
                for p in args.screenshots_dir.iterdir()
                if p.suffix.lower() in IMAGE_SUFFIXES
            )

            if not images:
                raise SystemExit(
                    f"no images and no part subfolders under "
                    f"{args.screenshots_dir}"
                )

            print(
                f"\n=== flat/mixed folder: {args.screenshots_dir} "
                f"({len(images)} image(s)) ==="
            )

            last_part: str | None = None

            for i, img_path in enumerate(images, 1):
                total_seen += 1

                if args.dry_run:
                    print(
                        f"  [{i}/{len(images)}] {img_path.name} "
                        f"-> would prompt for object_id and create a new crop"
                    )
                    total_saved += 1
                    continue

                part_id = prompt_part_id(
                    img_path.name,
                    last_part,
                )

                if part_id is None:
                    print(
                        "  [quit] stopping early; progress is saved so far."
                    )
                    break

                last_part = part_id

                object_dir = resolve_object_dir(
                    args.refbank_dir,
                    part_id,
                    assembly_map,
                )

                # IMPORTANT:
                # There is deliberately NO "already exists" check here.
                # Every run can create another crop from the same screenshot.
                result = crop_one(img_path)

                if result is None:
                    print(
                        f"  [{i}/{len(images)}] skipped {img_path.name}"
                    )
                    total_skipped += 1
                    continue

                img, (x, y, w, h) = result
                crop = img[y : y + h, x : x + w]

                saved = save_new_crop(
                    object_dir,
                    img_path.stem,
                    crop,
                    dry_run=False,
                )

                print(
                    f"  [{i}/{len(images)}] saved {saved} "
                    f"({w}x{h})"
                )
                total_saved += 1

        else:
            for part_id, part_dir in iter_part_dirs(
                args.screenshots_dir,
                args.part,
                args.parts,
            ):
                images = sorted(
                    p
                    for p in part_dir.iterdir()
                    if p.suffix.lower() in IMAGE_SUFFIXES
                )

                if not images:
                    print(
                        f"[skip] no images under {part_dir}",
                        file=sys.stderr,
                    )
                    continue

                object_dir = resolve_object_dir(
                    args.refbank_dir,
                    part_id,
                    assembly_map,
                )

                print(
                    f"\n=== {part_id} -> {object_dir} "
                    f"({len(images)} image(s)) ==="
                )

                for i, img_path in enumerate(images, 1):
                    total_seen += 1

                    if args.dry_run:
                        print(
                            f"  [{i}/{len(images)}] {img_path.name} "
                            f"-> would create a new crop"
                        )
                        total_saved += 1
                        continue

                    # IMPORTANT:
                    # No existing-file check. Rerunning the script will
                    # process the same source image again.
                    result = crop_one(img_path)

                    if result is None:
                        print(
                            f"  [{i}/{len(images)}] skipped "
                            f"{img_path.name}"
                        )
                        total_skipped += 1
                        continue

                    img, (x, y, w, h) = result
                    crop = img[y : y + h, x : x + w]

                    saved = save_new_crop(
                        object_dir,
                        img_path.stem,
                        crop,
                        dry_run=False,
                    )

                    print(
                        f"  [{i}/{len(images)}] saved {saved} "
                        f"({w}x{h})"
                    )
                    total_saved += 1

    except KeyboardInterrupt:
        print(
            "\n[interrupted] progress is saved so far; "
            "re-run to continue."
        )
    finally:
        cv2.destroyAllWindows()

    print(
        f"\nDone: {total_saved} saved, "
        f"{total_skipped} skipped, "
        f"{total_seen} image(s) seen."
    )


if __name__ == "__main__":
    main()