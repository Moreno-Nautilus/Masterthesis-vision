# Building the DINO Reference Bank

Two tools under `tools/` fill `Data/ZED_screens/<assembly>/<object_id>/*.png`
— the **real** reference-bank layout `dino_identifier.py` embeds at startup
(see [README.md §2.3](../README.md#23-the-data-folder-you-must-create-this)).
One crops raw screenshots by hand; the other auto-crops rendered
image/mask pairs. Both are resumable (re-running skips images that already
have a crop) and never touch their input files.

---

## Quickstart

```bash
# Manual: drag a box around the part in each raw screenshot
python3 tools/refbank_crop_screenshots.py --screenshots-dir Data/raw_screenshots

# Automatic: bbox from the render's segmentation mask, no GUI
python3 tools/refbank_autocrop_masks.py --input-dir Data/raw_synthetic
```

Always sanity-check with `--dry-run` first — it prints every target path
(and, for the manual tool, the image count) without opening a window or
writing anything.

---

## 1. Manual crop — `refbank_crop_screenshots.py`

For raw screenshots (e.g. off the ZED viewer) that need a human to draw the
box. Two input layouts:

- **Structured** — one subfolder per part, named after the object_id:
  `screenshots/pb_base/*.png`, `screenshots/pb_top/*.png`, ... The subfolder
  name is used directly as the object_id.
- **Flat/mixed** — images sit directly in `--screenshots-dir` in no
  particular order (this is what `Data/raw_screenshots` is). Before each
  crop you're prompted in the terminal for that image's object_id; press
  Enter to repeat the last one you typed (consecutive screenshots are
  usually the same part), or `q` to stop early.

```bash
python3 tools/refbank_crop_screenshots.py --screenshots-dir Data/raw_screenshots        # flat, prompts per image
python3 tools/refbank_crop_screenshots.py --screenshots-dir /path/to/pb_base_only --part pb_base   # single part, no prompt
python3 tools/refbank_crop_screenshots.py --screenshots-dir /path/to/screenshots        # structured, all subfolders
python3 tools/refbank_crop_screenshots.py --screenshots-dir /path/to/screenshots --parts pb_base pb_top  # only these subfolders
```

**Controls** (each image opens a `cv2.selectROI` window):

| Action | Key |
|---|---|
| Drag a box | mouse |
| Confirm crop | ENTER / SPACE |
| Redraw the box | `c` |
| Skip this image (nothing saved) | ENTER/SPACE with no box, or ESC |
| Quit the whole run | Ctrl-C, or `q` at a flat-mode prompt |

The crop is written to `Data/ZED_screens/<assembly>/<object_id>/<source
stem>.png`, where `<assembly>` is looked up in
`Data/assembly_part_ids.json` (a part with no known assembly, e.g. a loose
cube, is written directly under the refbank root). If that target file
already exists, the image is skipped — safe to stop with Ctrl-C anytime and
resume later, or re-run after adding a `--part`/`--parts` filter.

---

## 2. Auto-crop from renders — `refbank_autocrop_masks.py`

For imgpy render sessions (`Data/raw_synthetic`, a copy of imgpy's
`workdir/`) that already have a segmentation mask, so no human needs to draw
a box. Expected layout:

```
<input-dir>/<session>/render/<idx>_image.png
<input-dir>/<session>/render/<idx>_mask.exr      # or _mask.png
<input-dir>/<job_name>.json                      # imgpy render job config
```

`<session>` is named `<timestamp>-render-<job_name>`. For each session the
job config (`protagonist` + `scene.clutter`) is read to build a
`mask_value -> object_id` table:

- imgpy encodes each object's `class_id` into the mask as `class_id + 1`
  (`0` is background).
- Each object's Blender name (`<assembly>_<index>`, e.g.
  `plumbers_block_2`) resolves to a part id via
  `Data/assembly_part_ids.json[<assembly>][<index>]` (e.g. `pb_base`).
- Two instances of the same part (e.g. two screws) share a `class_id`, so
  they crop as one bounding box covering both.

```bash
python3 tools/refbank_autocrop_masks.py --input-dir Data/raw_synthetic
python3 tools/refbank_autocrop_masks.py --input-dir Data/raw_synthetic --sessions 20260817-144737-render-plumbers_block_base_only
```

If a session's job config is missing, unparseable, or doesn't resolve any
`class_id` (this happens for ad-hoc/test renders with no `class_id` set),
the whole session is skipped with a warning — pass `--object-id-map` to
manually say "every nonzero mask pixel in this session is this one part":

```bash
echo '{"20260814-101957-render-test": "pb_base"}' > /tmp/map.json
python3 tools/refbank_autocrop_masks.py --input-dir Data/raw_synthetic --object-id-map /tmp/map.json
```

**Per frame:** downsample image (area-averaging) and mask
(nearest-neighbor, to keep integer labels) to fit within 1280x720, then for
each mapped `class_id` take the bounding box of matching mask pixels, pad
it, and crop. Two filters drop bad detections before saving:

| Flag | Default | Skips a region when... |
|---|---|---|
| `--min-area` | 400 px | mask pixel count is below this (barely visible / occluded) |
| `--max-fill-fraction` | 0.7 | mask covers more than this fraction of the frame — imgpy's randomized camera occasionally lands on an extreme close-up with no recognizable object shape left, which is worse than useless as a reference image |

`--padding` (default 12px) controls how much context is kept around the
tight mask bbox. Output goes to
`Data/ZED_screens/<assembly>/<object_id>/<session-name>_<frame-idx>.png`;
an existing target is skipped, same resumability as the manual tool.

---

## Flags reference

Both tools share:

- `--refbank-dir` (default `Data/ZED_screens`) — where crops are written.
- `--assembly-map` (default `Data/assembly_part_ids.json`) — used to place
  each object_id under `<assembly>/` (or the refbank root, if the part
  isn't listed for any assembly).
- `--dry-run` — report targets without touching pixels or writing files.

`refbank_crop_screenshots.py` only: `--part` (treat `--screenshots-dir` as
one part's folder), `--parts` (filter which subfolders to process in
structured mode).

`refbank_autocrop_masks.py` only: `--configs-dir` (where `<job_name>.json`
files live, default `--input-dir`), `--sessions` (filter which session
folders to process), `--object-id-map`, `--padding`, `--min-area`,
`--max-fill-fraction`.
