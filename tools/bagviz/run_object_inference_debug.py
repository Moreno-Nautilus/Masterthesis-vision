"""
Standalone "mini pipeline" debug script: for each camera's raw RGB+depth
frame captured by tools/bagviz/capture_pipeline_snapshots.py, this runs
FRESH inference (Grounding DINO -> SAM2 -> DINOv2 re-ID -> FoundationPose) --
it does NOT read the bag's cached DebugFrame detections the way
tools/bagviz/view_pointclouds.py does. For each camera, separately:

  1. The raw image is cropped to the camera's ROI polygon (same
     crop_rgb_to_polygon_bbox() + polygon-fill blackout the live node's
     _generate_and_filter_masks() does; empty/default polygon => the whole
     frame, no-op crop).
  2. Grounding DINO proposes boxes on that crop. By default (matching the
     live node's --gdino-use-items-prompt=True default), the prompt is the
     single class-agnostic phrase "items" -- GDINO here is a pure box
     proposer, NOT the thing that names the object. Pass
     --no-gdino-use-items-prompt to fall back to the specific per-part text
     prompts instead (see DEFAULT_TEXT_PROMPTS below), matching the live
     node's --no-gdino-use-items-prompt path.
  3. SAM2 segments one mask per box, also on that crop -- both DINO and SAM
     only ever see the ROI crop, same order as the live pipeline. Masks/
     boxes are lifted back to full-image coordinates right after (see
     lift_mask_bbox_to_full()), same as lift_crop_masks_to_full_image().
  4. Object IDENTITY comes from a batched DINOv2 embedding classifier
     (batch_dino_classify(), a verbatim adaptation of the live node's own
     function of the same name) run against a reference bank built from
     --reference-dir (+ --reference-renders-dir when --reference-source
     includes renders) -- exactly what the live node's DINOIdentifier does
     against Data/ZED_screens. The same accept/reject gating the live node's
     _classify_masks_batched() applies (--dino-min-score / --dino-min-margin,
     with a looser small-object carve-out below a 5000px bbox area) decides
     "unknown" here too. Anything that comes back "unknown" -- or whose
     classified object_id doesn't resolve to a known CAD mesh under
     --cad-dir -- is DROPPED right there: not tracked, not added to the
     point cloud, not drawn, not saved. This matches the live pipeline's own
     object_id == "unknown" gate in _select_top_candidates().
  5. The remaining (known) detections' pixels are back-projected into ONE
     combined point cloud per camera (only the objects -- not the full
     scene).
  6. FoundationPose is run once per remaining object (standard single-shot
     registration, iteration=0) to estimate its pose.
  7. All of this run's outputs -- pointcloud_objects_debug.ply,
     poses_objects_debug.yaml, and detections_overlay.png (bbox + mask +
     pose axes per kept object) -- are saved under a dedicated
     frame_dir/offline_inference/ subfolder, kept separate from the
     capture-time files already sitting in frame_dir itself
     (rgb_native.png, depth_m.npy, frame_info.yaml, poses.yaml, and --
     whenever capture_pipeline_snapshots.py's bag had them -- the LIVE
     pipeline's own cached rgb_raw/dino_overlay/sam_overlay/pose_overlay/
     track_overlay/axes_overlay PNGs). Keeping this script's own overlay in
     its own subfolder is deliberate: an offline detections_overlay.png
     would otherwise land at the exact same path/name pattern as one of
     those live-pipeline overlays and silently look like it came from the
     live run.

This script does the COMPUTE only -- it never opens an Open3D window
(mirrors capture_pipeline_snapshots.py, which also only ever reads/writes
files). It needs GPU + the full inference stack (torch, transformers,
sam2, FoundationPose/nvdiffrast), so it runs inside the `vision` docker
container, which is normally headless -- an Open3D window would fail there
anyway (GLFW/XDG_RUNTIME_DIR errors). Visualization is a SEPARATE,
lightweight companion script that needs only numpy/open3d/pyyaml and runs
on a display-capable host in the `bagviz` conda env, same split as
capture_pipeline_snapshots.py -> view_pointclouds.py:

    conda activate bagviz   # on the host, NOT in the container
    python -m tools.bagviz.view_object_inference_debug \\
        --run-dir outputs/bagviz/<run> --frame 0

This is intentionally a SIMPLIFIED stand-in for the live pipeline
(src/perception/ros/learn_runners/run_pipeline_track_multicam_realsense.py):
the ROI-polygon crop -> GDINO -> SAM -> DINOv2 stage order and identity
gating now match, but there is still no border/dedup mask filtering, no
per-camera overlap-based top-candidate dedup or --max-objects cap, no
depth-coverage gate on the classified candidates, no tracking, and no
cross-camera fusion. See FILTERING_REPORT below (also printed at the
start/end of every run) for exactly what the live pipeline additionally
filters that this script does not. This script also can't import the live
node's own batch_dino_classify()/_classify_masks_batched() directly -- that
file imports rclpy + fp_debug_msgs at module level, which this script must
not need -- so the relevant pieces are verbatim adaptations kept in sync by
hand (see the comments on each).

Needs GPU + the full inference stack (torch, transformers, sam2,
FoundationPose/nvdiffrast) -- unlike the lightweight `bagviz` conda env
used by capture_pipeline_snapshots.py/view_pointclouds.py. Run this the
same place run_pipeline_track_multicam_realsense.py runs: inside the
`vision` docker container (docker exec -it vision bash). It does NOT need
ROS -- it only reads what capture_pipeline_snapshots.py already saved to
disk (rgb_native.png, depth_m.npy, frame_info.yaml).

Requires a run captured with a capture_pipeline_snapshots.py new enough to
save frame_info.yaml["K"] (camera intrinsics) -- re-run it if your run
predates that field.

Usage:
    docker exec -it vision bash
    python -m tools.bagviz.run_object_inference_debug \\
        --run-dir outputs/bagviz/<run> --frame 0
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
import yaml

from tools.bagviz.view_pointclouds import camera_frame_dir

from src.perception.learned.GDINO.grounding_dino_proposal import (
    GDINOConfig,
    GroundingDINOProposer,
)
from src.perception.learned.SAM.sam_segmentation import (
    SAMSegmenter,
    SAMSegmenterConfig,
)
from src.perception.learned.DINO.dino_identifier import (
    DINOIdentifier,
    DINOIdentifierConfig,
    DINOResult,
)
from src.perception.learned.FP.pose_foundation import (
    FoundationPoseConfig,
    FoundationPoseWrapper,
)


ALL_CAMS = ["zed2i_1", "realsense_1", "realsense_2"]

# Matches the live pipeline's own cam1/cam2/cam3 convention (see its
# --cam1-roi-polygon/--cam2-roi-polygon/--cam3-roi-polygon and the comment
# "cam1 = zed2i_1 (static). cam2/cam3 = realsense_1/realsense_2").
CAM_ROI_ARG = {
    "zed2i_1": "cam1_roi_polygon",
    "realsense_1": "cam2_roi_polygon",
    "realsense_2": "cam3_roi_polygon",
}

# All of this script's own outputs go under frame_dir/OFFLINE_SUBDIR -- kept
# separate from capture_pipeline_snapshots.py's files (including the LIVE
# pipeline's own cached overlay PNGs) sitting directly in frame_dir.
OFFLINE_SUBDIR = "offline_inference"

# RGB tuples -- drawn directly onto the RGB-ordered `rgb` array, converted
# to BGR only at the final cv2.imwrite (same convention as
# capture_pipeline_snapshots.py).
DETECTION_PALETTE = [
    (66, 135, 245), (129, 66, 245), (66, 245, 135), (66, 188, 245),
    (245, 66, 188), (66, 245, 227), (66, 66, 245), (66, 245, 154),
]

# Only used when --no-gdino-use-items-prompt is passed -- mirrors the live
# pipeline's own --gdino-text-prompts default (the current object set for
# this project) for its (non-default) non-items code path. By default GDINO
# instead gets the single class-agnostic "items" prompt (see
# --gdino-use-items-prompt below); it only proposes BOXES, object identity
# comes from the DINOv2 classifier (see build_dino_identifier() /
# batch_dino_classify() below).
DEFAULT_TEXT_PROMPTS = (
    "cooling base,cooling f,cooling screw,pb base,pb pipe,pb screw,pb top"
)

# Live pipeline's own small-object carve-out in _classify_masks_batched():
# below this bbox area, use a looser absolute-score floor but still require
# a margin over the runner-up class.
SMALL_OBJECT_BBOX_AREA_PX = 5000
SMALL_OBJECT_MIN_SCORE = 0.40
SMALL_OBJECT_MIN_MARGIN = 0.025

FILTERING_REPORT = """
============================================================
 Point-cloud / mask filtering: LIVE pipeline vs. THIS script
============================================================

Baked into SAMSegmenter itself (src/perception/learned/SAM/sam_segmentation.py
:_filter_mask) -- applies HERE too, every SAM mask goes through it:
  - area in [min_mask_area=800, max_mask_area_ratio=0.06 * H*W]
  - bbox side >= min_bbox_side_px=20
  - aspect ratio <= max_aspect_ratio=4.5
  - fill ratio (mask_area / bbox_area) >= 0.20
  - finite score
  - masks > 5000px must stay one connected blob (before AND after a 5x5
    erode x3) else rejected as "fragmented"
  - optional HSV shadow rejection (shadow_filter_enabled, default OFF)

Geometric filters in the live node's _generate_and_filter_masks()
(run_pipeline_track_multicam_realsense.py):
  - crop_rgb_to_polygon_bbox(): APPLIED here too now -- DINO+SAM only see
    the per-camera ROI polygon crop (--cam1/2/3-roi-polygon), same order
    as the live node. Defaults to empty (whole frame, no-op) for all three
    cameras, same as the live node's own current defaults.
  - per-camera min_mask_area / min_bbox_side_px overrides -- NOT applied
    (SAMSegmenterConfig's own single global min_mask_area/min_bbox_side_px
    still apply, see above)
  - reject_large_masks(): per-camera max_mask_area_ratio / max_bbox_area_ratio
    -- NOT applied
  - reject_border_masks(): drop masks mostly touching the image border --
    NOT applied
  - reject_outside_roi_polygon(): mask bbox center must be inside the ROI
    -- NOT applied (the ROI blackout above already keeps SAM from seeing
    outside it, but nothing separately re-checks the mask center)
  - dedup_masks_by_bbox_iou(): greedy IoU + containment dedup across boxes
    -- NOT applied

Object identity (DINOv2 re-ID against the reference bank) -- APPLIED here
now, via batch_dino_classify() + the same --dino-min-score/--dino-min-margin
(with the small-object carve-out) gating as the live node's
_classify_masks_batched(). NOT applied here:
  - _select_top_candidates()'s overlap-based dedup (drop a candidate whose
    mask overlaps >15% with an already-selected one) and --max-objects cap
  - _select_top_candidates()'s per-candidate depth-coverage gate
    (min_valid_z_m/max_valid_z_m/min_depth_coverage)
  - per-camera cross-run identity smoothing/tracking memory

Depth / point-cloud filters used elsewhere in the pipeline
(src/perception/tracking/icp_refiner.py, fused_multicam_helpers.py,
multicam_fusion.py) -- NOT applied here beyond a basic depth-validity check:
  - depth clipped to (min_depth, max_depth); this script only applies
    (0 < z < --max-depth-m)
  - optional mask morphological close/erode before back-projection
    (mask_morph_close_kernel, mask_interior_erosion -- default 0/off anyway)
  - optional depth hole-filling median blur inside the mask
    (fill_depth_holes_in_mask)
  - voxel downsampling after back-projection (voxel_size ~ 2-3mm)
  - statistical outlier removal (remove_statistical_outlier,
    nb_neighbors=20, std_ratio=2.0) -- OFF by default during live tracking
    ("Cutie mask is clean"), but view_pointclouds.py always applies it to
    the raw *scene* cloud for viewing

Cross-camera fusion filters (multicam_fusion.py) -- N/A here, this script
never fuses detections across cameras; each camera's objects/poses are
independent:
  - match_detections_across_cameras(): label agreement + cloud-bbox-diagonal
    gating before two cameras' detections are treated as the same object
  - fused_gate_min_mask_area(_ratio): gates a fused detection on mask size

This script's own combined cloud = union of (known-object) SAM masks'
pixels, back-projected keeping only finite depth with 0 < z <
--max-depth-m. ROI cropping is applied (see above); no border/dedup, no
morphology, no voxel downsampling, no outlier removal.
============================================================
"""


@dataclass
class ObjectDetection:
    gdino_label: str  # GDINO's own box label -- usually "items" (see above)
    object_id: str  # DINOv2-classified object id, e.g. "cooling_screw"
    mask: np.ndarray
    bbox_xyxy: tuple[int, int, int, int]
    sam_score: float
    gdino_score: float
    dino_score: float
    dino_margin: float
    mesh_path: Optional[str] = None
    T_object_camera: Optional[np.ndarray] = None
    pose_error: Optional[str] = None


def build_mesh_map(cad_dir: str) -> dict[str, str]:
    """Same scan the live pipeline node's _build_mesh_map() does: mesh
    files directly under cad_dir or one level down in an assembly
    subfolder (e.g. cad_dir/cooling_manifold/cooling_screw.obj)."""
    cad_root = Path(cad_dir)
    mesh_map: dict[str, str] = {}
    if not cad_root.is_dir():
        return mesh_map
    for ext in ("*.obj", "*.stl"):
        for mesh_file in list(cad_root.glob(ext)) + list(cad_root.glob(f"*/{ext}")):
            name = mesh_file.stem
            for key in (name, name.lower(), name.replace(" ", "_"),
                        name.replace(" ", "_").lower()):
                mesh_map[key] = str(mesh_file)
    return mesh_map


def resolve_mesh_for_label(label: str, mesh_map: dict[str, str]) -> Optional[str]:
    """DINOv2-classified object_id -> mesh path. The reference-bank folder
    names (under --reference-dir) and the CAD mesh filenames are supposed
    to use the same object-id convention (e.g. both "cooling_screw"), so
    this is normally an exact match; the case/underscore variants are a
    safety net, not the primary mechanism (unlike the old GDINO-label
    string-match this replaced)."""
    candidates = [
        label,
        label.replace(" ", "_"),
        label.strip().lower(),
        label.strip().lower().replace(" ", "_"),
    ]
    for c in candidates:
        if c in mesh_map:
            return mesh_map[c]
    return None


def parse_polygon_string(s: str) -> np.ndarray:
    """Verbatim copy of the live node's parse_polygon_string(): empty/
    whitespace => no ROI (resolved to the full frame at use time)."""
    if not s or not s.strip():
        return np.zeros((0, 2), dtype=np.int32)
    vals = [int(v.strip()) for v in s.split(",")]
    if len(vals) % 2 != 0:
        raise ValueError(f"Polygon string must have even number of values: {s}")
    return np.array(vals, dtype=np.int32).reshape(-1, 2)


def crop_rgb_to_polygon_bbox(
    rgb: np.ndarray,
    polygon: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Verbatim copy of the live node's crop_rgb_to_polygon_bbox(): crop an
    RGB frame to the ROI bbox and shift polygon coords into crop space."""
    h, w = rgb.shape[:2]
    xs = polygon[:, 0]
    ys = polygon[:, 1]

    x0 = max(0, int(xs.min()))
    y0 = max(0, int(ys.min()))
    x1 = min(w, int(xs.max()) + 1)
    y1 = min(h, int(ys.max()) + 1)

    rgb_crop = rgb[y0:y1, x0:x1].copy()
    polygon_crop = polygon.copy()
    polygon_crop[:, 0] -= x0
    polygon_crop[:, 1] -= y0

    return rgb_crop, polygon_crop.astype(np.int32), x0, y0


def lift_mask_bbox_to_full(
    mask_crop: np.ndarray,
    bbox_crop: tuple[int, int, int, int],
    full_h: int,
    full_w: int,
    x0: int,
    y0: int,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Adaptation of the live node's lift_crop_masks_to_full_image() for a
    single mask+bbox: paste the crop-space mask into a full-size canvas and
    shift the bbox by the crop origin."""
    full_mask = np.zeros((full_h, full_w), dtype=mask_crop.dtype)
    h, w = mask_crop.shape[:2]
    full_mask[y0:y0 + h, x0:x0 + w] = mask_crop

    bx0, by0, bx1, by1 = bbox_crop
    return full_mask, (bx0 + x0, by0 + y0, bx1 + x0, by1 + y0)


def bbox_size_xyxy(b: tuple[int, int, int, int]) -> tuple[int, int]:
    """Verbatim copy of the live node's bbox_size_xyxy()."""
    x0, y0, x1, y1 = b
    return x1 - x0, y1 - y0


def bbox_crop_with_local_mask(
    rgb: np.ndarray,
    mask: np.ndarray,
    bbox_xyxy: tuple[int, int, int, int],
    pad_frac: float = 0.15,
    min_pad_px: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """Verbatim copy of the live node's bbox_crop_with_local_mask(): crop
    RGB and mask around a bbox, keeping a little context for the DINOv2
    classifier crop (same padding as the live node's memory/DINO crops)."""
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = [int(v) for v in bbox_xyxy]

    bw = x1 - x0
    bh = y1 - y0

    pad_x = max(min_pad_px, int(round(bw * pad_frac)))
    pad_y = max(min_pad_px, int(round(bh * pad_frac)))

    x0p = max(0, x0 - pad_x)
    y0p = max(0, y0 - pad_y)
    x1p = min(w, x1 + pad_x)
    y1p = min(h, y1 + pad_y)

    return (
        rgb[y0p:y1p, x0p:x1p].copy(),
        mask[y0p:y1p, x0p:x1p].copy(),
    )


def upscale_crop_if_small(
    rgb: np.ndarray,
    mask: np.ndarray,
    min_side: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Verbatim copy of the live node's upscale_crop_if_small(): bicubic-
    upscale a small crop so DINOv2's input-size downsample doesn't throw
    away detail."""
    if min_side <= 0:
        return rgb, mask
    h, w = rgb.shape[:2]
    short = min(h, w)
    if short == 0 or short >= min_side:
        return rgb, mask
    scale = float(min_side) / float(short)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    rgb_up = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    mask_up = cv2.resize(
        mask.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST
    ).astype(mask.dtype)
    return rgb_up, mask_up


def batch_dino_classify(
    dino: DINOIdentifier,
    crops_rgb: list[np.ndarray],
    crops_mask: list[np.ndarray | None],
    objectness_priors: list[float | None] | None = None,
) -> list[DINOResult]:
    """Verbatim adaptation of the live node's module-level batch_dino_classify():
    embed `crops_rgb` in one forward pass, then classify each via the ref
    bank. Not imported directly -- run_pipeline_track_multicam_realsense.py
    imports rclpy + fp_debug_msgs at module level, which this script must
    not need."""

    if not crops_rgb:
        return []
    if objectness_priors is not None and len(objectness_priors) != len(crops_rgb):
        raise ValueError(
            f"objectness_priors len {len(objectness_priors)} != crops {len(crops_rgb)}"
        )

    tensors = []
    for rgb, mask in zip(crops_rgb, crops_mask):
        rgb_proc = dino._ensure_rgb(rgb)
        rgb_masked = dino._apply_mask(rgb_proc, mask)
        t = dino._preprocess(rgb_masked)
        tensors.append(t)

    batch = torch.cat(tensors, dim=0)

    with torch.inference_mode():
        out = dino.model.forward_features(batch)
    if not isinstance(out, dict):
        raise RuntimeError(
            f"forward_features returned {type(out)}; expected dict"
        )
    cls_tok = out["x_norm_clstoken"].reshape(batch.shape[0], -1)
    patch_toks = out["x_norm_patchtokens"]  # (B, N, D)
    gem = dino._gem_pool(patch_toks, p=float(dino.cfg.gem_p)).reshape(
        batch.shape[0], -1
    )
    cls_n = F.normalize(cls_tok, dim=1)
    gem_n = F.normalize(gem, dim=1)
    embeddings = torch.stack([cls_n, gem_n], dim=1).detach().cpu().numpy()

    results: list[DINOResult] = []
    for i, emb in enumerate(embeddings):
        prior = None
        if objectness_priors is not None:
            p = objectness_priors[i]
            if p is not None:
                prior = float(p)
        results.append(
            dino.classify_embedding(
                emb,
                objectness_prior=prior,
            )
        )
    return results


def build_dino_identifier(args: argparse.Namespace) -> DINOIdentifier:
    """Build + populate the DINOv2 reference-bank classifier the same way
    the live node's __init__ does (see its "DINO owns object identity"
    block): one bank shared across all cameras for this run."""
    ref_source = args.reference_source
    primary_ref = (
        args.reference_renders_dir if ref_source == "renders"
        else args.reference_dir
    )
    extra_refs: list[str] = []
    if ref_source == "both":
        extra_refs.append(args.reference_renders_dir)

    dino = DINOIdentifier(
        DINOIdentifierConfig(
            model_name=args.dino_model_name,
            device=args.dino_device or args.device,
            reference_dir=primary_ref,
            use_masked_background=False,
            gem_p=float(args.dino_gem_p),
            verbose=True,
        )
    )
    print(
        f"[*] building DINOv2 reference bank | source={ref_source} "
        f"primary={primary_ref}" + (f" extra={extra_refs}" if extra_refs else "")
    )
    dino.build_reference_bank_from_folder(extra_dirs=extra_refs)
    print(
        f"[*] DINOv2 ready | objects="
        f"{sorted(set(r.object_id for r in dino.reference_bank))}"
    )
    return dino


def classify_candidates(
    dino: DINOIdentifier,
    rgb: np.ndarray,
    candidates: list[tuple[np.ndarray, tuple[int, int, int, int], float]],
    dino_min_score: float,
    dino_min_margin: float,
    dino_min_crop_side: int,
) -> list[tuple[float, float, str]]:
    """Classify a list of (full-image mask, full-image bbox, gdino_score)
    candidates, returning one (dino_score, dino_margin, object_id) per
    candidate -- object_id == "unknown" whenever the same accept/reject
    gating the live node's _classify_masks_batched() applies would reject
    it. Order/length matches `candidates`."""
    crops_rgb: list[np.ndarray] = []
    crops_mask: list[np.ndarray] = []
    crops_prior: list[Optional[float]] = []
    valid_indices: list[int] = []

    for i, (mask, bbox, gdino_score) in enumerate(candidates):
        crop_rgb, crop_mask = bbox_crop_with_local_mask(rgb, mask, bbox)
        if crop_rgb.size == 0 or int(crop_mask.sum()) == 0:
            continue
        crop_rgb, crop_mask = upscale_crop_if_small(crop_rgb, crop_mask, dino_min_crop_side)
        crops_rgb.append(crop_rgb)
        crops_mask.append(crop_mask)
        crops_prior.append(gdino_score)
        valid_indices.append(i)

    results: list[tuple[float, float, str]] = [(0.0, 0.0, "unknown")] * len(candidates)
    if not crops_rgb:
        return results

    dino_results = batch_dino_classify(dino, crops_rgb, crops_mask, objectness_priors=crops_prior)

    for j, res in enumerate(dino_results):
        idx = valid_indices[j]
        _mask, bbox, _gdino_score = candidates[idx]

        base_scores = {k: float(v) for k, v in res.scores_by_object.items() if np.isfinite(float(v))}
        if not base_scores:
            continue
        sorted_scores = sorted(base_scores.items(), key=lambda kv: kv[1], reverse=True)
        top1_name, top1_score = sorted_scores[0]
        _top2_name, top2_score = sorted_scores[1] if len(sorted_scores) > 1 else ("", -1.0)

        object_id = top1_name
        decision_best_score = float(top1_score)
        margin = float(top1_score - top2_score)

        if not np.isfinite(decision_best_score) or not np.isfinite(margin):
            results[idx] = (decision_best_score, margin, "unknown")
            continue

        bw, bh = bbox_size_xyxy(bbox)
        is_small_object = (bw * bh) < SMALL_OBJECT_BBOX_AREA_PX

        if is_small_object:
            if decision_best_score < SMALL_OBJECT_MIN_SCORE:
                object_id = "unknown"
            elif margin < SMALL_OBJECT_MIN_MARGIN:
                object_id = "unknown"
        else:
            if decision_best_score < dino_min_score:
                object_id = "unknown"
            if dino_min_margin > 0.0 and margin < dino_min_margin:
                object_id = "unknown"

        results[idx] = (decision_best_score, margin, object_id)

    return results


def backproject_mask_points(
    rgb: np.ndarray,
    depth_m: np.ndarray,
    K: np.ndarray,
    mask: np.ndarray,
    max_depth_m: float,
    min_depth_m: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project mask pixels to camera-frame 3D points + RGB colors.

    Deliberately minimal -- only a finite/positive-range depth check. See
    FILTERING_REPORT for what the live pipeline additionally does that
    this intentionally skips.
    """
    mask_bool = mask.astype(bool)
    vs, us = np.where(mask_bool)
    if len(vs) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3))

    z = depth_m[vs, us].astype(np.float64)
    valid = np.isfinite(z) & (z > min_depth_m) & (z < max_depth_m)
    vs, us, z = vs[valid], us[valid], z[valid]
    if len(z) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3))

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x = (us.astype(np.float64) - cx) * z / fx
    y = (vs.astype(np.float64) - cy) * z / fy
    pts = np.stack([x, y, z], axis=-1)
    cols = rgb[vs, us].astype(np.float64) / 255.0
    return pts, cols


def _draw_axes_inplace(
    img: np.ndarray,
    K: np.ndarray,
    T_cam_obj: np.ndarray,
    axis_len_m: float,
) -> None:
    """Project the object-frame axis triad through T_cam_obj + K and draw it
    (X=red, Y=green, Z=blue). No-ops if the origin is behind the camera."""
    pts_obj = np.array([
        [0.0, 0.0, 0.0],
        [axis_len_m, 0.0, 0.0],
        [0.0, axis_len_m, 0.0],
        [0.0, 0.0, axis_len_m],
    ], dtype=np.float64)
    pts_cam = (T_cam_obj[:3, :3] @ pts_obj.T).T + T_cam_obj[:3, 3]
    z = pts_cam[:, 2]
    if np.any(z <= 1e-6):
        return

    u = K[0, 0] * pts_cam[:, 0] / z + K[0, 2]
    v = K[1, 1] * pts_cam[:, 1] / z + K[1, 2]
    origin = (int(round(u[0])), int(round(v[0])))
    for i, color in enumerate(((255, 0, 0), (0, 255, 0), (0, 0, 255)), start=1):
        tip = (int(round(u[i])), int(round(v[i])))
        cv2.line(img, origin, tip, color, 2, cv2.LINE_AA)


def draw_detections_overlay(
    rgb: np.ndarray,
    K: np.ndarray,
    detections: list["ObjectDetection"],
    axis_len_m: float = 0.05,
) -> np.ndarray:
    """RGB + per-object mask fill, bbox + label, and (once a pose was
    estimated) the projected coordinate-axes triad. Deliberately
    self-contained cv2 drawing rather than importing
    src/perception/ros/learn_runners/overlay_draw_utils.py, which pulls in
    ROS (fp_debug_msgs) at module level -- this script must not need ROS."""
    img = rgb.copy()
    for i, d in enumerate(detections):
        color = DETECTION_PALETTE[i % len(DETECTION_PALETTE)]
        mask = d.mask.astype(bool)

        tinted = img.copy()
        tinted[mask] = color
        img = cv2.addWeighted(tinted, 0.35, img, 0.65, 0)

        x0, y0, x1, y1 = d.bbox_xyxy
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
        cv2.putText(
            img, f"{d.object_id} ({d.dino_score:.2f})", (x0, max(12, y0 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
        )

        if d.T_object_camera is not None:
            _draw_axes_inplace(img, K, np.asarray(d.T_object_camera, dtype=np.float64), axis_len_m)
        elif d.pose_error:
            cv2.putText(
                img, "pose failed", (x0, min(rgb.shape[0] - 4, y1 + 16)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
            )

    return img


def T_to_pose_dict(T: np.ndarray) -> dict:
    """4x4 -> {t, quat_xyzw}, for a human-readable poses_objects_debug.yaml."""
    t = T[:3, 3].tolist()
    R = T[:3, :3]
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    else:
        i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        if i == 0:
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif i == 1:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S
    return {
        "t": [float(v) for v in t],
        "quat_xyzw": [float(qx), float(qy), float(qz), float(qw)],
    }


def process_camera(
    cam_id: str,
    frame_dir: Path,
    gdino: GroundingDINOProposer,
    sam: SAMSegmenter,
    dino: DINOIdentifier,
    fp: FoundationPoseWrapper,
    mesh_map: dict[str, str],
    roi_polygon: np.ndarray,
    args: argparse.Namespace,
) -> bool:
    """Run crop -> GDINO -> SAM -> DINOv2 re-ID -> FoundationPose on one
    camera's raw captured frame, saving pointcloud_objects_debug.ply,
    poses_objects_debug.yaml, and detections_overlay.png into
    frame_dir/OFFLINE_SUBDIR. Returns False if the frame was unusable
    (nothing written), True otherwise -- including when detection/pose came
    up empty, since (empty) outputs are still written in that case."""

    rgb_path = frame_dir / "rgb_native.png"
    depth_path = frame_dir / "depth_m.npy"
    info_path = frame_dir / "frame_info.yaml"

    if not rgb_path.exists() or not depth_path.exists():
        print(f"[{cam_id}] missing rgb_native.png/depth_m.npy in {frame_dir} -- skipping")
        return False

    info = yaml.safe_load(info_path.read_text()) or {} if info_path.exists() else {}
    K = info.get("K")
    if K is None:
        print(
            f"[{cam_id}] frame_info.yaml has no 'K' (camera intrinsics) -- "
            f"re-run capture_pipeline_snapshots.py, it now saves this field -- skipping"
        )
        return False
    K = np.asarray(K, dtype=np.float64)

    rgb = cv2.cvtColor(cv2.imread(str(rgb_path)), cv2.COLOR_BGR2RGB)
    depth_m = np.load(depth_path)
    full_h, full_w = rgb.shape[:2]

    # --- crop to ROI polygon, same order as the live node's
    # _generate_and_filter_masks(): DINO and SAM only ever see this crop. ---
    polygon_full = roi_polygon
    if polygon_full.shape[0] < 3:
        polygon_full = np.array(
            [[0, 0], [full_w, 0], [full_w, full_h], [0, full_h]], dtype=np.int32
        )

    rgb_crop, polygon_crop, crop_x0, crop_y0 = crop_rgb_to_polygon_bbox(rgb, polygon_full)
    crop_h, crop_w = rgb_crop.shape[:2]

    roi_mask_crop = np.zeros((crop_h, crop_w), dtype=np.uint8)
    cv2.fillPoly(roi_mask_crop, [polygon_crop], 255)
    rgb_crop_masked = rgb_crop.copy()
    rgb_crop_masked[roi_mask_crop == 0] = 0

    print(
        f"\n[{cam_id}] ROI crop: {crop_w}x{crop_h} from {full_w}x{full_h} "
        f"(origin {crop_x0},{crop_y0}) -- running Grounding DINO on the crop "
        f"(prompt={gdino._prompt_string!r}) ..."
    )
    proposals = gdino.propose(rgb_crop_masked)
    print(
        f"[{cam_id}] {len(proposals)} DINO box proposal(s) (crop-space): "
        f"{[(p.label, round(p.score, 2)) for p in proposals]}"
    )

    # --- SAM: one mask per GDINO box, lifted to full-image space -----------
    sam_dets: list[tuple[str, np.ndarray, tuple[int, int, int, int], float, float]] = []
    for p in proposals:
        cands = sam.generate_from_boxes(
            rgb_crop_masked,
            np.array([p.bbox_xyxy], dtype=np.float32),
            box_scores=np.array([p.score], dtype=np.float32),
        )
        if not cands:
            print(f"[{cam_id}]   '{p.label}' box -> SAM produced no mask (rejected by SAM's own filters)")
            continue
        c = cands[0]
        mask_full, bbox_full = lift_mask_bbox_to_full(
            c.mask, c.bbox_xyxy, full_h, full_w, crop_x0, crop_y0,
        )
        sam_dets.append((p.label, mask_full, bbox_full, c.score, p.score))

    # --- DINOv2 re-ID: batch-classify every SAM mask against the reference
    # bank, same accept/reject gating as the live node's
    # _classify_masks_batched(). Unknown objects (or ones that don't resolve
    # to a known CAD mesh) are dropped right here -- not tracked, not
    # pointcloud'd, not saved, not drawn.
    classify_in = [(mask, bbox, gdino_score) for (_lbl, mask, bbox, _sam_s, gdino_score) in sam_dets]
    classified = classify_candidates(
        dino, rgb, classify_in,
        dino_min_score=args.dino_min_score,
        dino_min_margin=args.dino_min_margin,
        dino_min_crop_side=args.dino_min_crop_side,
    )

    detections: list[ObjectDetection] = []
    n_unknown = 0
    n_no_mesh = 0
    for (gdino_label, mask_full, bbox_full, sam_score, gdino_score), (dino_score, dino_margin, object_id) in zip(sam_dets, classified):
        if object_id == "unknown":
            n_unknown += 1
            print(
                f"[{cam_id}]   '{gdino_label}' box -> DINOv2 unknown "
                f"(score={dino_score:.3f}, margin={dino_margin:.3f}) -- dropped, not tracked"
            )
            continue

        mesh_path = resolve_mesh_for_label(object_id, mesh_map)
        if mesh_path is None:
            n_no_mesh += 1
            print(
                f"[{cam_id}]   '{gdino_label}' box -> DINOv2 '{object_id}' "
                f"(score={dino_score:.3f}) has no matching CAD mesh -- dropped, not tracked"
            )
            continue

        detections.append(ObjectDetection(
            gdino_label=gdino_label, object_id=object_id, mask=mask_full, bbox_xyxy=bbox_full,
            sam_score=sam_score, gdino_score=gdino_score, dino_score=dino_score,
            dino_margin=dino_margin, mesh_path=mesh_path,
        ))

    print(
        f"[{cam_id}] {len(detections)} known object(s) kept, "
        f"{n_unknown} DINOv2-unknown, {n_no_mesh} unmatched-mesh dropped"
    )

    # --- combined "objects only" point cloud (known objects only) -----------
    union_mask = np.zeros(depth_m.shape[:2], dtype=bool)
    for d in detections:
        union_mask |= d.mask.astype(bool)

    pts, cols = backproject_mask_points(rgb, depth_m, K, union_mask, args.max_depth_m)
    objects_cloud = o3d.geometry.PointCloud()
    if len(pts) > 0:
        objects_cloud.points = o3d.utility.Vector3dVector(pts)
        objects_cloud.colors = o3d.utility.Vector3dVector(cols)

    offline_dir = frame_dir / OFFLINE_SUBDIR
    offline_dir.mkdir(parents=True, exist_ok=True)

    out_ply = offline_dir / "pointcloud_objects_debug.ply"
    o3d.io.write_point_cloud(str(out_ply), objects_cloud, write_ascii=False)
    print(f"[{cam_id}] combined objects cloud: {len(objects_cloud.points)} points -> {out_ply}")

    # --- FoundationPose per (known) object -----------------------------------
    for d in detections:
        object_id = Path(d.mesh_path).stem
        try:
            result = fp.estimate_pose(
                object_id=object_id, mesh_path=d.mesh_path,
                rgb=rgb, depth=depth_m, K=K, mask=d.mask,
            )
            d.T_object_camera = result.T_object_camera
            print(
                f"[{cam_id}]   '{d.object_id}' -> {object_id}: pose estimated "
                f"(mask_area={result.mask_area})"
            )
        except Exception as e:
            d.pose_error = f"FoundationPose failed: {e}"
            print(f"[{cam_id}]   '{d.object_id}' -> {object_id}: {d.pose_error}")

    # --- overlay (bbox + mask + pose axes, known objects only) --------------
    overlay = draw_detections_overlay(rgb, K, detections)
    out_overlay = offline_dir / "detections_overlay.png"
    cv2.imwrite(str(out_overlay), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"[{cam_id}] overlay -> {out_overlay}")

    # --- save poses manifest -------------------------------------------------
    poses_out = []
    for d in detections:
        entry = {
            "gdino_label": d.gdino_label,
            "object_id": d.object_id,
            "gdino_score": float(d.gdino_score),
            "sam_score": float(d.sam_score),
            "dino_score": float(d.dino_score),
            "dino_margin": float(d.dino_margin),
            "bbox_xyxy": list(d.bbox_xyxy),
            "mesh_path": d.mesh_path,
        }
        if d.T_object_camera is not None:
            entry["pose_camera"] = T_to_pose_dict(np.asarray(d.T_object_camera))
        if d.pose_error:
            entry["pose_error"] = d.pose_error
        poses_out.append(entry)
    (offline_dir / "poses_objects_debug.yaml").write_text(
        yaml.safe_dump({"detections": poses_out}, sort_keys=False)
    )

    return True


def run(args: argparse.Namespace) -> None:
    print(FILTERING_REPORT)

    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"Run dir not found: {run_dir}")

    cams = [c.strip() for c in args.cameras.split(",")] if args.cameras else ALL_CAMS

    print("[*] loading models (Grounding DINO, SAM2, DINOv2, FoundationPose) ...")
    if args.gdino_use_items_prompt:
        text_prompts = ["items"]
    else:
        text_prompts = [p.strip() for p in args.gdino_text_prompts.split(",") if p.strip()]
    gdino = GroundingDINOProposer(GDINOConfig(
        model_id=args.gdino_model_id,
        device=args.device,
        box_threshold=args.gdino_box_threshold,
        text_threshold=args.gdino_text_threshold,
        max_boxes_per_image=args.gdino_max_boxes,
        text_prompts=text_prompts,
    ))
    print(f"[*] GDINO prompt(s): {text_prompts}")
    sam = SAMSegmenter(SAMSegmenterConfig(
        repo_root=args.sam_repo_root,
        checkpoint=args.sam_checkpoint,
        model_cfg=args.sam_model_cfg,
        device=args.device,
    ))
    dino = build_dino_identifier(args)
    fp = FoundationPoseWrapper(FoundationPoseConfig(mesh_scale=args.mesh_scale))

    mesh_map = build_mesh_map(args.cad_dir)
    print(f"[*] {len(mesh_map)} mesh name variant(s) resolvable under {args.cad_dir}")

    n_ok = 0
    for cam_id in cams:
        frame_dir = camera_frame_dir(run_dir, cam_id, args.frame)
        if frame_dir is None:
            print(f"[{cam_id}] no frame_{args.frame:02d} in {run_dir} -- skipping")
            continue
        roi_arg = CAM_ROI_ARG.get(cam_id)
        roi_polygon = (
            parse_polygon_string(getattr(args, roi_arg))
            if roi_arg is not None
            else np.zeros((0, 2), dtype=np.int32)
        )
        if process_camera(cam_id, frame_dir, gdino, sam, dino, fp, mesh_map, roi_polygon, args):
            n_ok += 1

    if n_ok == 0:
        raise SystemExit("No camera produced usable results.")

    print(f"\n[*] done -- {n_ok}/{len(cams)} camera(s) processed for frame {args.frame}.")
    print("[*] view the results on a display-capable host (bagviz conda env, no GPU needed):")
    print(
        f"    python -m tools.bagviz.view_object_inference_debug "
        f"--run-dir {run_dir} --frame {args.frame}"
    )

    print(FILTERING_REPORT)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument("--run-dir", required=True,
                    help="Output dir from capture_pipeline_snapshots.py (needs its 'K' field).")
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--cameras", default=None,
                    help="Comma-separated cam_ids; default zed2i_1,realsense_1,realsense_2.")

    p.add_argument("--device", default="cuda")

    p.add_argument("--gdino-model-id", default="IDEA-Research/grounding-dino-base")
    p.add_argument("--gdino-box-threshold", type=float, default=0.20)
    p.add_argument("--gdino-text-threshold", type=float, default=0.25)
    p.add_argument("--gdino-max-boxes", type=int, default=40)
    p.add_argument("--gdino-text-prompts", default=DEFAULT_TEXT_PROMPTS,
                    help="Only used with --no-gdino-use-items-prompt.")
    p.add_argument("--gdino-use-items-prompt", dest="gdino_use_items_prompt",
                    action="store_true", default=True,
                    help="Use the MUSE class-agnostic literal 'items' prompt for GDINO box "
                         "proposals (matches the live node's own default) -- identity comes "
                         "from the DINOv2 classifier below, not from this prompt.")
    p.add_argument("--no-gdino-use-items-prompt", dest="gdino_use_items_prompt",
                    action="store_false",
                    help="Use --gdino-text-prompts for GDINO box proposals instead of 'items'.")

    p.add_argument("--sam-repo-root", default="external/sam2")
    p.add_argument("--sam-checkpoint", default="external/sam2/checkpoints/sam2.1_hiera_base_plus.pt")
    p.add_argument("--sam-model-cfg", default="configs/sam2.1/sam2.1_hiera_b+.yaml")

    # DINOv2 re-ID (mirrors the live node's own --reference-*/--dino-* flags).
    p.add_argument("--reference-dir", default="Data/ZED_screens")
    p.add_argument("--reference-source", choices=["real", "renders", "both"], default="real")
    p.add_argument("--reference-renders-dir", default="Data/reference_renders")
    p.add_argument("--dino-model-name", default="dinov2_vitg14")
    p.add_argument("--dino-device", choices=["", "cpu", "cuda"], default="",
                    help="Device for DINOv2 image embeddings. Empty reuses --device.")
    p.add_argument("--dino-gem-p", type=float, default=1.5)
    p.add_argument("--dino-min-score", type=float, default=0.50)
    p.add_argument("--dino-min-margin", type=float, default=0.05)
    p.add_argument("--dino-min-crop-side", type=int, default=0)

    p.add_argument("--cad-dir", default="Data/CAD_Models_centered")
    p.add_argument("--mesh-scale", type=float, default=0.01,
                    help="CAD meshes are in mm; scale factor to meters.")

    p.add_argument("--max-depth-m", type=float, default=3.0)

    # Same "x0,y0,x1,y1,..." format and cam1/cam2/cam3 = zed2i_1/realsense_1/
    # realsense_2 convention as the live node's own flags -- pass the same
    # value here once you've tuned one there. Empty (default) => no ROI,
    # DINO/SAM see the whole frame (matches the live node's own defaults).
    p.add_argument("--cam1-roi-polygon", type=str, default="",
                    help="zed2i_1 ROI polygon, e.g. 'x0,y0,x1,y1,x2,y2,...'.")
    p.add_argument("--cam2-roi-polygon", type=str, default="",
                    help="realsense_1 ROI polygon.")
    p.add_argument("--cam3-roi-polygon", type=str, default="",
                    help="realsense_2 ROI polygon.")

    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
