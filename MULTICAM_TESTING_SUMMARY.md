# Multicam Testing Summary

Date: 2026-06-05

Handoff summary for continuing multicam init/tracking testing in a new chat.

## Current GT

Units are meters.

```text
pb_base        (-0.406,   0.215)
pb_top         (-0.108,  -0.195)
pb_pipe        (-0.0475,  0.500)   # revised x, roughly -4.7/-4.8 cm
pb_screw1      (-0.253,  -0.040)
pb_screw2      (-0.260,   0.432)
cooling_base   ( 0.000,   0.3065)
cooling_screw1 ( 0.045,  -0.016)
cooling_screw2 ( 0.093,   0.030)
cooling_screw3 (-0.151,   0.210)
cooling_screw4 (-0.174,   0.117)
cooling_f      (-0.216,   0.224)
```

## Usability Rule Used

Runs counted as usable only if they had `MULTICAM INIT TOTAL`, no CUDA/traceback, all 11 GT objects matched, no low-mask camera collapse, and no very large matched XY error above about 30 mm. Extra wrong detections were excluded from object accuracy averages.

## Best Baseline Init Command

Init only, no tracking:

```bash
python -m src.perception.ros.learn_runners.run_pipeline_track_multicam \
  --num-cameras 3 \
  --mask-source gdino_sam \
  --gdino-device cpu \
  --gdino-box-threshold 0.30 \
  --gdino-text-threshold 0.20 \
  --gdino-max-boxes 20 \
  --sam-max-image-side 1536 \
  --reference-source real \
  --dino-min-crop-side 112 \
  --icp-grid-n-rot 60 \
  --icp-grid-prescreen \
  --icp-grid-cross-cam-chamfer \
  --depth-fill-holes-kernel 3 \
  --run-mode init_only \
  --tracking-backend cutie \
  --tracking-profile default \
  --debug-logging \
  --debug-verbose-logs \
  --log-init-poses \
  --track-pose-log-path outputs/logs/multicam_init_baseline_current.csv \
  2>&1 | tee outputs/logs/multicam_init_baseline_current.log
```

## Baseline Tracking Command

Same baseline init config, but with tracking:

```bash
python -m src.perception.ros.learn_runners.run_pipeline_track_multicam \
  --num-cameras 3 \
  --mask-source gdino_sam \
  --gdino-device cpu \
  --gdino-box-threshold 0.30 \
  --gdino-text-threshold 0.20 \
  --gdino-max-boxes 20 \
  --sam-max-image-side 1536 \
  --reference-source real \
  --dino-min-crop-side 112 \
  --icp-grid-n-rot 60 \
  --icp-grid-prescreen \
  --icp-grid-cross-cam-chamfer \
  --depth-fill-holes-kernel 3 \
  --run-mode track \
  --tracking-backend cutie \
  --tracking-profile default \
  --debug-logging \
  --debug-verbose-logs \
  --log-track-poses \
  --track-pose-log-path outputs/logs/multicam_track_baseline_current.csv \
  2>&1 | tee outputs/logs/multicam_track_baseline_current.log
```

## Results So Far

All stats use current GTs and usable runs only.

### Baseline

Log: `outputs/logs/multicam_init_baseline_3cam_fixed.log`

```text
completed: 43
usable:    39
all XY:    mean 3.94 mm, median 3.54 mm, max 20.34 mm
no pipe:   mean 3.45 mm, median 3.43 mm, max 10.32 mm
time:      mean 76.22 s, median 75.82 s
extras:    cooling_base 39, cooling_f 15, pb_top 1
chamfer:   all mean 3.30 mm, no-pipe mean 2.91 mm
```

Verdict: best default so far. Fastest and best no-pipe accuracy.

Known baseline biases:

```text
pb_base:       slight +x
pb_screw2:     slight +x
cooling_base:  slight -x / -y
pb_pipe:       often off; GT may also be less certain
```

### Full Second ICP Pass

Extra flags:

```text
--icp-grid-second-pass
--icp-grid-second-pass-k 3
--icp-grid-second-pass-n 8
--icp-grid-second-pass-jitter-deg 15.0
```

Log: `outputs/logs/multicam_init_second_icp.log`

```text
usable:    9/10
all XY:    mean 3.59 mm, median 3.04 mm, max 6.71 mm
no pipe:   mean 3.74 mm, median 3.75 mm, max 6.71 mm
time:      mean 92.47 s, median 91.57 s
extras:    cooling_f 7
chamfer:   all mean 3.17 mm, no-pipe mean 3.06 mm
```

Verdict: mixed. It helps `pb_pipe` and worst-case pipe behavior, but worsens non-pipe mean and costs about +16 s. Not default.

### ICP Grid n_rot 90

Changed flag:

```text
--icp-grid-n-rot 90
```

Log: `outputs/logs/multicam_init_icp_nrot90.log`

```text
usable:    8/11
all XY:    mean 4.25 mm, median 3.91 mm, max 19.26 mm
no pipe:   mean 3.73 mm, median 3.91 mm, max 12.37 mm
time:      mean 92.86 s, median 91.14 s
extras:    cooling_f 7
```

Verdict: reject. Slower than baseline and no accuracy win. Also had low-mask discarded runs.

### Tuned Smaller Second ICP, k2 n6 j10

Extra flags:

```text
--icp-grid-second-pass
--icp-grid-second-pass-k 2
--icp-grid-second-pass-n 6
--icp-grid-second-pass-jitter-deg 10.0
```

Log: `outputs/logs/multicam_init_second_icp_k2n6j10.log`

```text
completed: 5
usable:    3
all XY:    mean 4.17 mm, median 4.00 mm, max 15.08 mm
no pipe:   mean 3.98 mm, median 4.16 mm, max 7.32 mm
time:      mean 88.17 s, median 87.27 s
extras:    cooling_f 1
```

Verdict: reject. Faster than full second pass but still slower and worse than baseline. Had 2/5 low-mask runs.

### ICP Grid Tie By Inliers

Extra flag:

```text
--icp-grid-tie-by-inliers
```

Log: `outputs/logs/multicam_init_tie_inliers.log`

```text
completed: 5
usable:    3
all XY:    mean 6.41 mm, median 5.19 mm, max 14.80 mm
no pipe:   mean 6.09 mm, median 5.18 mm, max 11.49 mm
time:      mean 87.70 s, median 85.27 s
extras:    cooling_f 3
```

Bad object shifts:

```text
pb_pipe:       mean 9.62 mm, dx -9.57 mm
pb_screw1:     mean 9.74 mm, dy +9.70 mm
pb_screw2:     mean 9.29 mm, dx +9.07 mm
pb_top:        mean 8.08 mm, dx +6.60 mm, dy +4.67 mm
cooling_base:  mean 7.79 mm, dx -4.10 mm, dy +6.53 mm
```

Verdict: hard reject.

## Current Camera / Stability Issue

Later baseline sanity log:

```text
outputs/logs/multicam_init_baseline_sanity_current.log
```

Result:

```text
completed cycles: 3
usable cycles:    0
then crash:       CUDA illegal memory access
```

Main problem is `zed2i_2` / cam2:

```text
Run 1:
  cam2: 15 boxes -> 7 masks, 6 selected

Runs 2-4:
  cam2 image mean 26.6, max 134
  GDINO: 1 huge box across almost the whole crop
  SAM:   1 box -> 0 masks, rejected area_large
```

This means the current test bench is not in the same state as the good baseline. Likely causes:

```text
- cam2 view blocked or partly blocked
- exposure/lighting changed
- wrong/dark/frozen cam2 image
- ROI/image state changed
```

Do not continue flag tests until cam2 is healthy again.

## CUDA Error Seen

Recurring failure:

```text
CUDA error: an illegal memory access was encountered
```

It appeared during FoundationPose/nvdiffrast work:

```text
FP failed ... register() failed ... Cuda error: 700[cudaStreamSynchronize(stream);]
FP failed ... _build_estimator() failed ... CUDA error: an illegal memory access was encountered
```

After that the CUDA context was poisoned and cleanup also failed:

```text
torch.cuda.empty_cache()
RuntimeError: CUDA error: an illegal memory access was encountered
```

Practical rule: after this happens, restart the Python process. `torch.cuda.empty_cache()` alone is not enough.

## Rejected Flags / Configs

Do not add these to the current baseline:

```text
--icp-grid-second-pass              # mixed; only helped pipe, costs runtime
--icp-grid-n-rot 90                 # slower, not better
--icp-grid-second-pass-k 2
--icp-grid-second-pass-n 6
--icp-grid-second-pass-jitter-deg 10.0
--icp-grid-tie-by-inliers           # clear accuracy regression
```

## Remaining Testing Plan

### Step 1: Recover Clean Baseline

Fix cam2 first, then rerun 5-10 baseline init-only cycles.

Pass condition:

```text
cam2: 10+ masks normally, no repeated 1-box/0-mask failure
no CUDA crash
usable rate close to baseline
no-pipe mean around 3.5-4.0 mm
```

### Step 2: Depth Fill Tuning

Next real flag test after baseline is clean:

```text
--depth-fill-holes-kernel 5
```

Compare to current baseline `--depth-fill-holes-kernel 3`.

If kernel 5 helps, optionally try:

```text
--depth-fill-holes-kernel 7
```

If kernel 5 worsens or does not help, keep kernel 3.

### Step 3: Weighted Cloud Merge

Test later:

```text
--use-weighted-cloud-merge
```

If promising, tune:

```text
--cloud-merge-distance-exponent 1.0
--cloud-merge-distance-exponent 3.0
```

Note: check whether this affects init-only in the current code path or mainly tracking/fused tracking.

### Step 4: CLAHE

Test:

```text
--clahe-enabled
```

If promising, tune:

```text
--clahe-clip-limit 1.5
--clahe-clip-limit 2.5
--clahe-grid-size 8
--clahe-grid-size 12
```

### Step 5: Detection Thresholds

Only tune after geometry/visibility is stable:

```text
--dino-min-score
--gdino-box-threshold 0.25 or 0.35
--gdino-text-threshold 0.15 or 0.25
```

These change candidate population, so test them late.

### Step 6: ROI / SAM Camera-Specific Filters

If cam2 continues to make huge boxes, tune ROI or filters before testing more ICP flags:

```text
--cam2-roi-polygon ...
--cam2-sam-max-mask-area-ratio
--cam2-sam-max-bbox-area-ratio
```

### Step 7: Tracking Baseline

Once init is stable, run baseline tracking and measure:

```text
- drift
- lost/missed objects
- per-object jumps
- fused vs single-cam stability
- runtime/frame
```

