# Multicam Tracking Plan

Date: 2026-05-29

## Current Status

The current Cutie + fused ICP tracker can track, but it is brittle for manipulation-style motion.

Observed issues:

- Tracking latency is high: about 607 ms per tick for 6 objects in the clean Cutie static run.
- Fused ICP alone is about 203 ms per tick for 6 objects, so Cutie/per-object tracking and Python/ROS overhead dominate the rest.
- Faster object motion looks like a large jump because the tick rate is low.
- Reinit under memory pressure can still cause CUDA OOM.
- Init is sensitive to lighting and background changes.
- Same-class and similar-shape objects can still cause identity swaps or wrong init labels.
- We only recently added a sanity gate for "projected pose origin must lie inside the tracker mask"; full projected-model overlap is still unchecked.

The good news: when tracking follows correctly, the pose quality can be good. In the clean Cutie static run, good tracks were around 3 mm mean XY error, and tracking improved over init for several objects.

## Lessons From Commit 742ce81

Commit `742ce81202505eb556ece3be1baa0da425cbaa9b` was described as:

> Tracking is now very good estimation while init got worse

Important differences from now:

- The old tracking path was simpler and more direct.
- It keyed trackers by `cam_id + object_id + idx`; current code uses persistent `track_id` to handle same-class instances better.
- The old path did not have the current partial-loss/reinit state machine.
- The old path had less appearance-memory and identity-reuse machinery.
- The old path used fused tracking with Cutie + ICP and chamfer/jump rescue, but before several later additions.
- Current code has lower default tracking ICP iterations than the old commit: current `15`, old commit diff showed `30`.
- Current code added per-track CSV logging, memory crops, partial reinit handling, pose-status hold/lost states, and the new pose-origin-in-mask gate.
- The failed SAM2 realtime tracking branch was removed from the runner; Cutie is now the only realtime video tracker path there.

The likely lesson is not "go back to that commit", because it was weaker for multi-instance identity. The lesson is: the fast/stable tracker should stay lean and should not carry init/recovery/identity machinery in the hot path.

## Recovery Behavior

Desired staged recovery:

1. Good pose: publish.
2. One or a few bad ticks: hold previous pose.
3. Track lost but other tracks healthy: drop only the lost track; keep the rest alive.
4. Local recovery for the lost track only.
5. Full global reinit only if all tracks are lost, or if the user explicitly requests it.

Current state:

- Partial global reinit was fixed: by default, losing one track no longer triggers global init while other tracks persist.
- Old behavior can be forced with `--reinit-lost-tracks-while-tracking`.
- Local recovery is still not ideal. The next recovery step should be per-track and local, not full-scene.

## Why Reject Pose Outside Mask

The current gate rejects a tracking pose if the object-frame origin projects outside the tracker mask:

```bash
--track-require-pose-origin-in-mask
--track-pose-mask-margin-px 8
```

This is intentionally a reject/hold gate, not a correction.

Reason: forcing the pose into the mask is not a clean 6D operation. If we just shift the pose so the origin lands inside the mask, we can create a physically wrong pose and then let ICP refine from a bad seed.

Better future recovery:

1. Reject/hold the current bad pose.
2. Build a recovery seed from the mask depth centroid plus previous orientation.
3. Run local ICP.
4. Accept only if chamfer/RMSE/projected-model overlap are good.

The stronger future gate is projected-model overlap, not just origin-inside-mask.

## Tracking Profiles

### 1. Current Robust Cutie

Goal: best accuracy and safest behavior with the current architecture.

Use this for baseline quality comparisons.

```bash
--track-require-pose-origin-in-mask \
--track-pose-mask-margin-px 8 \
--track-icp-num-points 2000 \
--fused-track-icp-max-iteration 15 \
--median-pose-buffer-size 3 \
--fused-track-max-chamfer-m 0.015 \
--fused-track-max-translation-speed-mps 0.20 \
--fused-track-max-rotation-speed-degps 120 \
--fused-track-min-translation-jump-m 0.015 \
--fused-track-min-rotation-jump-deg 8
```

Expected behavior:

- Most robust of the current profiles.
- Still too slow for fast manipulation.
- Useful for measuring pose quality when tracking follows.

### 2. Fast Cutie

Goal: keep Cutie, but remove or shrink anything not essential to the hot tracking loop.

Starting flags:

```bash
--track-require-pose-origin-in-mask \
--track-pose-mask-margin-px 8 \
--track-icp-num-points 800 \
--fused-track-icp-max-iteration 6 \
--fused-track-icp-relative-fitness 1e-3 \
--fused-track-icp-relative-rmse 1e-3 \
--median-pose-buffer-size 1 \
--fused-track-max-chamfer-m 0.018 \
--fused-track-max-translation-speed-mps 0.25 \
--fused-track-max-rotation-speed-degps 180 \
--fused-track-min-translation-jump-m 0.02 \
--fused-track-min-rotation-jump-deg 10 \
--no-memory-crop
```

Things to skim or disable for Fast Cutie:

- Per-cam ICP during tracking: already skipped by default via `--skip-per-cam-icp-tracking`.
- Median smoothing: set `--median-pose-buffer-size 1` or add a true no-buffer path.
- Kalman soft reject: currently cheap computationally, but can be skipped in a fast profile to simplify gating and reduce brittleness.
- Axis-dominant jump gate: cheap, but potentially over-conservative for real motion; consider disabling or relaxing.
- Chamfer computation: already skipped when fitness/rmse/motion are clean. For fast mode, make skip more aggressive or compute chamfer every N frames only.
- Model points: reduce `--track-icp-num-points` from 2000 to 800, maybe 500.
- ICP iterations: reduce `--fused-track-icp-max-iteration` from 15 to 6, maybe 4 after testing.
- Debug frame publishing: avoid masks and large debug messages in speed runs.
- Verbose logs: keep `--debug-verbose-logs` off.
- Track CSV: use only during evaluation, not in speed measurements.
- Memory crop saving: disable with `--no-memory-crop` in speed tests.
- Appearance memory reranking: not in tracking hot path except at reinit, but disable memory crops for clean timing.
- Per-object memory/crop saving after every stable interval: can cost DINO embedding and crop work; disable in Fast Cutie.
- Realtime tracker pose force update: useful, but check if it costs meaningful time. Keep first, profile later.
- Fused pose publication under many topics: keep only fused pose topic in speed mode if possible.
- Per-camera pose/debug publication: keep off unless actively debugging.
- Multi-object all-at-once tracking: for manipulation, track target-only when possible.

Potential code flags to add for Fast Cutie:

```text
--tracking-profile fast_cutie
--disable-fused-kalman
--disable-axis-jump-gate
--chamfer-every-n-frames N
--no-debug-frame-publish
--publish-fused-only
--target-track-id / --target-object-id
```

### 3. Ultrafast Projected ICP

Goal: remove Cutie from the main tracking loop.

This does not exist yet as a complete path. It needs a new backend:

```text
--tracking-backend projected_icp
```

Loop:

```text
previous pose
-> project CAD silhouette / ROI into each camera
-> lift depth only inside projected ROI
-> short local ICP
-> gate by RMSE, chamfer, projected-model/mask overlap, motion
-> publish fused pose
```

Cutie becomes recovery, not the main loop:

```text
fast projected ICP every tick
Cutie every N frames or on failure
full init only when local recovery fails
```

Expected tradeoff:

- Much faster when previous pose is close.
- Smaller convergence basin than Cutie.
- More brittle under occlusion and fast pose jumps.
- Potentially accurate when depth and CAD fit are good.

## Init Light Sensitivity

The init path is still too light-sensitive.

CLAHE only on live/query crops caused domain mismatch because the reference bank was not processed the same way.

Better options:

1. Apply the same color normalization/CLAHE to both reference-bank crops and live crops.
2. Expand the reference bank with lighting variation.
3. Capture real reference images under the likely lab lighting conditions.
4. Avoid high-contrast checkerboards/backgrounds during init when testing recognition.
5. Use memory crops only after a stable correct init.
6. Consider a detection prompt/reference strategy that is less sensitive to background texture.

## Memory Crops

Memory crops are not useless, but they should not be treated as always-good.

They can help:

- Reinit the same object instance under current session lighting.
- Disambiguate same-class objects after a stable correct track.
- Reduce reliance on old reference-bank images.

They can hurt:

- If the initial identity is wrong, memory reinforces the wrong identity.
- They add crop + DINO embedding work while tracking.
- They can make experiments harder to interpret.

Recommendation:

- Disable for clean speed and tracking tests:

```bash
--no-memory-crop
```

- Enable only for recovery/production-style tests after basic tracking is stable.
- Keep track consistency required for rerank.

## Immediate Test Plan

1. Run current robust Cutie with the pose-origin-in-mask gate.
2. Confirm:
   - bad pose-outside-mask cases become rejected/held;
   - losing one track does not global-reinit while others survive;
   - XY error and chamfer stay reasonable when tracking follows.
3. Run Fast Cutie flags with memory crop off and reduced ICP cost.
4. Compare:
   - `FUSED TRACK total`;
   - accepted ratio;
   - mean/p95 XY error;
   - chamfer/RMSE;
   - visible drift/identity swaps.
5. Only after that, implement projected-ICP backend.

## Sequential TODO

High-level order:

1. Test current robust Cutie now.
2. Test current Cutie with fast-ish flags before adding new code.
3. Implement the Fast Cutie profile.
4. Test Fast Cutie.
5. Implement Ultrafast projected ICP.
6. Test Ultrafast projected ICP.
7. Improve based on findings from all tracking tests.
8. Tackle full recovery behavior.
9. Tackle init lighting sensitivity.

Note: do minimal recovery fixes immediately when they block testing, but do the full recovery design after choosing which tracking backend is worth keeping. If init lighting becomes too unstable to run tracking tests, move the lighting/reference-bank work earlier.

### Phase 0: Clean Baseline

Implementation:

- [x] Remove failed SAM2 realtime tracking path from the runner.
- [x] Remove SAM2 realtime tracker config/imports from `RealtimeTracker`.
- [x] Keep SAM/SAM2 segmentation assets for init, because `gdino_sam` still needs them.
- [x] Add track-pose CSV logging.
- [x] Add partial-loss behavior: dropping one lost track no longer forces global reinit while other tracks survive.
- [x] Add opt-in pose-origin-in-mask gate.

Test:

- [ ] Run robust Cutie on a simple 2-object scene.
- [ ] Use `--log-track-poses` and save logs under `outputs/logs/`.
- [ ] Keep objects static for at least 1-2 minutes after init.
- [ ] Move one object slowly once tracking is stable.
- [ ] Check `FUSED TRACK total`, accepted ratio, XY error, chamfer/RMSE, and lost/held/stale counts.
- [ ] Confirm partial loss does not trigger global reinit.
- [ ] Confirm pose-outside-mask cases are rejected/held instead of published.

### Phase 1: Make Robust Cutie Trustworthy

Implementation:

- [ ] Replace origin-only mask gate with projected-model overlap gate.
- [ ] Keep origin gate as a cheap optional fallback.
- [ ] Add a local recovery candidate when pose is outside mask:
  previous orientation + mask depth centroid -> short ICP -> accept only if metrics are good.
- [ ] Add a per-track local recovery path before full init.
- [ ] Make full global reinit happen only when all tracks are gone or a flag explicitly requests it.
- [ ] Add clearer log reasons for rejected/held/stale/lost poses.

Test:

- [ ] Repeat the 2-object static + slow-motion test.
- [ ] Intentionally occlude one object briefly.
- [ ] Confirm the other object keeps tracking.
- [ ] Confirm the lost object either recovers locally or is dropped cleanly.
- [ ] Compare XY error before/after projected-model overlap gate.
- [ ] Check that no valid poses are falsely rejected because of mesh-origin placement.

### Phase 2: Fast Cutie Profile

Implementation:

- [ ] Add `--tracking-profile robust_cutie|fast_cutie` or equivalent preset helper.
- [ ] Add `--disable-fused-kalman`.
- [ ] Add `--disable-axis-jump-gate`.
- [ ] Add `--chamfer-every-n-frames`.
- [ ] Add `--no-debug-frame-publish`.
- [ ] Add `--publish-fused-only`.
- [ ] Add target-only tracking selection:
  `--target-track-id` or `--target-object-id`.
- [ ] Make memory crop saving easy to disable in speed profiles.
- [ ] Consider skipping `rt.force_pose_update()` in fast mode only if profiling shows it matters.

Test:

- [ ] Run Fast Cutie on the same 2-object scene.
- [ ] First run with objects static.
- [ ] Second run with slow movement.
- [ ] Third run with faster movement.
- [ ] Compare against robust Cutie:
  `FUSED TRACK total`, `Fused ICP all objects`, accepted ratio, XY error, chamfer/RMSE, identity stability.
- [ ] Try `track-icp-num-points` values: 800, 500, 300.
- [ ] Try `fused-track-icp-max-iteration` values: 6, 4.
- [ ] Decide the best speed/accuracy tradeoff.

### Phase 3: Init Light Robustness

Implementation:

- [ ] Add shared live/reference image normalization path.
- [ ] If using CLAHE, apply it to both reference-bank crops and live/query crops.
- [ ] Add a way to rebuild or invalidate DINO cache when preprocessing changes.
- [ ] Expand reference images with lighting variation.
- [ ] Add optional memory-crop use only after tracks are stable and correctly identified.

Test:

- [ ] Test init under normal lighting.
- [ ] Test init under changed lighting.
- [ ] Test with and without checkerboard/background texture.
- [ ] Compare object selection, especially cooling_screw vs cooling_f and cooling_base vs pb_pipe.
- [ ] Confirm the fix helps detection without hurting the previous baseline.

### Phase 4: Memory Crop Decision

Implementation:

- [ ] Keep `--no-memory-crop` as the default for clean tracking experiments if we decide speed/interpretability matters more.
- [ ] If memory crops stay enabled by default, add stricter save conditions:
  stable track, correct class, good chamfer, good mask overlap, not warmup.
- [ ] Add a simple debug summary of which memory crops influenced reinit decisions.

Test:

- [ ] Run reinit tests with `--no-memory-crop`.
- [ ] Run the same reinit tests with memory crops enabled.
- [ ] Compare whether memory crops reduce identity swaps or reinforce wrong identities.
- [ ] Keep memory crops only if they clearly help reinit.

### Phase 5: Ultrafast Projected ICP Backend

Implementation:

- [ ] Add `--tracking-backend cutie|projected_icp`.
- [ ] Implement mesh projection from previous pose into each camera.
- [ ] Build projected ROI or silhouette mask.
- [ ] Lift depth only inside projected ROI.
- [ ] Run short local ICP from previous pose.
- [ ] Fuse per-camera clouds or per-camera pose candidates.
- [ ] Gate by RMSE, chamfer, projected-model overlap, depth coverage, and motion.
- [ ] Use Cutie only as recovery, not every frame.
- [ ] Add fallback from projected ICP -> Cutie recovery -> local reinit -> full init.

Test:

- [ ] Start with 1 object static.
- [ ] Then 1 object slow motion.
- [ ] Then 2 objects static.
- [ ] Then 2 objects slow motion.
- [ ] Only then test 6 objects.
- [ ] Compare projected ICP against robust and fast Cutie:
  latency, XY error, pose stability, failure cases.
- [ ] Check whether projected ICP can reach the manipulation target of roughly 10-15 Hz for 1-2 active objects.

### Phase 6: Final Manipulation Profile

Implementation:

- [ ] Decide the production profile:
  robust Cutie, fast Cutie, projected ICP, or hybrid.
- [ ] Add one documented command for the final thesis/demo setup.
- [ ] Add one documented command for evaluation logging.
- [ ] Add one documented command for speed-only timing.

Test:

- [ ] Run the final profile on the intended manipulation scene.
- [ ] Report final mean/p95 latency.
- [ ] Report final mean/p95 XY error.
- [ ] Report failure/recovery behavior.
- [ ] Confirm the robot-use case:
  static pick/place, slow correction, or reactive visual servoing.

## Metrics To Keep Reporting

For every tracking profile:

- Mean/p50/p95 `FUSED TRACK total`.
- Mean/p50/p95 `Fused ICP all objects`.
- Accepted ratio.
- Lost/held/stale counts.
- Mean XY error per object.
- Mean XY error excluding known wrong identities.
- Max XY jump and standard deviation.
- Chamfer/RMSE distribution.
- Number of reinit attempts.
- Number of OOMs or tracker resets.

## Bottom Line

The current tracker is a good research baseline, but not yet a manipulation-speed tracker. The right direction is:

1. Stabilize robust Cutie.
2. Strip it down into Fast Cutie.
3. Build Ultrafast Projected ICP.
4. Keep recovery local and per-track.
5. Fix init light sensitivity by matching live/reference preprocessing.
