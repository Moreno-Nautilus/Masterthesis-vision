# Joint Bundle-Adjustment Hand-Eye Calibration (dual-arm, CAD-prior)

`calibrate_handeye.py --method joint` (the default): instead of solving each
arm's `T_flange_cam` independently via closed-form `cv2.calibrateHandEye`
(AX=XB, `--method direct`), this jointly refines **both** arms at once
against every checkerboard corner's reprojection error, with two extra
priors that the closed-form solve has no way to use:

1. Both arms carry the same physical camera mount, so their `T_flange_cam`
   values should be close to each other — a soft prior pulls them together.
2. A CAD-derived nominal offset already exists (`realsense_nominal` in
   `config/camera_extrinsics_realsense.yaml`) — used as both the initial
   guess and a second soft prior.

It reads whatever samples `capture_handeye_data.py` already captured under
`outputs/calibration_debug/handeye/<cam_id>/` — see
[calibration_cheatsheet.md](calibration_cheatsheet.md) for Stage A/B of the
routine this fits into.

> **This is not a capture step.** `calibrate_handeye.py` never moves the
> robot and never grabs an image — it's pure offline re-optimization over
> `sample_*.json` files that `capture_handeye_data.py` already wrote. If
> that directory is empty for the arm(s) you care about, run Stage A first
> ([calibration_cheatsheet.md](calibration_cheatsheet.md)) — this script has
> nothing to do until then.

---

## Quick start

Assumes you already have captured samples for at least one arm under
`outputs/calibration_debug/handeye/<cam_id>/` (from `capture_handeye_data.py`).

```bash
# Dry run first -- prints the solved T_flange_cam + diagnostics, doesn't touch
# config/camera_extrinsics_realsense.yaml. --method joint is the default.
python3 -m src.calibration.calibrate_handeye -v
```

Check the printed diagnostics (see [Interpreting the output](#interpreting-the-output)
below) look sane, then write the result:

```bash
python3 -m src.calibration.calibrate_handeye --write
```

✅ Checkpoint: `pose residual` / `reprojection error (px)` lines are small (a
fraction of a degree and a few mm for pose residuals; well under 1px for
reprojection) for every arm listed. If not — don't `--write`; see
[Troubleshooting](#troubleshooting) below before trusting the result.

Works fine with only **one** arm's data present (e.g. only `realsense_2` captured so
far): it just solves that arm alone with the CAD-nominal prior; the cross-arm prior
kicks in automatically once both arms have samples.

Useful flags (all `--method joint`-specific):

```bash
--cam-ids realsense_1              # solve just one arm instead of both
--no-nominal                       # disable the CAD nominal init/prior entirely
--shared-mount-sigma-deg / -m      # loosen/tighten the cross-arm prior (default 2deg / 1cm)
--nominal-sigma-deg / -m           # loosen/tighten the CAD prior (default 5deg / 2cm)
--loss cauchy --f-scale 2.0        # opt into robust loss -- see the gotcha below first
```

For the original closed-form solve instead, pass `--method direct`
(no priors, no tuning knobs, per-arm independent) — see
[calibration_cheatsheet.md](calibration_cheatsheet.md).

---

## Where this lives

The actual bundle-adjustment solver is **not** in this repo — it's a standalone
Python package/git submodule at
[`src/calibration/joint_handeye_calib/`](../src/calibration/joint_handeye_calib/)
(own tests, own README, own commit history — see that README for the math/API
details). `src/calibration/calibrate_handeye.py`'s `--method joint` path is the
thin integration code that loads this repo's sample data and CAD nominal
transform, calls into that package, and writes the result into
`config/camera_extrinsics_realsense.yaml` using the same
`update_extrinsics_yaml_preserving_header` helper the other calibration scripts use
(backs up the previous file to `.yaml.bak` first).

**The submodule's remote is currently local-only** (`.gitmodules` points at
`/home/pdzuser/repos/joint_handeye_calib` on disk, no GitHub repo created yet) —
migrate by pushing that directory to a real remote, updating the url in
`.gitmodules`, and running `git submodule sync` once one exists.

## Why not just use the closed-form solve (`--method direct`)?

`cv2.calibrateHandEye` only ever sees the *solved* board pose per sample (via PnP),
and solves each camera completely independently — it has no way to use a CAD prior,
and no way to let a well-conditioned arm's data help a less-well-conditioned one.
The joint solve:

- Minimizes actual per-corner reprojection error (when available — see below),
  not just the already-summarized board pose.
- Ties both arms' `T_flange_cam` together with a soft prior, so if one arm's capture
  session had poor rotational diversity (a classic AX=XB-degenerate case), the
  other arm's better-conditioned data pulls it toward a sane answer instead of
  leaving it essentially unconstrained.
- Starts from, and is softly pulled toward, the CAD nominal offset instead of an
  arbitrary/identity guess.

Camera intrinsics are **not** jointly optimized (matching the reference method this
is based on — [Allegro/Terreran/Ghidoni, RA-L/ICRA
2025](https://github.com/davidea97/Multi-Camera-Hand-Eye-Calibration)): this rig's
images are already rectified upstream (zero distortion assumed), and refining
intrinsics would need raw/unrectified images with a real distortion model, which the
capture pipeline doesn't currently produce. Intrinsics are taken as fixed, per-sample
`K` from the RealSense driver's own `camera_info`.

## Reprojection vs. pose-level residuals (mixed sample fidelity)

`capture_handeye_data.py` persists the raw detected checkerboard corner
pixels + intrinsics (`corners_px`, `K`) alongside each sample, in addition to the
solved board pose it always saves. Samples with these fields feed a true 2D
reprojection residual; **older captures that don't have them** (e.g. any
`sample_*.json` written by a pre-refactor capture script) automatically fall back to a
6-DOF pose-level residual on the already-solved board pose instead. Both kinds mix
freely in the same solve — nothing needs recapturing to benefit from the joint,
multi-arm, CAD-prior solve.

## Interpreting the output

For each arm, the script prints:

- `T_flange_cam` — the solved flange-to-camera transform (what gets written to the
  YAML).
- `reprojection error (px)` — mean/max corner reprojection error, only for samples
  that have `corners_px`/`K`.
- `pose residual (legacy samples)` — mean/max rotation (deg) and translation (mm)
  disagreement between the solved board pose and what the fitted `T_flange_cam` +
  `T_base_board` predict, for samples without `corners_px`/`K`.
- `vs. CAD nominal` — how far the solved `T_flange_cam` ended up from
  `realsense_nominal`.
- `[a vs b] flange-cam disagreement` — how far apart the two arms' solved
  `T_flange_cam` ended up (only printed once both arms are included).

A large `vs. CAD nominal` number isn't necessarily a bug — it just means this
particular camera's actual as-built mount differs substantially from the CAD design
value. (Concretely: `realsense_2`'s real calibration currently differs from
`realsense_nominal` by ~140°, which is well outside what the priors' default sigmas
assume as "manufacturing tolerance" — worth checking whether `realsense_nominal`'s
CAD derivation actually matches how the camera is physically mounted before trusting
it as a strong prior for both arms.)

## Troubleshooting

**Result looks wildly wrong (tens of degrees / centimeters of pose residual), but
the same samples solve fine with `--method direct`'s `cv2.calibrateHandEye`:**
almost certainly the `--loss` choice. The default is `linear` for exactly this
reason — a robust loss (`soft_l1`/`cauchy`, closer to what the reference paper
uses) down-weights *every* residual near-uniformly when the initial guess is more
than a few degrees off (nothing looks like a "good" inlier to anchor to yet), which
can stall the solver within a few dozen iterations, far from the optimum. This was
caught during verification: `cauchy` converged to 50-70° of error on a real capture
session starting from the CAD nominal guess, where `linear` converged to <1°. Don't
pass `--loss cauchy`/`soft_l1` unless you have a specific outlier-rejection need and
have verified convergence from your actual initial guess looks correct.

**Only one arm prints results:** the other arm has no `sample_*.json` files under
`outputs/calibration_debug/handeye/<cam_id>/` yet — capture some with
`capture_handeye_data.py` (Stage A of the routine) first.
