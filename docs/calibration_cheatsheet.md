# Dual-Arm RealSense Calibration — Execution Cheatsheet

Commands only. For controller/mode background see
[calibration_control_modes.md](calibration_control_modes.md); for the joint
bundle-adjustment solver's math/troubleshooting see
[joint_handeye_calibration.md](joint_handeye_calibration.md).

Assumes: dual-arm hardware is racked, checkerboard (8×11 inner corners, 30mm
squares) is on hand, and you're starting from nothing running.

---

## Quickstart

**Only the ZED needs recalibrating and the checkerboard hasn't moved?**
Skip everything below and run just:

```bash
python3 -m src.calibration.autocalibrate_dual_realsense --zed-only
```

Reuses the `T_base_board` already in `config/base_board_pose.yaml`, no arm
motion or MoveGroup/hand-eye needed — see
[`autocalibrate_dual_realsense.py`](#autocalibrate_dual_realsensepy-board-pose--zed)
in the flags reference.

Reasonable defaults, from nothing running to a finished calibration. Order:
**host camera stack, then robot bring-up, then calibration commands inside
the `vision` container.** See the numbered sections below for what each step
actually does, and [Flags reference](#flags-reference) at the very end for
every flag any of these scripts takes.

```bash
# 1. Host camera stack (own terminal, on the host)
scripts/launch_host_realsense.sh
```

```bash
# 2. Robot bring-up (two more terminals, on the host)
ros2 launch lbr_dual_arm_bringup hardware.launch.py use_gripper:=true
ros2 launch lbr_dual_arm_bringup move_group.launch.py mode:=hardware rviz:=true use_gripper:=true
# then start the LBRServer app on each pendant you're actually using
```

```bash
# 3. Calibration (inside the 'vision' container)
python3 -m src.calibration.capture_handeye_data                  # Stage A: jog 7 poses/arm via RViz
python3 -m src.calibration.capture_handeye_data --mode augment   # -> 10 well-conditioned samples/arm
python3 -m src.calibration.calibrate_handeye --write              # Stage B: solve T_flange_cam
python3 -m src.calibration.autocalibrate_dual_realsense           # Board pose + ZED (auto-augments to 5)
```

Only one arm connected? Add `arms:=lbr_one` (left) or `arms:=lbr_two`
(right) to every bring-up launch above, and `--arm left`/`--arm right` to
`capture_handeye_data.py` — everything else is unchanged, including
`autocalibrate_dual_realsense.py` (no `--arm` flag needed there; it just
uses whichever arm(s) actually produce a successful sample).

---

## 0. Bring everything up

```bash
# Terminal 1 — host camera stack (ZED + both RealSense + flange_pose_publisher x2)
scripts/launch_host_realsense.sh
```

```bash
# Terminal 2 — hardware interface (both arms)
# use_gripper defaults to true (Y-gripper attached, arm_one/arm_two tipped at
# the gripper TCP); pass use_gripper:=false for the bare flange instead.
#
# Default Stage A controller (moveit/RViz-jogged) needs hardware.launch.py.
# Swap for `admittance.launch.py` or `calibration.launch.py` if you're using
# --controller admittance / handguided instead — see step 1 below and
# calibration_control_modes.md for full launch-param reference.
ros2 launch lbr_dual_arm_bringup hardware.launch.py use_gripper:=true

# Only have/control one arm? Add arms:=lbr_one (left) or arms:=lbr_two
# (right) to hardware.launch.py / admittance.launch.py / calibration.launch.py
# — the other arm loads as a mock component, no pendant/hardware needed for it:
ros2 launch lbr_dual_arm_bringup admittance.launch.py arms:=lbr_one use_gripper:=true
```

```bash
# Terminal 3 — MoveIt + RViz (needed for --controller moveit AND for
# --mode replay/augment; NOT needed for --controller admittance/handguided).
# use_gripper must match Terminal 2's value.
ros2 launch lbr_dual_arm_bringup move_group.launch.py mode:=hardware rviz:=true use_gripper:=true
```

Then on **each pendant you're actually using** (left first, then right, if
both): start the `LBRServer` app to open the FRI connection Terminal 2 is
waiting on. With `arms:=lbr_one`/`lbr_two`, only that one pendant needs it.

**Verify before continuing** (separate shell):

```bash
ros2 topic echo /left/ee_pose --once
ros2 topic echo /right/ee_pose --once
ros2 topic hz /realsense_1/camera/color/image_rect
ros2 topic hz /realsense_2/camera/color/image_rect
ros2 topic hz /zed2i_1/zed_node/rgb/color/rect/image
```

All five must return real data. Place the checkerboard where **both** arms
can see it and **do not move it again** until Step 3 (board pose) finishes.

---

## 1. Stage A — capture flange poses + checkerboard images (manual, ~5 min per arm)

One script, both arms by default, controller/mode picked by flags:

```bash
python3 -m src.calibration.capture_handeye_data
```

This defaults to `--arm both --controller moveit --mode interactive --num-samples 7`
— jog each arm in turn via RViz's MotionPlanning panel (Plan & Execute), 7
poses left then 7 poses right, in one run. For each prompt: vary orientation
as much as position, let the arm settle, press **Enter**.

**Alternatives** — pick one `--controller` and use it for the whole session
(the positioning method doesn't matter to what gets saved, so left/right can
even use different controllers if you want, though there's no reason to):

```bash
# Admittance-guided (software compliance, position interface) — needs:
#   ros2 launch lbr_dual_arm_bringup admittance.launch.py use_gripper:=true
python3 -m src.calibration.capture_handeye_data --controller admittance
python3 -m src.calibration.capture_handeye_data --controller admittance --gain-profile insertion

# Gravity-compensation hand-guiding (torque interface) — needs:
#   ros2 launch lbr_dual_arm_bringup calibration.launch.py use_gripper:=true
python3 -m src.calibration.capture_handeye_data --controller handguided

# One arm only:
python3 -m src.calibration.capture_handeye_data --arm left --controller admittance
```

`--controller admittance` stays one-arm-at-a-time under the hood even with
`--arm both` (running two admittance loops concurrently roughly halves the
achievable control rate for the arm you're guiding — see
[calibration_control_modes.md §3](calibration_control_modes.md#3-admittance--force-driven-compliance-on-the-position-interface));
the script just runs left-then-right in one process instead of needing two
terminals.

Every accepted sample saves BOTH the flange pose + joint configuration
(`config/flange_poses/<arm>.json`) AND the checkerboard image + detected
corners + intrinsics (`outputs/calibration_debug/handeye/<cam_id>/sample_NN.json`
+ `.png`) — the latter is what lets Stage B (next) run completely offline,
no robot or camera needed.

✅ Checkpoint: `config/flange_poses/left.json` and `right.json` each have 7
entries, and `outputs/calibration_debug/handeye/realsense_1/` /
`realsense_2/` each have 7 `sample_*.json` + `.png` pairs. (`--append` to add
more later without starting over — only if the checkerboard hasn't moved
since; otherwise see "Recalibrating later" below.)

**Standard next step — augment to 10 well-conditioned samples/arm** (no
manual positioning; works with however many arms are connected):

```bash
python3 -m src.calibration.capture_handeye_data --mode augment
```

Replays the 7 poses above, keeps whichever N still detect the checkerboard,
then perturbs random already-successful poses (±3cm/±7° per axis — see
`src/calibration/pose_augmentation.py`) and drives there via Cartesian IK to
fill the gap to `--target-samples` (default 10), retrying on any failure
(unreachable, no detection, reprojection too high). Rewrites
`config/flange_poses/<arm>.json` and the debug dir with the full resulting
set (archiving what was there first).

---

## 2. Stage B — solve T_flange_cam (offline, no hardware needed)

```bash
python3 -m src.calibration.calibrate_handeye -v      # dry run, joint method (default)
python3 -m src.calibration.calibrate_handeye --write # looks sane? write it
```

Two methods, picked with `--method`:

| `--method` | What it does |
|---|---|
| `joint` (default) | Bundle-adjustment refinement, both arms jointly, CAD-nominal + cross-arm priors — see [joint_handeye_calibration.md](joint_handeye_calibration.md) |
| `direct` | Original per-arm closed-form `cv2.calibrateHandEye` (AX=XB), no priors |

```bash
python3 -m src.calibration.calibrate_handeye --method direct --write
```

Both read whatever Stage A already saved under
`outputs/calibration_debug/handeye/<cam_id>/` and write
`config/camera_extrinsics_realsense.yaml` (backing up the previous file to
`.yaml.bak` first) when `--write` is passed.

✅ Checkpoint: printed reprojection-error / AX=XB-residual lines are small
(well under 1px reprojection, sub-degree/sub-mm pose residuals) for every
arm — if not, don't `--write`; re-run Stage A with more rotationally varied
samples instead.

---

## 3. Board pose + ZED calibration (hands-off, one command)

**Just want to redo the 5-sample board-pose average by hand — checkerboard
moved, hand-eye is still fine, don't want to touch the ZED stage?** Use
[`capture_board_pose_data.py`](#capture_board_pose_datapy-board-pose-only-manual)
instead of the auto command below — same interactive jog-and-press-Enter
loop as Stage A, scoped to only `config/base_board_pose.yaml`:

```bash
python3 -m src.calibration.capture_board_pose_data
```

**Checkerboard hasn't moved and you already ran Stage A?** Skip the robot
entirely with `--source handeye` — recomputes `config/base_board_pose.yaml`
straight from Stage A's already-saved images/detections
(`outputs/calibration_debug/handeye/<cam_id>/`), no camera stack or
bring-up needed, pools every saved sample for both arms:

```bash
python3 -m src.calibration.capture_board_pose_data --source handeye
```

Otherwise, the fully automatic hands-off path (drives to saved flange poses,
augments, and runs ZED too):

```bash
python3 -m src.calibration.autocalibrate_dual_realsense
```

Requires Stage B already done (both `realsense_1`/`realsense_2` have a real
`T_flange_cam` in `config/camera_extrinsics_realsense.yaml` — it checks this
at startup and errors out with a pointer back to Stages A/B if not). It then
runs:

| Stage | What happens | Writes |
|---|---|---|
| Board pose | Each arm moves through its last 2 saved poses (`--num-board-poses`), then standard augmentation (see below) fills the gap up to `--target-board-samples` (default 5) | `config/base_board_pose.yaml` |
| ZED | Calibrates the static ZED (`zed2i_1`) against that board pose | `config/camera_extrinsics_base.yaml` |

Standard augmentation (always on, same procedure as `capture_handeye_data.py
--mode augment`): if the known poses above yield fewer than
`--target-board-samples` total accepted samples, perturbs poses drawn from
whichever arm(s) actually produced a successful sample (±3cm/±7° — see
`src/calibration/pose_augmentation.py`), drives there via Cartesian IK, and
retries on any failure until the target is hit. Works with however many arms
are connected — an arm with no camera or no successful sample just never
gets drawn from.

Useful flags:

```bash
--skip-zed                    # stop after the board-pose stage; run ZED manually later
--num-board-poses 3            # use the last 3 saved poses per arm instead of 2
--target-board-samples 6       # augment to 6 total instead of 5
```

If you stopped before the ZED stage, run it separately whenever ready:

```bash
scripts/calibrate_zed_from_board_pose.sh
```

✅ Checkpoint: script exits with `=== All stages complete ===` and no
`RuntimeError`. Each stage refuses to write bad results (translation/rotation
spread or reprojection error too high) rather than silently writing a wrong
calibration — if it raises, re-run Stage A for the failing arm, don't just
retry the same poses.

---

## 4. Sanity-check the result

```bash
cat outputs/calibration_logs/camera_transforms.json        # per-camera QA metrics, latest run last
cat outputs/calibration_logs/checkerboard_transforms.json  # board-pose QA metrics
cat outputs/calibration_logs/flange_transforms.json        # which saved poses were used, per arm
```

Look for:
- **Hand-eye (`camera_transforms.json`, `stage=handeye_flange_cam`)**: `ax_xb_residual_rot_deg_mean` should be small (large → redo Stage A with more rotational spread); `T_flange_cam.t` magnitude should be a few cm, not tens of cm.
- **Board pose (`checkerboard_transforms.json`)**: `translation_std_m` / `rotation_std_deg` should be tight (the script already rejects and raises if not — this is just for your own review).
- **ZED (`camera_transforms.json`, `stage=base_to_cams_static`)**: `reproj_err_px_mean` low, `translation_std_m`/`rotation_std_deg` tight.

Then bring up the pipeline normally and confirm fused poses from the
RealSense cameras look geometrically sane (no more `bad_distance` rejects):

```bash
scripts/launch_pipeline_realsense.sh init-only
```

---

## Recalibrating later

**Checkerboard moved slightly, robot poses should still roughly work:**
re-drive to the saved poses and recapture fresh checkerboard detections,
then re-solve:

```bash
python3 -m src.calibration.capture_handeye_data --mode replay   # needs move_group.launch.py up
python3 -m src.calibration.calibrate_handeye --write
```

This archives the previous `outputs/calibration_debug/handeye/<cam_id>/`
samples to a timestamped backup first (never silently mixes samples from
two different checkerboard placements into one solve) and doesn't touch
`config/flange_poses/*.json` (the poses themselves didn't change).

**One camera physically moved (new mount), or you want a manual
single-camera-at-a-time capture:** Stage A/B still work scoped to one arm/camera,
and the original manual `board_pose_from_flange_realsense.py` still works standalone:

```bash
python3 -m src.calibration.capture_handeye_data --arm left    # or --arm right
python3 -m src.calibration.calibrate_handeye --cam-ids realsense_1 --write
python3 -m src.calibration.board_pose_from_flange_realsense --cam-id realsense_1
```

**Starting completely fresh (new checkerboard position, full recapture):**
just re-run Stage A without `--append` — it archives everything old to a
timestamped backup automatically before capturing:

```bash
python3 -m src.calibration.capture_handeye_data
```

**Checkerboard moved and hand-eye is still fine — just want to replace the
5 pooled board-pose samples that feed `config/base_board_pose.yaml`:**
`capture_board_pose_data.py` is the manual counterpart to
`autocalibrate_dual_realsense.py`'s board-pose stage — same interactive
jog-and-press-Enter capture loop as `capture_handeye_data.py`'s
`--mode interactive`, but it captures directly against the live board pose
(no saved-flange-pose replay, no augmentation/perturbation) and writes only
`config/base_board_pose.yaml`. It archives the previous
`outputs/calibration_debug/board_pose/dual/` samples and
`config/base_board_pose.yaml` to a timestamped backup first (same
backup/auto-restore policy as Stage A) and never touches
`config/flange_poses/*.json` or runs ZED calibration:

```bash
python3 -m src.calibration.capture_board_pose_data                       # pool 5 samples, left then right
python3 -m src.calibration.capture_board_pose_data --arm left            # left arm's camera only
python3 -m src.calibration.capture_board_pose_data --append --num-samples 8  # grow instead of replace
```

---

## Flags reference

Defaults below are what you get by passing no flag at all — see
[Quickstart](#quickstart) for the happy path using just those defaults.

### `capture_handeye_data.py` (Stage A)

| Flag | Default | Meaning |
|---|---|---|
| `--arm {left,right,both}` | `both` | Which arm(s) to run for. |
| `--controller {moveit,admittance,handguided}` | `moveit` | How the arm gets positioned in `--mode interactive` — see [calibration_control_modes.md](calibration_control_modes.md). |
| `--mode {interactive,replay,augment}` | `interactive` | `interactive`: manual jog+capture, builds the initial set. `replay`: re-detect at already-saved poses (checkerboard moved slightly, poses unchanged). `augment`: standard way to grow to `--target-samples`, no manual positioning — replays known poses, then perturbs successful ones to fill the gap (or randomly trims if there are already more than the target). |
| `--num-samples N` | `7` | `--mode interactive` only — new samples to prompt for, per arm. |
| `--target-samples N` | `10` | `--mode augment` only — total samples to reach per arm (known + perturbed). |
| `--append` | off | `--mode interactive` only — extend existing data instead of archiving it first (only valid if the checkerboard hasn't moved). No effect in `replay`/`augment`. |
| `--gain-profile {holding,insertion}` | none | `--controller admittance` only — see `config/admittance_gain_profiles.yaml`. |
| `--robot-namespace NAME` | `lbr_dual_arm` | `--mode replay`/`augment` only — must match `move_group.launch.py`'s `robot_name`. |
| `--no-debug-topic` | off | Skip publishing the `/calibration/capture_handeye_data/<arm>/debug_image` topic. |

### `calibrate_handeye.py` (Stage B)

| Flag | Default | Meaning |
|---|---|---|
| `--method {direct,joint}` | `joint` | `joint`: bundle-adjustment, both arms jointly, CAD priors — see [joint_handeye_calibration.md](joint_handeye_calibration.md). `direct`: original per-arm closed-form `cv2.calibrateHandEye` (AX=XB), no priors. |
| `--cam-ids ID [ID ...]` | `realsense_1 realsense_2` | Which cameras to solve. |
| `--write` | off | Write the solved `T_flange_cam` into `config/camera_extrinsics_realsense.yaml` (backs up the old file first). Without it, results only print. |
| `-v` / `-vv` | off | Verbose output, repeatable. |
| `--min-samples N` | `5` | `--method direct` only — minimum samples required to solve. |
| `--no-nominal` | off | `--method joint` only — disable the CAD nominal init/prior entirely. |
| `--loss`, `--f-scale`, `--shared-mount-sigma-*`, `--nominal-sigma-*` | script defaults | `--method joint` only — solver-tuning knobs; see [joint_handeye_calibration.md](joint_handeye_calibration.md)'s Troubleshooting section before changing any of these. |

### `autocalibrate_dual_realsense.py` (board pose + ZED)

| Flag | Default | Meaning |
|---|---|---|
| `--num-board-poses N` | `2` | Use the last N saved poses per arm as the "known" board-pose set (before augmentation). |
| `--target-board-samples N` | `5` | Total board-pose samples to reach (known + perturbed), pooled across both arms. |
| `--skip-zed` | off | Stop after the board-pose stage; run `scripts/calibrate_zed_from_board_pose.sh` manually later. |
| `--zed-only` | off | Skip the board-pose stage entirely and just re-run ZED calibration against the existing `config/base_board_pose.yaml` — for when the checkerboard hasn't moved and only the ZED needs recalibrating. No arm motion. Mutually exclusive with `--skip-zed`. |
| `--robot-namespace NAME` | `lbr_dual_arm` | Must match `move_group.launch.py`'s `robot_name`. |
| `--no-debug-topic` | off | Skip publishing the `/calibration/autocalibrate/<arm>/debug_image` topic. |

### `capture_board_pose_data.py` (board pose only, manual)

| Flag | Default | Meaning |
|---|---|---|
| `--source {live,handeye}` | `live` | `live`: jog the real arm(s), capture fresh images (needs bring-up + camera stack). `handeye`: no robot/camera at all — recompute straight from Stage A's already-saved `outputs/calibration_debug/handeye/<cam_id>/` detections. `--controller`/`--gain-profile`/`--no-debug-topic` are ignored under `handeye`, and `--num-samples` is ignored too (pools *every* saved sample per requested arm instead). |
| `--arm {left,right,both}` | `both` | Which arm(s)' wrist camera may contribute a sample. `--source live` + `both`: pooled, left captures first, right only tops up if left didn't reach `--num-samples`. `--source handeye`: restricts which arm's saved sample directory is read. |
| `--controller {moveit,admittance,handguided}` | `moveit` | `--source live` only. How the arm gets positioned — same semantics as `capture_handeye_data.py --mode interactive`. |
| `--num-samples N` | `5` | `--source live` only. Pooled total across whichever arm(s) are used (matches `autocalibrate_dual_realsense.py`'s `--target-board-samples`). |
| `--append` | off | Extend the existing pooled sample set under `outputs/calibration_debug/board_pose/dual/` instead of archiving it first (only valid if the checkerboard hasn't moved). Works with either `--source`. |
| `--gain-profile {holding,insertion}` | none | `--source live --controller admittance` only. |
| `--no-debug-topic` | off | `--source live` only. Skip publishing the `/calibration/capture_board_pose_data/<arm>/debug_image` topic. |
