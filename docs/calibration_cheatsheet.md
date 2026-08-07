# Dual-Arm RealSense Calibration — Execution Cheatsheet

Commands only. For controller/mode background see
[calibration_control_modes.md](calibration_control_modes.md); for the joint
bundle-adjustment solver's math/troubleshooting see
[joint_handeye_calibration.md](joint_handeye_calibration.md).

Assumes: dual-arm hardware is racked, checkerboard (8×11 inner corners, 30mm
squares) is on hand, and you're starting from nothing running.

---

## 0. Bring everything up

```bash
# Terminal 1 — hardware interface (both arms)
# use_gripper defaults to true (Y-gripper attached, arm_one/arm_two tipped at
# the gripper TCP); pass use_gripper:=false for the bare flange instead.
#
# Default Stage A controller (moveit/RViz-jogged) needs hardware.launch.py.
# Swap for `admittance.launch.py` or `calibration.launch.py` if you're using
# --controller admittance / handguided instead — see step 1 below.
ros2 launch lbr_dual_arm_bringup hardware.launch.py use_gripper:=true
```

```bash
# Terminal 2 — MoveIt + RViz (needed for --controller moveit AND for
# --mode replay; NOT needed for --controller admittance/handguided).
# use_gripper must match Terminal 1's value.
ros2 launch lbr_dual_arm_bringup move_group.launch.py mode:=hardware rviz:=true use_gripper:=true
```

Then on **both pendants** (left first, then right): start the `LBRServer`
app to open the FRI connections Terminal 1 is waiting on.

```bash
# Terminal 3 — host camera stack (ZED + both RealSense + flange_pose_publisher x2)
scripts/launch_host_realsense.sh
```

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

```bash
python3 -m src.calibration.autocalibrate_dual_realsense
```

Requires Stage B already done (both `realsense_1`/`realsense_2` have a real
`T_flange_cam` in `config/camera_extrinsics_realsense.yaml` — it checks this
at startup and errors out with a pointer back to Stages A/B if not). It then
runs:

| Stage | What happens | Writes |
|---|---|---|
| Board pose | Each arm moves through its last 2 saved poses (`--num-board-poses`); solves the checkerboard's pose in the robot base frame | `config/base_board_pose.yaml` |
| ZED | Calibrates the static ZED (`zed2i_1`) against that board pose | `config/camera_extrinsics_base.yaml` |

Useful flags:

```bash
--skip-zed                # stop after the board-pose stage; run ZED manually later
--num-board-poses 3        # use the last 3 saved poses per arm instead of 2
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
