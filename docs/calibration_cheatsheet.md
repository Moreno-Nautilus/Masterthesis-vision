# Dual-Arm RealSense Calibration — Execution Cheatsheet

Commands only. For the full explanation of what each stage does and why, see
[getting_started_realsense.md §4](getting_started_realsense.md#4-hand-eye-calibration-camera-to-flange-offset).

Assumes: dual-arm hardware is racked, checkerboard (8×11 inner corners, 30mm
squares) is on hand, and you're starting from nothing running.

---

## 0. Bring everything up

```bash
# Terminal 1 — hardware interface (both arms)
# use_gripper defaults to true (Y-gripper attached, arm_one/arm_two tipped at
# the gripper TCP); pass use_gripper:=false for the bare flange instead.
#
# Default Step 1 (admittance-guided capture): swap this for
#   ros2 launch lbr_dual_arm_bringup admittance.launch.py use_gripper:=true
# instead — see Step 1 below. hardware.launch.py is only needed here for the
# RViz-jogging alternative (and always for Step 2's replay).
ros2 launch lbr_dual_arm_bringup hardware.launch.py use_gripper:=true
```

```bash
# Terminal 2 — MoveIt + RViz (needed for Step 1 RViz-jogging AND Step 2's
# automatic moves; NOT needed for the default admittance-guided or
# gravity-compensation hand-guided Step 1 alternatives).
# use_gripper must match Terminal 1's value.
ros2 launch lbr_dual_arm_bringup move_group.launch.py mode:=hardware rviz:=true use_gripper:=true
```

Then on **both pendants** (left first, then right): start the `LBRServer` app
to open the FRI connections Terminal 1 is waiting on.

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
can see it and **do not move it again** until Step 2 finishes.

---

## 1. Capture flange poses (manual, ~5 min per arm)

**Default: admittance-guided capture.** Bring the rig up in software-
admittance mode (position interface, no torque mode needed) instead of
`hardware.launch.py`:

```bash
ros2 launch lbr_dual_arm_bringup admittance.launch.py use_gripper:=true
```

Inside the `vision` container:

```bash
python3 -m src.calibration.capture_flange_poses_dual_admittance --arm left
```

This script runs the admittance control loop itself for the whole session —
**only the arm you passed `--arm` for is compliant** (the other arm's
position controller just holds its last commanded pose; running both arms'
admittance loops concurrently was found to roughly halve the achievable
control-loop rate for the arm you're actually guiding, which is what made
it feel stuck/rigid before this fix). For each of 7 prompts: physically
push the **left** arm to a pose where the checkerboard is fully visible to
`realsense_1`, vary orientation on at least the first 5, let it settle,
press **Enter**.

Ctrl-C when done with the left arm, then (either in the same terminal or a
fresh launch of `admittance.launch.py` — a completely separate session is
fine, nothing carries over between arms):

```bash
python3 -m src.calibration.capture_flange_poses_dual_admittance --arm right
```

Same again for the **right** arm / `realsense_2`. This is the standard
routine: **one arm at a time, left then right**, and it's fine to split
across two entirely separate launch sessions rather than one continuous
one. Add `--gain-profile insertion` to either command if the default
("holding") still feels too stiff. Also saves the joint configuration
alongside the Cartesian pose, so
Step 2's replay reproduces the exact captured posture (see
[calibration_control_modes.md §3](calibration_control_modes.md#3-admittance--force-driven-compliance-on-the-position-interface)).

✅ Checkpoint: `config/flange_poses/left.json` and `right.json` each have 7
entries. (`--append` if you need to add more later without starting over.)

**Alternatives**, both still supported — pick one and use it for both arms
(captures from different capture scripts are not interchangeable, see
below):

- **RViz jogging** (the original flow): bring up `hardware.launch.py` +
  `move_group.launch.py` (Step 0 above) and run
  `python3 -m src.calibration.capture_flange_poses_dual --arm left` — for
  each prompt, drag the interactive marker in RViz's MotionPlanning panel
  and click **Plan & Execute** instead of physically pushing the arm.
- **Gravity-compensation hand-guiding**: bring the rig up in gravity-
  compensation mode (`ros2 launch lbr_dual_arm_bringup calibration.launch.py`)
  and physically push each arm into place, then run
  `python3 -m src.calibration.capture_flange_poses_dual_handguided --arm left`
  (same for `right`) — same interaction as admittance-guided capture, just
  torque-mode compliance from the hardware controller instead of the
  software admittance loop.

`capture_flange_poses_dual_admittance.py` and
`capture_flange_poses_dual_handguided.py` both save joint positions (unlike
`capture_flange_poses_dual.py`'s plain RViz-jogging captures) — required by
`autocalibrate_dual_realsense.py`'s joint-space replay (see
[hand_guided_calibration.md](hand_guided_calibration.md)).

---

## 2. Automatic calibration (hands-off, one command)

```bash
python3 -m src.calibration.autocalibrate_dual_realsense
```

This alone runs all three stages in order:

| Stage | What happens | Writes |
|---|---|---|
| A | Both arms move **simultaneously** through 5 pose-pairs; solves `T_flange_cam` for both RealSense cameras | `config/camera_extrinsics_realsense.yaml` |
| B | Each arm moves through its remaining 2 poses; solves the checkerboard's pose in the robot base frame | `config/base_board_pose.yaml` |
| C | Calibrates the static ZED (`zed2i_1`) against that board pose | `config/camera_extrinsics_base.yaml` |

Useful flags:

```bash
--skip-zed                  # stop after Stage B; run Stage C manually later
--min-handeye-samples 4     # loosen Stage A's minimum (default 5)
```

If you stopped Stage C, run it separately whenever ready:

```bash
scripts/calibrate_zed_from_board_pose.sh
```

✅ Checkpoint: script exits with `=== All stages complete ===` and no
`RuntimeError`. Each stage refuses to write bad results (translation/rotation
spread or reprojection error too high) rather than silently writing a wrong
calibration — if it raises, re-run captures for the failing arm/stage, don't
just retry the same poses.

---

## 3. Sanity-check the result

```bash
cat outputs/calibration_logs/camera_transforms.json        # per-camera QA metrics, latest run last
cat outputs/calibration_logs/checkerboard_transforms.json  # board-pose QA metrics
cat outputs/calibration_logs/flange_transforms.json        # which saved poses were used, per arm
```

Look for:
- **Hand-eye (`camera_transforms.json`, `stage=handeye_flange_cam`)**: `ax_xb_residual_rot_deg_mean` should be small (large → redo with more rotational spread); `T_flange_cam.t` magnitude should be a few cm, not tens of cm.
- **Board pose (`checkerboard_transforms.json`)**: `translation_std_m` / `rotation_std_deg` should be tight (the script already rejects and raises if not — this is just for your own review).
- **ZED (`camera_transforms.json`, `stage=base_to_cams_static`)**: `reproj_err_px_mean` low, `translation_std_m`/`rotation_std_deg` tight.

Then bring up the pipeline normally and confirm fused poses from the
RealSense cameras look geometrically sane (no more `bad_distance` rejects):

```bash
scripts/launch_pipeline_realsense.sh init-only
```

---

## 4. Optional: joint bundle-adjustment refinement (better than Stage A alone)

**Not a capture step — offline only.** Reuses whatever samples Step 2 above
already captured; this command never moves the robot or grabs an image, it just
re-optimizes existing `sample_*.json` files. No new captures needed.
Jointly refines both arms' `T_flange_cam` against per-corner reprojection error
instead of Stage A's closed-form per-arm solve, with two extra priors: the two
arms' offsets are pulled toward each other (same physical mount) and toward the
CAD-derived `realsense_nominal` value. Full explanation:
[joint_handeye_calibration.md](joint_handeye_calibration.md).

```bash
python3 -m src.calibration.joint_calibrate_dual_realsense -v      # dry run, prints diagnostics
python3 -m src.calibration.joint_calibrate_dual_realsense --write # looks sane? write it
```

✅ Checkpoint: printed `pose residual` / `reprojection error (px)` lines are small
(sub-degree / a few mm, or well under 1px) for every arm — if not, don't `--write`;
see the linked doc's Troubleshooting section (usually a `--loss` choice issue, not
bad data).

---

## Recalibrating later (one camera only)

If just one wrist camera physically moved and you don't want to redo the
whole dual-arm routine, the original manual single-camera scripts still work
— see [getting_started_realsense.md §4.7](getting_started_realsense.md#47-manual-single-camera-fallback-original-scripts-still-available):

```bash
python3 -m src.calibration.handeye_flange_cam_realsense --cam-id realsense_1
python3 -m src.calibration.board_pose_from_flange_realsense --cam-id realsense_1
```
