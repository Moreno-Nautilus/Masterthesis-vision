# Getting Started — RealSense Trio Variant (1 ZED + 2 end-effector RealSense)

A step-by-step guide to bringing up the **experimental RealSense pipeline**:
`zed2i_1` (static, tripod-mounted) + two Intel RealSense D405 cameras mounted on
the robot end-effector.

This is a **separate, parallel pipeline** from the standard 3-ZED setup described in
[getting_started.md](getting_started.md). Nothing in the original scripts, launch
files, or pipeline code was changed — every file used here is new:

| Purpose | Original (3-ZED) | RealSense variant |
|---|---|---|
| Host launch script | `scripts/launch_host.sh` | `scripts/launch_host_realsense.sh` |
| Pipeline launch script | `scripts/launch_pipeline.sh` | `scripts/launch_pipeline_realsense.sh` |
| Camera launch file | `mv_launch` → `zed2i_pair.launch.py` | `mv_launch` → `zed_realsense_trio.launch.py` |
| Extrinsics config | `config/camera_extrinsics_base.yaml` | `config/camera_extrinsics_realsense.yaml` |
| Grabber module | `src/perception/ros/multicam_grabber.py` | `src/perception/ros/multicam_grabber_realsense.py` |
| Pipeline runner | `run_pipeline_track_multicam.py` | `run_pipeline_track_multicam_realsense.py` |
| Flange pose (new, no 3-ZED equivalent) | — | `mv_launch` → `flange_pose_publisher.py` (§5) |
| Hand-eye calibration (new, no 3-ZED equivalent) | — | `src/calibration/handeye_flange_cam_realsense.py` + `board_pose_from_flange_realsense.py` (§4) |
| RGB/depth rectification (new, no 3-ZED equivalent — ZED rectifies internally) | — | `image_proc` rectify containers in `zed_realsense_trio.launch.py` (§8) |

> **Status: runs end-to-end, verified working** (see §1 below for the exact
> tested sequence). `config/camera_extrinsics_realsense.yaml` ships **identity**
> for the RealSense cameras until you run the §4 calibration scripts — until
> then, fused poses from those cameras are geometrically wrong. Everything else
> — camera drivers, serials, the flange-pose topic, the pipeline itself — is
> real and working.
>
> **Note on the robot**: despite the `franka_ros2_ws` directory name (just where
> the workspace happens to live on disk), the arm actually running on this rig is
> a **KUKA LBR iiwa** (`lbr_fri_ros2_stack`, `iiwa7` by default), not a Franka.
> Everything below uses iiwa-flavored naming (`lbr_link_0`, `lbr_link_ee`,
> `/lbr/state`, `/iiwa/ee_pose`) accordingly.

---

## 1. Run it start to finish (the tested sequence)

Follow this top to bottom, in order. Each step tells you what to check before
moving to the next — don't skip the checks, since a later step failing silently
almost always traces back to one of these.

You'll want **3 terminals**: one for the robot/tf step, one for the host camera
stack, one for the pipeline.

### Step 0 — before you start

```bash
cd ~/Masterthesis-vision
docker ps -a --filter name=vision      # 'vision' container should be listed
nvidia-smi                             # GPU should be mostly free
rs-enumerate-devices -s                # both RealSense D405s should be listed
```

If the GPU is busy with another job, wait — a saturated GPU makes SAM/DINO
produce garbage (NaNs), same as in the 3-ZED pipeline. If a RealSense doesn't
show up, check its USB3 connection (not a hub, not a USB2 port).

### Step 1 — get a base→flange transform into tf2 (terminal 1)

The pipeline needs `lbr_link_0 → lbr_link_ee` resolvable via tf2 before the two
RealSense cameras can ever be marked "ready" — this is how their live extrinsic
gets computed every frame (see §5 for why). Pick **one** of these:

```bash
# Option A — real robot. Needs the KUKA FRI connection already live on the
# controller/pendant side; if this hangs on "Awaiting robot heartbeat" the FRI
# application on the cabinet hasn't started streaming yet — that's on the
# KUKA side, not something to fix here.
ros2 launch lbr_bringup hardware.launch.py model:=iiwa7
```

```bash
# Option B — no robot connected right now. Publishes a fixed identity
# transform so the rest of the pipeline can be exercised. Fused poses from
# the RealSense cameras will be wrong with this (expected — see §4), but
# camera sync, detection, and the fusion code paths all still run for real.
ros2 run tf2_ros static_transform_publisher \
    --frame-id lbr_link_0 --child-frame-id lbr_link_ee \
    --x 0 --y 0 --z 0 --qx 0 --qy 0 --qz 0 --qw 1
```

Leave this running. **Verify it before moving on** (separate shell):

```bash
source /opt/ros/humble/setup.bash
ros2 run tf2_ros tf2_echo lbr_link_0 lbr_link_ee
```

You should see a transform streaming repeatedly. If you instead see `"lbr_link_0"
... does not exist`, neither option above is actually running yet — go back and
check.

### Step 2 — start the host camera stack (terminal 2)

```bash
cd ~/Masterthesis-vision
scripts/launch_host_realsense.sh          # start (and attach) the tmux session
# scripts/launch_host_realsense.sh attach # re-attach later if already running
# scripts/launch_host_realsense.sh stop   # kill it when done
```

This brings up `zed2i_1`, both RealSense D405s (serials already pinned, see §3),
the Foxglove bridge, a `visualize_pipeline` per camera, and `flange_pose_publisher`
— all as tmux windows in one session (`Ctrl+b` `0`..`5` to switch, `Ctrl+b d` to
detach without killing anything).

**Verify every camera and the flange pose before moving on** (separate shell):

```bash
source /opt/ros/humble/setup.bash
ros2 topic hz /zed2i_1/zed_node/rgb/color/rect/image
ros2 topic hz /realsense_1/camera/color/image_raw     # raw driver output (still published, unused by the pipeline)
ros2 topic hz /realsense_2/camera/color/image_raw
ros2 topic hz /realsense_1/camera/color/image_rect    # rectified — this is what the pipeline actually consumes, see §8
ros2 topic hz /realsense_2/camera/color/image_rect
ros2 topic echo /iiwa/ee_pose --once
```

All must return real data before you continue. If a RealSense `image_raw` topic is
silent, check the `cams` tmux window (`Ctrl+b` `0`) for the actual error — see
§7 Troubleshooting for the specific failure modes already hit and fixed once. If
`image_raw` is fine but `image_rect` is silent, the rectify container failed to
start — check the same tmux window; see §8.

### Step 3 — start the pipeline (terminal 3)

**Must be a real interactive terminal** — `docker exec -it` fails silently (`the
input device is not a TTY`) if this is run through a non-interactive
shell/script/tool. Run it directly at a terminal prompt, or inside its own tmux
window.

```bash
cd ~/Masterthesis-vision
scripts/launch_pipeline_realsense.sh init-only
```

`init-only` is the right first choice — it detects and poses every frame with no
tracking complexity, easiest to sanity-check. (`fast-track` / `accurate-track`
are available the same as the 3-ZED pipeline once this works — see §6.)

This opens a tmux session (`mv_pipeline_realsense` by default) with two windows:
`run` (the pipeline, attached by default) and `keys` (a keyboard helper for
pausing/resuming tracking without reloading models — see §6). Switch windows
with `Ctrl+b` then `0`/`1`; detach without killing anything with `Ctrl+b d`.

### Step 4 — confirm it's actually working

A healthy startup log looks like this, in order:

```
Subscribing flange pose topic=/iiwa/ee_pose for dynamic cameras=['realsense_1', 'realsense_2']
Subscribing cam_id=zed2i_1 ... dynamic=False
Subscribing cam_id=realsense_1 ... dynamic=True
Subscribing cam_id=realsense_2 ... dynamic=True
...
DINO ready | objects=[...]
GDINO proposer ready | ...
FoundationPoseTrackerNode started | run_mode=init_only
```

then it cycles repeatedly through, for **all three cameras** every cycle:

```
========== MULTICAM INIT START ==========
[VRAM] zed2i_1: ...
[TIMING] SAM zed2i_1: ... masks
DINO cand ... 
FP+ICP [zed2i_1] <object>: fitness=... chamfer=...
...
[VRAM] realsense_1: ...
[VRAM] realsense_2: ...
========== MULTICAM INIT TOTAL: ...ms | N objects ==========
```

No `Traceback` or `Exception` anywhere. Seeing `FP pose reject <cam> <object>:
bad_distance` for a RealSense camera is **expected right now**, not a bug — it
means a geometrically implausible pose was correctly rejected, which is exactly
what should happen while the hand-eye calibration is still the identity
placeholder (§4).

Watch it in Foxglove the same way as the 3-ZED pipeline
([getting_started.md §1](getting_started.md#watch-the-result-in-foxglove)) — the
per-camera overlay topics exist for `zed2i_1`, `realsense_1`, `realsense_2` under
the same `/perception/fp/...` naming pattern.

**To stop for good**: `scripts/launch_pipeline_realsense.sh stop` (kills the
whole tmux session, both windows), then `scripts/launch_host_realsense.sh stop`
for the host stack. If you started a placeholder `static_transform_publisher`
in Step 1, `Ctrl+C` that too.

**To pause/resume without reloading models** (SAM/DINO/GDINO/FoundationPose/Cutie
all take a while to load and warm up): switch to the `keys` tmux window (`Ctrl+b
1`) and press `x`/`s` instead of stopping the session — see §6 below.

---

## 2. Why this pipeline is different: dynamic extrinsics

`zed2i_1` is tripod-mounted, so its camera-to-base transform is fixed and lives in
the YAML exactly like the 3-ZED setup.

The two RealSense cameras are mounted on the **end-effector**, so they move with
the arm. Their pose in the base frame changes every frame. The pipeline handles
this by splitting the transform in two:

```
T_base_cam(t) = T_base_flange(t)   @   T_flange_cam
                 ^ live, per-frame     ^ static, mechanical mount offset
                 (subscribed pose)     (from config/camera_extrinsics_realsense.yaml)
```

- `T_flange_cam` — how the camera sits relative to the robot flange. Fixed once
  the camera is bolted on; this is the **hand-eye calibration** (§4 below).
- `T_base_flange(t)` — the robot's current flange pose in the base frame, published
  as a `geometry_msgs/PoseStamped`. The grabber node
  (`multicam_grabber_realsense.py`) subscribes to this every frame and composes it
  with `T_flange_cam` to get the camera's live base-frame extrinsic.

`config/camera_extrinsics_realsense.yaml` reuses the same `R`/`t` YAML shape as the
3-ZED file, but the meaning of the `realsense_1`/`realsense_2` entries is
**camera-to-flange**, not camera-to-base — read the comments at the top of that
file before editing it.

---

## 3. Packages and camera serials — already set up, reference only

`realsense2_camera` (the ROS2 wrapper) is already built from source inside
`franka_ros2_ws` (`src/realsense-ros/realsense2_camera`, installed at
`franka_ros2_ws/install/realsense2_camera`), the same overlay workspace `mv_launch`
and the ZED wrapper live in. It comes online automatically the moment
`franka_ros2_ws/install/setup.bash` is sourced — which `launch_host_realsense.sh`
already does — so **no `apt install` step is needed**. The `librealsense2-utils`
CLI tools (`rs-enumerate-devices`, etc.) are also already installed on the host.

**If you ever need to rebuild `mv_launch`** (e.g. after editing
`zed_realsense_trio.launch.py` or `flange_pose_publisher.py`):

```bash
cd ~/franka_ros2_ws
colcon build --packages-select mv_launch
source install/setup.bash
```

**Camera serials** are pinned so a given physical camera always comes up as the
same `cam_id`, regardless of USB enumeration order:

| cam_id | Serial | Where it's set |
|---|---|---|
| `zed2i_1` | `38580376` | `cam1_serial` launch arg in `zed_realsense_trio.launch.py`, default |
| `realsense_1` | `260322275185` | `RS1_SERIAL` env var / `rs1_serial` launch arg, default |
| `realsense_2` | `260522275434` | `RS2_SERIAL` env var / `rs2_serial` launch arg, default |

Just running `scripts/launch_host_realsense.sh` with no overrides uses all three
— nothing to do here unless the hardware changes. **If you swap a camera**,
re-enumerate and override:

```bash
rs-enumerate-devices -s      # short summary: one row per connected device
```

USB port order is **not** guaranteed stable across reboots/replugs, so to map
serial → physical camera, plug in **only one** RealSense at a time, note its
serial, and label the camera. Then override:

```bash
RS1_SERIAL=<new_serial> RS2_SERIAL=<existing_serial> scripts/launch_host_realsense.sh
```

or for the ZED:

```bash
ros2 launch mv_launch zed_realsense_trio.launch.py cam1_serial:=<serial> ...
```

---

## 4. Hand-eye calibration (camera-to-flange offset)

`config/camera_extrinsics_realsense.yaml` ships **identity** for `realsense_1` /
`realsense_2` until you run the calibration below: zero translation, no rotation
— i.e. "camera optical frame == flange frame exactly." This is a deliberate
placeholder so the pipeline can be brought up and exercised end-to-end (grabber
sync, tf2 lookup, fusion code paths) before real calibration exists — **it is
not a real measurement and fused poses from these two cameras will be
geometrically wrong** until replaced.

What you need per RealSense camera: the rigid transform from the **flange frame**
(`lbr_link_ee`) to the **camera's optical frame** (`T_flange_cam`), i.e. where
the camera sits and points relative to the robot's flange. This repo has two
scripts under `src/calibration/` for a two-stage routine that gets you this
*and* a computed (not hand-measured) checkerboard pose in the robot base frame,
using the same checkerboard as the ZED calibration
([getting_started.md §2](getting_started.md#2-calibrate-the-cameras): 8×11 inner
corners, 30 mm squares).

If instead the camera mount is a known, rigid, machined part, you can skip both
scripts and pull `T_flange_cam` directly from the mount's CAD model — edit the
`realsense_1`/`realsense_2` blocks in
[config/camera_extrinsics_realsense.yaml](../config/camera_extrinsics_realsense.yaml)
directly (same row-major 3×3 `R` + 3-vector `t` convention as the rest of the
repo). The rest of this section covers the checkerboard-based routine.

### 4.0 Two-script dual-arm routine (current workflow)

The routine below drives **both** LBR arms of the dual-arm rig
(`lbr_dual_arm_bringup`, arms `lbr_one`/left and `lbr_two`/right — see
`~/Desktop/KUKA_dual_arm_bringup_README.md`), each carrying its own wrist
RealSense (`realsense_1` on `lbr_one`, `realsense_2` on `lbr_two`), plus the
one static ZED (`zed2i_1`). It's split into two scripts so the
slow/manual/human-judgment part (deciding where to jog the arms) is done
once and its raw result kept forever, while the actual calibration solve is
scripted, repeatable, and doesn't need an operator standing at the robot:

1. **`capture_flange_poses_dual.py`** (manual, once per arm) — jog each arm
   with MoveIt RViz to 7 poses where its own wrist camera sees the fixed
   checkerboard, press Enter to save that arm's flange pose. Nothing is
   calibrated yet; only the poses themselves are recorded, permanently, to
   `config/flange_poses/left.json` / `right.json` (saved after **every**
   single capture, not just at the end — a crash mid-session loses nothing
   already accepted).
2. **`autocalibrate_dual_realsense.py`** (automatic, replay) — reads those
   saved poses back and, assuming the checkerboard hasn't moved since step 1,
   drives both arms there via MoveIt automatically (no jogging), solving:
   - **Stage A**: both arms' `T_flange_cam` (hand-eye), moving both arms
     *simultaneously* per pose-pair using the first 5 poses/arm.
   - **Stage B**: the checkerboard's pose in the robot base frame, using the
     remaining 2 poses/arm (4 samples total), now that both hand-eye
     transforms are known.
   - **Stage C**: the static ZED's extrinsic, from that now-known board pose
     (`scripts/calibrate_zed_from_board_pose.sh`).

   Every stage's result + quality metrics is appended (never overwritten) to
   `outputs/calibration_logs/{camera,checkerboard,flange}_transforms.json`
   (see §4.6) in addition to updating the live YAML configs.

The original single-arm, single-camera manual scripts
(`handeye_flange_cam_realsense.py`, `board_pose_from_flange_realsense.py`,
§4.7 below) still work standalone if you ever need to recalibrate just one
camera by hand — the dual-arm routine reuses their PnP/AX=XB solving code
directly, it doesn't replace or remove them.

**Hand-guided alternative to step 1**: `capture_flange_poses_dual_handguided.py`
does the same job as `capture_flange_poses_dual.py` but expects the arm to
be physically hand-guided (gravity-compensation mode,
`ros2 launch lbr_dual_arm_bringup calibration.launch.py`) instead of jogged
in RViz, and additionally saves the joint configuration so step 2's replay
reproduces the exact captured posture (not just an equivalent Cartesian
pose) via joint-space MoveGroup goals. See
[hand_guided_calibration.md](hand_guided_calibration.md) for the full
writeup. The two capture scripts' outputs are not interchangeable for
`autocalibrate_dual_realsense.py` — pick one and use it for both arms.

### 4.1 Bring everything up

Real dual-arm hardware bringup (**not** `mock.launch.py` — Stage A/B need
real, varied flange motion) plus MoveIt, in two terminals:

```bash
# Terminal 1 — use_gripper defaults to true (Y-gripper attached); pass
# use_gripper:=false for the bare flange instead.
ros2 launch lbr_dual_arm_bringup hardware.launch.py use_gripper:=true

# Terminal 2 — use_gripper must match Terminal 1's value.
ros2 launch lbr_dual_arm_bringup move_group.launch.py mode:=hardware rviz:=true use_gripper:=true
```

Then start the `LBRServer` app on **both** pendants (left, then right) to
open the FRI UDP connections `hardware.launch.py` is waiting on — see
`~/Desktop/KUKA_dual_arm_CHEATSHEET.md` for the full pre-flight checklist
(T1 mode, port 30200/30201, etc.).

Also bring up the host camera stack and verify both RealSense RGB topics and
both flange-pose topics are live:

```bash
scripts/launch_host_realsense.sh
ros2 topic echo /left/ee_pose --once
ros2 topic echo /right/ee_pose --once
```

For **Step 1** (`capture_flange_poses_dual.py`), jog each arm in RViz's
**MotionPlanning** panel exactly as in
[moveit_robot_control.md](moveit_robot_control.md) — drag the interactive
marker on that arm's flange, **Plan & Execute**. **Step 2**
(`autocalibrate_dual_realsense.py`) needs no jogging at all; it drives the
arms itself through the same `move_group` this terminal started, via the
`moveit_msgs/action/MoveGroup` action directly (see
[src/calibration/moveit_dual_arm.py](../src/calibration/moveit_dual_arm.py) —
there is no `moveit_commander`/`moveit_py` Python package installed in this
environment, so this talks to the same action `move_group`'s own RViz plugin
uses under the hood).

### 4.2 Recommended Foxglove layout

`capture_flange_poses_dual.py` publishes
`/calibration/capture_flange_poses/<left|right>/debug_image`;
`autocalibrate_dual_realsense.py` publishes
`/calibration/autocalibrate/<left|right>/debug_image` — add an **Image**
panel on whichever is running so you can confirm the checkerboard is
visible. Also useful: a **3D** panel with the robot model, and **Raw
Messages** panels on `/left/ee_pose` and `/right/ee_pose`. Same
`foxglove_bridge` as `launch_host_realsense.sh` already starts.

### 4.3 Step 1 — capture flange poses (manual, per arm)

Inside the `vision` container:

```bash
python3 -m src.calibration.capture_flange_poses_dual --arm left
python3 -m src.calibration.capture_flange_poses_dual --arm right
```

For each of the recommended **7 samples per arm**: jog that arm with MoveIt
to a pose where the checkerboard is fully visible to its own wrist
RealSense, let it settle, press Enter. A live checkerboard detection on that
arm's camera gates the capture (reject if the board isn't actually
in view or the reprojection error is too high) — but only the **flange
pose** is saved, not the checkerboard's own pose. **Vary orientation, not
just position** across at least the first 5 of the 7 — those are the ones
`autocalibrate_dual_realsense.py` uses for the `AX=XB` hand-eye solve by
default (`--num-handeye-poses`, default 5), which is only well-conditioned
with real rotational diversity; the last 2 (`--num-board-poses`, default 2)
are used for the checkerboard-pose solve, where the pose itself matters less
since `T_flange_cam` is already known by that point.

Every capture is written immediately to
`config/flange_poses/<left|right>.json` — safe to stop with `q` early (if
you already have >= 1 sample) or Ctrl-C between captures; nothing already
accepted is lost. Pass `--append` to add to a previous session's captures
instead of starting over.

Keep the checkerboard **fixed** for the rest of this whole section — Step 2
replays these exact poses assuming the board hasn't moved.

### 4.4 Step 2 — automatic calibration (replay + solve)

```bash
python3 -m src.calibration.autocalibrate_dual_realsense
```

Runs Stage A → B → C in order (see §4.0), aborting before any later stage if
an earlier one doesn't have enough accepted samples — nothing partial is
ever written to the YAML configs. Pass `--skip-zed` to stop after Stage B
(e.g. if you want to inspect the checkerboard pose before trusting the ZED
solve to it), then run `scripts/calibrate_zed_from_board_pose.sh` separately
once satisfied. `--min-handeye-samples` (default 5) can loosen the Stage A
gate if you deliberately captured fewer poses.

Stage A drives both arms **simultaneously** per pose-pair (a single
`both_arms` MoveGroup goal covering both tip links at once, see
`moveit_dual_arm.ArmTarget`/`DualArmMoveitClient.move_to`) — MoveIt supports
this directly since `lbr_dual_arm_moveit_config`'s SRDF already defines a
`both_arms` planning group. Stage B moves one arm at a time (each board-pose
sample only needs one camera).

Stage A writes `T_flange_cam` for **both** `realsense_1` and `realsense_2`
into `config/camera_extrinsics_realsense.yaml` together (backing up to
`.yaml.bak` first). Stage B overwrites `config/base_board_pose.yaml`. Stage C
(via `scripts/calibrate_zed_from_board_pose.sh`) overwrites
`config/camera_extrinsics_base.yaml`'s `zed2i_1` entry, reusing
[src/calibration/base_to_cams_calib_3.py](../src/calibration/base_to_cams_calib_3.py)'s
existing PnP + averaging + QA-gate logic (that script now takes a
`--cam-ids` list — still defaults to the original 3-ZED trio with no args —
scoped here to the single ZED this rig has).

### 4.5 Sanity-checking the result

Compare each `T_flange_cam` translation magnitude against a rough physical
estimate (camera is bolted a few cm from the flange, not tens of
centimeters) — a wildly large translation usually means too few/too
correlated samples rather than a real mechanical offset. Also see the
`lbr_link_0 == base` assumption flagged in
[§5.2](#52-whats-implemented-flange_pose_publisher) below: Stage B's
computed board pose is a good way to *verify* that assumption, since it's
now derived from real tf data instead of a hand-typed guess.

### 4.6 Where the calibration history + quality metrics live

Every run of `autocalibrate_dual_realsense.py` (and any standalone run of
`base_to_cams_calib_3.py`) **appends** an entry — never overwrites — to:

| File | What's logged |
|---|---|
| `outputs/calibration_logs/camera_transforms.json` | One entry per camera per run: `T_flange_cam` (RealSense, Stage A) or `T_base_cam` (ZED, Stage C) plus its QA metrics (`AX=XB` rotation/translation residuals for hand-eye; reprojection error + translation/rotation std for the static ZED solve). |
| `outputs/calibration_logs/checkerboard_transforms.json` | One entry per Stage B run: the solved `T_base_board`, sample count, translation/rotation std, mean reprojection error. |
| `outputs/calibration_logs/flange_transforms.json` | One entry per arm per Stage A+B run: exactly which saved `config/flange_poses/<arm>.json` capture indices were used for hand-eye vs. board-pose, with each capture's own pose inlined (self-contained even if the flange-pose file is later overwritten by a fresh capture session). |

See [src/calibration/calibration_log.py](../src/calibration/calibration_log.py)
for the exact schema. The live YAML configs
(`config/camera_extrinsics_realsense.yaml`, `config/camera_extrinsics_base.yaml`,
`config/base_board_pose.yaml`) still only ever hold the *latest* calibration —
these JSON logs are the append-only history + QA trail alongside them.

### 4.7 Manual single-camera fallback (original scripts, still available)

If you need to recalibrate just one RealSense by hand instead of running the
full dual-arm routine (e.g. only one camera physically moved), the original
manual, single-arm/single-camera scripts are unchanged and still work:

```bash
python3 -m src.calibration.handeye_flange_cam_realsense --cam-id realsense_1
python3 -m src.calibration.board_pose_from_flange_realsense --cam-id realsense_1
```

[config/robot_bases.yaml](../config/robot_bases.yaml) records the two
robots' base-frame offsets and an `active_robot` selector these two scripts
read purely for logging/output-header purposes (they always compute poses
directly in whatever robot's `lbr_link_0`/`lbr_{one,two}_link_0` is actually
streaming tf on the machine you run them on). **Set `active_robot` to match
the arm you're actually calibrating before running either script.** See each
script's own docstring for the full per-sample walkthrough (jog with
MoveIt → press Enter → repeat >= 10-15 times for hand-eye, >= 5 for
board-pose).

### 4.8 Joint bundle-adjustment refinement (upgrade over Stage A's closed-form solve)

Reuses samples already captured by `handeye_flange_cam_realsense.py` or Stage A
above (no new captures needed) to jointly refine both arms' `T_flange_cam` against
per-corner reprojection error, with soft priors tying the two arms together (same
physical mount) and toward the CAD-derived `realsense_nominal` offset:

```bash
python3 -m src.calibration.joint_calibrate_dual_realsense -v      # dry run
python3 -m src.calibration.joint_calibrate_dual_realsense --write
```

See [joint_handeye_calibration.md](joint_handeye_calibration.md) for the full
explanation, quick start, and troubleshooting.

---

## 5. The live flange-pose topic — how it works

The grabber subscribes to a `geometry_msgs/PoseStamped` topic for the robot's
current base→flange pose (`--flange-pose-topic`, default `/iiwa/ee_pose`).

### 5.1 Why not `/lbr/state` directly

`/lbr/state` (`lbr_fri_idl/msg/LBRState`) only carries raw joint angles
(`measured_joint_position`, 7 floats) and FRI session diagnostics — no Cartesian
pose. Getting a pose requires forward kinematics.

### 5.2 What's implemented: `flange_pose_publisher`

Rather than hand-rolling FK, `flange_pose_publisher` (a new node in
[mv_launch/mv_launch/flange_pose_publisher.py](../../franka_ros2_ws/src/mv_launch/mv_launch/flange_pose_publisher.py))
reads the transform from **tf2**. The LBR's own `hardware.launch.py` already starts
`robot_state_publisher` (namespaced `lbr`), fed by `joint_state_broadcaster`, which
publishes the full kinematic chain as tf — including the flange frame
`lbr_link_ee` (fixed joint `lbr_joint_ee` off `lbr_link_7`; see
`lbr_description/urdf/iiwa7/iiwa7_description.xacro`). `flange_pose_publisher` just
looks up `lbr_link_0 → lbr_link_ee` via `tf2_ros.Buffer.lookup_transform` at 30 Hz
and republishes it as `PoseStamped` on `/iiwa/ee_pose` (configurable via launch
args: `flange_base_frame`, `flange_frame`, `flange_pose_topic`).

It's wired into `zed_realsense_trio.launch.py` and starts automatically with
`launch_host_realsense.sh` — **nothing to run separately** as long as Step 1 (§1
above) has provided a resolvable `lbr_link_0 → lbr_link_ee` transform.

> **ASSUMPTION TO VERIFY**: `flange_pose_publisher` treats `lbr_link_0` as
> coincident with the Masterthesis-vision pipeline's "base" frame. That "base"
> frame is defined operationally by the checkerboard measurement in
> [config/base_board_pose.yaml](../config/base_board_pose.yaml)
> (`translation_xyz_m` is "the checkerboard origin corner in robot base frame",
> entered by hand — not derived from tf2). If that measurement wasn't taken
> relative to `lbr_link_0` specifically, this assumption is wrong, and every
> RealSense-derived pose will be off by whatever fixed transform actually relates
> the two frames. **Verify before trusting fused RealSense poses** — e.g. put a
> known object at a known base-frame position, compare the ZED-only detection
> (trusted) against the RealSense-fused one, and check they agree once §4's real
> hand-eye calibration is also in place.

### 5.3 Overriding

```bash
scripts/launch_pipeline_realsense.sh fast-track --flange-pose-topic /your/actual/topic
```

or edit `COMMON_ARGS` in
[scripts/launch_pipeline_realsense.sh](../scripts/launch_pipeline_realsense.sh), or
pass different `flange_base_frame`/`flange_frame`/`flange_pose_topic` launch args to
`zed_realsense_trio.launch.py` if you rename frames or switch robots.

**If the tf lookup fails** (e.g. `hardware.launch.py` isn't running, or the frame
names don't match your actual URDF), `flange_pose_publisher` logs a throttled
warning and simply doesn't publish — `MultiCamGrabberRealsense.ready()` then never
returns `True` for the RealSense cameras (it waits for a fresh flange pose no older
than `--flange-pose-max-age-s`, default 0.25 s), so the pipeline sits waiting for
frames rather than running with a wrong/frozen extrinsic.

---

## 6. Pipeline presets and flags

Same presets as the 3-ZED pipeline, same container, different module:

```bash
scripts/launch_pipeline_realsense.sh init-only        # detect & pose every frame, no tracking
scripts/launch_pipeline_realsense.sh fast-track        # detect once, then fast tracking
scripts/launch_pipeline_realsense.sh accurate-track    # detect once, then accurate tracking
```

These restart the same `vision` container as the 3-ZED scripts (same `CONTAINER`
env var override applies) and run
`src.perception.ros.learn_runners.run_pipeline_track_multicam_realsense` instead of
the original module. Logs land in `outputs/logs/*_realsense*`. Each opens a tmux
session (default name `mv_pipeline_realsense`, override with `SESSION=...`) with
a `run` window (the pipeline) and a `keys` window (keyboard start/stop/reset
control, see below); `... stop` / `... attach` manage that session.

Extra flags specific to this variant (on top of everything in
[getting_started.md §6](getting_started.md#6-experimenting-with-flags)):

| Flag | Default | Meaning |
|---|---|---|
| `--flange-pose-topic` | `/iiwa/ee_pose` | Published by `flange_pose_publisher`, see §5 above. |
| `--flange-pose-max-age-s` | `0.25` | Reject a flange pose older than this; grabber stays "not ready" until a fresh one arrives. |

`--num-cameras` is fixed to `3` for this variant (1 ZED + 2 RealSense, always all
three) — passing anything else raises an error at startup.

### Start/stop control without reloading models

Same as the 3-ZED pipeline (see
[pipeline_walkthrough.md](pipeline_walkthrough.md#startstop-control-without-reloading-models)):
model loading (SAM, DINO, GDINO, FoundationPose, Cutie) happens once at startup
and is the expensive part, so prefer these services over `Ctrl+C` when you just
want to pause/resume tracking.

| Service | Type | Effect |
|---|---|---|
| `/foundationpose_tracker/set_tracking_active` | `std_srvs/srv/SetBool` | `data: false` stops ticking and clears all track state. `data: true` resumes (re-runs multicam init on the next tick). |
| `/foundationpose_tracker/reset_tracking` | `std_srvs/srv/Trigger` | Clears all track state without stopping — forces a fresh re-init while staying "running". |

```bash
# stop tracking, keep all models resident
ros2 service call /foundationpose_tracker/set_tracking_active std_srvs/srv/SetBool "{data: false}"

# resume — instant, no model reload
ros2 service call /foundationpose_tracker/set_tracking_active std_srvs/srv/SetBool "{data: true}"

# force a re-init without stopping
ros2 service call /foundationpose_tracker/reset_tracking std_srvs/srv/Trigger {}
```

Pass `--start-paused` to load everything but leave tracking inactive until the
first `set_tracking_active` call.

`scripts/launch_pipeline_realsense.sh` already starts
`src.perception.ros.tracking_keyboard_control` for you in the `keys` tmux
window (mapping `s`/`x`/`r` to these same services — see
[pipeline_walkthrough.md](pipeline_walkthrough.md#keyboard-control-local-debugging)).
Switch to it with `Ctrl+b 1`. To run it by hand elsewhere it works unchanged
against this variant (same node name, `foundationpose_tracker`); no
`--node-name` override needed.

---

## 7. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| Pipeline hangs, never processes a frame | `flange_pose_publisher` isn't running or its tf lookup is failing (check its terminal/log for the throttled warning), or `--flange-pose-topic` points at a different name than it's publishing on. `MultiCamGrabberRealsense.ready()` blocks until every dynamic camera has a fresh pose. |
| `flange_pose_publisher` logs "tf lookup ... failed" repeatedly | Step 1 (§1) wasn't actually done — neither the real LBR bringup nor the placeholder `static_transform_publisher` is running. Check directly: `ros2 run tf2_ros tf2_echo lbr_link_0 lbr_link_ee`. |
| `ros2_control_node` stuck on "Awaiting robot heartbeat" | The KUKA FRI application on the controller/pendant hasn't started streaming yet — this is on the robot-controller side, not fixable from the ROS side. Use the identity placeholder (§1 Step 1, Option B) in the meantime. |
| `docker exec` fails with "the input device is not a TTY" | `scripts/launch_pipeline_realsense.sh` needs a real interactive terminal (or a tmux window) — it was run through a non-interactive shell/script/tool. Run it directly at a terminal prompt instead. |
| RealSense node crashes immediately with `parameter 'serial_no' has invalid type ... {integer} is not allowed` | Already fixed in `zed_realsense_trio.launch.py` — the serial is wrapped as `["'", rs1_serial, "'"]` at the point it's passed to `rs_launch.py` so it loads as a string, not an int. If you see this again, check that wrapping wasn't accidentally reverted. |
| ZED reports `CAMERA NOT DETECTED` even though a ZED is plugged in | The `cam1_serial` launch arg doesn't match the physically mounted ZED's serial. Check with `ZED_Diagnostic` or the ZED Explorer app, then override: `ros2 launch mv_launch zed_realsense_trio.launch.py cam1_serial:=<actual_serial>` (or edit the default in the launch file if the mounted ZED changed permanently). |
| `ros2 topic hz /realsense_*/camera/color/image_raw` shows nothing | Wrong/duplicate serial number (§3), camera on a USB2 port/hub (needs USB3), or the second RealSense wasn't power/bandwidth-capable alongside the first — try separate USB3 controllers. |
| Fused pose is clearly wrong for objects only seen by a RealSense | Expected until §4's calibration has been run for that camera (identity placeholder until then). Also double check the `lbr_link_0 == base` assumption (§5.2) once real calibration is in. |
| `run_pipeline_track_multicam_realsense` errors on `--num-cameras` | This variant only accepts `--num-cameras 3` — don't pass `2` (that flag exists for the original 3-ZED runner, not this one). |
| Depth looks misaligned with color on a RealSense | Check `align_depth.enable:=true` is actually taking effect — it's set in `zed_realsense_trio.launch.py`; confirm via `ros2 topic echo /realsense_1/camera/aligned_depth_to_color/camera_info` matches the color camera_info resolution. |
| `image_rect` topics silent, `image_raw` topics fine | The `<cam>_rectify_container` (an `image_proc` composable-node container) failed to start or crashed — check the `cams` tmux window for its output. Most likely cause: `ros-humble-image-proc` isn't installed on the **host** (see §8) — `sudo apt install ros-humble-image-proc ros-humble-image-pipeline`. |

For anything not specific to the RealSense cameras (GPU saturation, docker
container missing, general pipeline flags), see
[getting_started.md §4](getting_started.md#4-troubleshooting) — it all applies here
unchanged.

---

## 8. RGB/depth rectification (undistorting the RealSense color stream)

**Why this exists**: unlike the ZED wrapper, which rectifies the color image
internally before publishing (`rgb/color/rect/image`), `realsense2_camera`
publishes the D405 color stream **as-is, still distorted** —
`color/image_raw`, not `image_rect_raw` (confirmed in
`realsense2_camera`'s `rs_node_setup.cpp`: only `RS2_STREAM_DEPTH` and IR/`Y8`
formats are marked `rectified_image`, color is not). The D405's actual
distortion is real, not a rounding-error formality — checked live via
`rs-enumerate-devices -c -v` on both mounted units:

```
Intrinsic of "Color" / 640x480 / {YUYV/RGB8/BGR8/RGBA8/BGRA8}
  Distortion: Inverse Brown Conrady
  Coeffs:     -0.0552926  0.0611028  0.000408301  3.47888e-05  -0.0209464
```

(By contrast the D405's **depth** stream reports `Coeffs: 0 0 0 0 0` — it's
already rectified, nothing to do there on its own.)

The `realsense2_camera` package has **no built-in flag** for this — there is
no `enable_rectify`/`color.rectify`/equivalent parameter anywhere in
`rs_launch.py`'s ~90 configurable parameters, and no mention of
rectification/undistortion anywhere else in the vendored package (README,
launch files, or source). It expects a downstream consumer to do it, same as
any other unrectified ROS camera driver.

**What was added** — a per-camera `image_proc` rectify stage in
`zed_realsense_trio.launch.py` (`_rectify_container(...)`, one
`ComposableNodeContainer` per RealSense with two `image_proc::RectifyNode`
instances):

| Node | Subscribes | Publishes | `interpolation` |
|---|---|---|---|
| `rectify_color_node` | `camera/color/image_raw` + `camera/color/camera_info` | `camera/color/image_rect` | `1` (linear — fine for RGB) |
| `rectify_aligned_depth_node` | `camera/aligned_depth_to_color/image_raw` + `camera/aligned_depth_to_color/camera_info` | `camera/aligned_depth_to_color/image_rect` | `0` (nearest-neighbor — avoids inventing blended foreground/background depth at object silhouettes) |

**Why depth needs rectifying too, not just color**: `aligned_depth_to_color`
is registered to the *distorted* color pixel grid (that's what "aligned to
color" means for this driver), and `MultiCamGrabberRealsense` backprojects
using the **depth** stream's `camera_info.K`
(`multicam_grabber.py`/`multicam_grabber_realsense.py`, `_K_depth`, matching
what the ZED path does with `depth_registered`). If only color were
rectified, RGB pixels and depth pixels covering the same physical point would
land at different pixel coordinates, and detection masks/silhouettes (SAM,
DINO — everything reading `View.rgb`, see `src/perception/view.py`) would no
longer line up with `View.depth` at the same indices. Both rectify nodes are
fed the color stream's distortion coefficients (`aligned_depth_to_color`'s
`camera_info` is derived from the color intrinsics by the driver), so
`color/image_rect` and `aligned_depth_to_color/image_rect` end up back on one
shared, correctly-registered pixel grid.

**Why no new `camera_info` was needed**: `image_proc::RectifyNode` publishes
*only* the rectified image, not a corresponding rectified `CameraInfo` — the
correct rectified intrinsics are conventionally `CameraInfo.P`'s 3×3 block
(what `image_geometry::PinholeCameraModel` reprojects into via
`cv::initUndistortRectifyMap`). `realsense2_camera` already sets
`P` = a straight copy of `K` (`Tx=Ty=0`, see `base_realsense_node.cpp`), so
for this driver the original (still-distorted) `camera_info.K` **is already**
the correct rectified `K` — only the image pixels change, not the published
intrinsics. That's why `run_pipeline_track_multicam_realsense.py`'s
`ALL_CAMERAS` entries for `realsense_1`/`realsense_2` still point
`info_topic`/`rgb_info_topic` at the original `camera_info` topics, only the
image topics changed to `.../image_rect`.

**What consumes the rectified topics now**:
- `src/perception/ros/learn_runners/run_pipeline_track_multicam_realsense.py`
  (`ALL_CAMERAS`) — `depth_topic`/`rgb_topic` for both RealSense cameras.
- `scripts/launch_host_realsense.sh` (`VIZ2_CMD`/`VIZ3_CMD`) — the debug
  visualizers, so what you see in Foxglove matches what the pipeline
  actually consumes.
- `src/calibration/handeye_flange_cam_realsense.py` (`_camera_topics`, also
  used by `board_pose_from_flange_realsense.py`) — this was a **real
  pre-existing accuracy bug**, not just a topic-naming inconsistency:
  `_solve_board_pose`/`_compute_reproj_err_px` hardcode `dist = 0` for
  `cv2.solvePnP`/`cv2.projectPoints`, which was silently wrong while reading
  the distorted `image_raw` (any hand-eye calibration run before this change
  solved PnP against a distorted image as if undistorted, biasing
  `T_flange_cam`). Reading `image_rect` instead makes the `dist=0` assumption
  correct.

**Where it runs**: the rectify containers are declared inside
`zed_realsense_trio.launch.py` and start automatically as part of
`scripts/launch_host_realsense.sh` (the `cams` tmux window) — **on the host**,
same as the camera drivers, not inside the `vision` Docker container. See §3
for why: the whole camera stack (ZED, both RealSenses, `flange_pose_publisher`)
runs bare-metal via `launch_host_realsense.sh`; only the perception pipeline
runner (`launch_pipeline_realsense.sh`) goes through `docker exec` into
`vision`. `image_proc`/`image_pipeline` were therefore installed on the
**host** system, not in the container:

```bash
sudo apt install ros-humble-image-proc ros-humble-image-pipeline
```

If you ever rebuild `mv_launch` after editing `zed_realsense_trio.launch.py`
(§3's rebuild note also applies here — it's a `launch/` file, not compiled
source, but the package needs reinstalling for the change to be picked up by
`ros2 launch`):

```bash
cd ~/franka_ros2_ws
colcon build --packages-select mv_launch
source install/setup.bash
```

The raw `color/image_raw` and `aligned_depth_to_color/image_raw` topics are
still published by the driver (nothing subscribes to them anymore within this
repo, but they're harmless to leave running and useful as an independent
liveness check — see the Step 2 verify block in §1 and the troubleshooting
table above).

---

## 9. Where to read more

- **[getting_started.md](getting_started.md)** — the original 3-ZED guide this
  variant is based on; calibration mental model, Foxglove setup, remote access, and
  the full troubleshooting table still apply.
- **[calibration_cheatsheet.md](calibration_cheatsheet.md)** — condensed,
  copy-pasteable command sequence for the whole dual-arm calibration routine
  (§4 below), print-and-tape-to-the-rig style.
- **[moveit_robot_control.md](moveit_robot_control.md)** — how to jog the KUKA
  LBR via MoveIt's RViz plugin, used to position the arm during §4.3 (Step 1)
  captures.
- **[README §6](../README.md#6-realsense-trio-variant)**
  — condensed version of §1 above, for quick reference.
- **[config/camera_extrinsics_realsense.yaml](../config/camera_extrinsics_realsense.yaml)**
  — extrinsics file, with inline comments on the camera-to-flange convention.
- **[config/robot_bases.yaml](../config/robot_bases.yaml)** — named robot base
  offsets (Robot A / Robot B) and the `active_robot` selector used by §4's
  calibration scripts.
- **[config/flange_poses/](../config/flange_poses/)** — permanent per-arm
  flange pose captures written by `capture_flange_poses_dual.py` (§4.3) and
  read back by `autocalibrate_dual_realsense.py` (§4.4). See
  `src/calibration/flange_pose_store.py` for the JSON schema.
- **[outputs/calibration_logs/](../outputs/calibration_logs/)** — append-only
  JSON history of every calibration run's transforms + quality metrics (§4.6),
  written by `calibration_log.py`.
- **[src/calibration/capture_flange_poses_dual.py](../src/calibration/capture_flange_poses_dual.py)**
  — dual-arm routine Step 1 (manual): jog each arm, save its flange poses
  permanently. See §4.3.
- **[src/calibration/autocalibrate_dual_realsense.py](../src/calibration/autocalibrate_dual_realsense.py)**
  — dual-arm routine Step 2 (automatic): replays the saved poses to solve
  hand-eye (Stage A), checkerboard pose (Stage B), and the ZED extrinsic
  (Stage C). See §4.4.
- **[src/calibration/moveit_dual_arm.py](../src/calibration/moveit_dual_arm.py)**
  — `MoveGroup` action-client helper used by Step 2 above; the only code in
  this repo that sends motion commands to the robot (`ArmTarget`,
  `DualArmMoveitClient.move_to`, incl. the simultaneous `both_arms` goal).
- **[src/calibration/flange_pose_store.py](../src/calibration/flange_pose_store.py)**
  — JSON schema + save/load for `config/flange_poses/<left,right>.json`.
- **[src/calibration/calibration_log.py](../src/calibration/calibration_log.py)**
  — append-only JSON run logs under `outputs/calibration_logs/` (§4.6).
- **[src/calibration/handeye_flange_cam_realsense.py](../src/calibration/handeye_flange_cam_realsense.py)**
  — manual single-camera fallback (§4.7): solves `T_flange_cam` via
  `cv2.calibrateHandEye`. Its PnP/reprojection/AX=XB-solving helpers are
  reused directly by the dual-arm Step 2 script above.
- **[src/calibration/board_pose_from_flange_realsense.py](../src/calibration/board_pose_from_flange_realsense.py)**
  — manual single-camera fallback (§4.7): computes the checkerboard's pose in
  the robot base frame using Stage 1's result. Its averaging/RPY helpers are
  reused directly by the dual-arm Step 2 script above.
- **[src/calibration/base_to_cams_calib_3.py](../src/calibration/base_to_cams_calib_3.py)**
  — static camera-to-base calibration, now generalized to an arbitrary
  `--cam-ids` list (defaults to the 3-ZED trio). §4.4 Stage C pins this to
  the single `zed2i_1` via `scripts/calibrate_zed_from_board_pose.sh`.
- **[src/calibration/realsense_devices.py](../src/calibration/realsense_devices.py)**
  — wraps `rs-enumerate-devices -s` to list/match connected RealSense cameras.
- **[src/perception/ros/multicam_grabber_realsense.py](../src/perception/ros/multicam_grabber_realsense.py)**
  — the dynamic-extrinsics grabber; `_FlangeComposedExtrinsicsMap` is where the
  per-frame `T_base_flange @ T_flange_cam` composition happens.
- **[../../franka_ros2_ws/src/mv_launch/launch/zed_realsense_trio.launch.py](../../franka_ros2_ws/src/mv_launch/launch/zed_realsense_trio.launch.py)**
  — camera launch file (host side), also starts `flange_pose_publisher` and (§8)
  the per-camera `image_proc` rectify containers.
- **[../../franka_ros2_ws/src/mv_launch/mv_launch/flange_pose_publisher.py](../../franka_ros2_ws/src/mv_launch/mv_launch/flange_pose_publisher.py)**
  — tf2-based flange pose node; read its docstring for the `lbr_link_0 == base`
  assumption in full.
- **§8** — RGB/depth rectification: why the RealSense color stream needs it
  (real Brown-Conrady distortion, unlike the ZED which rectifies internally),
  why depth needs rectifying too (pixel-grid alignment with `View.rgb`), and
  the `ros-humble-image-proc`/`ros-humble-image-pipeline` host dependency it
  introduces.
