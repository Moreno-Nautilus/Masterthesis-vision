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
ros2 topic hz /realsense_1/camera/color/image_raw
ros2 topic hz /realsense_2/camera/color/image_raw
ros2 topic echo /iiwa/ee_pose --once
```

All four must return real data before you continue. If a RealSense topic is
silent, check the `cams` tmux window (`Ctrl+b` `0`) for the actual error — see
§7 Troubleshooting for the specific failure modes already hit and fixed once.

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

**To stop**: `Ctrl+C` in the pipeline terminal, then
`scripts/launch_host_realsense.sh stop` for the host stack. If you started a
placeholder `static_transform_publisher` in Step 1, `Ctrl+C` that too.

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

### 4.0 Which robot, and its base frame

[config/robot_bases.yaml](../config/robot_bases.yaml) records the two robots'
base-frame offsets (same coordinate-frame convention, Robot B offset
`[0, 0.84, 0]` m from Robot A) and an `active_robot` selector. Both calibration
scripts below read `active_robot` purely for logging/output-header purposes —
they always compute poses directly in whatever robot's `lbr_link_0` is actually
streaming tf on the machine you run them on, via `/iiwa/ee_pose`. **Set
`active_robot` in that file to match the robot you're actually calibrating
before running either script**, so the output files are correctly labeled. We
start with Robot B.

### 4.1 Bring everything up

Same as [§1](#1-run-it-start-to-finish-the-tested-sequence) Steps 0–2: real
robot bringup (`lbr_bringup hardware.launch.py`, *not* the mock/identity
placeholder — you need real, varied flange motion for this), the host camera
stack (`scripts/launch_host_realsense.sh`), and verify `/iiwa/ee_pose` and both
RealSense RGB topics are live.

For jogging the arm to calibration poses, also bring up MoveIt with a real
motion-planning GUI panel (not the RViz-only scene viewer used for pipeline
visualization):

```bash
ros2 launch lbr_bringup move_group.launch.py model:=iiwa7 rviz:=true
```

This one launch call brings up both `move_group` and RViz, already loaded
with `iiwa7_moveit_config`'s `moveit.rviz`, which includes the MotionPlanning
panel — no separate `rviz.launch.py` call needed. In RViz's **MotionPlanning**
panel, drag the interactive marker on the flange to a candidate pose, then
**Plan & Execute**. This is how you'll position the arm before each captured
sample in both stages below. See
[moveit_robot_control.md](moveit_robot_control.md) for the full walkthrough
(jogging via joints vs. the interactive marker, verifying the executed pose
via tf, etc.).

Confirm which RealSense cameras are actually connected before picking a
`--cam-id`:

```bash
python3 -m src.calibration.realsense_devices
# or, equivalently: rs-enumerate-devices -s
```

### 4.2 Recommended Foxglove layout

Both scripts publish a live debug image topic
(`/calibration/handeye/<cam_id>/debug_image`,
`/calibration/board_pose/<cam_id>/debug_image`) with detected checkerboard
corners overlaid — add an **Image** panel on that topic so you can confirm the
full board is visible and well-detected *before* pressing Enter to capture.
Also useful: a **3D** panel with the robot model (to see the jogged pose) and a
**Raw Messages** panel on `/iiwa/ee_pose` (confirms the flange pose topic is
actually streaming). This reuses the same `foxglove_bridge` already started by
`launch_host_realsense.sh` — nothing extra to launch.

### 4.3 Stage 1 — flange ↔ camera (`T_flange_cam`)

Run once per RealSense camera, inside the `vision` container:

```bash
python3 -m src.calibration.handeye_flange_cam_realsense --cam-id realsense_1
```

For each of **at least 10 samples (15 recommended)**: jog the arm with MoveIt to
a pose where the checkerboard is fully visible to `realsense_1`, let it settle,
then press Enter in the terminal to capture. **Vary orientation, not just
position** — the classic `AX = XB` hand-eye solve
(`cv2.calibrateHandEye`, method `TSAI`) is only well-conditioned with enough
rotational diversity between samples; a dozen samples that only translate the
wrist will not solve reliably. Each sample pairs a checkerboard-in-camera pose
(PnP on the detected corners, same method as
[src/calibration/base_to_cams_calib_3.py](../src/calibration/base_to_cams_calib_3.py))
with the flange-in-base pose read from `/iiwa/ee_pose` at capture time.

The script prints pairwise `AX=XB` residuals as a QA signal (large residuals →
redo with more rotational spread or check the arm had actually stopped moving
before you pressed Enter), then writes `T_flange_cam` into the `realsense_1`
block of `config/camera_extrinsics_realsense.yaml` — backing up the previous
file to `.yaml.bak` first, and leaving the file's header comment and the other
cam_id entries (`zed2i_1`, `realsense_2`) untouched.

Repeat for `realsense_2`.

### 4.4 Stage 2 — checkerboard pose in the robot base frame

Once Stage 1 has given a real (non-identity) `T_flange_cam` for a camera, use
that same camera to compute where the checkerboard actually sits in the robot
base frame — replacing the hand-measured value in
[config/base_board_pose.yaml](../config/base_board_pose.yaml):

```bash
python3 -m src.calibration.board_pose_from_flange_realsense --cam-id realsense_1
```

Keep the checkerboard **fixed** for this whole run (its own pose is what's
being solved for). For each of 5–8 samples, optionally re-jog the arm to a
different pose for redundancy (or just capture repeatedly from one pose), press
Enter to capture. Each sample computes

```
T_base_board = T_base_flange @ T_flange_cam @ T_cam_board
```

directly — no `AX=XB` solve needed here since `T_flange_cam` is already known.
Samples are averaged with the same rotation/translation spread quality gates as
`base_to_cams_calib_3.py`; the result overwrites `config/base_board_pose.yaml`
(backing up to `.yaml.bak` first), labeled with the `active_robot` frame it was
computed in.

This computed board pose is also what
[src/calibration/base_to_cams_calib_3.py](../src/calibration/base_to_cams_calib_3.py)
reads for the static ZED calibration — so re-running that script after Stage 2
picks up the improved (measured, not hand-eyeballed) board pose too.

### 4.5 Sanity-checking the result

Compare `T_flange_cam` translation magnitude against a rough physical estimate
(camera is bolted a few cm from the flange, not tens of centimeters) — a wildly
large translation usually means too few/too correlated samples rather than a
real mechanical offset. Also see the `lbr_link_0 == base` assumption flagged in
[§5.2](#52-whats-implemented-flange_pose_publisher) below: Stage 2's computed
board pose is a good way to *verify* that assumption, since it's now derived
from real tf data instead of a hand-typed guess.

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
the original module. Logs land in `outputs/logs/*_realsense*`.

Extra flags specific to this variant (on top of everything in
[getting_started.md §6](getting_started.md#6-experimenting-with-flags)):

| Flag | Default | Meaning |
|---|---|---|
| `--flange-pose-topic` | `/iiwa/ee_pose` | Published by `flange_pose_publisher`, see §5 above. |
| `--flange-pose-max-age-s` | `0.25` | Reject a flange pose older than this; grabber stays "not ready" until a fresh one arrives. |

`--num-cameras` is fixed to `3` for this variant (1 ZED + 2 RealSense, always all
three) — passing anything else raises an error at startup.

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

For anything not specific to the RealSense cameras (GPU saturation, docker
container missing, general pipeline flags), see
[getting_started.md §4](getting_started.md#4-troubleshooting) — it all applies here
unchanged.

---

## 8. Where to read more

- **[getting_started.md](getting_started.md)** — the original 3-ZED guide this
  variant is based on; calibration mental model, Foxglove setup, remote access, and
  the full troubleshooting table still apply.
- **[moveit_robot_control.md](moveit_robot_control.md)** — how to jog the KUKA
  LBR via MoveIt's RViz plugin, used to position the arm between §4's
  calibration samples.
- **[README §6](../README.md#6-realsense-trio-variant)**
  — condensed version of §1 above, for quick reference.
- **[config/camera_extrinsics_realsense.yaml](../config/camera_extrinsics_realsense.yaml)**
  — extrinsics file, with inline comments on the camera-to-flange convention.
- **[config/robot_bases.yaml](../config/robot_bases.yaml)** — named robot base
  offsets (Robot A / Robot B) and the `active_robot` selector used by §4's
  calibration scripts.
- **[src/calibration/handeye_flange_cam_realsense.py](../src/calibration/handeye_flange_cam_realsense.py)**
  — Stage 1: solves `T_flange_cam` via `cv2.calibrateHandEye`.
- **[src/calibration/board_pose_from_flange_realsense.py](../src/calibration/board_pose_from_flange_realsense.py)**
  — Stage 2: computes the checkerboard's pose in the robot base frame using
  Stage 1's result.
- **[src/calibration/realsense_devices.py](../src/calibration/realsense_devices.py)**
  — wraps `rs-enumerate-devices -s` to list/match connected RealSense cameras.
- **[src/perception/ros/multicam_grabber_realsense.py](../src/perception/ros/multicam_grabber_realsense.py)**
  — the dynamic-extrinsics grabber; `_FlangeComposedExtrinsicsMap` is where the
  per-frame `T_base_flange @ T_flange_cam` composition happens.
- **[../../franka_ros2_ws/src/mv_launch/launch/zed_realsense_trio.launch.py](../../franka_ros2_ws/src/mv_launch/launch/zed_realsense_trio.launch.py)**
  — camera launch file (host side), also starts `flange_pose_publisher`.
- **[../../franka_ros2_ws/src/mv_launch/mv_launch/flange_pose_publisher.py](../../franka_ros2_ws/src/mv_launch/mv_launch/flange_pose_publisher.py)**
  — tf2-based flange pose node; read its docstring for the `lbr_link_0 == base`
  assumption in full.
