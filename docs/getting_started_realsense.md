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

> **Status: runs end-to-end, verified working** (see §1 below for the exact
> tested sequence). One placeholder remains: the hand-eye (camera-to-flange)
> calibration is currently **identity** (§4), so fused poses from the RealSense
> cameras are geometrically wrong until that's replaced with a real measurement.
> Everything else — camera drivers, serials, the flange-pose topic, the pipeline
> itself — is real and working.
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

## 4. Hand-eye calibration (camera-to-flange offset) — still identity, needs real values

`config/camera_extrinsics_realsense.yaml` currently ships **identity** for
`realsense_1` / `realsense_2`: zero translation, no rotation — i.e. "camera optical
frame == flange frame exactly." This is a deliberate placeholder chosen so the
pipeline can be brought up and exercised end-to-end (grabber sync, tf2 lookup,
fusion code paths) before real calibration exists — **it is not a real
measurement and fused poses from these two cameras will be geometrically wrong**
until replaced.

What you need per RealSense camera: the rigid transform from the **flange frame**
(`lbr_link_ee`) to the **camera's optical frame**, i.e. where the camera sits and
points relative to the robot's flange.

Two common ways to get this:

- **CAD/mechanical measurement** — if the camera mount is a known, rigid,
  machined part, measure or pull the offset directly from the mount's CAD model.
  Fastest if the mount design is precise and you trust the drawing.
- **Hand-eye calibration routine** — show a checkerboard (or reuse the board
  described in [getting_started.md §2](getting_started.md#2-calibrate-the-cameras))
  to the wrist camera at several different robot poses, and solve the classic
  `AX = XB` hand-eye problem. This repo does not yet have a ready-made script for
  this — [src/calibration/base_to_cams_calib_3.py](../src/calibration/base_to_cams_calib_3.py)
  solves the *static* multi-camera case and would need adapting (or a new script
  written) for the moving-camera case.

Once you have `R`/`t` (camera-to-flange, same row-major 3×3 `R` + 3-vector `t`
convention as the rest of the repo), edit the `realsense_1`/`realsense_2` blocks in
[config/camera_extrinsics_realsense.yaml](../config/camera_extrinsics_realsense.yaml)
directly — no code changes needed.

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
| Fused pose is clearly wrong for objects only seen by a RealSense | Expected right now — hand-eye calibration (§4) is still identity. Also double check the `lbr_link_0 == base` assumption (§5.2) once real calibration is in. |
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
- **[README §6](../README.md#6-realsense-trio-variant)**
  — condensed version of §1 above, for quick reference.
- **[config/camera_extrinsics_realsense.yaml](../config/camera_extrinsics_realsense.yaml)**
  — extrinsics file, with inline comments on the camera-to-flange convention.
- **[src/perception/ros/multicam_grabber_realsense.py](../src/perception/ros/multicam_grabber_realsense.py)**
  — the dynamic-extrinsics grabber; `_FlangeComposedExtrinsicsMap` is where the
  per-frame `T_base_flange @ T_flange_cam` composition happens.
- **[../../franka_ros2_ws/src/mv_launch/launch/zed_realsense_trio.launch.py](../../franka_ros2_ws/src/mv_launch/launch/zed_realsense_trio.launch.py)**
  — camera launch file (host side), also starts `flange_pose_publisher`.
- **[../../franka_ros2_ws/src/mv_launch/mv_launch/flange_pose_publisher.py](../../franka_ros2_ws/src/mv_launch/mv_launch/flange_pose_publisher.py)**
  — tf2-based flange pose node; read its docstring for the `lbr_link_0 == base`
  assumption in full.
