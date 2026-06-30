# Getting Started — Calibrate & Run (for students)

A step-by-step guide to **calibrating the cameras** and **running the pipeline**
on the lab rig. Follow it top to bottom the first time.

This guide assumes the rig is **already set up**: the `vision` docker container is
built, `Data/` holds the meshes + reference crops, and the three ZED 2i cameras are
mounted and plugged in. If you are setting up a *fresh machine* instead, do the
one-time setup in the [README](../README.md#2-setup) (submodules, Cutie, Docker,
`Data/`) first, then come back here.

>  **Just want the commands?** There's a one-page
> [cheat sheet](cheatsheet.md) to print and tape next to the rig.
>
>  **Working from your own laptop, not at the rig?** Jump to
> [§5 Remote access](#5-remote-access-from-your-own-laptop) first, then follow the
> rest over SSH.

> **Mental model.** Two halves run at the same time:
> 1. the **host stack** (cameras + viewer) — runs on the host, in tmux;
> 2. the **pipeline** (the perception node) — runs inside the `vision` docker container.
>
> You always start the host stack first (so the cameras publish), then either
> calibrate or run the pipeline.

---

## 0. Before you start — a 30-second check

```bash
cd ~/Masterthesis-vision

docker ps -a --filter name=vision     # the 'vision' container should be listed
nvidia-smi                            # GPU should be mostly free (see warning below)
ls Data/CAD_Models_centered           # meshes should be here
ls Data/ZED_screens                   # one folder per object
```

> ⚠️ **GPU must be free.** If another job is using most of the GPU, the pipeline
> green-outs and SAM/DINO produce garbage (NaNs). Check `nvidia-smi` and wait for
> the GPU before launching. (Recorded bags still work; the live pipeline does not.)

---

## 1. Start the host stack (cameras + viewer)

This launches a tmux session with the ZED driver, the Foxglove bridge, and a
visualizer per camera.

```bash
scripts/launch_host.sh            # start and attach
# scripts/launch_host.sh attach   # re-attach later if it's already running
# scripts/launch_host.sh stop     # kill it when you're done
```

Inside tmux:
- `Ctrl+b` then `0`..`5` — switch windows (`0 = cams`, `1 = foxglove`, `2-4 = viz`, `5 = axes`).
- `Ctrl+b d` — detach (leaves everything running).

**Verify the cameras are publishing** before moving on. In a separate terminal:

```bash
source /opt/ros/humble/setup.bash
ros2 topic hz /zed2i_1/zed_node/rgb/color/rect/image    # ~15-30 Hz expected
```

You should see all three (`zed2i_1`, `zed2i_2`, `zed2i_3`) publishing. If a camera
is missing, check its USB connection and the `cams` tmux window for errors.

### Watch the result in Foxglove

The host stack runs a Foxglove bridge on port **8765**. Open **Foxglove Studio** →
*Open connection* → `ws://localhost:8765` if Studio runs on the desktop PC itself,
or `ws://10.5.6.204:8765` from another machine (see [§5](#5-remote-access-from-your-own-laptop)).

**Import the shared layout once** so you get the same panels as everyone else:
Foxglove Studio → *Layouts* (left sidebar) → *Import from file…* →
[docs/Franka_Perception.json](Franka_Perception.json). After that it's just a saved
layout you can pick from the list.

Useful image topics (per camera, `zed2i_1/2/3`):

| Topic | Shows |
|-------|-------|
| `/perception/fp/sam_overlay/zed2i_1_external`  | SAM masks |
| `/perception/fp/dino_overlay/zed2i_1_external` | DINO class labels |
| `/perception/fp/pose_overlay/zed2i_1_external` | FoundationPose result |
| `/perception/fp/track_overlay/zed2i_1_external`| live tracking overlay |

The pose axis markers (from the `axes` window) show up as 3D markers in the base
frame.

---

## 2. Calibrate the cameras

**You only need to do this when the cameras have moved** (bumped, remounted, or you
don't trust the current extrinsics). If the cameras haven't moved since the last
good calibration, **skip to [§3 Run the pipeline](#3-run-the-pipeline)** — the saved
`config/camera_extrinsics_base.yaml` is still valid.

Calibration finds where each camera sits **relative to the robot base frame**, using
a checkerboard held in a known spot. The result is written to
`config/camera_extrinsics_base.yaml`, which the pipeline loads at startup.

> **Two places hold the board info — keep them straight:**
> - **`config/base_board_pose.yaml`** — *where* the board is: the offset/orientation
>   of its **first (origin) corner** in the robot base frame. Edit this every time
>   the board sits somewhere new.
> - **`src/calibration/base_to_cams_calib_3.py`** (constants at the top) — *what* the
>   board is: **how many inner corners** and the **square size**. Only edit this if
>   you physically swap the checkerboard.

### 2.1 Prerequisites

- The **host stack from §1 must be running** (the cameras must be publishing).
- You need the **physical checkerboard**: **8 × 11 inner corners**, **30 mm** squares.
  (These are the defaults baked into the script — see the constants below.)

### 2.2 Tell the script where the board is

Edit [config/base_board_pose.yaml](../config/base_board_pose.yaml) — this is the pose
of the checkerboard's **origin corner** in the robot base frame:

```yaml
base_board:
  translation_xyz_m: [-0.0095, 0.089, 0.0065]   # board origin corner in base frame [m]
  rotation_rpy_deg:  [0.0, 180.0, 0.0]          # board orientation in base frame [deg]
```

- **`translation_xyz_m`** — measure (or read off the robot) where the board's origin
  corner sits in the base frame, in metres.
- **`rotation_rpy_deg`** — for a flat board lying on the table, start with
  `[0, 180, 0]`. If the resulting poses come out mirrored/upside-down, try flipping
  one sign (e.g. `[180, 0, 0]`).

If you ever change the physical board, also update the geometry constants at the top
of [src/calibration/base_to_cams_calib_3.py](../src/calibration/base_to_cams_calib_3.py):
`CHESS_COLS = 8`, `CHESS_ROWS = 11`, `SQUARE_SIZE_M = 0.03`.

### 2.3 Run the calibration

Open a shell **inside the container** (it has all the Python deps and ROS sourced):

```bash
scripts/launch_pipeline.sh        # no args → restarts container, drops you in a shell
```

Then, in that container shell:

```bash
python3 -m src.calibration.base_to_cams_calib_3
```

**Hold the checkerboard so all three cameras see it at once, and keep it steady.**
The script collects 8 good 3-camera samples, so hold position for a few seconds. It
will:

- reject blurry / unsynchronized / high-error frames automatically;
- average the 8 samples and check the spread is tight (translation < 1 cm, rotation < 1°);
- **back up** the old `camera_extrinsics_base.yaml` to `.yaml.bak` and write the new one;
- save annotated corner images to `outputs/calibration_debug/` so you can inspect them.

### 2.4 Check the result

Before trusting the calibration, read the script's final printout:

- **Mean reprojection error** — should be roughly **≤ 1–2 px**. High values mean bad corner detection (lighting, motion blur, wrong board constants).
- **`T_cam1_cam2` / `T_cam1_cam3` consistency** — the camera-to-camera transforms should look sane (cameras ~0.5–1 m apart, not metres off).
- If the script **raises and writes nothing**, the spread was too large → re-run, holding the board more steadily and making sure all three cameras have a clear, well-lit view.

Open a couple of images in `outputs/calibration_debug/` to confirm the corners were
detected correctly on every camera.

---

## 3. Run the pipeline

With the host stack running (§1) and a valid calibration in place (§2), start the
perception node. There are three presets — pick one:

```bash
scripts/launch_pipeline.sh init-only        # detect & pose every frame, no tracking
scripts/launch_pipeline.sh fast-track       # detect once, then fast tracking
scripts/launch_pipeline.sh accurate-track   # detect once, then accurate (settled-axis) tracking
```

Each command restarts the `vision` container, sources everything, runs the node, and
tees the log to `outputs/logs/`.

**Which one should I use?**

| Preset | Use it when |
|--------|-------------|
| `init-only`      | You just want per-frame detections + poses (no tracking). Good for a first smoke test and for checking detection/pose quality. |
| `fast-track`     | The object **moves** (up to ~1 m/s) and you want the pose to keep up. Position tracking is solid; rotation about the object's long axis can be loose. |
| `accurate-track` | The object is **slow/settling** and you care about a good orientation. Adds rotation reseed + PCA axis + light damping. |

> **Known limitation:** the rotation **about a shaft's long axis** is poorly
> observed (the objective is depth-based and nearly flat along that axis).
> `accurate-track` helps but does not fully fix it. Translation and the other two
> rotation axes are reliable.

### The output — the live ROS pose stream

**This is the part other students consume.** For every tracked object the node
publishes a base-frame `geometry_msgs/PoseStamped` (frame `base`) on:

```
/perception/fp/pose_base/fused/<track_id>
```

Subscribe to it from your own ROS node, or inspect it from a sourced ROS shell:

```bash
source /opt/ros/humble/setup.bash
ros2 topic list | grep /perception/fp/pose_base     # list the per-object topics
ros2 topic echo /perception/fp/pose_base/fused/<track_id>
```

Visual confirmation (not the interface, just for checking):
- in **Foxglove**, the overlay topics (§1) light up with masks, labels and pose axes — the `axes` window draws each pose as a 3D axis marker in the base frame;
- the **terminal** logs each tick (detections, fused objects, pose timing).

<details><summary>Debug / offline files (optional)</summary>

Written only when the logging flags are set (the presets enable them); for offline
analysis, **not** the live interface:

| Path | What |
|------|------|
| `outputs/logs/*.log`              | full run log (per preset) |
| `outputs/logs/live_*_track_q.csv` | per-frame track poses (quaternions) |
| `init_pose_log.csv`               | per-init pose rows |
| `init_renders/`                   | rendered pose overlays saved at init |

</details>

To stop: `Ctrl+C` in the pipeline terminal. Stop the host stack with
`scripts/launch_host.sh stop` when you're completely done.

---

## 4. Troubleshooting

| Symptom | Likely cause / fix |
|---------|--------------------|
| Image goes green / SAM & DINO output NaNs | **GPU is saturated** by another job. Check `nvidia-smi`, wait for it to free up, relaunch. |
| A camera topic is missing / `ros2 topic hz` shows nothing | Camera USB unplugged or driver crashed. Check the `cams` tmux window; replug and re-run `scripts/launch_host.sh`. |
| Calibration raises "spread too large" and writes nothing | Board wasn't steady, or a camera's view was poor. Re-run holding the board still with all three cameras seeing it clearly; check lighting. |
| Poses look mirrored / upside-down after calibration | Wrong board orientation. Flip a sign in `rotation_rpy_deg` in `config/base_board_pose.yaml` and recalibrate. |
| Nothing is detected | Check the SAM/DINO overlays in Foxglove to see where it breaks down. Usually the GPU is busy (green-out) or the object is out of view; lowering `--gdino-box-threshold` makes detection less strict. |
| Object's rotation about its long axis drifts | Known limitation (shaft axis is weakly observed). Use `accurate-track`; expect residual error there. |
| Container won't start | `docker ps -a` — if `vision` is missing, the rig isn't set up; see [README §2.2](../README.md#22-docker). |

---

## 5. Remote access (from your own laptop)

You don't have to sit at the rig — the whole thing runs over SSH, and Foxglove can
view it from your own machine.

Everything runs on the **desktop PC** at `10.5.6.204`. From your own machine you'll
typically have **three terminals open**:

1. one SSH terminal running the **host stack** (§1),
2. a second SSH terminal running the **pipeline** (§3),
3. (Mac/own PC) a third terminal holding the **Foxglove SSH tunnel** (§5.2).

Foxglove Studio itself runs on your own machine.

### 5.1 SSH into the desktop PC

```bash
ssh moreno@10.5.6.204      # PW on Cheatsheet
```

Open a **second** SSH session the same way for the pipeline, so the host stack and the
pipeline each have their own terminal. Both run inside tmux/`docker exec`, so they keep
running even if SSH drops — for the host stack just detach with `Ctrl+b d` and
re-`ssh` + `scripts/launch_host.sh attach` to get back.

### 5.2 View Foxglove on your own machine

Foxglove Studio runs **on your laptop**, not on the desktop PC. The bridge listens on
`0.0.0.0:8765`, so pick whichever is easier:

**A — Direct** (your laptop can reach the desktop PC on the network): in Foxglove
Studio, *Open connection* → `ws://10.5.6.204:8765`.

**B — SSH tunnel** (Mac / when the desktop isn't directly reachable) — in its **own
terminal**:

```bash
ssh -N -L 8765:localhost:8765 moreno@10.5.6.204
```

Leave that terminal open (it prints nothing — that's normal), then point Foxglove at
`ws://localhost:8765`.

> **Foxglove layout — download it to your own machine.** Foxglove imports a *local*
> file, so when you run Studio on your laptop you need a local copy of
> [docs/Franka_Perception.json](Franka_Perception.json) (download it from this repo,
> or `scp moreno@10.5.6.204:~/Masterthesis-vision/docs/Franka_Perception.json .`).
> Then Foxglove Studio → *Layouts* → *Import from file…* → that file.

---

## 6. Experimenting with flags

The presets are just locked sets of flags. **Anything you add after the preset name is
passed straight through** to the pipeline node, so you can tune without editing
scripts:

```bash
scripts/launch_pipeline.sh fast-track --gdino-box-threshold 0.25         # detect more/less aggressively
scripts/launch_pipeline.sh fast-track --num-cameras 2                    # run with two cameras
scripts/launch_pipeline.sh accurate-track --fused-track-rot-lowpass 0.3  # smooth rotation more
```

- The preset flag sets live in [scripts/launch_pipeline.sh](../scripts/launch_pipeline.sh)
  (`COMMON_ARGS`, `FAST_TRACK_ARGS`, `ACCURATE_TRACK_ARGS`) — copy one as a starting
  point for your own preset.
- Full flag list: every `p.add_argument(...)` in
  [run_pipeline_track_multicam.py](../src/perception/ros/learn_runners/run_pipeline_track_multicam.py).
- A flag given twice — once by the preset, once by you — takes **your** value (it comes
  later on the command line), so you can override any preset default this way.

---

## 7. Where to read more

- **[Cheat sheet](cheatsheet.md)** — one-page, commands-only summary to print.
- **[Franka_Perception.json](Franka_Perception.json)** — the shared Foxglove layout (import via *Layouts → Import from file…*).
- **[README](../README.md)** — full repo layout, every launch flag, and from-scratch setup.
- **[docs/pipeline_walkthrough.md](pipeline_walkthrough.md)** — how the algorithm works, stage by stage (GDINO → SAM → DINO → fusion → FoundationPose → ICP, then tracking).
