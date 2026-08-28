# RealSense Pipeline — Startup Behavior & CLI Reference

Commands-and-behavior reference for the RealSense-trio pipeline (1 static ZED
+ 2 end-effector-mounted RealSense cameras, dual-arm KUKA rig). For the full
walkthrough (hardware, calibration, first run) see
[getting_started_realsense.md](getting_started_realsense.md).

---

## 1. Startup no longer blocks on flange pose

`MultiCamGrabberRealsense` (`src/perception/ros/multicam_grabber_realsense.py`)
and the runner it feeds
(`src/perception/ros/learn_runners/run_pipeline_track_multicam_realsense.py`)
start ticking as soon as **some** configured cameras are usable — it does not
wait for every camera, or for a flange pose, before doing anything.

**Per-camera readiness** (`_camera_individually_ready`): a camera counts as
ready once its RGB + depth images and both `CameraInfo` messages have
arrived, and — only for end-effector-mounted (dynamic) cameras — a flange
pose no older than `--flange-pose-max-age-s` (default `0.25s`) has been
received on its `flange_pose_topic`.

**Grabber readiness** (`ready()` / `active_cameras()`): the grabber is
"ready" once **`--min-active-cameras`** (default **1**) of the configured
cameras are individually ready. With the default, the pipeline starts
ticking the moment the *first* camera comes up — commonly the static ZED,
which needs no flange pose at all — instead of waiting for all 3. Every
tick recomputes which cameras are currently ready
(`active_cameras()`/`get_latest_views()`), so:

- The RealSense cameras join automatically once their flange pose starts
  publishing — no restart needed.
- If a flange pose gap causes a camera to drop out, it's simply excluded
  from that tick's `views` (and rejoins once fresh again); tracking for the
  other cameras is unaffected.
- The runner logs `active camera set changed: [...]` whenever the set of
  ready cameras changes, so you can see this happening live.

**Stale vs. missing flange pose** (`--strict-flange-freshness`, default
**off**): a flange pose older than `--flange-pose-max-age-s` is, by default,
still used (with a throttled `flange pose is stale` warning) rather than
dropping the camera. Pass `--strict-flange-freshness` to restore the old
hard cutoff (stale ⇒ treated as missing ⇒ camera drops out of
`active_cameras()`), which is what calibration capture scripts want (a
stale pose there means a bad/desynced sample, not "keep going").

**What this does *not* fix:** if robot bringup (`hardware.launch.py`) was
never launched, `flange_pose_publisher` never publishes anything, so a
dynamic camera's flange pose is not "stale" — it is permanently absent, and
that camera never becomes individually ready. With the default
`--min-active-cameras 1`, the pipeline still starts and ticks using
whichever camera(s) *are* up (e.g. the static ZED alone); it just never
picks up the RealSense cameras until bringup is actually running and
publishing flange poses. Set `--min-active-cameras 3` to restore the old
all-cameras-required behavior (e.g. for a baseline/init-only run where a
partial camera set isn't useful).

Set `--min-active-cameras` to the runner directly, or add it after the mode
name when using the launch scripts, e.g.:

```bash
scripts/launch_pipeline_realsense.sh fast-track --min-active-cameras 3
```

> This behavior is specific to the RealSense-trio grabber
> (`multicam_grabber_realsense.py`). The 3-ZED pipeline's grabber
> (`multicam_grabber.py`, used by `scripts/launch_pipeline.sh`) has no
> `min_active_cameras`/stale-flange handling — it still requires every
> configured camera before `ready()` returns `True` (moot there since none
> of the 3 ZEDs are end-effector-mounted/dynamic).

---

## 2. `scripts/launch_host_realsense.sh`

Host-side stack (camera drivers + debug windows) in one tmux session
(`$SESSION`, default `mv_host_realsense`).

| Command / flag | Effect |
|---|---|
| *(no args)* | Start the session (or attach if already running) |
| `stop` | Kill the tmux session |
| `attach` | Attach to the already-running session |
| `--no-visualize` | Skip the 3 `viz1`/`viz2`/`viz3` windows (`visualize_pipeline` per camera). **On by default** so the mask + tracked-axes overlays show in Foxglove — each instance renders+publishes 5 overlay images at 5 Hz whether or not anything subscribes, so pass this for a lean unattended run. |
| `--no-foxglove` | Skip the `foxglove` window (`foxglove_bridge`). **On by default** — its `topic_whitelist` is `.*`, so any client that connects can pull full-res images/IMU/point cloud for anything on the graph; pass this for a lean unattended run. |

Env var overrides:

| Var | Default | Purpose |
|---|---|---|
| `SESSION` | `mv_host_realsense` | tmux session name |
| `RS1_SERIAL` | `260522275434` | realsense_1 camera serial |
| `RS2_SERIAL` | `260322275185` | realsense_2 camera serial |
| `CAM1_SERIAL` | `33137761` | zed2i_1 camera serial |

Windows created: `cams` (camera drivers), `viz1`/`viz2`/`viz3` (unless
`--no-visualize`), `axes` (`debug_pose_axes`), then `foxglove` (unless
`--no-foxglove`).

---

## 3. `scripts/launch_pipeline_realsense.sh`

Restarts the `vision` docker container and either drops into an interactive
shell or runs a pinned preset inside a tmux session (`$SESSION`, default
`mv_pipeline_realsense`).

| Command / flag | Effect |
|---|---|
| *(no args)* | Restart container, interactive shell (sourced, no pipeline run) |
| `init-only` / `baseline` | Locked 3-camera init baseline preset |
| `fast-track` | Fast tracking preset |
| `accurate-track` | Settled-axis accuracy preset |
| `stop` | Kill the tmux session |
| `attach` | Attach to the already-running session |
| `--disable-debug-frames` | Skip building/publishing `fp_debug_msgs/DebugFrame` (and the `/perception/fp/*_overlay/*` topics). **On by default** (this flag adds `--no-debug-frame-publish` to the runner). |
| anything else | Passed straight through as runner args (raw `run_pipeline_track_multicam_realsense` CLI) |

Env var overrides:

| Var | Default | Purpose |
|---|---|---|
| `CONTAINER` | `vision` | docker container name |
| `SESSION` | `mv_pipeline_realsense` | tmux session name |

Preset windows: `run` (the pipeline itself), `keys`
(`tracking_keyboard_control`, `s`=start `x`=stop `r`=reset), `cam-scene`
(`publish_camera_scene_objects.py --dual-arm` — publishes the ZED
camera+holder meshes as `CollisionObjects`, safe alongside a real MoveIt on
the same network).

Readiness flags you can append after a mode name (see §1 for what they do):

```bash
scripts/launch_pipeline_realsense.sh fast-track --min-active-cameras 3
scripts/launch_pipeline_realsense.sh fast-track --strict-flange-freshness
scripts/launch_pipeline_realsense.sh fast-track --flange-pose-max-age-s 0.5
```

---

## 4. `scripts/launch_pipeline.sh` (3-ZED variant)

Same container/tmux mechanics as §3, different module
(`run_pipeline_track_multicam` instead of the RealSense one), same presets
(`init-only`/`baseline`/`fast-track`/`accurate-track`), same
`--disable-debug-frames` flag and default-on behavior. No
`--min-active-cameras`/`--strict-flange-freshness`/`--flange-pose-max-age-s`
readiness flags — see the note at the end of §1.

---

## 5. Debug/logging runner flags

Flags accepted directly by
`run_pipeline_track_multicam_realsense` (and `run_pipeline_track_multicam`
for the 3-ZED variant), useful when appending raw args after a preset name
or in the interactive shell:

| Flag | Default | Effect |
|---|---|---|
| `--debug-frame-publish` / `--no-debug-frame-publish` | `--debug-frame-publish` (True) at the runner level — the launch scripts leave this on unless `--disable-debug-frames` is passed (see §3/§4) | Publish `fp_debug_msgs/DebugFrame` messages |
| `--debug-logging` | off | Enable all pipeline logging (prints + ROS logger info/warn) |
| `--debug-verbose-logs` | off | Per-frame `INFO` logs and `[TIMING]` prints |
| `--debug-per-cam-pose-publish` | off | Publish per-camera (pre-fusion) poses for debugging |
| `--min-active-cameras N` | `1` (RealSense variant only) | See §1 |
| `--flange-pose-max-age-s S` | `0.25` (RealSense variant only) | See §1 |
| `--strict-flange-freshness` | off (RealSense variant only) | See §1 |
