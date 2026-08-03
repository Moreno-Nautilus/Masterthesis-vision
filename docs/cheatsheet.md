# Main Pipeline (3-ZED) — Execution Cheatsheet

Commands only. For the full explanation, see [getting_started.md](getting_started.md).

Assumes: the `vision` container is built, `Data/` is populated, all three
ZED 2i cameras are mounted and plugged in.

---

## 0. Before you start

```bash
cd ~/Masterthesis-vision
docker ps -a --filter name=vision     # 'vision' container should be listed
nvidia-smi                            # GPU should be mostly free
ls Data/CAD_Models_centered           # meshes should be here
ls Data/ZED_screens                   # one folder per object
```

---

## 1. Start the host stack

```bash
scripts/launch_host.sh            # start and attach
# scripts/launch_host.sh attach   # re-attach later
# scripts/launch_host.sh stop     # kill it when done
```

**Verify** (separate shell):

```bash
source /opt/ros/humble/setup.bash
ros2 topic hz /zed2i_1/zed_node/rgb/color/rect/image
```

Foxglove: `ws://localhost:8765` (or `ws://10.5.6.204:8765` remote) — import
[Franka_Perception.json](Franka_Perception.json) once via
*Layouts → Import from file…*.

---

## 2. Calibrate the cameras (only if they moved)

Skip straight to §3 if `config/camera_extrinsics_base.yaml` is still valid.

```bash
# Better corner detection (HD2K/15fps instead of HD1080/30fps):
scripts/launch_host.sh stop
scripts/launch_host.sh calibrate
```

Edit [config/base_board_pose.yaml](../config/base_board_pose.yaml) with the
checkerboard's origin-corner pose in the base frame, then:

```bash
scripts/launch_pipeline.sh                              # shell inside the container
python3 -m src.calibration.base_to_cams_calib_3          # all 3 cameras
python3 -m src.calibration.base_to_cams_calib_3 --cam-ids zed2i_2 zed2i_3   # subset, if one failed to open
```

Hold the checkerboard steady, visible to all cameras being calibrated, until
it collects 8 good samples. Writes `config/camera_extrinsics_base.yaml`
(backing up the old one to `.yaml.bak`) and debug corner images to
`outputs/calibration_debug/`.

✅ Checkpoint: mean reprojection error ≤ 1–2 px; if the script raises
"spread too large", re-run holding the board steadier.

---

## 3. Run the pipeline

```bash
scripts/launch_pipeline.sh init-only        # detect & pose every frame, no tracking
scripts/launch_pipeline.sh fast-track       # fast tracking, moving objects
scripts/launch_pipeline.sh accurate-track   # settled objects, best orientation
```

Detach: `Ctrl+b d`. Reattach: `scripts/launch_pipeline.sh attach`. Stop:
`scripts/launch_pipeline.sh stop`. Pause/resume without reloading models:
`Ctrl+b 1` then `x`/`s`.

Live output (one shared topic for every part):

```bash
ros2 topic echo /perception/fp/pose_base/fused/assembly
```

---

## 4. Troubleshooting

| Symptom | Fix |
|---|---|
| Green image / NaNs | GPU saturated — check `nvidia-smi`, wait, relaunch. |
| Camera topic silent | Check `cams` tmux window; replug; re-run `launch_host.sh`. |
| Calibration "spread too large" | Hold board steadier, all cameras with a clear view. |
| Poses mirrored/upside-down | Flip a sign in `rotation_rpy_deg` (`config/base_board_pose.yaml`), recalibrate. |
| Nothing detected | Check SAM/DINO overlays in Foxglove — usually GPU busy or object out of view. |
| Container won't start | `docker ps -a` — if `vision` is missing, redo [README §2](../README.md#2-setup). |

---

Full guide: [getting_started.md](getting_started.md). RealSense/dual-arm
variant: [getting_started_realsense.md](getting_started_realsense.md) /
[calibration_cheatsheet.md](calibration_cheatsheet.md).
