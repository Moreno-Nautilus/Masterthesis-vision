# Offline Camera Extrinsics Tuning for Bagviz Visualization

## Overview

The `view_pointclouds.py` bagviz viewer can now load camera extrinsics from config YAML files instead of from the captured `frame_info.yaml`. This enables **offline tuning** of camera parameters without re-running the capture pipeline.

## Why This Matters

Previously:
- Camera poses (T_base_cam) were computed during capture and frozen in `frame_info.yaml`
- To test different camera calibrations, you had to re-run the entire `capture_pipeline_snapshots.py`
- This was slow and impractical for calibration iteration

Now:
- Edit the camera extrinsics YAML files (`config/camera_extrinsics_base.yaml`, `config/camera_extrinsics_realsense.yaml`)
- Re-run `view_pointclouds.py` with `--use-config-extrinsics` to visualize with the updated parameters
- Iteration cycle is seconds instead of minutes

## Usage

### Default Behavior (Backward Compatible)
```bash
python -m tools.bagviz.view_pointclouds --run-dir outputs/bagviz/<run>
```
Uses T_base_cam from the captured `frame_info.yaml` (original behavior).

### With Config Extrinsics (Offline Tuning)
```bash
python -m tools.bagviz.view_pointclouds \
  --run-dir outputs/bagviz/<run> \
  --use-config-extrinsics
```

Loads camera extrinsics from the default config files:
- `config/camera_extrinsics_base.yaml` — static cameras (ZED)
- `config/camera_extrinsics_realsense.yaml` — eye-in-hand cameras (RealSense)

### Custom Config Paths
```bash
python -m tools.bagviz.view_pointclouds \
  --run-dir outputs/bagviz/<run> \
  --use-config-extrinsics \
  --extrinsics-yaml config/camera_extrinsics_base.yaml \
  --extrinsics-realsense-yaml config/camera_extrinsics_realsense.yaml
```

## Workflow: Tuning Camera Extrinsics

1. **Capture a bagviz snapshot** (with the current calibration):
   ```bash
   python tools/bagviz/capture_pipeline_snapshots.py --bag ~/Desktop/rosbag_...
   ```

2. **View the result** to understand the current state:
   ```bash
   python -m tools.bagviz.view_pointclouds --run-dir outputs/bagviz/<run>
   ```

3. **Edit the camera extrinsics YAML** based on what you see:
   - Open `config/camera_extrinsics_base.yaml` (for ZED)
   - Open `config/camera_extrinsics_realsense.yaml` (for RealSense)
   - Adjust the R and t values

4. **Re-visualize with updated parameters** (no re-capture needed):
   ```bash
   python -m tools.bagviz.view_pointclouds \
     --run-dir outputs/bagviz/<run> \
     --use-config-extrinsics
   ```

5. **Compare visualizations** side-by-side:
   - Terminal 1: `--use-config-extrinsics` (updated)
   - Terminal 2: without flag (original)

6. **Repeat** steps 3–5 until camera poses look correct

## Technical Details

### Static Cameras (ZED)
For static cameras, the extrinsic is directly read from `config/camera_extrinsics_base.yaml`:
```
T_base_cam = config[cam_id]
```

### Eye-in-Hand Cameras (RealSense)
RealSense cameras are mounted on the robot end-effector, so their pose depends on:
- Camera-to-flange offset: `config/camera_extrinsics_realsense.yaml`
- Flange pose at that frame: extracted from the captured `frame_info.yaml`

Composition:
```
T_base_cam = T_base_flange(frame) @ T_flange_cam(config)
```

The flange pose is **extracted** from the captured T_base_cam using the relationship:
```
T_base_flange = T_base_cam_captured @ inv(T_flange_cam_original)
```

This allows tuning the camera-to-flange offset while preserving the flange pose from the original capture.

## Config File Format

### camera_extrinsics_base.yaml
```yaml
zed2i_1:
  R:
  - 0.004403812204537894
  - -0.5179704072767931
  - 0.8553872009935499
  - -0.9990571041368691
  # ... 9 floats total, row-major 3×3 matrix
  t:
  - 0.0645038744163499
  - -0.3624129187844245
  - 0.24622335804524623
```

### camera_extrinsics_realsense.yaml
```yaml
realsense_1:
  R: [...]  # camera-to-flange offset
  t: [...]
realsense_2:
  R: [...]  # camera-to-flange offset
  t: [...]
zed2i_1:
  R: [...]  # camera-to-base (same as camera_extrinsics_base.yaml)
  t: [...]
```

## Coordinate Frame Convention

All R/t matrices follow:
```
p_dst = R @ p_cam + t
```

Where:
- For `camera_extrinsics_base.yaml` and ZED in `camera_extrinsics_realsense.yaml`:
  - `p_cam` = point in camera optical frame
  - `p_dst` = point in robot base frame (lbr_link_0)

- For RealSense in `camera_extrinsics_realsense.yaml`:
  - `p_cam` = point in camera optical frame
  - `p_dst` = point in robot flange frame (lbr_link_ee)

## Limitations

1. **Pointcloud files unchanged**: The point cloud PLY files in `outputs/bagviz/<run>/*/frame_*/` were already transformed to base frame during capture. Only the **camera triads** (visualization axes) update when you use `--use-config-extrinsics`.

2. **RealSense flange pose fixed**: The flange pose for RealSense cameras is locked to what was captured. If you need to tune both camera extrinsics AND flange poses, re-run the capture pipeline.

3. **Requires io_extrinsics module**: The config-loading path depends on `src.calibration.io_extrinsics`. If imports fail, falls back to frame_info.yaml.

## Troubleshooting

### "io_extrinsics module not available"
Make sure you can import the calibration module:
```bash
python -c "from src.calibration.io_extrinsics import load_extrinsics_yaml; print('OK')"
```

If that fails, check that `src/calibration/io_extrinsics.py` exists and `src/` is on the Python path.

### Camera triad doesn't move
- Make sure `--use-config-extrinsics` is passed
- Check that the YAML files are being read (verify paths with `--extrinsics-yaml` and `--extrinsics-realsense-yaml`)
- Confirm the camera ID matches (zed2i_1, realsense_1, realsense_2)

### "Config files not found"
Pass explicit paths:
```bash
python -m tools.bagviz.view_pointclouds \
  --run-dir outputs/bagviz/<run> \
  --use-config-extrinsics \
  --extrinsics-yaml /full/path/to/camera_extrinsics_base.yaml \
  --extrinsics-realsense-yaml /full/path/to/camera_extrinsics_realsense.yaml
```

## Related Files

- `tools/bagviz/view_pointclouds.py` — visualization viewer (this file)
- `tools/bagviz/capture_pipeline_snapshots.py` — capture pipeline (still needed for initial bagviz snapshot)
- `config/camera_extrinsics_base.yaml` — ZED extrinsics
- `config/camera_extrinsics_realsense.yaml` — RealSense extrinsics
- `src/calibration/io_extrinsics.py` — YAML loading/saving utilities
