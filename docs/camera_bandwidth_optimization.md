# Camera Bandwidth Optimization

This document explains the bandwidth-saving optimizations implemented in the codebase.

## Camera Configuration Summary

### ZED 2i (Model: zed2i, Serial: 38580376)
- **Depth Mode:** NEURAL (AI-based, not stereo)
- **Frame Rate:** 30 FPS
- **Resolution:** 1080p (HD1080)
- **Status:** Automatically launched with RealSense cameras via `zed_realsense_trio.launch.py`

### RealSense Cameras
- **Models:** D405 (wrist-mounted on both arms)
- **Serials:** realsense_1=260322275185, realsense_2=260522275434
- **Frame Rate:** 30 FPS (matched to ZED)
- **Resolution:** 640×480 (QVGA)
- **Status:** Automatically launched with ZED via `zed_realsense_trio.launch.py`

## Changes Made

### 1. QoS Profile Depth Reduction (Implemented)

**File:** `src/perception/ros/qos_profiles.py`

All ROS image/camera subscriptions now use `qos_profile_sensor_data_low_latency` instead of the standard `qos_profile_sensor_data`:

- **Old depth:** 5 frames buffered (default `qos_profile_sensor_data`)
- **New depth:** 1 frame buffered (`qos_profile_sensor_data_low_latency`)

**Files updated:**
- `src/perception/ros/multicam_grabber.py`
- `src/perception/ros/multicam_grabber_realsense.py`
- `src/calibration/capture_flange_poses_dual.py`
- `src/calibration/capture_flange_poses_dual_handguided.py`
- `src/calibration/capture_flange_poses_dual_admittance.py`
- `src/calibration/autocalibrate_dual_realsense.py`
- `src/calibration/handeye_flange_cam_realsense.py`
- `src/calibration/board_pose_from_flange_realsense.py`

**Impact:** ~80% reduction in ROS middleware buffering overhead. Reduces memory usage and network stack load.

### 2. Camera Resolution & Frame Rate Configuration (Manual Setup)

Two configuration files have been created to override camera defaults:

#### ZED 2i Configuration
**File:** `config/zed_camera_bandwidth.yaml` (or directly in `/home/pdzuser/zed_ros2_ws/src/mv_launch/config/zed_override_native.yaml`)

Default settings:
- **Resolution:** 1080p (HD1080) for maximum detail
- **Frame rate:** 30 FPS (real-time with neural depth)
- **Depth mode:** NEURAL (AI-based depth estimation, not stereo)
  - Lower latency than stereo depth
  - Better handling of occlusions and textureless areas
  - More robust to lighting variations
- **Confidence threshold:** 50 (valid depth only above this score)

**Usage:**
```bash
# Launch ZED camera with bandwidth-optimized parameters
ros2 launch ros2_zed zed_camera.launch.py params_file:=$(pwd)/config/zed_camera_bandwidth.yaml
```

#### RealSense Configuration
**File:** `config/realsense_camera_bandwidth.yaml`

Default settings:
- **Resolution:** 640×480 (QVGA)
- **Frame rate:** 30 FPS (synchronized with ZED neural depth)
- **Depth filtering:** Disabled (preserves quality at 640×480 resolution)
- **IMU streams:** Disabled (accel/gyro not used in pipeline)
- **Auto-exposure:** Enabled for stable image capture

**Usage:**
```bash
# Launch RealSense camera with bandwidth-optimized parameters
ros2 launch realsense2_camera rs_launch.py param_file:=$(pwd)/config/realsense_camera_bandwidth.yaml
```

## Bandwidth Impact

### RealSense Optimization
| Factor | Reduction |
|--------|-----------|
| Resolution (640×480 vs 1280×720) | ~75% |
| Frame rate | No reduction (30 FPS for sync) |
| **Combined RealSense video** | **~75%** |
| QoS buffering overhead | ~80% |

### Example Numbers (per RealSense D405)
- **Before:** 1280×720 RGB @ 30 FPS ≈ 27 Mbps + depth
- **After:** 640×480 RGB @ 30 FPS ≈ 6.8 Mbps (uncompressed)
- With JPEG compression: ~1-2 Mbps per camera

### ZED 2i (Neural vs Stereo)
- **Neural depth mode:** Lower latency, better occlusion handling (no FPS penalty)
- **1080p @ 30 FPS:** Full resolution maintained for detail
- **Combined 3 cameras (1 ZED + 2 RealSense):**
  - Uncompressed: ~41 Mbps (27 Mbps ZED + 6.8 Mbps × 2 RealSense)
  - With JPEG: ~30-40 Mbps (depends on quality)

## Additional Bandwidth-Saving Options

### 3. Image Compression (via `image_transport`)

Can be applied at the ROS middleware level without modifying camera drivers:

```python
# In your grabber/pipeline node
from image_transport import create_subscription, CameraSubscriber

# Instead of raw sensor_msgs/Image, use image_transport's compressed plugin
# Automatically applies JPEG compression (configurable quality)
```

**Impact:** 80-95% reduction in transmitted bytes (JPEG quality-dependent)

### 4. Frequency Filtering / Frame Skipping

Reduce processed frames without reducing network load:

```python
# Skip every Nth frame
frame_counter = 0
def on_image(msg):
    global frame_counter
    if frame_counter % 2 == 0:  # Process every 2nd frame (effective 7.5 FPS from 15 FPS capture)
        process_image(msg)
    frame_counter += 1
```

### 5. Region of Interest (ROI)

Only stream/process relevant parts of the image:

```python
# Crop depth/RGB to a smaller region before transmission
def crop_to_roi(image, x, y, w, h):
    return image[y:y+h, x:x+w]
```

### 6. Depth-Only or RGB-Only Streams

If full RGB+depth is not needed:

```yaml
# In camera config, disable RGB if only depth is needed
enable_rgb: false
enable_depth: true
```

**Impact:** 50% reduction if one stream is disabled

## How to Verify Bandwidth Impact

### On the ROS side

```bash
# Monitor image topic bandwidth
ros2 topic hz /camera/rgb/image_raw  # Check frame rate
ros2 topic bw /camera/rgb/image_raw  # Check bytes/sec (if plugin available)
```

### Network-level monitoring

```bash
# Ubuntu/Linux
iftop -i eth0  # Real-time bandwidth monitor

# Docker container
docker stats <container>  # Monitor network I/O
```

### ROS QoS monitoring

```bash
# List all ROS topics with their QoS settings
ros2 topic list -v
```

## Launch Configuration

### Default Setup (Recommended)
The standard host launch script already includes all cameras:

```bash
scripts/launch_host_realsense.sh
```

This launches:
- **1 × ZED 2i** (zed2i_1) — tripod-mounted, static extrinsics
  - 1080p @ 30 FPS with neural depth
  - Uses `/home/pdzuser/zed_ros2_ws/src/mv_launch/config/zed_override_native.yaml`
  
- **2 × RealSense D405** (realsense_1, realsense_2) — wrist-mounted, dynamic extrinsics
  - 640×480 @ 30 FPS via default realsense2_camera params
  - Automatically rectified for alignment

All three cameras are synchronized at 30 FPS by the pipeline.

## Testing the Changes

The QoS depth reduction is automatically active for all calibration and pipeline scripts. To test with alternative configurations:

```bash
# Terminal 1: Launch hardware (or mock)
ros2 launch lbr_dual_arm_bringup hardware.launch.py  # or mock.launch.py

# Terminal 2: Start ZED camera with optimized params
ros2 launch ros2_zed zed_camera.launch.py params_file:=$(pwd)/config/zed_camera_bandwidth.yaml

# Terminal 3: Start RealSense cameras with optimized params
ros2 launch realsense2_camera rs_launch.py param_file:=$(pwd)/config/realsense_camera_bandwidth.yaml

# Terminal 4: Run pipeline/calibration as usual
python3 -m src.perception.ros.learn_runners.run_pipeline_track_multicam_realsense --help
```

## Notes

- The configuration files assume ROS 2 with `ros2_zed` and `realsense-ros` packages installed
- Parameter names vary by driver version; verify against your installed driver's documentation
- 15 FPS is suitable for hand-eye calibration (offline, not real-time). For live tracking, use higher FPS
- Depth filtering should be disabled during calibration to avoid biasing the solution
