# RealSense (D405) Intrinsics — Where They Come From

This document traces how the RGB/depth camera intrinsics used by
`scripts/launch_pipeline_realsense.sh` (via `run_pipeline_track_multicam_realsense.py`)
are actually determined, end to end: driver → topic → pipeline. There is **no
hardcoded/static intrinsics matrix anywhere in this repo** for the RealSense
cameras — every K is read live off a ROS `sensor_msgs/CameraInfo` topic on
every frame.

## 1. The chain, camera → pipeline

```
D405 on-chip factory calibration (per-resolution table, in camera firmware)
   │  read via librealsense: rs2::video_stream_profile::get_intrinsics()
   ▼
realsense2_camera driver node (ros2 launch realsense2_camera rs_launch.py,
included from mv_launch's zed_realsense_trio.launch.py)
   │  BaseRealSenseNode::updateStreamCalibData() copies fx/fy/ppx/ppy straight
   │  into a sensor_msgs/CameraInfo message (K and P), per stream, per resolution.
   ▼
/realsense_1/camera/color/camera_info            (raw color intrinsics)
/realsense_1/camera/aligned_depth_to_color/camera_info  (== color intrinsics,
                                                          since depth is
                                                          reprojected onto the
                                                          color pixel grid)
   │  (same for realsense_2)
   ▼
image_proc::RectifyNode (started per-camera in zed_realsense_trio.launch.py)
   rectifies camera/color/image_raw → camera/color/image_rect
   rectifies camera/aligned_depth_to_color/image_raw → .../image_rect
   ▼
multicam_grabber_realsense.py (MultiCamGrabberRealsense)
   subscribes rgb_info_topic / info_topic (the *raw*, non-rectified
   camera_info — see §3) and stores K via _K_from_camerainfo():
       self._K_rgb[cam_id]   = np.array(msg.k).reshape(3, 3)
       self._K_depth[cam_id] = np.array(msg.k).reshape(3, 3)
   ▼
View.K passed into SAM/DINO/FoundationPose/ICP for that frame
```

Key point: **the principal point, focal length, and distortion coefficients
are never computed or set by this repo.** They come straight from the D405's
own factory calibration, retrieved by librealsense for whatever stream
profile (width × height × fps × format) the driver actually opened, and
published verbatim into `CameraInfo.k` / `.p` by
`BaseRealSenseNode::updateStreamCalibData()` in
`realsense-ros/realsense2_camera/src/base_realsense_node.cpp:786-822`
(`~/franka_ros2_ws/src/realsense-ros/`).

## 2. What decides *which* intrinsics (resolution → different principal point)

The D405 stores a **separate calibration table entry per supported stream
resolution/format** in its firmware. Which entry you get is entirely decided
by which stream profile the driver opens — controlled by the
`rgb_camera.color_profile` / `depth_module.depth_profile` /
`depth_module.color_profile` launch parameters of `realsense2_camera`
(declared in `rs_launch.py`, default `'0,0,0'` = "no explicit width/height/fps
requested").

**`zed_realsense_trio.launch.py`** (`~/franka_ros2_ws/src/mv_launch/launch/`)
— the file `launch_host_realsense.sh` actually runs — does **not** pass any
of those profile parameters for either `realsense_1` or `realsense_2`:

```python
realsense_1 = IncludeLaunchDescription(
    PythonLaunchDescriptionSource(rs_launch),
    launch_arguments={
        "camera_namespace": "realsense_1",
        "camera_name": "camera",
        "serial_no": ["'", rs1_serial, "'"],
        "enable_color": "true",
        "enable_depth": "true",
        "align_depth.enable": "true",
        "publish_tf": "true",
        "config_file": info_qos_override,   # only overrides *_info QoS, see below
    }.items(),
)
```

With no profile requested, `realsense2_camera`'s `ProfileManager::getDefaultProfiles()`
(`realsense2_camera/src/profile_manager.cpp:121-141`) falls back to whatever
profile **librealsense itself flags as `is_default()`** for that stream on
that specific device/firmware — i.e. the SDK's own default, not a value this
repo chooses. **This is the setting that decides which intrinsics you get** —
change `rgb_camera.color_profile:=WxH,FPS` (and matching `depth_module.*`) on
the `zed_realsense_trio.launch.py` command line and the driver opens a
different calibration table entry, with a different `fx`/`fy`/`ppx`/`ppy`
(not just a linear rescale — it's whatever the firmware has stored for that
exact profile).

**Empirical evidence of the currently-active default:** a bag captured
2026-08-17 (`outputs/bagviz/rosbag_wo_fixture_20260817T140050Z/realsense_1/`)
has `depth_m.npy` shape `(480, 848)` and `rgb_native.png` size `848×480` —
i.e. the live default profile is currently **848×480**, not 640×480 (see the
discrepancy note in §4).

## 3. Depth alignment and rectification (why `rgb_info_topic` reads the *raw* topic)

- `align_depth.enable:=true` reprojects the depth stream onto the color
  sensor's pixel grid, so `.../aligned_depth_to_color/camera_info` carries
  the **same intrinsics as color**, not the physical depth sensor's own.
- The D405 color sensor has real (non-zero) Brown-Conrady distortion, unlike
  the ZED (rectified in-SDK before publishing). `zed_realsense_trio.launch.py`
  therefore runs an `image_proc::RectifyNode` per camera to produce
  `camera/color/image_rect` and `camera/aligned_depth_to_color/image_rect`.
- `ALL_CAMERAS` in
  `src/perception/ros/learn_runners/run_pipeline_track_multicam_realsense.py:283-315`
  deliberately points `rgb_topic`/`depth_topic` at the **rectified**
  `.../image_rect` images but `rgb_info_topic`/`info_topic` at the
  **unrectified** `camera/color/camera_info` /
  `camera/aligned_depth_to_color/camera_info` topics. This is intentional,
  not a bug: for a monocular stream with `Tx=Ty=0`, `image_proc::RectifyNode`
  itself rectifies using `P` from the raw `CameraInfo` (`P == K` here), so the
  raw topic's `K` is already the correct intrinsics matrix for the rectified
  image. There is no separate "rectified camera_info" topic to subscribe to.

## 4. Known discrepancy: `docs/camera_bandwidth_optimization.md` / `config/realsense_camera_bandwidth.yaml`

`docs/camera_bandwidth_optimization.md` and
`config/realsense_camera_bandwidth.yaml` describe a 640×480 "bandwidth
optimized" RealSense configuration. **That yaml is not referenced by any
launch file, script, or Python module in this repo or in
`~/franka_ros2_ws/src/mv_launch`** (confirmed by grep) — it is a standalone
file that was never wired into `zed_realsense_trio.launch.py`. The actually
running pipeline uses the unmodified SDK-default profile (§2), which the
2026-08-17 bag capture shows to be 848×480, not 640×480. Treat that doc's
resolution/bandwidth numbers as aspirational/stale, not as what's live.

## 5. How to read the live numeric intrinsics yourself

Nothing in this repo persists the numeric K anywhere (checked
`outputs/calibration_debug/`, bagviz `frame_info.yaml` dumps, and
`config/*.yaml` — none of them store RealSense fx/fy/ppx/ppy). To get the
actual live numbers, with the host stack running
(`scripts/launch_host_realsense.sh`) or inside the container:

```bash
ros2 topic echo /realsense_1/camera/color/camera_info --once
ros2 topic echo /realsense_1/camera/aligned_depth_to_color/camera_info --once
# or, without ROS running, straight from librealsense:
rs-enumerate-devices -c
```

`CameraInfo.k` is row-major `[fx, 0, ppx, 0, fy, ppy, 0, 0, 1]`; that's
exactly what `_K_from_camerainfo()` in `multicam_grabber_realsense.py:29-30`
reshapes into the 3×3 `K` consumed by the rest of the pipeline.
