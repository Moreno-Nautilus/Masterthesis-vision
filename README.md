# Masterthesis — Vision

Multi-camera 6-DoF object pose estimation and tracking for robotic manipulation.
The default rig — one static ZED 2i plus two end-effector-mounted Intel
RealSense D405 cameras — feeds a learned perception stack —
**Grounding-DINO → SAM2 → DINOv2 → cross-camera fusion → FoundationPose → ICP** —
that publishes a canonical pose per object in the **robot base frame**. The
original 3-ZED-only rig is still supported as a variant — see
[§6](#6-3-zed-trio-variant-original).

> **Brand-new machine, nothing installed yet?** Start with
> **[docs/setup_and_docker.md](docs/setup_and_docker.md)** — the complete
> from-scratch walkthrough: OS/GPU/ROS 2 prerequisites, camera SDKs, the host
> ROS 2 workspace (`mv_launch`/`fp_debug_msgs`), and Docker (image, container,
> in-container build) — nothing to run yet after this, just a machine that's
> ready to.
>
> **Rig already set up / just want to calibrate and run it?** Start with
> **[docs/getting_started_realsense.md](docs/getting_started_realsense.md)** —
> a linear, tested end-to-end calibrate → run → view guide for the default
> 1-ZED + 2-RealSense rig, or its condensed
> **[calibration cheat sheet](docs/calibration_cheatsheet.md)**.
>
> **On the original 3-ZED-only rig instead?** See
> **[docs/getting_started.md](docs/getting_started.md)** and its
> **[cheat sheet](docs/cheatsheet.md)** — [§6 below](#6-3-zed-trio-variant-original)
> has the condensed launch commands.
>
> **For a step-by-step description of how the pipeline actually runs,
> read [docs/pipeline_walkthrough.md](docs/pipeline_walkthrough.md).** This README
> covers setup, how to launch things, and what each piece of code does; the
> walkthrough explains the algorithm itself.
>
> **Just want to view tracked objects / camera poses in RViz?** See
> **[docs/visualization.md](docs/visualization.md)** for the MoveIt2 planning
> scene view and camera-extrinsics TF frames — the 2D per-camera overlays
> (Foxglove) are covered in
> [docs/getting_started.md §1](docs/getting_started.md#watch-the-result-in-foxglove)
> instead.
>
> **Have a rosbag and want to sanity-check the pipeline offline** (depth maps,
> DINO/SAM overlays, raw + segmented point clouds, detected object coordinate
> frames)? See **[docs/bagviz_quickstart.md](docs/bagviz_quickstart.md)** —
> no docker, no live pipeline, just `scripts/visualize_bag_pipeline.sh <bag>`.
>
> **Looking for anything else** (compliant control, hand-eye calibration,
> annotating the DINO reference bank, ...)? Every topic has its own guide
> under **[docs/](docs/)** — check there before digging through the code.

---

## 1. Repository layout

| Path | What it is |
|------|------------|
| [scripts/](scripts/) | Launch scripts (host stack and pipeline) — see §3 |
| [src/perception/ros/learn_runners/run_pipeline_track_multicam.py](src/perception/ros/learn_runners/run_pipeline_track_multicam.py) | **Main pipeline node.** Grabs synced views, runs the full detect→fuse→pose loop, publishes base-frame poses. This is what the launch files start. |
| [src/perception/ros/learn_runners/visualize_pipeline.py](src/perception/ros/learn_runners/visualize_pipeline.py) | Standalone RViz/Foxglove visualizer node — subscribes to the pipeline's debug topics and draws SAM/DINO/pose overlays per camera. |
| [src/perception/ros/multicam_grabber.py](src/perception/ros/multicam_grabber.py) | ROS subscriptions; returns time-synchronized `View`s (rgb + depth + `K`) per camera. |
| [src/perception/view.py](src/perception/view.py) | The `View` dataclass (one camera's rgb/depth/intrinsics/extrinsics for a frame). |
| [src/perception/multicam_fusion.py](src/perception/multicam_fusion.py) | Cross-camera fusion: back-projects masks to the base frame, clusters detections across cameras, arbitrates the class label by DINO-score voting. |
| [src/perception/fused_multicam_helpers.py](src/perception/fused_multicam_helpers.py) | Point-cloud helpers — lift masked depth to base frame, merge per-cam clouds, mesh→point-cloud cache, Chamfer/weighted-pose utilities. |
| [src/perception/learned/GDINO/grounding_dino_proposal.py](src/perception/learned/GDINO/grounding_dino_proposal.py) | Grounding-DINO box proposer (text prompt → boxes). |
| [src/perception/learned/SAM/sam_segmentation.py](src/perception/learned/SAM/sam_segmentation.py) | SAM2 wrapper — segments the GDINO boxes into masks, with bf16/fp32 collapse recovery. |
| [src/perception/learned/DINO/dino_identifier.py](src/perception/learned/DINO/dino_identifier.py) | DINOv2 MUSE identifier — embeds crops and classifies them against the reference bank. |
| [src/perception/learned/FP/pose_foundation.py](src/perception/learned/FP/pose_foundation.py) | FoundationPose wrapper (one estimator, one nvdiffrast CUDA context, GPU worker thread). |
| [src/perception/tracking/realtime_tracker.py](src/perception/tracking/realtime_tracker.py) | Per-object tracker state (pose + ICP refinement) used in `track` mode. In the multicam loop the runner drives one shared Cutie session per camera and feeds each object's mask into its own `RealtimeTracker`; the canonical pose comes from the fused multi-camera ICP. |
| [src/perception/tracking/cutie_tracker.py](src/perception/tracking/cutie_tracker.py) | Cutie (video object segmentation) wrapper. |
| [src/perception/tracking/icp_refiner.py](src/perception/tracking/icp_refiner.py) | Generic ICP refiner used by `RealtimeTracker`; the fused base-frame ICP and rotation grid live in the multicam runner/helpers. |
| [src/calibration/base_to_cams_calib_3.py](src/calibration/base_to_cams_calib_3.py) | **N-camera extrinsic calibration** (checkerboard → base frame) — see §6.2. Defaults to the 3-ZED trio; accepts `--cam-ids` for any subset (e.g. the RealSense-trio rig's single `zed2i_1`). |
| [src/calibration/io_extrinsics.py](src/calibration/io_extrinsics.py) | Load/save extrinsics YAML (`R` row-major + `t`) ↔ `SE3`. |
| [src/calibration/capture_handeye_data.py](src/calibration/capture_handeye_data.py) | Dual-arm calibration, Stage A (manual or replay): position each arm via `--controller {moveit,admittance,handguided}`, save its flange pose/joint config and checkerboard image/detection — see §4. |
| [src/calibration/calibrate_handeye.py](src/calibration/calibrate_handeye.py) | Stage B (offline, no hardware): solves `T_flange_cam` from Stage A's saved data, `--method direct` or `--method joint` — see §4. |
| [src/calibration/autocalibrate_dual_realsense.py](src/calibration/autocalibrate_dual_realsense.py) | Dual-arm calibration, Stage C (automatic replay): solves checkerboard pose + ZED extrinsic, run after Stage A/B — see §4. |
| [src/calibration/moveit_dual_arm.py](src/calibration/moveit_dual_arm.py) | `MoveGroup` action-client helper — the only code in this repo that sends motion commands to the robot (used by the Step 2 script above); Cartesian (`ArmTarget`/`move_to()`) and joint-space (`JointTarget`/`move_to_joint()`) goal types. |
| [src/calibration/flange_pose_store.py](src/calibration/flange_pose_store.py) | JSON schema + save/load for the permanently-stored flange pose captures. |
| [src/calibration/calibration_log.py](src/calibration/calibration_log.py) | Append-only JSON run logs (camera/checkerboard/flange transforms + quality metrics) under `outputs/calibration_logs/`. |
| [src/utils/se3.py](src/utils/se3.py) | Minimal immutable `SE3` rigid-transform type. |
| [tools/generate_dino_reference_renders.py](tools/generate_dino_reference_renders.py) | Renders synthetic reference views from the CAD meshes (optional DINO reference source). |
| [tools/refbank_crop_screenshots.py](tools/refbank_crop_screenshots.py) | Interactively crop raw screenshots into `Data/ZED_screens/` (manual bbox, one `cv2.selectROI` window per image) — see [docs/annotate_refbank.md](docs/annotate_refbank.md). |
| [tools/refbank_autocrop_masks.py](tools/refbank_autocrop_masks.py) | Auto-crop imgpy render sessions (image + segmentation mask) into `Data/ZED_screens/`, no GUI — see [docs/annotate_refbank.md](docs/annotate_refbank.md). |
| [debug_pose_axes.py](debug_pose_axes.py) | Publishes RViz/Foxglove axis markers for the poses on `/perception/fp/pose_base/...`. |
| [config/](config/) | Calibration inputs/outputs (board pose, camera extrinsics). |
| [external/](external/) | Third-party deps as submodules + the FoundationPose patch — see §2. |
| [Data/](Data/) | **Not in git** — CAD models + reference crops. You must create it, see §2.3. |

---

## 2. Setup

### 2.1 Third-party code (`external/`)

The heavy models are pinned as submodules, not vendored. See
[external/README.md](external/README.md) for details. In short:

```bash
git submodule update --init --recursive
bash external/apply_patches.sh        # applies the FoundationPose thesis patch (idempotent)
```

**Cutie** has no public upstream and is git-ignored, so it is not pulled by the
submodule update. Clone it separately into `external/Cutie` and check out the
pinned commit `ec5cdd4cf16f75c73ad785a2f96fb97dbad4125a` (see
[external/README.md](external/README.md)).

### 2.2 Docker

The pipeline runs inside a CUDA container built from
[Dockerfile.thesisnewcuda](Dockerfile.thesisnewcuda) (Python packages pinned in
[requirements-docker.txt](requirements-docker.txt)). The launch scripts assume
a container named `vision` already exists (they `docker start`/`stop` it, they
do not build it). Override the name with the `CONTAINER` env var.

**Setting up a fresh machine from scratch** (Docker image + container, GPU
passthrough, the host ROS 2 workspace that `mv_launch`/`fp_debug_msgs` live
in, camera SDKs)? See **[docs/setup_and_docker.md](docs/setup_and_docker.md)**
for the full walkthrough — this section only covers the submodules/`Data/`
half of one-time setup.

### 2.3 The `Data/` folder (you must create this)

`Data/` is **git-ignored**, so cloning this repo does **not** give you the meshes
or the reference images. Create it with this layout before running anything:

```
Data/
├── CAD_Models/              # raw object meshes (.obj or .stl)
├── CAD_Models_centered/     # origin-centered meshes — USED BY THE PIPELINE (--cad-dir)
│   ├── cooling_manifold/    # one subfolder per assembly
│   │   ├── cooling_base.obj
│   │   ├── cooling_f.obj
│   │   └── cooling_screw.obj
│   └── plumbers_block/
│       ├── pb_base.obj
│       ├── pb_pipe.obj
│       ├── pb_screw.obj
│       └── pb_top.obj
├── ZED_screens/             # REAL reference crops, one folder per object (--reference-dir)
│   ├── cooling_manifold/
│   │   ├── cooling_base/ *.png
│   │   ├── cooling_f/    *.png
│   │   └── cooling_screw/ *.png
│   ├── plumbers_block/
│   │   ├── pb_base/ *.png
│   │   ├── pb_pipe/ *.png
│   │   ├── pb_screw/ *.png
│   │   └── pb_top/  *.png
│   └── blue_cube/ *.png   # objects with no assembly stay directly under the root
└── reference_renders/       # OPTIONAL synthetic renders (--reference-renders-dir)
    ├── cooling_manifold/<object_id>/ *.png
    └── plumbers_block/<object_id>/ *.png
```

- **`CAD_Models_centered/<assembly_name>/<object_id>.obj` or `.stl`** — the mesh
  FoundationPose registers against. The default `--cad-dir` points here; the
  `object_id` is the filename stem. Meshes belonging to a known assembly
  (`cooling_manifold`, `plumbers_block`) live in that assembly's subfolder;
  objects with no assembly (cubes, screwdrivers) may sit directly under
  `CAD_Models_centered/`. Meshes are assumed to be in centimeters
  (`--mesh-scale 0.01`).
- **`ZED_screens/<assembly_name>/<object_id>/`** — the **DINO reference bank**:
  a handful of cropped photos of each object. DINOv2 embeds these once at
  startup and every candidate crop is classified against them. This is the
  default reference source (`--reference-source real`). Populate it with
  [tools/refbank_crop_screenshots.py](tools/refbank_crop_screenshots.py)
  (manual bbox from raw screenshots) and/or
  [tools/refbank_autocrop_masks.py](tools/refbank_autocrop_masks.py)
  (auto bbox from imgpy render + mask pairs) — see
  [docs/annotate_refbank.md](docs/annotate_refbank.md) for the full guide.
- **`reference_renders/<assembly_name>/<object_id>/`** — optional CAD-rendered
  alternative/extra reference views, produced by
  [tools/generate_dino_reference_renders.py](tools/generate_dino_reference_renders.py).
  Used when `--reference-source renders` or `both`.

The folder names under `ZED_screens/` / `reference_renders/` and the mesh
filenames must use the **same `object_id`** so labels line up across detection
and pose. The assembly-name grouping (`cooling_manifold`, `plumbers_block`) is
optional structure for organizing parts on disk — objects without a known
assembly prefix are read directly from the `Data/*` root instead.

---

## 3. Running the pipeline

There are two halves: the **host stack** (cameras + visualization, runs on the
host) and the **pipeline node** (runs inside the docker container). This
section covers the default rig — 1 static ZED 2i (`zed2i_1`) + 2
end-effector-mounted RealSense D405 cameras. For the full tested start-to-finish
sequence (including the required base→flange tf2 step before the cameras can
be marked "ready"), see
**[docs/getting_started_realsense.md §1](docs/getting_started_realsense.md#1-run-it-start-to-finish-the-tested-sequence)**.

### 3.0 Prerequisite — base→flange transform in tf2

The pipeline needs `lbr_link_0 → lbr_link_ee` resolvable via tf2 before the
RealSense cameras can be marked "ready" (their extrinsic is computed live
every frame from this):

```bash
# real robot (needs the KUKA FRI connection already live):
ros2 launch lbr_bringup hardware.launch.py model:=iiwa7

# or, with no robot connected, a fixed identity placeholder (fused RealSense
# poses will be wrong, but camera sync/detection/fusion all still run):
ros2 run tf2_ros static_transform_publisher \
    --frame-id lbr_link_0 --child-frame-id lbr_link_ee \
    --x 0 --y 0 --z 0 --qx 0 --qy 0 --qz 0 --qw 1
```

### 3.1 Host stack — cameras + viz ([scripts/launch_host_realsense.sh](scripts/launch_host_realsense.sh))

Starts a tmux session with `zed2i_1`, both RealSense D405s, and
`flange_pose_publisher`; the Foxglove bridge and per-camera `visualize_pipeline`
windows come up by default so the mask + tracked-axes overlays show in Foxglove.

```bash
scripts/launch_host_realsense.sh                  # start (and attach) the tmux session
scripts/launch_host_realsense.sh attach           # re-attach if already running
scripts/launch_host_realsense.sh stop             # kill the session
scripts/launch_host_realsense.sh --no-visualize   # skip the 3 visualize_pipeline windows
scripts/launch_host_realsense.sh --no-foxglove    # skip the foxglove_bridge window
```

tmux: `Ctrl+b` then a number to switch windows, `Ctrl+b d` to detach.

### 3.2 Pipeline node — the locked baseline ([scripts/launch_pipeline_realsense.sh](scripts/launch_pipeline_realsense.sh))

Restarts the docker container, sources ROS + the venv, and runs
`run_pipeline_track_multicam_realsense`:

```bash
scripts/launch_pipeline_realsense.sh init-only       # locked trio init baseline
scripts/launch_pipeline_realsense.sh fast-track      # fast tracking preset
scripts/launch_pipeline_realsense.sh accurate-track  # settled-axis accuracy preset
scripts/launch_pipeline_realsense.sh baseline        # alias for init-only
scripts/launch_pipeline_realsense.sh                 # interactive shell in the container
scripts/launch_pipeline_realsense.sh --run-mode track --num-cameras 3 ...  # custom args
CONTAINER=other-name scripts/launch_pipeline_realsense.sh fast-track       # different container
```

`init-only`/`baseline` runs in `init_only` mode (re-detect every tick, never
tracks) and tees output to `outputs/logs/multicam_init_realsense_baseline.log`.
`fast-track` keeps tracking responsive with centroid recovery and no rotation
reseed/PCA/damping. `accurate-track` adds the rotation reseed + cautious PCA +
light damping preset for better settled screw-axis estimates. The exact pinned
flags are listed in the launch file and the init-only baseline is explained in
[docs/pipeline_walkthrough.md](docs/pipeline_walkthrough.md). Named presets
also launch a `cam-scene` tmux window (`publish_camera_scene_objects.py
--dual-arm`) that publishes the ZED camera + holder meshes as
`CollisionObject`s alongside the tracked parts — see §7.

### Output

Per detected object the pipeline publishes a base-frame `fp_debug_msgs/DebugPoseItem`
(identified by `assembly_name`/`part_id`) on the shared
`/perception/fp/pose_base/fused/assembly` topic; with logging flags, it writes
CSV rows (`init_pose_log.csv`, `outputs/logs/...csv`) and saves a render under
`init_renders/`. Same message/topic shape on the [3-ZED variant](#6-3-zed-trio-variant-original).

---

## 4. Hand-eye / camera calibration (default rig)

Camera-to-flange offset for both RealSense cameras, plus the
checkerboard-in-base-frame and ZED extrinsic, are solved by a **three-stage
dual-arm routine** — see
**[docs/getting_started_realsense.md §4](docs/getting_started_realsense.md#4-hand-eye-calibration-camera-to-flange-offset)**
for the full walkthrough, or
**[docs/calibration_cheatsheet.md](docs/calibration_cheatsheet.md)** for the
condensed command sequence:

```bash
# Stage A (manual, both arms by default) — position each arm, save its
# flange pose/joint config and a checkerboard capture; nothing solved yet
python3 -m src.calibration.capture_handeye_data

# Stage B (offline, no hardware) — solves T_flange_cam from Stage A's data
python3 -m src.calibration.calibrate_handeye --write

# Stage C (automatic replay) — drives both arms itself, solves checkerboard
# pose + ZED extrinsic
python3 -m src.calibration.autocalibrate_dual_realsense
```

Stage A's default `--controller moveit` needs jogging the arm interactively
via MoveIt between samples; see
**[docs/moveit_robot_control.md](docs/moveit_robot_control.md)** for that
part — or skip jogging entirely with `--controller admittance` or
`--controller handguided`, see
**[docs/calibration_control_modes.md](docs/calibration_control_modes.md)**.
Stage C needs no jogging — it drives both arms itself over the
`moveit_msgs/action/MoveGroup` action (see
[src/calibration/moveit_dual_arm.py](src/calibration/moveit_dual_arm.py)),
including one simultaneous `both_arms_flange` goal per pose-pair for the
hand-eye stage (calibration always targets the bare flange, regardless of
whether the Y-gripper is attached — see `moveit_dual_arm.py`'s docstring).

The original single-arm, single-camera manual scripts
(`handeye_flange_cam_realsense.py`, `board_pose_from_flange_realsense.py`)
still work standalone — see
[docs/getting_started_realsense.md §4.7](docs/getting_started_realsense.md#47-manual-single-camera-fallback-original-scripts-still-available).

**On the 3-ZED variant instead?** Its 3-camera checkerboard extrinsic
calibration is different (single-stage, no hand-eye/flange offset involved) —
see **[§6.2 below](#62-3-zed-camera-to-base-calibration)**.

---

## 5. How the pipeline works

See **[docs/pipeline_walkthrough.md](docs/pipeline_walkthrough.md)** for the full
per-tick breakdown, in two halves:

- **Init** (Stages 0–4) — model loading, GDINO proposal, SAM segmentation, DINO
  classification + candidate selection, cross-camera fusion, per-object
  FoundationPose + ICP (incl. the conditional symmetry rotation grid and the
  polishing ICP).
- **Tracking** (Stage 5) — once objects are initialized, `track` mode drops the
  learned front-end and runs the cheap Cutie-mask → masked-depth → fused-ICP loop
  with its accept/hold/lost gates; this section also covers the `fast-track` vs
  `accurate-track` presets and the optional rotation fixes (rot-reseed, PCA
  shaft-axis, rotation damping).

---

## 6. 3-ZED trio variant (original)

The pipeline's original rig: three static, calibrated ZED 2i cameras
(`zed2i_1/2/3`), no end-effector-mounted cameras and no hand-eye calibration
step. A separate set of scripts/config/launch files that doesn't touch
anything in §3/§4 above — see
**[docs/getting_started.md](docs/getting_started.md)** for the full
student-facing calibrate → run → view guide, or the one-page
**[cheat sheet](docs/cheatsheet.md)**.

### 6.1 Running the pipeline (3-ZED)

Same two-halves split as §3 — **host stack** (cameras + viz, on the host) and
**pipeline node** (in the docker container).

**Host stack — cameras + viz** ([scripts/launch_host.sh](scripts/launch_host.sh)).
Starts a tmux session with one window each for: the ZED camera driver, the
Foxglove bridge, a `visualize_pipeline` per camera, and `debug_pose_axes`.

```bash
scripts/launch_host.sh            # start (and attach) the tmux session
scripts/launch_host.sh attach     # re-attach if already running
scripts/launch_host.sh stop       # kill the session
scripts/launch_host.sh calibrate  # same, but ZEDs grab/publish at HD2K/15fps (see §6.2)
```

tmux: `Ctrl+b` then `0..5` to switch windows, `Ctrl+b d` to detach.

**Pipeline node — the locked baseline** ([scripts/launch_pipeline.sh](scripts/launch_pipeline.sh)).
Restarts the docker container, sources ROS + the venv, and runs
`run_pipeline_track_multicam`:

```bash
scripts/launch_pipeline.sh init-only       # locked 3-camera init baseline
scripts/launch_pipeline.sh fast-track      # fast tracking preset
scripts/launch_pipeline.sh accurate-track  # settled-axis accuracy preset
scripts/launch_pipeline.sh baseline        # alias for init-only
scripts/launch_pipeline.sh                 # interactive shell in the container
scripts/launch_pipeline.sh --run-mode track --num-cameras 3 ...  # custom args
CONTAINER=other-name scripts/launch_pipeline.sh fast-track       # different container
```

Same `init-only`/`fast-track`/`accurate-track` preset semantics and output
format (`fp_debug_msgs/DebugPoseItem` on `/perception/fp/pose_base/fused/assembly`)
as §3 — see [docs/pipeline_walkthrough.md](docs/pipeline_walkthrough.md).

### 6.2 3-ZED camera-to-base calibration

The pipeline loads camera extrinsics from
[config/camera_extrinsics_base.yaml](config/camera_extrinsics_base.yaml) (`T_base_cam`
per camera). Regenerate this with the 3-camera checkerboard calibration whenever
the cameras move:

**1. Launch the cameras** so the ZED RGB/`camera_info` topics are publishing —
   easiest is the `cams` window of the host stack:

```bash
scripts/launch_host.sh          # window 0 runs the ZED driver
```

   For better checkerboard corner detection, launch with `calibrate` instead —
   this grabs/publishes at the ZED2i's HD2K resolution (2208x1242 @ 15fps)
   rather than the normal HD1080 @ 30fps used for tracking:

```bash
scripts/launch_host.sh calibrate    # same tmux session, cameras at HD2K/15fps
```

   Under the hood this passes `override_path:=config/zed_override_2k.yaml`
   (vs. the default `config/zed_override_native.yaml`) to
   `mv_launch`'s `zed2i_pair.launch.py` — both files live in the separate
   `~/franka_ros2_ws` ROS workspace, not this repo. Equivalent to setting
   `CALIBRATE_2K=1 scripts/launch_host.sh`.

   If a camera fails to open (`CAMERA NOT DETECTED` in the `cams` window —
   USB enumeration flakiness happens), calibrate with just the cameras that
   came up, e.g. `--cam-ids zed2i_2 zed2i_3` in step 3 below (see
   `base_to_cams_calib_3.py --help`; any subset of size N≥1 works).

**2. Set the board pose** in
   [config/base_board_pose.yaml](config/base_board_pose.yaml): the translation
   (m) and roll/pitch/yaw (deg) of the **checkerboard origin corner** in the
   robot base frame. For a flat board on the table start with `rpy = [0,180,0]`
   (you may need one sign flip). Also confirm the board geometry constants at the
   top of the script (`CHESS_COLS=8`, `CHESS_ROWS=11` inner corners,
   `SQUARE_SIZE_M=0.03`).

**3. Run the calibration** (it is a ROS node — run it where it can see the camera
   topics, i.e. inside the container shell or any sourced ROS env):

```bash
python3 -m src.calibration.base_to_cams_calib_3
```

Hold the checkerboard so **all three cameras** see it and keep it steady. The
script:
- rejects stale / unsynchronized / high-reprojection-error captures,
- collects `NUM_SAMPLES` (8) good 3-camera samples,
- per sample solves the board pose per camera (`solvePnP`) and computes
  `T_base_cam = T_base_board · T_cam_board⁻¹`,
- averages the samples, checks the translation/rotation spread is tight enough
  (else it raises and writes nothing),
- backs up the old YAML to `.yaml.bak` and writes the new
  `config/camera_extrinsics_base.yaml`,
- saves annotated corner images to `outputs/calibration_debug/` for inspection.

Check the printed mean reprojection error and the `T_cam1_cam2` / `T_cam1_cam3`
consistency before trusting the result.

---

## 7. MoveIt2 planning scene visualization

Both pipeline runners (`run_pipeline_track_multicam_realsense.py` and the
3-ZED variant, `run_pipeline_track_multicam.py`) publish each tracked part as
a `moveit_msgs/CollisionObject` on
`/planning_scene`, plus each camera's calibrated pose as a static TF frame —
so RViz can show the tracked parts and cameras alongside a mock robot without
any real hardware.

See **[docs/visualization.md](docs/visualization.md)** for the full guide:
what gets published and why, how to bring up the RViz scene view with
[scripts/view_scene.sh](scripts/view_scene.sh), and how to view/check the
camera extrinsics as TF frames.
If you need the other robot's frame, pass a different `--base-yaml`.

---

## 8. Compliant control (Cartesian impedance / admittance)

Two additive execution paths for the dual-arm rig, alongside the
position-controlled MoveGroup pipeline above — a torque-mode Cartesian
impedance controller with runtime-adjustable gains, and a software
admittance loop on the position interface. See
**[docs/compliant_control.md](docs/compliant_control.md)** for bring-up
commands, the client APIs
(`src/calibration/cartesian_impedance_dual_arm.py`,
`src/calibration/admittance_dual_arm.py`), named gain profiles, and known
caveats.

**New to these modes, or picking one for a calibration session?** Start
with **[docs/calibration_control_modes.md](docs/calibration_control_modes.md)**
instead — a walkthrough of all three dual-arm control modes (gravity
compensation, Cartesian impedance, admittance) side by side: what each
does, when to reach for it, and how each fits (or doesn't yet) into the
existing calibration routine.

**Just want to park both arms at a known pose** (start/end of a session)?
See **[docs/robot_init_pose_quickstart.md](docs/robot_init_pose_quickstart.md)**
— `scripts/launch_robots_to_init_pose.sh`, Cartesian-impedance (compliant)
by default, `CONTROL_MODE=position` for a stiff MoveGroup-executed move
instead.
