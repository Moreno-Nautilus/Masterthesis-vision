# Masterthesis — Vision

Multi-camera 6-DoF object pose estimation and tracking for robotic manipulation.
Three calibrated ZED 2i cameras feed a learned perception stack —
**Grounding-DINO → SAM2 → DINOv2 → cross-camera fusion → FoundationPose → ICP** —
that publishes a canonical pose per object in the **robot base frame**.

> **New here / just want to calibrate and run it?** Start with
> **[docs/getting_started.md](docs/getting_started.md)** — a linear, student-facing
> calibrate → run → view guide for the lab rig. There's also a one-page
> **[cheat sheet](docs/cheatsheet.md)** (commands only) to print and tape to the rig.
>
> **For a step-by-step description of how the pipeline actually runs,
> read [docs/pipeline_walkthrough.md](docs/pipeline_walkthrough.md).** This README
> covers setup, how to launch things, and what each piece of code does; the
> walkthrough explains the algorithm itself.
>
> **Experimenting with the 1-ZED + 2-RealSense (end-effector) variant?** See
> **[docs/getting_started_realsense.md §1](docs/getting_started_realsense.md#1-run-it-start-to-finish-the-tested-sequence)**
> for the full step-by-step run sequence, or [§6 below](#6-realsense-trio-variant)
> for a quick reference — a separate, parallel pipeline (own
> scripts/config/launch files); nothing above is affected by it.

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
| [src/calibration/base_to_cams_calib_3.py](src/calibration/base_to_cams_calib_3.py) | **N-camera extrinsic calibration** (checkerboard → base frame) — see §4. Defaults to the 3-ZED trio; accepts `--cam-ids` for any subset (e.g. the RealSense-trio rig's single `zed2i_1`). |
| [src/calibration/io_extrinsics.py](src/calibration/io_extrinsics.py) | Load/save extrinsics YAML (`R` row-major + `t`) ↔ `SE3`. |
| [src/calibration/capture_flange_poses_dual.py](src/calibration/capture_flange_poses_dual.py) | Dual-arm calibration, Step 1 (manual): jog + save each arm's flange poses permanently to `config/flange_poses/` — see §6. |
| [src/calibration/autocalibrate_dual_realsense.py](src/calibration/autocalibrate_dual_realsense.py) | Dual-arm calibration, Step 2 (automatic): replays the saved poses to solve hand-eye + checkerboard pose + ZED extrinsic — see §6. |
| [src/calibration/moveit_dual_arm.py](src/calibration/moveit_dual_arm.py) | `MoveGroup` action-client helper — the only code in this repo that sends motion commands to the robot (used by the Step 2 script above). |
| [src/calibration/flange_pose_store.py](src/calibration/flange_pose_store.py) | JSON schema + save/load for the permanently-stored flange pose captures. |
| [src/calibration/calibration_log.py](src/calibration/calibration_log.py) | Append-only JSON run logs (camera/checkerboard/flange transforms + quality metrics) under `outputs/calibration_logs/`. |
| [src/utils/se3.py](src/utils/se3.py) | Minimal immutable `SE3` rigid-transform type. |
| [tools/generate_dino_reference_renders.py](tools/generate_dino_reference_renders.py) | Renders synthetic reference views from the CAD meshes (optional DINO reference source). |
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
[Dockerfile.thesisnewcuda](Dockerfile.thesisnewcuda). The launch scripts assume a
container named `vision` already exists (they `docker start`/`stop` it, they do
not build it). Override the name with the `CONTAINER` env var.

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
  default reference source (`--reference-source real`).
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
host) and the **pipeline node** (runs inside the docker container).

### 3.1 Host stack — cameras + viz ([scripts/launch_host.sh](scripts/launch_host.sh))

Starts a tmux session with one window each for: the ZED camera driver, the
Foxglove bridge, a `visualize_pipeline` per camera, and `debug_pose_axes`.

```bash
scripts/launch_host.sh            # start (and attach) the tmux session
scripts/launch_host.sh attach     # re-attach if already running
scripts/launch_host.sh stop       # kill the session
```

tmux: `Ctrl+b` then `0..5` to switch windows, `Ctrl+b d` to detach.

### 3.2 Pipeline node — the locked baseline ([scripts/launch_pipeline.sh](scripts/launch_pipeline.sh))

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

`init-only`/`baseline` runs in `init_only` mode (re-detect every tick, never
tracks) and tees output to `outputs/logs/multicam_init_final_baseline.log`.
`fast-track` keeps tracking responsive with centroid recovery and no rotation
reseed/PCA/damping. `accurate-track` adds the rotation reseed + cautious PCA +
light damping preset for better settled screw-axis estimates. The exact pinned
flags are listed in the launch file and the init-only baseline is explained in
[docs/pipeline_walkthrough.md](docs/pipeline_walkthrough.md).

### Output

Per detected object the pipeline publishes a base-frame `fp_debug_msgs/DebugPoseItem`
(identified by `assembly_name`/`part_id`) on the shared
`/perception/fp/pose_base/fused/assembly` topic; with logging flags, it writes
CSV rows (`init_pose_log.csv`, `outputs/logs/...csv`) and saves a render under
`init_renders/`.

---

## 4. Camera-to-base calibration

The pipeline loads camera extrinsics from
[config/camera_extrinsics_base.yaml](config/camera_extrinsics_base.yaml) (`T_base_cam`
per camera). Regenerate this with the 3-camera checkerboard calibration whenever
the cameras move:

**1. Launch the cameras** so the ZED RGB/`camera_info` topics are publishing —
   easiest is the `cams` window of the host stack:

```bash
scripts/launch_host.sh          # window 0 runs the ZED driver
```

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

## 6. RealSense trio variant

Experimental parallel pipeline: `zed2i_1` (static) + 2 end-effector-mounted Intel
RealSense D405 cameras — a separate set of scripts/config/launch files that don't
touch anything above.

**For the full start-to-finish run sequence** (tested and verified working —
tf2/robot setup, host stack, pipeline, what a healthy run looks like, and every
bug already hit and fixed along the way), see
**[docs/getting_started_realsense.md §1](docs/getting_started_realsense.md#1-run-it-start-to-finish-the-tested-sequence)**.

Quick reference once you've read that once:

```bash
# terminal 1 — tf2 for the flange pose (real robot, or identity placeholder — see docs §1 Step 1)
ros2 launch lbr_bringup hardware.launch.py model:=iiwa7

# terminal 2 — host camera stack (ZED + both RealSense + flange_pose_publisher)
scripts/launch_host_realsense.sh

# terminal 3 — the pipeline itself (needs a real terminal, not a backgrounded/piped shell)
scripts/launch_pipeline_realsense.sh init-only
```

**Hand-eye calibration for the two RealSense cameras** (camera-to-flange
offset) plus the checkerboard-in-base-frame and ZED calibration are now a
**two-script dual-arm routine** — see
**[docs/getting_started_realsense.md §4](docs/getting_started_realsense.md#4-hand-eye-calibration-camera-to-flange-offset)**
for the full walkthrough, or
**[docs/calibration_cheatsheet.md](docs/calibration_cheatsheet.md)** for the
condensed command sequence:

```bash
# Step 1 (manual, per arm) — jog + save flange poses, nothing calibrated yet
python3 -m src.calibration.capture_flange_poses_dual --arm left
python3 -m src.calibration.capture_flange_poses_dual --arm right

# Step 2 (automatic replay) — drives both arms itself, solves hand-eye +
# checkerboard pose + ZED extrinsic, in one run
python3 -m src.calibration.autocalibrate_dual_realsense
```

Step 1 still needs jogging the arm interactively via MoveIt between samples;
see **[docs/moveit_robot_control.md](docs/moveit_robot_control.md)** for
that part. Step 2 needs no jogging — it drives both arms itself over the
`moveit_msgs/action/MoveGroup` action (see
[src/calibration/moveit_dual_arm.py](src/calibration/moveit_dual_arm.py)),
including one simultaneous `both_arms` goal per pose-pair for the hand-eye
stage.

The original single-arm, single-camera manual scripts
(`handeye_flange_cam_realsense.py`, `board_pose_from_flange_realsense.py`)
still work standalone — see
[docs/getting_started_realsense.md §4.7](docs/getting_started_realsense.md#47-manual-single-camera-fallback-original-scripts-still-available).

---

## 7. MoveIt2 planning scene visualization

Both pipeline runners (`run_pipeline_track_multicam.py` and the RealSense
variant) publish each tracked part as a `moveit_msgs/CollisionObject` on
`/planning_scene`, in addition to the existing pose topics. The mesh geometry
(from `Data/CAD_Models_centered/`) is embedded directly in the message — a
subscriber (RViz, `move_group`, ...) needs no filesystem access to the CAD
files at all.

- **Identity**: each object is keyed `"{assembly_name}/{part_id}"` (e.g.
  `plumbers_block/0`), matching the same identity already used for the
  `pub_fused_assembly` pose topic. Repeated same-mesh parts (e.g. multiple
  `pb_screw` instances) get one distinct `CollisionObject` per slot, all
  sharing the same mesh geometry.
- **Frame**: every `CollisionObject` header uses a fixed frame name from
  `--planning-scene-frame-id` (default `world`) — **not** a tf2 lookup. Set
  this to whatever frame your robot/world is actually spawned under if it
  isn't `world`.
- **Non-blocking, fail-soft**: publishing is a plain topic publish (never a
  blocking service call), and `_publish_planning_scene_object`/
  `_remove_planning_scene_objects` swallow all failures internally (missing
  mesh, bad pose, publish error) after logging once — a problem here never
  slows or crashes the detection/tracking loop.
- **Removal**: when a track's `pose_status` transitions to `lost` (see
  `_tick()`'s `_force_reinit_tracks` handling), its `CollisionObject` is
  retracted from the scene with a `REMOVE` op.

### Viewing it in RViz

RViz's `PlanningScene`/`MotionPlanning` display needs a `robot_description`
to initialize against, and needs a `move_group` (or similar) already
maintaining a base scene before it can apply our `is_diff:=true` updates —
otherwise it reports "no planning scene loaded" even though the topic is
publishing correctly. There's no need for the real robot hardware for any of
this — a mock robot is enough.

[scripts/launch_moveit_scene_viewer.launch.py](scripts/launch_moveit_scene_viewer.launch.py)
bundles a mock `iiwa7`, `move_group`, and RViz together for exactly this:

```bash
source /opt/ros/humble/setup.bash
source ~/franka_ros2_ws/install/setup.bash   # wherever the lbr_fri_ros2_stack workspace lives
ros2 launch /path/to/Masterthesis-vision/scripts/launch_moveit_scene_viewer.launch.py
```

Then in RViz: **Add → By display type → moveit_ros_visualization →
PlanningScene**, and set its **Planning Scene Topic** to `/planning_scene`.
Fixed Frame is already `world` in the bundled RViz config, matching
`--planning-scene-frame-id`'s default.

Notes/quirks (already hit and fixed once, so no need to rediscover them):
- `lbr_bringup`'s own `move_group.launch.py`/`rviz.launch.py` don't expose a
  namespace argument, but the mock robot (`lbr_bringup mock.launch.py`) runs
  everything under `/lbr` — the bundled launch file builds `move_group`/RViz
  as raw `Node` actions with `namespace="lbr"` instead of including those
  launch files directly.
- Namespacing `move_group` under `/lbr` also remaps its `/planning_scene`
  subscription to `/lbr/planning_scene` by default, disconnecting it from the
  bare `/planning_scene` topic the pipeline publishes on — the bundled launch
  file remaps it back explicitly (`("/lbr/planning_scene", "/planning_scene")`,
  same for `monitored_planning_scene`/`planning_scene_world`/
  `collision_object`/`attached_collision_object`).
- Collision meshes render with flat per-triangle shading that shifts as you
  orbit the camera (`shape_msgs/Mesh` carries no vertex normals) — this is a
  cosmetic limitation of RViz's collision-object rendering, not a sign of bad
  geometry or a wrong pose; it doesn't affect MoveIt's actual collision
  checking, which uses the raw triangle mesh directly.
