# Masterthesis — Vision

Multi-camera 6-DoF object pose estimation and tracking for robotic manipulation.
Three calibrated ZED 2i cameras feed a learned perception stack —
**Grounding-DINO → SAM2 → DINOv2 → cross-camera fusion → FoundationPose → ICP** —
that publishes a canonical pose per object in the **robot base frame**.

> **New here / just want to calibrate and run it?** Start with
> **[docs/getting_started.md](docs/getting_started.md)** — a linear, student-facing
> calibrate → run → view guide for the lab rig. The one-page lab cheat sheet is
> shared separately because it contains workstation-specific credentials.
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
>
> **Just want to view tracked objects / camera poses in RViz?** See
> **[docs/visualization.md](docs/visualization.md)** for the MoveIt2 planning
> scene view and camera-extrinsics TF frames — the 2D per-camera overlays
> (Foxglove) are covered in
> [docs/getting_started.md §1](docs/getting_started.md#watch-the-result-in-foxglove)
> instead.

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
| [src/calibration/capture_flange_poses_dual_handguided.py](src/calibration/capture_flange_poses_dual_handguided.py) | Step 1's hand-guided twin: gravity-compensation bring-up instead of RViz jogging, also saves the joint configuration — see [docs/hand_guided_calibration.md](docs/hand_guided_calibration.md). |
| [src/calibration/autocalibrate_dual_realsense.py](src/calibration/autocalibrate_dual_realsense.py) | Dual-arm calibration, Step 2 (automatic): replays the saved poses to solve hand-eye + checkerboard pose + ZED extrinsic — see §6. |
| [src/calibration/moveit_dual_arm.py](src/calibration/moveit_dual_arm.py) | `MoveGroup` action-client helper — the only code in this repo that sends motion commands to the robot (used by the Step 2 script above); Cartesian (`ArmTarget`/`move_to()`) and joint-space (`JointTarget`/`move_to_joint()`) goal types. |
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

Fresh workstation setup has this order:

1. Install host-side prerequisites: NVIDIA driver + NVIDIA Container Toolkit,
   Docker, tmux, ROS 2 Humble, and the camera/robot ROS workspaces used by the
   host launch scripts.
2. Clone this repo, initialize submodules, clone Cutie, and apply the
   FoundationPose patch.
3. Build the Docker image and create the long-lived `vision` container.
4. Download model weights/checkpoints into the paths listed below.
5. Run the one-time build/install commands inside the container.
6. Create/populate `Data/`, then calibrate cameras or run the saved calibration.

### Host prerequisites

The camera drivers and visualization run on the host, not in Docker. The launch
scripts assume these are already available on the workstation:

- NVIDIA driver new enough for CUDA 12.6 and `nvidia-smi`.
- Docker plus the NVIDIA Container Toolkit, so `docker run --gpus all ...` works.
- `tmux`, used by all launch scripts.
- ROS 2 Humble on Ubuntu 22.04.
- Host ROS overlay workspaces sourced by the scripts:
  - [scripts/launch_host.sh](scripts/launch_host.sh) sources
    `$HOME/franka_ros2_ws/install/setup.bash`.
  - [scripts/launch_host_realsense.sh](scripts/launch_host_realsense.sh) sources
    `$HOME/zed_ros2_ws/install/setup.bash`.

If your workstation uses different host workspace names, edit the `SRC_HOST`
line in the corresponding launch script. Leave the rest of this repo in its own
checkout; the scripts compute the repo root dynamically.

#### Building the host camera/robot workspaces from zero

The root of this repo contains zipped copies of the lab-specific host packages:

- `mv_launch.zip` — custom ROS 2 package with `zed2i_pair.launch.py`,
  `zed_realsense_trio.launch.py`, `thesis_stack.launch.py`, and
  `flange_pose_publisher`.
- `fri.zip` — KUKA FRI client SDK ROS package (`fri_client_sdk`).
- `lbr_fri_idl.zip` — KUKA FRI message definitions.
- `lbr_fri_ros2_stack.zip` — KUKA LBR ROS 2 / MoveIt stack used by the robot
  side.

These packages are for the **host** workspaces, not the Docker container. Put
`mv_launch` in the host overlay sourced by the launch script you use. With the
current scripts, [scripts/launch_host_realsense.sh](scripts/launch_host_realsense.sh)
expects it in `~/zed_ros2_ws`, while [scripts/launch_host.sh](scripts/launch_host.sh)
expects it through `~/franka_ros2_ws`.

Camera workspace example:

```bash
sudo apt install unzip python3-colcon-common-extensions python3-rosdep
sudo apt install ros-humble-foxglove-bridge ros-humble-image-proc ros-humble-image-pipeline
sudo rosdep init || true
rosdep update

mkdir -p ~/zed_ros2_ws/src
cd ~/zed_ros2_ws/src
unzip /path/to/Masterthesis-vision/mv_launch.zip

cd ~/zed_ros2_ws
source /opt/ros/humble/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --packages-select mv_launch
source install/setup.bash
```

`mv_launch` also needs the camera driver packages to be discoverable in that same
sourced environment:

- `zed_wrapper` for the ZED cameras.
- `realsense2_camera` for the RealSense variant.

Install/build those according to the lab workstation setup or the vendor docs
before running the host launch scripts. After unzipping `mv_launch`, check the
hardcoded `override_path` in
`~/zed_ros2_ws/src/mv_launch/launch/zed2i_pair.launch.py` and
`~/zed_ros2_ws/src/mv_launch/launch/zed_realsense_trio.launch.py`; the zipped
version points at `/home/pdzuser/zed_ros2_ws/src/mv_launch/config/zed_override_native.yaml`.
Change it if the workspace lives somewhere else. If you put `mv_launch` in
`~/franka_ros2_ws` for the standard 3-ZED script, check the same launch files
under that workspace instead.

Robot workspace example:

```bash
sudo apt install unzip python3-colcon-common-extensions python3-rosdep
sudo rosdep init || true
rosdep update

mkdir -p ~/franka_ros2_ws/src
cd ~/franka_ros2_ws/src
unzip /path/to/Masterthesis-vision/fri.zip
unzip /path/to/Masterthesis-vision/lbr_fri_idl.zip
unzip /path/to/Masterthesis-vision/lbr_fri_ros2_stack.zip

cd ~/franka_ros2_ws
source /opt/ros/humble/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build
source install/setup.bash
```

If `rosdep` reports missing MoveIt, controller, Gazebo, ZED, or RealSense system
packages, install the reported `ros-humble-*` packages on the host and rerun the
same `rosdep`/`colcon build` commands.

### 2.1 Third-party code (`external/`)

Most third-party code is pinned as submodules, not vendored. This includes
`external/FoundationPose`, `external/dinov2`, `external/sam2`, and the custom ROS
message package `src/fp_debug_msgs`.

```bash
git submodule update --init --recursive
bash external/apply_patches.sh        # applies the FoundationPose thesis patch (idempotent)
```

`src/fp_debug_msgs` is a submodule using the SSH URL
`git@github.com:Moreno-Nautilus/fp_debug_msgs.git` on branch
`assembly-cell-interfaces`. If submodule checkout fails there, the workstation
needs GitHub SSH access to that repo, or the submodule URL needs to be changed to
an accessible HTTPS URL.

**Cutie** is git-ignored in this repo and is not pulled by `git submodule
update`. Clone it separately into `external/Cutie` and check out the pinned
commit used by this thesis code:

```bash
git clone https://github.com/hkchengrex/Cutie.git external/Cutie
git -C external/Cutie checkout ec5cdd4cf16f75c73ad785a2f96fb97dbad4125a
```

See [external/README.md](external/README.md) for more background on the
third-party imports and the FoundationPose patch.

### 2.2 Docker

The pipeline runs inside a CUDA container built from
[Dockerfile.thesisnewcuda](Dockerfile.thesisnewcuda). The launch scripts assume a
container named `vision` already exists: they `docker start`/`stop` it and
`docker exec` into it, but they do not build or create it. Override the name with
the `CONTAINER` env var if needed.

Build the image from the repo root:

```bash
docker build -f Dockerfile.thesisnewcuda -t masterthesis:cu126-vision .
```

Create the long-lived container. The repo is mounted as
`/workspace/Masterthesis-vision`:

```bash
docker create -it \
  --name vision \
  --gpus all \
  --network host \
  --ipc host \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v "$PWD:/workspace/Masterthesis-vision" \
  -w /workspace/Masterthesis-vision \
  masterthesis:cu126-vision \
  bash
```

The container uses host networking so it can see the ROS 2 camera topics
published by the host stack. The camera drivers themselves stay on the host.

The Dockerfile creates one Python environment:

```text
/opt/thesis-venv
```

This is a `python3 -m venv --system-site-packages` environment, not conda. The
`--system-site-packages` flag is intentional: ROS 2 Python packages such as
`rclpy`, `ament_index_python`, and the `rosidl` tooling come from Ubuntu/ROS apt
packages in the system Python, while the ML stack is installed with pip in the
venv.

Older notes may call this environment `ros-thesis-venv`; in the current Docker
setup the environment is `/opt/thesis-venv`.

PyTorch and PyTorch3D are installed by the Dockerfile, not manually and not via
conda:

- `torch==2.7.0`, `torchvision==0.22.0`, `torchaudio==2.7.0` from the PyTorch
  CUDA 12.6 wheel index.
- `nvdiffrast` from `git+https://github.com/NVlabs/nvdiffrast.git`.
- `pytorch3d` from
  `git+https://github.com/facebookresearch/pytorch3d.git@stable`.

`nvdiffrast` and `pytorch3d` are installed after Torch with
`--no-build-isolation`, so their CUDA extensions build against the Torch version
already present in `/opt/thesis-venv`.

#### One-time build/install inside the container

After creating the container, start it and run the project-local installs/builds
once:

```bash
docker start vision
docker exec -it vision bash
```

Inside the container:

```bash
# The launch scripts currently still use the legacy path /workspace/MasterThesis.
# Keep this symlink on every workstation unless the launch scripts are updated.
ln -sfn /workspace/Masterthesis-vision /workspace/MasterThesis

cd /workspace/Masterthesis-vision
source /opt/ros/humble/setup.bash
source /opt/thesis-venv/bin/activate

# SAM2 is installed editable, but without dependency resolution so pip does not
# replace the Torch/CUDA versions pinned by the Dockerfile.
pip install --no-deps -e external/sam2

# FoundationPose's C++ helper used by Utils.py. Use build_all_conda.sh here;
# build_all.sh assumes the upstream FoundationPose /kaolin Docker layout.
cd external/FoundationPose
bash build_all_conda.sh
cd /workspace/Masterthesis-vision

# fp_debug_msgs is now a ROS 2 interface package under src/ and is built by colcon.
colcon build --packages-select fp_debug_msgs
source install/setup.bash
```

`external/COLCON_IGNORE` is tracked on purpose, so `colcon build` only sees this
repo's ROS packages such as `src/fp_debug_msgs`; it does not try to build all of
`external/`.

The generated `install/` directory is git-ignored, but the host launch scripts
source it so `visualize_pipeline` can import `fp_debug_msgs`. On a fresh clone,
run the container-side `colcon build` above before starting the host stack.

Do project builds inside the container path you will run from. `colcon` and CMake
cache absolute paths; if a stale build was created from a different path, remove
`build/`, `install/`, and `log/`, then rebuild inside the container.

### 2.3 The `Data/` folder (you must create this)

`Data/` is **git-ignored**, so cloning this repo does **not** give you the meshes
or the reference images. Create it with this layout before running anything. If
you want to copy my current lab `Data/` folder instead of recreating it, reach
out to me directly.

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
scripts/launch_host.sh calibrate  # same, but ZEDs grab/publish at HD2K/15fps (see §4)
```

tmux: `Ctrl+b` then `0..5` to switch windows, `Ctrl+b d` to detach.

If the wrapper cannot stop it for some reason, the raw tmux command is:

```bash
tmux kill-session -t mv_host
```

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

The preset modes run inside tmux session `mv_pipeline`. Stop through the wrapper
when possible:

```bash
scripts/launch_pipeline.sh stop
```

Raw fallback:

```bash
tmux kill-session -t mv_pipeline
```

For the RealSense variant, the sessions are `mv_host_realsense` and
`mv_pipeline_realsense`, so the equivalent raw fallbacks are:

```bash
tmux kill-session -t mv_host_realsense
tmux kill-session -t mv_pipeline_realsense
```

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
that part — or skip jogging entirely with the hand-guided alternative
(gravity-compensation bring-up + `capture_flange_poses_dual_handguided.py`),
see **[docs/hand_guided_calibration.md](docs/hand_guided_calibration.md)**.
Step 2 needs no jogging — it drives both arms itself over the
`moveit_msgs/action/MoveGroup` action (see
[src/calibration/moveit_dual_arm.py](src/calibration/moveit_dual_arm.py)),
including one simultaneous `both_arms_flange` goal per pose-pair for the
hand-eye stage (calibration always targets the bare flange, regardless of
whether the Y-gripper is attached — see `moveit_dual_arm.py`'s docstring).

The original single-arm, single-camera manual scripts
(`handeye_flange_cam_realsense.py`, `board_pose_from_flange_realsense.py`)
still work standalone — see
[docs/getting_started_realsense.md §4.7](docs/getting_started_realsense.md#47-manual-single-camera-fallback-original-scripts-still-available).

---

## 7. MoveIt2 planning scene visualization

Both pipeline runners (`run_pipeline_track_multicam.py` and the RealSense
variant) publish each tracked part as a `moveit_msgs/CollisionObject` on
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
