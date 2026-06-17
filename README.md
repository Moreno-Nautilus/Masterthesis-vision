# Masterthesis — Vision

Multi-camera 6-DoF object pose estimation and tracking for robotic manipulation.
Three calibrated ZED 2i cameras feed a learned perception stack —
**Grounding-DINO → SAM2 → DINOv2 → cross-camera fusion → FoundationPose → ICP** —
that publishes a canonical pose per object in the **robot base frame**.

> **For a micro-step-by-micro-step description of how the pipeline actually runs,
> read [docs/pipeline_walkthrough.md](docs/pipeline_walkthrough.md).** This README
> covers setup, how to launch things, and what each piece of code does; the
> walkthrough explains the algorithm itself.

---

## 1. Repository layout

| Path | What it is |
|------|------------|
| [scripts/](scripts/) | Launch scripts (host stack, pipeline, supervised pipeline) — see §3 |
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
| [src/perception/tracking/realtime_tracker.py](src/perception/tracking/realtime_tracker.py) | Real-time tracker state machine (Cutie mask tracking + ICP pose refinement) used in `track` mode. |
| [src/perception/tracking/cutie_tracker.py](src/perception/tracking/cutie_tracker.py) | Cutie (video object segmentation) wrapper. |
| [src/perception/tracking/icp_refiner.py](src/perception/tracking/icp_refiner.py) | ICP refinement in the base frame + symmetry rotation grid. |
| [src/calibration/base_to_cams_calib_3.py](src/calibration/base_to_cams_calib_3.py) | **3-camera extrinsic calibration** (checkerboard → base frame) — see §4. |
| [src/calibration/io_extrinsics.py](src/calibration/io_extrinsics.py) | Load/save extrinsics YAML (`R` row-major + `t`) ↔ `SE3`. |
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
├── CAD_Models/              # raw object meshes (.obj)
├── CAD_Models_centered/     # origin-centered meshes — USED BY THE PIPELINE (--cad-dir)
├── ZED_screens/             # REAL reference crops, one folder per object (--reference-dir)
│   ├── pb_screw/  *.png
│   ├── pb_pipe/   *.png
│   ├── cooling_f/ *.png
│   └── ... (one subfolder per object_id)
└── reference_renders/       # OPTIONAL synthetic renders (--reference-renders-dir)
    └── <object_id>/ *.png
```

- **`CAD_Models_centered/<object_id>.obj`** — the mesh FoundationPose registers
  against. The default `--cad-dir` points here; the `object_id` is the filename
  stem. Meshes are assumed to be in centimeters (`--mesh-scale 0.01`).
- **`ZED_screens/<object_id>/`** — the **DINO reference bank**: a handful of
  cropped photos of each object. DINOv2 embeds these once at startup and every
  candidate crop is classified against them. This is the default reference source
  (`--reference-source real`).
- **`reference_renders/<object_id>/`** — optional CAD-rendered alternative/extra
  reference views, produced by
  [tools/generate_dino_reference_renders.py](tools/generate_dino_reference_renders.py).
  Used when `--reference-source renders` or `both`.

The folder names under `ZED_screens/` / `reference_renders/` and the `.obj`
filenames must use the **same `object_id`** so labels line up across detection
and pose.

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

### 3.3 Supervised run ([scripts/run_pipeline_supervised.sh](scripts/run_pipeline_supervised.sh))

Wraps the runner and relaunches it **only** on exit code 42 (the rare
"bad SAM session" that produces zero masks for the whole process and is only
cured by a restart). Real crashes / Ctrl-C / clean exits are not looped.

```bash
LOG=outputs/logs/run.log MAX_RESTARTS=10 \
  scripts/run_pipeline_supervised.sh --num-cameras 3 --mask-source gdino_sam ...
```

### Output

Per detected object the pipeline publishes a base-frame `PoseStamped` on
`/perception/fp/pose_base/fused/<track_id>`, writes CSV rows
(`init_pose_log.csv`, `outputs/logs/...csv`), and saves a render under
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
per-tick breakdown: model loading, GDINO proposal, SAM segmentation, DINO
classification + candidate selection, cross-camera fusion, per-object
FoundationPose + ICP (incl. the conditional symmetry rotation grid and the
polishing ICP), and how `init_only` vs `track` mode differ.
