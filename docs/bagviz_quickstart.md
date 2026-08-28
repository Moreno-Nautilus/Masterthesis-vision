# Bag Pipeline Visualizer — Quickstart

Offline sanity-check dump of a recorded rosbag's perception-pipeline debug
output: per-camera RGB/depth frames, whatever DINO/SAM/pose/track overlays
the bag already contains, reconstructed binary masks, raw + segmented point
clouds, and a redrawn coordinate-axes overlay + numeric poses for every
detected object.

It only reads what's already in the bag — no docker, no replaying into the
live pipeline, no GPU. Point it at any `ros2bag` (sqlite3 storage) recorded
while `run_pipeline_track_multicam_realsense` was running and it does the
rest.

---

## One-time setup

Already done on this machine — skip to [Run it](#run-it). Documented here
in case this ever needs to run somewhere else.

```bash
conda create -n bagviz python=3.10
conda activate bagviz
pip install -r requirements-bagviz.txt
```

(exact pinned versions + rationale in
[requirements-bagviz.txt](../requirements-bagviz.txt))

Python **3.10 specifically** — that's what ROS 2 Humble's compiled `rclpy`
bindings are built against on this machine. The env deliberately does **not**
pip-install `rclpy`/`rosbag2_py`/`fp_debug_msgs`: this machine's `~/.bashrc`
already sources `/opt/ros/humble/setup.bash` and
`~/franka_ros2_ws/install/setup.bash` into every shell's `PYTHONPATH`, which
is where those come from (including the already-built `fp_debug_msgs`
package). A plain terminal is enough — no docker container needed, since the
custom messages are also built on the host, outside the `vision` container.

---

## Run it

```bash
scripts/visualize_bag_pipeline.sh ~/Desktop/rosbag_20260807_173538
```

Output lands in `outputs/bagviz/<bag_name>_<timestamp>/` by default (gitignored).
Console output looks like:

```
[*] bag: /home/pdzuser/Desktop/rosbag_20260807_173538
[*] cameras: ['realsense_1', 'realsense_2', 'zed2i_1']
    realsense_1: 4 debug frames available, sampling every 1-th -> up to 10
    ...
    [zed2i_1] captured frame 1/10 (pose_items=3, sam=0, dino=0)
    ...
[*] done -> outputs/bagviz/rosbag_20260807_173538_20260816T120000Z
```

If a camera's bag has fewer `DebugFrame` messages than requested, you get
however many actually exist (no error) — the log line for each camera says
exactly how many were available.

---

## What you get

```
outputs/bagviz/<run>/
  manifest.json                        # every frame's summary, one place to skim
  <cam_id>/
    frame_00/
      rgb_native.png                   # full-resolution color frame
      depth_colormap.png               # depth, false-colored (TURBO, clipped to --max-depth-m)
      depth_m.npy                      # raw depth in meters, float32, same resolution
      rgb_raw.png                      # already-rendered by the pipeline's own external
      dino_overlay.png                 # visualizer (visualize_pipeline.py), saved verbatim
      sam_overlay.png                  # when the bag has that topic -- these show whatever
      pose_overlay.png                 # the live pipeline actually rendered at capture time
      track_overlay.png
      mask_track.png                   # binary masks reconstructed from the DebugFrame's
      mask_sam_<i>_<object_id>.png     # own candidate crops (not the overlay images above --
      mask_dino_<i>_<object_id>.png    # these only exist when this frame's DebugFrame
                                        # actually carried that candidate data)
      pointcloud_raw.ply               # full-scene colored point cloud, back-projected from
                                        # depth + camera_info (camera frame)
      pointcloud_raw_base.ply          # same cloud re-expressed in the shared base frame --
                                        # only when T_base_cam was resolvable (always for
                                        # zed2i_1; for realsense_1/realsense_2, only when the
                                        # bag has /left/ee_pose or /right/ee_pose -- see below)
      pointcloud_segmented.ply         # same, masked by the best available mask (priority:
                                        # track mask > top SAM candidate > top DINO candidate)
      pointcloud_segmented_<part>.ply  # one more per pose_item that carries its own mask
      axes_overlay.png                 # rgb + redrawn object axes (camera frame, always;
                                        # base frame too when extrinsics are available)
      poses.yaml                       # numeric pose_camera / pose_base per detected object
      frame_info.yaml                  # this frame's own manifest entry + human-readable notes
    frame_01/
      ...
```

**Read `frame_info.yaml`'s `notes` list first** if something looks missing —
it explains itself instead of silently producing a blank/empty file. The two
notes worth knowing about ahead of time:

- *"This DebugFrame's own sam_candidates/dino_ranked_candidates are empty"* —
  normal for a bag recorded while the pipeline was **tracking**, not
  detecting; the pipeline only fills those fields on a fresh (re-)detection.
  `dino_overlay.png`/`sam_overlay.png` may still show real boxes from the
  external visualizer's cache even when this note fires — check those before
  assuming the bag has nothing. For guaranteed fresh DINO/SAM data, point
  this at a bag recorded during `init-only` (or any moment the pipeline
  re-detected).
- *"Only N/M pose_item origins project inside this camera's frame"* — expected
  for the two eye-in-hand RealSense cameras with cross-camera fused tracking:
  an object tracked mainly by `zed2i_1` still shows up in a RealSense
  camera's `DebugFrame.pose_items`, just not inside that RealSense's current
  view. `axes_overlay.png` correctly only draws what's actually visible;
  `poses.yaml` still has the numeric pose for everything regardless.

---

## Viewing the point clouds

`tools/bagviz/view_pointclouds.py` opens two Open3D windows in sequence for
one sampled frame: first `zed2i_1` alone (in the shared base frame), then
every camera the run has, each transformed into that same base frame and
overlaid, with a coordinate triad per camera plus one per tracked object.
Close a window to advance to the next.

```bash
conda activate bagviz
python -m tools.bagviz.view_pointclouds --run-dir outputs/bagviz/<run> --frame 0
```

The two eye-in-hand RealSense cameras only contribute to the combined view
when their live flange pose was in the bag (`/left/ee_pose` / `/right/ee_pose`)
-- `capture_pipeline_snapshots.py` composes it with
`config/camera_extrinsics_realsense.yaml` and `config/robot_bases.yaml` the
same way the live pipeline does, into the same "base" frame `--extrinsics-yaml`
uses for `zed2i_1`. A camera without it is skipped in the combined view with
the reason printed (and recorded in its `frame_info.yaml` notes) -- `poses.yaml`
and `pointcloud_raw.ply` for that camera are still there, just camera-frame-only.

---

## Useful flags

```bash
scripts/visualize_bag_pipeline.sh <bag> \
    --num-frames 5 \                  # cap per camera (default 10)
    --cameras zed2i_1,realsense_1 \   # default: every camera the bag has debug frames for
    --out-dir /tmp/my_run \           # default: outputs/bagviz/<bag_name>_<timestamp>/
    --voxel-size-m 0.001 \            # point-cloud downsample voxel size, 0 disables it
    --min-depth-m 0.05 --max-depth-m 2.0
```

`--extrinsics-yaml` (default `config/camera_extrinsics_base.yaml`) gives the
static base-frame extrinsic for `zed2i_1`. The two eye-in-hand RealSense
cameras instead get a live per-frame base extrinsic composed from
`--extrinsics-realsense-yaml` (default `config/camera_extrinsics_realsense.yaml`,
the camera-to-flange offset) and `--robot-bases-yaml` (default
`config/robot_bases.yaml`, the cross-arm offset) plus whatever
`/left/ee_pose` / `/right/ee_pose` messages the bag itself has — same math
the live pipeline uses. All three cameras land in the same frame either way
(the active robot's `lbr_link_0`, per `config/robot_bases.yaml`).

Full flag list and behavior: the module docstring in
[tools/bagviz/capture_pipeline_snapshots.py](../tools/bagviz/capture_pipeline_snapshots.py).

---

## Running fresh inference on a captured frame (not the bag's cached detections)

`tools/bagviz/run_object_inference_debug.py` re-runs the actual detection/pose
stages (Grounding DINO → SAM2 → DINOv2 re-ID → FoundationPose) on a captured
frame's `rgb_native.png`/`depth_m.npy`, instead of reading whatever the bag
already had cached in `DebugFrame`.

By default (matching the live node's own `--gdino-use-items-prompt=True`
default), Grounding DINO only ever gets the single class-agnostic prompt
`"items"` — it's a pure box proposer here, same as in the live pipeline.
Object *identity* comes entirely from a DINOv2 embedding classifier run
against a reference bank (`--reference-dir`, default `Data/ZED_screens`),
using the same accept/reject gating (`--dino-min-score`/`--dino-min-margin`,
with the live node's small-object carve-out) as
`_classify_masks_batched()` in `run_pipeline_track_multicam_realsense.py`.

**This used to be different and it mattered:** an earlier version of this
script fed Grounding DINO the *specific* per-part text prompts directly
(`"cooling base,cooling f,..."`) and resolved identity by string-matching
GDINO's own returned label against mesh filenames. Grounding DINO's box
recall for narrow, jargon-y multi-word prompts like `"pb screw"` is
unreliable — it does clean class-agnostic ("items") proposals, but weak
phrase-grounding for specific technical part names — so that version could
detect *nothing* on frames where the object was clearly visible, even
though the live pipeline (which uses the generic prompt) found it fine. If
you need the old text-prompt behavior for some other reason, pass
`--no-gdino-use-items-prompt` (and `--gdino-text-prompts`), but expect
weaker recall than the default.

Any detection whose DINOv2 classification comes back `"unknown"` — or whose
classified `object_id` doesn't resolve to a known CAD mesh under
`--cad-dir` — is treated as **unknown and dropped on the spot**: not added
to the point cloud, not pose-estimated, not drawn, not saved. This mirrors
the live pipeline's own behavior: `_select_top_candidates()` in
`run_pipeline_track_multicam_realsense.py` already drops every candidate
whose DINOv2 classification comes back `"unknown"` before it ever reaches
tracking.

For each camera, the surviving (known) detections' outputs are saved under
a dedicated **`<cam_id>/frame_NN/offline_inference/`** subfolder — kept
separate from everything `capture_pipeline_snapshots.py` writes directly
into `<cam_id>/frame_NN/` (see [What you get](#what-you-get) above),
*including* the live pipeline's own cached
`rgb_raw/dino_overlay/sam_overlay/pose_overlay/track_overlay/axes_overlay`
PNGs whenever the bag had them — otherwise this script's own overlay would
land at the same path/name and look like it came from the live run:

```
<cam_id>/frame_NN/offline_inference/
  pointcloud_objects_debug.ply   # combined cloud, known objects only
  poses_objects_debug.yaml       # per-object pose_camera + mesh_path
  detections_overlay.png         # bbox + mask fill + pose axes, per known object
```

It is a **simplified** stand-in for the live pipeline — no border/dedup mask
filtering, no overlap-based top-candidate dedup or `--max-objects` cap, no
depth-coverage gate on classified candidates, no tracking, no cross-camera
fusion, and no point-cloud filtering beyond a basic depth-validity check.
The module docstring (and its own `FILTERING_REPORT`, printed at the
start/end of every run) lists exactly what the live pipeline additionally
filters that this script skips.

Unlike everything else on this page, it needs the **full GPU inference
stack** (torch, transformers, sam2, FoundationPose/nvdiffrast), so it runs
inside the `vision` docker container, the same place
`run_pipeline_track_multicam_realsense.py` runs — which is normally
**headless**. So, unlike `capture_pipeline_snapshots.py`, this script never
opens an Open3D window itself; it only computes and saves:

```bash
docker exec -it vision bash
python -m tools.bagviz.run_object_inference_debug \
    --run-dir outputs/bagviz/<run> --frame 0
```

It also requires the run to have been captured with a
`capture_pipeline_snapshots.py` new enough to save `frame_info.yaml["K"]`
(camera intrinsics) — re-capture if your run predates that field.

To actually **see** the result, run the companion viewer —
`tools/bagviz/view_object_inference_debug.py` — on a display-capable host, in
the lightweight `bagviz` conda env (same split as `capture_pipeline_snapshots.py`
→ `view_pointclouds.py`; it needs only numpy/open3d/pyyaml, no GPU):

```bash
conda activate bagviz
python -m tools.bagviz.view_object_inference_debug \
    --run-dir outputs/bagviz/<run> --frame 0
```

It opens one Open3D window per camera (that camera's objects-only cloud +
a coordinate triad per detected object, camera frame) and then one combined
window with all cameras' objects + poses overlaid in the shared base frame
(same `T_base_cam` resolution as `view_pointclouds.py` above, including
`--use-config-extrinsics`).

`scripts/debug_object_inference.sh` runs both halves back to back (`docker
exec` into the container for stage 1, then the host `bagviz` conda env for
stage 2) so you don't have to juggle the two environments yourself:

```bash
scripts/debug_object_inference.sh --run-dir outputs/bagviz/<run> --frame 0
```

`--skip-inference` re-views a frame already processed by a previous run;
`--skip-view` runs inference only (e.g. no display anywhere); anything after
a literal `--` is forwarded to stage 1 only (`--gdino-text-threshold`,
`--sam-checkpoint`, etc.). See the script's own header comment for the full
flag list, including the `CONTAINER=`/`CONDA_ENV_NAME=` env overrides.

---

## Where to read more

- **[tools/bagviz/capture_pipeline_snapshots.py](../tools/bagviz/capture_pipeline_snapshots.py)**
  — the tool itself; single file, module docstring has the full design notes.
- **[scripts/visualize_bag_pipeline.sh](../scripts/visualize_bag_pipeline.sh)**
  — thin wrapper: activates the `bagviz` conda env, forwards args.
- **[tools/bagviz/view_pointclouds.py](../tools/bagviz/view_pointclouds.py)**
  — the Open3D point-cloud viewer, see [Viewing the point clouds](#viewing-the-point-clouds).
- **[tools/bagviz/run_object_inference_debug.py](../tools/bagviz/run_object_inference_debug.py)**
  — fresh GDINO/SAM/FoundationPose inference debug tool (GPU, compute-only), see
  [Running fresh inference on a captured frame](#running-fresh-inference-on-a-captured-frame-not-the-bags-cached-detections).
- **[tools/bagviz/view_object_inference_debug.py](../tools/bagviz/view_object_inference_debug.py)**
  — its lightweight Open3D viewer companion, run separately on a display-capable host.
- **[scripts/debug_object_inference.sh](../scripts/debug_object_inference.sh)**
  — thin wrapper: runs both of the above back to back (container, then host).
- **[src/perception/ros/learn_runners/overlay_draw_utils.py](../src/perception/ros/learn_runners/overlay_draw_utils.py)**
  — pure numpy/cv2 drawing + pose-math helpers (axes, mask-crop overlays,
  quaternion/matrix conversions), shared with the live
  `visualize_pipeline.py` external visualizer so the two never drift apart.
- **[src/fp_debug_msgs/msg/DebugFrame.msg](../src/fp_debug_msgs/msg/DebugFrame.msg)**
  (+ `DebugCandidate.msg`, `DebugPoseItem.msg`, `DebugMaskCrop.msg`) — the
  message schema this tool reads; the fields directly explain what's
  available per frame (ROI polygon, SAM/DINO candidates, per-object pose
  items, the Cutie track mask).
- **[getting_started_realsense.md](getting_started_realsense.md)** — how to
  actually run the live pipeline and record a bag in the first place
  (`init-only` for guaranteed fresh DINO/SAM data, `fast-track`/`accurate-track`
  for tracking-mode bags like the one this quickstart's examples use).
