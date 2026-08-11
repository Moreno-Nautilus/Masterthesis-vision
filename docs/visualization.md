# Visualizing tracked objects: MoveIt2 planning scene + camera TF frames

This covers the **3D scene view** — RViz showing each tracked part as a
`moveit_msgs/CollisionObject` in the robot's planning scene, plus the
calibrated camera poses as TF frames alongside it.

> Looking for the **2D per-camera overlays** (SAM/DINO/pose masks drawn on
> the raw camera feed) instead? Those run in Foxglove via `visualize_pipeline`
> and are covered in
> [getting_started.md §1 "Watch the result in Foxglove"](getting_started.md#watch-the-result-in-foxglove)
> — that's the `foxglove`/`viz1-3` windows of `scripts/launch_host.sh`. This
> doc is about the separate `move_group`/RViz scene view below, which needs
> its own launch step.

---

## 1. What gets published

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

## 2. Viewing it in RViz

RViz's `PlanningScene`/`MotionPlanning` display needs a `robot_description`
to initialize against, and needs a `move_group` (or similar) already
maintaining a base scene before it can apply our `is_diff:=true` updates —
otherwise it reports "no planning scene loaded" even though the topic is
publishing correctly. There's no need for the real robot hardware for any of
this — a mock robot is enough.

[../scripts/launch_moveit_scene_viewer.launch.py](../scripts/launch_moveit_scene_viewer.launch.py)
bundles the mock **dual-arm** robot (via `lbr_dual_arm_bringup`'s own
`mock.launch.py`, Y-gripper attached to each flange by default), `move_group`,
RViz, **and** a static-TF broadcast of the calibrated camera extrinsics (§3
below) together for exactly this:

```bash
source /opt/ros/humble/setup.bash
source ~/franka_ros2_ws/install/setup.bash   # wherever the lbr_fri_ros2_stack workspace lives
scripts/view_scene.sh
```

> `ros2 launch` always treats its first argument as a package name, not a
> path — a bare relative path like `scripts/launch_moveit_scene_viewer.launch.py`
> (or even `./scripts/...`) fails with `ValueError: ... is not a valid
> package name`, and it needs an **absolute path** instead.
> [../scripts/view_scene.sh](../scripts/view_scene.sh) resolves that path for
> you, so it works from any directory without hand-editing a path into the
> command. Calling `ros2 launch` on the absolute path directly still works
> too, if you prefer.

Then in RViz: **Add → By display type → moveit_ros_visualization →
PlanningScene**, and set its **Planning Scene Topic** to `/planning_scene`.
Fixed Frame is already `base_link` in the bundled RViz config -- note this
is *not* `world` (there is no `world` link in the dual-arm URDF at all), so
`--dual-arm` is required on `publish_camera_scene_objects`/the tracking
pipeline's `--planning-scene-frame-id` here to match; otherwise `move_group`
logs `Unknown frame: world` for every CollisionObject and the scene never
populates.

Notes/quirks (already hit and fixed once, so no need to rediscover them):
- `lbr_dual_arm_bringup`'s own `move_group.launch.py` doesn't expose a way to
  inject remappings into its internal `move_group` node, and the mock robot
  (`lbr_dual_arm_bringup mock.launch.py`) runs everything under
  `/lbr_dual_arm` — the bundled launch file builds `move_group`/RViz as raw
  `Node` actions with `namespace="lbr_dual_arm"` instead of including that
  launch file directly.
- Namespacing `move_group` under `/lbr_dual_arm` also remaps its
  `/planning_scene` subscription to `/lbr_dual_arm/planning_scene` by
  default, disconnecting it from the bare `/planning_scene` topic the
  pipeline publishes on — the bundled launch file remaps it back explicitly
  (`("/lbr_dual_arm/planning_scene", "/planning_scene")`, same for
  `monitored_planning_scene`/`planning_scene_world`/`collision_object`/
  `attached_collision_object`).
- **Don't "simplify" this by swapping the manual `move_group`/RViz `Node`s
  for an `IncludeLaunchDescription` of `lbr_dual_arm_bringup`'s own
  `move_group.launch.py`.** That's not leftover duplication from folding in
  the dual-arm setup — it's the one piece that has to stay custom, precisely
  because that upstream launch file gives no way to inject the remap above.
  Swap it in and `move_group`/RViz go back to listening on
  `/lbr_dual_arm/planning_scene`; the pipeline's (and
  `publish_camera_scene_objects`') `CollisionObject`s would just silently
  stop showing up in RViz, with nothing erroring to point at why.
- Collision meshes render with flat per-triangle shading that shifts as you
  orbit the camera (`shape_msgs/Mesh` carries no vertex normals) — this is a
  cosmetic limitation of RViz's collision-object rendering, not a sign of bad
  geometry or a wrong pose; it doesn't affect MoveIt's actual collision
  checking, which uses the raw triangle mesh directly.

## 3. Viewing the camera extrinsics as TF frames

[../src/calibration/publish_extrinsics_tf.py](../src/calibration/publish_extrinsics_tf.py)
reads [../config/camera_extrinsics_base.yaml](../config/camera_extrinsics_base.yaml)
and [../config/camera_extrinsics_realsense.yaml](../config/camera_extrinsics_realsense.yaml)
and broadcasts each camera's calibrated pose as a **static TF frame**, so you
can sanity-check calibration results visually instead of reading raw `R`/`t`
numbers:

- `zed2i_1` / `zed2i_2` / `zed2i_3` are published as children of
  `--base-frame` (the active robot's base frame — defaults to `lbr_link_0`,
  the single-arm mock's frame name), matching their camera-to-base meaning.
- `realsense_1` / `realsense_2` are published as children of `--ee-frame`
  (the flange, defaults to `lbr_link_ee`), matching their camera-to-flange
  mount-offset meaning — see the header comment in
  `camera_extrinsics_realsense.yaml` and
  [../src/calibration/io_extrinsics.py](../src/calibration/io_extrinsics.py)
  for the full frame-semantics explanation.

It's already wired into
[../scripts/launch_moveit_scene_viewer.launch.py](../scripts/launch_moveit_scene_viewer.launch.py)
(runs automatically alongside the mock dual-arm robot/`move_group`/RViz), so
nothing extra is needed beyond the launch command above (§2) — the bundled
launch file resolves `config/robot_bases.yaml`'s `active_robot` to that arm's
own link names in the dual-arm URDF (`robot_a` → `lbr_one_link_0`/
`lbr_one_link_ee`, `robot_b` → `lbr_two_link_0`/`lbr_two_link_ee`) and passes
those as `--base-frame`/`--ee-frame`, since the dual-arm mock has no bare
`lbr_link_0`/`lbr_link_ee` frames of its own. To run it standalone against a
different tf tree instead:

```bash
python3 -m src.calibration.publish_extrinsics_tf
# or override the parent frame names / input YAMLs, e.g. for the dual-arm mock:
python3 -m src.calibration.publish_extrinsics_tf --base-frame lbr_two_link_0 --ee-frame lbr_two_link_ee
```

In RViz: **Add → TF**. You should see `zed2i_1/2/3` hanging off the
`--base-frame` and `realsense_1/2` hanging off the `--ee-frame`, alongside
the robot's own tf tree. To check a specific transform numerically:

```bash
ros2 run tf2_ros tf2_echo lbr_two_link_0 zed2i_1
```

Note: this always publishes from `camera_extrinsics_base.yaml`, i.e. the
frame of whichever robot is currently `active_robot` in
`config/robot_bases.yaml`. The robot_a-frame re-expression of these same
poses is not kept as its own config file -- it's logged per-run in
`outputs/calibration_logs/camera_transforms.json` (`T_robotA_cam`) instead.

## 4. Viewing the camera rig itself (ZED2 + holder meshes)

[../src/calibration/publish_camera_scene_objects.py](../src/calibration/publish_camera_scene_objects.py)
publishes the ZED cameras and their mounting holders as
`moveit_msgs/CollisionObject`s on `/planning_scene`, the same mechanism §1
describes for tracked parts -- so they show up in the same PlanningScene
display, at the calibrated `zed2i_1/2/3` poses from
`camera_extrinsics_base.yaml`.

- **Meshes**: [../Assets/ZED2.stl](../Assets/ZED2.stl) (camera) and
  [../Assets/zed_camer_holder.stl](../Assets/zed_camer_holder.stl) (mount),
  both authored in millimeters and scaled to meters on load.
- **Camera-to-holder offset**: no mount has been measured yet, so the holder
  is published at the same pose as the camera (`CAMERA_TO_HOLDER` in the
  script is a mock identity transform) -- replace it once a real measurement
  exists.
- **realsense_1/realsense_2 are not published here**: they're wrist-mounted
  and move with the arm, so they have no fixed scene pose (see §3).
- Already wired into `lbr_dual_arm_bringup/launch/mock.launch.py` (which
  [../scripts/launch_moveit_scene_viewer.launch.py](../scripts/launch_moveit_scene_viewer.launch.py)
  includes for its mock robot, alongside `publish_extrinsics_tf`); to run it
  standalone:

```bash
python3 -m src.calibration.publish_camera_scene_objects
```

- **Dual-arm bringup**: `hardware.launch.py`, `mock.launch.py`,
  `cartesian_impedance.launch.py`, and `admittance.launch.py` in
  `lbr_dual_arm_bringup/launch/` all auto-start this publisher with
  `--dual-arm` (re-expresses every camera pose into the dual-arm's own
  `base_link` frame instead of the active robot's `lbr_link_0` -- see the
  script's module docstring), so move_group's collision checking always
  knows where the camera rig is once any of those bring-ups (or the scene
  viewer above, via `mock.launch.py`) is running. `calibration.launch.py`
  deliberately does **not** start it, since that's the procedure that
  produces `camera_extrinsics_base.yaml` in the first place.
