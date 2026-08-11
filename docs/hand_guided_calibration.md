# Hand-Guided Calibration Capture + Joint-Space Replay

New (2026-07-31) alternative to the RViz-jogged capture workflow described
in [moveit_robot_control.md](moveit_robot_control.md) and
[getting_started_realsense.md §4](getting_started_realsense.md#4-hand-eye-calibration-camera-to-flange-offset):
instead of dragging an interactive marker and clicking **Plan & Execute** to
position each arm before a capture, put both arms in gravity-compensation
mode and physically push them into place by hand. This page covers what
changed, why, and how the pieces fit together — see those two pages for the
capture routine's non-mechanical background (stages, YAML outputs, QA
metrics) which didn't change.

> Gravity compensation is one of three dual-arm control modes now
> available — see
> **[calibration_control_modes.md](calibration_control_modes.md)** for a
> walkthrough alongside the two newer ones (Cartesian impedance,
> admittance) and when to reach for each.

---

## 1. Why

The old flow (`capture_flange_poses_dual.py`) has an operator jog the arm
via RViz's MotionPlanning panel — technically still a MoveGroup goal
underneath, just driven interactively instead of programmatically. Hand
guiding removes that layer entirely: the arm is torque-controlled with (close
to) zero commanded effort, so the operator moves it directly, no planner or
IK in the loop for positioning.

The other half of this change: **capture now saves the arm's joint
configuration, not just its Cartesian flange pose**, and
`autocalibrate_dual_realsense.py`'s replay now drives to that exact joint
configuration (a `JointConstraint` goal) instead of re-deriving *some*
configuration from the Cartesian pose via IK. This matters because the LBR
arm is 7-DOF for a 6-DOF pose task — position + orientation alone
under-determine the elbow/null-space configuration, so a Cartesian-only
replay is not guaranteed to reproduce the physical posture that was actually
captured. The calibration math itself (AX=XB hand-eye solve) only ever uses
the Cartesian pose, so this doesn't change calibration *accuracy* — it
matters if you want the arm to physically look the same as it did during
capture (collision-consistency in a specific room setup, exact
repeatability across a session).

## 2. Gravity compensation: where it comes from, and its limits

Neither `lbr_fri_ros2_stack` nor this repo shipped a gravity-compensation
controller before this change. It comes from a separate repo,
[`kuka_lbr_control`](https://github.com/idra-lab/kuka_lbr_control) (an
`idra-lab` fork built around `ros2_effort_controller`), cloned into
`~/franka_ros2_ws/src/kuka_lbr_control`.

Two non-obvious things worth knowing if you ever touch this again:

- **Branch matters.** `kuka_lbr_control`'s `.gitmodules` originally pinned
  its `controllers` submodule to a branch called `kuka-prop-ctrl`. That
  branch has `kuka_clik_controller` (CLIK) but **no** `gravity_compensation`
  package at all — it was dropped in a refactor and the reference in
  `kuka_control/config/controllers.yaml` was left dangling. `gravity_compensation`
  only exists on that fork's `main` branch (and `ergodic_3D`), which in turn
  doesn't have `kuka_clik_controller`. The submodule here is pointed at
  `main` — so `kuka_clik_controller` is **not** available if you clone this
  fresh, only `gravity_compensation` / `cartesian_impedance_controller` /
  `joint_impedance_controller`.
- **The nested `lbr-stack` submodules were removed.** `kuka_lbr_control`
  vendors its own copies of `lbr_fri_idl` / `lbr_fri_ros2_stack` / `fri`
  under `lbr-stack/`, which collide (duplicate package names, colcon hard
  errors) with the top-level `~/franka_ros2_ws/src/lbr_fri_ros2_stack` this
  workspace already uses for the dual-arm rig. Those three submodules were
  `git submodule deinit`'d — only `controllers` (where `gravity_compensation`
  lives) is actually checked out. If `kuka_lbr_control` is ever re-cloned
  from scratch, redo this (`git submodule deinit -f lbr-stack/fri
  lbr-stack/lbr_fri_ros2_stack lbr-stack/lib_fri_idl`) before building.

**What `gravity_compensation` actually does**: its torque term is a single
KDL `ChainDynParam::JntToGravity()` call over `robot_description` — a
model-based gravity torque computed from whatever masses/COMs are in the
URDF, nothing more. There's no runtime "calibrate this specific tool"
routine anywhere in this codebase. Concretely:

- It **does** account for the Y-gripper's mass (`y_gripper.xacro` has real
  `<inertial>` tags: 0.8 kg body + 2×0.08 kg fingers). "Y-gripper" is this
  repo's existing name for the Y-shaped two-finger parallel gripper mounted
  on each arm's `link_7` (`y_gripper.xacro`, meshes under
  `lbr_dual_arm_description/meshes/y_gripper/`) — pre-existing naming, not
  introduced by this change.
- It does **not** account for whatever RealSense camera + mount is bolted
  on beyond that — no inertial tags exist for it anywhere in this URDF.
  Expect a small, currently-uncalibrated residual pull/droop near the
  wrist-mounted camera while hand-guiding. If that residual turns out to
  matter, the fix is adding an `<inertial>` block for the camera mount to
  `y_gripper.xacro` (or wherever it's actually attached), not new code.
- On real KUKA hardware over FRI, the Sunrise cabinet already applies its
  own internal gravity compensation (from whatever Tool/Load Data is
  configured on the cabinet) before FRI torque commands are added on top.
  That's why every controller's `compensate_gravity` in
  `kuka_control/config/controllers.yaml` (the single-arm reference config)
  defaults to `false` — it avoids double-compensating.

  **Tested on real hardware (2026-08-03) — the theory above does NOT hold
  for this rig/app combination.** First hardware run with
  `compensate_gravity: true` produced a hard jerk on activation, which read
  like double-compensation (cabinet's own Tool-Data term + our URDF-model
  term stacking) — so `compensate_gravity` was flipped to `false` on the
  theory the cabinet's own compensation would be enough on its own. It
  wasn't: with `compensate_gravity: false` the arms **fell straight down**
  under their own weight, proving this FRI app configuration does **not**
  apply automatic gravity compensation the way the single-arm reference
  config's `compensate_gravity: false` default assumes. `compensate_gravity`
  is back to `true` on both arms — our URDF-model gravity term (`gravity_compensation.cpp`'s
  `computeTorque()`, gated by `if (m_compensate_gravity)`) is the *only*
  thing holding the arms up in this setup.

  The actual cause of the jerk was unrelated: `effort_controller_base.cpp`'s
  `computeJointEffortCmds()` already rate-limits how much the commanded
  torque can change per cycle (`delta_tau_max`, default 1.0 Nm — a smooth
  ramp to a ~50 Nm gravity torque takes ~50 ms at the 1000 Hz update rate).
  The startup-crash fix below (`m_first_update`) originally *bypassed* that
  ramp on the very first cycle by setting `m_efforts[i] = tau[i]` directly —
  jumping straight to the full torque in a single 1 ms step, which is what
  actually jerked the arm. Fixed by keeping the normal rate-limited step on
  the first cycle too, and only skipping the crash-inducing large-jump guard
  (see below) rather than skipping the ramp itself.

## 3. Hardware bring-up: `calibration.launch.py`

```bash
source ~/franka_ros2_ws/install/setup.bash
ros2 launch lbr_dual_arm_bringup calibration.launch.py
```

Location: `~/franka_ros2_ws/src/lbr_fri_ros2_stack/lbr_demos/lbr_dual_arm/lbr_dual_arm_bringup/launch/calibration.launch.py`.
Twin of `hardware.launch.py`, differing only in:

- `robot_description` is built with `command_mode:=torque` (an arg on
  `lbr_dual_arm.xacro`, default `position`) — selects
  `lbr_{one,two}_system_config_torque.yaml` instead of
  `lbr_{one,two}_system_config_position.yaml`
  (`lbr_dual_arm_description/ros2_control/`), the only difference being
  `client_command_mode: torque` vs `position`. This is what gives
  `gravity_compensation` an effort command interface to write to at all.
- `ros2_control_node` loads `dual_arm_gravity_compensation_controllers.yaml`
  instead of `dual_arm_controllers.yaml`, and spawns
  `gravity_compensation_lbr_one` + `gravity_compensation_lbr_two` instead of
  `joint_trajectory_controller` — automatically, no `ctrl:=` argument to
  pick. **Two separate controller instances**, not one combined one:
  `effort_controller_base`'s kinematic chain is single-chain
  (`robot_base_link` → `end_effector_link`), so each arm needs its own.
- `use_gripper:=true` (default, matches `hardware.launch.py`) — the
  Y-gripper's mass is what makes `compensate_gravity: true` meaningful; see
  §2.

**Pendant-side settings (`LBRServerSelect` → `LBRServer` app "Inputs"
screen)**, both pendants, before pressing play:

- **FRI send period: `1 ms`** (1000 Hz). Must match
  `dual_arm_gravity_compensation_controllers.yaml`'s `controller_manager`
  `update_rate: 1000` — the ros2_control real-time loop and the KUKA-side
  FRI cycle have to agree, or the FRI session either refuses to reach
  `COMMANDING_ACTIVE` or gets dropped mid-session by the cabinet's
  cycle-time-violation monitor. This is **different from position-mode**
  bring-up (`hardware.launch.py` / `dual_arm_controllers.yaml`,
  `update_rate: 100` → `10 ms` send period) — if the pendant was last set
  up for RViz jogging, this has to change before switching to gravity
  compensation.
- **FRI control mode: `JOINT_IMPEDANCE_CONTROL`.** Per the command-interface
  table in `~/franka_ros2_ws/src/lbr_fri_idl/README.md`, the `TORQUE` client
  command mode is only valid paired with `JOINT_IMPEDANCE_CONTROL` — this is
  also what makes the arm physically backdrivable at all: the KUKA-side
  controller runs a soft joint-impedance loop instead of a stiff position
  hold, so the FRI-commanded gravity torque (§2) is what you feel, not a
  rigid setpoint fighting you.
- **FRI client command mode: `TORQUE`.** Must match
  `client_command_mode: torque` in `lbr_{one,two}_system_config_torque.yaml`
  (§3 above) — a mismatch here is a common cause of the connection never
  reaching `COMMANDING_ACTIVE`.
- IP address: unchanged from whatever's already configured for
  `hardware.launch.py` — same PC either way.

Set these on **both** pendants (left, then right — same order the app gets
started in) before `calibration.launch.py`'s `ros2_control_node` will stop
waiting on "Awaiting robot heartbeat".

Both arms come up hand-guidable immediately. `robot_state_publisher` +
`joint_state_broadcaster` are unaffected by which controller is active, so
`/left/ee_pose`, `/right/ee_pose` (via `flange_pose_publisher`, launched
separately — `scripts/launch_host_realsense.sh` / `zed_realsense_trio.launch.py`,
unchanged) and `/lbr_dual_arm/joint_states` work exactly as they do under
`hardware.launch.py`.

**SRDF note**: bringing this up for the first time surfaced a genuine
pre-existing bug, not something this feature introduced — the mock
config's `lbr_dual_arm_moveit_config/config/lbr_dual_arm.srdf.xacro` was
missing `lbr_{one,two}_link_6` ↔ `lbr_{one,two}_gripper_base_link` in its
`disable_collisions` list (every other link in the chain had the
one-link-back entry — `link_5`↔`link_6`, `link_1..4`↔`link_6`/`link_7` — the
gripper pair alone was missing). Without it, the home/default configuration
registers as self-colliding, so MoveGroup can't establish a valid start
state and **no** goal plans at all, Cartesian or joint-space. Fixed by
adding the two missing lines (one per arm), mirroring the existing pattern
exactly.

## 4. Capture: `capture_flange_poses_dual_handguided.py`

```bash
python3 -m src.calibration.capture_flange_poses_dual_handguided --arm left
python3 -m src.calibration.capture_flange_poses_dual_handguided --arm right
```

Same interaction model as `capture_flange_poses_dual.py` (checkerboard
quality gate, save-after-every-capture reliability, same
`config/flange_poses/<arm>.json` output) — the only differences:

- Instructions say to hand-guide the arm (after `calibration.launch.py` is
  up) instead of jogging it in RViz.
- It additionally subscribes to `/lbr_dual_arm/joint_states` and, on each
  accepted capture, saves that arm's 7 raw joint angles alongside the
  Cartesian flange pose.

## 5. Schema: `FlangePoseCapture.joint_positions`

`src/calibration/flange_pose_store.py` — `FlangePoseCapture` gained a
`joint_positions: dict` field (`{"lbr_one_A1": 0.12, ..., "lbr_one_A7": -0.4}`),
serialized alongside `T_armBase_flange` in `config/flange_poses/<arm>.json`.
Backward compatible: `_capture_from_dict` defaults missing/absent
`joint_positions` to `{}` for anything captured before this field existed.

**This means captures made with the old `capture_flange_poses_dual.py` have
`joint_positions == {}`** and cannot be used for replay anymore (see §7) —
they need to be recaptured with the hand-guided script.

## 6. Joint-space MoveIt goals: `moveit_dual_arm.py`

Two goal styles now coexist in `DualArmMoveitClient`:

| | `ArmTarget` / `move_to()` / `plan_only()` | `JointTarget` / `move_to_joint()` / `plan_only_joint()` |
|---|---|---|
| Goal type | Cartesian (`PositionConstraint` + `OrientationConstraint`) | Joint-space (`JointConstraint` per joint) |
| IK involved | Yes, MoveGroup's IK sampler picks *a* solution | No — skipped entirely |
| Used for | Reachability probing, the startup "is move_group ready" check | Replaying a captured configuration exactly (what `autocalibrate_dual_realsense.py` uses) |

`move_to_joint()` additionally **reads back and permanently logs** what was
actually achieved: after the goal completes (success *or* failure — a
failed/partial move is exactly the case most worth a record of), it waits
~0.2s for `/lbr_dual_arm/joint_states` to settle, reads the achieved values
for the requested joints, and writes both requested-vs-achieved plus the
MoveIt error code to `outputs/calibration_debug/moveit_joint_targets/<timestamp>.json`
(same incremental-write philosophy as the capture scripts; if `outputs/` is
root-owned on your checkout, this warns and returns the record to the
caller instead of crashing — same handling as `mock_reachability_check.py`).

**SRDF groups**: `arm_one`/`arm_two`/`both_arms` now tip at the Y-gripper
TCP by default (`use_gripper` toggle). Calibration — both the old Cartesian
path and the new joint-space one — uses `arm_one_flange`/`arm_two_flange`/
`both_arms_flange` instead, which are **always** pinned to the bare flange
regardless of `use_gripper`, since the calibration math and
`config/flange_poses/*.json` are anchored to the physical flange. This is
what `flange_pose_store.ARM_KEYS[...]["group_name"]` points to.

## 7. Replay: `autocalibrate_dual_realsense.py`

Stage A (`both_arms_flange`, simultaneous) and Stage B (single-arm) now
build `JointTarget`s from each capture's `joint_positions` and call
`move_to_joint()` instead of `move_to()`. `_require_joint_positions()` runs
before any motion and raises immediately if a selected capture has empty
`joint_positions` — pointing at `capture_flange_poses_dual_handguided.py` as
the fix, rather than failing confusingly mid-run.

The readiness probe (`wait_for_valid_state`, confirming move_group's
current-state monitor is actually up before the first real goal) is
unchanged — still Cartesian/`ArmTarget`-based, since its job is generic
readiness, not exercising the production goal type.

## 8. Smoke test: `mock_joint_target_check.py`

```bash
ros2 launch lbr_dual_arm_bringup mock.launch.py
ros2 launch lbr_dual_arm_bringup move_group.launch.py mode:=mock rviz:=false
python3 -m src.calibration.mock_joint_target_check           # plan-only
python3 -m src.calibration.mock_joint_target_check --execute # also moves the mock robot
```

Simulation-only validation of the `JointTarget`/`move_to_joint`/
`plan_only_joint` code path itself — mirrors `mock_reachability_check.py`
but for joint-space goals. Since no real hand-guided captures exist yet on
this checkout (`joint_positions` is `{}` for every current entry in
`config/flange_poses/*.json` — see §5), it builds **synthetic** targets from
whatever configuration the mock robot reports as its home state, plus a
±0.15 rad perturbation, and checks `arm_one_flange`, `arm_two_flange`, and
`both_arms_flange` can all plan to both.

**Result as of 2026-07-31** (after the SRDF fix in §3): 6/6 plan-only checks
pass. `--execute` currently fails for **single-arm** goals with
`Joints on incoming trajectory don't match the controller joints` from
`joint_trajectory_controller` — a separate, pre-existing gap: both
`moveit_controllers.yaml` and `dual_arm_controllers.yaml` configure
`joint_trajectory_controller` with all 14 joints (both arms) and no
`allow_partial_joints_goal: true`, so it rejects any trajectory that only
covers one arm's 7 joints. This affects execution generically (Cartesian or
joint-space, doesn't matter) — planning/reachability itself is unaffected.

**Fixed 2026-08-01** — added `allow_partial_joints_goal: true` under
`joint_trajectory_controller`'s `ros__parameters` in `dual_arm_controllers.yaml`
(`~/franka_ros2_ws/src/lbr_fri_ros2_stack/...`, an uncommitted local diff, not
this repo). Confirmed via `mock_reachability_check.py --execute`: single-arm
goals went from 0/14 to 13/14 actually executing (visible in RViz); the one
remaining flake was tracked down to an RRTConnect sampling-planner artifact,
not a controller or reachability problem — see the 2026-08-01 update in
`README_mock_calibration_reachability.md`
(`/home/pdzuser/Desktop/claude_docs`) for the full investigation.

## 9. Files touched, for reference

**`~/franka_ros2_ws/src/kuka_lbr_control`** (new clone):
- `.gitmodules` — `controllers` submodule branch `kuka-prop-ctrl` → `main`.
- `lbr-stack/{fri,lbr_fri_ros2_stack,lib_fri_idl}` submodules deinitialized.

**`~/franka_ros2_ws/src/lbr_fri_ros2_stack`** (new files only, except the
two-line SRDF fix):
- `lbr_demos/lbr_dual_arm/lbr_dual_arm_description/ros2_control/lbr_{one,two}_system_config_torque.yaml`
- `lbr_demos/lbr_dual_arm/lbr_dual_arm_description/ros2_control/dual_arm_gravity_compensation_controllers.yaml`
- `lbr_demos/lbr_dual_arm/lbr_dual_arm_bringup/launch/calibration.launch.py`
- `lbr_demos/lbr_dual_arm/lbr_dual_arm_moveit_config/config/lbr_dual_arm.srdf.xacro` — two `disable_collisions` lines added (§3).

**`Masterthesis-vision`**:
- `src/calibration/flange_pose_store.py` — `joint_positions` field.
- `src/calibration/capture_flange_poses_dual_handguided.py` — new script.
- `src/calibration/moveit_dual_arm.py` — `JointTarget`, `move_to_joint()`, `plan_only_joint()`, achieved-state logging.
- `src/calibration/autocalibrate_dual_realsense.py` — joint-space replay, `_require_joint_positions()`.
- `src/calibration/mock_joint_target_check.py` — new smoke-test script.

## 10. Migration checklist

If you're picking this up fresh:

1. `ros2 launch lbr_dual_arm_bringup calibration.launch.py` (real hardware) or `mock.launch.py` + `move_group.launch.py mode:=mock` (sim only — §8).
2. Any `config/flange_poses/*.json` captured before today has empty `joint_positions` and **will be rejected** by `autocalibrate_dual_realsense.py` (`_require_joint_positions`, §7) — recapture with `capture_flange_poses_dual_handguided.py` (§4) before running it.
3. If you need single-arm `--execute` (not just plan-only) in mock, add `allow_partial_joints_goal: true` first — see §8.
