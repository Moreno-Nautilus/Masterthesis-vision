# Calibration Control Modes — Gravity Compensation / Cartesian Impedance / Admittance

A walkthrough of the **three** ways the dual-arm rig can be driven during
calibration, when to reach for each, and exactly how to bring each one up.
This is the entry point — it cross-references
[compliant_control.md](compliant_control.md) for the full technical detail
behind cartesian impedance and admittance rather than repeating it here.

> **TL;DR table**

| | Gravity compensation | Cartesian impedance | Admittance |
|---|---|---|---|
| Status | Supported alternative — Stage A `--controller handguided` | Additive, not yet wired into any calibration script | **Default** — Stage A `--controller admittance` |
| Command interface | `effort` (torque, ~zero task stiffness) | `effort` (torque, task-space spring) | `position` |
| Controller | `gravity_compensation_lbr_one`/`_two` | `cartesian_impedance_lbr_one`/`_two` | `lbr_state_broadcaster_lbr_{one,two}` + `lbr_joint_position_command_controller_lbr_{one,two}` |
| Bring-up | `calibration.launch.py` | `cartesian_impedance.launch.py` | `admittance.launch.py` |
| Client (Masterthesis-vision) | `capture_handeye_data.py --controller handguided` | `src/calibration/cartesian_impedance_dual_arm.py` | `src/calibration/admittance_dual_arm.py` (control loop) + `capture_handeye_data.py --controller admittance` |
| Restoring/goal term | No — arm floats, operator positions it by hand | Yes — spring toward a commanded `target_frame` | No — holds wherever force stops |
| Gains changeable at runtime | N/A (no task stiffness by design) | Yes, via `set_parameters` service, stationary-guarded | Yes, via `set_parameters` service, stationary-guarded |

All three run on the **same URDF/rig**, just different `ros2_control`
controllers activated by different launch files — you can only have **one**
of these three (plus the default `hardware.launch.py`, used by
`--controller moveit`) active at a time per arm, since they contend for the
same command interface.

Note: `capture_handeye_data.py` always saves each arm's joint configuration
alongside the Cartesian flange pose, regardless of which `--controller` you
pick — needed later because the LBR arm is 7-DOF for a 6-DOF pose task, so a
Cartesian-only replay is not guaranteed to reproduce the physical posture
that was actually captured (the calibration math itself only ever uses the
Cartesian pose, so this doesn't affect calibration *accuracy* — it matters
for exact repeatability and collision-consistency during replay).

---

## 1. Gravity compensation — hand-guiding the arm into place

**What it is.** Near-zero commanded torque beyond a model-based gravity
term — the arm floats, and you physically push it wherever you want. No
task-space spring, no goal pose, nothing holding it in place once you let
go except the KUKA cabinet's own brake/friction.

**Where it fits today.** A supported `--controller` for Stage A capture
(`capture_handeye_data.py --controller handguided`) — hand-guide each arm to
a pose where the checkerboard is visible, then press Enter; the script
records both the Cartesian flange pose **and** the literal joint
configuration.

**Bring-up:**

```bash
source ~/franka_ros2_ws/install/setup.bash
ros2 launch lbr_dual_arm_bringup calibration.launch.py use_gripper:=true
```

**Pendant settings required first** (`LBRServerSelect` → `LBRServer` app
"Inputs" screen, both pendants), before pressing play:

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
  hold, so the FRI-commanded gravity torque (below) is what you feel, not a
  rigid setpoint fighting you.
- **FRI client command mode: `TORQUE`.** Must match
  `client_command_mode: torque` in `lbr_{one,two}_system_config_torque.yaml`
  — a mismatch here is a common cause of the connection never reaching
  `COMMANDING_ACTIVE`.
- IP address: unchanged from whatever's already configured for
  `hardware.launch.py` — same PC either way.

Set these on **both** pendants (left, then right) before
`calibration.launch.py`'s `ros2_control_node` will stop waiting on "Awaiting
robot heartbeat".

**Where it comes from, and its limits.** Neither `lbr_fri_ros2_stack` nor
this repo ships a gravity-compensation controller natively — it comes from a
separate fork, [`kuka_lbr_control`](https://github.com/idra-lab/kuka_lbr_control)
(`idra-lab`'s fork, built around `ros2_effort_controller`), cloned into
`~/franka_ros2_ws/src/kuka_lbr_control`, pinned to its `main` branch (not
`kuka-prop-ctrl` — that branch dropped the `gravity_compensation` package in
a refactor; `main` has `gravity_compensation`/`cartesian_impedance_controller`/
`joint_impedance_controller` but not `kuka_clik_controller`). Its nested
`lbr-stack/{fri,lbr_fri_ros2_stack,lib_fri_idl}` submodules are deinitialized
(they'd collide with this workspace's top-level `lbr_fri_ros2_stack`) — redo
that (`git submodule deinit -f lbr-stack/fri lbr-stack/lbr_fri_ros2_stack
lbr-stack/lib_fri_idl`) if `kuka_lbr_control` is ever re-cloned from scratch.

Its torque term is a single KDL `ChainDynParam::JntToGravity()` call over
`robot_description` — model-based gravity from whatever masses/COMs are in
the URDF, nothing more. It **does** account for the Y-gripper's mass
(`y_gripper.xacro` has real `<inertial>` tags), but does **not** account for
whatever RealSense camera + mount is bolted on beyond that (no inertial tags
exist for it anywhere in this URDF) — expect a small, currently-uncalibrated
residual pull/droop near the wrist-mounted camera while hand-guiding.

**Hardware finding (2026-08-03) worth knowing if you ever touch
`compensate_gravity`:** the theory that the KUKA cabinet's own internal
gravity compensation makes this controller's `compensate_gravity: false`
safe (matching the single-arm reference config's default) does **not** hold
for this rig/app combination. With `compensate_gravity: false` the arms fell
straight down under their own weight — this FRI app configuration doesn't
apply automatic gravity compensation the way that assumption expects.
`compensate_gravity: true` (our URDF-model gravity term) is the only thing
holding the arms up here, and stays `true` on both arms. A separate, now-fixed
bug (`effort_controller_base.cpp`'s per-cycle torque rate limit being bypassed
on the very first control cycle) was the actual cause of an activation jerk
that initially looked like double-compensation — not `compensate_gravity`
itself; see the `m_first_update` fix in `gravity_compensation.cpp` if you
need the full story.

**SRDF note**: bringing gravity-compensation mode up for the first time
surfaced a genuine pre-existing bug, not something this feature introduced
— the mock config's `lbr_dual_arm_moveit_config/config/lbr_dual_arm.srdf.xacro`
was missing `lbr_{one,two}_link_6` ↔ `lbr_{one,two}_gripper_base_link` in its
`disable_collisions` list (every other link in the chain had the
one-link-back entry — the gripper pair alone was missing). Without it, the
home/default configuration registers as self-colliding, so MoveGroup can't
establish a valid start state and **no** goal plans at all. Fixed by adding
the two missing lines (one per arm), mirroring the existing pattern exactly.

**Capture:**

```bash
python3 -m src.calibration.capture_handeye_data --controller handguided
```

---

## 2. Cartesian impedance — compliant execution of a Cartesian or joint-space goal

**What it is.** A task-space spring-damper: command a pose
(`target_frame`), and the controller applies torque proportional to the
error, with `stiffness.*` setting how stiff that spring is per axis. Unlike
gravity compensation, it **does** hold a goal and resist small pushes;
unlike `joint_trajectory_controller`, it's genuinely compliant — push hard
enough and it yields, rather than fighting back at full torque.

**Where it fits.** Not wired into `capture_handeye_data.py` or
`autocalibrate_dual_realsense.py` today — those still replay/position via
`joint_trajectory_controller`'s hard position control. This mode is
available as a building block for anything calibration-adjacent that
benefits from compliant motion instead — e.g. a compliant final approach
before a capture, or driving the arm through a MoveIt-planned path without
the trajectory controller's rigidity:

```python
from src.calibration.cartesian_impedance_dual_arm import CartesianImpedanceDualArmClient
from src.calibration.moveit_dual_arm import ArmTarget, JointTarget

client = CartesianImpedanceDualArmClient(node)

# A Cartesian pose (e.g. from FlangePoseCapture.T_armBase_flange):
client.move_to_cartesian(ArmTarget(group_name="arm_one_flange", base_frame="lbr_one_link_0",
                                    tip_link="lbr_one_link_ee", T_armBase_flange=pose))

# The literal captured joint configuration (e.g. from a capture's
# joint_positions) -- FK'd to a target_frame, with the joint vector itself
# set as a soft nullspace bias toward that elbow posture:
client.move_to_joint_compliant("left", JointTarget(group_name="arm_one_flange",
                                                     joint_positions=capture.joint_positions))
```

**Bring-up:**

```bash
source ~/franka_ros2_ws/install/setup.bash
ros2 launch lbr_dual_arm_bringup cartesian_impedance.launch.py use_gripper:=true
```

**Defaults** (already reasonable out of the box, no configuration needed
before first use): 1000 N/m translational / 30 Nm/rad rotational stiffness,
`nullspace_stiffness: 0.0` (no elbow bias applied until you explicitly set
one — see below). Named profiles for quick retuning:

```python
from src.calibration import gain_profiles
from src.calibration.cartesian_impedance_dual_arm import GainSettings

client.set_gains("left", GainSettings(**gain_profiles.load_impedance_profile("insertion")))
```

**Nothing about its compliance behavior changes except through a service
call**, and gains specifically only while the arm is stationary — see
[compliant_control.md §1](compliant_control.md#1-cartesian-impedance) for
the full mechanism (`set_gains()` vs `set_nullspace_target()`, why the
nullspace bias is soft, the `set_parameters`-based design).

---

## 3. Admittance — force-driven compliance on the position interface

**What it is.** Software compliance built on top of *position* control, not
torque: it reads the FRI-estimated external joint torque, converts it to a
task-space force, and integrates a position setpoint that moves in response
— push it and it yields in that direction; stop pushing and it just holds
there (no spring back to a nominal pose, unlike cartesian impedance — see
[compliant_control.md §2](compliant_control.md#2-admittance)).

**Where it fits.** This is the **default** Stage A controller —
`capture_handeye_data.py --controller admittance` runs `AdmittanceDualArmNode`
itself (in a background thread, alongside the interactive capture prompt),
scoped to **only the arm being captured** — not both arms. Running two
`AdmittanceController` instances (each its own Jacobian pseudo-inverse
solve per incoming `LBRState`) concurrently was measured to roughly halve
the achievable control-loop rate for the arm actually being guided, which
on real hardware showed up as the arm feeling rigid/unresponsive even with
the node confirmed running and publishing. The routine is therefore
**sequential and one-arm-at-a-time** even under `--arm both` — the script
just runs left-then-right within one process instead of needing two
terminals. The untouched arm's position controller just holds its last
commanded pose (`controller_manager`'s own fixed-rate loop keeps streaming
that to FRI regardless of this node, so there's no connection-dropout risk
from leaving it out). Gravity-compensation hand-guiding (§1) and RViz
jogging (`--controller moveit`) remain available as alternatives. Prefer
admittance when you want per-axis-tunable deadband/responsiveness (via
`--gain-profile`) instead of gravity compensation's uniform floating, or
want to avoid switching the rig into torque mode at all.

**Bring-up:**

```bash
source ~/franka_ros2_ws/install/setup.bash
ros2 launch lbr_dual_arm_bringup admittance.launch.py use_gripper:=true
```

**Capture:**

```bash
python3 -m src.calibration.capture_handeye_data --controller admittance
# optionally: --gain-profile insertion  (yields more readily than the "holding" default)
```

`AdmittanceDualArmNode` can also be run standalone (general hand-guiding,
independent of any capture script) via its own entry point:

```bash
python3 -m src.calibration.admittance_dual_arm            # both arms
python3 -m src.calibration.admittance_dual_arm --arm left  # single arm
python3 -m src.calibration.admittance_dual_arm --gain-profile insertion
```

**Defaults** are applied automatically at startup from
`config/admittance_gain_profiles.yaml`'s `"holding"` profile — nothing to
configure before first use. (Rotational axes were retuned 2026-08-03 — 30%
lower force deadband, 4x higher velocity gain — after hand-guiding felt too
stiff rotationally; see that file's comments.) Changing gains afterward is
possible either through the node's standard `~/set_parameters` service (an
external caller) or, for a caller that already holds the node object
in-process, `admittance_dual_arm.apply_gain_profile()` — both routes hit the
same validating callback, and only while that arm is stationary — see
[compliant_control.md §2](compliant_control.md#2-admittance) for the exact
mechanics.

**New dependency**: needs `optas` (added to
[Dockerfile.thesisnewcuda](../Dockerfile.thesisnewcuda), not yet in the
built `vision` image — rebuild it before running this).

---

## 4. Picking a mode during an actual calibration session

Walking the routine ([calibration_cheatsheet.md](calibration_cheatsheet.md))
end to end with where each mode could apply:

1. **Bring up hardware** — `admittance.launch.py` (§3, **default**) for
   admittance-guided capture, `hardware.launch.py` (position mode) if you're
   jogging in RViz instead, or `calibration.launch.py` (§1, gravity
   compensation) if you're hand-guiding without admittance. Pick one; they
   all feed `capture_handeye_data.py`, just via different `--controller`
   values.
2. **Stage A, capture** — `--controller admittance` (§3, **default**),
   `--controller handguided` (§1), or `--controller moveit` (RViz jogging,
   `hardware.launch.py` + `move_group.launch.py`) — pick whichever
   positioning method you prefer, all three feed the same downstream
   schema. Cartesian impedance isn't wired into a capture script yet.
3. **Board pose + ZED, automatic replay** — `autocalibrate_dual_realsense.py`,
   still `joint_trajectory_controller` under the hood (exact joint-space
   replay, unaffected by anything in this doc); `capture_handeye_data.py
   --mode replay` uses the same mechanism for hand-eye recapture. Cartesian
   impedance (§2) is what you'd reach for if replay ever needs to be
   compliant instead of rigid (e.g. the checkerboard/fixture might be in the
   way and you'd rather the arm yield than fault) — `move_to_joint_compliant()`
   is a drop-in candidate for that, but nothing currently calls it from
   either replay path.
4. **Anything requiring contact** (e.g. probing a fixture, force-sensitive
   alignment) is a genuinely new capability none of the existing
   calibration scripts need yet — admittance (§3) or cartesian impedance
   with a low `insertion`-style stiffness profile (§2) are the two starting
   points, not the current gravity-compensation/position-control routine.

**Bottom line for calibration as it exists today**: §3 (admittance) is the
default way to run Stage A capture end to end; §1 (gravity compensation) and
RViz jogging remain supported alternatives. §2 is a ready building block for
a compliant replay variant, not yet integrated into either replay path.

---

## 5. Quick command reference

```bash
# Admittance (DEFAULT Stage A controller)
ros2 launch lbr_dual_arm_bringup admittance.launch.py use_gripper:=true
python3 -m src.calibration.capture_handeye_data --controller admittance
# ...or standalone, independent of any capture script:
python3 -m src.calibration.admittance_dual_arm

# Gravity compensation (Stage A alternative)
ros2 launch lbr_dual_arm_bringup calibration.launch.py use_gripper:=true
python3 -m src.calibration.capture_handeye_data --controller handguided

# Cartesian impedance (additive)
ros2 launch lbr_dual_arm_bringup cartesian_impedance.launch.py use_gripper:=true

# Position-controlled pipeline (Stage A RViz-jogging alternative, default;
# also needed for --mode replay and for autocalibrate_dual_realsense.py)
ros2 launch lbr_dual_arm_bringup hardware.launch.py use_gripper:=true
ros2 launch lbr_dual_arm_bringup move_group.launch.py mode:=hardware rviz:=true use_gripper:=true
python3 -m src.calibration.capture_handeye_data
python3 -m src.calibration.autocalibrate_dual_realsense
```

Only one of `hardware.launch.py` / `calibration.launch.py` /
`cartesian_impedance.launch.py` / `admittance.launch.py` should be running
per arm at a time — `Ctrl+C` the current bring-up before switching modes.

## Where to read more

- **[compliant_control.md](compliant_control.md)** — cartesian impedance
  and admittance in full: the controller patch, the `set_parameters`-only
  design, gain profiles, and known caveats.
- **[calibration_cheatsheet.md](calibration_cheatsheet.md)** — the
  commands-only walkthrough of the calibration routine itself.
- **[joint_handeye_calibration.md](joint_handeye_calibration.md)** —
  `calibrate_handeye.py --method joint`'s math and troubleshooting.
- **[moveit_robot_control.md](moveit_robot_control.md)** — the default
  RViz-jogging / MoveGroup workflow these modes sit alongside.
