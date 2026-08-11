# Calibration Control Modes — Gravity Compensation / Cartesian Impedance / Admittance

A walkthrough of the **three** ways the dual-arm rig can be driven during
calibration, when to reach for each, and exactly how to bring each one up.
This is the entry point — it cross-references
[hand_guided_calibration.md](hand_guided_calibration.md) and
[compliant_control.md](compliant_control.md) for the full technical detail
behind each mode rather than repeating it here.

> **TL;DR table**

| | Gravity compensation | Cartesian impedance | Admittance |
|---|---|---|---|
| Status | Used today — Step 1 alternative | Additive, not yet wired into any calibration script | **Default** — Step 1 of calibration |
| Command interface | `effort` (torque, ~zero task stiffness) | `effort` (torque, task-space spring) | `position` |
| Controller | `gravity_compensation_lbr_one`/`_two` | `cartesian_impedance_lbr_one`/`_two` | `lbr_state_broadcaster_lbr_{one,two}` + `lbr_joint_position_command_controller_lbr_{one,two}` |
| Bring-up | `calibration.launch.py` | `cartesian_impedance.launch.py` | `admittance.launch.py` |
| Client (Masterthesis-vision) | `capture_flange_poses_dual_handguided.py` | `src/calibration/cartesian_impedance_dual_arm.py` | `src/calibration/admittance_dual_arm.py` (control loop) + `capture_flange_poses_dual_admittance.py` (capture) |
| Restoring/goal term | No — arm floats, operator positions it by hand | Yes — spring toward a commanded `target_frame` | No — holds wherever force stops |
| Gains changeable at runtime | N/A (no task stiffness by design) | Yes, via `set_parameters` service, stationary-guarded | Yes, via `set_parameters` service, stationary-guarded |

All three run on the **same URDF/rig**, just different `ros2_control`
controllers activated by different launch files — you can only have **one**
of these three (plus the default `hardware.launch.py`) active at a time per
arm, since they contend for the same command interface.

---

## 1. Gravity compensation — hand-guiding the arm into place

**What it is.** Near-zero commanded torque beyond a model-based gravity
term — the arm floats, and you physically push it wherever you want. No
task-space spring, no goal pose, nothing holding it in place once you let
go except the KUKA cabinet's own brake/friction.

**Where it fits today.** This is **Step 1** of the dual-arm hand-eye
calibration routine — the alternative to RViz-jogging described in
[calibration_cheatsheet.md §1](calibration_cheatsheet.md#1-capture-flange-poses-manual-5-min-per-arm).
You hand-guide each arm to a pose where the checkerboard is visible, then
`capture_flange_poses_dual_handguided.py` records both the Cartesian flange
pose **and** the literal joint configuration (needed later because the
7-DOF arm is redundant — see
[hand_guided_calibration.md §1](hand_guided_calibration.md#1-why)).

**Bring-up:**

```bash
source ~/franka_ros2_ws/install/setup.bash
ros2 launch lbr_dual_arm_bringup calibration.launch.py use_gripper:=true
```

**Pendant settings required first** (`LBRServerSelect` → `LBRServer` app
"Inputs" screen, both pendants): **FRI send period `1 ms`**, **FRI control
mode `JOINT_IMPEDANCE_CONTROL`**, **FRI client command mode `TORQUE`** — see
[hand_guided_calibration.md §3](hand_guided_calibration.md#3-hardware-bring-up-calibrationlaunchpy)
for why each of these specifically (and how they differ from the
position-mode/RViz-jogging pendant setup).

**Capture:**

```bash
python3 -m src.calibration.capture_flange_poses_dual_handguided --arm left
python3 -m src.calibration.capture_flange_poses_dual_handguided --arm right
```

**Full detail** (why it's a separate controller instance per arm, what the
gravity term does/doesn't account for, the SRDF fix it surfaced): see
[hand_guided_calibration.md](hand_guided_calibration.md).

---

## 2. Cartesian impedance — compliant execution of a Cartesian or joint-space goal

**What it is.** A task-space spring-damper: command a pose
(`target_frame`), and the controller applies torque proportional to the
error, with `stiffness.*` setting how stiff that spring is per axis. Unlike
gravity compensation, it **does** hold a goal and resist small pushes;
unlike `joint_trajectory_controller`, it's genuinely compliant — push hard
enough and it yields, rather than fighting back at full torque.

**Where it fits.** Not wired into `autocalibrate_dual_realsense.py`
today — that script still replays captures via
`joint_trajectory_controller`'s hard position control (exact joint-space
replay, see [hand_guided_calibration.md §7](hand_guided_calibration.md#7-replay-autocalibrate_dual_realsensepy)).
This mode is available as a building block for anything calibration-adjacent
that benefits from compliant motion instead — e.g. a compliant final
approach before a capture, or driving the arm through a MoveIt-planned path
without the trajectory controller's rigidity. Two ways to command it,
matching what `capture_flange_poses_dual*.py` already saves:

```python
from src.calibration.cartesian_impedance_dual_arm import CartesianImpedanceDualArmClient
from src.calibration.moveit_dual_arm import ArmTarget, JointTarget

client = CartesianImpedanceDualArmClient(node)

# A Cartesian pose (e.g. from FlangePoseCapture.T_armBase_flange):
client.move_to_cartesian(ArmTarget(group_name="arm_one_flange", base_frame="lbr_one_link_0",
                                    tip_link="lbr_one_link_ee", T_armBase_flange=pose))

# The literal captured joint configuration (e.g. from a hand-guided
# capture's joint_positions) -- FK'd to a target_frame, with the joint
# vector itself set as a soft nullspace bias toward that elbow posture:
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

**Where it fits.** This is now the **default** Step 1 method (see
[calibration_cheatsheet.md §1](calibration_cheatsheet.md#1-capture-flange-poses-manual-5-min-per-arm))
— `capture_flange_poses_dual_admittance.py` runs `AdmittanceDualArmNode`
itself (in a background thread, alongside the interactive capture prompt),
scoped to **only the arm being captured** — not both arms. Running two
`AdmittanceController` instances (each its own Jacobian pseudo-inverse
solve per incoming `LBRState`) concurrently was measured to roughly halve
the achievable control-loop rate for the arm actually being guided, which
on real hardware showed up as the arm feeling rigid/unresponsive even with
the node confirmed running and publishing. The routine is therefore
**sequential and one-arm-at-a-time**: capture left fully, Ctrl-C, then
capture right — safe to do as two entirely separate launch sessions, not
just two calls in one process. The untouched arm's position controller
just holds its last commanded pose (`controller_manager`'s own fixed-rate
loop keeps streaming that to FRI regardless of this node, so there's no
connection-dropout risk from leaving it out). Gravity-compensation
hand-guiding (§1) and RViz jogging remain available as alternatives. Prefer
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
python3 -m src.calibration.capture_flange_poses_dual_admittance --arm left
python3 -m src.calibration.capture_flange_poses_dual_admittance --arm right
# optionally: --gain-profile insertion  (yields more readily than the "holding" default)
```

Same interaction model, output schema (including `joint_positions`), and
checkerboard quality gate as `capture_flange_poses_dual_handguided.py` (§1)
— only the compliance mechanism differs. `AdmittanceDualArmNode` can also be
run standalone (general hand-guiding, independent of any capture script)
via its own entry point:

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

Walking the existing routine
([calibration_cheatsheet.md](calibration_cheatsheet.md)) end to end with
where each mode could apply:

1. **Bring up hardware** — `admittance.launch.py` (§3, **default**) for
   admittance-guided capture, `hardware.launch.py` (position mode) if you're
   jogging in RViz instead, or `calibration.launch.py` (§1, gravity
   compensation) if you're hand-guiding without admittance. Pick one; they
   all feed the same capture scripts, just via different positioning
   methods.
2. **Step 1, capture** — admittance-guided capture (§3,
   `capture_flange_poses_dual_admittance.py`, **default**),
   gravity-compensation hand-guiding (§1), or RViz jogging
   (`hardware.launch.py` + `move_group.launch.py`, unchanged) — pick
   whichever positioning method you prefer, all three feed the same
   downstream schema. Cartesian impedance isn't wired into a capture script
   yet.
3. **Step 2, automatic replay** — `autocalibrate_dual_realsense.py`, still
   `joint_trajectory_controller` under the hood (exact joint-space replay,
   unaffected by anything in this doc). Cartesian impedance (§2) is what
   you'd reach for if replay ever needs to be compliant instead of rigid
   (e.g. the checkerboard/fixture might be in the way and you'd rather the
   arm yield than fault) — `move_to_joint_compliant()` is a drop-in
   candidate for that, but nothing currently calls it from the replay
   script.
4. **Anything requiring contact** (e.g. probing a fixture, force-sensitive
   alignment) is a genuinely new capability none of the existing
   calibration scripts need yet — admittance (§3) or cartesian impedance
   with a low `insertion`-style stiffness profile (§2) are the two starting
   points, not the current gravity-compensation/position-control routine.

**Bottom line for calibration as it exists today**: §3 (admittance) is the
default way to run Step 1 end to end; §1 (gravity compensation) and RViz
jogging remain supported alternatives. §2 is a ready building block for a
compliant replay variant, not yet integrated into
`autocalibrate_dual_realsense.py`.

---

## 5. Quick command reference

```bash
# Admittance (DEFAULT Step 1, admittance-guided capture)
ros2 launch lbr_dual_arm_bringup admittance.launch.py use_gripper:=true
python3 -m src.calibration.capture_flange_poses_dual_admittance --arm left
python3 -m src.calibration.capture_flange_poses_dual_admittance --arm right
# ...or standalone, independent of any capture script:
python3 -m src.calibration.admittance_dual_arm

# Gravity compensation (Step 1 alternative, hand-guided capture)
ros2 launch lbr_dual_arm_bringup calibration.launch.py use_gripper:=true
python3 -m src.calibration.capture_flange_poses_dual_handguided --arm left
python3 -m src.calibration.capture_flange_poses_dual_handguided --arm right

# Cartesian impedance (additive)
ros2 launch lbr_dual_arm_bringup cartesian_impedance.launch.py use_gripper:=true

# Position-controlled pipeline (Step 1 RViz-jogging alternative, Step 2 replay)
ros2 launch lbr_dual_arm_bringup hardware.launch.py use_gripper:=true
ros2 launch lbr_dual_arm_bringup move_group.launch.py mode:=hardware rviz:=true use_gripper:=true
python3 -m src.calibration.autocalibrate_dual_realsense
```

Only one of `hardware.launch.py` / `calibration.launch.py` /
`cartesian_impedance.launch.py` / `admittance.launch.py` should be running
per arm at a time — `Ctrl+C` the current bring-up before switching modes.

## Where to read more

- **[hand_guided_calibration.md](hand_guided_calibration.md)** — gravity
  compensation in full: where it comes from, its limits, the SRDF fix it
  surfaced, and how `capture_flange_poses_dual_handguided.py` /
  `autocalibrate_dual_realsense.py` changed to support it.
- **[compliant_control.md](compliant_control.md)** — cartesian impedance
  and admittance in full: the controller patch, the `set_parameters`-only
  design, gain profiles, and known caveats.
- **[calibration_cheatsheet.md](calibration_cheatsheet.md)** — the
  commands-only walkthrough of the calibration routine itself.
- **[moveit_robot_control.md](moveit_robot_control.md)** — the default
  RViz-jogging / MoveGroup workflow these modes sit alongside.
