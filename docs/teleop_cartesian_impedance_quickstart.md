# Cartesian Impedance Teleop — Quickstart

Keyboard jogging of the dual-LBR rig through `src/calibration/teleop_cartesian_impedance.py`,
which drives the **custom** `cartesian_impedance_lbr_one`/`_two` controllers
(`kuka_lbr_control`, patched — see [compliant_control.md §3](compliant_control.md#3-the-cartesian_impedance_controller-patch))
directly over their `target_frame` topic. This is **not** the KUKA Sunrise
cabinet's built-in Cartesian impedance mode, and no MoveGroup goal is ever
sent — MoveIt is only reused for its frame/planning-group conventions (same
`arm_one_flange`/`arm_two_flange`, `lbr_{one,two}_link_0` as
`moveit_dual_arm.py`).

For the general impedance-vs-admittance background and the client's full
API, see [compliant_control.md §1](compliant_control.md#1-cartesian-impedance).
This page is just the fast path to get the teleop script moving the arm.

## 1. SmartPad settings (both pendants)

Same torque-mode settings as `calibration.launch.py`
(gravity-compensation) — see
[calibration_control_modes.md](calibration_control_modes.md#pendant-settings-per-mode-both-pendants-before-ros2_control_node-stops-waiting):

| | Setting |
|---|---|
| FRI send period | 1 ms (1000 Hz) |
| FRI control mode | `JOINT_IMPEDANCE_CONTROL` |
| Client command mode | `TORQUE` |

`cartesian_impedance.launch.py` has no `arms:=` argument — it always brings
up both arms, so **both** pendants need these settings applied and the FRI
application started before `ros2_control_node` will finish activating.

## 2. Launch

```bash
# Terminal 1 — controller bring-up (both arms)
source ~/franka_ros2_ws/install/setup.bash
ros2 launch lbr_dual_arm_bringup cartesian_impedance.launch.py use_gripper:=true

# Terminal 2 — teleop
cd ~/Masterthesis-vision
python3 -m src.calibration.teleop_cartesian_impedance
```

Useful flags:

```bash
python3 -m src.calibration.teleop_cartesian_impedance --arm left
python3 -m src.calibration.teleop_cartesian_impedance --gain-profile holding
python3 -m src.calibration.teleop_cartesian_impedance --linear-step 0.002 --angular-step 0.01
python3 -m src.calibration.teleop_cartesian_impedance --dry-run          # see §4
```

`use_gripper` only matters for which link the physical flange corresponds
to in RViz/collision-checking — the impedance controller's `compliance_ref_link`
is always `lbr_{one,two}_link_ee` (the bare flange) regardless, per
`dual_arm_cartesian_impedance_controllers.yaml` (§3).

## 3. Key bindings

```
Translation (arm base frame):  w/s = +x/-x   a/d = -y/+y   r/f = +z/-z
Rotation (tool frame):         u/o = roll -/+   i/k = pitch -/+   j/l = yaw -/+
[ / ]   halve / double the step size
1 / 2   select left / right arm as active (only when --arm was omitted)
g       cycle gain profile (holding <-> insertion) on the active arm
h       show this help again
q       quit (holds the last-published target; controller keeps running)
```

Translation moves in the arm's own base frame (intuitive world directions);
rotation is about the flange's own current axes (tool frame) — the usual
jog convention. Defaults: 5 mm / ~1.1° per keypress.

`g` calls `set_gains()`, which **refuses the change unless the active arm's
joint velocities are already settled** (`GainChangeUnsafeError`, printed to
the terminal rather than raised) — stop jogging that arm for a moment before
switching profiles.

## 4. No hardware handy? `--dry-run`

There is currently **no simulation backend** that can run this controller:
`lbr_dual_arm_bringup` has no Gazebo launch file, and `mode:=gazebo` on the
shared `lbr_system_interface.xacro` deliberately omits the `effort` command
interface this controller needs (a `gz_ros2_control` limitation — one
command interface per joint — not a config gap). `mock.launch.py` has no
physics either.

`--dry-run` skips ROS/hardware entirely: it starts each selected arm's
target at an identity pose (not the real robot's pose) and prints every
computed target instead of publishing it, so you can sanity-check key
bindings and step sizes:

```bash
python3 -m src.calibration.teleop_cartesian_impedance --dry-run --arm left
```

```
[dry-run] arm_one_flange -> target_frame in lbr_one_link_0: t=(+0.0050, +0.0000, +0.0000) m  rpy=(+0.000, -0.000, +0.000) rad
[dry-run] [left] would set_gains(GainSettings(stiffness={...}, nullspace_stiffness=5.0))
```

This only exercises the script's own pose bookkeeping — it says nothing
about the controller's actual compliant response, which needs the real
arm.

## 5. Stiffness / control config

| What | Where |
|---|---|
| Named gain profiles (`holding`, `insertion`) used by `--gain-profile` / the `g` key | [`config/impedance_gain_profiles.yaml`](../config/impedance_gain_profiles.yaml) |
| Profile loader | `src/calibration/gain_profiles.py` (`load_impedance_profile()`) |
| Controller's out-of-the-box defaults (before any profile is applied) | `~/franka_ros2_ws/src/lbr_fri_ros2_stack/lbr_demos/lbr_dual_arm/lbr_dual_arm_description/ros2_control/dual_arm_cartesian_impedance_controllers.yaml` |
| Controller source (the live-reread patch) | `~/franka_ros2_ws/src/kuka_lbr_control/controllers/cartesian_impedance_controller` |

Stiffness units: N/m for `trans_*`, Nm/rad for `rot_*`, both w.r.t.
`compliance_ref_link` (`lbr_{one,two}_link_ee`). Changing gains is **only
safe while the arm is stationary** — enforced by `set_gains()` itself, not
just documentation (see [compliant_control.md §1](compliant_control.md#runtime-gains-and-nullspace-target)).
There is no way to change stiffness except through the controller's own
`set_parameters` service (`set_gains()` wraps this) — no free topic for it.

## Where to read more

- [compliant_control.md](compliant_control.md) — impedance/admittance mechanism, full `CartesianImpedanceDualArmClient` API, the controller patch (§3).
- [calibration_control_modes.md](calibration_control_modes.md) — pendant settings and launch args across all bring-up modes, side by side.
- [moveit_robot_control.md](moveit_robot_control.md) — the other way to jog the arm (RViz interactive marker, position control, no compliance).
