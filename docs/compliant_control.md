# Compliant Control: Cartesian Impedance + Admittance

Two new, additive execution paths for the dual-LBR rig, alongside the
existing position-controlled MoveGroup pipeline described in
[moveit_robot_control.md](moveit_robot_control.md) — unchanged by this;
nothing here replaces `joint_trajectory_controller` as the default.

> Picking a mode for a calibration session, or want gravity compensation
> covered alongside these two? See
> **[calibration_control_modes.md](calibration_control_modes.md)** for the
> walkthrough across all three modes — this page is the technical reference
> for these two specifically.

| | Cartesian impedance | Admittance |
|---|---|---|
| Command interface | `effort` (torque) | `position` |
| Controller | `cartesian_impedance_lbr_one`/`_two` (`kuka_lbr_control`, patched — see §3) | `lbr_joint_position_command_controller_lbr_one`/`_two` (`lbr_ros2_control`, unmodified) |
| Bring-up | `cartesian_impedance.launch.py` | `admittance.launch.py` |
| Client | `src/calibration/cartesian_impedance_dual_arm.py` | `src/calibration/admittance_dual_arm.py` |
| Restoring/goal term | Yes — spring toward `target_frame` | No — holds wherever force stops (see §2) |

Both talk to their controllers directly, not through MoveGroup execution —
MoveGroup planning/collision-checking is still used where it adds value
(`plan_joint_trajectory()`, `compute_fk`), but neither controller executes a
MoveIt trajectory the way `joint_trajectory_controller` does. **Compliance
behavior (stiffness, nullspace, admittance gains) is only ever changed
through a `set_parameters` service call, never a topic or a bare Python
method** — `target_frame` (the motion goal, not "the impedance") is the one
thing that stays a topic; everything that shapes how the arm responds goes
through a service, with reasonable defaults applied out of the box so
nothing needs to be configured before first use.

---

## 1. Cartesian impedance

### Bring-up

```bash
source ~/franka_ros2_ws/install/setup.bash
ros2 launch lbr_dual_arm_bringup cartesian_impedance.launch.py use_gripper:=true
```

Torque-mode twin of `hardware.launch.py`, same shape as
`calibration.launch.py` (see [calibration_control_modes.md §1](calibration_control_modes.md#1-gravity-compensation--hand-guiding-the-arm-into-place)) but spawning
`cartesian_impedance_lbr_one`/`_two` instead of `gravity_compensation_lbr_one`/`_two`.
Config: `lbr_dual_arm_description/ros2_control/dual_arm_cartesian_impedance_controllers.yaml`.
Bring up `move_group.launch.py` separately if you need a plan (§1.3) — this
launch file only changes which controller executes motion.

### Reaching a target

```python
from src.calibration.cartesian_impedance_dual_arm import CartesianImpedanceDualArmClient
from src.calibration.moveit_dual_arm import ArmTarget, JointTarget

client = CartesianImpedanceDualArmClient(node)

# Cartesian goal -- straight to target_frame:
client.move_to_cartesian(ArmTarget(group_name="arm_one_flange", base_frame="lbr_one_link_0",
                                    tip_link="lbr_one_link_ee", T_armBase_flange=pose))

# Joint-space goal (e.g. a hand-guided capture) -- FK'd via MoveIt's
# compute_fk to get target_frame; the joint vector itself is set as the
# controller's nullspace target via set_nullspace_target() (a
# set_parameters service call, see below), biasing the elbow toward it:
client.move_to_joint_compliant("left", JointTarget(group_name="arm_one_flange",
                                                     joint_positions=capture.joint_positions))
```

**The nullspace bias is soft, not a hard constraint.** It's a spring
(`nullspace_stiffness`) pulling the elbow toward the last-set
`nullspace_desired_configuration`, competing against the task-space spring
toward `target_frame` — it will not reproduce a captured configuration as
exactly as `moveit_dual_arm.py`'s `JointConstraint`-based replay does.
Validate by diffing the achieved joint vector against the literal capture
before relying on this for precision-sensitive stages (hand-eye calibration
itself still uses the exact-replay path unchanged).

### Multi-waypoint trajectories

`plan_joint_trajectory(targets, group_name)` calls MoveGroup plan-only (no
execution) and returns the `RobotTrajectory` — MoveIt still does the
planning/collision-checking. `execute_planned_trajectory(arm_key, traj)`
resamples it and streams `target_frame` plus a `set_nullspace_target()` call
per waypoint to the controller, turning it into a compliant trajectory
follower rather than a snap-to-pose one.

### Runtime gains and nullspace target

Both go exclusively through the controller's own `set_parameters` service —
there is no topic for either, and the patched controller re-reads both
every control cycle (§3), so no restart happens:

```python
from src.calibration import gain_profiles
from src.calibration.cartesian_impedance_dual_arm import GainSettings

# Gains (stiffness.*, nullspace_stiffness) -- these are the compliance
# characteristics, so set_gains() enforces the arm is stationary first:
client.set_gains("left", GainSettings(**gain_profiles.load_impedance_profile("insertion")))

# Nullspace TARGET (which configuration to bias toward) -- not a gain, no
# stationary guard, since move_to_joint_compliant()/execute_planned_trajectory()
# both need to set it while the arm may be moving:
client.set_nullspace_target("left", {"lbr_one_A1": 0.1, ...})
```

Defaults shipped in
[dual_arm_cartesian_impedance_controllers.yaml](../../../franka_ros2_ws/src/lbr_fri_ros2_stack/lbr_demos/lbr_dual_arm/lbr_dual_arm_description/ros2_control/dual_arm_cartesian_impedance_controllers.yaml)
are the "reasonable out of the box" values (1000 N/m trans / 30 Nm/rad rot,
`nullspace_stiffness` 0.0 — deliberately off until a caller both raises it
*and* provides a configuration via `set_nullspace_target()`, since there's
no universally-safe default elbow posture). Named profiles for `set_gains()`
live in [config/impedance_gain_profiles.yaml](../config/impedance_gain_profiles.yaml)
(`holding`, `insertion` — tune per-axis for a real task). `set_gains()`
**refuses the change and raises `GainChangeUnsafeError`** unless that arm's
recent joint velocities are already below threshold — a hard requirement,
not a suggestion: don't catch and ignore this exception to force a change
mid-motion. `set_nullspace_target()` has no such guard.

## 2. Admittance

### Bring-up

```bash
source ~/franka_ros2_ws/install/setup.bash
ros2 launch lbr_dual_arm_bringup admittance.launch.py use_gripper:=true
```

Position-mode (no `command_mode:=torque` — `client_command_mode` stays
`position` throughout, same as `hardware.launch.py`). Per arm: 
`lbr_state_broadcaster_lbr_{one,two}` (publishes `LBRState`, incl.
`external_torque`) + `lbr_joint_position_command_controller_lbr_{one,two}`
(position setpoint streaming). Config:
`lbr_dual_arm_description/ros2_control/dual_arm_admittance_controllers.yaml`.

**Known caveat** (pre-existing in the URDF, not introduced by this change):
`lbr_system_interface.xacro`'s FRI-session auxiliary sensor is hardcoded to
the literal name `auxiliary_sensor`, not robot-name-prefixed, so both arms'
hardware components export it under the same key. The per-joint fields
`lbr_state_broadcaster` actually needs (`measured_joint_position`,
`external_torque`) are correctly filtered per arm and unaffected — but the
FRI session/control-mode/connection-quality metadata fields in each arm's
published `LBRState` may be ambiguous between arms. Don't use those fields
to tell which arm a message is about; the topic namespace already does that.

### Running it

```python
from src.calibration.admittance_dual_arm import AdmittanceDualArmNode

node = AdmittanceDualArmNode()
rclpy.spin(node)  # admittance runs itself, one control loop per LBRState message
```

There is **no restoring/goal-tracking term** in the reused
[`lbr_demos_advanced_py.AdmittanceController`](https://lbr-stack.readthedocs.io)
law: with zero external force it holds wherever it currently is (its
internal `dq` decays via exponential smoothing; it does not spring back to
a nominal pose). You cannot "give it a goal pose" the way you can with the
impedance controller's `target_frame` — push it and let go, it stays
displaced. Gain profiles here are purely about compliance/responsiveness
(how much force it takes before it yields, how fast), not pose-holding.

### Runtime gains

`AdmittanceDualArmNode` declares `f_ext_th.<arm>`, `dq_gains.<arm>`,
`dx_gains.<arm>` as ordinary ROS parameters, seeded at startup from
[config/admittance_gain_profiles.yaml](../config/admittance_gain_profiles.yaml)'s
`"holding"` profile (`DEFAULT_GAIN_PROFILE`) — the "reasonable by default"
values, applied automatically before any external call is needed. The
**only** way to change them afterward is this node's standard
`~/set_parameters` service (automatic once parameters are declared — no
custom `.srv`, no other public method exists):

```python
import rclpy
from rcl_interfaces.srv import SetParameters
from rclpy.parameter import Parameter
from src.calibration import gain_profiles

profile = gain_profiles.load_admittance_profile("insertion")
client = caller_node.create_client(SetParameters, "/admittance_dual_arm/set_parameters")
req = SetParameters.Request(parameters=[
    Parameter(name=f"{gain_name}.left", value=Parameter.Type.DOUBLE_ARRAY.to_parameter_type().to_value(values)).to_parameter_msg()
    for gain_name, values in profile.items()
])
future = client.call_async(req)
rclpy.spin_until_future_complete(caller_node, future)
# future.result().results[i].successful / .reason
```

The validating callback (`_on_set_parameters`) checks array lengths and
refuses (`successful=False`, with a `reason`) unless that arm's admittance
loop is currently stationary — checked against the controller's own current
`|dq|` (the joint velocity it's presently commanding), same
stationary-only contract as the impedance client's `set_gains()`. There is
no exception thrown across the service boundary — callers must check
`result.results[i].successful`.

**New dependency**: the reused `AdmittanceController` needs `optas`
(Jacobian via `optas.RobotModel`) — added to
[Dockerfile.thesisnewcuda](../Dockerfile.thesisnewcuda)'s pip install list.
Rebuild the `vision` image before running `admittance_dual_arm.py`.

## 3. The `cartesian_impedance_controller` patch

`~/franka_ros2_ws/src/kuka_lbr_control/controllers/cartesian_impedance_controller`
(a third-party `idra-lab/ros2_effort_controller` git submodule) needed
changes the upstream code didn't have, required for the design above:

1. **Live nullspace target** — `nullspace_desired_configuration` (a
   `double[]` parameter that previously was only read once, at
   `on_configure()`) is now re-read every `update()` cycle, exactly like
   `stiffness.*` below. No topic was added for this — deliberately: the
   whole point is that nothing about the controller's behavior is settable
   except through this node's own `set_parameters` service (see
   `cartesian_impedance_dual_arm.py`'s `set_gains()`/`set_nullspace_target()`).
   An empty or wrong-length array throttle-warns and forces
   `nullspace_stiffness` to 0 for that cycle rather than erroring, since
   this now runs at control-loop rate, not just at configure time.
2. **Live gain re-read** — `stiffness.*`/`nullspace_stiffness` parameters
   are now re-read every `update()` cycle (`updateGainsFromParameters()`)
   instead of only at `on_configure()`, so `set_gains()`/`ros2 param set`
   takes effect without a controller lifecycle transition. Per the agreed
   constraint (only change gains while stationary), there's no low-pass
   filtering — a value takes effect on the very next control cycle.
   `nullspace_desired_configuration` shares this same re-read path but has
   no stationary requirement of its own (see `set_nullspace_target()`).
3. **Capped reference jumps** — `target_frame` commands (`targetFrameCallback()`)
   are clamped through `limitTargetStep()` so the resulting spring
   force/torque (`stiffness.* * error`) never exceeds `max_impedance_force`
   (N) / `max_impedance_torque` (Nm), both new tunable parameters
   (defaults 70 N / 20 Nm, set explicitly in
   `dual_arm_cartesian_impedance_controllers.yaml`). Without this, a
   target far from the currently-applied one — e.g. a MoveIt waypoint
   streamed by `execute_planned_trajectory()` — turned straight into a
   large instantaneous force, i.e. fast/violent motion, and the higher
   `stiffness.*` was set the worse it got. The allowed step now scales
   DOWN as stiffness goes up so the resulting force/torque stays bounded
   regardless of gain; the clamp is direction-preserving (shrinks the
   step, doesn't clip per-axis) and only applies to this manual/streamed
   path — the `FollowJointTrajectory` action path in
   `updateTrajectoryExecution()` already interpolates smoothly via
   `hermiteSample()`.

This patch is **local and uncommitted** in that submodule's own repo
(`https://github.com/idra-lab/ros2_effort_controller.git`, branch `main`) —
it will not survive a fresh clone of `kuka_lbr_control` until committed
there (or forked). `git diff` inside
`~/franka_ros2_ws/src/kuka_lbr_control/controllers` to review it.

---

## 4. Files touched, for reference

**`~/franka_ros2_ws`** (new files except the controller patch):
- `src/kuka_lbr_control/controllers/cartesian_impedance_controller/{src,include}/.../cartesian_impedance_controller.{cpp,h}` — patched, §3.
- `src/lbr_fri_ros2_stack/lbr_demos/lbr_dual_arm/lbr_dual_arm_description/ros2_control/dual_arm_cartesian_impedance_controllers.yaml`
- `src/lbr_fri_ros2_stack/lbr_demos/lbr_dual_arm/lbr_dual_arm_description/ros2_control/dual_arm_admittance_controllers.yaml`
- `src/lbr_fri_ros2_stack/lbr_demos/lbr_dual_arm/lbr_dual_arm_bringup/launch/cartesian_impedance.launch.py`
- `src/lbr_fri_ros2_stack/lbr_demos/lbr_dual_arm/lbr_dual_arm_bringup/launch/admittance.launch.py`

**`Masterthesis-vision`**:
- `src/calibration/cartesian_impedance_dual_arm.py` — new client.
- `src/calibration/admittance_dual_arm.py` — new node.
- `src/calibration/gain_profiles.py` — shared profile loader.
- `config/impedance_gain_profiles.yaml`, `config/admittance_gain_profiles.yaml` — new.
- `Dockerfile.thesisnewcuda` — `optas` added to the pip install list.

## 5. Verification status

The controller patch compiles cleanly (`colcon build --packages-select
cartesian_impedance_controller`) and both new bring-up packages build and
install. The Python modules import cleanly against a live ROS environment
(`cartesian_impedance_dual_arm.py` fully; `admittance_dual_arm.py`
structurally, stubbed against `optas` since that dependency isn't in the
image yet — see §2). `admittance_dual_arm.py`'s `_on_set_parameters`
callback was unit-tested directly (successful change while stationary,
wrong-array-length rejection, moving-arm rejection) against a stubbed
controller — the three behaviors matching the design all pass. **None of
this has been run against real or mock hardware yet** — no `ros2 launch`,
no controller activation, no arm motion. Before trusting any of it: bring
up `cartesian_impedance.launch.py` / `admittance.launch.py`, confirm `ros2
control list_controllers` shows the new instances `active`, and work
through the checks in the approved plan (`tf2_echo` tracking +
yield-under-push, live gain-set with no reconfigure transition,
nullspace-replay-vs-literal-capture diff, admittance push/yield + profile
switch) plus the new `set_parameters`-based gain/nullspace paths
specifically (confirm a real `ros2 service call .../set_parameters`
round-trip, not just the in-process unit test).
