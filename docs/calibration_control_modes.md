# Calibration Control Modes — Launch Param Reference

Four bring-ups, one per `--controller` value (plus one not yet wired into
capture). Only one may run per arm at a time (`Ctrl+C` before switching).
For the physics/math of impedance vs. admittance, see
[compliant_control.md](compliant_control.md) — not repeated here.

## Quickstart

Left arm (`lbr_one`) shown; swap `lbr_one`→`lbr_two` and `left`→`right` for
the right arm, or drop `arms:=`/`--arm` entirely to run both.

**Admittance** (3 commands):

```bash
ros2 launch lbr_dual_arm_bringup admittance.launch.py arms:=lbr_one use_gripper:=true
scripts/launch_host_realsense.sh
python3 -m src.calibration.capture_handeye_data --arm left --controller admittance
```

**MoveIt/RViz jogging** (4 commands — needs a separate MoveIt terminal):

```bash
ros2 launch lbr_dual_arm_bringup hardware.launch.py arms:=lbr_one use_gripper:=true
ros2 launch lbr_dual_arm_bringup move_group.launch.py mode:=hardware rviz:=true use_gripper:=true
scripts/launch_host_realsense.sh
python3 -m src.calibration.capture_handeye_data --arm left
```

| Mode | Bring-up | `capture_handeye_data.py --controller` | Interface | Controller manager rate |
|---|---|---|---|---|
| MoveIt/RViz jogging (default) | `hardware.launch.py` | `moveit` | position | 100 Hz |
| Admittance (**default Stage A**) | `admittance.launch.py` | `admittance` | position | 100 Hz |
| Gravity compensation | `calibration.launch.py` | `handguided` | torque | 1000 Hz |
| Cartesian impedance (not wired into any capture script — use `CartesianImpedanceDualArmClient` directly) | `cartesian_impedance.launch.py` | — | torque | 1000 Hz |

## Launch args

**`hardware.launch.py` / `admittance.launch.py` / `calibration.launch.py`** — all three:
- `arms:={both,lbr_one,lbr_two}` (default `both`) — real FRI hardware for the selected arm(s) only; the other arm loads as a mock `ros2_control` component so `ros2_control_node` doesn't block in `on_activate` waiting for a connection, and that arm's per-arm controllers aren't spawned. `lbr_one`=left, `lbr_two`=right.
- `use_gripper:={true,false}` (default `true`) — must match across every terminal in a session.
- `robot_name` (default `lbr_dual_arm`) — namespace.

**`hardware.launch.py` only** — `ctrl` (fixed to `joint_trajectory_controller`).

**`cartesian_impedance.launch.py`** — `use_gripper`, `robot_name` only. No `arms` arg yet — always brings up both.

**`move_group.launch.py`** — `mode:={mock,hardware}`, `use_gripper`, `rviz:={true,false}`, `robot_name`. No `arms` arg — MoveIt/RViz always model both arms. Needed for `--controller moveit` and `--mode replay`; not needed for admittance/handguided.

## Pendant settings (per mode, both pendants, before `ros2_control_node` stops waiting)

| | FRI send period | FRI control mode | Client command mode |
|---|---|---|---|
| `hardware.launch.py` / `admittance.launch.py` | 10 ms (100 Hz) | position | `POSITION` |
| `calibration.launch.py` / `cartesian_impedance.launch.py` | 1 ms (1000 Hz) | `JOINT_IMPEDANCE_CONTROL` | `TORQUE` |

With `arms:=lbr_one`/`lbr_two`, only that pendant needs to be started — the mocked arm's pendant/cabinet doesn't need to be on.

## Commands

```bash
# Admittance (default) — both arms, or one:
ros2 launch lbr_dual_arm_bringup admittance.launch.py use_gripper:=true
ros2 launch lbr_dual_arm_bringup admittance.launch.py arms:=lbr_one use_gripper:=true
python3 -m src.calibration.capture_handeye_data --controller admittance
python3 -m src.calibration.capture_handeye_data --arm left --controller admittance --gain-profile insertion
# standalone hand-guiding, independent of capture:
python3 -m src.calibration.admittance_dual_arm --arm left

# Gravity compensation
ros2 launch lbr_dual_arm_bringup calibration.launch.py arms:=lbr_one use_gripper:=true
python3 -m src.calibration.capture_handeye_data --arm left --controller handguided

# Cartesian impedance (both arms only)
ros2 launch lbr_dual_arm_bringup cartesian_impedance.launch.py use_gripper:=true

# Position-controlled / RViz jogging (also needed for --mode replay, autocalibrate_dual_realsense.py)
ros2 launch lbr_dual_arm_bringup hardware.launch.py arms:=lbr_one use_gripper:=true
ros2 launch lbr_dual_arm_bringup move_group.launch.py mode:=hardware rviz:=true use_gripper:=true
python3 -m src.calibration.capture_handeye_data --arm left
python3 -m src.calibration.autocalibrate_dual_realsense
```

`admittance` stays one-arm-at-a-time internally even under `--arm both` (two concurrent admittance loops roughly halve the achievable control rate) — the script just runs left-then-right in one process.

Admittance gain profiles: `config/admittance_gain_profiles.yaml` (`holding` default, `insertion` for lower stiffness). Requires `optas` (in `Dockerfile.thesisnewcuda`, rebuild the `vision` image if missing).

## Where to read more

- [compliant_control.md](compliant_control.md) — impedance/admittance mechanism, gain-service design.
- [calibration_cheatsheet.md](calibration_cheatsheet.md) — full calibration routine.
- [joint_handeye_calibration.md](joint_handeye_calibration.md) — solver math/troubleshooting.
- [moveit_robot_control.md](moveit_robot_control.md) — default RViz-jogging workflow.
