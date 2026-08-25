# Robot Init Pose — Quickstart

Parks both LBR arms at a saved, known joint configuration
(`config/robot_init_pose.yaml`) — e.g. at the start of a session to
guarantee a known starting state, or at the end to leave the rig somewhere
predictable for the next person.

Two scripts:

| | What it does |
|---|---|
| `src/calibration/capture_robot_init_pose.py` | **Snapshot.** Reads `/lbr_dual_arm/joint_states` and writes the current joint positions (both arms) to `config/robot_init_pose.yaml`. Run this once, with the robot at the pose you want as "init". |
| `src/calibration/move_to_init_pose.py` | **Replay.** Drives back to those exact joint values (`JointTarget`, no IK — see [moveit_robot_control.md](moveit_robot_control.md)). This is what you'll normally run. |
| `scripts/launch_robots_to_init_pose.sh` | One-terminal wrapper: brings up bringup + MoveIt in the background, waits for both to be ready, then runs `move_to_init_pose.py` in the foreground. **This is the normal entry point** — the two scripts above are its building blocks. Defaults to `CONTROL_MODE=cartesian_impedance`; set `CONTROL_MODE=position` for a stiff MoveGroup-executed move instead (see §3). |

## 1. Capture a pose (one-time, or whenever you want a new "init")

Jog the arms to the pose you want (RViz/MoveIt, admittance, hand-guiding —
any method), then:

```bash
python3 -m src.calibration.capture_robot_init_pose
```

Overwrites `config/robot_init_pose.yaml`.

## 2. Replay it

```bash
scripts/launch_robots_to_init_pose.sh
```

By default (`CONTROL_MODE=cartesian_impedance`), brings up
`cartesian_impedance.launch.py` + `move_group.launch.py`, waits for
MoveGroup and a plannable state, then plans (MoveIt) + executes (the
compliant controller) the move to the saved pose one arm at a time — see
§3. With `CONTROL_MODE=position`, brings up `hardware.launch.py` +
`move_group.launch.py` instead and plans+executes a single stiff
`both_arms_flange` MoveGroup goal. Bringup + MoveIt stay running afterward
either way (`Ctrl+C` to stop them, or keep going with other work — RViz is
already up).

### Env overrides

```bash
CONTROL_MODE=position scripts/launch_robots_to_init_pose.sh          # stiff MoveGroup move, see §3
ARM=left scripts/launch_robots_to_init_pose.sh                       # one arm only
TIMEOUT_S=300 scripts/launch_robots_to_init_pose.sh                  # slow pendant startup
CONFIG=config/some_other_pose.yaml scripts/launch_robots_to_init_pose.sh
```

`USE_GRIPPER`, `LOG_DIR` are also overridable — see the script's own header
comment for the full list.

## 3. `CONTROL_MODE=cartesian_impedance` (default)

Instead of `hardware.launch.py`'s `joint_trajectory_controller`, bring up
`cartesian_impedance.launch.py` and drive there via the compliant
controller — softer approach, useful if the arms might already be near an
obstacle or each other, and the default for this script. Set
`CONTROL_MODE=position` for the stiffer, MoveGroup-executed move instead.
MoveIt still **plans** the move (same
collision-checked planner as position mode, via `plan_joint_trajectory()`);
the impedance controller only **executes** the already-planned trajectory
(`execute_planned_trajectory()`), it never freelances its own path. See
[compliant_control.md §1](compliant_control.md#1-cartesian-impedance) for
the underlying client.

That controller has no both-arms composite goal, so with `ARM=both` the two
arms are planned+run per-arm, not as one joint goal. The left arm starts on
its own node/thread; the right arm is commanded `LEFT_ARM_LEAD_S` (10s)
later **without waiting for left to finish/settle** — an explicit,
accepted tradeoff (see `move_to_init_pose.py`'s `LEFT_ARM_LEAD_S` and its
module docstring). This means right's plan is checked against whatever live
joint state left happens to be in at that point, **not** against an
already-settled left arm — it is *not* guaranteed collision-safe against a
still-moving left arm. Only rely on this when left's trajectory is already
known to stay clear of the right arm's workspace.

`cartesian_impedance.launch.py` has **no `arms:=` argument** — it always
brings up both arms' controllers regardless of `ARM`; `ARM` only restricts
which arm `move_to_init_pose.py` then commands.

### Pendant settings

| | FRI send period | FRI control mode | Client command mode |
|---|---|---|---|
| `position` (`hardware.launch.py`) | 10 ms (100 Hz) | position | `POSITION` |
| `cartesian_impedance` (`cartesian_impedance.launch.py`) | 1 ms (1000 Hz) | `JOINT_IMPEDANCE_CONTROL` | `TORQUE` |

Source of truth:
[calibration_control_modes.md](calibration_control_modes.md#pendant-settings-per-mode-both-pendants-before-ros2_control_node-stops-waiting).
Set these on the pendant **before** starting `LBRServer` — a mismatch
just makes `ros2_control_node` hang waiting on the wrong FRI stream, not a
clean error.

## 4. Troubleshooting (all of this was hit live debugging this feature)

**Stuck on "Awaiting robot heartbeat. Attempt N..." for one arm** —
`LBRServer` hasn't actually been started (or isn't in its streaming state)
on that pendant yet. Both arms' hardware must connect before
`controller_manager`'s own services (`list_controllers`, `set_parameters`,
...) come up at all — so the *other* arm's controller spawner will also sit
at "Could not contact service .../list_controllers" the whole time, even if
that other arm connected fine.

**One arm connects, then something goes wrong on the pendant itself** — if
`ros2_control_node`'s log shows `Robot connected` / `COMMANDING_ACTIVE` for
that arm and no further state-transition or disconnect line appears, the
ROS side still considers it connected. Anything you observe going wrong
after that point is on the **KUKA SmartPad / Sunrise side** (proprietary,
no logs or API reachable from this host) — check the SmartPad screen
directly, this repo's tooling has no visibility into it.

**A MoveGroup "plannable state" check succeeding doesn't mean hardware is
ready.** `DualArmMoveitClient.wait_for_valid_state_joint()` (used by both
control modes as the first readiness gate) only checks that MoveGroup can
produce *some* plan — it can pass using a placeholder/empty joint state
before the FRI connection or even the controller exists. `cartesian_impedance`
mode additionally waits on `CartesianImpedanceDualArmClient.wait_for_controller()`
(the controller's own `set_parameters` service) before planning/executing,
specifically because of this gap — don't remove that second wait.

**Leftover `ros2_control_node` / `robot_state_publisher` / spawner /
`move_group` / `rviz2` processes after Ctrl+C or a crash** — the launch
script's `cleanup()` trap kills by process group (`set -m` + `kill -TERM --
-$PID`, with a `kill -KILL` fallback after a grace period), not just the
top-level `ros2 launch` PID, specifically because the single-PID version was
observed to leave orphans when `ros2_control_node` was mid-blocked awaiting
a heartbeat at shutdown time. If you still see stragglers (e.g. after a
`kill -9` on the script itself, which skips the trap entirely):

```bash
ps -eo pid,cmd | grep -E "ros2_control_node|robot_state_publisher|move_group|spawner|rviz2" | grep -v grep
kill <pids>          # SIGTERM first
kill -9 <pids>        # if still alive after a couple seconds
```

No `ros2_control_node` in that list but the graph still shows stale nodes
(`ros2 node list` warns about duplicate names)? That's the orphan case
above with `ros2_control_node` itself already exited — safe to kill the
rest and relaunch.

## Where to read more

- [calibration_control_modes.md](calibration_control_modes.md) — pendant
  settings and launch args across all bring-up modes, side by side.
- [compliant_control.md](compliant_control.md) — full
  `CartesianImpedanceDualArmClient` API, gain profiles, the controller
  patch.
- [moveit_robot_control.md](moveit_robot_control.md) — the position-mode
  MoveGroup pipeline this feature's default path reuses.
