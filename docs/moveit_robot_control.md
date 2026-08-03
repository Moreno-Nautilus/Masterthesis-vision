# Controlling the KUKA LBR with MoveIt

How to jog the robot to a pose via MoveIt's RViz plugin — this is what you
use to position the arm before capturing each sample in the RealSense
hand-eye calibration ([getting_started_realsense.md §4](getting_started_realsense.md#4-hand-eye-calibration-camera-to-flange-offset)),
or any time you want to move the arm interactively without writing code.

This jogging workflow is entirely off-the-shelf `lbr_fri_ros2_stack` +
MoveIt — nothing here is specific to this repo. Most of
`Masterthesis-vision`'s own scripts only ever *read* the robot's pose (via
`/left/ee_pose`, `/right/ee_pose`, or single-arm `/iiwa/ee_pose` — see
[getting_started_realsense.md §5](getting_started_realsense.md#5-the-live-flange-pose-topic--how-it-works));
the exception is `src/calibration/moveit_dual_arm.py`, used by
`autocalibrate_dual_realsense.py` to replay saved calibration poses
automatically through the same MoveGroup action this page's RViz panel
drives interactively.

> **Note on the robot**: despite the `franka_ros2_ws` directory name, the arm
> is a **KUKA LBR iiwa** (`iiwa7` by default), driven by `lbr_fri_ros2_stack`.

---

## 1. Bring up the hardware interface (terminal 1)

This page is linked from the **dual-arm** hand-eye calibration guide
([getting_started_realsense.md §4](getting_started_realsense.md#4-hand-eye-calibration-camera-to-flange-offset)),
so the commands below bring up the dual-arm rig (`lbr_dual_arm_bringup`),
not the single-arm stack — jogging one arm here is the same RViz workflow
either way, sections 3–5 don't change.

```bash
source ~/franka_ros2_ws/install/setup.bash
# use_gripper defaults to true (Y-gripper attached, arm_one/arm_two tipped at
# the gripper TCP); pass use_gripper:=false for the bare flange instead.
ros2 launch lbr_dual_arm_bringup hardware.launch.py use_gripper:=true
```

This needs the KUKA FRI application already streaming from **both**
pendants (left, then right) — if it hangs on "Awaiting robot heartbeat",
that's on the robot-controller side, not fixable from here.

It starts `robot_state_publisher`, `ros2_control_node`, and spawns
`joint_state_broadcaster` + **`joint_trajectory_controller`** for both arms
(the `ctrl:=` launch arg's default). The trajectory controller is what
actually executes motion, and it's the interface MoveIt talks to by
default — no extra config needed to connect the two.

If you don't have the real robot connected right now, `mock.launch.py`
substitutes a simulated (non-physical) robot state for both arms — see
**[visualization.md](visualization.md)** for the single-arm mock-viewer
equivalent. The mock path is fine for familiarizing yourself with the RViz
controls, but it does **not** move a real arm, so it can't be used to
actually collect hand-eye calibration samples.

## 2. Bring up MoveIt + RViz (terminal 2)

```bash
source ~/franka_ros2_ws/install/setup.bash
# use_gripper must match Terminal 1's value.
ros2 launch lbr_dual_arm_bringup move_group.launch.py mode:=hardware rviz:=true use_gripper:=true
```

One launch file gives you both `move_group` **and** RViz, preloaded with
`lbr_dual_arm_moveit_config`'s own `moveit.rviz` config — which already
includes the **MotionPlanning** panel. You do not need to separately add the
panel by hand; `rviz:=true` is sufficient.

Everything comes up namespaced under `/lbr_dual_arm` by default
(`robot_name:=lbr_dual_arm`); `move_group.launch.py` remaps its
planning-scene/robot-description/TF topics accordingly, so nothing else
needs to change. In the MotionPlanning panel's **Planning tab**, pick
`arm_one`, `arm_two`, or `both_arms` as the **Planning Group** to jog that
arm (or both at once) — its tip follows whatever `use_gripper` value both
terminals were launched with.

## 3. Jog the arm and execute

In RViz's **MotionPlanning** panel:

- **Planning tab** — drag the interactive marker (the ball/arrow gizmo on the
  flange) to a target pose. Alternatively, open **Joints** to jog individual
  joints directly.
- **Plan** previews the trajectory without moving the robot.
- **Execute** runs the previewed plan; **Plan & Execute** does both in one
  step.
- **Query Goal State** / **Query Start State** let you stage a target pose
  without committing to it yet, so you can inspect the plan first.

There's no code involved in any of this — it's the standard MoveIt RViz
workflow, unmodified.

## 4. Verify the arm actually moved where you think

```bash
# left arm; substitute lbr_two_* for the right arm. If use_gripper:=true
# (the default), the flange itself is still lbr_one_link_ee -- swap in
# lbr_one_gripper_tcp to check the gripper TCP pose instead.
ros2 run tf2_ros tf2_echo lbr_one_link_0 lbr_one_link_ee
```

Compare against the pose you commanded. This is also exactly what
`flange_pose_publisher` republishes as `/left/ee_pose` (`/right/ee_pose` for
the other arm) for the calibration scripts to consume — if this echoes
correctly, the calibration scripts will see the same pose.

## 5. Shutting down

`Ctrl+C` both terminals (MoveIt/RViz first, then the hardware interface).
There's no persistent state to clean up — the KUKA controller itself keeps
running independently of these ROS nodes.

---

## Where to read more

- **[getting_started_realsense.md §1](getting_started_realsense.md#1-run-it-start-to-finish-the-tested-sequence)**
  — the full RealSense pipeline bring-up sequence, of which this MoveIt step
  is one part.
- **[getting_started_realsense.md §4](getting_started_realsense.md#4-hand-eye-calibration-camera-to-flange-offset)**
  — the calibration routine that uses this workflow to position the arm
  between samples.
- `~/franka_ros2_ws/src/lbr_fri_ros2_stack/lbr_demos/lbr_dual_arm/lbr_dual_arm_bringup/launch/` —
  the actual launch files referenced above (`hardware.launch.py`,
  `move_group.launch.py`, `mock.launch.py`, plus `calibration.launch.py` for
  hand-guided/gravity-compensation calibration), if you need to check or
  override their arguments (`use_gripper`, `mode`, `robot_name`, ...).
- **[hand_guided_calibration.md](hand_guided_calibration.md)** — the
  gravity-compensation alternative to this page's RViz jogging: hand-guide
  the arm instead of dragging the interactive marker, and
  `moveit_dual_arm.py`'s joint-space (`JointTarget`/`move_to_joint()`) goals
  used to replay a captured configuration exactly, instead of this page's
  Cartesian-goal jogging.
- [lbr_fri_ros2_stack docs](https://lbr-stack.readthedocs.io/en/latest/lbr_fri_ros2_stack/lbr_fri_ros2_stack/doc/lbr_fri_ros2_stack.html#quick-start)
  — upstream quick-start, covers FRI setup and controller choices in more
  depth than this page does.
