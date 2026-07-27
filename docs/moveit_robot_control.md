# Controlling the KUKA LBR with MoveIt

How to jog the robot to a pose via MoveIt's RViz plugin — this is what you
use to position the arm before capturing each sample in the RealSense
hand-eye calibration ([getting_started_realsense.md §4](getting_started_realsense.md#4-hand-eye-calibration-camera-to-flange-offset)),
or any time you want to move the arm interactively without writing code.

This is entirely off-the-shelf `lbr_fri_ros2_stack` + MoveIt — nothing here
is specific to this repo, and none of `Masterthesis-vision`'s own scripts
send motion commands to the robot. They only ever *read* its pose (via
`/iiwa/ee_pose`) — see [getting_started_realsense.md §5](getting_started_realsense.md#5-the-live-flange-pose-topic--how-it-works).

> **Note on the robot**: despite the `franka_ros2_ws` directory name, the arm
> is a **KUKA LBR iiwa** (`iiwa7` by default), driven by `lbr_fri_ros2_stack`.

---

## 1. Bring up the hardware interface (terminal 1)

```bash
source ~/franka_ros2_ws/install/setup.bash
ros2 launch lbr_bringup hardware.launch.py model:=iiwa7
```

This needs the KUKA FRI application already streaming from the
controller/pendant side — if it hangs on "Awaiting robot heartbeat", that's
on the robot-controller side, not fixable from here.

It starts `robot_state_publisher`, `ros2_control_node`, and spawns
`joint_state_broadcaster` + `lbr_state_broadcaster` +
**`joint_trajectory_controller`** (the `ctrl:=` launch arg's default). The
trajectory controller is what actually executes motion, and it's the
interface MoveIt talks to by default — no extra config needed to connect the
two.

If you don't have the real robot connected right now, `mock.launch.py`
substitutes a simulated (non-physical) robot state — see
[scripts/launch_moveit_scene_viewer.launch.py](../scripts/launch_moveit_scene_viewer.launch.py)
for an existing example that wires up `mock.launch.py` + `move_group` + RViz
together for viewing tracked objects. That mock path is fine for
familiarizing yourself with the RViz controls, but it does **not** move a
real arm, so it can't be used to actually collect hand-eye calibration
samples.

## 2. Bring up MoveIt + RViz (terminal 2)

```bash
source ~/franka_ros2_ws/install/setup.bash
ros2 launch lbr_bringup move_group.launch.py model:=iiwa7 rviz:=true
```

One launch file gives you both `move_group` **and** RViz, preloaded with
`iiwa7_moveit_config`'s own `moveit.rviz` config — which already includes the
**MotionPlanning** panel. You do not need to separately run
`lbr_bringup rviz.launch.py` or add the panel by hand; `rviz:=true` is
sufficient and is the pairing the stack itself is built around.

Everything comes up namespaced under `/lbr` by default (`robot_name:=lbr`);
`move_group.launch.py` remaps its planning-scene/robot-description topics
accordingly, so nothing else needs to change.

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
ros2 run tf2_ros tf2_echo lbr_link_0 lbr_link_ee
```

Compare against the pose you commanded. This is also exactly what
`flange_pose_publisher` republishes as `/iiwa/ee_pose` for the calibration
scripts to consume — if this echoes correctly, the calibration scripts will
see the same pose.

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
- `~/franka_ros2_ws/src/lbr_fri_ros2_stack/lbr_bringup/launch/` — the actual
  launch files referenced above (`hardware.launch.py`, `move_group.launch.py`,
  `mock.launch.py`), if you need to check or override their arguments.
- [lbr_fri_ros2_stack docs](https://lbr-stack.readthedocs.io/en/latest/lbr_fri_ros2_stack/lbr_fri_ros2_stack/doc/lbr_fri_ros2_stack.html#quick-start)
  — upstream quick-start, covers FRI setup and controller choices in more
  depth than this page does.
