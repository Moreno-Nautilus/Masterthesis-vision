# Franka first grasp test checklist

This checklist is for the first end-to-end test of the perception-to-robot grasp pipeline with the Franka robot, the Cartesian impedance controller, the pb_pipe object, and the manual bridge triggers.

Assumptions:
- The trusted pose is the raw base-frame pose from perception.
- The object used is pb_pipe.
- The object stands upright on the table.
- The bridge uses the object centroid plus fixed offsets for side grasping.
- The Cartesian impedance controller is already installed and registered in `franka_bringup/config/controllers.yaml`.

Before starting:
- Confirm the robot IP.
- Confirm the robot workspace exists and builds.
- Confirm `/opt/ros/humble` can be sourced.
- Confirm `~/franka_ros2_ws/install/setup.bash` exists.
- Confirm `cartesian_impedance_controller` is listed in the Franka controllers config.
- Confirm the perception pipeline publishes the trusted raw base pose topic for pb_pipe.
- Confirm the bridge file exists at `src/scripts/franka_pb_pipe_bridge.py`.
- Confirm the bridge input topic matches the actual raw base pose topic.
- Confirm `dry_run:=false` only when you are ready for real motion.
- Confirm the robot workspace and your thesis workspace are the ones you intend to use.
- Confirm the object is placed upright and isolated in the scene.
- Confirm you know where the emergency stop is and keep the first motions conservative.

## Terminal 1
```bash
source /opt/ros/humble/setup.bash
source ~/franka_ros2_ws/install/setup.bash

## Terminal 2
source /opt/ros/humble/setup.bash
source ~/franka_ros2_ws/install/setup.bash
ros2 launch franka_bringup franka.launch.py robot_ip:=<ROBOT_IP> arm_id:=fr3 load_gripper:=true use_rviz:=false
## Terminal 3
source /opt/ros/humble/setup.bash
source ~/franka_ros2_ws/install/setup.bash
ros2 run controller_manager spawner cartesian_impedance_controller
## Terminal 4
source /opt/ros/humble/setup.bash
source ~/franka_ros2_ws/install/setup.bash
ros2 control list_controllers
ros2 service list | grep set_pose

## Terminal 5
source /opt/ros/humble/setup.bash
source ~/franka_ros2_ws/install/setup.bash
ros2 topic echo /perception/fp/pose_base/zed2i_2/pb_pipe_raw_0 --once

## terminal 6
source /opt/ros/humble/setup.bash
source ~/franka_ros2_ws/install/setup.bash
python3 src/scripts/franka_pb_pipe_bridge.py --ros-args \
  -p input_pose_topic:=/perception/fp/pose_base/zed2i_2/pb_pipe_raw_0 \
  -p object_name:=pb_pipe \
  -p object_diameter_m:=0.046 \
  -p approach_axis:=x \
  -p approach_sign:=-1.0 \
  -p pregrasp_clearance_m:=0.080 \
  -p grasp_clearance_m:=0.020 \
  -p lift_delta_z_m:=0.080 \
  -p dry_run:=false

  source /opt/ros/humble/setup.bash
source ~/franka_ros2_ws/install/setup.bash
ros2 topic list | grep franka_pb_pipe_bridge
ros2 topic echo /franka_pb_pipe_bridge/stage

## Terminal 7
Reset:

ros2 service call /franka_pb_pipe_bridge/reset_sequence std_srvs/srv/Trigger "{}"

Pregrasp:

ros2 service call /franka_pb_pipe_bridge/go_pregrasp std_srvs/srv/Trigger "{}"

Neutral:

ros2 service call /franka_pb_pipe_bridge/go_neutral std_srvs/srv/Trigger "{}"

Grasp:

ros2 service call /franka_pb_pipe_bridge/go_grasp std_srvs/srv/Trigger "{}"

Lift:

ros2 service call /franka_pb_pipe_bridge/go_lift std_srvs/srv/Trigger "{}"
