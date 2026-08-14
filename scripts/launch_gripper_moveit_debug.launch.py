"""Debug-only viewer, same gripper-alone model as
launch_gripper_debug.launch.py, but with move_group up too so RViz's
MotionPlanning display (planning scene, group state playback, "Query Goal
State" interactive marker) actually works, instead of just RobotModel/TF.

No controllers, no execution -- Planning Scene Topic and the "lbr_one_gripper"
planning group (see y_gripper_standalone.srdf, its own minimal SRDF since the
dual-arm's own SRDF references links this standalone model doesn't have) are
enough to drive the MotionPlanning panel's display and interactive markers.

Usage:
    ros2 launch /home/pdzuser/Masterthesis-vision/scripts/launch_gripper_moveit_debug.launch.py

`ros2 launch` always treats its first argument as a package name, not a
path -- needs an absolute path like above (see
scripts/launch_moveit_scene_viewer.launch.py's docstring).
"""
import os

from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder

# Source-tree paths (not the ROS package share dir) so this renders directly
# from what's on disk with no colcon build step needed.
GRIPPER_STANDALONE_XACRO = (
    "/home/pdzuser/franka_ros2_ws/src/lbr_fri_ros2_stack/lbr_demos/lbr_dual_arm/"
    "lbr_dual_arm_description/urdf/y_gripper_standalone.xacro"
)
GRIPPER_STANDALONE_SRDF = (
    "/home/pdzuser/franka_ros2_ws/src/lbr_fri_ros2_stack/lbr_demos/lbr_dual_arm/"
    "lbr_dual_arm_description/urdf/y_gripper_standalone.srdf"
)

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# lbr_one's current mount, matching lbr_dual_arm.xacro -- fixed, not a
# runtime launch arg, since MoveItConfigsBuilder.robot_description()
# resolves xacro mappings immediately (at generate_launch_description() time,
# not through the launch substitution system), so there's no clean way to
# also expose this as a `mount_yaw:=` CLI override here the way the non-MoveIt
# viewer does. Use launch_gripper_debug.launch.py for side-by-side mount_yaw
# comparisons.
MOUNT_YAW = "3.14159265358979"


def generate_launch_description() -> LaunchDescription:
    # planning_pipelines(...) still pulls its ompl planner config from
    # lbr_dual_arm_moveit_config -- robot_description/robot_description_semantic
    # are overridden below to point at the standalone gripper model instead of
    # the dual-arm one, same pattern launch_moveit_scene_viewer.launch.py uses.
    moveit_config = (
        MoveItConfigsBuilder("y_gripper_standalone", package_name="lbr_dual_arm_moveit_config")
        .robot_description(GRIPPER_STANDALONE_XACRO, mappings={"mount_yaw": MOUNT_YAW})
        .robot_description_semantic(file_path=GRIPPER_STANDALONE_SRDF)
        .planning_pipelines(default_planning_pipeline="ompl", pipelines=["ompl"])
        .to_moveit_configs()
    )

    move_group_params = dict(moveit_config.to_dict())
    move_group_params["publish_robot_description_semantic"] = True

    return LaunchDescription([
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            parameters=[{"robot_description": moveit_config.robot_description["robot_description"]}],
        ),
        Node(
            package="moveit_ros_move_group",
            executable="move_group",
            output="screen",
            parameters=[move_group_params],
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            arguments=["-d", os.path.join(REPO_DIR, "scripts", "gripper_moveit_debug.rviz")],
            parameters=[
                moveit_config.robot_description,
                moveit_config.robot_description_semantic,
                moveit_config.planning_pipelines,
                moveit_config.robot_description_kinematics,
            ],
        ),
    ])
