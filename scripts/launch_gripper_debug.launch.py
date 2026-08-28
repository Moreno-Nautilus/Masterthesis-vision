"""Debug-only viewer: just the Y-gripper mounted on a stand-in flange link
(lbr_one_link_7's real mesh, for a like-for-like silhouette), no arm, no
move_group, no controllers -- for inspecting the gripper mount rotation
(y_gripper.xacro's mount_yaw arg) and mesh clearance against the flange in
isolation, without booting the whole dual-arm mock stack.

Usage:
    ros2 launch /home/pdzuser/Masterthesis-vision/scripts/launch_gripper_debug.launch.py
    ros2 launch /home/pdzuser/Masterthesis-vision/scripts/launch_gripper_debug.launch.py mount_yaw:=0.0
    ros2 launch /home/pdzuser/Masterthesis-vision/scripts/launch_gripper_debug.launch.py mount_yaw:=3.14159265358979

mount_yaw defaults to pi (matching lbr_one's current mount in
lbr_dual_arm.xacro) so this reproduces exactly what the full dual-arm robot
shows; pass mount_yaw:=0.0 to see the un-rotated (lbr_two-style) mount for
comparison.

`ros2 launch` always treats its first argument as a package name, not a
path -- a bare relative path fails with "is not a valid package name", so
this needs an absolute path like above (matches
scripts/launch_moveit_scene_viewer.launch.py's own docstring note).
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, FindExecutable, LaunchConfiguration
from launch_ros.actions import Node

# Source-tree path (not the ROS package share dir) so this renders directly
# from what's on disk with no colcon build step needed -- its own
# $(find lbr_dual_arm_description) include of y_gripper.xacro still resolves
# via the installed/symlinked package share, which is unaffected by this.
GRIPPER_STANDALONE_XACRO = (
    "/home/pdzuser/franka_ros2_ws/src/lbr_fri_ros2_stack/lbr_demos/lbr_dual_arm/"
    "lbr_dual_arm_description/urdf/y_gripper_standalone.xacro"
)

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def generate_launch_description() -> LaunchDescription:
    ld = LaunchDescription()

    ld.add_action(
        DeclareLaunchArgument(
            name="mount_yaw",
            default_value="3.14159265358979",
            description="Gripper mount joint yaw (radians) about the flange Z axis.",
        )
    )

    robot_description = {
        "robot_description": Command(
            [
                FindExecutable(name="xacro"),
                " ",
                GRIPPER_STANDALONE_XACRO,
                " mount_yaw:=",
                LaunchConfiguration("mount_yaw"),
            ]
        )
    }

    ld.add_action(
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            parameters=[robot_description],
        )
    )

    # Publishes the two prismatic finger joints (open/close) with sliders --
    # they're not driven by any controller here, so without this
    # robot_state_publisher never gets a /joint_states for them and their
    # links just never appear in TF.
    ld.add_action(
        Node(
            package="joint_state_publisher_gui",
            executable="joint_state_publisher_gui",
        )
    )

    ld.add_action(
        Node(
            package="rviz2",
            executable="rviz2",
            arguments=["-d", os.path.join(REPO_DIR, "scripts", "gripper_debug.rviz")],
        )
    )

    return ld
