"""Mock KUKA iiwa7 + move_group + RViz, all under the /lbr namespace, so
RViz's PlanningScene display can render the CollisionObjects published by
our camera pipeline (run_pipeline_track_multicam*.py) on /planning_scene.

This exists only to *view* the tracked parts - no hardware, no real motion
planning. move_group is required so RViz has a base (non-diff) scene to
merge our ADD/REMOVE CollisionObject diffs onto; without it RViz reports
"no planning scene loaded" even though the topic is publishing correctly.

lbr_bringup's own move_group.launch.py / rviz.launch.py don't expose a
namespace argument, and mock.launch.py's robot already runs everything
under /lbr - so move_group and RViz are built here as raw Node actions
with namespace="lbr" instead of including those launch files.

Usage:
    ros2 launch /home/pdzuser/Masterthesis-vision/scripts/launch_moveit_scene_viewer.launch.py

Then in the pipeline's own terminal (docker exec into `vision`), run the
camera pipeline as usual; tracked parts should appear in RViz once you
Add -> moveit_ros_visualization -> PlanningScene and set its
"Planning Scene Topic" to /planning_scene (Fixed Frame is already "world",
matching --planning-scene-frame-id's default).
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description() -> LaunchDescription:
    lbr_bringup_share = get_package_share_directory("lbr_bringup")

    # Mock robot (robot_state_publisher + ros2_control + controllers), all
    # namespaced under /lbr internally by lbr_bringup's own launch file.
    mock_robot = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(lbr_bringup_share, "launch", "mock.launch.py")
        ),
        launch_arguments={"model": "iiwa7"}.items(),
    )

    moveit_config = (
        MoveItConfigsBuilder("iiwa7", package_name="iiwa7_moveit_config")
        .robot_description(
            os.path.join(
                get_package_share_directory("lbr_description"),
                "urdf/iiwa7/iiwa7.xacro",
            )
        )
        .to_moveit_configs()
    )

    move_group_params = dict(moveit_config.to_dict())
    move_group_params["publish_robot_description_semantic"] = True

    # namespace="lbr" would also remap /planning_scene to /lbr/planning_scene,
    # disconnecting move_group from the bare /planning_scene topic our camera
    # pipeline publishes CollisionObjects on - remap it back explicitly so
    # move_group's robot-state topics/services stay under /lbr but it still
    # monitors the same /planning_scene topic the pipeline uses.
    move_group = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        namespace="lbr",
        output="screen",
        parameters=[move_group_params],
        remappings=[
            ("/lbr/planning_scene", "/planning_scene"),
            ("/lbr/monitored_planning_scene", "/monitored_planning_scene"),
            ("/lbr/planning_scene_world", "/planning_scene_world"),
            ("/lbr/collision_object", "/collision_object"),
            ("/lbr/attached_collision_object", "/attached_collision_object"),
        ],
    )

    rviz_config = os.path.join(lbr_bringup_share, "config", "mock.rviz")
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        namespace="lbr",
        arguments=["-d", rviz_config],
    )

    return LaunchDescription([mock_robot, move_group, rviz])
