"""Mock dual-arm bringup + move_group + RViz, all under the /lbr_dual_arm
namespace, so RViz's PlanningScene display can render the CollisionObjects
published by our camera pipeline (run_pipeline_track_multicam*.py) on
/planning_scene.

This exists only to *view* the tracked parts - no hardware, no real motion
planning. move_group is required so RViz has a base (non-diff) scene to
merge our ADD/REMOVE CollisionObject diffs onto; without it RViz reports
"no planning scene loaded" even though the topic is publishing correctly.

Reuses lbr_dual_arm_bringup's own mock.launch.py for the robot itself
(robot_state_publisher + ros2_control + controllers + the Y-gripper on each
flange by default, plus its already-wired --dual-arm camera-rig publisher)
instead of duplicating that setup for a single mock arm -- this is a pure
viewer, so there's no reason to maintain a separate single-arm mock config
alongside the real dual-arm one.

lbr_dual_arm_bringup's own move_group.launch.py doesn't expose a way to
inject remappings into its internal move_group Node, and namespacing
move_group/RViz under /lbr_dual_arm (matching the mock robot's own
namespace) remaps /planning_scene et al. to /lbr_dual_arm/planning_scene,
disconnecting move_group from the bare /planning_scene topic our camera
pipeline publishes on -- so move_group/RViz are built here as raw Node
actions instead, with those topics remapped back explicitly (same pattern
this file used for the single-arm mock before it was folded into the
dual-arm setup).

WARNING: don't "simplify" this file by swapping the manual move_group/RViz
Nodes below for an IncludeLaunchDescription of lbr_dual_arm_bringup's own
move_group.launch.py -- that's not leftover duplication, it's the one piece
that has to stay custom. Without the explicit remap, move_group/RViz would
silently listen on /lbr_dual_arm/planning_scene instead of /planning_scene,
and the pipeline's CollisionObjects (and publish_camera_scene_objects') would
just never show up in RViz, with no error anywhere to point at why.

Usage:
    ros2 launch /home/pdzuser/Masterthesis-vision/scripts/launch_moveit_scene_viewer.launch.py

Then in the pipeline's own terminal (docker exec into `vision`), run the
camera pipeline as usual; tracked parts should appear in RViz once you
Add -> moveit_ros_visualization -> PlanningScene and set its
"Planning Scene Topic" to /planning_scene (Fixed Frame is already "world",
matching --planning-scene-frame-id's default).

This also broadcasts config/camera_extrinsics_base.yaml and
config/camera_extrinsics_realsense.yaml as static TF frames (see
src/calibration/publish_extrinsics_tf.py) hanging off whichever arm is
currently config/robot_bases.yaml's active_robot (robot_a -> lbr_one,
robot_b -> lbr_two), so Add -> TF in RViz shows the calibrated camera poses
(zed2i_1/2/3, realsense_1/2) alongside the robot.

The ZED cameras + their mounting holders (see
src/calibration/publish_camera_scene_objects.py, Assets/ZED2.stl,
Assets/zed_camer_holder.stl) are already published as CollisionObjects on
/planning_scene by lbr_dual_arm_bringup's mock.launch.py itself (it starts
publish_camera_scene_objects --dual-arm alongside the mock robot) -- these
show up in the same PlanningScene display as the tracked parts, alongside
the TF frames above.
"""
import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROBOT_NAME = "lbr_dual_arm"

# robot_bases.yaml's active_robot -> the dual-arm xacro's per-arm link
# prefix (robot_a == left == lbr_one, robot_b == right == lbr_two -- see
# src/calibration/flange_pose_store.py's ARM_KEYS).
ACTIVE_ROBOT_TO_ARM_PREFIX = {"robot_a": "lbr_one", "robot_b": "lbr_two"}


def _active_arm_link_prefix() -> str:
    robot_bases_yaml = os.path.join(REPO_DIR, "config", "robot_bases.yaml")
    with open(robot_bases_yaml) as f:
        active_robot = yaml.safe_load(f)["active_robot"]
    return ACTIVE_ROBOT_TO_ARM_PREFIX[active_robot]


def generate_launch_description() -> LaunchDescription:
    lbr_dual_arm_bringup_share = get_package_share_directory("lbr_dual_arm_bringup")

    # Mock dual-arm robot (robot_state_publisher + ros2_control + controllers
    # + Y-gripper on each flange by default), namespaced under /lbr_dual_arm
    # internally, plus its own --dual-arm publish_camera_scene_objects.
    mock_robot = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(lbr_dual_arm_bringup_share, "launch", "mock.launch.py")
        ),
        launch_arguments={"robot_name": ROBOT_NAME}.items(),
    )

    moveit_config = (
        MoveItConfigsBuilder("lbr_dual_arm", package_name="lbr_dual_arm_moveit_config")
        .robot_description(
            os.path.join(
                get_package_share_directory("lbr_dual_arm_description"),
                "urdf/lbr_dual_arm.xacro",
            ),
            mappings={"mode": "mock", "use_gripper": "true"},
        )
        .robot_description_semantic(
            file_path=os.path.join(
                get_package_share_directory("lbr_dual_arm_moveit_config"),
                "config",
                "lbr_dual_arm.srdf.xacro",
            ),
            mappings={"use_gripper": "true"},
        )
        .planning_pipelines(default_planning_pipeline="ompl", pipelines=["ompl"])
        .to_moveit_configs()
    )

    move_group_params = dict(moveit_config.to_dict())
    move_group_params["publish_robot_description_semantic"] = True

    # namespace=ROBOT_NAME would also remap /planning_scene to
    # /lbr_dual_arm/planning_scene, disconnecting move_group from the bare
    # /planning_scene topic our camera pipeline publishes CollisionObjects
    # on - remap it back explicitly so move_group's robot-state
    # topics/services stay under /lbr_dual_arm but it still monitors the
    # same /planning_scene topic the pipeline uses.
    move_group = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        namespace=ROBOT_NAME,
        output="screen",
        parameters=[move_group_params],
        remappings=[
            (f"/{ROBOT_NAME}/planning_scene", "/planning_scene"),
            (f"/{ROBOT_NAME}/monitored_planning_scene", "/monitored_planning_scene"),
            (f"/{ROBOT_NAME}/planning_scene_world", "/planning_scene_world"),
            (f"/{ROBOT_NAME}/collision_object", "/collision_object"),
            (f"/{ROBOT_NAME}/attached_collision_object", "/attached_collision_object"),
        ],
    )

    rviz_config = os.path.join(
        get_package_share_directory("lbr_dual_arm_moveit_config"),
        "config",
        "moveit.rviz",
    )
    # Deliberately NOT namespace=ROBOT_NAME: moveit.rviz's MotionPlanning
    # display already has "Move Group Namespace: lbr_dual_arm" baked in, so
    # it prefixes its own move_group-facing names (get_planning_scene,
    # move_action, compute_cartesian_path, attached_collision_object, ...)
    # with "lbr_dual_arm/" itself. Namespacing this Node on top of that
    # double-prefixes those names to /lbr_dual_arm/lbr_dual_arm/... which
    # don't exist -- move_group only serves the single-prefixed ones. That
    # double-prefixing is exactly what caused RViz's "Requesting initial
    # scene failed" (confirmed live: RViz was calling
    # /lbr_dual_arm/lbr_dual_arm/get_planning_scene, which doesn't exist).
    #
    # Its "Planning Scene Topic" (monitored_planning_scene) is NOT run
    # through that Move Group Namespace prefixing, so left un-namespaced it
    # resolves to the bare /monitored_planning_scene -- matching move_group's
    # own remap of that topic above (needed for the camera pipeline). With
    # namespace=ROBOT_NAME it resolved to /lbr_dual_arm/monitored_planning_scene
    # instead, which move_group never published to, so RViz's scene display
    # silently went stale relative to the live TF -- "TF and visualization do
    # not match PlanningScene".
    #
    # The one thing that *did* depend on the node namespace was robot model
    # loading (robot_description / robot_description_semantic, published by
    # robot_state_publisher/move_group under /lbr_dual_arm/...), so those are
    # remapped explicitly instead of relying on namespacing for them too.
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", rviz_config],
        parameters=[
            moveit_config.planning_pipelines,
            moveit_config.robot_description_kinematics,
        ],
        remappings=[
            ("robot_description", f"/{ROBOT_NAME}/robot_description"),
            ("robot_description_semantic", f"/{ROBOT_NAME}/robot_description_semantic"),
        ],
    )

    # Not an installed ROS package executable, so run it as a plain module
    # via ExecuteProcess instead of launch_ros's Node action.
    #
    # zed2i_* are calibrated against the active arm's link_0 specifically
    # (see config/camera_extrinsics_base.yaml), so --base-frame stays
    # single/active-arm. realsense_1/realsense_2 are a flange-mount offset
    # only, so the same calibrated offset is republished under both arms'
    # link_ee here (--ee-frame given twice), assuming both arms carry an
    # identical wrist-camera mount.
    extrinsics_tf = ExecuteProcess(
        cmd=[
            "python3", "-m", "src.calibration.publish_extrinsics_tf",
            "--base-frame", f"{_active_arm_link_prefix()}_link_0",
            "--ee-frame", "lbr_one_link_ee",
            "--ee-frame", "lbr_two_link_ee",
        ],
        cwd=REPO_DIR,
        output="screen",
    )

    return LaunchDescription([mock_robot, move_group, rviz, extrinsics_tf])
