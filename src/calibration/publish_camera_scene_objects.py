"""Publish the ZED cameras + their mounting holders as MoveIt CollisionObjects
on /planning_scene, using the calibrated extrinsics in
config/camera_extrinsics_base.yaml -- the same mechanism the tracking
pipeline uses for tracked parts (see run_pipeline_track_multicam.py's
_publish_planning_scene_object / docs/visualization.md), just for the fixed
camera rig instead of a moving tracked object.

Assets/ZED2.stl and Assets/zed_camer_holder.stl are the camera and holder
meshes. CAMERA_TO_HOLDER below is the measured camera-to-holder mount offset
(x=0.183081, y=0.046914, z=0.197348 m; roll=-2.617994, pitch=0.0,
yaw=3.141593 rad, i.e. R = Rz(yaw) @ Ry(pitch) @ Rx(roll), the same
fixed-axis roll/pitch/yaw convention as base_to_cams_calib_3.py's
_rpy_deg_to_R) -- the holder base frame's pose expressed in the camera
frame.

Only the zed2i_* entries of camera_extrinsics_base.yaml are used: those are
static camera-to-base transforms (dst = active robot's lbr_link_0, see
io_extrinsics.load_extrinsics_yaml), which is what "the calibration
procedure is done" means here -- realsense_1/realsense_2 are wrist-mounted
and move with the arm, so they have no fixed scene pose to publish.

Usage:
    python3 -m src.calibration.publish_camera_scene_objects

Then in RViz (see docs/visualization.md §2): Add -> PlanningScene, Planning
Scene Topic /planning_scene, Fixed Frame "world" -- the ZED2 + holder meshes
should appear at the calibrated zed2i_1/2/3 poses alongside tracked parts.

--dual-arm: camera_extrinsics_base.yaml's zed2i_* transforms are expressed
in config/robot_bases.yaml's active_robot frame (a single arm's lbr_link_0).
The single mock robot's URDF (iiwa7.xacro) roots itself at an actual "world"
link, fixed-jointed to lbr_link_0 -- so publishing poses in "world" as-is
works there. The dual-arm mock has NO "world" link at all: its URDF
(lbr_dual_arm.xacro) roots the kinematic tree at `base_link` (the midpoint
between robot_a/robot_b -- see src/utils/robot_bases.get_dual_arm_base_link),
and there is no static transform publishing a "world" frame either --
move_group genuinely has no concept of "world" there and logs
"Unknown frame: world" for every CollisionObject stamped with it. Pass
--dual-arm to (a) re-express every camera pose into the base_link frame,
exactly like run_pipeline_track_multicam_realsense.py's main() does for
tracked parts, and (b) default --frame-id to "base_link" instead of "world"
(matching lbr_dual_arm_moveit_config's own moveit.rviz Fixed Frame) -- this
is what lbr_dual_arm_bringup/launch/mock.launch.py (and, through it,
scripts/launch_moveit_scene_viewer.launch.py) always invokes.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import rclpy
from geometry_msgs.msg import Point, Pose
from moveit_msgs.msg import CollisionObject, PlanningScene, PlanningSceneWorld
from rclpy.node import Node
from shape_msgs.msg import Mesh, MeshTriangle
from std_msgs.msg import Header

import numpy as np

from src.calibration.io_extrinsics import load_extrinsics_yaml
from src.calibration.publish_extrinsics_tf import _rotation_matrix_to_quaternion_xyzw
from src.utils.robot_bases import get_active_robot_base, get_dual_arm_base_link
from src.utils.se3 import SE3

REPO_ROOT = Path(__file__).resolve().parents[2]


def _rpy_rad_to_R(roll: float, pitch: float, yaw: float) -> np.ndarray:
    # Fixed-axis roll/pitch/yaw, R = Rz(yaw) @ Ry(pitch) @ Rx(roll) -- same
    # convention as base_to_cams_calib_3.py's _rpy_deg_to_R (there in
    # degrees; here in radians, matching the measured holder offset below).
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)],
    ])
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)],
    ])
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1],
    ])
    return Rz @ Ry @ Rx


# Measured camera-to-holder mount offset -- see module docstring for the
# source x/y/z + roll/pitch/yaw values.
CAMERA_TO_HOLDER = SE3(
    R=_rpy_rad_to_R(roll=-2.617994, pitch=0.0, yaw=3.141593),
    t=np.array([0.183081, 0.046914, 0.197348]),
)

# Meshes are authored in millimeters (checked via each STL's bounding box,
# e.g. ZED2.stl is ~175mm long, matching the real ZED2's length) --
# CollisionObjects need meters.
DEFAULT_MESH_SCALE_MM_TO_M = 0.001


def _se3_to_pose_msg(T: SE3) -> Pose:
    msg = Pose()
    msg.position.x, msg.position.y, msg.position.z = (
        float(T.t[0]), float(T.t[1]), float(T.t[2])
    )
    qx, qy, qz, qw = _rotation_matrix_to_quaternion_xyzw(T.R)
    msg.orientation.x = float(qx)
    msg.orientation.y = float(qy)
    msg.orientation.z = float(qz)
    msg.orientation.w = float(qw)
    return msg


def _load_mesh_as_msg(mesh_path: str, mesh_scale: float) -> Mesh:
    """Load an STL/CAD mesh and convert it to shape_msgs/Mesh.

    Unlike run_pipeline_track_multicam.py's _load_mesh_as_msg, this does NOT
    recenter on the mesh centroid -- the camera/holder STLs' own origins are
    kept as authored, since there's no tracked-pose-estimation step here to
    match against.
    """
    import trimesh

    mesh = trimesh.load(mesh_path, force="mesh")
    if mesh_scale != 1.0:
        mesh.apply_scale(mesh_scale)

    msg = Mesh()
    msg.triangles = [
        MeshTriangle(vertex_indices=[int(i) for i in face])
        for face in mesh.faces
    ]
    msg.vertices = [
        Point(x=float(v[0]), y=float(v[1]), z=float(v[2])) for v in mesh.vertices
    ]
    return msg


def _make_collision_object(
    object_id: str, mesh: Mesh, pose: Pose, frame_id: str, stamp
) -> CollisionObject:
    obj = CollisionObject()
    obj.header = Header(frame_id=frame_id, stamp=stamp)
    obj.id = object_id
    obj.meshes = [mesh]
    obj.mesh_poses = [pose]
    obj.operation = CollisionObject.ADD
    return obj


class CameraScenePublisher(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("camera_scene_object_publisher")
        self.args = args
        self._pub = self.create_publisher(PlanningScene, "/planning_scene", 1)

        zed_mesh = _load_mesh_as_msg(args.zed_mesh, args.mesh_scale)
        holder_mesh = _load_mesh_as_msg(args.holder_mesh, args.mesh_scale)

        base_extr = load_extrinsics_yaml(args.base_yaml)
        if args.dual_arm:
            # Re-express every camera-to-active-robot-frame transform into
            # the dual-arm bringup's own base_link frame, same as
            # run_pipeline_track_multicam_realsense.py's main().
            _, T_robotA_activeRobot = get_active_robot_base()
            T_baseLink_robotA = get_dual_arm_base_link().inverse()
            T_baseLink_activeRobot = T_baseLink_robotA.compose(T_robotA_activeRobot)
            base_extr = {
                cam_id: T_baseLink_activeRobot.compose(T)
                for cam_id, T in base_extr.items()
            }

        self._objects: list[CollisionObject] = []
        stamp = self.get_clock().now().to_msg()
        for cam_id, T_base_camera in base_extr.items():
            T_base_holder = T_base_camera.compose(CAMERA_TO_HOLDER)
            self._objects.append(
                _make_collision_object(
                    f"zed_camera/{cam_id}", zed_mesh,
                    _se3_to_pose_msg(T_base_camera), args.frame_id, stamp,
                )
            )
            self._objects.append(
                _make_collision_object(
                    f"zed_camera_holder/{cam_id}", holder_mesh,
                    _se3_to_pose_msg(T_base_holder), args.frame_id, stamp,
                )
            )

        self.get_logger().info(
            f"Publishing {len(self._objects)} camera/holder CollisionObjects "
            f"for {list(base_extr.keys())} in frame '{args.frame_id}'"
        )
        # /planning_scene has no transient-local QoS, so a subscriber that
        # connects after the first publish (RViz opened late, move_group
        # restarted, ...) would otherwise never see these fixed-rig objects
        # -- republish periodically instead of once, same reasoning as the
        # tracking pipeline's continuous republish of tracked parts.
        self.create_timer(args.republish_period_s, self._publish)
        self._publish()

    def _publish(self) -> None:
        self._pub.publish(PlanningScene(is_diff=True, world=PlanningSceneWorld(collision_objects=self._objects)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-yaml",
        default=str(REPO_ROOT / "config" / "camera_extrinsics_base.yaml"),
        help="Static camera-to-base extrinsics (zed2i_* entries).",
    )
    parser.add_argument(
        "--zed-mesh",
        default=str(REPO_ROOT / "Assets" / "ZED2.stl"),
    )
    parser.add_argument(
        "--holder-mesh",
        default=str(REPO_ROOT / "Assets" / "zed_camer_holder.stl"),
    )
    parser.add_argument("--mesh-scale", type=float, default=DEFAULT_MESH_SCALE_MM_TO_M)
    parser.add_argument(
        "--frame-id",
        default=None,
        help="Fixed frame_id for the CollisionObjects, matching "
        "run_pipeline_track_multicam.py's --planning-scene-frame-id default. "
        "Defaults to 'world', or 'base_link' with --dual-arm (see module "
        "docstring) -- pass this explicitly to override either.",
    )
    parser.add_argument("--republish-period-s", type=float, default=2.0)
    parser.add_argument(
        "--dual-arm",
        action="store_true",
        help="Re-express camera poses into the dual-arm bringup's base_link "
        "frame instead of publishing them as-is in the active robot's own "
        "frame, and default --frame-id to that same base_link frame -- see "
        "module docstring.",
    )
    args = parser.parse_args()
    if args.frame_id is None:
        args.frame_id = "base_link" if args.dual_arm else "world"

    rclpy.init()
    node = CameraScenePublisher(args)
    rclpy.spin(node)


if __name__ == "__main__":
    main()
