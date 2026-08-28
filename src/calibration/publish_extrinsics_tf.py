"""Broadcast config/camera_extrinsics_*.yaml as static TF frames, so they
show up next to the robot's own tf tree in RViz (e.g. alongside
scripts/launch_moveit_scene_viewer.launch.py) instead of only being usable
as raw numbers.

zed2i_* entries are published as children of --base-frame (their dst frame,
see io_extrinsics.load_extrinsics_yaml docstring) -- only ever one parent,
since they're external/room-mounted cameras calibrated against one specific
arm's link_0, not something that needs duplicating per arm. realsense_1/
realsense_2 are published as children of --ee-frame, since those entries are
a static camera-to-flange mount offset, not a camera-to-base transform -- see
config/camera_extrinsics_realsense.yaml's header comment.

--ee-frame may be passed more than once (e.g. for a dual-arm rig where both
arms carry an identical wrist-camera mount): the same calibrated
flange-to-camera offset is then republished once per given ee-frame, with
each pair of child frames tagged by arm name (e.g. lbr_one_realsense_1,
lbr_two_realsense_1) instead of the bare cam_id, so they don't collide.

realsense_2 has a second, CAD-derived entry in that same YAML,
realsense_2_initial_guess_cad (a reference-only T_flange_cam computed from
the mount's CAD, not from hand-eye calibration). By default this script
publishes THAT one as the "realsense_2" TF frame -- in place of the
calibrated entry -- so it can be sanity-checked against the gripper/robot in
RViz. Pass --realsense2-source calibrated to publish the real hand-eye
result instead.

Usage:
    python3 -m src.calibration.publish_extrinsics_tf
    python3 -m src.calibration.publish_extrinsics_tf --base-frame lbr_link_0 --ee-frame lbr_link_ee
    python3 -m src.calibration.publish_extrinsics_tf \\
        --base-frame lbr_one_link_0 --ee-frame lbr_one_link_ee --ee-frame lbr_two_link_ee
    python3 -m src.calibration.publish_extrinsics_tf --realsense2-source calibrated

Then in RViz: Add -> TF, and look for zed2i_1/zed2i_2/zed2i_3 hanging off
--base-frame, and realsense_1/realsense_2 (or their arm-tagged names, if
--ee-frame was given more than once) hanging off --ee-frame.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster

from src.calibration.io_extrinsics import load_extrinsics_yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

# cam_id -> which robot frame its extrinsics entry is expressed relative to.
# realsense_nominal is the CAD-derived (not hand-eye calibrated) mount
# offset -- see config/camera_extrinsics_realsense.yaml's header comment.
FLANGE_CAM_IDS = {"realsense_1", "realsense_2", "realsense_nominal"}

# Suffix marking a YAML entry as a reference-only CAD guess for the cam_id
# it's named after (e.g. realsense_2_initial_guess_cad -> realsense_2). Never
# published as its own TF frame -- only swapped in for its named cam_id.
CAD_INITIAL_GUESS_SUFFIX = "_initial_guess_cad"


def _rotation_matrix_to_quaternion_xyzw(R: np.ndarray) -> np.ndarray:
    # Same Shepperd's-method implementation as
    # run_pipeline_track_multicam.py's rotation_matrix_to_quaternion_xyzw --
    # duplicated (not imported) to avoid pulling the whole pipeline runner
    # module in for one helper, and to skip scipy, whose compiled extension
    # is ABI-incompatible with the numpy build in this ROS env.
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    q = np.empty(4, dtype=np.float64)
    trace = np.trace(R)
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        q[3] = 0.25 * s
        q[0] = (R[2, 1] - R[1, 2]) / s
        q[1] = (R[0, 2] - R[2, 0]) / s
        q[2] = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        q[3] = (R[2, 1] - R[1, 2]) / s
        q[0] = 0.25 * s
        q[1] = (R[0, 1] + R[1, 0]) / s
        q[2] = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        q[3] = (R[0, 2] - R[2, 0]) / s
        q[0] = (R[0, 1] + R[1, 0]) / s
        q[1] = 0.25 * s
        q[2] = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        q[3] = (R[1, 0] - R[0, 1]) / s
        q[0] = (R[0, 2] + R[2, 0]) / s
        q[1] = (R[1, 2] + R[2, 1]) / s
        q[2] = 0.25 * s
    return q / (np.linalg.norm(q) + 1e-12)


def _make_transform(
    parent_frame: str, child_frame: str, R, t, stamp
) -> TransformStamped:
    msg = TransformStamped()
    msg.header.stamp = stamp
    msg.header.frame_id = parent_frame
    msg.child_frame_id = child_frame
    msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z = (
        float(t[0]), float(t[1]), float(t[2])
    )
    qx, qy, qz, qw = _rotation_matrix_to_quaternion_xyzw(R)
    msg.transform.rotation.x = float(qx)
    msg.transform.rotation.y = float(qy)
    msg.transform.rotation.z = float(qz)
    msg.transform.rotation.w = float(qw)
    return msg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-yaml",
        default=str(REPO_ROOT / "config" / "camera_extrinsics_base.yaml"),
        help="Static camera-to-base extrinsics (zed2i_* entries).",
    )
    parser.add_argument(
        "--realsense-yaml",
        default=str(REPO_ROOT / "config" / "camera_extrinsics_realsense.yaml"),
        help="Wrist-camera-to-flange + zed2i_1 extrinsics.",
    )
    parser.add_argument(
        "--base-frame",
        default="lbr_link_0",
        help="Parent frame for camera-to-base entries (active robot's base).",
    )
    parser.add_argument(
        "--ee-frame",
        action="append",
        default=None,
        help="Parent frame for camera-to-flange entries (realsense_1/realsense_2). "
        "May be given more than once to republish the same calibrated offset "
        "under multiple arms' flanges (assumes an identical camera mount on "
        "each); default is a single 'lbr_link_ee'.",
    )
    parser.add_argument(
        "--realsense2-source",
        choices=["cad_initial_guess", "calibrated"],
        default="cad_initial_guess",
        help="Which T_flange_cam to publish as the realsense_2 TF frame: the "
        "CAD-derived nominal mount transform (realsense_2_initial_guess_cad) "
        "or the real hand-eye calibration result (realsense_2). Defaults to "
        "the CAD guess.",
    )
    args = parser.parse_args()

    rclpy.init()
    node = Node("extrinsics_tf_publisher")
    broadcaster = StaticTransformBroadcaster(node)
    stamp = node.get_clock().now().to_msg()

    transforms: list[TransformStamped] = []

    base_extr = load_extrinsics_yaml(args.base_yaml)
    for cam_id, T in base_extr.items():
        transforms.append(
            _make_transform(args.base_frame, cam_id, T.R, T.t, stamp)
        )

    ee_frames = args.ee_frame or ["lbr_link_ee"]

    realsense_extr = load_extrinsics_yaml(args.realsense_yaml)
    for cam_id, T in realsense_extr.items():
        if cam_id.endswith(CAD_INITIAL_GUESS_SUFFIX):
            # Not published under its own name -- only swapped in below, in
            # place of the cam_id it's a guess for.
            continue
        if cam_id in base_extr:
            # zed2i_1 is duplicated across both files with the same meaning;
            # camera_extrinsics_base.yaml's copy already published above.
            continue
        if cam_id == "realsense_2" and args.realsense2_source == "cad_initial_guess":
            guess_key = cam_id + CAD_INITIAL_GUESS_SUFFIX
            if guess_key in realsense_extr:
                T = realsense_extr[guess_key]
        if cam_id not in FLANGE_CAM_IDS:
            transforms.append(_make_transform(args.base_frame, cam_id, T.R, T.t, stamp))
            continue
        for ee_frame in ee_frames:
            # Tag the child frame by arm when publishing under more than one
            # flange, so e.g. lbr_one's and lbr_two's realsense_1 don't both
            # try to claim the same TF child frame name.
            child_frame = (
                cam_id if len(ee_frames) == 1
                else f"{ee_frame.removesuffix('_link_ee')}_{cam_id}"
            )
            transforms.append(_make_transform(ee_frame, child_frame, T.R, T.t, stamp))

    broadcaster.sendTransform(transforms)
    node.get_logger().info(
        f"Published {len(transforms)} static camera TF frames: "
        f"{[t.child_frame_id for t in transforms]}"
    )
    rclpy.spin(node)


if __name__ == "__main__":
    main()
