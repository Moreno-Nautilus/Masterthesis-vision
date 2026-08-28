"""Fixed camera set for the 1-ZED + 2-RealSense trio variant.

Extracted from run_pipeline_track_multicam_realsense.py.
"""
from __future__ import annotations

from src.perception.ros.multicam_grabber_realsense import DynamicCameraTopics


# Fixed camera set for this variant: one static tripod ZED (cam1) plus two
# end-effector-mounted RealSense cameras (cam2, cam3). Unlike the original
# ALL_CAMERAS[:num_cameras] prefix-selection scheme, this set is not sliced
# by --num-cameras — it is always exactly these three, in this order, so it
# lines up 1:1 with the existing cam1_*/cam2_*/cam3_* per-camera CLI args.
#
# RealSense topic names follow the realsense2_camera ROS2 wrapper's default
# namespacing (camera_namespace/camera_name from the launch file). Depth is
# aligned to the color frame (align_depth.enable:=true in the launch file)
# so depth and RGB share the same intrinsics/resolution, same as the ZED's
# depth_registered topic.
ALL_CAMERAS: list[DynamicCameraTopics] = [
    DynamicCameraTopics(
        cam_id="zed2i_1",
        depth_topic="/zed2i_1/zed_node/depth/depth_registered",
        info_topic="/zed2i_1/zed_node/depth/depth_registered/camera_info",
        rgb_topic="/zed2i_1/zed_node/rgb/color/rect/image",
        rgb_info_topic="/zed2i_1/zed_node/rgb/color/rect/image/camera_info",
        is_dynamic=False,
    ),
    # realsense_1 is bolted to the LEFT arm (port_id 30200 -- see
    # lbr_dual_arm_description/ros2_control/lbr_one_system_config.yaml --
    # and robot_base_key="robot_a" in config/robot_bases.yaml).
    DynamicCameraTopics(
        cam_id="realsense_1",
        # /image_rect (not /image_raw): rectified by the per-camera
        # image_proc RectifyNode pair started in zed_realsense_trio.launch.py
        # (see comment there). camera_info topics are unchanged — the D405's
        # raw camera_info already carries P == K (Tx=Ty=0), which is what
        # RectifyNode uses as the rectified image's intrinsics, so the
        # original (distorted) camera_info's K is still the correct K to
        # read for the rectified image.
        depth_topic="/realsense_1/camera/aligned_depth_to_color/image_rect",
        info_topic="/realsense_1/camera/aligned_depth_to_color/camera_info",
        rgb_topic="/realsense_1/camera/color/image_rect",
        rgb_info_topic="/realsense_1/camera/color/camera_info",
        is_dynamic=True,
        flange_pose_topic="/left/ee_pose",
        robot_base_key="robot_a",
    ),
    # realsense_2 is bolted to the RIGHT arm (port_id 30201, robot_base_key
    # "robot_b"). This is the camera that already has a real hand-eye
    # calibration in camera_extrinsics_realsense.yaml, taken with
    # active_robot: robot_b.
    DynamicCameraTopics(
        cam_id="realsense_2",
        # See realsense_1's comment above — same /image_rect + unchanged
        # camera_info rationale.
        depth_topic="/realsense_2/camera/aligned_depth_to_color/image_rect",
        info_topic="/realsense_2/camera/aligned_depth_to_color/camera_info",
        rgb_topic="/realsense_2/camera/color/image_rect",
        rgb_info_topic="/realsense_2/camera/color/camera_info",
        is_dynamic=True,
        flange_pose_topic="/right/ee_pose",
        robot_base_key="robot_b",
    ),
]


def select_cameras(num_cameras: int) -> list[DynamicCameraTopics]:
    """Active camera set for this run.

    Kept for interface parity with the original runner (FoundationPoseTrackerNode
    calls select_cameras(args.num_cameras)), but this variant always runs all
    three cameras in ALL_CAMERAS regardless of num_cameras.
    """
    if num_cameras != len(ALL_CAMERAS):
        raise ValueError(
            f"run_pipeline_track_multicam_realsense fixes the camera set to "
            f"{len(ALL_CAMERAS)} cameras (1 ZED + 2 RealSense); got "
            f"--num-cameras {num_cameras}"
        )
    return list(ALL_CAMERAS)
