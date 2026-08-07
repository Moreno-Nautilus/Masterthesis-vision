"""Automatic dual-arm board-pose + ZED calibration -- runs AFTER hand-eye
calibration is already done (see src/calibration/capture_handeye_data.py +
calibrate_handeye.py; docs/calibration_cheatsheet.md for the full
walkthrough).

This script used to also solve the hand-eye stage itself (driving both arms
via MoveIt to replay saved poses, capturing + solving T_flange_cam inline).
That responsibility moved out: capture_handeye_data.py's --mode replay is
now the way to (re)capture hand-eye images against previously-saved poses,
and calibrate_handeye.py is the way to solve them (--method direct or
--method joint) -- doing both hand-eye capture and hand-eye solving inline
here was duplicate logic once those two scripts existed. This script now
starts from config/camera_extrinsics_realsense.yaml already having a real
T_flange_cam for both wrist cameras, and requires that (see
_require_handeye_done below) rather than re-deriving it.

Two stages, run in order, each gated on the previous succeeding:

  Board pose -- checkerboard pose in the robot base frame (T_base_board).
    Uses the LAST `--num-board-poses` (default 2) saved captures per arm
    from config/flange_poses/<arm>.json, moving each arm to its saved joint
    configuration (single-arm joint-space goals -- each sample only needs
    ONE camera) and computing
    T_base_board = T_base_flange @ T_flange_cam @ T_cam_board per sample,
    same formula as board_pose_from_flange_realsense.py. All samples (both
    arms combined) are averaged together since they're all observing the
    same fixed board in the same base frame (robot_b's captures are
    converted into robot_a's / the active robot's frame first via
    config/robot_bases.yaml, matching that script's convention).
    Overwrites config/base_board_pose.yaml and appends to
    outputs/calibration_logs/checkerboard_transforms.json.

  ZED calibration -- from the now-known board pose.
    Shells out to scripts/calibrate_zed_from_board_pose.sh, which calls the
    generalized src/calibration/base_to_cams_calib_3.py with
    --cam-ids zed2i_1 (see that script's --help) -- reusing the exact same
    PnP + averaging + QA-gate logic already used for the ZED trio, just
    scoped to the one ZED this rig actually has. That script itself appends
    to outputs/calibration_logs/camera_transforms.json for the ZED entry.

Run (inside the 'vision' container), with lbr_dual_arm_bringup's
hardware.launch.py AND move_group.launch.py already up (real hardware, not
mock -- board-pose replay needs real flange motion), the host camera stack
(scripts/launch_host_realsense.sh) up, hand-eye already calibrated (both
realsense_1/realsense_2 have a real T_flange_cam in
config/camera_extrinsics_realsense.yaml), and the checkerboard placed
exactly where it was during the capture_handeye_data.py session that
produced config/flange_poses/*.json:

    python3 -m src.calibration.autocalibrate_dual_realsense
"""

from __future__ import annotations

import argparse
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from src.perception.ros.qos_profiles import qos_profile_sensor_data_low_latency
from sensor_msgs.msg import CameraInfo, Image

from src.calibration.calibration_log import (
    log_checkerboard_transform,
    log_flange_transform_usage,
)
from src.calibration.flange_pose_store import ARM_KEYS, FlangePoseCapture, load_pose_set
from src.calibration.handeye_flange_cam_realsense import (
    MAX_REPROJ_ERR_PX,
    RGB_INFO_MAX_DT_S,
    _camera_topics,
    _draw_chessboard,
    _img_to_numpy_bgr,
    _K_from_camerainfo,
    _rgb_numpy_to_imgmsg,
    _solve_board_pose,
    _stamp_to_sec,
)
from src.calibration.io_extrinsics import load_extrinsics_yaml
from src.calibration.board_pose_from_flange_realsense import (
    BoardPoseSample,
    _average_se3,
    _rotation_matrix_to_rpy_deg,
    _save_sample_json as _save_board_sample_json,
)
from src.calibration.moveit_dual_arm import ArmTarget, DualArmMoveitClient, JointTarget
from src.perception.ros.multicam_grabber_realsense import _pose_msg_to_se3
from src.utils.robot_bases import get_active_robot_base, load_robot_bases
from src.utils.se3 import SE3

DEFAULT_NUM_BOARD_POSES = 2

FLANGE_POSE_MAX_AGE_S = 0.25
SETTLE_S = 1.5

CAMERA_EXTRINSICS_YAML = "config/camera_extrinsics_realsense.yaml"
BASE_BOARD_YAML = "config/base_board_pose.yaml"
ROBOT_BASES_YAML = "config/robot_bases.yaml"
BOARD_POSE_DEBUG_DIR = "outputs/calibration_debug/board_pose"
ZED_CALIB_SH = "scripts/calibrate_zed_from_board_pose.sh"

REALSENSE_CAM_IDS = ("realsense_1", "realsense_2")


@dataclass
class _ArmCamState:
    arm_key: str
    cam_id: str
    flange_pose_topic: str
    group_name: str
    base_frame: str
    tip_link: str
    img_msg: Optional[Image] = None
    img_t: Optional[float] = None
    info_t: Optional[float] = None
    K: Optional[np.ndarray] = None
    flange_pose: Optional[SE3] = None
    flange_pose_wall_t: float = 0.0

    def has_fresh_rgb_info(self) -> bool:
        if self.img_msg is None or self.K is None or self.img_t is None or self.info_t is None:
            return False
        return abs(self.img_t - self.info_t) <= RGB_INFO_MAX_DT_S

    def has_fresh_flange_pose(self) -> bool:
        if self.flange_pose is None:
            return False
        return (time.time() - self.flange_pose_wall_t) <= FLANGE_POSE_MAX_AGE_S


class _DualArmCalibNode(Node):
    """Subscribes both arms' wrist RealSense RGB/CameraInfo + both /left,
    /right ee_pose topics, and hosts the MoveGroup client used to drive both
    arms to saved poses."""

    def __init__(self, publish_debug: bool, robot_namespace: str = "lbr_dual_arm"):
        super().__init__("autocalibrate_dual_realsense")
        self.arms: dict[str, _ArmCamState] = {}

        for arm_key, arm in ARM_KEYS.items():
            state = _ArmCamState(
                arm_key=arm_key,
                cam_id=arm["cam_id"],
                flange_pose_topic=arm["flange_pose_topic"],
                group_name=arm["group_name"],
                base_frame=arm["base_frame"],
                tip_link=arm["flange_frame"],
            )
            self.arms[arm_key] = state

            rgb_topic, info_topic = _camera_topics(arm["cam_id"])
            self.create_subscription(
                Image, rgb_topic,
                lambda msg, k=arm_key: self._on_img(k, msg), qos_profile_sensor_data_low_latency,
            )
            self.create_subscription(
                CameraInfo, info_topic,
                lambda msg, k=arm_key: self._on_info(k, msg), qos_profile_sensor_data_low_latency,
            )
            self.create_subscription(
                PoseStamped, arm["flange_pose_topic"],
                lambda msg, k=arm_key: self._on_flange(k, msg), qos_profile_sensor_data_low_latency,
            )
            self.get_logger().info(
                f"[{arm_key}] cam={arm['cam_id']} rgb={rgb_topic} info={info_topic} "
                f"flange_pose={arm['flange_pose_topic']}"
            )

        self._debug_pubs = {}
        if publish_debug:
            for arm_key in ARM_KEYS:
                self._debug_pubs[arm_key] = self.create_publisher(
                    Image, f"/calibration/autocalibrate/{arm_key}/debug_image", 1
                )

        self.moveit = DualArmMoveitClient(self, namespace=robot_namespace)

    def _on_img(self, arm_key: str, msg: Image) -> None:
        st = self.arms[arm_key]
        st.img_msg = msg
        st.img_t = _stamp_to_sec(msg.header.stamp)

    def _on_info(self, arm_key: str, msg: CameraInfo) -> None:
        st = self.arms[arm_key]
        st.info_t = _stamp_to_sec(msg.header.stamp)
        st.K = _K_from_camerainfo(msg)

    def _on_flange(self, arm_key: str, msg: PoseStamped) -> None:
        st = self.arms[arm_key]
        st.flange_pose = _pose_msg_to_se3(msg)
        st.flange_pose_wall_t = time.time()

    def publish_debug(self, arm_key: str, img_bgr: np.ndarray) -> None:
        pub = self._debug_pubs.get(arm_key)
        if pub is None:
            return
        pub.publish(_rgb_numpy_to_imgmsg(img_bgr, self.arms[arm_key].cam_id, self.get_clock().now().to_msg()))

    def spin_briefly(self, duration_s: float) -> None:
        deadline = time.time() + duration_s
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)


def _require_joint_positions(captures: list[FlangePoseCapture], arm_key: str) -> None:
    missing = [c.idx for c in captures if not c.joint_positions]
    if missing:
        raise RuntimeError(
            f"{arm_key}: capture idx={missing} have no saved joint_positions -- "
            f"autocalibrate_dual_realsense.py drives to saved poses via joint-space "
            f"MoveGroup goals (JointTarget), not Cartesian IK, so every capture used here "
            f"needs joint_positions. Re-capture with capture_handeye_data.py (every capture "
            f"from that script saves joint_positions, regardless of --controller)."
        )


def _require_handeye_done() -> dict[str, SE3]:
    """Loads config/camera_extrinsics_realsense.yaml and confirms both
    RealSense cameras already have a real (non-identity) T_flange_cam --
    hand-eye is now solved beforehand by capture_handeye_data.py +
    calibrate_handeye.py, not by this script."""
    path = Path(CAMERA_EXTRINSICS_YAML)
    if not path.exists():
        raise RuntimeError(
            f"{path} doesn't exist yet -- run capture_handeye_data.py then "
            f"calibrate_handeye.py --write first (see docs/calibration_cheatsheet.md)."
        )
    extrinsics = load_extrinsics_yaml(path)
    missing_or_identity = []
    for cam_id in REALSENSE_CAM_IDS:
        T = extrinsics.get(cam_id)
        if T is None or (np.allclose(T.R, np.eye(3)) and np.allclose(T.t, 0.0)):
            missing_or_identity.append(cam_id)
    if missing_or_identity:
        raise RuntimeError(
            f"{path} has no real (non-identity) T_flange_cam for {missing_or_identity} -- "
            f"run capture_handeye_data.py then calibrate_handeye.py --write for "
            f"{missing_or_identity} first (see docs/calibration_cheatsheet.md)."
        )
    return {cam_id: extrinsics[cam_id] for cam_id in REALSENSE_CAM_IDS}


def _move_single_arm(node: _DualArmCalibNode, arm_key: str, capture: FlangePoseCapture) -> bool:
    arm = ARM_KEYS[arm_key]
    target = JointTarget(
        group_name=arm["group_name"], joint_positions=capture.joint_positions,
        label=f"{arm_key} idx={capture.idx}",
    )
    ok, _record = node.moveit.move_to_joint([target])
    return ok


def _try_capture_board_from_cam(
    node: _DualArmCalibNode, arm_key: str,
) -> Optional[tuple[SE3, np.ndarray, float]]:
    st = node.arms[arm_key]
    if not st.has_fresh_rgb_info():
        print(f"  [skip:{arm_key}] no fresh RGB/CameraInfo")
        return None
    img = _img_to_numpy_bgr(st.img_msg)
    K = st.K.copy()
    try:
        T_cam_board, corners, reproj_err = _solve_board_pose(img, K)
    except RuntimeError as e:
        print(f"  [skip:{arm_key}] {e}")
        node.publish_debug(arm_key, img)
        return None
    if reproj_err > MAX_REPROJ_ERR_PX:
        print(f"  [skip:{arm_key}] reprojection error too high: {reproj_err:.3f}px")
        return None
    vis = _draw_chessboard(img, corners, f"{arm_key} reproj={reproj_err:.3f}px")
    node.publish_debug(arm_key, vis)
    return T_cam_board, vis, reproj_err


def _run_board_pose_stage(
    node: _DualArmCalibNode,
    left_poses: list[FlangePoseCapture],
    right_poses: list[FlangePoseCapture],
    T_flange_cam_by_cam: dict[str, SE3],
) -> SE3:
    print("\n=== Board-pose stage: checkerboard pose in robot base frame (T_base_board) ===")
    active_robot, T_robotA_activeRobot = get_active_robot_base(ROBOT_BASES_YAML)
    robot_bases = load_robot_bases(ROBOT_BASES_YAML)

    debug_dir = Path(BOARD_POSE_DEBUG_DIR) / "dual"
    debug_dir.mkdir(parents=True, exist_ok=True)

    all_samples_robotA: list[BoardPoseSample] = []

    for arm_key, poses in (("left", left_poses), ("right", right_poses)):
        arm = ARM_KEYS[arm_key]
        cam_id = arm["cam_id"]
        T_flange_cam = T_flange_cam_by_cam[cam_id]
        # This arm's own base frame -> robot_a's frame (the global reference),
        # so left (robot_a) and right (robot_b) board-pose samples land in a
        # shared frame before being averaged together.
        T_robotA_armBase = robot_bases[arm["robot_base_key"]]

        for cap in poses:
            print(f"\nmoving {arm_key} to saved board-pose sample idx={cap.idx}...")
            if not _move_single_arm(node, arm_key, cap):
                raise RuntimeError(f"MoveGroup failed to reach {arm_key} board-pose sample {cap.idx}")
            node.spin_briefly(SETTLE_S)

            st = node.arms[arm_key]
            if not st.has_fresh_flange_pose():
                print(f"  [skip:{arm_key}] no fresh flange pose")
                continue
            result = _try_capture_board_from_cam(node, arm_key)
            if result is None:
                continue
            T_cam_board, vis, reproj_err = result

            T_armBase_board = st.flange_pose @ T_flange_cam @ T_cam_board
            T_robotA_board = T_robotA_armBase @ T_armBase_board

            idx = len(all_samples_robotA)
            cv2.imwrite(str(debug_dir / f"sample_{idx:02d}_{arm_key}.png"), vis)
            sample = BoardPoseSample(idx=idx, T_base_board=T_robotA_board, reproj_px=reproj_err)
            _save_board_sample_json(debug_dir, sample)
            all_samples_robotA.append(sample)
            print(f"  [ok:{arm_key}] reproj={reproj_err:.3f}px  T_robotA_board.t={T_robotA_board.t}")

    if len(all_samples_robotA) < 2:
        raise RuntimeError(
            f"Board-pose stage failed: only {len(all_samples_robotA)} accepted board-pose samples "
            f"(need >= 2 across both arms)."
        )

    T_robotA_board_avg, t_std, rot_std_deg = _average_se3([s.T_base_board for s in all_samples_robotA])
    reproj_mean = float(np.mean([s.reproj_px for s in all_samples_robotA]))

    print(f"\nT_board averaged over {len(all_samples_robotA)} samples (robot_a frame):")
    print(T_robotA_board_avg)
    print(f"translation std: {t_std:.6f}m  rotation std: {rot_std_deg:.6f}deg  mean reproj: {reproj_mean:.3f}px")

    # active_robot's own frame (matching board_pose_from_flange_realsense.py's
    # convention: base_board_pose.yaml's primary entry is in active_robot's frame).
    T_activeRobot_board = T_robotA_activeRobot.inverse() @ T_robotA_board_avg

    roll, pitch, yaw = _rotation_matrix_to_rpy_deg(T_activeRobot_board.R)
    roll_a, pitch_a, yaw_a = _rotation_matrix_to_rpy_deg(T_robotA_board_avg.R)

    out_path = Path(BASE_BOARD_YAML)
    if out_path.exists():
        backup = out_path.with_suffix(".yaml.bak")
        backup.write_text(out_path.read_text())
        print(f"Backed up existing YAML to: {backup}")

    out_data = {
        "base_board": {
            "translation_xyz_m": [round(float(v), 6) for v in T_activeRobot_board.t],
            "rotation_rpy_deg": [round(roll, 4), round(pitch, 4), round(yaw, 4)],
        },
        "base_board_robot_a_frame": {
            "translation_xyz_m": [round(float(v), 6) for v in T_robotA_board_avg.t],
            "rotation_rpy_deg": [round(roll_a, 4), round(pitch_a, 4), round(yaw_a, 4)],
        },
    }
    header = (
        f"# Checkerboard pose in the robot base frame ({active_robot}'s lbr_link_0),\n"
        f"# computed by src/calibration/autocalibrate_dual_realsense.py from\n"
        f"# {len(all_samples_robotA)} samples across both arms (realsense_1 + realsense_2).\n"
        f"#\n"
        f"# base_board_robot_a_frame is the SAME checkerboard pose, re-expressed in\n"
        f"# robot_a's base frame (the global reference -- see config/robot_bases.yaml).\n"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(header + yaml.safe_dump(out_data, sort_keys=False))
    print(f"Wrote computed board pose to: {out_path}")

    log_checkerboard_transform({
        "active_robot": active_robot,
        "num_samples": len(all_samples_robotA),
        "T_base_board": {"R": T_activeRobot_board.R.tolist(), "t": T_activeRobot_board.t.tolist()},
        "T_robotA_board": {"R": T_robotA_board_avg.R.tolist(), "t": T_robotA_board_avg.t.tolist()},
        "translation_std_m": t_std,
        "rotation_std_deg": rot_std_deg,
        "reproj_err_px_mean": reproj_mean,
    })

    return T_activeRobot_board


def _run_zed_calib_stage() -> None:
    print("\n=== ZED calibration stage: from computed board pose ===")
    sh_path = Path(ZED_CALIB_SH)
    if not sh_path.exists():
        raise RuntimeError(
            f"{sh_path} not found -- run `python3 -m src.calibration.base_to_cams_calib_3 "
            f"--cam-ids zed2i_1` directly, or regenerate the wrapper script."
        )
    print(f"Running {sh_path} ...")
    subprocess.run(["bash", str(sh_path)], check=True)


def _log_flange_usage(
    left_board: list[FlangePoseCapture], right_board: list[FlangePoseCapture],
) -> None:
    for arm_key, board in (("left", left_board), ("right", right_board)):
        log_flange_transform_usage({
            "arm_key": arm_key,
            "board_pose_capture_indices": [c.idx for c in board],
            "board_pose_captures": [
                {"idx": c.idx, "t": c.T_armBase_flange.t.tolist(), "captured_at_unix_s": c.captured_at_unix_s}
                for c in board
            ],
        })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num-board-poses", type=int, default=DEFAULT_NUM_BOARD_POSES)
    parser.add_argument("--skip-zed", action="store_true", help="Stop after the board-pose stage; run ZED calibration manually later.")
    parser.add_argument("--no-debug-topic", action="store_true")
    parser.add_argument(
        "--robot-namespace",
        default="lbr_dual_arm",
        help=(
            "Namespace lbr_dual_arm_bringup's move_group.launch.py was started with "
            "(its `robot_name` launch arg -- default matches that launch file's own "
            "default). The MoveGroup action is served at /<robot-namespace>/move_action."
        ),
    )
    args = parser.parse_args()

    T_flange_cam_by_cam = _require_handeye_done()

    left_all = load_pose_set("left").captures
    right_all = load_pose_set("right").captures
    if len(left_all) < args.num_board_poses:
        raise RuntimeError(f"left arm has only {len(left_all)} saved poses, need >= {args.num_board_poses}.")
    if len(right_all) < args.num_board_poses:
        raise RuntimeError(f"right arm has only {len(right_all)} saved poses, need >= {args.num_board_poses}.")

    left_board = left_all[-args.num_board_poses:]
    right_board = right_all[-args.num_board_poses:]

    _require_joint_positions(left_board, "left")
    _require_joint_positions(right_board, "right")

    rclpy.init()
    node = _DualArmCalibNode(publish_debug=not args.no_debug_topic, robot_namespace=args.robot_namespace)

    print("Waiting for /move_action (MoveGroup) action server...")
    if not node.moveit.wait_for_server(timeout_s=30.0):
        raise RuntimeError(
            "MoveGroup action server not available -- is lbr_dual_arm_bringup's "
            "move_group.launch.py running? (running inside Docker: DDS discovery "
            "across the container network can take noticeably longer than on bare "
            "metal, so this is a generous timeout already, not just a hang.)"
        )

    # wait_for_server() above only proves the /move_action action server has been
    # discovered -- it says nothing about whether move_group's own current-state
    # monitor has actually received a valid robot state yet. Sending the very
    # first goal too soon after move_group.launch.py comes up fails instantly
    # ("IKConstraintSampler received dirty robot state" in move_group's own log),
    # which looks like a generic planning failure everywhere else. Poll with
    # plan_only goals (no motion) using the first board-pose sample as the probe
    # until that settles, or bail with a clear error instead of a confusing one
    # from deep inside the board-pose stage.
    print("Confirming move_group's current-state monitor is ready (plan-only probe)...")
    probe_targets = [
        ArmTarget(
            group_name=ARM_KEYS["left"]["group_name"], base_frame=ARM_KEYS["left"]["base_frame"],
            tip_link=ARM_KEYS["left"]["flange_frame"], T_armBase_flange=left_board[0].T_armBase_flange,
        ),
        ArmTarget(
            group_name=ARM_KEYS["right"]["group_name"], base_frame=ARM_KEYS["right"]["base_frame"],
            tip_link=ARM_KEYS["right"]["flange_frame"], T_armBase_flange=right_board[0].T_armBase_flange,
        ),
    ]
    if not node.moveit.wait_for_valid_state(probe_targets, group_name="both_arms_flange"):
        raise RuntimeError(
            "move_group never became ready to plan for 'both_arms_flange' -- see the "
            "warnings above and move_group's own ~/.ros/log/move_group_*.log for "
            "the real reason (dirty robot state vs. an actual planning failure "
            "on the first probe pose)."
        )

    try:
        _run_board_pose_stage(node, left_board, right_board, T_flange_cam_by_cam)

        _log_flange_usage(left_board, right_board)

        if not args.skip_zed:
            _run_zed_calib_stage()
        else:
            print("\n--skip-zed passed: leaving the ZED calibration stage for a manual run.")

        print("\n=== All stages complete ===")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
