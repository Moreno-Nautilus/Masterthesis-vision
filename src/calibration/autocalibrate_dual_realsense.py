"""Automatic dual-arm RealSense calibration -- Step 2 of the "turned up a
notch" routine (see docs/getting_started_realsense.md section 4).

Consumes the flange poses saved by capture_flange_poses_dual.py
(config/flange_poses/left.json, config/flange_poses/right.json -- 7 each
recommended) and drives BOTH arms there automatically via MoveIt (single
"both_arms" MoveGroup goal per pose-pair, see moveit_dual_arm.py), instead of
requiring an operator to jog the arm to each pose by hand like the original
handeye_flange_cam_realsense.py / board_pose_from_flange_realsense.py.

The checkerboard must stay fixed in the same place it was in during
capture_flange_poses_dual.py for all of this to be valid -- these are the
SAME physical poses, replayed, not new ones.

Three stages, run in order, each gated on the previous succeeding:

  Stage A -- Hand-eye (T_flange_cam), BOTH arms, jointly.
    Uses the first `--num-handeye-poses` (default 5) saved captures per arm.
    For pose-pair i: moves arm_one to left capture[i] and arm_two to right
    capture[i] in ONE simultaneous both_arms MoveGroup goal, waits for
    settle, captures a checkerboard-in-camera PnP solve from EACH arm's own
    wrist RealSense, and stashes (T_armBase_flange, T_cam_board) pairs per
    arm exactly like handeye_flange_cam_realsense.py did manually. Once
    both arms have >= min samples, solves AX=XB independently per arm (the
    two arms' cameras are physically independent mounts -- there is no
    shared unknown between them) and writes T_flange_cam for realsense_1 AND
    realsense_2 into config/camera_extrinsics_realsense.yaml together.
    Every accepted sample is written to outputs/calibration_debug/ (JSON +
    PNG) as soon as it's captured, and the final per-arm solve + QA
    residuals are appended to outputs/calibration_logs/camera_transforms.json
    -- both immediately, so nothing needs to be redone if a later stage
    fails.

  Stage B -- Checkerboard pose in the robot base frame (T_base_board).
    Runs only once BOTH cameras from Stage A have a real (non-identity)
    T_flange_cam. Uses the LAST `--num-board-poses` (default 2) saved
    captures per arm (4 samples total: 2 arms x 2 poses), moving each arm
    to its saved pose (single-arm goals this time -- no need for
    simultaneous motion since each sample only needs ONE camera) and
    computing T_base_board = T_base_flange @ T_flange_cam @ T_cam_board per
    sample, same formula as board_pose_from_flange_realsense.py. All
    samples (both arms combined) are averaged together since they're all
    observing the same fixed board in the same base frame (mind: robot_b's
    captures are converted into robot_a's / the active robot's frame first
    via config/robot_bases.yaml, matching that script's convention).
    Overwrites config/base_board_pose.yaml and appends to
    outputs/calibration_logs/checkerboard_transforms.json.

  Stage C -- ZED calibration from the now-known board pose.
    Shells out to scripts/calibrate_zed_from_board_pose.sh, which calls the
    generalized src/calibration/base_to_cams_calib_3.py with
    --cam-ids zed2i_1 (see that script's --help) -- reusing the exact same
    PnP + averaging + QA-gate logic already used for the ZED trio, just
    scoped to the one ZED this rig actually has. That script itself appends
    to outputs/calibration_logs/camera_transforms.json for the ZED entry.

Run (inside the 'vision' container), with lbr_dual_arm_bringup's
hardware.launch.py AND move_group.launch.py already up (real hardware, not
mock -- Stage A/B need real varied flange motion), the host camera stack
(scripts/launch_host_realsense.sh) up, and the checkerboard placed exactly
where it was during capture_flange_poses_dual.py:

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
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

from src.calibration.calibration_log import (
    log_camera_transform,
    log_checkerboard_transform,
    log_flange_transform_usage,
)
from src.calibration.flange_pose_store import ARM_KEYS, FlangePoseCapture, load_pose_set
from src.calibration.handeye_flange_cam_realsense import (
    MAX_REPROJ_ERR_PX,
    RGB_INFO_MAX_DT_S,
    HandEyeSample,
    _camera_topics,
    _draw_chessboard,
    _img_to_numpy_bgr,
    _K_from_camerainfo,
    _rgb_numpy_to_imgmsg,
    _save_sample_json,
    _solve_board_pose,
    _solve_handeye,
    _stamp_to_sec,
)
from src.calibration.io_extrinsics import load_extrinsics_yaml, update_extrinsics_yaml_preserving_header
from src.calibration.board_pose_from_flange_realsense import (
    BoardPoseSample,
    _average_se3,
    _rotation_matrix_to_rpy_deg,
    _save_sample_json as _save_board_sample_json,
)
from src.calibration.moveit_dual_arm import ArmTarget, DualArmMoveitClient
from src.perception.ros.multicam_grabber_realsense import _pose_msg_to_se3
from src.utils.robot_bases import get_active_robot_base, load_robot_bases
from src.utils.se3 import SE3

DEFAULT_NUM_HANDEYE_POSES = 5
DEFAULT_NUM_BOARD_POSES = 2
MIN_HANDEYE_SAMPLES = 5

FLANGE_POSE_MAX_AGE_S = 0.25
SETTLE_S = 1.5

CAMERA_EXTRINSICS_YAML = "config/camera_extrinsics_realsense.yaml"
BASE_BOARD_YAML = "config/base_board_pose.yaml"
ROBOT_BASES_YAML = "config/robot_bases.yaml"
HANDEYE_DEBUG_DIR = "outputs/calibration_debug/handeye"
BOARD_POSE_DEBUG_DIR = "outputs/calibration_debug/board_pose"
ZED_CALIB_SH = "scripts/calibrate_zed_from_board_pose.sh"


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
                lambda msg, k=arm_key: self._on_img(k, msg), qos_profile_sensor_data,
            )
            self.create_subscription(
                CameraInfo, info_topic,
                lambda msg, k=arm_key: self._on_info(k, msg), qos_profile_sensor_data,
            )
            self.create_subscription(
                PoseStamped, arm["flange_pose_topic"],
                lambda msg, k=arm_key: self._on_flange(k, msg), qos_profile_sensor_data,
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


def _move_both_arms(
    node: _DualArmCalibNode,
    left_capture: FlangePoseCapture,
    right_capture: FlangePoseCapture,
) -> bool:
    left = ARM_KEYS["left"]
    right = ARM_KEYS["right"]
    targets = [
        ArmTarget(
            group_name=left["group_name"], base_frame=left["base_frame"],
            tip_link=left["flange_frame"], T_armBase_flange=left_capture.T_armBase_flange,
        ),
        ArmTarget(
            group_name=right["group_name"], base_frame=right["base_frame"],
            tip_link=right["flange_frame"], T_armBase_flange=right_capture.T_armBase_flange,
        ),
    ]
    return node.moveit.move_to(targets, group_name="both_arms")


def _move_single_arm(node: _DualArmCalibNode, arm_key: str, capture: FlangePoseCapture) -> bool:
    arm = ARM_KEYS[arm_key]
    target = ArmTarget(
        group_name=arm["group_name"], base_frame=arm["base_frame"],
        tip_link=arm["flange_frame"], T_armBase_flange=capture.T_armBase_flange,
    )
    return node.moveit.move_to([target])


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


def _run_stage_a_handeye(
    node: _DualArmCalibNode,
    left_poses: list[FlangePoseCapture],
    right_poses: list[FlangePoseCapture],
    min_samples: int,
) -> dict[str, SE3]:
    print("\n=== Stage A: dual hand-eye (T_flange_cam), both arms simultaneously ===")
    n_pairs = min(len(left_poses), len(right_poses))
    if n_pairs < min_samples:
        raise RuntimeError(
            f"Not enough hand-eye pose pairs: have {n_pairs}, need >= {min_samples} per arm. "
            f"Run capture_flange_poses_dual.py for both arms first."
        )

    debug_dirs = {
        arm_key: Path(HANDEYE_DEBUG_DIR) / ARM_KEYS[arm_key]["cam_id"]
        for arm_key in ARM_KEYS
    }
    for d in debug_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    samples: dict[str, list[HandEyeSample]] = {"left": [], "right": []}

    for i in range(n_pairs):
        left_cap, right_cap = left_poses[i], right_poses[i]
        print(f"\n[pair {i + 1}/{n_pairs}] moving both arms simultaneously (both_arms goal)...")
        if not _move_both_arms(node, left_cap, right_cap):
            raise RuntimeError(
                f"MoveGroup failed to reach pose pair {i} for both_arms -- aborting Stage A."
            )
        node.spin_briefly(SETTLE_S)

        for arm_key, cap in (("left", left_cap), ("right", right_cap)):
            arm = ARM_KEYS[arm_key]
            st = node.arms[arm_key]
            if not st.has_fresh_flange_pose():
                print(f"  [skip:{arm_key}] no fresh flange pose on {arm['flange_pose_topic']}")
                continue
            result = _try_capture_board_from_cam(node, arm_key)
            if result is None:
                continue
            T_cam_board, vis, reproj_err = result

            idx = len(samples[arm_key])
            cv2.imwrite(str(debug_dirs[arm_key] / f"sample_{idx:02d}.png"), vis)
            sample = HandEyeSample(
                idx=idx, T_base_flange=st.flange_pose, T_cam_board=T_cam_board, reproj_px=reproj_err,
            )
            _save_sample_json(debug_dirs[arm_key], sample)
            samples[arm_key].append(sample)
            print(f"  [ok:{arm_key}] reproj={reproj_err:.3f}px  T_base_flange.t={st.flange_pose.t}")

    results: dict[str, SE3] = {}
    for arm_key in ("left", "right"):
        cam_id = ARM_KEYS[arm_key]["cam_id"]
        arm_samples = samples[arm_key]
        if len(arm_samples) < min_samples:
            raise RuntimeError(
                f"Stage A failed for {arm_key} ({cam_id}): only {len(arm_samples)} accepted "
                f"samples, need >= {min_samples}."
            )
        T_flange_cam, residuals_deg, residuals_m = _solve_handeye(arm_samples)
        print(f"\n--- {arm_key} ({cam_id}) hand-eye result ---")
        print(T_flange_cam)
        print(
            f"AX=XB residuals: rotation mean={residuals_deg.mean():.4f}deg "
            f"max={residuals_deg.max():.4f}deg | translation mean={residuals_m.mean():.6f}m "
            f"max={residuals_m.max():.6f}m"
        )
        results[cam_id] = T_flange_cam

        log_camera_transform({
            "stage": "handeye_flange_cam",
            "cam_id": cam_id,
            "arm_key": arm_key,
            "num_samples": len(arm_samples),
            "T_flange_cam": {"R": T_flange_cam.R.tolist(), "t": T_flange_cam.t.tolist()},
            "ax_xb_residual_rot_deg_mean": float(residuals_deg.mean()),
            "ax_xb_residual_rot_deg_max": float(residuals_deg.max()),
            "ax_xb_residual_t_m_mean": float(residuals_m.mean()),
            "ax_xb_residual_t_m_max": float(residuals_m.max()),
        })

    out_path = Path(CAMERA_EXTRINSICS_YAML)
    if out_path.exists():
        backup = out_path.with_suffix(".yaml.bak")
        backup.write_text(out_path.read_text())
        print(f"\nBacked up existing YAML to: {backup}")
    update_extrinsics_yaml_preserving_header(out_path, results)
    print(f"Wrote T_flange_cam for {sorted(results)} into: {out_path}")

    return results


def _run_stage_b_board_pose(
    node: _DualArmCalibNode,
    left_poses: list[FlangePoseCapture],
    right_poses: list[FlangePoseCapture],
    T_flange_cam_by_cam: dict[str, SE3],
) -> SE3:
    print("\n=== Stage B: checkerboard pose in robot base frame (T_base_board) ===")
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
            f"Stage B failed: only {len(all_samples_robotA)} accepted board-pose samples "
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


def _run_stage_c_zed_calib() -> None:
    print("\n=== Stage C: ZED calibration from computed board pose ===")
    sh_path = Path(ZED_CALIB_SH)
    if not sh_path.exists():
        raise RuntimeError(
            f"{sh_path} not found -- run `python3 -m src.calibration.base_to_cams_calib_3 "
            f"--cam-ids zed2i_1` directly, or regenerate the wrapper script."
        )
    print(f"Running {sh_path} ...")
    subprocess.run(["bash", str(sh_path)], check=True)


def _log_flange_usage(
    left_handeye: list[FlangePoseCapture], right_handeye: list[FlangePoseCapture],
    left_board: list[FlangePoseCapture], right_board: list[FlangePoseCapture],
) -> None:
    for arm_key, handeye, board in (
        ("left", left_handeye, left_board), ("right", right_handeye, right_board),
    ):
        log_flange_transform_usage({
            "arm_key": arm_key,
            "handeye_capture_indices": [c.idx for c in handeye],
            "board_pose_capture_indices": [c.idx for c in board],
            "handeye_captures": [
                {"idx": c.idx, "t": c.T_armBase_flange.t.tolist(), "captured_at_unix_s": c.captured_at_unix_s}
                for c in handeye
            ],
            "board_pose_captures": [
                {"idx": c.idx, "t": c.T_armBase_flange.t.tolist(), "captured_at_unix_s": c.captured_at_unix_s}
                for c in board
            ],
        })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num-handeye-poses", type=int, default=DEFAULT_NUM_HANDEYE_POSES)
    parser.add_argument("--num-board-poses", type=int, default=DEFAULT_NUM_BOARD_POSES)
    parser.add_argument("--min-handeye-samples", type=int, default=MIN_HANDEYE_SAMPLES)
    parser.add_argument("--skip-zed", action="store_true", help="Skip Stage C (ZED calibration).")
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

    left_all = load_pose_set("left").captures
    right_all = load_pose_set("right").captures
    if len(left_all) < args.num_handeye_poses + args.num_board_poses:
        raise RuntimeError(
            f"left arm has only {len(left_all)} saved poses, need >= "
            f"{args.num_handeye_poses + args.num_board_poses}."
        )
    if len(right_all) < args.num_handeye_poses + args.num_board_poses:
        raise RuntimeError(
            f"right arm has only {len(right_all)} saved poses, need >= "
            f"{args.num_handeye_poses + args.num_board_poses}."
        )

    left_handeye = left_all[: args.num_handeye_poses]
    right_handeye = right_all[: args.num_handeye_poses]
    left_board = left_all[args.num_handeye_poses: args.num_handeye_poses + args.num_board_poses]
    right_board = right_all[args.num_handeye_poses: args.num_handeye_poses + args.num_board_poses]

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
    # plan_only goals (no motion) using the real first pose-pair as the probe
    # until that settles, or bail with a clear error instead of a confusing one
    # from deep inside Stage A.
    print("Confirming move_group's current-state monitor is ready (plan-only probe)...")
    probe_targets = [
        ArmTarget(
            group_name=ARM_KEYS["left"]["group_name"], base_frame=ARM_KEYS["left"]["base_frame"],
            tip_link=ARM_KEYS["left"]["flange_frame"], T_armBase_flange=left_handeye[0].T_armBase_flange,
        ),
        ArmTarget(
            group_name=ARM_KEYS["right"]["group_name"], base_frame=ARM_KEYS["right"]["base_frame"],
            tip_link=ARM_KEYS["right"]["flange_frame"], T_armBase_flange=right_handeye[0].T_armBase_flange,
        ),
    ]
    if not node.moveit.wait_for_valid_state(probe_targets, group_name="both_arms"):
        raise RuntimeError(
            "move_group never became ready to plan for 'both_arms' -- see the "
            "warnings above and move_group's own ~/.ros/log/move_group_*.log for "
            "the real reason (dirty robot state vs. an actual planning failure "
            "on pose-pair 0)."
        )

    try:
        T_flange_cam_by_cam = _run_stage_a_handeye(
            node, left_handeye, right_handeye, args.min_handeye_samples
        )

        _run_stage_b_board_pose(node, left_board, right_board, T_flange_cam_by_cam)

        _log_flange_usage(left_handeye, right_handeye, left_board, right_board)

        if not args.skip_zed:
            _run_stage_c_zed_calib()
        else:
            print("\n--skip-zed passed: leaving Stage C (ZED calibration) for a manual run.")

        print("\n=== All stages complete ===")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
