"""Checkerboard pose in the robot base frame (Stage 2 of 2), using a
wrist-mounted RealSense camera that already has a T_flange_cam from
src/calibration/handeye_flange_cam_realsense.py (Stage 1).

With the flange->camera offset known, one synced (camera image, flange
pose) pair is enough to place the checkerboard in the robot base frame:

    T_base_board = T_base_flange @ T_flange_cam @ T_cam_board

where T_cam_board comes from solvePnP on the checkerboard, exactly as in
Stage 1. This script takes several such samples (holding the checkerboard
fixed while the arm can optionally be re-jogged between samples, since the
board itself never moves) and averages them, then overwrites
config/base_board_pose.yaml -- the same file base_to_cams_calib_3.py reads
for the static ZED calibration -- replacing the previous hand-measured
value with a real computed one, in the frame of whichever robot is
'active_robot' in config/robot_bases.yaml.

Run (inside the 'vision' container):
    python3 -m src.calibration.board_pose_from_flange_realsense --cam-id realsense_1
"""

from __future__ import annotations

import argparse
import json
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

from src.calibration.handeye_flange_cam_realsense import (
    CHESS_COLS,
    CHESS_ROWS,
    FLANGE_POSE_TOPIC_DEFAULT,
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
from src.perception.ros.multicam_grabber_realsense import _pose_msg_to_se3
from src.utils.robot_bases import get_active_robot_base
from src.utils.se3 import SE3

# ---------------- USER SETTINGS ----------------
FLANGE_POSE_MAX_AGE_S = 0.25
MAX_REPROJ_ERR_PX = 2.5

NUM_SAMPLES = 8
MIN_SAMPLES_TO_SOLVE = 5
MAX_FINAL_TRANSLATION_STD_M = 0.01
MAX_FINAL_ROTATION_STD_DEG = 1.0

EXTRINSICS_YAML = "config/camera_extrinsics_realsense.yaml"
OUT_YAML = "config/base_board_pose.yaml"
DEBUG_DIR = "outputs/calibration_debug/board_pose"
ROBOT_BASES_YAML = "config/robot_bases.yaml"
# ------------------------------------------------


def _rotation_matrix_to_rotvec(R: np.ndarray) -> np.ndarray:
    rvec, _ = cv2.Rodrigues(R.astype(np.float64))
    return rvec.reshape(3)


def _rotvec_to_rotation_matrix(rvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(rvec.reshape(3, 1).astype(np.float64))
    return R


def _rotation_matrix_to_rpy_deg(R: np.ndarray) -> tuple[float, float, float]:
    # Inverse of base_to_cams_calib_3.py's _rpy_deg_to_R (R = Rz @ Ry @ Rx).
    sy = -R[2, 0]
    sy = float(np.clip(sy, -1.0, 1.0))
    pitch = np.arcsin(sy)
    cy = np.cos(pitch)

    if abs(cy) > 1e-6:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        # Gimbal lock: pitch = +-90deg, roll/yaw not independently observable.
        roll = np.arctan2(-R[1, 2], R[1, 1])
        yaw = 0.0

    return float(np.degrees(roll)), float(np.degrees(pitch)), float(np.degrees(yaw))


def _average_se3(poses: list[SE3]) -> tuple[SE3, float, float]:
    ts = np.stack([p.t for p in poses], axis=0)
    t_mean = ts.mean(axis=0)
    t_std = float(np.linalg.norm(ts.std(axis=0)))

    rvecs = np.stack([_rotation_matrix_to_rotvec(p.R) for p in poses], axis=0)
    R_mean = _rotvec_to_rotation_matrix(rvecs.mean(axis=0))

    ang_devs = []
    for p in poses:
        R_rel = R_mean.T @ p.R
        c = np.clip((np.trace(R_rel) - 1.0) * 0.5, -1.0, 1.0)
        ang_devs.append(float(np.degrees(np.arccos(c))))
    rot_std_deg = float(np.std(ang_devs))

    return SE3(R_mean, t_mean), t_std, rot_std_deg


@dataclass
class BoardPoseSample:
    idx: int
    T_base_board: SE3
    reproj_px: float


def _sample_to_json_dict(s: BoardPoseSample) -> dict:
    return {
        "idx": s.idx,
        "reproj_px": s.reproj_px,
        "T_base_board": {"R": s.T_base_board.R.tolist(), "t": s.T_base_board.t.tolist()},
    }


def _sample_from_json_dict(d: dict) -> BoardPoseSample:
    return BoardPoseSample(
        idx=d["idx"],
        T_base_board=SE3(np.array(d["T_base_board"]["R"]), np.array(d["T_base_board"]["t"])),
        reproj_px=d["reproj_px"],
    )


def _save_sample_json(debug_dir: Path, sample: BoardPoseSample) -> None:
    # Written immediately on capture so a crash in averaging/YAML-write (or the
    # script's own std-dev rejection check) doesn't lose the raw poses -- only
    # the debug PNG survived that before, which isn't enough to redo the
    # average without recapturing everything.
    path = debug_dir / f"sample_{sample.idx:02d}.json"
    path.write_text(json.dumps(_sample_to_json_dict(sample), indent=2))


def _load_samples_from_dir(debug_dir: Path) -> list[BoardPoseSample]:
    samples = [
        _sample_from_json_dict(json.loads(p.read_text()))
        for p in sorted(debug_dir.glob("sample_*.json"))
    ]
    samples.sort(key=lambda s: s.idx)
    return samples


class BoardPoseNode(Node):
    def __init__(self, cam_id: str, flange_pose_topic: str, publish_debug: bool):
        super().__init__("board_pose_from_flange_realsense_" + cam_id)
        self.cam_id = cam_id
        rgb_topic, info_topic = _camera_topics(cam_id)

        self.img_msg: Optional[Image] = None
        self.img_t: Optional[float] = None
        self.info_msg: Optional[CameraInfo] = None
        self.info_t: Optional[float] = None
        self.K: Optional[np.ndarray] = None

        self.flange_pose: Optional[SE3] = None
        self.flange_pose_wall_t: float = 0.0

        self.create_subscription(Image, rgb_topic, self._on_img, qos_profile_sensor_data_low_latency)
        self.create_subscription(CameraInfo, info_topic, self._on_info, qos_profile_sensor_data_low_latency)
        self.create_subscription(PoseStamped, flange_pose_topic, self._on_flange, qos_profile_sensor_data_low_latency)

        self._debug_pub = None
        if publish_debug:
            self._debug_pub = self.create_publisher(
                Image, f"/calibration/board_pose/{cam_id}/debug_image", 1
            )

        self.get_logger().info(
            f"[{cam_id}] rgb={rgb_topic} info={info_topic} flange_pose={flange_pose_topic}"
        )

    def _on_img(self, msg: Image) -> None:
        self.img_msg = msg
        self.img_t = _stamp_to_sec(msg.header.stamp)

    def _on_info(self, msg: CameraInfo) -> None:
        self.info_msg = msg
        self.info_t = _stamp_to_sec(msg.header.stamp)
        self.K = _K_from_camerainfo(msg)

    def _on_flange(self, msg: PoseStamped) -> None:
        self.flange_pose = _pose_msg_to_se3(msg)
        self.flange_pose_wall_t = time.time()

    def has_fresh_flange_pose(self) -> bool:
        if self.flange_pose is None:
            return False
        return (time.time() - self.flange_pose_wall_t) <= FLANGE_POSE_MAX_AGE_S

    def has_fresh_rgb_info(self) -> bool:
        if self.img_msg is None or self.K is None or self.img_t is None or self.info_t is None:
            return False
        return abs(self.img_t - self.info_t) <= RGB_INFO_MAX_DT_S

    def publish_debug(self, img_bgr: np.ndarray) -> None:
        if self._debug_pub is None:
            return
        self._debug_pub.publish(_rgb_numpy_to_imgmsg(img_bgr, self.cam_id, self.get_clock().now().to_msg()))


def _average_and_write(
    cam_id: str, active_robot: str, T_robotA_activeRobot: SE3,
    samples: list[BoardPoseSample], debug_dir: Path,
) -> None:
    T_base_board_avg, t_std, rot_std_deg = _average_se3([s.T_base_board for s in samples])

    # Same board pose, re-expressed in robot_a's base frame (the global
    # reference -- config/robot_bases.yaml: robot_a is the origin, robot_b is
    # +0.84m in y with identical orientation, so this is just a translation
    # when active_robot == robot_b, but stays general via T_robotA_activeRobot).
    T_robotA_board_avg = T_robotA_activeRobot @ T_base_board_avg

    print("\n=== FINAL RESULT ===")
    print(f"T_base_board (averaged over {len(samples)} samples, base = {active_robot}'s lbr_link_0):")
    print(T_base_board_avg)
    print(f"translation std: {t_std:.6f} m   rotation std: {rot_std_deg:.6f} deg")
    print(f"\nSame pose in robot_a's base frame (global reference):")
    print(T_robotA_board_avg)

    if t_std > MAX_FINAL_TRANSLATION_STD_M:
        raise RuntimeError(f"Calibration rejected: translation spread too high ({t_std:.6f}m)")
    if rot_std_deg > MAX_FINAL_ROTATION_STD_DEG:
        raise RuntimeError(f"Calibration rejected: rotation spread too high ({rot_std_deg:.6f}deg)")

    roll, pitch, yaw = _rotation_matrix_to_rpy_deg(T_base_board_avg.R)
    roll_a, pitch_a, yaw_a = _rotation_matrix_to_rpy_deg(T_robotA_board_avg.R)

    out_path = Path(OUT_YAML)
    if out_path.exists():
        backup = out_path.with_suffix(".yaml.bak")
        backup.write_text(out_path.read_text())
        print(f"\nBacked up existing YAML to: {backup}")

    out_data = {
        "base_board": {
            "translation_xyz_m": [round(float(v), 6) for v in T_base_board_avg.t],
            "rotation_rpy_deg": [round(roll, 4), round(pitch, 4), round(yaw, 4)],
        },
        "base_board_robot_a_frame": {
            "translation_xyz_m": [round(float(v), 6) for v in T_robotA_board_avg.t],
            "rotation_rpy_deg": [round(roll_a, 4), round(pitch_a, 4), round(yaw_a, 4)],
        },
    }
    header = (
        f"# Checkerboard pose in the robot base frame ({active_robot}'s lbr_link_0),\n"
        f"# computed by src/calibration/board_pose_from_flange_realsense.py from\n"
        f"# {len(samples)} samples via cam_id={cam_id}. Replaces the previous\n"
        f"# hand-measured value -- see config/robot_bases.yaml for the active robot.\n"
        f"#\n"
        f"# base_board_robot_a_frame is the SAME checkerboard pose, re-expressed in\n"
        f"# robot_a's base frame (the global reference -- see config/robot_bases.yaml\n"
        f"# for the per-robot offsets used to convert between the two).\n"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(header + yaml.safe_dump(out_data, sort_keys=False))

    print(f"Wrote computed board pose to: {out_path}")
    print(f"Saved debug corner images + pose data to: {debug_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cam-id", required=True, choices=["realsense_1", "realsense_2"])
    parser.add_argument("--flange-pose-topic", default=FLANGE_POSE_TOPIC_DEFAULT)
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES)
    parser.add_argument("--min-samples", type=int, default=MIN_SAMPLES_TO_SOLVE)
    parser.add_argument("--no-debug-topic", action="store_true")
    parser.add_argument(
        "--solve-only", action="store_true",
        help="Skip capture entirely; reload sample_*.json already saved under "
             "outputs/calibration_debug/board_pose/<cam_id>/ from a previous (e.g. crashed or "
             "rejected) run and just re-run the averaging + YAML write.",
    )
    args = parser.parse_args()

    active_robot, T_robotA_activeRobot = get_active_robot_base(ROBOT_BASES_YAML)
    debug_dir = Path(DEBUG_DIR) / args.cam_id
    debug_dir.mkdir(parents=True, exist_ok=True)

    if args.solve_only:
        samples = _load_samples_from_dir(debug_dir)
        print(f"Loaded {len(samples)} saved sample(s) from {debug_dir}")
        if len(samples) < args.min_samples:
            raise RuntimeError(f"Not enough saved samples: {len(samples)} < {args.min_samples}")
        _average_and_write(args.cam_id, active_robot, T_robotA_activeRobot, samples, debug_dir)
        return

    extr = load_extrinsics_yaml(EXTRINSICS_YAML)
    if args.cam_id not in extr:
        raise RuntimeError(f"{args.cam_id} not found in {EXTRINSICS_YAML}")
    T_flange_cam = extr[args.cam_id]
    if np.allclose(T_flange_cam.R, np.eye(3)) and np.allclose(T_flange_cam.t, 0.0):
        raise RuntimeError(
            f"{args.cam_id}'s entry in {EXTRINSICS_YAML} is still the identity placeholder. "
            "Run src.calibration.handeye_flange_cam_realsense for this camera first "
            "(Stage 1) before computing the board pose (Stage 2)."
        )

    rclpy.init()
    node = BoardPoseNode(args.cam_id, args.flange_pose_topic, publish_debug=not args.no_debug_topic)

    print("")
    print(f"=== Checkerboard pose in base frame, via {args.cam_id} ===")
    print(f"Active robot (config/robot_bases.yaml): {active_robot}")
    print(f"Using calibrated T_flange_cam:\n{T_flange_cam}")
    print("")
    print("The checkerboard must stay fixed for the whole run (this recovers ITS")
    print("pose, so it can't move between samples). You may re-jog the arm between")
    print("samples for redundancy/averaging, or just take several samples from one pose.")
    print("")
    print("Suggested Foxglove layout:")
    print(f"  - Image panel -> /calibration/board_pose/{args.cam_id}/debug_image")
    print("")

    try:
        samples: list[BoardPoseSample] = []
        while len(samples) < args.num_samples:
            input(f"[{len(samples)}/{args.num_samples}] Press Enter to capture a sample: ")

            t_deadline = time.time() + 2.0
            while time.time() < t_deadline:
                rclpy.spin_once(node, timeout_sec=0.05)

            if not node.has_fresh_rgb_info():
                print("  [skip] no fresh RGB/CameraInfo yet")
                continue
            if not node.has_fresh_flange_pose():
                print(f"  [skip] no fresh flange pose on {args.flange_pose_topic}")
                continue

            img = _img_to_numpy_bgr(node.img_msg)
            K = node.K.copy()
            T_base_flange = node.flange_pose

            try:
                T_cam_board, corners, reproj_err = _solve_board_pose(img, K)
            except RuntimeError as e:
                print(f"  [skip] {e}")
                node.publish_debug(img)
                continue

            if reproj_err > MAX_REPROJ_ERR_PX:
                print(f"  [skip] reprojection error too high: {reproj_err:.3f}px")
                continue

            # T_base_board = T_base_flange @ T_flange_cam @ T_cam_board
            T_base_board = T_base_flange @ T_flange_cam @ T_cam_board

            vis = _draw_chessboard(img, corners, f"sample {len(samples)} reproj={reproj_err:.3f}px")
            node.publish_debug(vis)
            cv2.imwrite(str(debug_dir / f"sample_{len(samples):02d}.png"), vis)

            sample = BoardPoseSample(idx=len(samples), T_base_board=T_base_board, reproj_px=reproj_err)
            _save_sample_json(debug_dir, sample)

            print(f"  [ok] reproj={reproj_err:.3f}px  T_base_board.t={T_base_board.t}")
            samples.append(sample)

        if len(samples) < args.min_samples:
            raise RuntimeError(f"Not enough accepted samples: {len(samples)} < {args.min_samples}")

        try:
            _average_and_write(args.cam_id, active_robot, T_robotA_activeRobot, samples, debug_dir)
        except Exception:
            print(f"\nAveraging/write step failed, but all {len(samples)} samples are already saved as")
            print(f"JSON under {debug_dir} -- nothing needs to be recaptured. Fix the underlying")
            print(f"issue (or drop bad samples), then re-run with:")
            print(f"  python3 -m src.calibration.board_pose_from_flange_realsense --cam-id {args.cam_id} --solve-only")
            raise
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
