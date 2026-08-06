"""Hand-eye calibration for a wrist-mounted RealSense camera (Stage 1 of 2).

Solves the classic AX=XB hand-eye problem to find T_flange_cam: the static
rigid offset between the robot flange (lbr_link_ee) and a RealSense camera's
optical frame, bolted to the end-effector so it moves with the arm.

Unlike src/calibration/base_to_cams_calib_3.py (which fixes the checkerboard
AND every camera, so a single PnP per camera solves everything), here only
the checkerboard is fixed -- the camera moves with the robot -- so a single
image can't disambiguate the mount offset from the board's own unknown pose.
Instead we collect the checkerboard-in-camera pose (via PnP) and the
flange-in-base pose (via /iiwa/ee_pose, from the lbr_fri_ros2_stack's tf) at
several distinct robot poses, then solve for the one offset consistent with
all of them via cv2.calibrateHandEye.

Workflow (see docs/getting_started_realsense.md section 4 for the full
walkthrough):
  1. Bring up the robot (lbr_bringup hardware.launch.py) and, for jogging,
     the MoveIt GUI (scripts/launch_moveit_scene_viewer.launch.py or
     lbr_bringup's own moveit launch) -- command Cartesian goals from the
     RViz MotionPlanning panel.
  2. Bring up the camera stack (scripts/launch_host_realsense.sh) and open
     Foxglove on the suggested layout/topics (printed at startup below).
  3. Run this script for one camera at a time (--cam-id realsense_1 or
     realsense_2). For each of several (>=10) very different wrist
     orientations with the checkerboard fully visible: jog there with
     MoveIt, let the arm settle, then press Enter in this terminal to
     capture a sample.
  4. After enough samples, the script solves T_flange_cam and writes it into
     config/camera_extrinsics_realsense.yaml (backing up the previous file
     first), preserving the header comment and the other cam_id entries.

Run (inside the 'vision' container, matching base_to_cams_calib_3.py):
    python3 -m src.calibration.handeye_flange_cam_realsense --cam-id realsense_1
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
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from src.perception.ros.qos_profiles import qos_profile_sensor_data_low_latency
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header

from src.calibration.io_extrinsics import update_extrinsics_yaml_preserving_header
from src.perception.ros.multicam_grabber_realsense import _pose_msg_to_se3
from src.utils.se3 import SE3

# ---------------- USER SETTINGS ----------------
CHESS_COLS = 8   # inner corners along x (matches base_to_cams_calib_3.py / the ZED board)
CHESS_ROWS = 11  # inner corners along y
SQUARE_SIZE_M = 0.03

RGB_INFO_MAX_DT_S = 0.50       # RGB vs CameraInfo freshness
FLANGE_POSE_MAX_AGE_S = 0.25   # /iiwa/ee_pose freshness at capture time
MAX_REPROJ_ERR_PX = 2.5        # reject a sample if PnP reprojection error is too high

MIN_SAMPLES_TO_SOLVE = 10      # AX=XB needs rotational diversity; more than the static case
RECOMMENDED_SAMPLES = 15

OUT_YAML = "config/camera_extrinsics_realsense.yaml"
DEBUG_DIR = "outputs/calibration_debug/handeye"

# RealSense USB serials, for the --detect convenience check (see
# docs/getting_started_realsense.md section 3). Update if hardware changes.
KNOWN_REALSENSE_SERIALS = {
    "realsense_1": "260322275185",
    "realsense_2": "260522275434",
}

FLANGE_POSE_TOPIC_DEFAULT = "/iiwa/ee_pose"
# ------------------------------------------------


def _camera_topics(cam_id: str) -> tuple[str, str]:
    # Matches the realsense2_camera ROS2 wrapper's default namespacing, as
    # used throughout src/perception/ros/learn_runners/run_pipeline_track_multicam_realsense.py.
    # /image_rect (not /image_raw): rectified by the image_proc RectifyNode
    # started per-camera in zed_realsense_trio.launch.py. Needed here
    # specifically because _solve_board_pose / _compute_reproj_err_px below
    # hardcode dist=0 for solvePnP/projectPoints — that assumption is only
    # correct once the image is actually undistorted upstream (the D405
    # color stream has real non-zero Brown-Conrady distortion, see
    # docs/getting_started_realsense.md).
    rgb = f"/{cam_id}/camera/color/image_rect"
    info = f"/{cam_id}/camera/color/camera_info"
    return rgb, info


def _stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def _K_from_camerainfo(msg: CameraInfo) -> np.ndarray:
    return np.array(msg.k, dtype=float).reshape(3, 3)


# Decode a color Image message to an OpenCV BGR array (handles padding + alpha + rgb/bgr).
def _img_to_numpy_bgr(msg: Image) -> np.ndarray:
    h, w = int(msg.height), int(msg.width)
    enc = msg.encoding.lower()

    if enc in ("rgb8", "bgr8"):
        channels = 3
    elif enc in ("bgra8", "rgba8"):
        channels = 4
    else:
        raise ValueError(f"Unsupported RGB encoding: {msg.encoding}")

    data = np.frombuffer(msg.data, dtype=np.uint8)
    step = int(msg.step)
    row_bytes = w * channels

    if step == row_bytes:
        img = data.reshape(h, w, channels)
    else:
        img = np.zeros((h, w, channels), dtype=np.uint8)
        for r in range(h):
            start = r * step
            img[r] = data[start:start + row_bytes].reshape(w, channels)

    if channels == 4:
        img = img[:, :, :3]
    if enc.startswith("rgb"):
        img = img[:, :, ::-1].copy()  # RGB -> BGR for OpenCV

    return img


def _rgb_numpy_to_imgmsg(bgr: np.ndarray, frame_id: str, stamp) -> Image:
    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
    msg = Image()
    msg.header = Header(frame_id=frame_id, stamp=stamp)
    msg.height = int(rgb.shape[0])
    msg.width = int(rgb.shape[1])
    msg.encoding = "rgb8"
    msg.is_bigendian = False
    msg.step = int(rgb.shape[1] * 3)
    msg.data = rgb.tobytes()
    return msg


def _make_objp() -> np.ndarray:
    objp = np.zeros((CHESS_ROWS * CHESS_COLS, 3), dtype=np.float32)
    grid = np.mgrid[0:CHESS_COLS, 0:CHESS_ROWS].T.reshape(-1, 2).astype(np.float32)
    objp[:, :2] = grid * float(SQUARE_SIZE_M)
    return objp


def _compute_reproj_err_px(
    objp: np.ndarray, corners: np.ndarray, K: np.ndarray, rvec: np.ndarray, tvec: np.ndarray
) -> float:
    dist = np.zeros((8, 1), dtype=np.float64)
    proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2)
    corners2 = corners.reshape(-1, 2)
    return float(np.linalg.norm(proj - corners2, axis=1).mean())


def _solve_board_pose(img_bgr: np.ndarray, K: np.ndarray) -> tuple[SE3, np.ndarray, float]:
    pattern_size = (CHESS_COLS, CHESS_ROWS)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if not ok or corners is None:
        raise RuntimeError("Chessboard NOT found")

    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), term)

    objp = _make_objp()
    dist = np.zeros((8, 1), dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(objp, corners, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise RuntimeError("solvePnP failed")

    reproj_err_px = _compute_reproj_err_px(objp, corners, K, rvec, tvec)
    R, _ = cv2.Rodrigues(rvec)
    return SE3(R, tvec.reshape(3)), corners.reshape(-1, 2), reproj_err_px


def _draw_chessboard(img_bgr: np.ndarray, corners: np.ndarray, text: str) -> np.ndarray:
    vis = img_bgr.copy()
    cv2.drawChessboardCorners(vis, (CHESS_COLS, CHESS_ROWS), corners.reshape(-1, 1, 2), True)
    cv2.putText(vis, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
    return vis


@dataclass
class HandEyeSample:
    idx: int
    T_base_flange: SE3
    T_cam_board: SE3
    reproj_px: float
    # Raw detected corner pixels + intrinsics at capture time. Optional so existing
    # legacy sample_*.json files (e.g. realsense_2's already-captured run) still load
    # fine -- cv2.calibrateHandEye never needed these, only T_base_flange/T_cam_board,
    # but a reprojection-error bundle adjustment (see joint_handeye_calib) does. New
    # captures always fill these in; a sample missing them falls back to a pose-level
    # residual there instead of a pixel one.
    corners_px: Optional[np.ndarray] = None
    K: Optional[np.ndarray] = None


def _sample_to_json_dict(s: HandEyeSample) -> dict:
    d = {
        "idx": s.idx,
        "reproj_px": s.reproj_px,
        "T_base_flange": {"R": s.T_base_flange.R.tolist(), "t": s.T_base_flange.t.tolist()},
        "T_cam_board": {"R": s.T_cam_board.R.tolist(), "t": s.T_cam_board.t.tolist()},
    }
    if s.corners_px is not None:
        d["corners_px"] = np.asarray(s.corners_px, dtype=float).tolist()
    if s.K is not None:
        d["K"] = np.asarray(s.K, dtype=float).reshape(3, 3).tolist()
    return d


def _sample_from_json_dict(d: dict) -> HandEyeSample:
    return HandEyeSample(
        idx=d["idx"],
        T_base_flange=SE3(np.array(d["T_base_flange"]["R"]), np.array(d["T_base_flange"]["t"])),
        T_cam_board=SE3(np.array(d["T_cam_board"]["R"]), np.array(d["T_cam_board"]["t"])),
        reproj_px=d["reproj_px"],
        corners_px=np.array(d["corners_px"], dtype=float) if "corners_px" in d else None,
        K=np.array(d["K"], dtype=float).reshape(3, 3) if "K" in d else None,
    )


def _save_sample_json(debug_dir: Path, sample: HandEyeSample) -> None:
    # Written immediately on capture (not just at the final solve) so a crash
    # in _solve_handeye or anywhere after capture doesn't lose the raw poses
    # -- only the debug PNG survived that before, which isn't enough to redo
    # the AX=XB solve without recapturing everything.
    path = debug_dir / f"sample_{sample.idx:02d}.json"
    path.write_text(json.dumps(_sample_to_json_dict(sample), indent=2))


def _load_samples_from_dir(debug_dir: Path) -> list[HandEyeSample]:
    samples = [
        _sample_from_json_dict(json.loads(p.read_text()))
        for p in sorted(debug_dir.glob("sample_*.json"))
    ]
    samples.sort(key=lambda s: s.idx)
    return samples


class HandEyeCalibNode(Node):
    def __init__(self, cam_id: str, flange_pose_topic: str, publish_debug: bool):
        super().__init__("handeye_flange_cam_realsense_" + cam_id)
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
                Image, f"/calibration/handeye/{cam_id}/debug_image", 1
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


def _try_capture_sample(node: HandEyeCalibNode, debug_dir: Path, sample_idx: int) -> Optional[HandEyeSample]:
    if not node.has_fresh_rgb_info():
        print("  [skip] no fresh RGB/CameraInfo yet")
        return None
    if not node.has_fresh_flange_pose():
        print(
            f"  [skip] no fresh flange pose (< {FLANGE_POSE_MAX_AGE_S}s old) on /iiwa/ee_pose -- "
            "is the robot/tf bringup running? (docs/getting_started_realsense.md section 5)"
        )
        return None

    img = _img_to_numpy_bgr(node.img_msg)
    K = node.K.copy()
    T_base_flange = node.flange_pose

    try:
        T_cam_board, corners, reproj_err = _solve_board_pose(img, K)
    except RuntimeError as e:
        print(f"  [skip] {e}")
        node.publish_debug(img)
        return None

    if reproj_err > MAX_REPROJ_ERR_PX:
        print(f"  [skip] reprojection error too high: {reproj_err:.3f}px > {MAX_REPROJ_ERR_PX}px")
        return None

    vis = _draw_chessboard(img, corners, f"sample {sample_idx} reproj={reproj_err:.3f}px")
    node.publish_debug(vis)
    cv2.imwrite(str(debug_dir / f"sample_{sample_idx:02d}.png"), vis)

    sample = HandEyeSample(
        idx=sample_idx, T_base_flange=T_base_flange, T_cam_board=T_cam_board, reproj_px=reproj_err,
        corners_px=corners, K=K,
    )
    _save_sample_json(debug_dir, sample)

    print(f"  [ok] reproj={reproj_err:.3f}px  T_base_flange.t={T_base_flange.t}")
    return sample


def _solve_handeye(samples: list[HandEyeSample]) -> tuple[SE3, np.ndarray]:
    """Solves AX=XB for T_flange_cam via cv2.calibrateHandEye.

    cv2's convention: given gripper2base (flange->base, i.e. T_base_flange)
    and target2cam (board->camera, i.e. T_cam_board... note cv2 wants
    board-in-camera, which is exactly what solvePnP already gave us as
    T_cam_board), it returns cam2gripper == T_flange_cam.
    """
    R_gripper2base = [s.T_base_flange.R for s in samples]
    t_gripper2base = [s.T_base_flange.t.reshape(3, 1) for s in samples]
    R_target2cam = [s.T_cam_board.R for s in samples]
    t_target2cam = [s.T_cam_board.t.reshape(3, 1) for s in samples]

    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base, t_gripper2base,
        R_target2cam, t_target2cam,
        method=cv2.CALIB_HAND_EYE_TSAI,
    )
    T_flange_cam = SE3(R_cam2gripper, t_cam2gripper.reshape(3))

    # Residual check: for every pair of samples, AX should equal XB.
    # Report per-pair rotation/translation residuals as a QA signal.
    residuals_deg = []
    residuals_m = []
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            A = samples[i].T_base_flange.inverse() @ samples[j].T_base_flange
            B = samples[i].T_cam_board @ samples[j].T_cam_board.inverse()
            lhs = A @ T_flange_cam
            rhs = T_flange_cam @ B
            R_err = lhs.R.T @ rhs.R
            c = np.clip((np.trace(R_err) - 1.0) * 0.5, -1.0, 1.0)
            residuals_deg.append(float(np.degrees(np.arccos(c))))
            residuals_m.append(float(np.linalg.norm(lhs.t - rhs.t)))

    return T_flange_cam, np.array(residuals_deg), np.array(residuals_m) if residuals_m else np.array([0.0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cam-id", required=True, choices=["realsense_1", "realsense_2"])
    parser.add_argument("--flange-pose-topic", default=FLANGE_POSE_TOPIC_DEFAULT)
    parser.add_argument("--num-samples", type=int, default=RECOMMENDED_SAMPLES)
    parser.add_argument("--min-samples", type=int, default=MIN_SAMPLES_TO_SOLVE)
    parser.add_argument("--no-debug-topic", action="store_true", help="Disable the Foxglove debug image publisher")
    parser.add_argument(
        "--solve-only", action="store_true",
        help="Skip capture entirely; reload sample_*.json already saved under "
             "outputs/calibration_debug/handeye/<cam_id>/ from a previous (e.g. crashed) "
             "run and just re-run the AX=XB solve + YAML write.",
    )
    args = parser.parse_args()

    if args.num_samples < args.min_samples:
        parser.error("--num-samples must be >= --min-samples")

    debug_dir = Path(DEBUG_DIR) / args.cam_id
    debug_dir.mkdir(parents=True, exist_ok=True)

    if args.solve_only:
        samples = _load_samples_from_dir(debug_dir)
        print(f"Loaded {len(samples)} saved sample(s) from {debug_dir}")
        if len(samples) < args.min_samples:
            raise RuntimeError(f"Not enough saved samples: {len(samples)} < {args.min_samples}")
        _solve_and_write(args.cam_id, samples)
        return

    rclpy.init()
    node = HandEyeCalibNode(args.cam_id, args.flange_pose_topic, publish_debug=not args.no_debug_topic)

    print("")
    print(f"=== Hand-eye calibration: {args.cam_id} (flange <-> camera) ===")
    print(f"Need >= {args.min_samples} accepted samples ({args.num_samples} recommended).")
    print("")
    print("Suggested Foxglove layout while collecting samples:")
    print(f"  - Image panel  -> /calibration/handeye/{args.cam_id}/debug_image")
    print("    (shows the detected checkerboard corners overlaid live, so you")
    print("     can confirm the board is fully visible before capturing)")
    print(f"  - Raw Messages panel -> {args.flange_pose_topic} (sanity-check the flange pose is streaming)")
    print("  - 3D panel with the robot model, to visually confirm each jogged pose")
    print("")
    print("For each sample: jog the arm with the MoveIt GUI (RViz MotionPlanning")
    print("panel -> plan & execute) to a NEW pose with the checkerboard fully")
    print("visible, let it settle, then press Enter here to capture.")
    print("Vary orientation as much as translation -- AX=XB needs rotational")
    print("diversity between samples to be well-conditioned.")
    print("Type 'q' + Enter instead to stop early (if you already have >= min).")
    print("")

    samples: list[HandEyeSample] = []
    while len(samples) < args.num_samples:
        rclpy.spin_once(node, timeout_sec=0.0)  # drain any pending callbacks first
        user_in = input(f"[{len(samples)}/{args.num_samples}] Press Enter to capture (or 'q' to finish early): ")
        if user_in.strip().lower() == "q":
            if len(samples) < args.min_samples:
                print(f"Need at least {args.min_samples} samples, only have {len(samples)}. Keep going.")
                continue
            break

        # Spin briefly so callbacks pick up the latest messages after the operator moved the arm.
        t_deadline = time.time() + 2.0
        while time.time() < t_deadline:
            rclpy.spin_once(node, timeout_sec=0.05)

        sample = _try_capture_sample(node, debug_dir, len(samples))
        if sample is not None:
            samples.append(sample)

    if len(samples) < args.min_samples:
        raise RuntimeError(f"Not enough accepted samples: {len(samples)} < {args.min_samples}")

    try:
        _solve_and_write(args.cam_id, samples)
    except Exception:
        print(f"\nSolve/write step failed, but all {len(samples)} samples are already saved as")
        print(f"JSON under {debug_dir} -- nothing needs to be recaptured. Fix the underlying")
        print(f"issue, then re-run with:")
        print(f"  python3 -m src.calibration.handeye_flange_cam_realsense --cam-id {args.cam_id} --solve-only")
        raise
    finally:
        node.destroy_node()
        rclpy.shutdown()


def _solve_and_write(cam_id: str, samples: list[HandEyeSample]) -> None:
    T_flange_cam, residuals_deg, residuals_m = _solve_handeye(samples)

    print("\n=== FINAL RESULT ===")
    print(f"T_flange_cam ({cam_id}):")
    print(T_flange_cam)
    print("\nAX=XB pairwise residuals (consistency across sample pairs):")
    print(f"  rotation:    mean={residuals_deg.mean():.4f} deg  max={residuals_deg.max():.4f} deg")
    print(f"  translation: mean={residuals_m.mean():.6f} m    max={residuals_m.max():.6f} m")
    print("Large residuals usually mean too little rotational diversity between")
    print("samples, or the flange-pose/RGB timestamps didn't correspond to the")
    print("same physical arm position (moved too soon after pressing Enter).")

    out_path = Path(OUT_YAML)
    if out_path.exists():
        backup = out_path.with_suffix(".yaml.bak")
        backup.write_text(out_path.read_text())
        print(f"\nBacked up existing YAML to: {backup}")

    update_extrinsics_yaml_preserving_header(out_path, {cam_id: T_flange_cam})
    print(f"Wrote T_flange_cam for '{cam_id}' into: {out_path}")
    print(f"Saved debug corner images + pose data to: {Path(DEBUG_DIR) / cam_id}")


if __name__ == "__main__":
    main()
