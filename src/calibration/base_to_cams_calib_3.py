from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
import yaml

from src.utils.se3 import SE3
from src.calibration.io_extrinsics import save_extrinsics_yaml
from src.calibration.calibration_log import log_camera_transform
from src.utils.robot_bases import get_active_robot_base


# ---------------- USER SETTINGS ----------------
# Default camera set: the original 3-ZED trio. --cam-ids on the CLI
# overrides this with an arbitrary N>=1 list (e.g. --cam-ids zed2i_1 for the
# 1-ZED rig in the RealSense-trio variant, see
# scripts/calibrate_zed_from_board_pose.sh) -- ZED topic naming is
# regular (zed_node/rgb/color/rect/...) so topics are derived from cam_id,
# not hand-listed per camera like the old CAM1/CAM2/CAM3 constants were.
DEFAULT_CAM_IDS = ["zed2i_1", "zed2i_2", "zed2i_3"]


def _zed_topics(cam_id: str) -> tuple[str, str]:
    return (
        f"/{cam_id}/zed_node/rgb/color/rect/image",
        f"/{cam_id}/zed_node/rgb/color/rect/camera_info",
    )

CHESS_COLS = 8   # number of INNER corners along x
CHESS_ROWS = 11  # number of INNER corners along y
SQUARE_SIZE_M = 0.03

# freshness/sync thresholds
SYNC_SLOP_S = 0.05              # cam1 RGB vs cam2 RGB vs cam3 RGB
RGB_INFO_MAX_DT_S = 0.50        # per camera: RGB vs CameraInfo
MAX_WAIT_S = 60.0               # max time waiting for valid triple
PRINT_EVERY_S = 5.0

# multi-sample robustness
NUM_SAMPLES = 8                 # number of accepted 3-camera captures to average
MIN_SAMPLES_TO_SOLVE = 5
INTER_SAMPLE_MIN_DT_S = 0.4     # avoid taking near-identical frames
MAX_REPROJ_ERR_PX = 2.5        # reject single sample if too high
MAX_FINAL_TRANSLATION_STD_M = 0.01
MAX_FINAL_ROTATION_STD_DEG = 1.0

BASE_BOARD_YAML = "config/base_board_pose.yaml"
OUT_YAML = "config/camera_extrinsics_base.yaml"
# Same poses, re-expressed in robot_a's frame (global reference) -- written
# as a separate sibling file (not extra keys in OUT_YAML) because OUT_YAML
# uses the flat cam_id -> {R,t} format that load_extrinsics_yaml() reads
# every top-level key from; mixing in "_robot_a_frame" keys there would get
# silently picked up as if they were extra cameras. Reference/documentation
# only -- the live pipeline computes this same conversion itself at runtime
# from OUT_YAML + config/robot_bases.yaml (see run_pipeline_track_multicam.py
# and run_pipeline_track_multicam_realsense.py), so this file is not read by
# anything; it exists so the robot_a-frame numbers are on disk, not just
# printed to a terminal that may not be saved.
OUT_YAML_ROBOT_A_FRAME = "config/camera_extrinsics_base_robot_a_frame.yaml"
DEBUG_DIR = "outputs/calibration_debug"
# ------------------------------------------------


def _stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def _K_from_camerainfo(msg: CameraInfo) -> np.ndarray:
    return np.array(msg.k, dtype=float).reshape(3, 3)


# Decode a color Image message to an OpenCV BGR array (handles padding + alpha + rgb/bgr).
def _img_to_numpy_color(msg: Image) -> np.ndarray:
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


# Roll/pitch/yaw (degrees) → rotation matrix using the Rz·Ry·Rx convention.
def _rpy_deg_to_R(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    r = np.deg2rad(roll_deg)
    p = np.deg2rad(pitch_deg)
    y = np.deg2rad(yaw_deg)

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(r), -np.sin(r)],
        [0, np.sin(r),  np.cos(r)],
    ], dtype=float)

    Ry = np.array([
        [ np.cos(p), 0, np.sin(p)],
        [0,          1, 0],
        [-np.sin(p), 0, np.cos(p)],
    ], dtype=float)

    Rz = np.array([
        [np.cos(y), -np.sin(y), 0],
        [np.sin(y),  np.cos(y), 0],
        [0,          0,         1],
    ], dtype=float)

    return Rz @ Ry @ Rx


# Read the board's known pose in the robot base frame from YAML.
def _load_T_base_board(path: Path) -> SE3:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    bb = cfg["base_board"]

    t = np.array(bb["translation_xyz_m"], dtype=float)
    rpy = bb["rotation_rpy_deg"]
    R = _rpy_deg_to_R(float(rpy[0]), float(rpy[1]), float(rpy[2]))

    return SE3(R, t)


# 3D coordinates of the checkerboard's inner corners in the board frame (z=0 plane).
def _make_objp() -> np.ndarray:
    objp = np.zeros((CHESS_ROWS * CHESS_COLS, 3), dtype=np.float32)
    grid = np.mgrid[0:CHESS_COLS, 0:CHESS_ROWS].T.reshape(-1, 2).astype(np.float32)
    objp[:, :2] = grid * float(SQUARE_SIZE_M)
    return objp


def _compute_reproj_err_px(
    objp: np.ndarray,
    corners: np.ndarray,
    K: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> float:
    # Mean pixel distance between the detected corners and the reprojected model corners.
    dist = np.zeros((8, 1), dtype=np.float64)
    proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2)
    corners2 = corners.reshape(-1, 2)
    err = np.linalg.norm(proj - corners2, axis=1)
    return float(err.mean())


def _solve_board_pose(img_bgr: np.ndarray, K: np.ndarray) -> Tuple[SE3, np.ndarray, float]:
    # Find chessboard corners, refine to sub-pixel, then solve the board→camera pose.
    pattern_size = (CHESS_COLS, CHESS_ROWS)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if not ok or corners is None:
        raise RuntimeError("Chessboard NOT found")

    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), term)

    # PnP from the known 3D corners to the detected 2D corners.
    objp = _make_objp()
    dist = np.zeros((8, 1), dtype=np.float64)

    ok, rvec, tvec = cv2.solvePnP(objp, corners, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise RuntimeError("solvePnP failed")

    reproj_err_px = _compute_reproj_err_px(objp, corners, K, rvec, tvec)

    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3)

    return SE3(R, t), corners.reshape(-1, 2), reproj_err_px


def _draw_chessboard(img_bgr: np.ndarray, corners: np.ndarray, text: str) -> np.ndarray:
    vis = img_bgr.copy()
    pattern_size = (CHESS_COLS, CHESS_ROWS)
    cv2.drawChessboardCorners(vis, pattern_size, corners.reshape(-1, 1, 2), True)
    cv2.putText(
        vis,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return vis


def _rotation_matrix_to_rotvec(R: np.ndarray) -> np.ndarray:
    rvec, _ = cv2.Rodrigues(R.astype(np.float64))
    return rvec.reshape(3)


def _rotvec_to_rotation_matrix(rvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(rvec.reshape(3, 1).astype(np.float64))
    return R


# Average a list of SE3 poses; also return the translation/rotation spread for QA.
def _average_se3(poses: list[SE3]) -> Tuple[SE3, float, float]:
    if len(poses) == 0:
        raise ValueError("No poses to average")

    # Translation: plain mean + spread.
    ts = np.stack([p.t for p in poses], axis=0)
    t_mean = ts.mean(axis=0)
    t_std = float(np.linalg.norm(ts.std(axis=0)))

    # Rotation: mean in rotation-vector space, then back to a matrix.
    rvecs = np.stack([_rotation_matrix_to_rotvec(p.R) for p in poses], axis=0)
    rvec_mean = rvecs.mean(axis=0)
    R_mean = _rotvec_to_rotation_matrix(rvec_mean)

    # Rotation spread = std of each sample's geodesic angle from the mean.
    ang_devs = []
    for p in poses:
        R_rel = R_mean.T @ p.R
        c = np.clip((np.trace(R_rel) - 1.0) * 0.5, -1.0, 1.0)
        ang = np.degrees(np.arccos(c))
        ang_devs.append(float(ang))
    rot_std_deg = float(np.std(ang_devs))

    return SE3(R_mean, t_mean), t_std, rot_std_deg


@dataclass
class CamState:
    name: str
    expected_rgb_frame_id: Optional[str] = None
    expected_info_frame_id: Optional[str] = None

    img_msg: Optional[Image] = None
    img_t: Optional[float] = None
    img_frame_id: Optional[str] = None

    info_msg: Optional[CameraInfo] = None
    info_t: Optional[float] = None
    info_frame_id: Optional[str] = None

    K: Optional[np.ndarray] = None

    rgb_count: int = 0
    info_count: int = 0
    bad_frame_id_count: int = 0

    def update_img(self, msg: Image) -> None:
        self.img_msg = msg
        self.img_t = _stamp_to_sec(msg.header.stamp)
        self.img_frame_id = msg.header.frame_id
        self.rgb_count += 1

        if self.expected_rgb_frame_id is not None and msg.header.frame_id != self.expected_rgb_frame_id:
            self.bad_frame_id_count += 1
            raise RuntimeError(
                f"[{self.name}] Unexpected RGB frame_id '{msg.header.frame_id}', "
                f"expected '{self.expected_rgb_frame_id}'"
            )

    def update_info(self, msg: CameraInfo) -> None:
        self.info_msg = msg
        self.info_t = _stamp_to_sec(msg.header.stamp)
        self.info_frame_id = msg.header.frame_id
        self.K = _K_from_camerainfo(msg)
        self.info_count += 1

        if self.expected_info_frame_id is not None and msg.header.frame_id != self.expected_info_frame_id:
            self.bad_frame_id_count += 1
            raise RuntimeError(
                f"[{self.name}] Unexpected CameraInfo frame_id '{msg.header.frame_id}', "
                f"expected '{self.expected_info_frame_id}'"
            )

    def has_fresh_pair(self) -> bool:
        # True when this camera has an image + intrinsics with close timestamps.
        if self.img_msg is None or self.K is None or self.img_t is None or self.info_t is None:
            return False
        return abs(self.img_t - self.info_t) <= RGB_INFO_MAX_DT_S


class CheckerboardBaseCalib(Node):
    """N-camera variant (N >= 1): every camera in `cam_ids` must see the
    checkerboard in the same synced sample for that sample to be accepted.
    cam_ids order is preserved throughout (accepted[i].T_base_cam[cam_id],
    the printed report, and the output YAML all key off cam_id, not a fixed
    cam1/cam2/cam3 slot)."""

    def __init__(self, cam_ids: list[str]):
        super().__init__("checkerboard_base_to_cameras_calib")
        self.cam_ids = list(cam_ids)
        self.cams: dict[str, CamState] = {cam_id: CamState(name=cam_id) for cam_id in cam_ids}

        for cam_id in cam_ids:
            rgb_topic, info_topic = _zed_topics(cam_id)
            self.create_subscription(
                Image, rgb_topic, lambda msg, c=cam_id: self._on_img(c, msg), 10
            )
            self.create_subscription(
                CameraInfo, info_topic, lambda msg, c=cam_id: self._on_info(c, msg), 10
            )

    def _on_img(self, cam_id: str, msg: Image) -> None:
        try:
            self.cams[cam_id].update_img(msg)
        except RuntimeError as e:
            self.get_logger().error(str(e))

    def _on_info(self, cam_id: str, msg: CameraInfo) -> None:
        try:
            self.cams[cam_id].update_info(msg)
        except RuntimeError as e:
            self.get_logger().error(str(e))

    def ready(self) -> bool:
        return all(self.cams[c].has_fresh_pair() for c in self.cam_ids)

    def get_synced_set(self) -> Optional[dict[str, tuple[np.ndarray, np.ndarray, float]]]:
        """Returns {cam_id: (img_bgr, K, rgb_info_dt_s)} if every camera has a
        fresh RGB/info pair AND all cameras' images share one timestamp
        window <= SYNC_SLOP_S; None otherwise."""
        if not self.ready():
            return None

        img_ts = [self.cams[c].img_t for c in self.cam_ids]
        dts = {c: abs(self.cams[c].img_t - self.cams[c].info_t) for c in self.cam_ids}
        if any(dt > RGB_INFO_MAX_DT_S for dt in dts.values()):
            return None

        cross_dt = max(img_ts) - min(img_ts)
        if cross_dt > SYNC_SLOP_S:
            return None

        return {
            c: (_img_to_numpy_color(self.cams[c].img_msg), self.cams[c].K.copy(), dts[c])
            for c in self.cam_ids
        }

    def print_status(self) -> None:
        img_ts = [self.cams[c].img_t for c in self.cam_ids if self.cams[c].img_t is not None]
        if len(img_ts) == len(self.cam_ids):
            cross_dt = max(img_ts) - min(img_ts)
            self.get_logger().info(f"rgb_cross_dt={cross_dt:.4f}s (limit {SYNC_SLOP_S:.4f}s)")
        for cam_id in self.cam_ids:
            st = self.cams[cam_id]
            if st.img_t is not None and st.info_t is not None:
                self.get_logger().info(
                    f"{cam_id}: rgb_count={st.rgb_count}, info_count={st.info_count}, "
                    f"img_frame_id={st.img_frame_id}, info_frame_id={st.info_frame_id}, "
                    f"rgb_info_dt={abs(st.img_t - st.info_t):.4f}s"
                )
            else:
                self.get_logger().info(f"{cam_id}: waiting...")


@dataclass
class SampleResult:
    idx: int
    T_cam_board: dict[str, SE3]
    T_base_cam: dict[str, SE3]
    reproj_px: dict[str, float]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--cam-ids", nargs="+", default=None,
        help=f"Camera cam_ids to calibrate, e.g. --cam-ids zed2i_1 for a single-ZED rig. "
             f"Topics are derived as /<cam_id>/zed_node/rgb/color/rect/{{image,camera_info}}. "
             f"Defaults to the original 3-ZED trio {DEFAULT_CAM_IDS} if omitted.",
    )
    args = parser.parse_args()
    cam_ids = args.cam_ids or DEFAULT_CAM_IDS

    rclpy.init()
    node = CheckerboardBaseCalib(cam_ids)

    debug_dir = Path(DEBUG_DIR)
    debug_dir.mkdir(parents=True, exist_ok=True)

    # The board's known pose in the base frame anchors every camera solve.
    T_base_board = _load_T_base_board(Path(BASE_BOARD_YAML))
    print("Loaded T_base_board:")
    print(T_base_board)

    print("")
    print(f"=== Robust {len(cam_ids)}-camera base calibration: {cam_ids} ===")
    print(f"Need {NUM_SAMPLES} accepted {len(cam_ids)}-camera samples")
    print(f"Show the checkerboard to {'ALL' if len(cam_ids) > 1 else 'THE'} camera(s) and hold it steady.")
    print("This script will reject stale, unsynced, or poor-quality captures.")
    print("")

    accepted: list[SampleResult] = []
    t_start = time.time()
    t_last_print = 0.0
    t_last_reject_print = 0.0
    last_accept_t = -1e9

    # Collect NUM_SAMPLES good synced sets, solving the board pose in each.
    while len(accepted) < NUM_SAMPLES:
        rclpy.spin_once(node, timeout_sec=0.05)

        now = time.time()
        if now - t_start > MAX_WAIT_S:
            raise RuntimeError(f"Timed out waiting for enough valid {len(cam_ids)}-camera captures.")

        if now - t_last_print > PRINT_EVERY_S:
            print(f"[status] accepted {len(accepted)}/{NUM_SAMPLES}")
            node.print_status()
            t_last_print = now

        synced = node.get_synced_set()
        if synced is None:
            continue

        # Space out samples so we don't average near-identical frames.
        if now - last_accept_t < INTER_SAMPLE_MIN_DT_S:
            continue

        # Solve the board pose per camera; any failed detection rejects the whole set.
        T_cam_board: dict[str, SE3] = {}
        reproj_px: dict[str, float] = {}
        corners_by_cam: dict[str, np.ndarray] = {}
        rejected = False
        for cam_id in cam_ids:
            img, K, _dt = synced[cam_id]
            try:
                T_cam_board[cam_id], corners_by_cam[cam_id], reproj_px[cam_id] = _solve_board_pose(img, K)
            except RuntimeError as e:
                if now - t_last_reject_print > 1.0:
                    print(f"[reject] {cam_id}: {e}")
                    t_last_reject_print = now
                rejected = True
                break
        if rejected:
            continue

        if any(r > MAX_REPROJ_ERR_PX for r in reproj_px.values()):
            print(
                f"[reject] reproj too high: "
                f"{ {c: round(r, 3) for c, r in reproj_px.items()} } "
                f"(limit {MAX_REPROJ_ERR_PX:.3f}px)"
            )
            continue

        # T_base_cam = T_base_board · (T_cam_board)⁻¹ for each camera.
        T_base_cam = {c: T_base_board @ T_cam_board[c].inverse() for c in cam_ids}

        sample = SampleResult(
            idx=len(accepted), T_cam_board=T_cam_board, T_base_cam=T_base_cam, reproj_px=reproj_px,
        )
        accepted.append(sample)
        last_accept_t = now

        dts_str = " ".join(f"{c}={synced[c][2]:.4f}s" for c in cam_ids)
        reproj_str = " ".join(f"{c}={reproj_px[c]:.3f}px" for c in cam_ids)
        print(f"[accept {len(accepted)}/{NUM_SAMPLES}] rgb-info dt: {dts_str} | reproj: {reproj_str}")

        # Save annotated corner images for visual inspection.
        for cam_id in cam_ids:
            img, _K, _dt = synced[cam_id]
            vis = _draw_chessboard(
                img, corners_by_cam[cam_id],
                f"{cam_id} sample {len(accepted)} reproj={reproj_px[cam_id]:.3f}px",
            )
            cv2.imwrite(str(debug_dir / f"{cam_id}_sample_{len(accepted):02d}.png"), vis)

    if len(accepted) < MIN_SAMPLES_TO_SOLVE:
        raise RuntimeError(f"Not enough accepted samples: {len(accepted)} < {MIN_SAMPLES_TO_SOLVE}")

    # Average each camera's extrinsic across all accepted samples.
    T_base_cam_avg: dict[str, SE3] = {}
    tstd: dict[str, float] = {}
    rstd: dict[str, float] = {}
    reproj_mean: dict[str, float] = {}
    for cam_id in cam_ids:
        poses = [s.T_base_cam[cam_id] for s in accepted]
        T_base_cam_avg[cam_id], tstd[cam_id], rstd[cam_id] = _average_se3(poses)
        reproj_mean[cam_id] = float(np.mean([s.reproj_px[cam_id] for s in accepted]))

    print("\n=== FINAL RESULTS ===")
    active_robot, T_robotA_activeRobot = get_active_robot_base()
    T_robotA_cam = {c: T_robotA_activeRobot.compose(T_base_cam_avg[c]) for c in cam_ids}

    for cam_id in cam_ids:
        print(f"\nT_base_{cam_id} (averaged, base = {active_robot}'s lbr_link_0):")
        print(T_base_cam_avg[cam_id])

    print("\nSame poses in robot_a's frame (global reference):")
    for cam_id in cam_ids:
        print(f"T_robotA_{cam_id}:\n{T_robotA_cam[cam_id]}")

    print("\nQuality metrics:")
    for cam_id in cam_ids:
        print(f"{cam_id} mean reproj error: {reproj_mean[cam_id]:.4f} px")
        print(f"{cam_id} translation std:   {tstd[cam_id]:.6f} m")
        print(f"{cam_id} rotation std:      {rstd[cam_id]:.6f} deg")

    # Relative camera transforms — a sanity check on inter-camera consistency
    # (only meaningful with >= 2 cameras).
    if len(cam_ids) >= 2:
        ref = cam_ids[0]
        print(f"\nConsistency checks relative to {ref}:")
        for cam_id in cam_ids[1:]:
            T_check = T_base_cam_avg[ref].inverse() @ T_base_cam_avg[cam_id]
            print(f"T_{ref}_{cam_id}:\n{T_check}")
            print("R det:", np.linalg.det(T_check.R), "valid:", T_check.is_valid())

    # Refuse to write a result whose sample spread is too loose to trust.
    bad_t = {c: v for c, v in tstd.items() if v > MAX_FINAL_TRANSLATION_STD_M}
    if bad_t:
        raise RuntimeError(f"Calibration rejected: translation spread too high ({bad_t})")

    bad_r = {c: v for c, v in rstd.items() if v > MAX_FINAL_ROTATION_STD_DEG}
    if bad_r:
        raise RuntimeError(f"Calibration rejected: rotation spread too high ({bad_r})")

    # Back up any existing extrinsics before overwriting.
    out_path = Path(OUT_YAML)
    if out_path.exists():
        backup = out_path.with_suffix(".yaml.bak")
        backup.write_text(out_path.read_text())
        print(f"Backed up existing YAML to: {backup}")

    save_extrinsics_yaml(out_path, T_base_cam_avg)
    print(f"\nWrote base-referenced camera extrinsics to: {out_path}")

    out_path_robot_a = Path(OUT_YAML_ROBOT_A_FRAME)
    if out_path_robot_a.exists():
        backup_a = out_path_robot_a.with_suffix(".yaml.bak")
        backup_a.write_text(out_path_robot_a.read_text())
        print(f"Backed up existing YAML to: {backup_a}")
    save_extrinsics_yaml(out_path_robot_a, T_robotA_cam)
    print(f"Wrote robot_a-frame camera extrinsics (reference only) to: {out_path_robot_a}")

    print(f"Saved debug corner images to: {debug_dir}")

    for cam_id in cam_ids:
        log_camera_transform({
            "stage": "base_to_cams_static",
            "cam_id": cam_id,
            "active_robot": active_robot,
            "num_samples": len(accepted),
            "T_base_cam": {"R": T_base_cam_avg[cam_id].R.tolist(), "t": T_base_cam_avg[cam_id].t.tolist()},
            "T_robotA_cam": {"R": T_robotA_cam[cam_id].R.tolist(), "t": T_robotA_cam[cam_id].t.tolist()},
            "reproj_err_px_mean": reproj_mean[cam_id],
            "translation_std_m": tstd[cam_id],
            "rotation_std_deg": rstd[cam_id],
        })

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
