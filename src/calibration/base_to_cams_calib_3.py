from __future__ import annotations

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


# ---------------- USER SETTINGS ----------------
CAM1 = "zed2i_1"
CAM2 = "zed2i_2"
CAM3 = "zed2i_3"

CAM1_RGB = "/zed2i_1/zed_node/rgb/color/rect/image"
CAM2_RGB = "/zed2i_2/zed_node/rgb/color/rect/image"
CAM3_RGB = "/zed2i_3/zed_node/rgb/color/rect/image"
CAM1_INFO = "/zed2i_1/zed_node/rgb/color/rect/camera_info"
CAM2_INFO = "/zed2i_2/zed_node/rgb/color/rect/camera_info"
CAM3_INFO = "/zed2i_3/zed_node/rgb/color/rect/camera_info"

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

# optional frame-id sanity checks; set to None to disable
EXPECTED_CAM1_RGB_FRAME_ID = None   # e.g. "zed2i_1_left_camera_frame"
EXPECTED_CAM2_RGB_FRAME_ID = None
EXPECTED_CAM3_RGB_FRAME_ID = None
EXPECTED_CAM1_INFO_FRAME_ID = None
EXPECTED_CAM2_INFO_FRAME_ID = None
EXPECTED_CAM3_INFO_FRAME_ID = None

BASE_BOARD_YAML = "config/base_board_pose.yaml"
OUT_YAML = "config/camera_extrinsics_base.yaml"
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
    def __init__(self):
        super().__init__("checkerboard_base_to_cameras_calib")

        self.cam1 = CamState(
            name=CAM1,
            expected_rgb_frame_id=EXPECTED_CAM1_RGB_FRAME_ID,
            expected_info_frame_id=EXPECTED_CAM1_INFO_FRAME_ID,
        )
        self.cam2 = CamState(
            name=CAM2,
            expected_rgb_frame_id=EXPECTED_CAM2_RGB_FRAME_ID,
            expected_info_frame_id=EXPECTED_CAM2_INFO_FRAME_ID,
        )
        self.cam3 = CamState(
            name=CAM3,
            expected_rgb_frame_id=EXPECTED_CAM3_RGB_FRAME_ID,
            expected_info_frame_id=EXPECTED_CAM3_INFO_FRAME_ID,
        )

        # Subscribe each camera's RGB image and CameraInfo.
        self.create_subscription(Image, CAM1_RGB, self._on_img1, 10)
        self.create_subscription(Image, CAM2_RGB, self._on_img2, 10)
        self.create_subscription(Image, CAM3_RGB, self._on_img3, 10)
        self.create_subscription(CameraInfo, CAM1_INFO, self._on_info1, 10)
        self.create_subscription(CameraInfo, CAM2_INFO, self._on_info2, 10)
        self.create_subscription(CameraInfo, CAM3_INFO, self._on_info3, 10)

    def _on_img1(self, msg: Image) -> None:
        try:
            self.cam1.update_img(msg)
        except RuntimeError as e:
            self.get_logger().error(str(e))

    def _on_img2(self, msg: Image) -> None:
        try:
            self.cam2.update_img(msg)
        except RuntimeError as e:
            self.get_logger().error(str(e))

    def _on_img3(self, msg: Image) -> None:
        try:
            self.cam3.update_img(msg)
        except RuntimeError as e:
            self.get_logger().error(str(e))

    def _on_info1(self, msg: CameraInfo) -> None:
        try:
            self.cam1.update_info(msg)
        except RuntimeError as e:
            self.get_logger().error(str(e))

    def _on_info2(self, msg: CameraInfo) -> None:
        try:
            self.cam2.update_info(msg)
        except RuntimeError as e:
            self.get_logger().error(str(e))

    def _on_info3(self, msg: CameraInfo) -> None:
        try:
            self.cam3.update_info(msg)
        except RuntimeError as e:
            self.get_logger().error(str(e))

    def ready(self) -> bool:
        return (
            self.cam1.has_fresh_pair()
            and self.cam2.has_fresh_pair()
            and self.cam3.has_fresh_pair()
        )

    def get_synced_triple(
        self,
    ) -> Optional[
        Tuple[
            np.ndarray, np.ndarray, np.ndarray,
            np.ndarray, np.ndarray, np.ndarray,
            float, float, float,
        ]
    ]:
        if not self.ready():
            return None

        # Per-camera RGB↔info must be fresh.
        dt_cam1 = abs(self.cam1.img_t - self.cam1.info_t)
        dt_cam2 = abs(self.cam2.img_t - self.cam2.info_t)
        dt_cam3 = abs(self.cam3.img_t - self.cam3.info_t)
        if dt_cam1 > RGB_INFO_MAX_DT_S or dt_cam2 > RGB_INFO_MAX_DT_S or dt_cam3 > RGB_INFO_MAX_DT_S:
            return None

        # And the three cameras' images must share a timestamp window.
        min_t = min(self.cam1.img_t, self.cam2.img_t, self.cam3.img_t)
        max_t = max(self.cam1.img_t, self.cam2.img_t, self.cam3.img_t)
        cross_dt = max_t - min_t
        if cross_dt > SYNC_SLOP_S:
            return None

        img1 = _img_to_numpy_color(self.cam1.img_msg)
        img2 = _img_to_numpy_color(self.cam2.img_msg)
        img3 = _img_to_numpy_color(self.cam3.img_msg)
        return (
            img1, img2, img3,
            self.cam1.K.copy(), self.cam2.K.copy(), self.cam3.K.copy(),
            dt_cam1, dt_cam2, dt_cam3,
        )

    def print_status(self) -> None:
        if self.cam1.img_t is not None and self.cam2.img_t is not None and self.cam3.img_t is not None:
            cross_dt = max(self.cam1.img_t, self.cam2.img_t, self.cam3.img_t) - min(
                self.cam1.img_t, self.cam2.img_t, self.cam3.img_t
            )
            self.get_logger().info(
                f"rgb_cross_dt={cross_dt:.4f}s (limit {SYNC_SLOP_S:.4f}s)"
            )
        self.get_logger().info(
            f"{CAM1}: rgb_count={self.cam1.rgb_count}, info_count={self.cam1.info_count}, "
            f"img_frame_id={self.cam1.img_frame_id}, info_frame_id={self.cam1.info_frame_id}, "
            f"rgb_info_dt={None if self.cam1.img_t is None or self.cam1.info_t is None else abs(self.cam1.img_t - self.cam1.info_t):.4f}s"
            if (self.cam1.img_t is not None and self.cam1.info_t is not None)
            else f"{CAM1}: waiting..."
        )
        self.get_logger().info(
            f"{CAM2}: rgb_count={self.cam2.rgb_count}, info_count={self.cam2.info_count}, "
            f"img_frame_id={self.cam2.img_frame_id}, info_frame_id={self.cam2.info_frame_id}, "
            f"rgb_info_dt={None if self.cam2.img_t is None or self.cam2.info_t is None else abs(self.cam2.img_t - self.cam2.info_t):.4f}s"
            if (self.cam2.img_t is not None and self.cam2.info_t is not None)
            else f"{CAM2}: waiting..."
        )
        self.get_logger().info(
            f"{CAM3}: rgb_count={self.cam3.rgb_count}, info_count={self.cam3.info_count}, "
            f"img_frame_id={self.cam3.img_frame_id}, info_frame_id={self.cam3.info_frame_id}, "
            f"rgb_info_dt={None if self.cam3.img_t is None or self.cam3.info_t is None else abs(self.cam3.img_t - self.cam3.info_t):.4f}s"
            if (self.cam3.img_t is not None and self.cam3.info_t is not None)
            else f"{CAM3}: waiting..."
        )


@dataclass
class SampleResult:
    idx: int
    T_cam1_board: SE3
    T_cam2_board: SE3
    T_cam3_board: SE3
    T_base_cam1: SE3
    T_base_cam2: SE3
    T_base_cam3: SE3
    reproj1_px: float
    reproj2_px: float
    reproj3_px: float
    img1: np.ndarray
    img2: np.ndarray
    img3: np.ndarray
    corners1: np.ndarray
    corners2: np.ndarray
    corners3: np.ndarray


def main() -> None:
    rclpy.init()
    node = CheckerboardBaseCalib()

    debug_dir = Path(DEBUG_DIR)
    debug_dir.mkdir(parents=True, exist_ok=True)

    # The board's known pose in the base frame anchors every camera solve.
    T_base_board = _load_T_base_board(Path(BASE_BOARD_YAML))
    print("Loaded T_base_board:")
    print(T_base_board)

    print("")
    print("=== Robust 3-camera base calibration ===")
    print(f"Need {NUM_SAMPLES} accepted 3-camera samples")
    print("Show the checkerboard to ALL THREE cameras and hold it steady.")
    print("This script will reject stale, unsynced, or poor-quality captures.")
    print("")

    accepted: list[SampleResult] = []
    t_start = time.time()
    t_last_print = 0.0
    t_last_reject_print = 0.0
    last_accept_t = -1e9

    # Collect NUM_SAMPLES good synced triples, solving the board pose in each.
    while len(accepted) < NUM_SAMPLES:
        rclpy.spin_once(node, timeout_sec=0.05)

        now = time.time()
        if now - t_start > MAX_WAIT_S:
            raise RuntimeError("Timed out waiting for enough valid 3-camera captures.")

        if now - t_last_print > PRINT_EVERY_S:
            print(f"[status] accepted {len(accepted)}/{NUM_SAMPLES}")
            node.print_status()
            t_last_print = now

        triple = node.get_synced_triple()
        if triple is None:
            continue

        # Space out samples so we don't average near-identical frames.
        if now - last_accept_t < INTER_SAMPLE_MIN_DT_S:
            continue

        img1, img2, img3, K1, K2, K3, dt1, dt2, dt3 = triple

        # Solve the board pose per camera; any failed detection rejects the whole triple.
        try:
            T_cam1_board, corners1, reproj1 = _solve_board_pose(img1, K1)
        except RuntimeError as e:
            if now - t_last_reject_print > 1.0:
                print(f"[reject] {CAM1}: {e}")
                t_last_reject_print = now
            continue

        try:
            T_cam2_board, corners2, reproj2 = _solve_board_pose(img2, K2)
        except RuntimeError as e:
            if now - t_last_reject_print > 1.0:
                print(f"[reject] {CAM2}: {e}")
                t_last_reject_print = now
            continue

        try:
            T_cam3_board, corners3, reproj3 = _solve_board_pose(img3, K3)
        except RuntimeError as e:
            if now - t_last_reject_print > 1.0:
                print(f"[reject] {CAM3}: {e}")
                t_last_reject_print = now
            continue

        # Reject the triple if any camera's reprojection error is too high.
        if (
            reproj1 > MAX_REPROJ_ERR_PX
            or reproj2 > MAX_REPROJ_ERR_PX
            or reproj3 > MAX_REPROJ_ERR_PX
        ):
            print(
                f"[reject] reproj too high: cam1={reproj1:.3f}px "
                f"cam2={reproj2:.3f}px cam3={reproj3:.3f}px "
                f"(limit {MAX_REPROJ_ERR_PX:.3f}px)"
            )
            continue

        # T_base_cam = T_base_board · (T_cam_board)⁻¹ for each camera.
        T_base_cam1 = T_base_board @ T_cam1_board.inverse()
        T_base_cam2 = T_base_board @ T_cam2_board.inverse()
        T_base_cam3 = T_base_board @ T_cam3_board.inverse()

        sample = SampleResult(
            idx=len(accepted),
            T_cam1_board=T_cam1_board,
            T_cam2_board=T_cam2_board,
            T_cam3_board=T_cam3_board,
            T_base_cam1=T_base_cam1,
            T_base_cam2=T_base_cam2,
            T_base_cam3=T_base_cam3,
            reproj1_px=reproj1,
            reproj2_px=reproj2,
            reproj3_px=reproj3,
            img1=img1,
            img2=img2,
            img3=img3,
            corners1=corners1,
            corners2=corners2,
            corners3=corners3,
        )
        accepted.append(sample)
        last_accept_t = now

        print(
            f"[accept {len(accepted)}/{NUM_SAMPLES}] "
            f"rgb-info dt: cam1={dt1:.4f}s cam2={dt2:.4f}s cam3={dt3:.4f}s | "
            f"reproj: cam1={reproj1:.3f}px cam2={reproj2:.3f}px cam3={reproj3:.3f}px"
        )

        # Save annotated corner images for visual inspection.
        vis1 = _draw_chessboard(img1, corners1, f"{CAM1} sample {len(accepted)} reproj={reproj1:.3f}px")
        vis2 = _draw_chessboard(img2, corners2, f"{CAM2} sample {len(accepted)} reproj={reproj2:.3f}px")
        vis3 = _draw_chessboard(img3, corners3, f"{CAM3} sample {len(accepted)} reproj={reproj3:.3f}px")
        cv2.imwrite(str(debug_dir / f"{CAM1}_sample_{len(accepted):02d}.png"), vis1)
        cv2.imwrite(str(debug_dir / f"{CAM2}_sample_{len(accepted):02d}.png"), vis2)
        cv2.imwrite(str(debug_dir / f"{CAM3}_sample_{len(accepted):02d}.png"), vis3)

    if len(accepted) < MIN_SAMPLES_TO_SOLVE:
        raise RuntimeError(f"Not enough accepted samples: {len(accepted)} < {MIN_SAMPLES_TO_SOLVE}")

    # Average each camera's extrinsic across all accepted samples.
    base_cam1_poses = [s.T_base_cam1 for s in accepted]
    base_cam2_poses = [s.T_base_cam2 for s in accepted]
    base_cam3_poses = [s.T_base_cam3 for s in accepted]

    T_base_cam1_avg, tstd1, rstd1 = _average_se3(base_cam1_poses)
    T_base_cam2_avg, tstd2, rstd2 = _average_se3(base_cam2_poses)
    T_base_cam3_avg, tstd3, rstd3 = _average_se3(base_cam3_poses)

    reproj1_mean = float(np.mean([s.reproj1_px for s in accepted]))
    reproj2_mean = float(np.mean([s.reproj2_px for s in accepted]))
    reproj3_mean = float(np.mean([s.reproj3_px for s in accepted]))

    # Relative camera transforms — a sanity check on inter-camera consistency.
    T_cam1_cam2_check = T_base_cam1_avg.inverse() @ T_base_cam2_avg
    T_cam1_cam3_check = T_base_cam1_avg.inverse() @ T_base_cam3_avg

    print("\n=== FINAL RESULTS ===")
    print("\nT_base_cam1 (averaged):")
    print(T_base_cam1_avg)
    print("\nT_base_cam2 (averaged):")
    print(T_base_cam2_avg)
    print("\nT_base_cam3 (averaged):")
    print(T_base_cam3_avg)

    print("\nQuality metrics:")
    print(f"cam1 mean reproj error: {reproj1_mean:.4f} px")
    print(f"cam2 mean reproj error: {reproj2_mean:.4f} px")
    print(f"cam3 mean reproj error: {reproj3_mean:.4f} px")
    print(f"cam1 translation std:   {tstd1:.6f} m")
    print(f"cam2 translation std:   {tstd2:.6f} m")
    print(f"cam3 translation std:   {tstd3:.6f} m")
    print(f"cam1 rotation std:      {rstd1:.6f} deg")
    print(f"cam2 rotation std:      {rstd2:.6f} deg")
    print(f"cam3 rotation std:      {rstd3:.6f} deg")

    print("\nConsistency T_cam1_cam2:")
    print(T_cam1_cam2_check)
    print("R det:", np.linalg.det(T_cam1_cam2_check.R), "valid:", T_cam1_cam2_check.is_valid())

    print("\nConsistency T_cam1_cam3:")
    print(T_cam1_cam3_check)
    print("R det:", np.linalg.det(T_cam1_cam3_check.R), "valid:", T_cam1_cam3_check.is_valid())

    # Refuse to write a result whose sample spread is too loose to trust.
    if (
        tstd1 > MAX_FINAL_TRANSLATION_STD_M
        or tstd2 > MAX_FINAL_TRANSLATION_STD_M
        or tstd3 > MAX_FINAL_TRANSLATION_STD_M
    ):
        raise RuntimeError(
            f"Calibration rejected: translation spread too high "
            f"(cam1={tstd1:.6f}m, cam2={tstd2:.6f}m, cam3={tstd3:.6f}m)"
        )

    if (
        rstd1 > MAX_FINAL_ROTATION_STD_DEG
        or rstd2 > MAX_FINAL_ROTATION_STD_DEG
        or rstd3 > MAX_FINAL_ROTATION_STD_DEG
    ):
        raise RuntimeError(
            f"Calibration rejected: rotation spread too high "
            f"(cam1={rstd1:.6f}deg, cam2={rstd2:.6f}deg, cam3={rstd3:.6f}deg)"
        )

    out = {
        CAM1: T_base_cam1_avg,
        CAM2: T_base_cam2_avg,
        CAM3: T_base_cam3_avg,
    }

    # Back up any existing extrinsics before overwriting.
    out_path = Path(OUT_YAML)
    if out_path.exists():
        backup = out_path.with_suffix(".yaml.bak")
        backup.write_text(out_path.read_text())
        print(f"Backed up existing YAML to: {backup}")

    save_extrinsics_yaml(out_path, out)
    print(f"\nWrote base-referenced camera extrinsics to: {out_path}")
    print(f"Saved debug corner images to: {debug_dir}")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
