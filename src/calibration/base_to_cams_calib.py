# Calibrate 2 cams to robot base
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

CAM1_RGB = "/zed2i_1/zed_node/rgb/color/rect/image"
CAM2_RGB = "/zed2i_2/zed_node/rgb/color/rect/image"
CAM1_INFO = "/zed2i_1/zed_node/rgb/color/rect/camera_info"
CAM2_INFO = "/zed2i_2/zed_node/rgb/color/rect/camera_info"

CHESS_COLS = 8   # number of INNER corners along x
CHESS_ROWS = 11  # number of INNER corners along y
SQUARE_SIZE_M = 0.03

SYNC_SLOP_S = 0.05

BASE_BOARD_YAML = "config/base_board_pose.yaml"
OUT_YAML = "config/camera_extrinsics_base.yaml"
DEBUG_DIR = "outputs/calibration_debug"
# ------------------------------------------------


def _stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def _K_from_camerainfo(msg: CameraInfo) -> np.ndarray:
    return np.array(msg.k, dtype=float).reshape(3, 3)


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
        img = img[:, :, ::-1].copy()

    return img


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


def _load_T_base_board(path: Path) -> SE3:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    bb = cfg["base_board"]

    t = np.array(bb["translation_xyz_m"], dtype=float)
    rpy = bb["rotation_rpy_deg"]
    R = _rpy_deg_to_R(float(rpy[0]), float(rpy[1]), float(rpy[2]))

    return SE3(R, t)


def _solve_board_pose(img_bgr: np.ndarray, K: np.ndarray) -> Tuple[SE3, np.ndarray]:
    pattern_size = (CHESS_COLS, CHESS_ROWS)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if not ok or corners is None:
        raise RuntimeError("Chessboard NOT found")

    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), term)

    objp = np.zeros((CHESS_ROWS * CHESS_COLS, 3), dtype=np.float32)
    grid = np.mgrid[0:CHESS_COLS, 0:CHESS_ROWS].T.reshape(-1, 2).astype(np.float32)
    objp[:, :2] = grid * float(SQUARE_SIZE_M)

    dist = np.zeros((8, 1), dtype=np.float64)

    ok, rvec, tvec = cv2.solvePnP(objp, corners, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise RuntimeError("solvePnP failed")

    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3)

    return SE3(R, t), corners.reshape(-1, 2)


def _draw_chessboard(img_bgr: np.ndarray, corners: np.ndarray) -> np.ndarray:
    vis = img_bgr.copy()
    pattern_size = (CHESS_COLS, CHESS_ROWS)
    cv2.drawChessboardCorners(vis, pattern_size, corners.reshape(-1, 1, 2), True)
    return vis


@dataclass
class CamState:
    img_msg: Optional[Image] = None
    img_t: Optional[float] = None
    K: Optional[np.ndarray] = None


class CheckerboardBaseCalib(Node):
    def __init__(self):
        super().__init__("checkerboard_base_to_cameras_calib")

        self.cam1 = CamState()
        self.cam2 = CamState()

        self.create_subscription(Image, CAM1_RGB, self._on_img1, 10)
        self.create_subscription(Image, CAM2_RGB, self._on_img2, 10)
        self.create_subscription(CameraInfo, CAM1_INFO, self._on_info1, 10)
        self.create_subscription(CameraInfo, CAM2_INFO, self._on_info2, 10)

    def _on_img1(self, msg: Image) -> None:
        self.cam1.img_msg = msg
        self.cam1.img_t = _stamp_to_sec(msg.header.stamp)

    def _on_img2(self, msg: Image) -> None:
        self.cam2.img_msg = msg
        self.cam2.img_t = _stamp_to_sec(msg.header.stamp)

    def _on_info1(self, msg: CameraInfo) -> None:
        self.cam1.K = _K_from_camerainfo(msg)

    def _on_info2(self, msg: CameraInfo) -> None:
        self.cam2.K = _K_from_camerainfo(msg)

    def ready(self) -> bool:
        return (
            self.cam1.img_msg is not None and self.cam2.img_msg is not None
            and self.cam1.K is not None and self.cam2.K is not None
            and self.cam1.img_t is not None and self.cam2.img_t is not None
        )

    def get_synced_pair(self) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        if not self.ready():
            return None

        t_ref = max(self.cam1.img_t, self.cam2.img_t)
        max_dt = max(abs(self.cam1.img_t - t_ref), abs(self.cam2.img_t - t_ref))
        if max_dt > SYNC_SLOP_S:
            return None

        img1 = _img_to_numpy_color(self.cam1.img_msg)
        img2 = _img_to_numpy_color(self.cam2.img_msg)
        return img1, img2, self.cam1.K, self.cam2.K


def main() -> None:
    rclpy.init()
    node = CheckerboardBaseCalib()

    debug_dir = Path(DEBUG_DIR)
    debug_dir.mkdir(parents=True, exist_ok=True)

    T_base_board = _load_T_base_board(Path(BASE_BOARD_YAML))
    print("Loaded T_base_board:")
    print(T_base_board)

    print("Show the checkerboard to BOTH cameras, keep it still for ~1s...")
    pair = None
    t0 = time.time()
    while pair is None:
        rclpy.spin_once(node, timeout_sec=0.1)
        pair = node.get_synced_pair()
        if time.time() - t0 > 10.0 and pair is None:
            print("Still waiting for a synced RGB pair... keep checkerboard visible.")
            t0 = time.time()

    img1, img2, K1, K2 = pair

    print("Finding checkerboard in cam1...")
    T_cam1_board, corners1 = _solve_board_pose(img1, K1)

    print("Finding checkerboard in cam2...")
    T_cam2_board, corners2 = _solve_board_pose(img2, K2)

    # Main result: base -> cam
    T_base_cam1 = T_base_board @ T_cam1_board.inverse()
    T_base_cam2 = T_base_board @ T_cam2_board.inverse()

    # Consistency check: cam1 -> cam2 from base-referenced result
    T_cam1_cam2_check = T_base_cam1.inverse() @ T_base_cam2

    print("\n=== RESULTS ===")
    print("T_cam1_board:", T_cam1_board)
    print("T_cam2_board:", T_cam2_board)
    print("\nT_base_cam1:", T_base_cam1)
    print("T_base_cam2:", T_base_cam2)
    print("\nConsistency T_cam1_cam2:", T_cam1_cam2_check)
    print("R det:", np.linalg.det(T_cam1_cam2_check.R), "valid:", T_cam1_cam2_check.is_valid())

    out = {
        CAM1: T_base_cam1,
        CAM2: T_base_cam2,
    }

    out_path = Path(OUT_YAML)
    if out_path.exists():
        backup = out_path.with_suffix(".yaml.bak")
        backup.write_text(out_path.read_text())
        print(f"Backed up existing YAML to: {backup}")

    save_extrinsics_yaml(out_path, out)
    print(f"\nWrote base-referenced camera extrinsics to: {out_path}")

    vis1 = _draw_chessboard(img1, corners1)
    vis2 = _draw_chessboard(img2, corners2)
    cv2.imwrite(str(debug_dir / f"{CAM1}_corners.png"), vis1)
    cv2.imwrite(str(debug_dir / f"{CAM2}_corners.png"), vis2)
    print(f"Saved debug corner images to: {debug_dir}")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()