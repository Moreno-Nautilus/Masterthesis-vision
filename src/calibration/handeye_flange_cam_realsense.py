"""Helpers for hand-eye calibration of a wrist-mounted RealSense camera.

Solves the classic AX=XB hand-eye problem to find T_flange_cam: the static
rigid offset between the robot flange (lbr_link_ee) and a RealSense camera's
optical frame, bolted to the end-effector so it moves with the arm.

Unlike src/calibration/base_to_cams_calib_3.py (which fixes the checkerboard
AND every camera, so a single PnP per camera solves everything), here only
the checkerboard is fixed -- the camera moves with the robot -- so a single
image can't disambiguate the mount offset from the board's own unknown pose.
Instead we collect the checkerboard-in-camera pose (via PnP) and the
flange-in-base pose at several distinct robot poses, then solve for the one
offset consistent with all of them via cv2.calibrateHandEye.

This module is now a **helpers-only library**, not a runnable script -- the
capture and solve responsibilities that used to live in its own main() moved
out into two dedicated entry points:
  - src/calibration/capture_handeye_data.py  -- Stage A, capture flange
    poses + checkerboard images/detections (HandEyeSample below), any
    controller, any arm(s).
  - src/calibration/calibrate_handeye.py     -- Stage B, solve T_flange_cam
    from captured HandEyeSamples (--method direct uses _solve_handeye below;
    --method joint uses the bundle-adjustment submodule instead).
Everything here (the checkerboard/PnP math, the HandEyeSample schema + its
JSON (de)serialization, and the closed-form _solve_handeye) is shared by
both, plus board_pose_from_flange_realsense.py.

See docs/calibration_cheatsheet.md for the end-to-end walkthrough.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header

from src.utils.se3 import SE3

# ---------------- USER SETTINGS ----------------
CHESS_COLS = 8   # inner corners along x (matches base_to_cams_calib_3.py / the ZED board)
CHESS_ROWS = 11  # inner corners along y
SQUARE_SIZE_M = 0.03

RGB_INFO_MAX_DT_S = 0.50       # RGB vs CameraInfo freshness
FLANGE_POSE_MAX_AGE_S = 0.25   # flange-pose-topic freshness at capture time
MAX_REPROJ_ERR_PX = 2.5        # reject a sample if PnP reprojection error is too high

OUT_YAML = "config/camera_extrinsics_realsense.yaml"
DEBUG_DIR = "outputs/calibration_debug/handeye"

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
