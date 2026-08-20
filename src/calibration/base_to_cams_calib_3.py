from __future__ import annotations

import argparse
import json
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
from src.calibration.io_extrinsics import save_extrinsics_yaml, update_extrinsics_yaml_preserving_header
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
# T_robotA_cam (same poses re-expressed in robot_a's frame) used to also be
# written to a separate sibling YAML (config/camera_extrinsics_base_robot_a_frame.yaml)
# for reference, since the live pipeline never reads it back -- it computes
# this same conversion itself at runtime from OUT_YAML + config/robot_bases.yaml
# (see run_pipeline_track_multicam.py / run_pipeline_track_multicam_realsense.py).
# Dropped: that file only duplicated OUT_YAML's data in a different frame, and
# T_robotA_cam is already persisted durably via log_camera_transform() below
# (outputs/calibration_logs/camera_transforms.json), so the "not just printed
# to a terminal that may not be saved" justification for a second YAML no
# longer held.
DEBUG_DIR = "outputs/calibration_debug"
# The realsense-trio tracking pipeline (run_pipeline_track_multicam_
# realsense.py) reads zed2i_1 from this file, not from OUT_YAML -- its
# zed2i_1 entry must be kept identical to OUT_YAML's (same dst frame:
# active robot's lbr_link_0, see io_extrinsics.load_extrinsics_yaml), or the
# pipeline silently tracks against a stale extrinsic. Synced automatically
# below via update_extrinsics_yaml_preserving_header() whenever this script
# recalibrates zed2i_1.
TRACKING_PIPELINE_YAML = "config/camera_extrinsics_realsense.yaml"
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


def _solve_board_pose_candidates(
    img_bgr: np.ndarray, K: np.ndarray
) -> Tuple[list, np.ndarray, list]:
    """Reverted to the original single-solution algorithm: cv2.solvePnP
    with SOLVEPNP_ITERATIVE, one pose per call. (Briefly replaced with
    cv2.solvePnPGeneric + SOLVEPNP_IPPE, which returns every pose consistent
    with the coplanar checkerboard corners for a near-fronto-parallel view,
    disambiguated by _score_camera_pose's camera-height heuristic -- that
    heuristic turned out to reject the genuinely correct, low-reprojection
    solution outright on a real ZED capture (0.2px vs the picked
    candidate's 31.8px), which is worse than SOLVEPNP_ITERATIVE's original
    single deterministic answer. Went back to that rather than trying to
    patch the heuristic further.) Still returns a list (of length 1) so the
    BoardPoseSolve/_save_solve_debug plumbing built around multi-candidate
    solves keeps working unchanged."""
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

    R, _ = cv2.Rodrigues(rvec)
    candidates = [SE3(R, tvec.reshape(3))]
    reproj_errs = [_compute_reproj_err_px(objp, corners, K, rvec, tvec)]

    return candidates, corners.reshape(-1, 2), reproj_errs


# Vestigial now that _solve_board_pose_candidates is back to a single
# SOLVEPNP_ITERATIVE solution (nothing left to disambiguate -- _solve_board_pose
# always has exactly one candidate to pick from). Kept only because
# BoardPoseSolve/_save_solve_debug still record each candidate's score in
# the debug JSON for inspection; not used to choose between candidates
# anymore. See _solve_board_pose_candidates's docstring for why the
# IPPE + height-heuristic disambiguation this used to drive was reverted.
def _score_camera_pose(t_base_cam: np.ndarray) -> float:
    x, y, z = t_base_cam
    return float(z - abs(x) - abs(y))


@dataclass
class _PoseCandidate:
    R: np.ndarray
    t: np.ndarray
    reproj_px: float
    score: float


@dataclass
class BoardPoseSolve:
    """Everything about one board-pose solve worth dumping to disk for
    inspection, not just the winning candidate -- see _save_solve_debug."""
    T_cam_board: SE3
    T_base_cam: SE3
    corners: np.ndarray
    reproj_px: float
    K: np.ndarray
    candidates: list       # list[_PoseCandidate], IPPE's original order
    chosen_index: int


def _solve_board_pose(img_bgr: np.ndarray, K: np.ndarray, T_base_board: SE3) -> BoardPoseSolve:
    """Wraps _solve_board_pose_candidates's single SOLVEPNP_ITERATIVE
    solution into a BoardPoseSolve (T_base_board @ T_cam_board^-1 for the
    implied camera pose, plus the debug/candidate bookkeeping
    _save_solve_debug expects) -- see that function's docstring for why
    there's only ever one candidate here now."""
    candidates, corners, reproj_errs = _solve_board_pose_candidates(img_bgr, K)

    scores = [
        _score_camera_pose((T_base_board @ candidates[i].inverse()).t)
        for i in range(len(candidates))
    ]
    best_i = max(range(len(candidates)), key=lambda i: scores[i])

    if len(candidates) > 1:
        cams = {i: T_base_board @ candidates[i].inverse() for i in range(len(candidates))}
        detail = "  ".join(
            f"cand{i}: t={np.round(cams[i].t, 3)} reproj={reproj_errs[i]:.3f}px score={scores[i]:.3f}"
            + (" <- chosen" if i == best_i else "")
            for i in range(len(candidates))
        )
        print(f"  [ambiguous PnP, {len(candidates)} candidates] {detail}")

    T_cam_board = candidates[best_i]
    T_base_cam = T_base_board @ T_cam_board.inverse()
    pose_candidates = [
        _PoseCandidate(R=candidates[i].R, t=candidates[i].t, reproj_px=reproj_errs[i], score=scores[i])
        for i in range(len(candidates))
    ]
    return BoardPoseSolve(
        T_cam_board=T_cam_board, T_base_cam=T_base_cam, corners=corners,
        reproj_px=reproj_errs[best_i], K=K, candidates=pose_candidates, chosen_index=best_i,
    )


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


# Draws the detected corners (as _draw_chessboard) PLUS the board's own
# coordinate frame (origin + RGB=XYZ axes) as solved for the CHOSEN
# candidate -- lets you see at a glance, from the image alone, which
# physical corner PnP thinks is the origin and which way X/Y/Z point,
# which is exactly what's ambiguous when IPPE returns >1 candidate (see
# _solve_board_pose). Not published live -- written straight to disk
# alongside the matching JSON, see _save_solve_debug.
def _draw_chessboard_with_axes(
    img_bgr: np.ndarray, solve: "BoardPoseSolve", text: str,
) -> np.ndarray:
    vis = _draw_chessboard(img_bgr, solve.corners, text)
    dist = np.zeros((8, 1), dtype=np.float64)
    rvec = _rotation_matrix_to_rotvec(solve.T_cam_board.R).reshape(3, 1)
    tvec = solve.T_cam_board.t.reshape(3, 1)
    axis_len = 4 * SQUARE_SIZE_M  # 12cm -- clearly visible against 3cm squares
    cv2.drawFrameAxes(vis, solve.K, dist, rvec, tvec, axis_len, thickness=3)
    return vis


# Dumps everything about a solve -- not just the winning candidate, every
# candidate IPPE returned with its score -- so a rejected run (like the
# one that motivated this: every sample rejected on reprojection, meaning
# _solve_board_pose's own picture of "chosen" was never trustworthy to
# begin with) can still be inspected after the fact instead of only ever
# seeing images for samples that happened to pass the accept gate.
def _save_solve_debug(
    debug_dir: Path, cam_id: str, attempt_idx: int, img_bgr: np.ndarray,
    solve: "BoardPoseSolve", accepted: bool, reject_reason: str = "",
) -> None:
    cam_dir = debug_dir / "base_to_cams" / cam_id
    cam_dir.mkdir(parents=True, exist_ok=True)
    status = "accepted" if accepted else "rejected"
    stem = f"sample_{attempt_idx:03d}_{status}"

    text = f"{cam_id} #{attempt_idx} {status} reproj={solve.reproj_px:.3f}px"
    if len(solve.candidates) > 1:
        text += f" ({len(solve.candidates)} candidates)"
    vis = _draw_chessboard_with_axes(img_bgr, solve, text)
    cv2.imwrite(str(cam_dir / f"{stem}.png"), vis)

    data = {
        "cam_id": cam_id,
        "attempt_idx": attempt_idx,
        "accepted": accepted,
        "reject_reason": reject_reason,
        "chosen_index": solve.chosen_index,
        "chosen": {
            "T_cam_board": {"R": solve.T_cam_board.R.tolist(), "t": solve.T_cam_board.t.tolist()},
            "T_base_cam": {"R": solve.T_base_cam.R.tolist(), "t": solve.T_base_cam.t.tolist()},
            "reproj_px": solve.reproj_px,
        },
        "candidates": [
            {"R": c.R.tolist(), "t": c.t.tolist(), "reproj_px": c.reproj_px, "score": c.score}
            for c in solve.candidates
        ],
        "corners_px": np.asarray(solve.corners, dtype=float).tolist(),
        "K": np.asarray(solve.K, dtype=float).reshape(3, 3).tolist(),
    }
    (cam_dir / f"{stem}.json").write_text(json.dumps(data, indent=2))


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
    t_last_debug_save = -1e9
    DEBUG_SAVE_MIN_DT_S = 1.0   # cap disk writes to <=1 debug PNG+JSON set/sec, accepted or rejected alike
    last_accept_t = -1e9
    attempt_idx = 0

    print(
        f"Per-attempt debug (corners + axes, every candidate, JSON+PNG, "
        f"max 1/s) -> {debug_dir / 'base_to_cams'}"
    )

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
        # T_base_cam comes back already disambiguated against T_base_board
        # (see _solve_board_pose) -- no separate T_base_board @ T_cam_board^-1
        # step needed here.
        solves: dict[str, BoardPoseSolve] = {}
        rejected = False
        for cam_id in cam_ids:
            img, K, _dt = synced[cam_id]
            try:
                solves[cam_id] = _solve_board_pose(img, K, T_base_board)
            except RuntimeError as e:
                if now - t_last_reject_print > 1.0:
                    print(f"[reject] {cam_id}: {e}")
                    t_last_reject_print = now
                rejected = True
                break
        if rejected:
            continue

        T_cam_board = {c: s.T_cam_board for c, s in solves.items()}
        T_base_cam = {c: s.T_base_cam for c, s in solves.items()}
        reproj_px = {c: s.reproj_px for c, s in solves.items()}

        reproj_bad = any(r > MAX_REPROJ_ERR_PX for r in reproj_px.values())
        is_accepted = not reproj_bad
        reject_reason = "" if is_accepted else f"reproj too high (limit {MAX_REPROJ_ERR_PX:.3f}px)"

        # Dump corners+axes PNG and full-candidate JSON, accepted or
        # rejected alike -- a run that rejects every single sample (like
        # the one that motivated this) previously left NO debug images
        # behind at all, since the old code only ever saved images for
        # samples that already passed the accept gate. Rate-limited to
        # DEBUG_SAVE_MIN_DT_S regardless of accept/reject status (a single
        # shared gate, not one per branch) so a stuck reject-loop -- which
        # can attempt several samples per second -- can't flood the disk.
        if now - t_last_debug_save > DEBUG_SAVE_MIN_DT_S:
            for cam_id in cam_ids:
                img, _K, _dt = synced[cam_id]
                _save_solve_debug(
                    debug_dir, cam_id, attempt_idx, img, solves[cam_id],
                    accepted=is_accepted, reject_reason=reject_reason,
                )
            t_last_debug_save = now
            attempt_idx += 1

        if reproj_bad:
            print(
                f"[reject] reproj too high: "
                f"{ {c: round(r, 3) for c, r in reproj_px.items()} } "
                f"(limit {MAX_REPROJ_ERR_PX:.3f}px)"
            )
            continue

        sample = SampleResult(
            idx=len(accepted), T_cam_board=T_cam_board, T_base_cam=T_base_cam, reproj_px=reproj_px,
        )
        accepted.append(sample)
        last_accept_t = now

        dts_str = " ".join(f"{c}={synced[c][2]:.4f}s" for c in cam_ids)
        reproj_str = " ".join(f"{c}={reproj_px[c]:.3f}px" for c in cam_ids)
        print(f"[accept {len(accepted)}/{NUM_SAMPLES}] rgb-info dt: {dts_str} | reproj: {reproj_str}")

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
    print(
        "(robot_a-frame numbers printed above and in the log_camera_transform() "
        "call below -- no longer written to a separate YAML, see T_robotA_cam "
        "comment near OUT_YAML.)"
    )

    # Keep the realsense-trio pipeline's copy of zed2i_1 in sync -- both
    # files store the same dst frame (active robot's lbr_link_0), so this is
    # a direct copy, no re-projection between robot_a/robot_b/base_link
    # needed here; that resolution happens at pipeline runtime (see
    # run_pipeline_track_multicam_realsense.py's use of get_active_robot_base
    # / get_dual_arm_base_link).
    synced = {c: T for c, T in T_base_cam_avg.items() if c == "zed2i_1"}
    if synced:
        rs_path = Path(TRACKING_PIPELINE_YAML)
        if rs_path.exists():
            backup = rs_path.with_suffix(".yaml.bak")
            backup.write_text(rs_path.read_text())
            print(f"Backed up existing YAML to: {backup}")
        update_extrinsics_yaml_preserving_header(rs_path, synced)
        print(f"Synced {list(synced)} into: {rs_path}")

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
