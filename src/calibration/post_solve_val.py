#!/usr/bin/env python3

import json
import math
import threading
from pathlib import Path
from typing import Optional, List

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.duration import Duration
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException


def make_homogeneous(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)
    return T


def invert_transform(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4, dtype=float)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def quat_xyzw_to_rot(q):
    x, y, z, w = q
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        raise ValueError("Quaternion norm too small.")
    x /= n
    y /= n
    z /= n
    w /= n

    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ], dtype=float)
    return R


def rot_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    m = R
    tr = np.trace(m)

    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (m[2, 1] - m[1, 2]) / S
        y = (m[0, 2] - m[2, 0]) / S
        z = (m[1, 0] - m[0, 1]) / S
    elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
        S = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / S
        x = 0.25 * S
        y = (m[0, 1] + m[1, 0]) / S
        z = (m[0, 2] + m[2, 0]) / S
    elif m[1, 1] > m[2, 2]:
        S = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / S
        x = (m[0, 1] + m[1, 0]) / S
        y = 0.25 * S
        z = (m[1, 2] + m[2, 1]) / S
    else:
        S = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / S
        x = (m[0, 2] + m[2, 0]) / S
        y = (m[1, 2] + m[2, 1]) / S
        z = 0.25 * S

    q = np.array([x, y, z, w], dtype=float)
    q /= np.linalg.norm(q)
    return q


def board_object_points(cols: int, rows: int, square_size_m: float) -> np.ndarray:
    objp = np.zeros((rows * cols, 3), np.float32)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp[:, :2] = grid
    objp *= square_size_m
    return objp


def rotation_angle_deg(R: np.ndarray) -> float:
    trace = np.trace(R)
    c = (trace - 1.0) / 2.0
    c = max(-1.0, min(1.0, float(c)))
    return math.degrees(math.acos(c))


class HandeyePostSolveValidator(Node):
    def __init__(self) -> None:
        super().__init__("validate_wrist_handeye_postsolve")

        self.declare_parameter("image_topic", "/zed_mini/zed_node/rgb/image_rect_color")
        self.declare_parameter("camera_info_topic", "/zed_mini/zed_node/rgb/camera_info")
        self.declare_parameter("base_frame", "base")
        self.declare_parameter("ee_frame", "franka_hand")
        self.declare_parameter("result_json", "./handeye_result.json")
        self.declare_parameter("use_transform", "T_ee_cam")  # or T_cam_ee
        self.declare_parameter("board_cols", 9)
        self.declare_parameter("board_rows", 6)
        self.declare_parameter("square_size_m", 0.025)
        self.declare_parameter("sharpness_threshold", 80.0)
        self.declare_parameter("status_period_s", 1.0)
        self.declare_parameter("max_reproj_rmse_px", 2.0)

        self.image_topic = self.get_parameter("image_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value
        self.base_frame = self.get_parameter("base_frame").value
        self.ee_frame = self.get_parameter("ee_frame").value
        self.result_json = Path(self.get_parameter("result_json").value).expanduser().resolve()
        self.use_transform = self.get_parameter("use_transform").value
        self.board_cols = int(self.get_parameter("board_cols").value)
        self.board_rows = int(self.get_parameter("board_rows").value)
        self.square_size_m = float(self.get_parameter("square_size_m").value)
        self.sharpness_threshold = float(self.get_parameter("sharpness_threshold").value)
        self.status_period_s = float(self.get_parameter("status_period_s").value)
        self.max_reproj_rmse_px = float(self.get_parameter("max_reproj_rmse_px").value)

        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.latest_camera_info: Optional[CameraInfo] = None
        self.latest_valid = None
        self.latest_valid_lock = threading.Lock()

        self.pending_capture = False
        self.pending_capture_lock = threading.Lock()

        self.samples: List[dict] = []

        self.status = {
            "tf_ok": False,
            "camera_info_ok": False,
            "board_visible": False,
            "sharpness": None,
            "sharpness_ok": False,
            "valid": False,
            "last_reason": "waiting",
        }
        self.status_lock = threading.Lock()

        self.T_ee_cam = self.load_handeye_transform()

        self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_cb, 10)
        self.create_subscription(Image, self.image_topic, self.image_cb, 10)

        self.status_timer = self.create_timer(self.status_period_s, self.print_status)

        self.input_thread = threading.Thread(target=self.keyboard_loop, daemon=True)
        self.input_thread.start()

        self.get_logger().info("=== Post-solve wrist hand-eye validator ===")
        self.get_logger().info(f"image_topic      : {self.image_topic}")
        self.get_logger().info(f"camera_info_topic: {self.camera_info_topic}")
        self.get_logger().info(f"base_frame       : {self.base_frame}")
        self.get_logger().info(f"ee_frame         : {self.ee_frame}")
        self.get_logger().info(f"result_json      : {self.result_json}")
        self.get_logger().info(f"use_transform    : {self.use_transform}")
        self.get_logger().info("Fix checkerboard in the world, move robot to many poses, hold still, then press Enter.")
        self.get_logger().info("Type 'p' + Enter to print current summary. Type 'q' + Enter to quit.")

    def load_handeye_transform(self) -> np.ndarray:
        if not self.result_json.exists():
            raise FileNotFoundError(f"Result json not found: {self.result_json}")

        with open(self.result_json, "r") as f:
            data = json.load(f)

        if self.use_transform not in data:
            raise KeyError(f"{self.use_transform} not found in {self.result_json}")

        T = np.array(data[self.use_transform]["matrix_4x4"], dtype=float)

        # We want T_base_board = T_base_ee * T_ee_cam * T_cam_board
        # So internally we need T_ee_cam.
        if self.use_transform == "T_ee_cam":
            return T
        elif self.use_transform == "T_cam_ee":
            return invert_transform(T)
        else:
            raise ValueError("use_transform must be T_ee_cam or T_cam_ee")

    def camera_info_cb(self, msg: CameraInfo) -> None:
        self.latest_camera_info = msg
        with self.status_lock:
            self.status["camera_info_ok"] = True

    def keyboard_loop(self) -> None:
        while rclpy.ok():
            try:
                s = input()
            except EOFError:
                break

            cmd = s.strip().lower()
            if cmd == "q":
                self.get_logger().info("Quit requested.")
                rclpy.shutdown()
                return
            elif cmd == "p":
                self.print_summary()
            else:
                with self.pending_capture_lock:
                    self.pending_capture = True

    def print_status(self) -> None:
        with self.status_lock:
            st = dict(self.status)

        sharpness_str = "n/a" if st["sharpness"] is None else f"{st['sharpness']:.1f}"
        self.get_logger().info(
            "[STATUS] "
            f"TF={'OK' if st['tf_ok'] else 'MISS'} | "
            f"CamInfo={'OK' if st['camera_info_ok'] else 'MISS'} | "
            f"Board={'YES' if st['board_visible'] else 'NO'} | "
            f"Sharp={sharpness_str} ({'OK' if st['sharpness_ok'] else 'LOW'}) | "
            f"Valid={'YES' if st['valid'] else 'NO'} | "
            f"N={len(self.samples)} | "
            f"Reason={st['last_reason']}"
        )

    def get_base_to_ee(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.ee_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.2),
            )
            tr = tf.transform.translation
            ro = tf.transform.rotation
            t = np.array([tr.x, tr.y, tr.z], dtype=float)
            q = np.array([ro.x, ro.y, ro.z, ro.w], dtype=float)
            R = quat_xyzw_to_rot(q)
            return make_homogeneous(R, t), tf
        except (LookupException, ConnectivityException, ExtrapolationException):
            return None, None

    def sharpness_score(self, gray: np.ndarray) -> float:
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def solve_board_pose(self, gray: np.ndarray, sample_camera_info: CameraInfo):
        pattern_size = (self.board_cols, self.board_rows)
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)

        if not found:
            return None

        corners = cv2.cornerSubPix(
            gray,
            corners,
            winSize=(11, 11),
            zeroZone=(-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
        )

        expected = self.board_cols * self.board_rows
        if len(corners) != expected:
            return None

        obj_pts = board_object_points(self.board_cols, self.board_rows, self.square_size_m).astype(np.float32)
        K = np.array(sample_camera_info.k, dtype=float).reshape(3, 3)
        D = np.array(sample_camera_info.d, dtype=float).reshape(-1, 1)

        ok, rvec, tvec = cv2.solvePnP(
            objectPoints=obj_pts,
            imagePoints=corners,
            cameraMatrix=K,
            distCoeffs=D,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return None

        R_cam_board, _ = cv2.Rodrigues(rvec)
        t_cam_board = tvec.reshape(3)

        proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, D)
        proj = proj.reshape(-1, 2)
        err = np.linalg.norm(proj - corners.reshape(-1, 2), axis=1)
        rmse = float(np.sqrt(np.mean(err ** 2)))

        return {
            "corners": corners,
            "R_cam_board": R_cam_board,
            "t_cam_board": t_cam_board,
            "rmse": rmse,
        }

    def image_cb(self, msg: Image) -> None:
        if self.latest_camera_info is None:
            with self.status_lock:
                self.status["camera_info_ok"] = False
                self.status["tf_ok"] = False
                self.status["board_visible"] = False
                self.status["sharpness"] = None
                self.status["sharpness_ok"] = False
                self.status["valid"] = False
                self.status["last_reason"] = "waiting for CameraInfo"
            self.handle_trigger_no_valid()
            return

        T_base_ee, tf_msg = self.get_base_to_ee()
        if T_base_ee is None:
            with self.status_lock:
                self.status["tf_ok"] = False
                self.status["camera_info_ok"] = True
                self.status["board_visible"] = False
                self.status["sharpness"] = None
                self.status["sharpness_ok"] = False
                self.status["valid"] = False
                self.status["last_reason"] = f"missing TF {self.base_frame}<-{self.ee_frame}"
            self.handle_trigger_no_valid()
            return

        with self.status_lock:
            self.status["tf_ok"] = True
            self.status["camera_info_ok"] = True

        try:
            image_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            with self.status_lock:
                self.status["board_visible"] = False
                self.status["sharpness"] = None
                self.status["sharpness_ok"] = False
                self.status["valid"] = False
                self.status["last_reason"] = f"cv_bridge failed: {e}"
            self.handle_trigger_no_valid()
            return

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        sharpness = self.sharpness_score(gray)
        sharpness_ok = sharpness >= self.sharpness_threshold

        with self.status_lock:
            self.status["sharpness"] = sharpness
            self.status["sharpness_ok"] = sharpness_ok

        if not sharpness_ok:
            with self.latest_valid_lock:
                self.latest_valid = None
            with self.status_lock:
                self.status["board_visible"] = False
                self.status["valid"] = False
                self.status["last_reason"] = "image too blurry"
            self.handle_trigger_no_valid()
            return

        board_pose = self.solve_board_pose(gray, self.latest_camera_info)
        if board_pose is None:
            with self.latest_valid_lock:
                self.latest_valid = None
            with self.status_lock:
                self.status["board_visible"] = False
                self.status["valid"] = False
                self.status["last_reason"] = "checkerboard not found / incomplete / solvePnP failed"
            self.handle_trigger_no_valid()
            return

        with self.status_lock:
            self.status["board_visible"] = True

        if board_pose["rmse"] > self.max_reproj_rmse_px:
            with self.latest_valid_lock:
                self.latest_valid = None
            with self.status_lock:
                self.status["valid"] = False
                self.status["last_reason"] = f"reprojection RMSE too high: {board_pose['rmse']:.3f}px"
            self.handle_trigger_no_valid()
            return

        T_cam_board = make_homogeneous(board_pose["R_cam_board"], board_pose["t_cam_board"])
        T_base_board = T_base_ee @ self.T_ee_cam @ T_cam_board

        valid_data = {
            "msg": msg,
            "tf_msg": tf_msg,
            "T_base_ee": T_base_ee,
            "T_cam_board": T_cam_board,
            "T_base_board": T_base_board,
            "rmse": board_pose["rmse"],
        }

        with self.latest_valid_lock:
            self.latest_valid = valid_data

        with self.status_lock:
            self.status["valid"] = True
            self.status["last_reason"] = "ready to capture"

        should_capture = False
        with self.pending_capture_lock:
            if self.pending_capture:
                should_capture = True
                self.pending_capture = False

        if not should_capture:
            return

        self.capture_current_valid()

    def handle_trigger_no_valid(self) -> None:
        should_capture = False
        with self.pending_capture_lock:
            if self.pending_capture:
                should_capture = True
                self.pending_capture = False

        if should_capture:
            with self.status_lock:
                reason = self.status["last_reason"]
            self.get_logger().warn(f"No valid validation sample right now: {reason}")

    def capture_current_valid(self) -> None:
        with self.latest_valid_lock:
            valid = self.latest_valid

        if valid is None:
            self.get_logger().warn("No valid sample available right now.")
            return

        T = valid["T_base_board"]
        t = T[:3, 3].copy()
        q = rot_to_quat_xyzw(T[:3, :3])

        sample = {
            "translation_xyz_m": t.tolist(),
            "quaternion_xyzw": q.tolist(),
            "reproj_rmse_px": float(valid["rmse"]),
            "T_base_board": T.tolist(),
        }
        self.samples.append(sample)

        self.get_logger().info(
            f"[CAPTURED] N={len(self.samples)} | board xyz = "
            f"[{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}] m | reproj_rmse={valid['rmse']:.3f}px"
        )

        self.print_summary()

    def print_summary(self) -> None:
        n = len(self.samples)
        if n == 0:
            self.get_logger().info("No captured validation samples yet.")
            return

        Ts = [np.array(s["T_base_board"], dtype=float) for s in self.samples]
        translations = np.array([T[:3, 3] for T in Ts], dtype=float)

        mean_t = translations.mean(axis=0)
        std_t = translations.std(axis=0)
        rms_t = float(np.sqrt(np.mean(np.sum((translations - mean_t) ** 2, axis=1))))
        max_dev_t = float(np.max(np.linalg.norm(translations - mean_t, axis=1)))

        R_ref = Ts[0][:3, :3]
        ang_devs = []
        for T in Ts:
            R_rel = R_ref.T @ T[:3, :3]
            ang_devs.append(rotation_angle_deg(R_rel))
        ang_devs = np.array(ang_devs, dtype=float)

        rms_reproj = float(np.sqrt(np.mean([s["reproj_rmse_px"] ** 2 for s in self.samples])))

        self.get_logger().info("=== Validation summary ===")
        self.get_logger().info(f"N samples: {n}")
        self.get_logger().info(
            f"Mean board position in {self.base_frame}: "
            f"[{mean_t[0]:.5f}, {mean_t[1]:.5f}, {mean_t[2]:.5f}] m"
        )
        self.get_logger().info(
            f"Translation std [m]: "
            f"[{std_t[0]:.5f}, {std_t[1]:.5f}, {std_t[2]:.5f}]"
        )
        self.get_logger().info(f"Translation RMS spread [m]: {rms_t:.5f}")
        self.get_logger().info(f"Translation max deviation from mean [m]: {max_dev_t:.5f}")
        self.get_logger().info(
            f"Orientation deviation wrt first sample [deg]: "
            f"mean={ang_devs.mean():.3f}, max={ang_devs.max():.3f}"
        )
        self.get_logger().info(f"RMS checkerboard reprojection RMSE [px]: {rms_reproj:.4f}")

        if n >= 5:
            self.get_logger().info(
                "Interpretation: lower spread is better. "
                "If board position/orientation in base stays tight across many poses, hand-eye is likely good."
            )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = HandeyePostSolveValidator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()