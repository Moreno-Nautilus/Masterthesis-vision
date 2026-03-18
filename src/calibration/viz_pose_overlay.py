from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
import yaml

from src.utils.se3 import SE3


# ---------------- USER SETTINGS ----------------
CAM1 = "zed2i_1"
CAM2 = "zed2i_2"

CAM1_RGB = "/zed2i_1/zed_node/rgb/color/rect/image"
CAM2_RGB = "/zed2i_2/zed_node/rgb/color/rect/image"
CAM1_INFO = "/zed2i_1/zed_node/rgb/color/rect/camera_info"
CAM2_INFO = "/zed2i_2/zed_node/rgb/color/rect/camera_info"

SYNC_SLOP_S = 0.05
EXTRINSICS_YAML = "config/camera_extrinsics_base.yaml"

AXIS_LEN_M = 0.12
SAVE_DIR = Path("outputs/base_axes_overlay")
SHOW_WINDOWS = True

# Extra sanity-check points in BASE frame [m]
TEST_POINTS_BASE = np.array([
    [0.5, 0.0, 0.0],
    [0.5, 0.2, 0.0],
    [0.5, -0.2, 0.0],
], dtype=float)
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


def _parse_se3_from_entry(entry: dict) -> SE3:
    if "translation_xyz_m" in entry and "rotation_matrix" in entry:
        t = np.array(entry["translation_xyz_m"], dtype=float)
        R = np.array(entry["rotation_matrix"], dtype=float).reshape(3, 3)
        return SE3(R, t)

    if "translation" in entry and "rotation_matrix" in entry:
        t = np.array(entry["translation"], dtype=float)
        R = np.array(entry["rotation_matrix"], dtype=float).reshape(3, 3)
        return SE3(R, t)

    if "t" in entry and "R" in entry:
        t = np.array(entry["t"], dtype=float).reshape(3)
        R = np.array(entry["R"], dtype=float).reshape(3, 3)
        return SE3(R, t)

    if "matrix" in entry:
        T = np.array(entry["matrix"], dtype=float).reshape(4, 4)
        return SE3(T[:3, :3], T[:3, 3])

    raise ValueError(f"Unsupported extrinsics entry format: {entry.keys()}")


def _load_extrinsics(path: Path) -> Dict[str, SE3]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    out: Dict[str, SE3] = {}
    for cam_name, entry in cfg.items():
        out[cam_name] = _parse_se3_from_entry(entry)
    return out


def _project_points(pts_3d_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    zs = pts_3d_cam[:, 2]
    uv = np.zeros((pts_3d_cam.shape[0], 2), dtype=float)
    uv[:, 0] = K[0, 0] * (pts_3d_cam[:, 0] / zs) + K[0, 2]
    uv[:, 1] = K[1, 1] * (pts_3d_cam[:, 1] / zs) + K[1, 2]
    return uv


def _draw_labeled_point(
    vis: np.ndarray,
    uv: Tuple[int, int],
    label: str,
    color_bgr: Tuple[int, int, int],
) -> None:
    cv2.circle(vis, uv, 8, color_bgr, -1)
    cv2.putText(
        vis,
        label,
        (uv[0] + 10, uv[1]),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color_bgr,
        2,
        cv2.LINE_AA,
    )


def _overlay_base_axes(
    img_bgr: np.ndarray,
    K: np.ndarray,
    T_base_cam: SE3,
    cam_name: str,
    axis_len_m: float,
) -> np.ndarray:
    vis = img_bgr.copy()

    # Need base points expressed in camera frame
    T_cam_base = T_base_cam.inverse()

    # Base origin and axis tips, expressed in BASE frame
    axis_points_base = np.array([
        [0.0, 0.0, 0.0],               # origin
        [axis_len_m, 0.0, 0.0],        # +x
        [0.0, axis_len_m, 0.0],        # +y
        [0.0, 0.0, axis_len_m],        # +z
    ], dtype=float)

    axis_points_cam = (T_cam_base.R @ axis_points_base.T).T + T_cam_base.t.reshape(1, 3)

    # Check if base origin/axes are in front of camera
    if np.any(axis_points_cam[:, 2] <= 1e-6):
        cv2.putText(
            vis,
            f"{cam_name}: base origin/axes behind camera",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return vis

    uv_axes = _project_points(axis_points_cam, K)
    uv_axes = np.round(uv_axes).astype(int)

    o = tuple(uv_axes[0])
    px = tuple(uv_axes[1])
    py = tuple(uv_axes[2])
    pz = tuple(uv_axes[3])

    # Draw base origin
    cv2.circle(vis, o, 8, (0, 255, 255), -1)
    cv2.putText(
        vis,
        "base (0,0,0)",
        (o[0] + 10, o[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Draw axes
    cv2.arrowedLine(vis, o, px, (0, 0, 255), 3, tipLength=0.12)   # +X red
    cv2.arrowedLine(vis, o, py, (0, 255, 0), 3, tipLength=0.12)   # +Y green
    cv2.arrowedLine(vis, o, pz, (255, 0, 0), 3, tipLength=0.12)   # +Z blue

    cv2.putText(vis, "+X", (px[0] + 8, px[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, "+Y", (py[0] + 8, py[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(vis, "+Z", (pz[0] + 8, pz[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

    # Extra sanity-check table points in BASE frame
    test_points_cam = (T_cam_base.R @ TEST_POINTS_BASE.T).T + T_cam_base.t.reshape(1, 3)

    for pt_base, pt_cam in zip(TEST_POINTS_BASE, test_points_cam):
        if pt_cam[2] <= 1e-6:
            continue
        uv = _project_points(pt_cam.reshape(1, 3), K)
        uv = np.round(uv).astype(int)[0]
        uv_t = (int(uv[0]), int(uv[1]))
        label = f"({pt_base[0]:.1f},{pt_base[1]:.1f},{pt_base[2]:.1f})"
        _draw_labeled_point(vis, uv_t, label, (255, 0, 255))  # purple

    # Text with camera position in base
    t = T_base_cam.t
    txt = f"{cam_name}: cam in base = [{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}] m"
    cv2.putText(
        vis,
        txt,
        (30, vis.shape[0] - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return vis


@dataclass
class CamState:
    img_msg: Optional[Image] = None
    img_t: Optional[float] = None
    K: Optional[np.ndarray] = None


class BaseAxesOverlay(Node):
    def __init__(self):
        super().__init__("base_axes_overlay")

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
    node = BaseAxesOverlay()

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    extrinsics = _load_extrinsics(Path(EXTRINSICS_YAML))

    if CAM1 not in extrinsics or CAM2 not in extrinsics:
        raise RuntimeError(f"Expected both {CAM1} and {CAM2} in {EXTRINSICS_YAML}")

    print("Loaded base-referenced camera extrinsics.")
    print(f"{CAM1}: {extrinsics[CAM1]}")
    print(f"{CAM2}: {extrinsics[CAM2]}")

    print("Waiting for synced camera frames...")
    pair = None
    t0 = time.time()
    while pair is None:
        rclpy.spin_once(node, timeout_sec=0.1)
        pair = node.get_synced_pair()
        if time.time() - t0 > 10.0 and pair is None:
            print("Still waiting for synced images...")
            t0 = time.time()

    img1, img2, K1, K2 = pair

    vis1 = _overlay_base_axes(img1, K1, extrinsics[CAM1], CAM1, AXIS_LEN_M)
    vis2 = _overlay_base_axes(img2, K2, extrinsics[CAM2], CAM2, AXIS_LEN_M)

    out1 = SAVE_DIR / f"{CAM1}_base_axes_overlay.png"
    out2 = SAVE_DIR / f"{CAM2}_base_axes_overlay.png"
    cv2.imwrite(str(out1), vis1)
    cv2.imwrite(str(out2), vis2)

    print(f"Saved: {out1}")
    print(f"Saved: {out2}")

    if SHOW_WINDOWS:
        cv2.imshow(f"{CAM1} base axes overlay", vis1)
        cv2.imshow(f"{CAM2} base axes overlay", vis2)
        print("Press any key in an image window to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()