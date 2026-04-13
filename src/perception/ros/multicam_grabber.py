from dataclasses import dataclass
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo

from src.utils.se3 import SE3
from src.perception.view import View

import time

@dataclass
class CameraTopics:
    cam_id: str
    depth_topic: str
    info_topic: str
    rgb_topic: str
    rgb_info_topic: str


def _stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def _K_from_camerainfo(msg: CameraInfo) -> np.ndarray:
    return np.array(msg.k, dtype=float).reshape(3, 3)


def _depth_to_meters(depth: np.ndarray, encoding: str) -> np.ndarray:
    if encoding == "32FC1":
        return depth.astype(np.float32)
    if encoding == "16UC1":
        return depth.astype(np.float32) * 1e-3
    raise ValueError(f"Unsupported depth encoding: {encoding}")


def _img_to_numpy_depth(msg: Image) -> np.ndarray:
    h, w = int(msg.height), int(msg.width)
    enc = msg.encoding

    if enc == "32FC1":
        dtype = np.float32
        elem_size = 4
    elif enc == "16UC1":
        dtype = np.uint16
        elem_size = 2
    else:
        raise ValueError(f"Unsupported depth Image encoding: {enc}")

    data = np.frombuffer(msg.data, dtype=dtype)
    step_elems = int(msg.step // elem_size)

    if step_elems == w:
        return data.reshape(h, w)

    out = np.zeros((h, w), dtype=dtype)
    for r in range(h):
        out[r, :] = data[r * step_elems : r * step_elems + w]
    return out


def _img_to_numpy_rgb(msg: Image) -> np.ndarray:
    """
    Returns uint8 HxWx3 (RGB).
    Supports: rgb8, bgr8, rgba8, bgra8.
    """
    h, w = int(msg.height), int(msg.width)
    enc = msg.encoding.lower()

    if enc in ("rgb8", "bgr8"):
        channels = 3
    elif enc in ("rgba8", "bgra8"):
        channels = 4
    else:
        raise ValueError(f"Unsupported RGB Image encoding: {msg.encoding}")

    data = np.frombuffer(msg.data, dtype=np.uint8)
    step = int(msg.step)

    row_bytes = w * channels
    if step == row_bytes:
        img = data.reshape(h, w, channels)
    else:
        # padded rows
        img = np.zeros((h, w, channels), dtype=np.uint8)
        for r in range(h):
            start = r * step
            img[r, :, :] = data[start : start + row_bytes].reshape(w, channels)

    if channels == 4:
        img = img[:, :, :3]

    if enc.startswith("bgr"):
        img = img[:, :, ::-1]  # BGR->RGB
    return img


class MultiCamGrabber(Node):
    def __init__(
        self,
        cameras: list[CameraTopics],
        sync_slop_s: float = 0.015,
        use_best_effort_if_unsynced: bool = True,
        static_extrinsics_base_cam: Optional[dict[str, SE3]] = None,
        rgb_depth_max_dt_s: float = 0.10,
    ):
        super().__init__("multicam_grabber")

        self.cameras = cameras
        self.sync_slop_s = float(sync_slop_s)
        self.use_best_effort_if_unsynced = bool(use_best_effort_if_unsynced)
        self.rgb_depth_max_dt_s = float(rgb_depth_max_dt_s)

        self.T_base_cam_map: dict[str, SE3] = static_extrinsics_base_cam or {}

        # Buffers
        self._depth_msg: dict[str, Image] = {}
        self._depth_stamp_s: dict[str, float] = {}
        self._K_depth: dict[str, np.ndarray] = {}

        self._rgb_msg: dict[str, Image] = {}
        self._rgb_stamp_s: dict[str, float] = {}
        self._K_rgb: dict[str, np.ndarray] = {}

        self._subs = []
        for c in cameras:
            self.get_logger().info(
                f"Subscribing cam_id={c.cam_id} depth={c.depth_topic} depth_info={c.info_topic} "
                f"rgb={c.rgb_topic} rgb_info={c.rgb_info_topic}"
            )

            self._subs.append(
                self.create_subscription(
                    Image,
                    c.depth_topic,
                    lambda msg, cam_id=c.cam_id: self._on_depth(cam_id, msg),
                    10,
                )
            )
            self._subs.append(
                self.create_subscription(
                    CameraInfo,
                    c.info_topic,
                    lambda msg, cam_id=c.cam_id: self._on_depth_info(cam_id, msg),
                    10,
                )
            )

            self._subs.append(
                self.create_subscription(
                    Image,
                    c.rgb_topic,
                    lambda msg, cam_id=c.cam_id: self._on_rgb(cam_id, msg),
                    10,
                )
            )
            self._subs.append(
                self.create_subscription(
                    CameraInfo,
                    c.rgb_info_topic,
                    lambda msg, cam_id=c.cam_id: self._on_rgb_info(cam_id, msg),
                    10,
                )
            )

    def _on_depth(self, cam_id: str, msg: Image) -> None:
        self._depth_msg[cam_id] = msg
        self._depth_stamp_s[cam_id] = _stamp_to_sec(msg.header.stamp)

    def _on_depth_info(self, cam_id: str, msg: CameraInfo) -> None:
        self._K_depth[cam_id] = _K_from_camerainfo(msg)

    def _on_rgb(self, cam_id: str, msg: Image) -> None:
        self._rgb_msg[cam_id] = msg
        self._rgb_stamp_s[cam_id] = _stamp_to_sec(msg.header.stamp)

    def _on_rgb_info(self, cam_id: str, msg: CameraInfo) -> None:
        self._K_rgb[cam_id] = _K_from_camerainfo(msg)

    def ready(self) -> bool:
        for c in self.cameras:
            has_depth_msg = c.cam_id in self._depth_msg
            has_K_depth = c.cam_id in self._K_depth
            has_rgb_msg = c.cam_id in self._rgb_msg
            has_K_rgb = c.cam_id in self._K_rgb

            # print(
            #     f"[READY DEBUG] {c.cam_id} "
            #     f"depth_msg={has_depth_msg} "
            #     f"K_depth={has_K_depth} "
            #     f"rgb_msg={has_rgb_msg} "
            #     f"K_rgb={has_K_rgb}"
            # )

            if c.cam_id not in self._depth_msg:
                return False
            if c.cam_id not in self._K_depth:
                return False
            if c.cam_id not in self._rgb_msg:
                return False
            if c.cam_id not in self._K_rgb:
                return False
        return True

    def get_latest_views(self) -> Optional[list[View]]:
        """
        Returns a synced list[View] in base frame if possible.

        """
        if not self.ready():
            return None

        # ------------------------
        # 1) Multi-cam depth sync
        # ------------------------
        stamps_d = [self._depth_stamp_s[c.cam_id] for c in self.cameras]
        t_ref = max(stamps_d)
        max_dt = max(abs(t - t_ref) for t in stamps_d)

        if max_dt > self.sync_slop_s and not self.use_best_effort_if_unsynced:
            # throttle log spam
            now = time.time()
            last = getattr(self, "_last_warn_unsynced_depth_t", 0.0)
            if (now - last) > 1.0:
                self._last_warn_unsynced_depth_t = now
                self.get_logger().warn(
                    f"unsynced depth set: max_dt={max_dt:.3f}s > slop={self.sync_slop_s:.3f}s"
                )
            return None

        views: list[View] = []

        # init health-monitor state lazily (no need to touch __init__)
        if not hasattr(self, "_rgb_frozen_state"):
            # cam_id -> dict
            self._rgb_frozen_state = {}
        if not hasattr(self, "_last_warn_rgb_depth_dt_t"):
            self._last_warn_rgb_depth_dt_t = 0.0
        if not hasattr(self, "_last_warn_shape_t"):
            self._last_warn_shape_t = 0.0
        if not hasattr(self, "_last_warn_frozen_t"):
            self._last_warn_frozen_t = 0.0

        now_wall = time.time()

        for c in self.cameras:
            cam_id = c.cam_id

            depth_msg = self._depth_msg[cam_id]
            rgb_msg = self._rgb_msg[cam_id]

            t_d = self._depth_stamp_s[cam_id]
            t_rgb = self._rgb_stamp_s[cam_id]

            # --------------------------------
            # 2) Option 3: rgb-depth time check
            # --------------------------------
            dt_rgb_depth = abs(t_rgb - t_d)
            if dt_rgb_depth > self.rgb_depth_max_dt_s and not self.use_best_effort_if_unsynced:
                if (now_wall - self._last_warn_rgb_depth_dt_t) > 1.0:
                    self._last_warn_rgb_depth_dt_t = now_wall
                    self.get_logger().warn(
                        f"{cam_id} rgb-depth unsynced: dt={dt_rgb_depth:.3f}s > "
                        f"{self.rgb_depth_max_dt_s:.3f}s"
                    )
                return None

            # Convert depth (meters)
            K = self._K_depth[cam_id]  # keep depth K for backprojection
            depth_raw = _img_to_numpy_depth(depth_msg)
            depth_m = _depth_to_meters(depth_raw, depth_msg.encoding)

            # --------------------------------
            # 3) Option 1: RGB numpy conversion
            # --------------------------------
            try:
                rgb = _img_to_numpy_rgb(rgb_msg)  # HxWx3 uint8 RGB
            except Exception as e:
                if not self.use_best_effort_if_unsynced:
                    self.get_logger().warn(f"{cam_id} failed RGB conversion: {e}")
                    return None
                rgb = None

            # -----------------------------------------
            # 4) Option 3: shape sanity (throttled warn)
            # -----------------------------------------
            if rgb is not None:
                Hd, Wd = depth_m.shape[:2]
                Hr, Wr = rgb.shape[:2]
                if (Hd != Hr) or (Wd != Wr):
                    if (now_wall - self._last_warn_shape_t) > 5.0:
                        self._last_warn_shape_t = now_wall
                        self.get_logger().warn(
                            f"{cam_id} depth shape {depth_m.shape} != rgb shape {rgb.shape}. "
                            f"Masking/colorizing assumes pixel alignment; ROI/chroma masking still works "
                            f"only if indices correspond."
                        )

            # -----------------------------------------
            # 5) Option 3: frozen RGB detection (cheap)
            # -----------------------------------------
            if rgb is not None:
                st = self._rgb_frozen_state.get(cam_id)
                if st is None:
                    st = {"sig": None, "same_count": 0}
                    self._rgb_frozen_state[cam_id] = st

                # Subsample signature (very cheap)
                small = rgb[::16, ::16, :].reshape(-1)
                sig = hash(bytes(small[:2048]))  # cap bytes; good enough for "frozen" detection

                if st["sig"] == sig:
                    st["same_count"] += 1
                else:
                    st["sig"] = sig
                    st["same_count"] = 0

                # Warn if frozen for ~N ticks; throttled
                # (Assumes you call this at ~5–10 Hz)
                if st["same_count"] >= 30:  # ~3–6 seconds frozen
                    if (now_wall - self._last_warn_frozen_t) > 5.0:
                        self._last_warn_frozen_t = now_wall
                        self.get_logger().warn(
                            f"{cam_id} RGB appears frozen (same frame signature for {st['same_count']} cycles)."
                        )

            # Build view
            T_base_cam = self.T_base_cam_map.get(cam_id, SE3.identity())

            views.append(
                View(
                    cam_id=cam_id,
                    rgb=rgb,          # Option 1/2 input
                    depth=depth_m,
                    K=K,
                    T_base_cam=T_base_cam,
                    stamp_s=t_d,
                )
            )
        return views