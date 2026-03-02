from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo

from src.utils.se3 import SE3
from src.perception.view import View


@dataclass
class CameraTopics:
    cam_id: str
    depth_topic: str
    info_topic: str


def _stamp_to_sec(stamp) -> float:
    # stamp is builtin_interfaces/msg/Time
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def _K_from_camerainfo(msg: CameraInfo) -> np.ndarray:
    # msg.k is row-major 3x3
    K = np.array(msg.k, dtype=float).reshape(3, 3)
    return K


def _depth_to_meters(depth: np.ndarray, encoding: str) -> np.ndarray:
    """
    Returns float32 meters. Supports common encodings:
      - 32FC1: float meters
      - 16UC1: uint16 millimeters
    """
    if encoding == "32FC1":
        d = depth.astype(np.float32)
        return d
    if encoding == "16UC1":
        # millimeters -> meters
        d = depth.astype(np.float32) * 1e-3
        return d
    raise ValueError(f"Unsupported depth encoding: {encoding}")


def _img_to_numpy(msg: Image) -> np.ndarray:
    """
    Minimal conversion without cv_bridge.
    Assumes tightly packed rows or uses step.
    """
    h, w = int(msg.height), int(msg.width)
    enc = msg.encoding

    if enc == "32FC1":
        dtype = np.float32
        elem_size = 4
    elif enc == "16UC1":
        dtype = np.uint16
        elem_size = 2
    else:
        raise ValueError(f"Unsupported Image encoding: {enc}")

    data = np.frombuffer(msg.data, dtype=dtype)

    # handle row stride
    step_elems = int(msg.step // elem_size)
    if step_elems == w:
        return data.reshape(h, w)
    # padded rows
    out = np.zeros((h, w), dtype=dtype)
    for r in range(h):
        out[r, :] = data[r * step_elems : r * step_elems + w]
    return out


class MultiCamGrabber(Node):
    """
    Buffers latest depth + camera_info for multiple cameras and produces a synced View set.
    """

    def __init__(
        self,
        cameras: list[CameraTopics],
        sync_slop_s: float = 0.05,
        use_best_effort_if_unsynced: bool = False,
        static_extrinsics_base_cam: Optional[dict[str, SE3]] = None,
    ):
        super().__init__("multicam_grabber")

        self.cameras = cameras
        self.sync_slop_s = float(sync_slop_s)
        self.use_best_effort_if_unsynced = bool(use_best_effort_if_unsynced)

        self.T_base_cam_map: dict[str, SE3] = static_extrinsics_base_cam or {}

        # Buffers
        self._depth_msg: dict[str, Image] = {}
        self._depth_stamp_s: dict[str, float] = {}
        self._K: dict[str, np.ndarray] = {}

        # subs
        self._subs = []
        for c in cameras:
            self.get_logger().info(f"Subscribing cam_id={c.cam_id} depth={c.depth_topic} info={c.info_topic}")

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
                    lambda msg, cam_id=c.cam_id: self._on_info(cam_id, msg),
                    10,
                )
            )

    def _on_depth(self, cam_id: str, msg: Image) -> None:
        self._depth_msg[cam_id] = msg
        self._depth_stamp_s[cam_id] = _stamp_to_sec(msg.header.stamp)

    def _on_info(self, cam_id: str, msg: CameraInfo) -> None:
        self._K[cam_id] = _K_from_camerainfo(msg)

    def ready(self) -> bool:
        for c in self.cameras:
            if c.cam_id not in self._depth_msg:
                return False
            if c.cam_id not in self._K:
                return False
        return True

    def get_latest_views(self) -> Optional[list[View]]:
        """
        Returns a synced list[View] in base frame if possible.
        If not synced:
          - returns None (default), or
          - returns best-effort set if use_best_effort_if_unsynced=True
        """
        if not self.ready():
            return None

        stamps = [self._depth_stamp_s[c.cam_id] for c in self.cameras]
        t_ref = max(stamps)
        max_dt = max(abs(t - t_ref) for t in stamps)

        if max_dt > self.sync_slop_s and not self.use_best_effort_if_unsynced:
            self.get_logger().warn(f"unsynced set: max_dt={max_dt:.3f}s > slop={self.sync_slop_s:.3f}s")
            return None

        views: list[View] = []
        for c in self.cameras:
            depth_msg = self._depth_msg[c.cam_id]
            K = self._K[c.cam_id]

            depth_raw = _img_to_numpy(depth_msg)
            depth_m = _depth_to_meters(depth_raw, depth_msg.encoding)

            finite = np.isfinite(depth_m)
            if finite.any():
                dmin = float(depth_m[finite].min())
                dmed = float(np.median(depth_m[finite]))
                dmax = float(depth_m[finite].max())
                self.get_logger().info(f"{c.cam_id} depth stats [m]: min={dmin:.3f} med={dmed:.3f} max={dmax:.3f}")
            else:
                self.get_logger().warn(f"{c.cam_id} depth is all NaN/Inf")
            T_base_cam = self.T_base_cam_map.get(c.cam_id, SE3.identity())

            views.append(
                View(
                    cam_id=c.cam_id,
                    rgb=None,
                    depth=depth_m,
                    K=K,
                    T_base_cam=T_base_cam,
                    stamp_s=self._depth_stamp_s[c.cam_id],
                )
            )

        return views