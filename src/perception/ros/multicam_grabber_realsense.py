"""Variant of multicam_grabber.py for the 1-ZED + 2-RealSense trio.

Adds support for cameras whose extrinsic is not fixed: end-effector-mounted
RealSense cameras move with the robot arm, so their camera-to-base transform
must be recomputed every frame as

    T_base_cam(t) = T_base_flange(t) @ T_flange_cam

instead of being read from a static map. Everything else (image decoding,
sync logic, View assembly) is unchanged from multicam_grabber.py.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped

from src.utils.se3 import SE3
from src.perception.view import View

import time

# Re-exported so callers can `from multicam_grabber_realsense import CameraTopics`.
from src.perception.ros.multicam_grabber import (  # noqa: F401
    CameraTopics,
    _stamp_to_sec,
    _K_from_camerainfo,
    _depth_to_meters,
    _img_to_numpy_depth,
    _img_to_numpy_rgb,
)


@dataclass
class DynamicCameraTopics(CameraTopics):
    """A CameraTopics entry whose extrinsic is looked up dynamically.

    is_dynamic=False behaves exactly like the base CameraTopics (static
    camera-to-base extrinsic from the loaded YAML map).

    is_dynamic=True means the YAML entry for this cam_id holds a
    camera-to-FLANGE offset instead of a camera-to-base transform; the
    live camera-to-base extrinsic is computed each frame from the most
    recent flange PoseStamped message.
    """
    is_dynamic: bool = False


def _pose_msg_to_se3(msg: PoseStamped) -> SE3:
    p = msg.pose.position
    q = msg.pose.orientation
    # Quaternion (x, y, z, w) -> rotation matrix.
    x, y, z, w = q.x, q.y, q.z, q.w
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=float)
    t = np.array([p.x, p.y, p.z], dtype=float)
    return SE3(R, t)


class _FlangeComposedExtrinsicsMap:
    """dict[str, SE3]-like object: static passthrough for most cameras,
    live T_base_flange @ T_flange_cam composition for dynamic ones.

    Drop-in replacement for the plain `dict[str, SE3]` that
    MultiCamGrabber / FoundationPoseTrackerNode / multicam_fusion read via
    `T_base_cam_map[cam_id]` / `.get(cam_id, ...)`.
    """

    def __init__(
        self,
        static_map: dict[str, SE3],
        dynamic_cam_ids: set[str],
        flange_pose_provider,
    ):
        self._static_map = static_map
        self._dynamic_cam_ids = set(dynamic_cam_ids)
        # Callable[[], Optional[SE3]] returning the latest T_base_flange.
        self._flange_pose_provider = flange_pose_provider

    def _resolve(self, cam_id: str) -> Optional[SE3]:
        if cam_id in self._dynamic_cam_ids:
            T_base_flange = self._flange_pose_provider()
            if T_base_flange is None:
                return None
            T_flange_cam = self._static_map[cam_id]
            return T_base_flange.compose(T_flange_cam)
        return self._static_map.get(cam_id)

    def __getitem__(self, cam_id: str) -> SE3:
        T = self._resolve(cam_id)
        if T is None:
            raise KeyError(
                f"No extrinsic available yet for cam_id={cam_id} "
                f"(dynamic camera with no flange pose received yet?)"
            )
        return T

    def __contains__(self, cam_id: str) -> bool:
        if cam_id in self._dynamic_cam_ids:
            return self._flange_pose_provider() is not None
        return cam_id in self._static_map

    def get(self, cam_id: str, default=None):
        T = self._resolve(cam_id)
        return default if T is None else T


# ROS node that buffers each camera's latest RGB/depth/intrinsics and serves
# time-synced View sets, with dynamic (end-effector-mounted) camera support.
class MultiCamGrabberRealsense(Node):
    def __init__(
        self,
        cameras: list[DynamicCameraTopics],
        sync_slop_s: float = 0.015,
        use_best_effort_if_unsynced: bool = True,
        static_extrinsics_base_cam: Optional[dict[str, SE3]] = None,
        rgb_depth_max_dt_s: float = 0.10,
        flange_pose_topic: str = "/iiwa/ee_pose",
        flange_pose_max_age_s: float = 0.25,
        T_robotA_activeRobot: Optional[SE3] = None,
    ):
        super().__init__("multicam_grabber_realsense")

        self.cameras = cameras
        self.sync_slop_s = float(sync_slop_s)
        self.use_best_effort_if_unsynced = bool(use_best_effort_if_unsynced)
        self.rgb_depth_max_dt_s = float(rgb_depth_max_dt_s)
        self.flange_pose_max_age_s = float(flange_pose_max_age_s)
        # None = leave everything in whichever robot is physically connected's
        # frame (old behavior). If given, static (non-dynamic) entries -- true
        # base-frame poses -- are shifted into robot_a's frame at load time,
        # and the live flange pose is shifted the same way on receipt
        # (_on_flange_pose), so the static zed2i_1 lookup and the live
        # T_base_flange(t) @ T_flange_cam composition stay in the same frame.
        # T_flange_cam entries (dynamic cams) are camera-to-FLANGE, not
        # base-frame, so they must NOT be converted.
        self._T_robotA_activeRobot = T_robotA_activeRobot

        dynamic_ids = {c.cam_id for c in cameras if c.is_dynamic}
        raw_static_map = static_extrinsics_base_cam or {}
        if T_robotA_activeRobot is not None:
            raw_static_map = {
                cam_id: (T_robotA_activeRobot.compose(T) if cam_id not in dynamic_ids else T)
                for cam_id, T in raw_static_map.items()
            }

        self._latest_flange_pose: Optional[SE3] = None
        self._latest_flange_pose_wall_t: float = 0.0

        self.T_base_cam_map = _FlangeComposedExtrinsicsMap(
            static_map=raw_static_map,
            dynamic_cam_ids=dynamic_ids,
            flange_pose_provider=self._get_flange_pose,
        )

        if dynamic_ids:
            self.get_logger().info(
                f"Subscribing flange pose topic={flange_pose_topic} "
                f"for dynamic cameras={sorted(dynamic_ids)}"
            )
            self._flange_sub = self.create_subscription(
                PoseStamped,
                flange_pose_topic,
                self._on_flange_pose,
                qos_profile_sensor_data,
            )

        # Buffers
        self._depth_msg: dict[str, Image] = {}
        self._depth_stamp_s: dict[str, float] = {}
        self._K_depth: dict[str, np.ndarray] = {}

        self._rgb_msg: dict[str, Image] = {}
        self._rgb_stamp_s: dict[str, float] = {}
        self._K_rgb: dict[str, np.ndarray] = {}

        # Subscribe each camera's depth + rgb image and their camera_info topics.
        self._subs = []
        for c in cameras:
            self.get_logger().info(
                f"Subscribing cam_id={c.cam_id} depth={c.depth_topic} depth_info={c.info_topic} "
                f"rgb={c.rgb_topic} rgb_info={c.rgb_info_topic} dynamic={c.is_dynamic}"
            )

            self._subs.append(
                self.create_subscription(
                    Image,
                    c.depth_topic,
                    lambda msg, cam_id=c.cam_id: self._on_depth(cam_id, msg),
                    qos_profile_sensor_data,
                )
            )
            self._subs.append(
                self.create_subscription(
                    CameraInfo,
                    c.info_topic,
                    lambda msg, cam_id=c.cam_id: self._on_depth_info(cam_id, msg),
                    qos_profile_sensor_data,
                )
            )

            self._subs.append(
                self.create_subscription(
                    Image,
                    c.rgb_topic,
                    lambda msg, cam_id=c.cam_id: self._on_rgb(cam_id, msg),
                    qos_profile_sensor_data,
                )
            )
            self._subs.append(
                self.create_subscription(
                    CameraInfo,
                    c.rgb_info_topic,
                    lambda msg, cam_id=c.cam_id: self._on_rgb_info(cam_id, msg),
                    qos_profile_sensor_data,
                )
            )

    def _on_flange_pose(self, msg: PoseStamped) -> None:
        T = _pose_msg_to_se3(msg)
        if self._T_robotA_activeRobot is not None:
            T = self._T_robotA_activeRobot.compose(T)
        self._latest_flange_pose = T
        self._latest_flange_pose_wall_t = time.time()

    def _get_flange_pose(self) -> Optional[SE3]:
        if self._latest_flange_pose is None:
            return None
        age = time.time() - self._latest_flange_pose_wall_t
        if age > self.flange_pose_max_age_s:
            return None
        return self._latest_flange_pose

    # Subscription callbacks: stash each latest message (and its stamp / intrinsics) per camera.
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
        # True only once every camera has both images, both intrinsics buffered,
        # and (for dynamic cameras) a fresh flange pose.
        for c in self.cameras:
            if c.cam_id not in self._depth_msg:
                return False
            if c.cam_id not in self._K_depth:
                return False
            if c.cam_id not in self._rgb_msg:
                return False
            if c.cam_id not in self._K_rgb:
                return False
            if c.is_dynamic and self._get_flange_pose() is None:
                return False
        return True

    def get_latest_views(self) -> Optional[list[View]]:
        """
        Returns a synced list[View] in base frame if possible.
        """
        if not self.ready():
            return None

        # Reject the set if the cameras' depth stamps span more than the sync slop.
        stamps_d = [self._depth_stamp_s[c.cam_id] for c in self.cameras]
        t_ref = max(stamps_d)
        max_dt = max(abs(t - t_ref) for t in stamps_d)

        if max_dt > self.sync_slop_s and not self.use_best_effort_if_unsynced:
            now = time.time()
            last = getattr(self, "_last_warn_unsynced_depth_t", 0.0)
            if (now - last) > 1.0:
                self._last_warn_unsynced_depth_t = now
                self.get_logger().warn(
                    f"unsynced depth set: max_dt={max_dt:.3f}s > slop={self.sync_slop_s:.3f}s"
                )
            return None

        views: list[View] = []

        if not hasattr(self, "_rgb_frozen_state"):
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

            dt_rgb_depth = abs(t_rgb - t_d)
            if dt_rgb_depth > self.rgb_depth_max_dt_s and not self.use_best_effort_if_unsynced:
                if (now_wall - self._last_warn_rgb_depth_dt_t) > 1.0:
                    self._last_warn_rgb_depth_dt_t = now_wall
                    self.get_logger().warn(
                        f"{cam_id} rgb-depth unsynced: dt={dt_rgb_depth:.3f}s > "
                        f"{self.rgb_depth_max_dt_s:.3f}s"
                    )
                return None

            K = self._K_depth[cam_id]
            depth_raw = _img_to_numpy_depth(depth_msg)
            depth_m = _depth_to_meters(depth_raw, depth_msg.encoding)

            try:
                rgb = _img_to_numpy_rgb(rgb_msg)
            except Exception as e:
                if not self.use_best_effort_if_unsynced:
                    self.get_logger().warn(f"{cam_id} failed RGB conversion: {e}")
                    return None
                rgb = None

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

            if rgb is not None:
                st = self._rgb_frozen_state.get(cam_id)
                if st is None:
                    st = {"sig": None, "same_count": 0}
                    self._rgb_frozen_state[cam_id] = st

                small = rgb[::16, ::16, :].reshape(-1)
                sig = hash(bytes(small[:2048]))

                if st["sig"] == sig:
                    st["same_count"] += 1
                else:
                    st["sig"] = sig
                    st["same_count"] = 0

                if st["same_count"] >= 30:
                    if (now_wall - self._last_warn_frozen_t) > 5.0:
                        self._last_warn_frozen_t = now_wall
                        self.get_logger().warn(
                            f"{cam_id} RGB appears frozen (same frame signature for {st['same_count']} cycles)."
                        )

            # Assemble the per-camera View with its base-frame extrinsic
            # (static lookup, or live T_base_flange @ T_flange_cam for
            # dynamic end-effector-mounted cameras).
            T_base_cam = self.T_base_cam_map.get(cam_id, SE3.identity())

            views.append(
                View(
                    cam_id=cam_id,
                    rgb=rgb,
                    depth=depth_m,
                    K=K,
                    T_base_cam=T_base_cam,
                    stamp_s=t_d,
                )
            )
        return views
