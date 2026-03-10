from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import ColorRGBA, Header
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker

from src.calibration.io_extrinsics import load_extrinsics_yaml
from src.perception.pipeline_fast import GraspPerceptionPipeline, PipelineConfig
from src.perception.pipeline_multiview_fast import MultiViewConfig, MultiViewRunner
from src.perception.pose_icp_fast import load_cad_as_pointcloud
from src.perception.ros.multicam_grabber import CameraTopics, MultiCamGrabber


CAMERAS = [
    CameraTopics(
        cam_id="zed2i_1",
        depth_topic="/zed2i_1/zed_node/depth/depth_registered",
        info_topic="/zed2i_1/zed_node/depth/depth_registered/camera_info",
        rgb_topic="/zed2i_1/zed_node/rgb/color/rect/image",
        rgb_info_topic="/zed2i_1/zed_node/rgb/color/rect/image/camera_info",
    ),
    CameraTopics(
        cam_id="zed2i_2",
        depth_topic="/zed2i_2/zed_node/depth/depth_registered",
        info_topic="/zed2i_2/zed_node/depth/depth_registered/camera_info",
        rgb_topic="/zed2i_2/zed_node/rgb/color/rect/image",
        rgb_info_topic="/zed2i_2/zed_node/rgb/color/rect/image/camera_info",
    ),
]


BASE_FRAME = "base"
TIMER_PERIOD_S = 0.10
MIN_PROCESS_DT_S = 0.08
PUBLISH_RGB = False
RGB_PUBLISH_EVERY_N = 5
PUBLISH_FUSED_CLOUD = True
CLOUD_PUBLISH_EVERY_N = 1
MAX_CLOUD_POINTS_PUBLISH = 100000
ENABLE_AABB_MARKERS = True
ENABLE_CAD_MARKERS = True


FAST_QOS = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
)


def _mat3_to_quat_xyzw(R: np.ndarray) -> Tuple[float, float, float, float]:
    m = np.asarray(R, dtype=float)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0.0:
        S = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (m[2, 1] - m[1, 2]) / S
        qy = (m[0, 2] - m[2, 0]) / S
        qz = (m[1, 0] - m[0, 1]) / S
    elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
        S = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        qw = (m[2, 1] - m[1, 2]) / S
        qx = 0.25 * S
        qy = (m[0, 1] + m[1, 0]) / S
        qz = (m[0, 2] + m[2, 0]) / S
    elif m[1, 1] > m[2, 2]:
        S = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        qw = (m[0, 2] - m[2, 0]) / S
        qx = (m[0, 1] + m[1, 0]) / S
        qy = 0.25 * S
        qz = (m[1, 2] + m[2, 1]) / S
    else:
        S = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        qw = (m[1, 0] - m[0, 1]) / S
        qx = (m[0, 2] + m[2, 0]) / S
        qy = (m[1, 2] + m[2, 1]) / S
        qz = 0.25 * S
    return float(qx), float(qy), float(qz), float(qw)


def _rgb_numpy_to_imgmsg(rgb: np.ndarray, frame_id: str, stamp) -> Image:
    rgb = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8))
    msg = Image()
    msg.header = Header(frame_id=frame_id, stamp=stamp)
    msg.height = int(rgb.shape[0])
    msg.width = int(rgb.shape[1])
    msg.encoding = "rgb8"
    msg.is_bigendian = False
    msg.step = int(rgb.shape[1] * 3)
    msg.data = rgb.tobytes()
    return msg


def _points_to_pointcloud2(points_xyz: np.ndarray, frame_id: str, stamp) -> PointCloud2:
    pts = np.asarray(points_xyz, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        pts = pts.reshape(-1, 3)

    if pts.size > 0:
        finite = np.isfinite(pts).all(axis=1)
        pts = pts[finite]

    pts = np.ascontiguousarray(pts, dtype=np.float32)

    msg = PointCloud2()
    msg.header = Header(frame_id=frame_id, stamp=stamp)
    msg.height = 1
    msg.width = int(pts.shape[0])
    msg.is_bigendian = False
    msg.is_dense = True
    msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    msg.point_step = 12
    msg.row_step = msg.point_step * msg.width
    msg.data = pts.tobytes()
    return msg


def _downsample_points_for_publish(points_xyz: np.ndarray, max_points: int) -> np.ndarray:
    pts = np.asarray(points_xyz, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        pts = pts.reshape(-1, 3)
    if len(pts) <= max_points:
        return pts
    stride = max(1, int(np.ceil(len(pts) / max_points)))
    return pts[::stride]


def _try_get_view_stamp_ns(view: Any) -> Optional[int]:
    for attr in (
        "stamp_ns",
        "timestamp_ns",
        "depth_stamp_ns",
        "rgb_stamp_ns",
        "stamp",
        "depth_stamp",
        "rgb_stamp",
    ):
        value = getattr(view, attr, None)
        if value is None:
            continue
        if hasattr(value, "nanoseconds"):
            return int(value.nanoseconds)
        sec = getattr(value, "sec", None)
        nanosec = getattr(value, "nanosec", None)
        if sec is not None and nanosec is not None:
            return int(sec) * 1_000_000_000 + int(nanosec)
        if isinstance(value, (int, np.integer)):
            return int(value)
        if isinstance(value, float):
            return int(value * 1e9)
    return None


class LiveMultiViewDebug(Node):
    def __init__(self, grabber: MultiCamGrabber):
        super().__init__("live_multiview_debug")
        self.grabber = grabber

        self._busy = False
        self._frame = 0
        self._last_process_wall_t = 0.0
        self._last_views_signature: Optional[Tuple[Tuple[str, int], ...]] = None
        self._last_result: Any = None

        self.cad_library = {
            "cube": load_cad_as_pointcloud("Data/CAD_Models/Cube.stl", scale=0.003822, center=True),
        }

        cube_pts = self.cad_library["cube"]
        cube_ext = cube_pts.max(axis=0) - cube_pts.min(axis=0)
        self.cube_size_xyz = tuple(float(x) for x in cube_ext)

        pipe_cfg = PipelineConfig(
            plane_distance_threshold=0.004,
            expected_num_objects=3,
            dbscan_eps=0.035,
            dbscan_min_points=25,
            voxel_size=0.005,
            max_rms_nn=0.020,
            min_margin=1.2,
            accept_icp_rmse_max=0.015,
            accept_icp_fitness_min=0.15,
            accept_cluster_points_min=120,
        )
        pipe = GraspPerceptionPipeline(cad_library=self.cad_library, cfg=pipe_cfg)

        mv_cfg = MultiViewConfig(
            voxel_size_fusion=0.005,
            stride=4,
            zmin=0.30,
            zmax=1.05,
            roi_x_min=-0.15,
            roi_x_max=0.30,
            roi_y_min=-0.15,
            roi_y_max=0.30,
            roi_z_min=0.30,
            roi_z_max=1.05,
        )
        self.runner = MultiViewRunner(pipe, cfg=mv_cfg)

        self.pub_cloud_fused = self.create_publisher(PointCloud2, "/perception/cloud/fused", FAST_QOS)

        self.pub_rgb: Dict[str, Any] = {}
        for c in CAMERAS:
            self.pub_rgb[c.cam_id] = self.create_publisher(Image, f"/perception/rgb/{c.cam_id}", FAST_QOS)

        self.pub_pose: Dict[str, Any] = {}
        self.tf_broadcaster = TransformBroadcaster(self)
        self.pub_markers = self.create_publisher(Marker, "/perception/markers", FAST_QOS)

        self.timer = self.create_timer(TIMER_PERIOD_S, self._tick)

        self.get_logger().info("LiveMultiViewDebug started (fast version)")
        self.get_logger().info(f"[CAD] cube extents (marker scale) = {self.cube_size_xyz}")
        self.get_logger().info(
            f"[CFG] stride={mv_cfg.stride} voxel_fusion={mv_cfg.voxel_size_fusion:.3f} "
            f"pipe_voxel={pipe_cfg.voxel_size:.3f} timer={TIMER_PERIOD_S:.2f}s"
        )

    def _make_marker_base(self, ns: str, marker_id: int, stamp, frame_id: str = BASE_FRAME) -> Marker:
        m = Marker()
        m.header.frame_id = frame_id
        m.header.stamp = stamp
        m.ns = ns
        m.id = int(marker_id)
        m.action = Marker.ADD
        m.lifetime.sec = 0
        m.lifetime.nanosec = 0
        return m

    def _publish_aabb_marker(self, obj_cloud: np.ndarray, marker_id: int, stamp, obj_id: str) -> None:
        pts = np.asarray(obj_cloud, dtype=np.float32)
        if pts.size == 0:
            return

        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        center = (mins + maxs) / 2.0
        size = maxs - mins

        m = self._make_marker_base(ns="aabb", marker_id=marker_id, stamp=stamp)
        m.type = Marker.CUBE
        m.pose.position.x = float(center[0])
        m.pose.position.y = float(center[1])
        m.pose.position.z = float(center[2])
        m.pose.orientation.w = 1.0
        eps = 1e-4
        m.scale.x = float(max(size[0], eps))
        m.scale.y = float(max(size[1], eps))
        m.scale.z = float(max(size[2], eps))
        m.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.30)
        m.text = f"{obj_id}_aabb"
        self.pub_markers.publish(m)

    def _publish_fitted_cube_marker(self, T: Any, marker_id: int, stamp, obj_id: str) -> None:
        qx, qy, qz, qw = _mat3_to_quat_xyzw(T.R)

        m = self._make_marker_base(ns="cad", marker_id=marker_id, stamp=stamp)
        m.type = Marker.CUBE
        m.pose.position.x = float(T.t[0])
        m.pose.position.y = float(T.t[1])
        m.pose.position.z = float(T.t[2])
        m.pose.orientation.x = qx
        m.pose.orientation.y = qy
        m.pose.orientation.z = qz
        m.pose.orientation.w = qw

        sx, sy, sz = self.cube_size_xyz
        m.scale.x = float(max(sx, 1e-4))
        m.scale.y = float(max(sy, 1e-4))
        m.scale.z = float(max(sz, 1e-4))
        m.color = ColorRGBA(r=1.0, g=0.1, b=0.1, a=0.35)
        m.text = f"{obj_id}_cad"
        self.pub_markers.publish(m)

    def _views_signature(self, views: Any) -> Optional[Tuple[Tuple[str, int], ...]]:
        signature = []
        for v in views:
            stamp_ns = _try_get_view_stamp_ns(v)
            if stamp_ns is None:
                return None
            signature.append((str(v.cam_id), int(stamp_ns)))
        signature.sort(key=lambda x: x[0])
        return tuple(signature)

    def _maybe_publish_rgb(self, views: Any, stamp) -> None:
        if not PUBLISH_RGB or (self._frame % RGB_PUBLISH_EVERY_N != 0):
            return
        for v in views:
            pub = self.pub_rgb[v.cam_id]
            if pub.get_subscription_count() == 0:
                continue
            if getattr(v, "rgb", None) is None:
                continue
            pub.publish(_rgb_numpy_to_imgmsg(v.rgb, frame_id=v.cam_id, stamp=stamp))

    def _select_cloud_to_publish(self, result: Any) -> Optional[np.ndarray]:
        for attr in ("debug_points_world_raw", "points_world_raw", "points_world_roi", "points_world"):
            pts = getattr(result, attr, None)
            if pts is not None:
                return pts
        return None

    def _publish_result(self, result: Any, stamp) -> None:
        if result is None:
            return

        if (
            PUBLISH_FUSED_CLOUD
            and self.pub_cloud_fused.get_subscription_count() > 0
            and self._frame % CLOUD_PUBLISH_EVERY_N == 0
        ):
            pts = self._select_cloud_to_publish(result)
            if pts is not None:
                pts_pub = _downsample_points_for_publish(pts, MAX_CLOUD_POINTS_PUBLISH)
                self.pub_cloud_fused.publish(_points_to_pointcloud2(pts_pub, frame_id=BASE_FRAME, stamp=stamp))

        objs = getattr(result, "objects", None) or []
        for idx, obj in enumerate(objs):
            obj_id = getattr(obj, "object_id", f"obj{idx}")
            if obj_id not in self.pub_pose:
                self.pub_pose[obj_id] = self.create_publisher(PoseStamped, f"/perception/pose/{obj_id}", FAST_QOS)

            T = getattr(obj, "T_object_to_world", None)
            if T is None:
                continue

            qx, qy, qz, qw = _mat3_to_quat_xyzw(T.R)

            pose_pub = self.pub_pose[obj_id]
            if pose_pub.get_subscription_count() > 0:
                pose = PoseStamped()
                pose.header.frame_id = BASE_FRAME
                pose.header.stamp = stamp
                pose.pose.position.x = float(T.t[0])
                pose.pose.position.y = float(T.t[1])
                pose.pose.position.z = float(T.t[2])
                pose.pose.orientation.x = qx
                pose.pose.orientation.y = qy
                pose.pose.orientation.z = qz
                pose.pose.orientation.w = qw
                pose_pub.publish(pose)

            tf = TransformStamped()
            tf.header.frame_id = BASE_FRAME
            tf.header.stamp = stamp
            tf.child_frame_id = f"object/{obj_id}"
            tf.transform.translation.x = float(T.t[0])
            tf.transform.translation.y = float(T.t[1])
            tf.transform.translation.z = float(T.t[2])
            tf.transform.rotation.x = qx
            tf.transform.rotation.y = qy
            tf.transform.rotation.z = qz
            tf.transform.rotation.w = qw
            self.tf_broadcaster.sendTransform(tf)

            cloud = getattr(obj, "point_cloud", None)
            if ENABLE_AABB_MARKERS and cloud is not None and len(cloud) > 0:
                self._publish_aabb_marker(cloud, marker_id=1000 + idx, stamp=stamp, obj_id=obj_id)

            if ENABLE_CAD_MARKERS and obj_id == "cube":
                self._publish_fitted_cube_marker(T, marker_id=2000 + idx, stamp=stamp, obj_id=obj_id)

    def _tick(self) -> None:
        now = time.perf_counter()
        if self._busy:
            return
        if now - self._last_process_wall_t < MIN_PROCESS_DT_S:
            return

        views = self.grabber.get_latest_views()
        if views is None:
            return

        signature = self._views_signature(views)
        if signature is not None and signature == self._last_views_signature:
            return

        self._busy = True
        t0 = time.perf_counter()
        try:
            self._frame += 1
            stamp = self.get_clock().now().to_msg()

            self._maybe_publish_rgb(views, stamp)

            t_run0 = time.perf_counter()
            result = self.runner.run(views)
            t_run1 = time.perf_counter()

            self._last_result = result
            self._last_views_signature = signature
            self._last_process_wall_t = time.perf_counter()

            t_pub0 = time.perf_counter()
            self._publish_result(result, stamp)
            t_pub1 = time.perf_counter()

            self.get_logger().info(
                f"[TIMING run_multiview_ros_fast] "
                f"runner.run={(t_run1 - t_run0) * 1000:.1f} ms | "
                f"publish={(t_pub1 - t_pub0) * 1000:.1f} ms | "
                f"total={(t_pub1 - t0) * 1000:.1f} ms"
            )

            if self._frame % 10 == 0:
                pts = self._select_cloud_to_publish(result)
                n_pts = 0 if pts is None else len(pts)
                n_obj = len(getattr(result, "objects", None) or [])
                dt_ms = (time.perf_counter() - t0) * 1000.0
                hz = 1000.0 / max(dt_ms, 1e-6)
                self.get_logger().info(
                    f"[frame {self._frame}] pts={n_pts} objects={n_obj} proc={dt_ms:.1f} ms ({hz:.1f} Hz)"
                )
        finally:
            self._busy = False


def main() -> None:
    rclpy.init()

    T_map = load_extrinsics_yaml("config/camera_extrinsics.yaml")
    grabber = MultiCamGrabber(
        cameras=CAMERAS,
        sync_slop_s=0.10,
        use_best_effort_if_unsynced=True,
        static_extrinsics_base_cam=T_map,
        rgb_depth_max_dt_s=0.08,
    )

    node = LiveMultiViewDebug(grabber)

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(grabber)
    executor.add_node(node)

    try:
        executor.spin()
    finally:
        executor.remove_node(node)
        executor.remove_node(grabber)
        node.destroy_node()
        grabber.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()