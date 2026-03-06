from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import rclpy
from src.utils.se3 import SE3
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header, ColorRGBA
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker

from src.calibration.io_extrinsics import load_extrinsics_yaml
from src.perception.pipeline import GraspPerceptionPipeline, PipelineConfig
from src.perception.pipeline_multiview import MultiViewRunner, MultiViewConfig
from src.perception.pose_icp import load_cad_as_pointcloud
from src.perception.ros.multicam_grabber import MultiCamGrabber, CameraTopics
from src.perception.backproject import depth_to_points_cam


# IMPORTANT: these match your `ros2 topic list | grep zed`
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


def _mat3_to_quat_xyzw(R: np.ndarray) -> Tuple[float, float, float, float]:
    """Convert 3x3 rotation matrix to quaternion (x,y,z,w)."""
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
    rgb = np.asarray(rgb, dtype=np.uint8)
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


class LiveMultiViewDebug(Node):
    """
    Continuous loop:
      - fetches synced views
      - runs the pipeline repeatedly
      - publishes topics for Foxglove
      - publishes:
          * AABB marker around each detected cluster (axis-aligned)
          * Fitted CAD cube marker (pose + known cube size)
    """

    def __init__(self, grabber: MultiCamGrabber):
        super().__init__("live_multiview_debug")
        self.grabber = grabber

        # ---- CAD library ----
        self.cad_library = {
            "cube": load_cad_as_pointcloud("Data/CAD_Models/Cube.stl", scale=0.003822, center=True),
        }

        # Precompute cube size (scale) from CAD points, so marker matches real cube size
        cube_pts = self.cad_library["cube"]
        cube_ext = cube_pts.max(axis=0) - cube_pts.min(axis=0)
        # For a cube, extents should be near equal; still use per-axis extents to be safe
        self.cube_size_xyz = tuple(float(x) for x in cube_ext)

        pipe_cfg = PipelineConfig(
            plane_distance_threshold=0.002,
            dbscan_eps=0.03,
            dbscan_min_points=40,
            voxel_size=0.005,
            max_rms_nn=0.012,
            min_margin=1.5,
        )
        pipe = GraspPerceptionPipeline(cad_library=self.cad_library, cfg=pipe_cfg)

        mv_cfg = MultiViewConfig(
            voxel_size_fusion=0.003,
            stride=1,
            zmin=0.25,
            zmax=1.1,
            # >>> TABLE ROI (edit these) <<<
            roi_x_min=-0.25,
            roi_x_max=0.40,
            roi_y_min=-0.25,
            roi_y_max=0.40,
            roi_z_min=0.35,
            roi_z_max=1.10,
        )
        self.runner = MultiViewRunner(pipe, cfg=mv_cfg)

        # ---- publishers ----
        self.pub_cloud_fused = self.create_publisher(PointCloud2, "/perception/cloud/fused", 1)

        self.pub_rgb: Dict[str, any] = {}
        for c in CAMERAS:
            self.pub_rgb[c.cam_id] = self.create_publisher(Image, f"/perception/rgb/{c.cam_id}", 1)

        self.pub_pose: Dict[str, any] = {}  # per object_id
        self.tf_broadcaster = TransformBroadcaster(self)

        # markers (AABB + fitted CAD)
        self.pub_markers = self.create_publisher(Marker, "/perception/markers", 10)

        # loop rate
        self.timer = self.create_timer(0.2, self._tick)  # 5 Hz
        self._frame = 0

        self.get_logger().info(
            "LiveMultiViewDebug started (publishing RGB, cloud, poses, TF, markers)"
        )
        self.get_logger().info(f"[CAD] cube extents (marker scale) = {self.cube_size_xyz}")

        #self.pub_cloud_cam = {
        #    "zed2i_1": self.create_publisher(PointCloud2, "/perception/cloud/cam/zed2i_1", 1),
        #    "zed2i_2": self.create_publisher(PointCloud2, "/perception/cloud/cam/zed2i_2", 1),
        #}

    def _make_marker_base(
        self,
        ns: str,
        marker_id: int,
        stamp,
        frame_id: str = "base",
    ) -> Marker:
        m = Marker()
        m.header.frame_id = frame_id
        m.header.stamp = stamp
        m.ns = ns
        m.id = int(marker_id)
        m.action = Marker.ADD
        m.lifetime.sec = 0  # persistent until overwritten
        m.lifetime.nanosec = 0
        return m

    def _publish_aabb_marker(self, obj_cloud: np.ndarray, marker_id: int, stamp, obj_id: str) -> None:
        pts = np.asarray(obj_cloud, dtype=np.float32)
        if pts.size == 0:
            return

        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        center = (mins + maxs) / 2.0
        size = (maxs - mins)

        m = self._make_marker_base(ns="aabb", marker_id=marker_id, stamp=stamp, frame_id="base")
        m.type = Marker.CUBE

        m.pose.position.x = float(center[0])
        m.pose.position.y = float(center[1])
        m.pose.position.z = float(center[2])
        # axis-aligned box -> identity orientation
        m.pose.orientation.w = 1.0

        # Avoid zero-size markers (Foxglove may hide them)
        eps = 1e-4
        m.scale.x = float(max(size[0], eps))
        m.scale.y = float(max(size[1], eps))
        m.scale.z = float(max(size[2], eps))

        # translucent green
        m.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.30)

        # helpful label (Foxglove may show it depending on settings)
        m.text = f"{obj_id}_aabb"

        self.pub_markers.publish(m)

    def _publish_fitted_cube_marker(self, T, marker_id: int, stamp, obj_id: str) -> None:
        """
        Publish a cube marker at the estimated object pose using known cube size.
        This represents the fitted CAD cube in the scene.
        """
        qx, qy, qz, qw = _mat3_to_quat_xyzw(T.R)

        m = self._make_marker_base(ns="cad", marker_id=marker_id, stamp=stamp, frame_id="base")
        m.type = Marker.CUBE

        m.pose.position.x = float(T.t[0])
        m.pose.position.y = float(T.t[1])
        m.pose.position.z = float(T.t[2])
        m.pose.orientation.x = qx
        m.pose.orientation.y = qy
        m.pose.orientation.z = qz
        m.pose.orientation.w = qw

        # use CAD extents as scale
        sx, sy, sz = self.cube_size_xyz
        m.scale.x = float(max(sx, 1e-4))
        m.scale.y = float(max(sy, 1e-4))
        m.scale.z = float(max(sz, 1e-4))

        # translucent red so it stands out from the AABB
        m.color = ColorRGBA(r=1.0, g=0.1, b=0.1, a=0.35)
        m.text = f"{obj_id}_cad"

        self.pub_markers.publish(m)

    def _tick(self) -> None:
        views = self.grabber.get_latest_views()
        if views is None:
            return

        self._frame += 1
        stamp = self.get_clock().now().to_msg()

        # publish RGB images
        for v in views:
            if v.rgb is None:
                continue
            self.pub_rgb[v.cam_id].publish(_rgb_numpy_to_imgmsg(v.rgb, frame_id=v.cam_id, stamp=stamp))
    
        #for v in views:
        #    pts_cam = depth_to_points_cam(v.depth, v.K, stride=2, zmin=0.25, zmax=1.6)
        #    pts_base = v.T_base_cam.transform_points(pts_cam)
        #    self.pub_cloud_cam[v.cam_id].publish(_points_to_pointcloud2(pts_base, frame_id="base", stamp=stamp))
        # run pipeline
        result = self.runner.run(views)

        # publish fused cloud (raw scene)
        if getattr(result, "points_world_raw", None) is not None:
            self.pub_cloud_fused.publish(
                _points_to_pointcloud2(result.points_world_raw, frame_id="base", stamp=stamp)
            )

        # publish poses + TF + markers
        objs = getattr(result, "objects", None) or []
        for idx, obj in enumerate(objs):
            obj_id = getattr(obj, "object_id", f"obj{idx}")

            # create pub on demand
            if obj_id not in self.pub_pose:
                self.pub_pose[obj_id] = self.create_publisher(PoseStamped, f"/perception/pose/{obj_id}", 1)

            # pipeline convention: OBJ -> BASE
            T = getattr(obj, "T_object_to_world", None)
            if T is None:
                continue

            # Pose topic
            qx, qy, qz, qw = _mat3_to_quat_xyzw(T.R)

            pose = PoseStamped()
            pose.header.frame_id = "base"
            pose.header.stamp = stamp
            pose.pose.position.x = float(T.t[0])
            pose.pose.position.y = float(T.t[1])
            pose.pose.position.z = float(T.t[2])
            pose.pose.orientation.x = qx
            pose.pose.orientation.y = qy
            pose.pose.orientation.z = qz
            pose.pose.orientation.w = qw
            self.pub_pose[obj_id].publish(pose)

            # TF
            tf = TransformStamped()
            tf.header.frame_id = "base"
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

            # AABB around the object's cluster point cloud (if available)
            cloud = getattr(obj, "point_cloud", None)
            if cloud is not None and len(cloud) > 0:
                self._publish_aabb_marker(cloud, marker_id=1000 + idx, stamp=stamp, obj_id=obj_id)

            # CAD cube marker at fitted pose (only meaningful for cube)
            if obj_id == "cube":
                self._publish_fitted_cube_marker(T, marker_id=2000 + idx, stamp=stamp, obj_id=obj_id)

        # occasional log
        if self._frame % 10 == 0:
            n_raw = 0 if getattr(result, "points_world_raw", None) is None else len(result.points_world_raw)
            self.get_logger().info(f"[frame {self._frame}] raw_pts={n_raw} objects={len(objs)}")


def main() -> None:
    rclpy.init()

    # Grabber node (callbacks are handled by executor)
    T_map = load_extrinsics_yaml("config/camera_extrinsics.yaml")
    grabber = MultiCamGrabber(
        cameras=CAMERAS,
        sync_slop_s=0.15,
        use_best_effort_if_unsynced=True,
        static_extrinsics_base_cam=T_map,
        rgb_depth_max_dt_s=0.10,
    )

    # Main processing node
    node = LiveMultiViewDebug(grabber)

    executor = MultiThreadedExecutor(num_threads=2)
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