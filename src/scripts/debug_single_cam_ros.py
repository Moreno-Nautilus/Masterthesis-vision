from __future__ import annotations

import time
import numpy as np
import open3d as o3d

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo

from src.perception.backproject import depth_to_points_cam


def _stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def _K_from_camerainfo(msg: CameraInfo) -> np.ndarray:
    return np.array(msg.k, dtype=float).reshape(3, 3)


def _img_to_numpy(msg: Image) -> np.ndarray:
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

    step_elems = int(msg.step // elem_size)
    if step_elems == w:
        return data.reshape(h, w)

    out = np.zeros((h, w), dtype=dtype)
    for r in range(h):
        out[r, :] = data[r * step_elems : r * step_elems + w]
    return out


def _depth_to_meters(depth: np.ndarray, encoding: str) -> np.ndarray:
    if encoding == "32FC1":
        return depth.astype(np.float32)
    if encoding == "16UC1":
        return depth.astype(np.float32) * 1e-3  # mm -> m
    raise ValueError(f"Unsupported depth encoding: {encoding}")


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if len(points) == 0:
        return points
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pcd = pcd.voxel_down_sample(voxel_size)
    return np.asarray(pcd.points)


class SingleCamDebug(Node):
    def __init__(
        self,
        cam_id: str,
        depth_topic: str,
        info_topic: str,
        stride: int = 2,
        zmin: float = 0.15,
        zmax: float = 2.0,
        voxel_size: float = 0.005,
        out_path: str | None = None,
    ):
        super().__init__("single_cam_debug")

        self.cam_id = cam_id
        self.depth_topic = depth_topic
        self.info_topic = info_topic

        self.stride = int(stride)
        self.zmin = float(zmin)
        self.zmax = float(zmax)
        self.voxel_size = float(voxel_size)

        self.out_path = out_path or f"/tmp/{cam_id}_cloud.ply"

        self._K: np.ndarray | None = None
        self._depth_msg: Image | None = None
        self._depth_stamp_s: float | None = None

        self.get_logger().info(f"Subscribing cam_id={cam_id}")
        self.get_logger().info(f"  depth: {depth_topic}")
        self.get_logger().info(f"  info : {info_topic}")

        self.create_subscription(CameraInfo, info_topic, self._on_info, 10)
        self.create_subscription(Image, depth_topic, self._on_depth, 10)

    def _on_info(self, msg: CameraInfo) -> None:
        self._K = _K_from_camerainfo(msg)

    def _on_depth(self, msg: Image) -> None:
        self._depth_msg = msg
        self._depth_stamp_s = _stamp_to_sec(msg.header.stamp)

    def ready(self) -> bool:
        return (self._K is not None) and (self._depth_msg is not None)

    def dump_once(self) -> bool:
        if not self.ready():
            return False

        assert self._K is not None
        assert self._depth_msg is not None

        depth_raw = _img_to_numpy(self._depth_msg)
        depth_m = _depth_to_meters(depth_raw, self._depth_msg.encoding)

        finite = np.isfinite(depth_m)
        if not finite.any():
            self.get_logger().warn("Depth is all NaN/Inf; not writing PLY.")
            return False

        dmin = float(depth_m[finite].min())
        dmed = float(np.median(depth_m[finite]))
        dmax = float(depth_m[finite].max())
        self.get_logger().info(
            f"Depth stats [m]: min={dmin:.3f} med={dmed:.3f} max={dmax:.3f} "
            f"encoding={self._depth_msg.encoding} stamp={self._depth_stamp_s:.3f}"
        )

        pts_cam = depth_to_points_cam(
            depth_m=depth_m,
            K=self._K,
            stride=self.stride,
            zmin=self.zmin,
            zmax=self.zmax,
        )
        self.get_logger().info(f"Backprojected points: {len(pts_cam)}")

        pts_cam = voxel_downsample(pts_cam, self.voxel_size)
        self.get_logger().info(f"After voxel downsample: {len(pts_cam)} (voxel={self.voxel_size})")

        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_cam.astype(np.float64)))
        ok = o3d.io.write_point_cloud(self.out_path, pcd)
        self.get_logger().info(f"Wrote PLY: {self.out_path} ok={ok}")
        return True


def main() -> None:
    # ---- EDIT THESE TOMORROW for the camera you want to debug ----
    CAM_ID = "zed2i_1"
    DEPTH_TOPIC = "/zed2i_1/zed_node/depth/depth_registered"
    INFO_TOPIC = "/zed2i_1/zed_node/left/camera_info"
    # -------------------------------------------------------------

    rclpy.init()
    node = SingleCamDebug(
        cam_id=CAM_ID,
        depth_topic=DEPTH_TOPIC,
        info_topic=INFO_TOPIC,
        stride=2,
        zmin=0.15,
        zmax=2.0,
        voxel_size=0.005,
        out_path=f"/tmp/{CAM_ID}_cloud.ply",
    )

    try:
        print("Waiting for one depth+camera_info pair...")
        t0 = time.time()
        while rclpy.ok() and not node.ready():
            rclpy.spin_once(node, timeout_sec=0.1)
            if time.time() - t0 > 5.0:
                print("Still waiting... check that the camera node is running and topics are correct.")
                t0 = time.time()

        # once ready, dump once and exit
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            if node.dump_once():
                break

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()