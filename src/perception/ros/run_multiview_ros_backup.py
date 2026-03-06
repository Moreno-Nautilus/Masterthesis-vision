from __future__ import annotations

import time
from typing import Any, List, Optional, Tuple

import numpy as np
import rclpy

from src.perception.ros.multicam_grabber import MultiCamGrabber, CameraTopics
from src.perception.pipeline import GraspPerceptionPipeline, PipelineConfig
from src.perception.pipeline_multiview import MultiViewRunner, MultiViewConfig
from src.perception.pose_icp import load_cad_as_pointcloud
from src.perception.viz_plotly import save_scene_plotly_html
from src.calibration.io_extrinsics import load_extrinsics_yaml

from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header, ColorRGBA
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import struct

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


def _pose_candidates_in_priority() -> List[str]:
    return [
        "T_object_to_world",  # preferred in our pipeline (OBJ -> BASE)
        "T_base_obj",
        "T_world_obj",
        "T_map_obj",
        "world_T_obj",
        "map_T_obj",
        "base_T_obj",
        "T_w_o",
        "T_m_o",
        "T_b_o",
        "pose_world",
        "pose_in_world",
        "pose_base",
        "pose_in_base",
        "T",
        "pose",
    ]


def _to_matrix4x4(T: Any) -> np.ndarray:
    if T is None:
        raise ValueError("T is None")
    if hasattr(T, "as_matrix"):
        M = np.asarray(T.as_matrix())
    elif hasattr(T, "matrix"):
        M = np.asarray(T.matrix())
    else:
        M = np.asarray(T)
    if M.shape != (4, 4):
        raise ValueError(f"Expected 4x4 transform, got shape={M.shape} type={type(T)}")
    return M


def _transform_points(T: Any, pts_xyz: np.ndarray) -> np.ndarray:
    if pts_xyz.size == 0:
        return pts_xyz
    if hasattr(T, "transform_points"):
        return np.asarray(T.transform_points(pts_xyz))
    M = _to_matrix4x4(T)
    pts_h = np.hstack([pts_xyz, np.ones((pts_xyz.shape[0], 1), dtype=pts_xyz.dtype)])
    return (M @ pts_h.T).T[:, :3]


def _pick_pose_field(o: Any) -> Tuple[Optional[str], Optional[Any]]:
    for name in _pose_candidates_in_priority():
        if hasattr(o, name):
            return name, getattr(o, name)
    return None, None

def _points_to_pointcloud2(points_xyz: np.ndarray, frame_id: str, stamp) -> PointCloud2:
    pts = np.asarray(points_xyz, dtype=np.float32)
    msg = PointCloud2()
    msg.header = Header(frame_id=frame_id, stamp=stamp)
    msg.height = 1
    msg.width = int(pts.shape[0])
    msg.is_bigendian = False
    msg.is_dense = True

    # fields: x,y,z float32
    from sensor_msgs.msg import PointField
    msg.fields = [
        PointField(name="x", offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8,  datatype=PointField.FLOAT32, count=1),
    ]
    msg.point_step = 12
    msg.row_step = msg.point_step * msg.width
    msg.data = pts.tobytes()
    return msg


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

def _line(a: np.ndarray, b: np.ndarray, n: int = 60) -> np.ndarray:
    a = np.asarray(a, dtype=float).reshape(3,)
    b = np.asarray(b, dtype=float).reshape(3,)
    ts = np.linspace(0.0, 1.0, n)
    return (1 - ts)[:, None] * a[None, :] + ts[:, None] * b[None, :]


def _base_axes(scale: float = 0.35, n: int = 60) -> list[np.ndarray]:
    o = np.zeros(3)
    return [
        _line(o, np.array([scale, 0, 0]), n),
        _line(o, np.array([0, scale, 0]), n),
        _line(o, np.array([0, 0, scale]), n),
    ]

def _mat3_to_quat_xyzw(R: np.ndarray) -> tuple[float, float, float, float]:
    # robust enough for normal rotations
    m = np.asarray(R, dtype=float)
    tr = m[0,0] + m[1,1] + m[2,2]
    if tr > 0.0:
        S = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (m[2,1] - m[1,2]) / S
        qy = (m[0,2] - m[2,0]) / S
        qz = (m[1,0] - m[0,1]) / S
    elif (m[0,0] > m[1,1]) and (m[0,0] > m[2,2]):
        S = np.sqrt(1.0 + m[0,0] - m[1,1] - m[2,2]) * 2.0
        qw = (m[2,1] - m[1,2]) / S
        qx = 0.25 * S
        qy = (m[0,1] + m[1,0]) / S
        qz = (m[0,2] + m[2,0]) / S
    elif m[1,1] > m[2,2]:
        S = np.sqrt(1.0 + m[1,1] - m[0,0] - m[2,2]) * 2.0
        qw = (m[0,2] - m[2,0]) / S
        qx = (m[0,1] + m[1,0]) / S
        qy = 0.25 * S
        qz = (m[1,2] + m[2,1]) / S
    else:
        S = np.sqrt(1.0 + m[2,2] - m[0,0] - m[1,1]) * 2.0
        qw = (m[1,0] - m[0,1]) / S
        qx = (m[0,2] + m[2,0]) / S
        qy = (m[1,2] + m[2,1]) / S
        qz = 0.25 * S
    return (float(qx), float(qy), float(qz), float(qw))
    
def _frame_lines(T: Any, scale: float = 0.20, n: int = 60) -> list[np.ndarray]:
    M = _to_matrix4x4(T)
    R = M[:3, :3]
    o = M[:3, 3]
    x = o + scale * R[:, 0]
    y = o + scale * R[:, 1]
    z = o + scale * R[:, 2]
    return [_line(o, x, n), _line(o, y, n), _line(o, z, n)]


def _plane_normal_line(nrm: np.ndarray, p0: np.ndarray, scale: float = 0.25, n: int = 80) -> np.ndarray:
    nrm = np.asarray(nrm, dtype=float).reshape(3,)
    p0 = np.asarray(p0, dtype=float).reshape(3,)
    p1 = p0 + scale * nrm
    return _line(p0, p1, n)


def main() -> None:
    rclpy.init()

    cad_library = {
        "cube": load_cad_as_pointcloud("Data/CAD_Models/Cube.stl", scale=0.003822, center=True),
    }

    cad = cad_library["cube"]
    cad_ext = cad.max(axis=0) - cad.min(axis=0)
    cad_diag = float(np.linalg.norm(cad_ext))
    print("[CAD] cube extent [m]:", cad_ext, "diag:", cad_diag)

    pipe_cfg = PipelineConfig(
        plane_distance_threshold=0.002,
        dbscan_eps=0.03,
        dbscan_min_points=40,
        voxel_size=0.005,
        max_rms_nn=0.012,
        min_margin=1.5,
    )
    pipe = GraspPerceptionPipeline(cad_library=cad_library, cfg=pipe_cfg)

    mv_cfg = MultiViewConfig(
        voxel_size_fusion=0.003,
        stride=1,
        zmin=0.25,
        zmax=1.6,
        roi_x_min=-0.7,
        roi_x_max=0.7,
        roi_y_min=-0.7,
        roi_y_max=0.7,
        roi_z_min=0.25,
        roi_z_max=1.6,
    )
    runner = MultiViewRunner(pipe, cfg=mv_cfg)

    T_map = load_extrinsics_yaml("config/camera_extrinsics.yaml")
    grabber = MultiCamGrabber(
        cameras=CAMERAS,
        sync_slop_s=0.05,
        use_best_effort_if_unsynced=False,
        static_extrinsics_base_cam=T_map,
    )

    try:
        print("Waiting for synced multi-view set...")
        views = None
        t0 = time.time()
        while views is None:
            rclpy.spin_once(grabber, timeout_sec=0.1)
            views = grabber.get_latest_views()
            if time.time() - t0 > 10.0 and views is None:
                print("Still waiting... (check camera topics / running nodes)")
                t0 = time.time()

        print("Got synced views:", [(v.cam_id, f"{v.stamp_s:.3f}") for v in views])

        # debug geometry (axes + camera frames)
        debug_clusters: List[np.ndarray] = []
        debug_clusters += _base_axes(scale=0.35)

        cam_origins = []
        for v in views:
            debug_clusters += _frame_lines(v.T_base_cam, scale=0.20)
            cam_origins.append(v.T_base_cam.transform_point(np.zeros(3)))
        if cam_origins:
            debug_clusters.append(np.asarray(cam_origins, dtype=float))

        # Run full pipeline
        result = runner.run(views)
        raw = result.points_world_raw

        print("[RAW] n=", len(raw))
        mins = raw.min(axis=0)
        maxs = raw.max(axis=0)
        ext = maxs - mins
        print("[RAW] mins:", mins, "maxs:", maxs)
        print("[RAW] extent XYZ [m]:", ext)

        # --- Choose plane normal sign by maximizing above-band count ---
        n0 = np.array(result.plane_model[:3], dtype=float)
        d0 = float(result.plane_model[3])
        nn = np.linalg.norm(n0) + 1e-12
        n0 = n0 / nn
        d0 = d0 / nn

        h0 = raw @ n0 + d0
        h1 = -h0

        h_min, h_max = 0.01, 0.12
        n_above0 = int(np.count_nonzero((h0 > h_min) & (h0 < h_max)))
        n_above1 = int(np.count_nonzero((h1 > h_min) & (h1 < h_max)))

        if n_above1 > n_above0:
            n = -n0
            d = -d0
            h = h1
        else:
            n = n0
            d = d0
            h = h0

        print(
            f"[UP_SIGN] above_count normal={n_above0} flipped={n_above1} -> using "
            f"{'flipped' if n_above1 > n_above0 else 'normal'}"
        )
        print("[H] min/med/max:", float(h.min()), float(np.median(h)), float(h.max()))
        print("[H] quantiles 50/90/95/99%:", np.quantile(h, [0.5, 0.9, 0.95, 0.99]))

        # Plane centroid marker + normal arrow
        plane_centroid = (
            np.mean(result.plane_points, axis=0)
            if result.plane_points is not None and len(result.plane_points)
            else np.zeros(3)
        )
        debug_clusters.append(np.asarray([plane_centroid], dtype=float))
        debug_clusters.append(_plane_normal_line(n, plane_centroid, scale=0.25))

        # Above-band points (for quick debug)
        mask = (h > h_min) & (h < h_max)
        above = raw[mask]
        print("[ABOVE] n=", len(above))

        out_above = "/home/moreno/MasterThesis/multiview_above_plane_debug.html"
        save_scene_plotly_html(
            scene=raw,
            plane=result.plane_points,
            clusters=debug_clusters + [above],
            objects_world=None,
            out_path=out_above,
        )
        print(f"Saved {out_above}")

        # --- Workspace plot: ALWAYS use pipeline pose (no manual ICP pick!) ---
        objects_world_ws: List[Tuple[str, np.ndarray]] = []
        if getattr(result, "objects", None):
            for i, o in enumerate(result.objects):
                obj_id = getattr(o, "object_id", f"obj{i}")
                if obj_id not in cad_library:
                    continue

                pose_name, T = _pick_pose_field(o)
                if T is None:
                    raise AttributeError(f"DetectedObject has no recognized pose field. attrs={dir(o)}")

                M = _to_matrix4x4(T)
                pts_fit = _transform_points(M, cad_library[obj_id])
                objects_world_ws.append((f"{obj_id}_pipeline", pts_fit))

                t = M[:3, 3]
                c = o.point_cloud.mean(axis=0) if hasattr(o, "point_cloud") else np.full(3, np.nan)
                print(
                    f"[PIPE_POSE] {obj_id} via {pose_name}: "
                    f"t={t} |t|={np.linalg.norm(t):.3f} "
                    f"cluster_centroid={c} diff={np.linalg.norm(t-c):.3f}"
                )

        out_ws = "/home/moreno/MasterThesis/multiview_workspace_with_icp.html"
        save_scene_plotly_html(
            scene=result.points_world_raw,
            plane=result.plane_points,
            clusters=debug_clusters + (result.clusters or []),
            objects_world=objects_world_ws,
            out_path=out_ws,
        )
        print(f"Saved workspace (pipeline pose): {out_ws}")

        # Detection plot (same as workspace, but you can keep both)
        out_det = "/home/moreno/MasterThesis/multiview_detection_debug.html"
        save_scene_plotly_html(
            scene=result.points_world_raw,
            plane=result.plane_points,
            clusters=debug_clusters + (result.clusters or []),
            objects_world=objects_world_ws,
            out_path=out_det,
        )
        print(f"Saved detection debug: {out_det}")

        if getattr(result, "objects", None):
            print("detections:", [(o.object_id, o.id_confidence, o.pose_confidence) for o in result.objects])

    finally:
        grabber.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()