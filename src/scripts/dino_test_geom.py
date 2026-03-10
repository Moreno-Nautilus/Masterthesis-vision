from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import open3d as o3d
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Header

from src.calibration.io_extrinsics import load_extrinsics_yaml
from src.perception.learned.DINO.dino_identifier import DINOIdentifier, DINOIdentifierConfig
from src.perception.classical.pipeline_fast import GraspPerceptionPipeline, PipelineConfig
from src.perception.classical.fusion_fast import fuse_views_to_points_base_with_colors, RGBMaskConfig
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

FAST_QOS = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
)


@dataclass
class GeometryDINOConfig:
    voxel_size_fusion: float = 0.005
    stride: int = 4

    zmin: float = 0.30
    zmax: float = 1.05

    rgb_mask: RGBMaskConfig = RGBMaskConfig(
        mode="chroma",
        min_chroma=15,
        min_v_chroma=25,
    )

    roi_x_min: float = -0.15
    roi_x_max: float = 0.30
    roi_y_min: float = -0.15
    roi_y_max: float = 0.30
    roi_z_min: float = 0.30
    roi_z_max: float = 1.05

    max_points_after_roi: int = 22000

    voxel_size_pipe: float = 0.005
    plane_distance_threshold: float = 0.004
    plane_ransac_n: int = 3
    plane_num_iterations: int = 400
    dbscan_eps: float = 0.035
    dbscan_min_points: int = 25
    max_clusters_considered: int = 6
    max_points_plane: int = 15000
    max_points_cluster: int = 6000

    max_clusters_for_dino: int = 6
    proposal_pad_px: int = 12
    min_bbox_side_px: int = 18
    min_projected_points: int = 30
    min_crop_side_after_resize: int = 96

    dino_score_threshold: float = 0.60
    dino_margin_threshold: float = 0.03

    choose_camera_by: str = "visible_points"  # or "bbox_area"


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


def _subsample_points(points: np.ndarray, max_pts: int) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if pts.shape[0] <= max_pts:
        return pts
    stride = max(1, int(np.ceil(pts.shape[0] / max_pts)))
    return pts[::stride]


def _cap_points_with_optional_colors(
    pts: np.ndarray,
    cols: Optional[np.ndarray],
    max_pts: int,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    if pts.shape[0] <= max_pts:
        return pts, cols
    stride = max(1, int(np.ceil(pts.shape[0] / max_pts)))
    pts = pts[::stride]
    if cols is not None:
        cols = cols[::stride]
    return pts, cols


def _mat_from_transform(T: Any) -> np.ndarray:
    if hasattr(T, "as_matrix"):
        M = np.asarray(T.as_matrix(), dtype=np.float64)
        if M.shape == (4, 4):
            return M

    if hasattr(T, "matrix"):
        M = np.asarray(T.matrix, dtype=np.float64)
        if M.shape == (4, 4):
            return M

    if hasattr(T, "R") and hasattr(T, "t"):
        R = np.asarray(T.R, dtype=np.float64).reshape(3, 3)
        t = np.asarray(T.t, dtype=np.float64).reshape(3)
        M = np.eye(4, dtype=np.float64)
        M[:3, :3] = R
        M[:3, 3] = t
        return M

    raise ValueError("Unsupported transform object for T_base_cam")


def _world_to_cam(points_world: np.ndarray, T_base_cam: Any) -> np.ndarray:
    pts = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)

    T_bc = _mat_from_transform(T_base_cam)
    T_cb = np.linalg.inv(T_bc)

    pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1)
    pts_cam_h = (T_cb @ pts_h.T).T
    return pts_cam_h[:, :3].astype(np.float32)


def _project_cam_points_to_image(points_cam: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points_cam, dtype=np.float32).reshape(-1, 3)
    if pts.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=bool)

    z = pts[:, 2]
    valid = np.isfinite(z) & (z > 1e-6)
    if not np.any(valid):
        return np.zeros((0, 2), dtype=np.float32), np.zeros((pts.shape[0],), dtype=bool)

    ptsv = pts[valid]
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    u = fx * (ptsv[:, 0] / ptsv[:, 2]) + cx
    v = fy * (ptsv[:, 1] / ptsv[:, 2]) + cy
    uv = np.stack([u, v], axis=1).astype(np.float32)

    return uv, valid


def _clip_bbox_xyxy(
    bbox_xyxy: Tuple[int, int, int, int],
    h: int,
    w: int,
) -> Optional[Tuple[int, int, int, int]]:
    x0, y0, x1, y1 = bbox_xyxy
    x0 = max(0, min(x0, w - 1))
    y0 = max(0, min(y0, h - 1))
    x1 = max(x0 + 1, min(x1, w))
    y1 = max(y0 + 1, min(y1, h))
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _bbox_from_uv(
    uv: np.ndarray,
    h: int,
    w: int,
    pad: int,
) -> Optional[Tuple[int, int, int, int]]:
    if uv.shape[0] == 0:
        return None

    x0 = int(np.floor(np.min(uv[:, 0]))) - pad
    y0 = int(np.floor(np.min(uv[:, 1]))) - pad
    x1 = int(np.ceil(np.max(uv[:, 0]))) + pad + 1
    y1 = int(np.ceil(np.max(uv[:, 1]))) + pad + 1
    return _clip_bbox_xyxy((x0, y0, x1, y1), h, w)


def _make_mask_from_projected_points(
    uv: np.ndarray,
    h: int,
    w: int,
    dilate_px: int = 7,
) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    if uv.shape[0] == 0:
        return mask.astype(bool)

    pts = np.round(uv).astype(np.int32)
    inside = (
        (pts[:, 0] >= 0) & (pts[:, 0] < w) &
        (pts[:, 1] >= 0) & (pts[:, 1] < h)
    )
    pts = pts[inside]
    if pts.shape[0] == 0:
        return mask.astype(bool)

    if pts.shape[0] >= 3:
        hull = cv2.convexHull(pts.reshape(-1, 1, 2))
        cv2.fillConvexPoly(mask, hull, 255)
    else:
        for p in pts:
            mask[p[1], p[0]] = 255

    if dilate_px > 0:
        k = 2 * dilate_px + 1
        kernel = np.ones((k, k), dtype=np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

    return mask > 0


def _masked_tight_crop(
    rgb: np.ndarray,
    mask: np.ndarray,
    bbox_xyxy: Tuple[int, int, int, int],
    min_side_after_resize: int = 96,
) -> Optional[np.ndarray]:
    x0, y0, x1, y1 = bbox_xyxy
    crop = rgb[y0:y1, x0:x1].copy()
    crop_mask = mask[y0:y1, x0:x1]
    if crop.size == 0:
        return None

    if crop_mask.shape[:2] == crop.shape[:2] and np.any(crop_mask):
        out = np.zeros_like(crop)
        out[crop_mask] = crop[crop_mask]
    else:
        out = crop

    h, w = out.shape[:2]
    min_side = min(h, w)
    if min_side > 0 and min_side < min_side_after_resize:
        scale = float(min_side_after_resize) / float(min_side)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        out = cv2.resize(out, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    return np.ascontiguousarray(out)


def _draw_mask_overlay(
    rgb: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int],
    alpha: float = 0.28,
) -> np.ndarray:
    out = rgb.copy()
    color_arr = np.array(color, dtype=np.uint8).reshape(1, 1, 3)
    mask3 = mask.astype(bool)[..., None]
    blended = ((1.0 - alpha) * out + alpha * color_arr).astype(np.uint8)
    return np.where(mask3, blended, out)


class GeometryDINODebugNode(Node):
    def __init__(self, grabber: MultiCamGrabber, reference_dir: str):
        super().__init__("geometry_dino_debug")
        self.grabber = grabber
        self.cfg = GeometryDINOConfig()

        self.pub_overlay: Dict[str, Any] = {}
        self.pub_raw: Dict[str, Any] = {}
        self.pub_geom: Dict[str, Any] = {}

        for cam in CAMERAS:
            self.pub_overlay[cam.cam_id] = self.create_publisher(
                Image,
                f"/perception/debug/{cam.cam_id}/dino_overlay",
                FAST_QOS,
            )
            self.pub_raw[cam.cam_id] = self.create_publisher(
                Image,
                f"/perception/debug/{cam.cam_id}/dino_raw",
                FAST_QOS,
            )
            self.pub_geom[cam.cam_id] = self.create_publisher(
                Image,
                f"/perception/debug/{cam.cam_id}/geometry_proposals",
                FAST_QOS,
            )

        self.palette = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
            (255, 128, 0),
            (128, 0, 255),
        ]

        self.dino = DINOIdentifier(
            DINOIdentifierConfig(
                model_name="dinov2_vitb14",
                reference_dir=reference_dir,
            )
        )
        self.get_logger().info("Building DINO reference bank...")
        self.dino.build_reference_bank_from_folder()
        self.get_logger().info(f"Reference bank size: {len(self.dino.reference_bank)}")

        pipe_cfg = PipelineConfig(
            voxel_size=self.cfg.voxel_size_pipe,
            plane_distance_threshold=self.cfg.plane_distance_threshold,
            plane_ransac_n=self.cfg.plane_ransac_n,
            plane_num_iterations=self.cfg.plane_num_iterations,
            dbscan_eps=self.cfg.dbscan_eps,
            dbscan_min_points=self.cfg.dbscan_min_points,
            max_clusters_considered=self.cfg.max_clusters_considered,
            max_points_plane=self.cfg.max_points_plane,
            max_points_cluster=self.cfg.max_points_cluster,
            expected_num_objects=None,
            max_objects_returned=self.cfg.max_clusters_for_dino,
        )
        self.pipe = GraspPerceptionPipeline(cad_library={}, cfg=pipe_cfg)

        self._busy = False
        self._last_views_signature = None
        self.timer = self.create_timer(0.25, self._tick)

        self.get_logger().info(
            f"Geometry+DINO runner started | stride={self.cfg.stride} "
            f"voxel_fusion={self.cfg.voxel_size_fusion:.3f} "
            f"dbscan_eps={self.cfg.dbscan_eps:.3f}"
        )

    def _classify_single_crop(self, rgb_crop: np.ndarray):
        res = self.dino.classify_crop(rgb_crop)
        scores = res.scores_by_object
        sorted_scores = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

        best_obj, best_score = sorted_scores[0]
        second_score = sorted_scores[1][1] if len(sorted_scores) > 1 else -1.0
        margin = best_score - second_score

        if best_score < self.cfg.dino_score_threshold or margin < self.cfg.dino_margin_threshold:
            label = "unknown"
        else:
            label = best_obj

        return label, best_score, scores

    def _views_signature(self, views: Any) -> Optional[Tuple[Tuple[str, int], ...]]:
        signature = []
        for v in views:
            stamp_ns = _try_get_view_stamp_ns(v)
            if stamp_ns is None:
                return None
            signature.append((str(v.cam_id), int(stamp_ns)))
        signature.sort(key=lambda x: x[0])
        return tuple(signature)

    def _publish_pair(self, cam_id: str, rgb: np.ndarray, overlay: np.ndarray) -> None:
        stamp = self.get_clock().now().to_msg()
        self.pub_raw[cam_id].publish(
            _rgb_numpy_to_imgmsg(rgb, frame_id=cam_id, stamp=stamp)
        )
        self.pub_overlay[cam_id].publish(
            _rgb_numpy_to_imgmsg(overlay, frame_id=cam_id, stamp=stamp)
        )

    def _publish_geom(self, cam_id: str, geom_vis: np.ndarray) -> None:
        stamp = self.get_clock().now().to_msg()
        self.pub_geom[cam_id].publish(
            _rgb_numpy_to_imgmsg(geom_vis, frame_id=cam_id, stamp=stamp)
        )

    def _fuse_points(self, views: List[Any]) -> tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
        t0 = time.perf_counter()

        debug_pts_base_raw, _ = fuse_views_to_points_base_with_colors(
            views,
            voxel_size=self.cfg.voxel_size_fusion,
            stride=self.cfg.stride,
            zmin=self.cfg.zmin,
            zmax=self.cfg.zmax,
            rgb_mask_cfg=RGBMaskConfig(mode="none"),
        )
        debug_pts_base_raw = np.asarray(debug_pts_base_raw, dtype=np.float32).reshape(-1, 3)

        pts_base_raw, cols_rgb = fuse_views_to_points_base_with_colors(
            views,
            voxel_size=self.cfg.voxel_size_fusion,
            stride=self.cfg.stride,
            zmin=self.cfg.zmin,
            zmax=self.cfg.zmax,
            rgb_mask_cfg=self.cfg.rgb_mask,
        )
        pts_base_raw = np.asarray(pts_base_raw, dtype=np.float32).reshape(-1, 3)

        if pts_base_raw.size == 0:
            self.get_logger().info(
                f"[TIMING geometry_dino] fusion={(time.perf_counter() - t0) * 1000:.1f} ms raw_pts=0"
            )
            return pts_base_raw, cols_rgb, debug_pts_base_raw

        m = (
            (pts_base_raw[:, 0] > self.cfg.roi_x_min)
            & (pts_base_raw[:, 0] < self.cfg.roi_x_max)
            & (pts_base_raw[:, 1] > self.cfg.roi_y_min)
            & (pts_base_raw[:, 1] < self.cfg.roi_y_max)
            & (pts_base_raw[:, 2] > self.cfg.roi_z_min)
            & (pts_base_raw[:, 2] < self.cfg.roi_z_max)
        )

        pts_base_roi = pts_base_raw[m]
        if cols_rgb is not None:
            cols_rgb = cols_rgb[m]

        pts_base_roi, cols_rgb = _cap_points_with_optional_colors(
            pts_base_roi,
            cols_rgb,
            self.cfg.max_points_after_roi,
        )

        self.get_logger().info(
            f"[TIMING geometry_dino] fusion+roi={(time.perf_counter() - t0) * 1000:.1f} ms "
            f"raw_pts={pts_base_raw.shape[0]} roi_pts={pts_base_roi.shape[0]}"
        )
        return pts_base_roi, cols_rgb, debug_pts_base_raw

    def _cluster_points(self, points_world: np.ndarray) -> list[np.ndarray]:
        pts = np.asarray(points_world, dtype=np.float32).reshape(-1, 3)
        if pts.size == 0:
            return []

        t0 = time.perf_counter()

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        pcd = pcd.voxel_down_sample(self.cfg.voxel_size_pipe)
        pts_voxel = np.asarray(pcd.points, dtype=np.float32)

        t1 = time.perf_counter()
        pts_noplane = self.pipe._remove_plane(pts_voxel)
        t2 = time.perf_counter()
        clusters = self.pipe._cluster(pts_noplane)
        t3 = time.perf_counter()

        self.get_logger().info(
            f"[TIMING geometry_dino] voxel={(t1 - t0) * 1000:.1f} ms "
            f"plane={(t2 - t1) * 1000:.1f} ms "
            f"cluster={(t3 - t2) * 1000:.1f} ms "
            f"pts_in={pts.shape[0]} pts_after={pts_noplane.shape[0]} clusters={len(clusters)}"
        )
        return clusters[: self.cfg.max_clusters_for_dino]

    def _best_camera_proposal(
        self,
        cluster_world: np.ndarray,
        views_by_cam: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        best = None

        for cam_id, v in views_by_cam.items():
            rgb = getattr(v, "rgb", None)
            if rgb is None:
                continue

            h, w = rgb.shape[:2]
            pts_cam = _world_to_cam(cluster_world, v.T_base_cam)
            uv, valid_z = _project_cam_points_to_image(pts_cam, v.K)

            if uv.shape[0] == 0:
                continue

            inside = (
                (uv[:, 0] >= 0) & (uv[:, 0] < w) &
                (uv[:, 1] >= 0) & (uv[:, 1] < h)
            )
            uv_in = uv[inside]
            if uv_in.shape[0] < self.cfg.min_projected_points:
                continue

            bbox = _bbox_from_uv(
                uv_in,
                h=h,
                w=w,
                pad=self.cfg.proposal_pad_px,
            )
            if bbox is None:
                continue

            x0, y0, x1, y1 = bbox
            if (x1 - x0) < self.cfg.min_bbox_side_px or (y1 - y0) < self.cfg.min_bbox_side_px:
                continue

            mask = _make_mask_from_projected_points(uv_in, h=h, w=w, dilate_px=7)

            metric = uv_in.shape[0]
            if self.cfg.choose_camera_by == "bbox_area":
                metric = (x1 - x0) * (y1 - y0)

            cand = {
                "cam_id": cam_id,
                "view": v,
                "bbox_xyxy": bbox,
                "mask": mask,
                "visible_points": int(uv_in.shape[0]),
                "metric": float(metric),
            }

            if best is None or cand["metric"] > best["metric"]:
                best = cand

        return best

    def _draw_cluster_overlay(
        self,
        rgb: np.ndarray,
        proposals: List[Dict[str, Any]],
        cam_id: str,
    ) -> np.ndarray:
        vis = rgb.copy()
        for i, p in enumerate(proposals):
            if p["cam_id"] != cam_id:
                continue

            color = self.palette[i % len(self.palette)]
            x0, y0, x1, y1 = p["bbox_xyxy"]

            vis = _draw_mask_overlay(vis, p["mask"], color, alpha=0.22)
            cv2.rectangle(vis, (x0, y0), (x1, y1), color, 2)

            txt = f'geom_{i} vp={p["visible_points"]}'
            cv2.putText(
                vis,
                txt,
                (x0, max(20, y0 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )
        return vis

    def _draw_dino_overlay(
        self,
        rgb: np.ndarray,
        preds: List[Dict[str, Any]],
        cam_id: str,
    ) -> np.ndarray:
        vis = rgb.copy()
        for i, pred in enumerate(preds):
            if pred["cam_id"] != cam_id:
                continue

            color = self.palette[i % len(self.palette)]
            x0, y0, x1, y1 = pred["bbox_xyxy"]

            vis = _draw_mask_overlay(vis, pred["mask"], color, alpha=0.22)
            cv2.rectangle(vis, (x0, y0), (x1, y1), color, 2)

            txt = f'{pred["obj_id"]} {pred["score"]:.2f}'
            cv2.putText(
                vis,
                txt,
                (x0, max(20, y0 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
                cv2.LINE_AA,
            )
        return vis

    def _predict_from_geometry(
        self,
        views_by_cam: Dict[str, Any],
        clusters: List[np.ndarray],
    ) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], List[Dict[str, Any]]]:
        geom_vis_by_cam = {}
        overlay_by_cam = {}
        preds: List[Dict[str, Any]] = []

        for cam_id, v in views_by_cam.items():
            rgb = getattr(v, "rgb", None)
            if rgb is None:
                continue
            geom_vis_by_cam[cam_id] = rgb.copy()
            overlay_by_cam[cam_id] = rgb.copy()

        proposals = []
        for cluster in clusters:
            p = self._best_camera_proposal(cluster, views_by_cam)
            if p is not None:
                proposals.append(p)

        for cam_id, v in views_by_cam.items():
            rgb = getattr(v, "rgb", None)
            if rgb is None:
                continue
            geom_vis_by_cam[cam_id] = self._draw_cluster_overlay(rgb, proposals, cam_id)

        t0 = time.perf_counter()
        for i, p in enumerate(proposals):
            rgb = p["view"].rgb
            crop = _masked_tight_crop(
                rgb,
                p["mask"],
                p["bbox_xyxy"],
                min_side_after_resize=self.cfg.min_crop_side_after_resize,
            )
            if crop is None or crop.size == 0:
                continue

            try:
                obj_id, score, scores = self._classify_single_crop(crop)
            except Exception as e:
                self.get_logger().warn(f"DINO classify failed on proposal {i}: {e}")
                continue

            preds.append(
                {
                    "cam_id": p["cam_id"],
                    "bbox_xyxy": p["bbox_xyxy"],
                    "mask": p["mask"],
                    "obj_id": obj_id,
                    "score": score,
                    "scores": scores,
                    "visible_points": p["visible_points"],
                }
            )
        t1 = time.perf_counter()

        self.get_logger().info(
            f"[TIMING geometry_dino] dino={(t1 - t0) * 1000:.1f} ms "
            f"proposals={len(proposals)} preds={len(preds)}"
        )

        for cam_id, v in views_by_cam.items():
            rgb = getattr(v, "rgb", None)
            if rgb is None:
                continue
            overlay_by_cam[cam_id] = self._draw_dino_overlay(rgb, preds, cam_id)

        return geom_vis_by_cam, overlay_by_cam, preds

    def _tick(self) -> None:
        if self._busy:
            return
        self._busy = True

        try:
            views = self.grabber.get_latest_views()
            if views is None:
                return

            signature = self._views_signature(views)
            if signature is not None and signature == self._last_views_signature:
                return
            self._last_views_signature = signature

            views_by_cam = {str(v.cam_id): v for v in views if getattr(v, "rgb", None) is not None}

            t0 = time.perf_counter()
            pts_world, _cols, _debug_pts = self._fuse_points(views)
            t1 = time.perf_counter()

            clusters = self._cluster_points(pts_world)
            t2 = time.perf_counter()

            geom_vis_by_cam, overlay_by_cam, preds = self._predict_from_geometry(views_by_cam, clusters)
            t3 = time.perf_counter()

            for cam_id, v in views_by_cam.items():
                self._publish_pair(cam_id, v.rgb, overlay_by_cam[cam_id])
                self._publish_geom(cam_id, geom_vis_by_cam[cam_id])

            self.get_logger().info(
                f"[TIMING geometry_dino_total] "
                f"fusion={(t1 - t0) * 1000:.1f} ms | "
                f"geometry={(t2 - t1) * 1000:.1f} ms | "
                f"dino+draw={(t3 - t2) * 1000:.1f} ms | "
                f"total={(t3 - t0) * 1000:.1f} ms | "
                f"clusters={len(clusters)} preds={len(preds)}"
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

    node = GeometryDINODebugNode(
        grabber=grabber,
        reference_dir="Data/ZED_screens",
    )

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