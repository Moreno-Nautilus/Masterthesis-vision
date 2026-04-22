"""
Fixed external visualizer for FoundationPose.

Changes from original:
1. _make_track_overlay: fixed indentation bug (try was outside if guard)
2. _make_track_overlay: catch ALL exceptions (not just LinAlgError)
3. _make_track_overlay: render error text ON the image so you see it in Foxglove
4. _make_track_overlay: always draw pose text even if axes fail
5. Added debug prints so you can see in terminal what's happening
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Pose
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header

from fp_debug_msgs.msg import DebugCandidate, DebugFrame, DebugMaskCrop, DebugPoseItem

FAST_QOS = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
)


def rgb_numpy_to_imgmsg(rgb: np.ndarray, frame_id: str, stamp) -> Image:
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


def imgmsg_to_rgb_numpy(msg: Image) -> np.ndarray:
    if msg.encoding == "rgb8":
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        return arr.copy()
    if msg.encoding == "bgr8":
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    if msg.encoding == "rgba8":
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 4)
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
    if msg.encoding == "bgra8":
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 4)
        return cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB)
    raise ValueError(f"Unsupported image encoding: {msg.encoding}")


def quaternion_xyzw_to_rotation_matrix(q_xyzw: np.ndarray) -> np.ndarray:
    q = np.asarray(q_xyzw, dtype=np.float64).reshape(4)
    x, y, z, w = q
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3, dtype=np.float32)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float32,
    )


def pose_msg_to_T(p: Pose) -> np.ndarray:
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = np.array([p.position.x, p.position.y, p.position.z], dtype=np.float32)
    q = np.array([p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w], dtype=np.float32)
    T[:3, :3] = quaternion_xyzw_to_rotation_matrix(q)
    return T


def project_points(K: np.ndarray, pts_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = pts_cam[:, 2]
    valid = z > 1e-6
    uv = np.zeros((pts_cam.shape[0], 2), dtype=np.float32)
    if np.any(valid):
        x = pts_cam[valid, 0] / z[valid]
        y = pts_cam[valid, 1] / z[valid]
        uv_valid = np.stack([K[0, 0] * x + K[0, 2], K[1, 1] * y + K[1, 2]], axis=1)
        uv[valid] = uv_valid
    return uv, valid


def draw_bbox_label_inplace(
    image: np.ndarray,
    bbox_xyxy: tuple[int, int, int, int],
    text: str,
    color: tuple[int, int, int],
    font_scale: float = 0.6,
) -> None:
    x0, y0, x1, y1 = [int(v) for v in bbox_xyxy]
    cv2.rectangle(image, (x0, y0), (x1, y1), color, 2)
    cv2.putText(image, text, (x0, max(20, y0 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2, cv2.LINE_AA)


def draw_pose_text_inplace(
    image: np.ndarray,
    object_id: str,
    score: float,
    T_display: np.ndarray,
    mode: str,
    obj_idx: int,
) -> None:
    t = T_display[:3, 3]
    lines = [
        f"[{obj_idx}] {mode}: {object_id}",
        f"  score: {score:.3f}",
        f"  t=[{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}]",
    ]
    y = 32 + obj_idx * 100
    for line in lines:
        cv2.putText(image, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1, cv2.LINE_AA)
        y += 26


def draw_roi_polygon_inplace(
    image: np.ndarray,
    polygon_flat: list[int],
    color: tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
    label: str = "ROI",
) -> None:
    if not polygon_flat:
        return
    polygon = np.array(polygon_flat, dtype=np.int32).reshape(-1, 2)
    cv2.polylines(image, [polygon], isClosed=True, color=color, thickness=thickness)
    cv2.putText(image, label, (int(polygon[0, 0]) + 8, max(20, int(polygon[0, 1]) - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)


def draw_axes_from_pose_inplace(
    image: np.ndarray,
    K: np.ndarray,
    T_camera_object: np.ndarray,
    axis_len_m: float = 0.03,
    thickness: int = 2,
) -> None:
    pts_obj = np.array(
        [[0, 0, 0], [axis_len_m, 0, 0], [0, axis_len_m, 0], [0, 0, axis_len_m]],
        dtype=np.float32,
    )
    pts_cam = (T_camera_object[:3, :3] @ pts_obj.T).T + T_camera_object[:3, 3]
    uv, valid = project_points(K, pts_cam)
    if not np.all(valid):
        return

    p0 = tuple(np.round(uv[0]).astype(int))
    px = tuple(np.round(uv[1]).astype(int))
    py = tuple(np.round(uv[2]).astype(int))
    pz = tuple(np.round(uv[3]).astype(int))

    cv2.line(image, p0, px, (255, 0, 0), thickness, cv2.LINE_AA)
    cv2.line(image, p0, py, (0, 255, 0), thickness, cv2.LINE_AA)
    cv2.line(image, p0, pz, (0, 0, 255), thickness, cv2.LINE_AA)
    cv2.circle(image, p0, 4, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(image, p0, 2, (0, 0, 0), -1, cv2.LINE_AA)


def overlay_mask_crop_in_bbox(
    image: np.ndarray,
    bbox_xyxy: tuple[int, int, int, int],
    mask_msg: DebugMaskCrop,
    color: tuple[int, int, int],
    alpha: float,
) -> None:
    if mask_msg.width == 0 or mask_msg.height == 0 or len(mask_msg.data) == 0:
        return
    x0, y0, x1, y1 = [int(v) for v in bbox_xyxy]
    if x1 <= x0 or y1 <= y0:
        return
    crop = np.array(mask_msg.data, dtype=np.uint8).reshape(mask_msg.height, mask_msg.width)
    target_w = x1 - x0
    target_h = y1 - y0
    if target_w <= 0 or target_h <= 0:
        return
    mask = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_NEAREST).astype(bool)
    roi = image[y0:y1, x0:x1]
    color_arr = np.array(color, dtype=np.float32)
    roi_f = roi.astype(np.float32)
    roi_f[mask] = (1.0 - alpha) * roi_f[mask] + alpha * color_arr
    roi[:] = roi_f.astype(np.uint8)


@dataclass
class FrameData:
    stamp_ns: int
    rgb: np.ndarray


class FoundationPoseExternalVisualizer(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("foundationpose_external_visualizer")
        self.args = args

        self.palette = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255),
        ]

        self.latest_frame: Optional[FrameData] = None
        self.latest_debug: Optional[DebugFrame] = None
        self.K: Optional[np.ndarray] = None

        self.cached_sam: list[DebugCandidate] = []
        self.cached_dino: list[DebugCandidate] = []

        self.create_subscription(Image, self.args.rgb_topic, self._on_rgb, FAST_QOS)
        self.create_subscription(CameraInfo, self.args.camera_info_topic, self._on_camera_info, FAST_QOS)
        self.create_subscription(DebugFrame, self.args.debug_topic, self._on_debug, FAST_QOS)

        self.pub_raw = self.create_publisher(Image, self.args.raw_out_topic, FAST_QOS)
        self.pub_sam = self.create_publisher(Image, self.args.sam_out_topic, FAST_QOS)
        self.pub_dino = self.create_publisher(Image, self.args.dino_out_topic, FAST_QOS)
        self.pub_pose = self.create_publisher(Image, self.args.pose_out_topic, FAST_QOS)
        self.pub_track = self.create_publisher(Image, self.args.track_out_topic, FAST_QOS)
        self.timer = self.create_timer(self.args.timer_period_s, self._tick)

        self.get_logger().info(
            f"Visualizer started | cam_id={args.cam_id} "
            f"rgb={args.rgb_topic} info={args.camera_info_topic} debug={args.debug_topic}"
        )

    def _on_rgb(self, msg: Image) -> None:
        rgb = imgmsg_to_rgb_numpy(msg)
        stamp_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
        self.latest_frame = FrameData(stamp_ns=stamp_ns, rgb=rgb)

    def _on_camera_info(self, msg: CameraInfo) -> None:
        if self.K is None:
            self.get_logger().info(f"Got camera_info K: fx={msg.k[0]:.1f} fy={msg.k[4]:.1f}")
        self.K = np.asarray(msg.k, dtype=np.float32).reshape(3, 3)

    def _on_debug(self, msg: DebugFrame) -> None:
        self.latest_debug = msg
        if msg.update_sam:
            self.cached_sam = list(msg.sam_candidates)
        if msg.update_dino:
            self.cached_dino = list(msg.dino_ranked_candidates)

    def _make_sam_overlay(self, rgb: np.ndarray, dbg: DebugFrame) -> np.ndarray:
        out = rgb.copy()
        draw_roi_polygon_inplace(out, list(dbg.roi_polygon_xy_flat), label=f"ROI {dbg.cam_id}")

        if dbg.has_tiny_roi:
            x0, y0, x1, y1 = [int(v) for v in dbg.tiny_roi_xyxy]
            cv2.rectangle(out, (x0, y0), (x1, y1), (0, 255, 255), 2)
            cv2.putText(out, "tiny_roi", (x0, max(20, y0 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        for i, cand in enumerate(self.cached_sam[: dbg.max_candidate_draw]):
            color = self.palette[i % len(self.palette)]
            bbox = tuple(int(v) for v in cand.bbox_xyxy)
            if cand.has_mask:
                overlay_mask_crop_in_bbox(out, bbox, cand.mask, color, alpha=0.22)
            draw_bbox_label_inplace(out, bbox, f"{i}: sam={cand.score:.2f}", color, font_scale=0.5)
        return out

    def _make_dino_overlay(self, rgb: np.ndarray, dbg: DebugFrame) -> np.ndarray:
        out = rgb.copy()
        for i, cand in enumerate(self.cached_dino[: dbg.max_candidate_draw]):
            color = self.palette[i % len(self.palette)]
            bbox = tuple(int(v) for v in cand.bbox_xyxy)
            if cand.has_mask:
                overlay_mask_crop_in_bbox(out, bbox, cand.mask, color, alpha=0.22)
            draw_bbox_label_inplace(out, bbox, f"{cand.object_id} {cand.score:.2f}", color, font_scale=0.55)
        return out

    def _make_pose_overlay(self, rgb: np.ndarray, dbg: DebugFrame) -> np.ndarray:
        out = rgb.copy()
        for i, item in enumerate(dbg.pose_items):
            color = self.palette[i % len(self.palette)]

            if item.has_bbox:
                bbox = tuple(int(v) for v in item.bbox_xyxy)
                draw_bbox_label_inplace(out, bbox, f"{item.object_id} {item.mode}", color, font_scale=0.6)
                if item.has_mask:
                    overlay_mask_crop_in_bbox(out, bbox, item.mask, color, alpha=0.18)

            T_base = pose_msg_to_T(item.pose_base)
            draw_pose_text_inplace(out, item.object_id, item.score, T_base, item.mode, i)

            if dbg.show_axes and self.K is not None:
                T_obj_cam = pose_msg_to_T(item.pose_camera)
                try:
                    T_cam_obj = np.linalg.inv(T_obj_cam.astype(np.float64)).astype(np.float32)
                    draw_axes_from_pose_inplace(out, self.K, T_cam_obj, axis_len_m=float(item.axis_len_m), thickness=2)
                except np.linalg.LinAlgError:
                    pass
        return out

    def _make_track_overlay(self, rgb: np.ndarray, dbg: DebugFrame) -> np.ndarray:
        """
        Create overlay showing Cutie tracking mask, axes, and ICP metrics.
        
        FIXES vs original:
        - try/except is INSIDE the if-guard (was outside → NameError)
        - Catches ALL exceptions, not just LinAlgError
        - Renders errors ON the image so you see them in Foxglove
        - Always draws pose text even if axes fail
        """
        out = rgb.copy()

        # ── Tracking mask overlay ──
        if dbg.has_track_mask:
            bbox = tuple(int(v) for v in dbg.track_mask_bbox_xyxy)
            color = (0, 255, 128)
            overlay_mask_crop_in_bbox(out, bbox, dbg.track_mask, color, alpha=0.35)
            label = f"TRACK: {dbg.track_object_id}"
            draw_bbox_label_inplace(out, bbox, label, color, font_scale=0.7)

            metrics_lines = [
                f"ICP fitness: {dbg.track_icp_fitness:.2f}",
                f"ICP rmse: {dbg.track_icp_rmse_mm:.1f}mm",
            ]
            y = 32
            for line in metrics_lines:
                cv2.putText(out, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(out, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 180, 90), 1, cv2.LINE_AA)
                y += 28
        else:
            cv2.putText(out, "MODE: INIT/SEARCH", (20, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)

        # ── DEBUG: Show diagnostic info on image ──
        n_items = len(dbg.pose_items) if dbg.pose_items else 0
        has_k = self.K is not None
        diag = f"pose_items={n_items} K={'YES' if has_k else 'NO'} axes={dbg.show_axes}"
        cv2.putText(out, diag, (20, out.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        # ── Pose axes + text for each tracked object ──
        for i, item in enumerate(dbg.pose_items):
            # ALWAYS draw base-pose text regardless of axes success
            T_base = pose_msg_to_T(item.pose_base)
            t = T_base[:3, 3]
            y_start = 100 + i * 80
            text_lines = [
                f"[{i}] {item.mode}: {item.object_id}",
                f"  base: [{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}]",
            ]
            for line in text_lines:
                cv2.putText(out, line, (20, y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(out, line, (20, y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 0), 1, cv2.LINE_AA)
                y_start += 24

            # Draw axes — fully guarded, catches ALL exceptions
            if self.K is not None:
                try:
                    T_obj_cam = pose_msg_to_T(item.pose_camera)
                    T_cam_obj = np.linalg.inv(
                        T_obj_cam.astype(np.float64)
                    ).astype(np.float32)
                    draw_axes_from_pose_inplace(
                        out, self.K, T_cam_obj,
                        axis_len_m=float(item.axis_len_m),
                        thickness=2,
                    )
                except Exception as e:
                    # Render error ON the image so we can see it in Foxglove
                    err_text = f"AXES ERR [{i}]: {type(e).__name__}: {e}"
                    cv2.putText(out, err_text, (20, y_start),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
                    print(f"[TRACK_VIZ] {err_text}")

        return out

    def _tick(self) -> None:
        if self.latest_frame is None or self.latest_debug is None:
            return

        dbg = self.latest_debug
        frame = self.latest_frame

        dbg_ns = int(dbg.stamp.sec) * 1_000_000_000 + int(dbg.stamp.nanosec)
        if abs(frame.stamp_ns - dbg_ns) > int(self.args.max_sync_dt_s * 1e9):
            return

        rgb = frame.rgb
        stamp = self.get_clock().now().to_msg()

        # Wrap each overlay in try/except so one failure doesn't kill others
        try:
            sam_overlay = self._make_sam_overlay(rgb, dbg)
        except Exception as e:
            print(f"[VIZ] sam overlay failed: {e}")
            sam_overlay = rgb.copy()

        try:
            dino_overlay = self._make_dino_overlay(rgb, dbg)
        except Exception as e:
            print(f"[VIZ] dino overlay failed: {e}")
            dino_overlay = rgb.copy()

        try:
            pose_overlay = self._make_pose_overlay(rgb, dbg)
        except Exception as e:
            print(f"[VIZ] pose overlay failed: {e}")
            pose_overlay = rgb.copy()

        try:
            track_overlay = self._make_track_overlay(rgb, dbg)
        except Exception as e:
            print(f"[VIZ] track overlay FAILED: {e}")
            import traceback
            traceback.print_exc()
            # Render error on a copy of rgb so user sees something
            track_overlay = rgb.copy()
            cv2.putText(track_overlay, f"TRACK OVERLAY CRASHED: {e}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if self.args.output_scale != 1.0:
            new_w = max(1, int(round(rgb.shape[1] * self.args.output_scale)))
            new_h = max(1, int(round(rgb.shape[0] * self.args.output_scale)))
            rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
            sam_overlay = cv2.resize(sam_overlay, (new_w, new_h), interpolation=cv2.INTER_AREA)
            dino_overlay = cv2.resize(dino_overlay, (new_w, new_h), interpolation=cv2.INTER_AREA)
            pose_overlay = cv2.resize(pose_overlay, (new_w, new_h), interpolation=cv2.INTER_AREA)
            track_overlay = cv2.resize(track_overlay, (new_w, new_h), interpolation=cv2.INTER_AREA)

        self.pub_raw.publish(rgb_numpy_to_imgmsg(rgb, self.args.cam_id, stamp))
        self.pub_sam.publish(rgb_numpy_to_imgmsg(sam_overlay, self.args.cam_id, stamp))
        self.pub_dino.publish(rgb_numpy_to_imgmsg(dino_overlay, self.args.cam_id, stamp))
        self.pub_pose.publish(rgb_numpy_to_imgmsg(pose_overlay, self.args.cam_id, stamp))
        self.pub_track.publish(rgb_numpy_to_imgmsg(track_overlay, self.args.cam_id, stamp))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cam-id", type=str, default="zed2i_2")
    p.add_argument("--rgb-topic", type=str, default="/zed2i_2/zed_node/rgb/color/rect/image")
    p.add_argument("--camera-info-topic", type=str, default="/zed2i_2/zed_node/rgb/color/rect/image/camera_info")
    p.add_argument("--debug-topic", type=str, default="/perception/fp/debug_frame/zed2i_2")

    p.add_argument("--raw-out-topic", type=str, default="/perception/fp/rgb_raw/zed2i_2_external")
    p.add_argument("--sam-out-topic", type=str, default="/perception/fp/sam_overlay/zed2i_2_external")
    p.add_argument("--dino-out-topic", type=str, default="/perception/fp/dino_overlay/zed2i_2_external")
    p.add_argument("--pose-out-topic", type=str, default="/perception/fp/pose_overlay/zed2i_2_external")
    p.add_argument("--track-out-topic", type=str, default="/perception/fp/track_overlay/zed2i_2_external")

    p.add_argument("--timer-period-s", type=float, default=0.2)
    p.add_argument("--max-sync-dt-s", type=float, default=0.25)
    p.add_argument("--output-scale", type=float, default=0.5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = FoundationPoseExternalVisualizer(args)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()