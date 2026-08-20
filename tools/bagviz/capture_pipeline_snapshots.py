"""
Offline sanity-check dump of the perception pipeline's recorded debug output.

Reads a ros2bag (sqlite3 storage) that was captured while
run_pipeline_track_multicam_realsense was running, and for each camera pulls
out a handful of evenly-spaced fp_debug_msgs/DebugFrame samples. For every
sample it saves:

  - the native-resolution RGB + depth frame (colormap PNG + raw .npy in meters)
  - whichever already-rendered pipeline overlays the bag contains
    (rgb_raw/dino_overlay/sam_overlay/pose_overlay/track_overlay, all
    published by the pipeline's own external visualizer -- see
    src/perception/ros/learn_runners/visualize_pipeline.py)
  - full-image binary masks reconstructed from the DebugFrame's DINO/SAM
    candidate crops, per-object pose_item crops, and the Cutie track mask
  - a raw colored point cloud (.ply) back-projected from depth + camera_info,
    and one segmented point cloud per available mask
  - the same raw point cloud re-expressed in the shared base frame
    (pointcloud_raw_base.ply), whenever T_base_cam is resolvable: a static
    lookup in --extrinsics-yaml for zed2i_1, or -- for the two eye-in-hand
    RealSense cams -- the live flange pose read straight from the bag
    (/left/ee_pose, /right/ee_pose) composed with --extrinsics-realsense-yaml
    and --robot-bases-yaml the same way the live pipeline does. This is what
    lets tools/bagviz/view_pointclouds.py overlay every camera's cloud in one
    shared frame.
  - a redrawn axes overlay + poses.yaml with each detected object's pose in
    camera frame (always) and base frame (whenever T_base_cam above was
    resolvable), so the detected coordinate systems can be eyeballed against
    the image
  - camera intrinsics (frame_info.yaml["K"], 3x3 row-major), whenever a
    camera_info message was seen -- lets offline tools (e.g.
    tools/bagviz/run_object_inference_debug.py) back-project rgb_native.png +
    depth_m.npy without re-reading the bag

This only reads what the bag already contains -- it does not replay the bag
into the live docker pipeline. If a bag was recorded in pure "track" mode
(no fresh DINO/SAM detections during the recording), DebugFrame.sam_candidates
and .dino_ranked_candidates will simply be empty for every sample; that is
reported in the manifest rather than silently producing empty files.

Must run where ROS 2 Humble + this repo's workspace overlay are sourced
(rclpy, sensor_msgs, fp_debug_msgs) -- normal terminals on this machine
already do this via ~/.bashrc. Use the dedicated `bagviz` conda env for the
non-ROS deps (opencv, open3d, numpy, pyyaml):

    conda activate bagviz
    python tools/bagviz/capture_pipeline_snapshots.py --bag ~/Desktop/rosbag_20260807_173538

or simply:

    scripts/visualize_bag_pipeline.sh ~/Desktop/rosbag_20260807_173538
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import open3d as o3d
import rosbag2_py
import yaml
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

from fp_debug_msgs.msg import DebugFrame
from src.perception.ros.learn_runners.overlay_draw_utils import (
    draw_axes_from_pose_inplace,
    draw_base_axes_at_object_origin_inplace,
    draw_bbox_label_inplace,
    imgmsg_to_rgb_numpy,
    mask_crop_to_full_image,
    object_z_vs_base_z_deg,
    overlay_mask_crop_in_bbox,
    pose_msg_to_T,
)

try:
    from src.calibration.io_extrinsics import load_extrinsics_yaml
except Exception:
    load_extrinsics_yaml = None

try:
    from src.utils.robot_bases import get_active_robot_base, load_robot_bases
except Exception:
    get_active_robot_base = None
    load_robot_bases = None


# Eye-in-hand RealSense cameras: which arm they're bolted to (robot_base_key
# into config/robot_bases.yaml) and which live flange-pose topic the bag
# carries for it. Mirrors run_pipeline_track_multicam_realsense.py's
# ALL_CAMERAS -- keep in sync if that ever changes. zed2i_1 is not here: it's
# the static camera, handled entirely through --extrinsics-yaml.
DYNAMIC_CAM_CONFIG = {
    "realsense_1": dict(robot_base_key="robot_a", flange_pose_topic="/left/ee_pose"),
    "realsense_2": dict(robot_base_key="robot_b", flange_pose_topic="/right/ee_pose"),
}

PALETTE = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255),
]

OVERLAY_KINDS = ["rgb_raw", "dino_overlay", "sam_overlay", "pose_overlay", "track_overlay"]


# Per-camera-family native RGB-D topic layout. Add a prefix here to support a
# new camera family; anything not matching a prefix is skipped with a warning.
def resolve_camera_topics(cam_id: str) -> Optional[dict]:
    if cam_id.startswith("realsense"):
        return dict(
            rgb=f"/{cam_id}/camera/color/image_raw",
            depth=f"/{cam_id}/camera/aligned_depth_to_color/image_raw",
            info=f"/{cam_id}/camera/aligned_depth_to_color/camera_info",
            depth_units="mm_uint16",
        )
    if cam_id.startswith("zed"):
        return dict(
            rgb=f"/{cam_id}/zed_node/rgb/color/rect/image",
            depth=f"/{cam_id}/zed_node/depth/depth_registered",
            info=f"/{cam_id}/zed_node/depth/depth_registered/camera_info",
            depth_units="m_float32",
        )
    return None


def decode_depth_to_meters(msg, depth_units: str) -> np.ndarray:
    if depth_units == "mm_uint16":
        arr = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
        return arr.astype(np.float32) * 1e-3
    if depth_units == "m_float32":
        arr = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
        return arr.copy()
    raise ValueError(f"Unhandled depth_units: {depth_units}")


def depth_colormap(depth_m: np.ndarray, max_depth_m: float) -> np.ndarray:
    valid = np.isfinite(depth_m) & (depth_m > 0)
    norm = np.clip(depth_m / max_depth_m, 0.0, 1.0)
    norm[~valid] = 0.0
    img8 = (norm * 255.0).astype(np.uint8)
    cmap = cv2.applyColorMap(img8, cv2.COLORMAP_TURBO)
    cmap[~valid] = (0, 0, 0)
    return cv2.cvtColor(cmap, cv2.COLOR_BGR2RGB)


def backproject(
    rgb: np.ndarray,
    depth_m: np.ndarray,
    K: np.ndarray,
    mask: Optional[np.ndarray],
    min_depth_m: float,
    max_depth_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = depth_m.shape
    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    z = depth_m
    valid = np.isfinite(z) & (z > min_depth_m) & (z < max_depth_m)
    if mask is not None:
        valid &= mask

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x = (us - cx) * z / fx
    y = (vs - cy) * z / fy

    pts = np.stack([x[valid], y[valid], z[valid]], axis=-1).astype(np.float64)
    cols = (rgb[valid].astype(np.float64) / 255.0)
    return pts, cols


def save_point_cloud(path: Path, pts: np.ndarray, cols: np.ndarray, voxel_size_m: float) -> int:
    pc = o3d.geometry.PointCloud()
    if pts.shape[0] > 0:
        pc.points = o3d.utility.Vector3dVector(pts)
        pc.colors = o3d.utility.Vector3dVector(cols)
        if voxel_size_m > 0:
            pc = pc.voxel_down_sample(voxel_size_m)
    o3d.io.write_point_cloud(str(path), pc, write_ascii=False)
    return len(pc.points)


def pose_to_dict(T: np.ndarray) -> dict:
    t = T[:3, 3].tolist()
    R = T[:3, :3]
    # Rotation matrix -> quaternion (xyzw), for a compact/readable dump.
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    else:
        i = np.argmax([R[0, 0], R[1, 1], R[2, 2]])
        if i == 0:
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif i == 1:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S
    return {"t": [float(v) for v in t], "quat_xyzw": [float(qx), float(qy), float(qz), float(qw)]}


@dataclass
class LatestState:
    rgb: Optional[tuple] = None       # (stamp_ns, np.ndarray)
    depth: Optional[tuple] = None     # (stamp_ns, msg)
    K: Optional[np.ndarray] = None
    overlays: dict = field(default_factory=dict)  # kind -> (stamp_ns, np.ndarray)
    # Dynamic (eye-in-hand) cameras only: latest live T_activeRobot_flange,
    # recomposed every time a fresh flange-pose message is read from the bag.
    flange_T_active_flange: Optional[tuple] = None  # (stamp_ns, np.ndarray 4x4)


def get_topic_counts(reader: rosbag2_py.SequentialReader) -> dict[str, int]:
    return {t.topic_metadata.name: t.message_count for t in reader.get_metadata().topics_with_message_count}


def pick_sample_stride(count: int, want: int) -> int:
    if want <= 0 or count <= want:
        return 1
    return max(1, count // want)


def stamp_to_ns(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def process_camera_snapshot(
    cam_id: str,
    frame_idx: int,
    dbg: DebugFrame,
    state: LatestState,
    out_dir: Path,
    T_base_cam: Optional[np.ndarray],
    base_cam_unavailable_reason: Optional[str],
    args: argparse.Namespace,
    T_base_flange: Optional[np.ndarray] = None,
) -> dict:
    frame_dir = out_dir / cam_id / f"frame_{frame_idx:02d}"
    frame_dir.mkdir(parents=True, exist_ok=True)

    info: dict = {
        "cam_id": cam_id,
        "frame_idx": frame_idx,
        "debug_frame_stamp_ns": stamp_to_ns(dbg.stamp),
        "n_pose_items": len(dbg.pose_items),
        "n_sam_candidates": len(dbg.sam_candidates),
        "n_dino_candidates": len(dbg.dino_ranked_candidates),
        "has_track_mask": bool(dbg.has_track_mask),
        "notes": [],
    }
    if T_base_cam is not None:
        info["T_base_cam"] = T_base_cam.tolist()
    if T_base_flange is not None:
        # Flange pose (in the active robot's base frame) at capture time,
        # saved separately from T_base_cam so tools/bagviz/view_pointclouds.py
        # can recompose T_base_cam = T_base_flange @ T_flange_cam offline with
        # an updated camera-to-flange offset, without needing to re-capture.
        info["T_base_flange"] = T_base_flange.tolist()

    if len(dbg.sam_candidates) == 0 and len(dbg.dino_ranked_candidates) == 0:
        info["notes"].append(
            "This DebugFrame's own sam_candidates/dino_ranked_candidates are empty -- the "
            "pipeline only fills them on frames where a fresh DINO/SAM (re-)detection ran, "
            "and this sample is a track-mode frame. mask_sam_*.png/mask_dino_*.png are "
            "therefore not reconstructed for this frame. dino_overlay.png/sam_overlay.png "
            "(if present below) may still show real boxes/masks -- those come from the "
            "external visualizer's own cache of the last detection, not from this message."
        )

    # Already-rendered pipeline overlays, saved verbatim if the bag has them.
    for kind in OVERLAY_KINDS:
        entry = state.overlays.get(kind)
        if entry is None:
            info["notes"].append(f"No '{kind}' topic recorded for this camera in the bag.")
            continue
        stamp_ns, img = entry
        cv2.imwrite(str(frame_dir / f"{kind}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        info[f"{kind}_dt_s"] = abs(stamp_ns - info["debug_frame_stamp_ns"]) / 1e9

    rgb_native = None
    depth_m = None
    K = state.K
    if K is not None:
        # Saved so offline tools (e.g. run_object_inference_debug.py) can
        # back-project without re-reading the bag's camera_info topic.
        info["K"] = K.tolist()

    if state.rgb is not None:
        rgb_stamp_ns, rgb_native = state.rgb
        cv2.imwrite(str(frame_dir / "rgb_native.png"), cv2.cvtColor(rgb_native, cv2.COLOR_RGB2BGR))
        info["rgb_native_dt_s"] = abs(rgb_stamp_ns - info["debug_frame_stamp_ns"]) / 1e9
    else:
        info["notes"].append("No native-resolution RGB topic found for point-cloud construction.")

    if state.depth is not None and K is not None:
        depth_stamp_ns, depth_msg, depth_units = state.depth
        depth_m = decode_depth_to_meters(depth_msg, depth_units)
        np.save(frame_dir / "depth_m.npy", depth_m)
        cv2.imwrite(
            str(frame_dir / "depth_colormap.png"),
            cv2.cvtColor(depth_colormap(depth_m, args.max_depth_m), cv2.COLOR_RGB2BGR),
        )
        info["depth_dt_s"] = abs(depth_stamp_ns - info["debug_frame_stamp_ns"]) / 1e9
    else:
        info["notes"].append("No native-resolution depth topic found for point-cloud construction.")

    # --- point clouds -----------------------------------------------------
    if rgb_native is not None and depth_m is not None and K is not None:
        if rgb_native.shape[:2] != depth_m.shape[:2]:
            info["notes"].append(
                f"RGB {rgb_native.shape[:2]} and depth {depth_m.shape[:2]} resolution mismatch; "
                "resizing depth (nearest) onto the RGB grid."
            )
            depth_m = cv2.resize(
                depth_m, (rgb_native.shape[1], rgb_native.shape[0]), interpolation=cv2.INTER_NEAREST
            )

        pts, cols = backproject(rgb_native, depth_m, K, None, args.min_depth_m, args.max_depth_m)
        n_raw = save_point_cloud(frame_dir / "pointcloud_raw.ply", pts, cols, args.voxel_size_m)
        info["pointcloud_raw_points"] = n_raw

        # Same cloud re-expressed in the shared base frame (active robot's
        # lbr_link_0 -- same convention as poses.yaml's pose_base), so a
        # viewer can combine multiple cameras' clouds without redoing any
        # extrinsics math. Only possible when T_base_cam was resolved above.
        if T_base_cam is not None and pts.shape[0] > 0:
            pts_base = (T_base_cam[:3, :3] @ pts.T).T + T_base_cam[:3, 3]
            save_point_cloud(frame_dir / "pointcloud_raw_base.ply", pts_base, cols, args.voxel_size_m)

        # Best available full-image segmentation mask, in priority order.
        seg_mask = None
        seg_label = None
        if dbg.has_track_mask:
            seg_mask = mask_crop_to_full_image(
                dbg.track_mask, tuple(dbg.track_mask_bbox_xyxy), depth_m.shape[:2]
            )
            seg_label = f"track_{dbg.track_object_id}" if dbg.track_object_id else "track"
        elif dbg.sam_candidates and dbg.sam_candidates[0].has_mask:
            c = dbg.sam_candidates[0]
            seg_mask = mask_crop_to_full_image(c.mask, tuple(c.bbox_xyxy), depth_m.shape[:2])
            seg_label = f"sam_{c.object_id or 'top'}"
        elif dbg.dino_ranked_candidates and dbg.dino_ranked_candidates[0].has_mask:
            c = dbg.dino_ranked_candidates[0]
            seg_mask = mask_crop_to_full_image(c.mask, tuple(c.bbox_xyxy), depth_m.shape[:2])
            seg_label = f"dino_{c.object_id or 'top'}"

        if seg_mask is not None:
            pts_s, cols_s = backproject(rgb_native, depth_m, K, seg_mask, args.min_depth_m, args.max_depth_m)
            n_seg = save_point_cloud(frame_dir / "pointcloud_segmented.ply", pts_s, cols_s, args.voxel_size_m)
            info["pointcloud_segmented_points"] = n_seg
            info["pointcloud_segmented_source"] = seg_label
        else:
            info["notes"].append("No mask available (track/SAM/DINO) to build a segmented point cloud.")

        # One extra segmented cloud per per-object pose_item mask, when present.
        for item in dbg.pose_items:
            if not item.has_mask:
                continue
            label = f"{item.assembly_name}_{item.part_id}" if item.assembly_name else str(item.part_id)
            m = mask_crop_to_full_image(item.mask, tuple(item.bbox_xyxy), depth_m.shape[:2])
            pts_i, cols_i = backproject(rgb_native, depth_m, K, m, args.min_depth_m, args.max_depth_m)
            save_point_cloud(frame_dir / f"pointcloud_segmented_{label}.ply", pts_i, cols_i, args.voxel_size_m)

    # --- masks reconstructed from the DebugFrame candidate crops -----------
    if depth_m is not None:
        shape_hw = depth_m.shape[:2]
        if dbg.has_track_mask:
            m = mask_crop_to_full_image(dbg.track_mask, tuple(dbg.track_mask_bbox_xyxy), shape_hw)
            cv2.imwrite(str(frame_dir / "mask_track.png"), (m.astype(np.uint8) * 255))
        for i, c in enumerate(dbg.sam_candidates):
            if not c.has_mask:
                continue
            m = mask_crop_to_full_image(c.mask, tuple(c.bbox_xyxy), shape_hw)
            cv2.imwrite(str(frame_dir / f"mask_sam_{i}_{c.object_id}.png"), (m.astype(np.uint8) * 255))
        for i, c in enumerate(dbg.dino_ranked_candidates):
            if not c.has_mask:
                continue
            m = mask_crop_to_full_image(c.mask, tuple(c.bbox_xyxy), shape_hw)
            cv2.imwrite(str(frame_dir / f"mask_dino_{i}_{c.object_id}.png"), (m.astype(np.uint8) * 255))

    # --- coordinate-system sanity overlay + numeric poses -------------------
    poses_out = []
    n_origin_in_frame = 0
    if rgb_native is not None and K is not None:
        h_img, w_img = rgb_native.shape[:2]
        axes_img = rgb_native.copy()
        for i, item in enumerate(dbg.pose_items):
            color = PALETTE[i % len(PALETTE)]
            label = f"{item.assembly_name}/{item.part_id}" if item.assembly_name else str(item.part_id)

            if item.has_bbox:
                draw_bbox_label_inplace(axes_img, tuple(item.bbox_xyxy), f"{label} {item.mode}", color)
                if item.has_mask:
                    overlay_mask_crop_in_bbox(axes_img, tuple(item.bbox_xyxy), item.mask, color, alpha=0.18)

            T_cam_obj = pose_msg_to_T(item.pose_camera.pose)

            # Origin visibility check purely for reporting -- draw_axes_from_pose_inplace
            # itself silently no-ops offscreen (cv2 clips), so a blank axes_overlay.png
            # would otherwise look like a bug rather than "this object isn't in this
            # camera's view right now" (expected for cross-camera-fused tracking).
            origin_cam = T_cam_obj[:3, 3]
            if origin_cam[2] > 1e-6:
                u = K[0, 0] * origin_cam[0] / origin_cam[2] + K[0, 2]
                v = K[1, 1] * origin_cam[1] / origin_cam[2] + K[1, 2]
                if 0 <= u < w_img and 0 <= v < h_img:
                    n_origin_in_frame += 1

            draw_axes_from_pose_inplace(
                axes_img, K, T_cam_obj, axis_len_m=max(0.03, float(item.axis_len_m)), label_prefix="O"
            )

            pose_entry = {
                "assembly_name": item.assembly_name,
                "part_id": int(item.part_id),
                "mode": item.mode,
                "score": float(item.score),
                "pose_camera": pose_to_dict(T_cam_obj),
            }

            if T_base_cam is not None:
                T_base_obj = pose_msg_to_T(item.pose_base.pose)
                draw_base_axes_at_object_origin_inplace(axes_img, K, T_base_cam, T_base_obj)
                pose_entry["pose_base"] = pose_to_dict(T_base_obj)
                pose_entry["z_vs_base_z_deg"] = object_z_vs_base_z_deg(T_base_obj)

            poses_out.append(pose_entry)

        if dbg.pose_items:
            cv2.imwrite(str(frame_dir / "axes_overlay.png"), cv2.cvtColor(axes_img, cv2.COLOR_RGB2BGR))
            info["pose_item_origins_in_frame"] = f"{n_origin_in_frame}/{len(dbg.pose_items)}"
            if n_origin_in_frame < len(dbg.pose_items):
                info["notes"].append(
                    f"Only {n_origin_in_frame}/{len(dbg.pose_items)} pose_item origins project inside "
                    f"this camera's frame -- the rest are tracked objects reported in this DebugFrame "
                    "but not currently visible to this camera (expected for cross-camera fused "
                    "tracking); their axes are correctly omitted from axes_overlay.png, not a bug. "
                    "See poses.yaml for their numeric pose regardless."
                )
        else:
            info["notes"].append("No pose_items in this DebugFrame -- nothing to draw axes for.")

        if T_base_cam is None:
            info["notes"].append(
                base_cam_unavailable_reason
                or f"No base extrinsics found for '{cam_id}' in {args.extrinsics_yaml} -- "
                "base-frame axes/pose skipped (camera-frame pose is still reported)."
            )

    (frame_dir / "poses.yaml").write_text(yaml.safe_dump({"pose_items": poses_out}, sort_keys=False))
    (frame_dir / "frame_info.yaml").write_text(yaml.safe_dump(info, sort_keys=False))
    return info


def run(args: argparse.Namespace) -> None:
    bag_path = Path(args.bag).expanduser().resolve()
    if not bag_path.exists():
        raise SystemExit(f"Bag not found: {bag_path}")

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (
        Path("outputs/bagviz") / f"{bag_path.name}_{datetime.now().strftime('%Y%m%dT%H%M%SZ')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_path), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("", ""),
    )
    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
    counts = get_topic_counts(reader)

    debug_topics = {t: c for t, c in counts.items() if t.startswith("/perception/fp/debug_frame/")}
    all_cams = sorted(t.rsplit("/", 1)[-1] for t in debug_topics)
    cams = [c.strip() for c in args.cameras.split(",")] if args.cameras else all_cams
    cams = [c for c in cams if c in all_cams]
    if not cams:
        raise SystemExit(f"No requested cameras found. Debug-frame topics present: {sorted(all_cams)}")

    extrinsics = {}
    if load_extrinsics_yaml is not None and Path(args.extrinsics_yaml).exists():
        try:
            extrinsics = load_extrinsics_yaml(args.extrinsics_yaml)
        except Exception as e:
            print(f"[WARN] failed to load {args.extrinsics_yaml}: {e}")

    # T_flange_cam for the eye-in-hand RealSense cams (static mount offset),
    # and T_activeRobot_<robot_base_key> for every robot_base_key any of
    # those cams need -- lets a dynamic cam's live T_base_cam(t) be built the
    # same way the live pipeline does: T_activeRobot_flange(t) @ T_flange_cam.
    extrinsics_realsense = {}
    if load_extrinsics_yaml is not None and Path(args.extrinsics_realsense_yaml).exists():
        try:
            extrinsics_realsense = load_extrinsics_yaml(args.extrinsics_realsense_yaml)
        except Exception as e:
            print(f"[WARN] failed to load {args.extrinsics_realsense_yaml}: {e}")

    T_active_robotkey = {}
    if get_active_robot_base is not None and load_robot_bases is not None and Path(args.robot_bases_yaml).exists():
        try:
            active_robot, T_robotA_activeRobot = get_active_robot_base(args.robot_bases_yaml)
            T_activeRobot_robotA = T_robotA_activeRobot.inverse()
            robot_bases_all = load_robot_bases(args.robot_bases_yaml)
            needed_keys = {cfg["robot_base_key"] for cfg in DYNAMIC_CAM_CONFIG.values() if cfg["robot_base_key"] in robot_bases_all}
            for key in needed_keys:
                T_active_robotkey[key] = T_activeRobot_robotA.compose(robot_bases_all[key]).as_matrix()
            print(f"[*] active_robot={active_robot} ({args.robot_bases_yaml}) -- dynamic cams resolve into this frame")
        except Exception as e:
            print(f"[WARN] failed to load {args.robot_bases_yaml}: {e}")

    flange_topic_by_cam = {
        c: DYNAMIC_CAM_CONFIG[c]["flange_pose_topic"]
        for c in cams if c in DYNAMIC_CAM_CONFIG
    }
    for c, topic in flange_topic_by_cam.items():
        if topic not in counts:
            print(f"[WARN] '{c}': flange-pose topic {topic} not in bag, base-frame pose/pointcloud will be skipped")

    cam_profiles = {}
    strides = {}
    counters = {c: 0 for c in cams}
    captured = {c: 0 for c in cams}
    for c in cams:
        profile = resolve_camera_topics(c)
        if profile is None:
            print(f"[WARN] unrecognized camera family for '{c}', skipping native RGB-D/pointcloud step")
        else:
            missing = [k for k in ("rgb", "depth", "info") if profile[k] not in counts]
            if missing:
                print(f"[WARN] '{c}': missing native topics {missing}, pointcloud step will be skipped")
        cam_profiles[c] = profile
        strides[c] = pick_sample_stride(debug_topics[f"/perception/fp/debug_frame/{c}"], args.num_frames)

    state = {c: LatestState() for c in cams}
    manifest = {"bag": str(bag_path), "out_dir": str(out_dir), "cameras": {}}

    print(f"[*] bag: {bag_path}")
    print(f"[*] cameras: {cams}")
    for c in cams:
        print(f"    {c}: {debug_topics[f'/perception/fp/debug_frame/{c}']} debug frames available, "
              f"sampling every {strides[c]}-th -> up to {args.num_frames}")

    while reader.has_next():
        topic, data, t = reader.read_next()
        msg_type = type_map.get(topic)
        if msg_type is None:
            continue

        for c in cams:
            profile = cam_profiles[c]
            st = state[c]

            if profile is not None and topic == profile["rgb"]:
                msg = deserialize_message(data, get_message(msg_type))
                st.rgb = (stamp_to_ns(msg.header.stamp), imgmsg_to_rgb_numpy(msg))
            elif profile is not None and topic == profile["depth"]:
                msg = deserialize_message(data, get_message(msg_type))
                st.depth = (stamp_to_ns(msg.header.stamp), msg, profile["depth_units"])
            elif profile is not None and topic == profile["info"]:
                msg = deserialize_message(data, get_message(msg_type))
                st.K = np.asarray(msg.k, dtype=np.float64).reshape(3, 3)
            elif topic == flange_topic_by_cam.get(c):
                robot_base_key = DYNAMIC_CAM_CONFIG[c]["robot_base_key"]
                T_ar_key = T_active_robotkey.get(robot_base_key)
                if T_ar_key is not None:
                    msg = deserialize_message(data, get_message(msg_type))
                    T_robotkey_flange = pose_msg_to_T(msg.pose)
                    st.flange_T_active_flange = (stamp_to_ns(msg.header.stamp), T_ar_key @ T_robotkey_flange)
            else:
                for kind in OVERLAY_KINDS:
                    if topic == f"/perception/fp/{kind}/{c}_external":
                        msg = deserialize_message(data, get_message(msg_type))
                        st.overlays[kind] = (stamp_to_ns(msg.header.stamp), imgmsg_to_rgb_numpy(msg))
                        break

            if topic == f"/perception/fp/debug_frame/{c}":
                idx = counters[c]
                counters[c] += 1
                if captured[c] >= args.num_frames:
                    continue
                if idx % strides[c] != 0:
                    continue

                dbg = deserialize_message(data, get_message(msg_type))
                T_base_cam = None
                T_base_flange = None
                base_cam_reason = None
                if c in extrinsics:
                    T = extrinsics[c]
                    T_base_cam = T.as_matrix() if hasattr(T, "as_matrix") else np.asarray(T).reshape(4, 4)
                elif c in DYNAMIC_CAM_CONFIG:
                    T_flange_cam = extrinsics_realsense.get(c)
                    if T_flange_cam is None:
                        base_cam_reason = (
                            f"No '{c}' entry in {args.extrinsics_realsense_yaml} (camera-to-flange "
                            "hand-eye offset) -- base-frame axes/pose/pointcloud skipped."
                        )
                    elif st.flange_T_active_flange is None:
                        topic = flange_topic_by_cam[c]
                        base_cam_reason = (
                            f"No '{topic}' message seen yet by this point in the bag -- base-frame "
                            "axes/pose/pointcloud skipped (camera-frame pose is still reported)."
                        )
                    else:
                        T_flange_cam_mat = T_flange_cam.as_matrix() if hasattr(T_flange_cam, "as_matrix") else np.asarray(T_flange_cam).reshape(4, 4)
                        T_base_flange = st.flange_T_active_flange[1]
                        T_base_cam = T_base_flange @ T_flange_cam_mat

                frame_info = process_camera_snapshot(
                    c, captured[c], dbg, st, out_dir, T_base_cam, base_cam_reason, args,
                    T_base_flange=T_base_flange,
                )
                manifest["cameras"].setdefault(c, []).append(frame_info)
                captured[c] += 1
                print(f"    [{c}] captured frame {captured[c]}/{args.num_frames} "
                      f"(pose_items={frame_info['n_pose_items']}, "
                      f"sam={frame_info['n_sam_candidates']}, dino={frame_info['n_dino_candidates']})")

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\n[*] done -> {out_dir}")
    for c in cams:
        print(f"    {c}: captured {captured[c]} frame(s) (bag had {debug_topics[f'/perception/fp/debug_frame/{c}']} debug frames total)")
    any_detections = any(
        fi["n_sam_candidates"] or fi["n_dino_candidates"]
        for frames in manifest["cameras"].values() for fi in frames
    )
    if not any_detections:
        print(
            "\n[!] Every sampled DebugFrame had empty sam_candidates/dino_ranked_candidates -- this "
            "bag looks like it was recorded entirely in tracking mode (no fresh (re-)detection ran "
            "during the recording), so mask_sam_*.png/mask_dino_*.png were not produced for any frame. "
            "Check each frame's dino_overlay.png/sam_overlay.png anyway -- those may still show real "
            "boxes/masks cached by the external visualizer from an earlier detection. Point this tool "
            "at an init_only (or freshly re-detecting) bag for a frame-accurate DINO/SAM capture."
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bag", required=True, help="Path to a ros2bag directory (sqlite3 storage).")
    p.add_argument("--out-dir", default=None, help="Default: outputs/bagviz/<bag_name>_<timestamp>/")
    p.add_argument("--num-frames", type=int, default=10, help="Max DebugFrame samples per camera.")
    p.add_argument("--cameras", default=None, help="Comma-separated cam_ids; default = all found in the bag.")
    p.add_argument("--extrinsics-yaml", default="config/camera_extrinsics_base.yaml")
    p.add_argument("--extrinsics-realsense-yaml", default="config/camera_extrinsics_realsense.yaml",
                    help="Camera-to-flange hand-eye offsets for realsense_1/realsense_2 (see DYNAMIC_CAM_CONFIG).")
    p.add_argument("--robot-bases-yaml", default="config/robot_bases.yaml",
                    help="Cross-robot base offsets, used to land dynamic cams in the same frame as --extrinsics-yaml.")
    p.add_argument("--voxel-size-m", type=float, default=0.003, help="0 disables downsampling.")
    p.add_argument("--min-depth-m", type=float, default=0.05)
    p.add_argument("--max-depth-m", type=float, default=2.0)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
