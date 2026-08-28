from __future__ import annotations

import cv2
import numpy as np

from fp_debug_msgs.msg import DebugMaskCrop


# Decode a ROS Image message to an RGB array across the common encodings.
def imgmsg_to_rgb_numpy(msg) -> np.ndarray:
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


# Quaternion [x,y,z,w] → 3x3 rotation matrix.
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


# ROS Pose message → 4x4 homogeneous transform.
def pose_msg_to_T(p) -> np.ndarray:
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = np.array(
        [p.position.x, p.position.y, p.position.z],
        dtype=np.float32,
    )
    q = np.array(
        [p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w],
        dtype=np.float32,
    )
    T[:3, :3] = quaternion_xyzw_to_rotation_matrix(q)
    return T


# Project camera-frame 3D points to pixels; returns uv plus an in-front-of-camera mask.
def project_points(K: np.ndarray, pts_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = pts_cam[:, 2]
    valid = z > 1e-6

    uv = np.zeros((pts_cam.shape[0], 2), dtype=np.float32)

    if np.any(valid):
        x = pts_cam[valid, 0] / z[valid]
        y = pts_cam[valid, 1] / z[valid]
        uv_valid = np.stack(
            [
                K[0, 0] * x + K[0, 2],
                K[1, 1] * y + K[1, 2],
            ],
            axis=1,
        )
        uv[valid] = uv_valid

    return uv, valid


def draw_bbox_label_inplace(
    image: np.ndarray,
    bbox_xyxy: tuple[int, int, int, int],
    text: str,
    color: tuple[int, int, int],
    font_scale: float = 0.6,
) -> None:
    # Draw a labeled bounding box.
    x0, y0, x1, y1 = [int(v) for v in bbox_xyxy]

    cv2.rectangle(image, (x0, y0), (x1, y1), color, 2)
    cv2.putText(
        image,
        text,
        (x0, max(20, y0 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_pose_text_inplace(
    image: np.ndarray,
    object_id: str,
    score: float,
    T_display: np.ndarray,
    mode: str,
    obj_idx: int,
) -> None:
    # Draw a per-object text block (id, score, translation) stacked by index.
    t = T_display[:3, 3]
    lines = [
        f"[{obj_idx}] {mode}: {object_id}",
        f"  score: {score:.3f}",
        f"  t=[{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}]",
    ]

    y = 32 + obj_idx * 100
    for line in lines:
        cv2.putText(
            image,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        y += 26


def draw_roi_polygon_inplace(
    image: np.ndarray,
    polygon_flat: list[int],
    color: tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
    label: str = "ROI",
) -> None:
    # Draw the closed ROI polygon plus a label at its first vertex.
    if not polygon_flat:
        return

    polygon = np.array(polygon_flat, dtype=np.int32).reshape(-1, 2)
    cv2.polylines(image, [polygon], isClosed=True, color=color, thickness=thickness)
    cv2.putText(
        image,
        label,
        (int(polygon[0, 0]) + 8, max(20, int(polygon[0, 1]) - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_axes_from_pose_inplace(
    image: np.ndarray,
    K: np.ndarray,
    T_camera_object: np.ndarray,
    axis_len_m: float = 0.03,
    thickness: int = 2,
    colors: tuple[
        tuple[int, int, int],
        tuple[int, int, int],
        tuple[int, int, int],
    ] | None = None,
    label_prefix: str = "",
) -> None:
    """
    Draw object local x/y/z axes.

    T_camera_object must map object-frame points into the camera frame.

    Important:
    - This function expects camera <- object.
    - Do NOT pass inverse(object <- camera).
    """
    if colors is None:
        colors = (
            (255, 0, 0),   # x red
            (0, 255, 0),   # y green
            (0, 0, 255),   # z blue
        )

    T_camera_object = np.asarray(T_camera_object, dtype=np.float32).reshape(4, 4)

    pts_obj = np.array(
        [
            [0.0, 0.0, 0.0],
            [axis_len_m, 0.0, 0.0],
            [0.0, axis_len_m, 0.0],
            [0.0, 0.0, axis_len_m],
        ],
        dtype=np.float32,
    )

    # Project the object-frame origin + unit axis tips into the image.
    pts_cam = (T_camera_object[:3, :3] @ pts_obj.T).T + T_camera_object[:3, 3]
    uv, valid = project_points(K, pts_cam)

    if not np.all(valid):
        return

    p0 = tuple(np.round(uv[0]).astype(int))
    px = tuple(np.round(uv[1]).astype(int))
    py = tuple(np.round(uv[2]).astype(int))
    pz = tuple(np.round(uv[3]).astype(int))

    cv2.line(image, p0, px, colors[0], thickness, cv2.LINE_AA)
    cv2.line(image, p0, py, colors[1], thickness, cv2.LINE_AA)
    cv2.line(image, p0, pz, colors[2], thickness, cv2.LINE_AA)

    cv2.circle(image, p0, 4, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(image, p0, 2, (0, 0, 0), -1, cv2.LINE_AA)

    if label_prefix:
        cv2.putText(
            image,
            f"{label_prefix}x",
            px,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            colors[0],
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            f"{label_prefix}y",
            py,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            colors[1],
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            f"{label_prefix}z",
            pz,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            colors[2],
            1,
            cv2.LINE_AA,
        )


# Angle (deg) between the object's z-axis and the base-frame vertical.
def object_z_vs_base_z_deg(T_base_object: np.ndarray) -> float:
    T_base_object = np.asarray(T_base_object, dtype=np.float64).reshape(4, 4)

    z_obj = T_base_object[:3, 2]
    z_obj = z_obj / (np.linalg.norm(z_obj) + 1e-12)

    base_z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    dot = float(np.clip(np.dot(z_obj, base_z), -1.0, 1.0))

    return float(np.degrees(np.arccos(dot)))


def draw_base_axes_at_object_origin_inplace(
    image: np.ndarray,
    K: np.ndarray,
    T_base_cam: np.ndarray,
    T_base_object: np.ndarray,
    axis_len_m: float = 0.06,
    thickness: int = 2,
) -> None:
    """
    Draw base-frame x/y/z axes at the object's base-frame origin,
    projected into the current camera image.

    This is the image equivalent of the Foxglove 3D marker.
    """
    T_base_cam = np.asarray(T_base_cam, dtype=np.float64).reshape(4, 4)
    T_base_object = np.asarray(T_base_object, dtype=np.float64).reshape(4, 4)

    # T_base_cam maps camera -> base.
    # For image projection we need camera <- base.
    T_cam_base = np.linalg.inv(T_base_cam)

    # Axis tips offset from the object origin along the base x/y/z directions.
    p0_base = T_base_object[:3, 3]

    pts_base = np.array(
        [
            p0_base,
            p0_base + np.array([axis_len_m, 0.0, 0.0], dtype=np.float64),
            p0_base + np.array([0.0, axis_len_m, 0.0], dtype=np.float64),
            p0_base + np.array([0.0, 0.0, axis_len_m], dtype=np.float64),
        ],
        dtype=np.float64,
    )

    pts_cam = (T_cam_base[:3, :3] @ pts_base.T).T + T_cam_base[:3, 3]
    uv, valid = project_points(K, pts_cam.astype(np.float32))

    if not np.all(valid):
        return

    p0 = tuple(np.round(uv[0]).astype(int))
    px = tuple(np.round(uv[1]).astype(int))
    py = tuple(np.round(uv[2]).astype(int))
    pz = tuple(np.round(uv[3]).astype(int))

    base_x_col = (255, 160, 160)
    base_y_col = (160, 255, 160)
    base_z_col = (160, 160, 255)

    # Black halo for visibility.
    cv2.line(image, p0, px, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.line(image, p0, py, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.line(image, p0, pz, (0, 0, 0), thickness + 2, cv2.LINE_AA)

    cv2.line(image, p0, px, base_x_col, thickness, cv2.LINE_AA)
    cv2.line(image, p0, py, base_y_col, thickness, cv2.LINE_AA)
    cv2.line(image, p0, pz, base_z_col, thickness, cv2.LINE_AA)

    cv2.putText(
        image,
        "Bx",
        px,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        base_x_col,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "By",
        py,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        base_y_col,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "Bz",
        pz,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        base_z_col,
        2,
        cv2.LINE_AA,
    )


def overlay_mask_crop_in_bbox(
    image: np.ndarray,
    bbox_xyxy: tuple[int, int, int, int],
    mask_msg: DebugMaskCrop,
    color: tuple[int, int, int],
    alpha: float,
) -> None:
    # Alpha-blend a (resized) mask crop as a colored overlay inside its bbox.
    if mask_msg.width == 0 or mask_msg.height == 0 or len(mask_msg.data) == 0:
        return

    x0, y0, x1, y1 = [int(v) for v in bbox_xyxy]

    if x1 <= x0 or y1 <= y0:
        return

    crop = np.array(mask_msg.data, dtype=np.uint8).reshape(
        mask_msg.height,
        mask_msg.width,
    )

    target_w = x1 - x0
    target_h = y1 - y0

    if target_w <= 0 or target_h <= 0:
        return

    mask = cv2.resize(
        crop,
        (target_w, target_h),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)

    roi = image[y0:y1, x0:x1]
    color_arr = np.array(color, dtype=np.float32)
    roi_f = roi.astype(np.float32)
    roi_f[mask] = (1.0 - alpha) * roi_f[mask] + alpha * color_arr
    roi[:] = roi_f.astype(np.uint8)


def mask_crop_to_full_image(
    mask_msg: DebugMaskCrop,
    bbox_xyxy: tuple[int, int, int, int],
    image_shape_hw: tuple[int, int],
) -> np.ndarray:
    """Place a cropped bool mask (sized to bbox) back into a full-image bool array."""
    full = np.zeros(image_shape_hw, dtype=bool)

    if mask_msg.width == 0 or mask_msg.height == 0 or len(mask_msg.data) == 0:
        return full

    x0, y0, x1, y1 = [int(v) for v in bbox_xyxy]
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(image_shape_hw[1], x1)
    y1 = min(image_shape_hw[0], y1)

    if x1 <= x0 or y1 <= y0:
        return full

    crop = np.array(mask_msg.data, dtype=np.uint8).reshape(mask_msg.height, mask_msg.width)
    resized = cv2.resize(crop, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST).astype(bool)
    full[y0:y1, x0:x1] = resized
    return full
