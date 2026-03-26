#!/usr/bin/env python3

import json
import math
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


METHOD_MAP = {
    "tsai": cv2.CALIB_HAND_EYE_TSAI,
    "park": cv2.CALIB_HAND_EYE_PARK,
    "horaud": cv2.CALIB_HAND_EYE_HORAUD,
    "andreff": cv2.CALIB_HAND_EYE_ANDREFF,
    "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def make_homogeneous(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)
    return T


def invert_transform(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4, dtype=float)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def quat_xyzw_to_rot(q: List[float]) -> np.ndarray:
    x, y, z, w = q
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        raise ValueError("Quaternion norm too small.")
    x /= n
    y /= n
    z /= n
    w /= n

    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ], dtype=float)
    return R


def rot_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    # Robust conversion
    m = R
    tr = np.trace(m)

    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (m[2, 1] - m[1, 2]) / S
        y = (m[0, 2] - m[2, 0]) / S
        z = (m[1, 0] - m[0, 1]) / S
    elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
        S = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / S
        x = 0.25 * S
        y = (m[0, 1] + m[1, 0]) / S
        z = (m[0, 2] + m[2, 0]) / S
    elif m[1, 1] > m[2, 2]:
        S = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / S
        x = (m[0, 1] + m[1, 0]) / S
        y = 0.25 * S
        z = (m[1, 2] + m[2, 1]) / S
    else:
        S = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / S
        x = (m[0, 2] + m[2, 0]) / S
        y = (m[1, 2] + m[2, 1]) / S
        z = 0.25 * S

    q = np.array([x, y, z, w], dtype=float)
    q /= np.linalg.norm(q)
    return q


def board_object_points(cols: int, rows: int, square_size_m: float) -> np.ndarray:
    objp = np.zeros((rows * cols, 3), np.float32)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp[:, :2] = grid
    objp *= square_size_m
    return objp


def solve_board_pose_from_sample(sample: dict) -> Tuple[np.ndarray, np.ndarray]:
    board = sample["board"]
    cols = int(board["cols_inner_corners"])
    rows = int(board["rows_inner_corners"])
    square_size_m = float(board["square_size_m"])

    obj_pts = board_object_points(cols, rows, square_size_m).astype(np.float32)
    img_pts = np.array(sample["detector"]["corners_px"], dtype=np.float32).reshape(-1, 1, 2)

    K = np.array(sample["camera_info"]["k"], dtype=float).reshape(3, 3)
    D = np.array(sample["camera_info"]["d"], dtype=float).reshape(-1, 1)

    ok, rvec, tvec = cv2.solvePnP(
        objectPoints=obj_pts,
        imagePoints=img_pts,
        cameraMatrix=K,
        distCoeffs=D,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        raise RuntimeError("cv2.solvePnP failed.")

    R_cam_board, _ = cv2.Rodrigues(rvec)
    t_cam_board = tvec.reshape(3)
    return R_cam_board, t_cam_board


def robot_pose_from_sample(sample: dict) -> Tuple[np.ndarray, np.ndarray]:
    pose = sample["robot_pose_base_to_ee"]
    t = np.array(pose["translation_xyz"], dtype=float)
    q = np.array(pose["quaternion_xyzw"], dtype=float)
    R = quat_xyzw_to_rot(q)
    return R, t


def reprojection_rmse(
    sample: dict,
    R_cam_board: np.ndarray,
    t_cam_board: np.ndarray,
) -> float:
    board = sample["board"]
    cols = int(board["cols_inner_corners"])
    rows = int(board["rows_inner_corners"])
    square_size_m = float(board["square_size_m"])

    obj_pts = board_object_points(cols, rows, square_size_m).astype(np.float32)
    img_pts = np.array(sample["detector"]["corners_px"], dtype=np.float32).reshape(-1, 2)

    K = np.array(sample["camera_info"]["k"], dtype=float).reshape(3, 3)
    D = np.array(sample["camera_info"]["d"], dtype=float).reshape(-1, 1)

    rvec, _ = cv2.Rodrigues(R_cam_board)
    proj, _ = cv2.projectPoints(obj_pts, rvec, t_cam_board.reshape(3, 1), K, D)
    proj = proj.reshape(-1, 2)

    err = np.linalg.norm(proj - img_pts, axis=1)
    return float(np.sqrt(np.mean(err ** 2)))


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Solve wrist hand-eye calibration from recorded checkerboard samples.")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Folder containing sample_XXXX.json files.")
    parser.add_argument("--method", type=str, default="tsai", choices=list(METHOD_MAP.keys()))
    parser.add_argument("--min_samples", type=int, default=10)
    parser.add_argument("--max_reproj_rmse_px", type=float, default=2.0,
                        help="Reject sample if checkerboard PnP reprojection RMSE exceeds this.")
    parser.add_argument("--output_json", type=str, default="handeye_result.json")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset dir does not exist: {dataset_dir}")

    sample_paths = sorted(dataset_dir.glob("sample_*.json"))
    if len(sample_paths) == 0:
        raise RuntimeError(f"No sample_*.json files found in {dataset_dir}")

    print(f"Found {len(sample_paths)} sample json files in {dataset_dir}")

    R_gripper2base = []
    t_gripper2base = []
    R_target2cam = []
    t_target2cam = []

    accepted = []
    rejected = []

    for path in sample_paths:
        try:
            sample = load_json(path)

            # Robot side: recorder stored T_base_ee
            R_base_ee, t_base_ee = robot_pose_from_sample(sample)
            T_base_ee = make_homogeneous(R_base_ee, t_base_ee)

            # OpenCV calibrateHandEye expects gripper->base
            T_ee_base = invert_transform(T_base_ee)
            R_ee_base = T_ee_base[:3, :3]
            t_ee_base = T_ee_base[:3, 3]

            # Vision side: solvePnP gives target(board)->camera
            R_cam_board, t_cam_board = solve_board_pose_from_sample(sample)

            rmse = reprojection_rmse(sample, R_cam_board, t_cam_board)

            if rmse > args.max_reproj_rmse_px:
                rejected.append((path.name, rmse, "high reprojection RMSE"))
                continue

            R_gripper2base.append(R_ee_base)
            t_gripper2base.append(t_ee_base.reshape(3, 1))
            R_target2cam.append(R_cam_board)
            t_target2cam.append(t_cam_board.reshape(3, 1))

            accepted.append({
                "sample": path.name,
                "reproj_rmse_px": rmse,
            })

        except Exception as e:
            rejected.append((path.name, None, str(e)))

    print(f"Accepted {len(accepted)} samples")
    print(f"Rejected {len(rejected)} samples")

    if len(accepted) < args.min_samples:
        raise RuntimeError(
            f"Not enough accepted samples for hand-eye. "
            f"Need at least {args.min_samples}, got {len(accepted)}."
        )

    method_flag = METHOD_MAP[args.method]

    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base=R_gripper2base,
        t_gripper2base=t_gripper2base,
        R_target2cam=R_target2cam,
        t_target2cam=t_target2cam,
        method=method_flag,
    )

    # OpenCV returns camera->gripper; here gripper == ee
    T_ee_cam = make_homogeneous(R_cam2gripper, t_cam2gripper.reshape(3))
    T_cam_ee = invert_transform(T_ee_cam)

    q_ee_cam = rot_to_quat_xyzw(T_ee_cam[:3, :3])
    q_cam_ee = rot_to_quat_xyzw(T_cam_ee[:3, :3])

    mean_rmse = float(np.mean([a["reproj_rmse_px"] for a in accepted]))
    med_rmse = float(np.median([a["reproj_rmse_px"] for a in accepted]))

    result = {
        "dataset_dir": str(dataset_dir),
        "method": args.method,
        "num_total_samples": len(sample_paths),
        "num_accepted_samples": len(accepted),
        "num_rejected_samples": len(rejected),
        "accepted_samples": accepted,
        "rejected_samples": [
            {
                "sample": name,
                "reproj_rmse_px": rmse,
                "reason": reason,
            }
            for (name, rmse, reason) in rejected
        ],
        "quality": {
            "mean_reprojection_rmse_px": mean_rmse,
            "median_reprojection_rmse_px": med_rmse,
        },
        "convention_notes": {
            "recorded_robot_pose": "T_base_ee",
            "pnp_pose": "T_cam_board (board/target to camera)",
            "opencv_output": "T_ee_cam (camera to ee/gripper)",
            "inverse_also_saved": "T_cam_ee (ee to camera)",
        },
        "T_ee_cam": {
            "matrix_4x4": T_ee_cam.tolist(),
            "translation_xyz_m": T_ee_cam[:3, 3].tolist(),
            "quaternion_xyzw": q_ee_cam.tolist(),
        },
        "T_cam_ee": {
            "matrix_4x4": T_cam_ee.tolist(),
            "translation_xyz_m": T_cam_ee[:3, 3].tolist(),
            "quaternion_xyzw": q_cam_ee.tolist(),
        },
    }

    output_path = Path(args.output_json).expanduser().resolve()
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    np.set_printoptions(precision=6, suppress=True)

    print("\n=== Hand-eye result ===")
    print(f"Method: {args.method}")
    print(f"Accepted samples: {len(accepted)} / {len(sample_paths)}")
    print(f"Mean reprojection RMSE:   {mean_rmse:.4f} px")
    print(f"Median reprojection RMSE: {med_rmse:.4f} px")

    print("\nT_ee_cam (camera -> ee)")
    print(T_ee_cam)
    print(f"translation xyz [m]: {T_ee_cam[:3, 3]}")
    print(f"quaternion xyzw    : {q_ee_cam}")

    print("\nT_cam_ee (ee -> camera)")
    print(T_cam_ee)
    print(f"translation xyz [m]: {T_cam_ee[:3, 3]}")
    print(f"quaternion xyzw    : {q_cam_ee}")

    print(f"\nSaved result to: {output_path}")


if __name__ == "__main__":
    main()