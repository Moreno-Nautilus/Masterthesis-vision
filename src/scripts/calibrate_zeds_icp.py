from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import rclpy

from src.calibration.extrinsics_icp import estimate_extrinsic_icp
from src.calibration.io_extrinsics import save_extrinsics_yaml
from src.utils.se3 import SE3
from src.perception.backproject import depth_to_points_cam
from src.perception.ros.multicam_grabber import MultiCamGrabber, CameraTopics


CAMERAS_2ZED = [
    CameraTopics(
        cam_id="zed2i_1",
        depth_topic="/zed2i_1/zed_node/depth/depth_registered",
        info_topic="/zed2i_1/zed_node/depth/depth_registered/camera_info"

    ),
    CameraTopics(
        cam_id="zed2i_2",
        depth_topic="/zed2i_2/zed_node/depth/depth_registered",
        info_topic="/zed2i_2/zed_node/depth/depth_registered/camera_info"
    ),
]


def _points_from_view(depth: np.ndarray, K: np.ndarray, stride: int, zmin: float, zmax: float) -> np.ndarray:
    pts = depth_to_points_cam(depth, K, stride=stride, zmin=zmin, zmax=zmax)
    pts = np.asarray(pts, dtype=np.float32)
    # remove NaN just in case
    finite = np.isfinite(pts).all(axis=1)
    pts = pts[finite]
    return pts


def main() -> None:
    # ---- params (keep them here for now; later expose argparse) ----
    out_yaml = Path("config/camera_extrinsics.yaml")   # match your run_multiview_ros.py
    stride = 2
    zmin = 0.15
    zmax = 2.0
    voxel = 0.01
    sync_slop_s = 0.20
    timeout_s = 60.0

    # --------------------------------------------------------------
    rclpy.init()
    grabber = MultiCamGrabber(
        cameras=CAMERAS_2ZED,
        sync_slop_s=sync_slop_s,
        use_best_effort_if_unsynced=True,
        static_extrinsics_base_cam=None,  # none for calibration
    )

    try:
        print("[calib] Waiting for synced depth frames...")
        views = None
        t0 = time.time()
        while views is None:
            rclpy.spin_once(grabber, timeout_sec=0.1)
            views = grabber.get_latest_views()
            if time.time() - t0 > timeout_s:
                raise RuntimeError(
                    f"[calib] Timeout after {timeout_s}s waiting for synced frames. "
                    f"Check that both ZED nodes are running and topics match."
                )

        # Expect exactly 2 views: cam1 then cam2 (same order as CAMERAS_2ZED)
        v1, v2 = views[0], views[1]
        print(f"[calib] Got views: {(v1.cam_id, v1.stamp_s):} and {(v2.cam_id, v2.stamp_s):}")

        # Backproject to point clouds in each camera frame
        pts1 = _points_from_view(v1.depth, v1.K, stride=stride, zmin=zmin, zmax=zmax)
        pts2 = _points_from_view(v2.depth, v2.K, stride=stride, zmin=zmin, zmax=zmax)

        print(f"[calib] points: {v1.cam_id}={len(pts1)}  {v2.cam_id}={len(pts2)}")
        if len(pts1) < 5000 or len(pts2) < 5000:
            print("[calib][warn] few points; consider stride=1 or increase zmax, or improve depth scene.")

        # Estimate T_cam1_cam2 (maps cam2 -> cam1)
        T_cam1_cam2, metrics = estimate_extrinsic_icp(
            pts_cam1=pts1,
            pts_cam2=pts2,
            voxel_size=voxel,
        )

        print("[calib] metrics:", metrics)
        print("[calib] T_cam1_cam2:\n", T_cam1_cam2.as_matrix())

        # Save as T_base_cam, choosing base = cam1
        extr = {
            v1.cam_id: SE3.identity(),
            v2.cam_id: T_cam1_cam2,
        }
        save_extrinsics_yaml(out_yaml, extr)
        print(f"[calib] Saved extrinsics to: {out_yaml.resolve()}")

        print("\n[calib] Next: run your multiview pipeline using that YAML.")
        print("        If fusion looks offset, re-run with more clutter / overlap, or voxel=0.02.")

    finally:
        grabber.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()