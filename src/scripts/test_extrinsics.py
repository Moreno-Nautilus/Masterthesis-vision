from __future__ import annotations

import numpy as np
import rclpy
import open3d as o3d

from src.perception.ros.multicam_grabber import MultiCamGrabber, CameraTopics
from src.calibration.io_extrinsics import load_extrinsics_yaml
from src.perception.backproject import depth_to_points_cam
from src.perception.viz_plotly import save_scene_plotly_html


CAMERAS = [
    CameraTopics(
        cam_id="zed2i_1",
        depth_topic="/zed2i_1/zed_node/depth/depth_registered",
        info_topic="/zed2i_1/zed_node/depth/depth_registered/camera_info",
    ),
    CameraTopics(
        cam_id="zed2i_2",
        depth_topic="/zed2i_2/zed_node/depth/depth_registered",
        info_topic="/zed2i_2/zed_node/depth/depth_registered/camera_info",
    ),
]


def _fit_plane(points: np.ndarray, dist=0.004, iters=2000):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points.astype(np.float64)))
    model, inliers = pcd.segment_plane(distance_threshold=dist, ransac_n=3, num_iterations=iters)
    a, b, c, d = [float(x) for x in model]
    n = np.array([a, b, c], dtype=np.float64)
    nn = np.linalg.norm(n) + 1e-12
    n = n / nn
    d = d / nn
    return n, d, np.asarray(inliers, dtype=int)


def _angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    u = u / (np.linalg.norm(u) + 1e-12)
    v = v / (np.linalg.norm(v) + 1e-12)
    c = float(np.clip(np.dot(u, v), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def main():
    rclpy.init()

    T_map = load_extrinsics_yaml("config/camera_extrinsics.yaml")
    grabber = MultiCamGrabber(
        cameras=CAMERAS,
        sync_slop_s=0.05,
        use_best_effort_if_unsynced=False,
        static_extrinsics_base_cam=T_map,
    )

    try:
        print("Waiting for synced views...")
        views = None
        for _ in range(250):
            rclpy.spin_once(grabber, timeout_sec=0.05)
            views = grabber.get_latest_views()
            if views is not None:
                break
        if views is None:
            raise RuntimeError("No synced views.")

        per_cam = {}
        for v in views:
            # IMPORTANT: tighten depth range here to bias toward the table/board
            pts_cam = depth_to_points_cam(v.depth, v.K, stride=3, zmin=0.25, zmax=1.4)
            pts_base = v.T_base_cam.transform_points(pts_cam)

            # OPTIONAL: very crude spatial crop around cam1 forward region (tune later)
            # This removes far wall points even before plane fit.
            m = (
                (pts_base[:, 2] > 0.25) & (pts_base[:, 2] < 1.4) &
                (pts_base[:, 0] > -0.8) & (pts_base[:, 0] < 0.8) &
                (pts_base[:, 1] > -0.8) & (pts_base[:, 1] < 0.8)
            )
            pts_base = pts_base[m]

            print(v.cam_id, "pts_base:", pts_base.shape[0])
            n, d, inl = _fit_plane(pts_base, dist=0.004, iters=2500)

            # signed distances
            h = pts_base @ n + d
            print(v.cam_id, "plane n=", np.round(n, 3), "d=", round(d, 4))
            print(v.cam_id, "h median/95%:", float(np.median(h)), float(np.quantile(h, 0.95)))
            print(v.cam_id, "plane inliers:", len(inl))

            per_cam[v.cam_id] = dict(pts=pts_base, n=n, d=d, inliers=inl)

        # Compare plane normals
        ids = list(per_cam.keys())
        if len(ids) >= 2:
            n1 = per_cam[ids[0]]["n"]
            n2 = per_cam[ids[1]]["n"]

            # allow sign flip
            ang = min(_angle_deg(n1, n2), _angle_deg(n1, -n2))
            print("\nPlane normal agreement angle [deg]:", ang)

            # Compare plane heights at origin (distance from base origin to plane)
            # plane: n^T x + d = 0 -> signed dist of origin = d
            print("Plane origin signed dist d:", ids[0], per_cam[ids[0]]["d"], "|", ids[1], per_cam[ids[1]]["d"])

        # Visualize two sets + their plane inliers as separate clusters
        clusters = []
        for cam_id in ids:
            pts = per_cam[cam_id]["pts"]
            inl = per_cam[cam_id]["inliers"]
            clusters.append(pts[::20])              # downsample for viz
            clusters.append(pts[inl][::20])         # plane inliers

        out = "/home/moreno/MasterThesis/extrinsics_plane_align.html"
        save_scene_plotly_html(
            scene=np.zeros((0, 3), dtype=float),
            plane=None,
            clusters=clusters,
            objects_world=None,
            out_path=out,
        )
        print("\nWrote:", out)
        print("Expected: both plane-inlier sets should lie on the same plane in base frame.")
        print("If they form clearly different planes -> extrinsics wrong (often inversion or axis mismatch).")

    finally:
        grabber.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()