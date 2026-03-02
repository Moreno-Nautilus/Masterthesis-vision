from __future__ import annotations

import time

import rclpy

from src.perception.ros.multicam_grabber import MultiCamGrabber, CameraTopics
from src.perception.pipeline import GraspPerceptionPipeline, PipelineConfig
from src.perception.pipeline_multiview import MultiViewRunner, MultiViewConfig
from src.perception.pose_icp import load_cad_as_pointcloud
from src.perception.segmentation import remove_plane_ransac
from src.perception.viz_plotly import save_scene_plotly_html
from src.calibration.io_extrinsics import load_extrinsics_yaml


CAMERAS = [
    CameraTopics(
        cam_id="zed2i_1",
        depth_topic="/zed2i_1/zed_node/depth/depth_registered",
        info_topic="/zed2i_1/zed_node/left/camera_info",
    ),
    CameraTopics(
        cam_id="zed2i_2",
        depth_topic="/zed2i_2/zed_node/depth/depth_registered",
        info_topic="/zed2i_2/zed_node/left/camera_info",
    ),
    # CameraTopics(
    #     cam_id="zed2i_3",
    #     depth_topic="/zed2i_3/zed_node/depth/depth_registered",
    #     info_topic="/zed2i_3/zed_node/left/camera_info",
    # ),
    CameraTopics(
        cam_id="zedmini_wrist",
        depth_topic="/zedmini_wrist/zed_node/depth/depth_registered",
        info_topic="/zedmini_wrist/zed_node/left/camera_info",
    ),
]


def main() -> None:
    # 1) init ROS
    rclpy.init()

    # 2) build perception pipeline
    cad_library = {
        "cube": load_cad_as_pointcloud("Data/CAD_Models/Cube.stl", scale=0.01, center=True),
        "cat": load_cad_as_pointcloud("Data/CAD_Models/Cat.stl", scale=0.007, center=True),
        "dolphin": load_cad_as_pointcloud("Data/CAD_Models/dolphin.stl", scale=0.001, center=True),
        "hand": load_cad_as_pointcloud("Data/CAD_Models/Hand.stl", scale=0.01, center=True),
    }

    pipe_cfg = PipelineConfig(
        plane_distance_threshold=0.003,
        dbscan_eps=0.02,
        dbscan_min_points=30,
        voxel_size=0.005,
        max_rms_nn=0.01,
        min_margin=1.2,
    )
    pipe = GraspPerceptionPipeline(cad_library=cad_library, cfg=pipe_cfg)

    mv_cfg = MultiViewConfig(
        voxel_size_fusion=0.005,
        stride=2,
        zmin=0.15,
        zmax=2.0,
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

        # 4) run fusion + pipeline
        result = runner.run(views)

        # 5) quick debug export
        
        wo_plane = result.points_wo_plane
        _, plane_pts, _ = remove_plane_ransac(wo_plane)

        save_scene_plotly_html(
            scene=wo_plane,
            plane=plane_pts,
            clusters=result.clusters,
            objects_world=None,
            out_path="/tmp/multiview_debug.html",
        )
        print("Saved /tmp/multiview_debug.html")
        print("detections:", [(o.object_id, o.id_confidence, o.pose_confidence) for o in result.objects])

    finally:
        grabber.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()