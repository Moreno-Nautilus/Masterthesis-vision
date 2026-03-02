from __future__ import annotations
import numpy as np
import open3d as o3d

from src.utils.se3 import SE3
from src.utils.geometry import random_rot_matrix
from src.perception.pose_icp import load_cad_as_pointcloud
from src.perception.sim_scene import make_synthetic_scene_single_object
from src.perception.pipeline import GraspPerceptionPipeline, PipelineConfig
from src.perception.segmentation import remove_plane_ransac
from src.perception.viz import save_scene_png
from src.perception.sim_scene import make_synthetic_scene_single_object_robust
from src.perception.viz_plotly import save_scene_plotly_html
from src.perception.sim_scene import make_synthetic_scene_multi_object_robust


def random_pose():
    R = random_rot_matrix()
    t = np.array([0.1, 0.05, 0.08])  # above table
    return SE3(R, t)


def save_scene_debug(scene: np.ndarray, clusters: list[np.ndarray], out_dir: str = "/tmp"):
    o3d.io.write_point_cloud(f"{out_dir}/scene.ply", o3d.geometry.PointCloud(o3d.utility.Vector3dVector(scene)))
    for k, c in enumerate(clusters):
        o3d.io.write_point_cloud(f"{out_dir}/cluster_{k}.ply", o3d.geometry.PointCloud(o3d.utility.Vector3dVector(c)))
    print(f"Wrote debug PLYs to {out_dir}/scene.ply and {out_dir}/cluster_*.ply")

def fixed_pose(tx, ty, tz, seed=0):
    # deterministic-ish random rotation per object
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(3,3))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:,0] *= -1
    return SE3(Q, np.array([tx, ty, tz], dtype=float))

poses_gt = {
    "cube": fixed_pose(0.10, 0.05, 0.08, seed=1),
    "cat": fixed_pose(0.25, -0.2, 0.08, seed=2),
    "dolphin": fixed_pose(-0.05, 0.12, 0.08, seed=3),
    "hand": fixed_pose(-0.2, -0.1, 0.08, seed=4),
}

def main():
    # CAD library (object frame, centered)
    cad_library = {
        "cube": load_cad_as_pointcloud("Data/CAD_Models/Cube.stl", scale=0.01, center=True),
        "cat": load_cad_as_pointcloud("Data/CAD_Models/Cat.stl", scale=0.007, center=True),
        "dolphin": load_cad_as_pointcloud("Data/CAD_Models/dolphin.stl", scale=0.001, center=True),
        "hand": load_cad_as_pointcloud("Data/CAD_Models/Hand.stl", scale=0.01, center=True),
    }

    cfg = PipelineConfig(
        plane_distance_threshold=0.003,
        dbscan_eps=0.02,
        dbscan_min_points=30,
        voxel_size=0.005,
        max_rms=0.01,
        min_margin=1.2,
    )
    pipe = GraspPerceptionPipeline(cad_library=cad_library, cfg=cfg)

    pose_gt = random_pose()
    scene, observed_objs = make_synthetic_scene_multi_object_robust(
        cad_library=cad_library,
        poses_gt=poses_gt,
        table_z=0.0,
        obj_noise_std=0.002,
        plane_noise_std=0.001,
        occlusion_keep_ratio=0.35,  # hard
        n_outliers=800,
        seed=0,
    )
    print({k: len(v) for k, v in observed_objs.items()})

    result = pipe.run(scene)
    # Recompute plane points for visualization only
    wo_plane, plane_pts, plane_model = remove_plane_ransac(scene)

    # Build transformed CADs for each detection
    objects_world = []
    for obj in result.objects:
        cad_pts = cad_library[obj.object_id]
        cad_world = obj.T_object_to_world.transform_points(cad_pts)
        objects_world.append((obj.object_id, cad_world))

    save_scene_png(
        scene=scene,
        plane=plane_pts,
        clusters=result.clusters,
        objects_world=objects_world,
        out_path="/home/moreno/MasterThesis/pipeline_debug.png",
    )
    save_scene_plotly_html(
        scene=scene,
        plane=plane_pts,
        clusters=result.clusters,
        objects_world=objects_world,
        out_path="/home/moreno/MasterThesis/pipeline_debug.html",
    )
    print("\n=== DETECTIONS ===")
    for obj in result.objects:
        print(
            f"- id={obj.object_id} id_conf={obj.id_confidence:.2f} pose_conf={obj.pose_confidence:.2f} "
            f"rms={obj.metrics['rms']:.4f} margin={obj.metrics['margin']:.2f}"
        )
        T = obj.T_object_to_world.as_matrix()
        print("  T_object_to_world:\n", np.array_str(T, precision=3, suppress_small=True))

    save_scene_debug(scene, result.clusters, out_dir="/tmp")


if __name__ == "__main__":
    main()