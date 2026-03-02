import numpy as np
import open3d as o3d

from src.utils.geometry import random_rot_matrix
from src.utils.se3 import SE3
from src.perception.pose_icp import load_cad_as_pointcloud
from src.perception.sim_scene import make_synthetic_scene_single_object
from src.perception.segmentation import remove_plane_ransac

def random_pose():
    R = random_rot_matrix()
    t = np.array([0.1, 0.05, 0.08])  # keep above table for now
    return SE3(R, t)

def save_debug_png(points_scene, points_wo_plane, plane_points, out_path="plane_ransac_debug.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def ds(pts, n=6000):
        if len(pts) <= n:
            return pts
        idx = np.random.choice(len(pts), n, replace=False)
        return pts[idx]

    S = ds(points_scene)
    O = ds(points_wo_plane)
    P = ds(plane_points)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(S[:,0], S[:,1], S[:,2], s=1, c="k", label="scene")
    ax.scatter(P[:,0], P[:,1], P[:,2], s=2, c="g", label="plane")
    ax.scatter(O[:,0], O[:,1], O[:,2], s=2, c="r", label="scene w/o plane")
    ax.legend()
    ax.set_title("RANSAC plane removal")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print("Saved:", out_path)

def main():
    cad = load_cad_as_pointcloud("Data/CAD_Models/Cube.stl", scale=0.01, center=True)

    pose_gt = random_pose()
    scene, observed_obj = make_synthetic_scene_single_object(
        cad_points=cad,
        pose_gt=pose_gt,
        table_z=0.0,
    )

    wo_plane, plane_pts, plane_model = remove_plane_ransac(
        scene,
        distance_threshold=0.003,
        num_iterations=2000,
    )

    print("plane_model [a,b,c,d]:", plane_model)
    print("scene points:", len(scene))
    print("plane points:", len(plane_pts))
    print("remaining points:", len(wo_plane))

    # Save PLY for inspection + PNG for quick look
    p_scene = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(scene))
    p_plane = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(plane_pts))
    p_wo = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(wo_plane))
    o3d.io.write_point_cloud("/tmp/scene.ply", p_scene)
    o3d.io.write_point_cloud("/tmp/plane.ply", p_plane)
    o3d.io.write_point_cloud("/tmp/wo_plane.ply", p_wo)
    print("Wrote: /tmp/scene.ply /tmp/plane.ply /tmp/wo_plane.ply")

    save_debug_png(scene, wo_plane, plane_pts, out_path="plane_ransac_debug.png")

if __name__ == "__main__":
    main()