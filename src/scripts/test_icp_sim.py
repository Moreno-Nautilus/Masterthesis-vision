from __future__ import annotations
import numpy as np
import open3d as o3d
from src.utils.se3 import SE3
from src.utils.geometry import random_rot_matrix, rotation_error_deg
from src.perception.pose_icp import load_cad_as_pointcloud, estimate_pose_icp
import matplotlib
import matplotlib.pyplot as plt

def random_pose():
    R = random_rot_matrix()
    t = np.random.uniform(-0.1, 0.1, 3)  # m
    return SE3(R, t)

def save_icp_debug_image(cad: np.ndarray, observed: np.ndarray, aligned: np.ndarray, out_path="/tmp/icp_debug.png"):
    """
    Save a debug image showing:
      Green = CAD
      Red   = observed
      Blue  = aligned(observed)

    Tries Open3D offscreen rendering; if that fails (common on headless),
    falls back to matplotlib 3D scatter and saves a PNG.
    """
    try:
        import open3d as o3d
        from open3d.visualization import rendering

        # Create point clouds
        cad_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cad))
        obs_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(observed))
        ali_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(aligned))

        cad_pcd.paint_uniform_color([0, 1, 0])  # green
        obs_pcd.paint_uniform_color([1, 0, 0])  # red
        ali_pcd.paint_uniform_color([0, 0, 1])  # blue

        w, h = 1280, 720
        renderer = rendering.OffscreenRenderer(w, h)
        scene = renderer.scene
        scene.set_background([1, 1, 1, 1])  # white

        mat = rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        mat.point_size = 3.0

        scene.add_geometry("cad", cad_pcd, mat)
        scene.add_geometry("obs", obs_pcd, mat)
        scene.add_geometry("ali", ali_pcd, mat)

        # Camera: look at the CAD center
        center = cad.mean(axis=0)
        extent = (cad.max(axis=0) - cad.min(axis=0))
        diag = float(np.linalg.norm(extent))
        eye = center + np.array([0.8 * diag, 0.6 * diag, 0.6 * diag])
        up = np.array([0, 0, 1], dtype=float)

        scene.camera.look_at(center, eye, up)

        img = renderer.render_to_image()
        o3d.io.write_image(out_path, img)
        print(f"Saved Open3D offscreen render: {out_path}")
        return
    except Exception as e:
        print(f"Open3D offscreen render failed, falling back to matplotlib. Reason: {e}")

    # downsample for faster plotting if needed
    def _ds(pts, n=5000):
        if pts.shape[0] <= n:
            return pts
        idx = np.random.choice(pts.shape[0], n, replace=False)
        return pts[idx]

    cad_ds = _ds(cad, 6000)
    obs_ds = _ds(observed, 6000)
    ali_ds = _ds(aligned, 6000)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(cad_ds[:, 0], cad_ds[:, 1], cad_ds[:, 2], s=1, c="g", label="CAD")
    ax.scatter(obs_ds[:, 0], obs_ds[:, 1], obs_ds[:, 2], s=1, c="r", label="Observed")
    ax.scatter(ali_ds[:, 0], ali_ds[:, 1], ali_ds[:, 2], s=1, c="b", label="Aligned")

    ax.set_title("ICP debug: Green=CAD, Red=Observed, Blue=Aligned(Observed)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend(loc="upper right")

    # set equalish axis scaling
    all_pts = np.vstack([cad_ds, obs_ds, ali_ds])
    mins = all_pts.min(axis=0)
    maxs = all_pts.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = np.max(maxs - mins) / 2.0
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)

    # choose a viewpoint
    ax.view_init(elev=20, azim=35)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved matplotlib render: {out_path}")


def main():
    # Adjust scale depending on STL units:
    # - if STL is in cm: scale=0.01
    # - if STL is in mm: scale=0.001
    cad = load_cad_as_pointcloud("Data/CAD_Models/Cube.stl", scale=0.01)

    mins = cad.min(axis=0)
    maxs = cad.max(axis=0)
    extent = maxs - mins
    print("CAD extent:", extent, " (units)")
    print("CAD diag:", np.linalg.norm(extent))

    pose_gt = random_pose() 
    observed = pose_gt.transform_points(cad)

    # add noise (2mm)
    observed = observed + np.random.normal(0, 0.002, observed.shape)

    # ICP returns T_obs_to_cad
    pose_est = estimate_pose_icp(observed, cad)

    # Convert to T_cad_to_obs for direct comparison with pose_gt
    pose_est_cad_to_obs = pose_est.inverse()

    t_err = np.linalg.norm(pose_est_cad_to_obs.t - pose_gt.t)
    r_err = rotation_error_deg(pose_est_cad_to_obs.R, pose_gt.R)
    print("t_err [m]:", t_err)
    print("r_err [deg]:", r_err)

    T_err = pose_est.compose(pose_gt)
    print("compose check |t| [m]:", float(np.linalg.norm(T_err.t)))
    print("compose check rot [deg]:", rotation_error_deg(T_err.R, np.eye(3)))

    # NN RMS alignment error 
    aligned = pose_est.transform_points(observed)  
    save_icp_debug_image(cad, observed, aligned, out_path="/home/moreno/MasterThesis/icp_debug.png")
    src = o3d.geometry.PointCloud()
    src.points = o3d.utility.Vector3dVector(aligned)

    tgt = o3d.geometry.PointCloud()
    tgt.points = o3d.utility.Vector3dVector(cad)

    dists = np.asarray(src.compute_point_cloud_distance(tgt))
    print("NN RMS [m]:", float(np.sqrt(np.mean(dists**2))))

    out_dir = "/tmp"
    o3d.io.write_point_cloud(f"{out_dir}/cad.ply", tgt)
    obs_pcd = o3d.geometry.PointCloud()
    obs_pcd.points = o3d.utility.Vector3dVector(observed)
    o3d.io.write_point_cloud(f"{out_dir}/observed.ply", obs_pcd)
    o3d.io.write_point_cloud(f"{out_dir}/aligned.ply", src)
    print(f"Wrote: {out_dir}/cad.ply {out_dir}/observed.ply {out_dir}/aligned.ply")


if __name__ == "__main__":
    main()