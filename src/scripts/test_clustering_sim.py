# src/scripts/test_clustering_sim.py
from __future__ import annotations

import numpy as np
import open3d as o3d
from pathlib import Path

from src.utils.geometry import random_rot_matrix
from src.utils.se3 import SE3
from src.perception.pose_icp import load_cad_as_pointcloud, estimate_pose_icp
from src.perception.sim_scene import generate_plane_points
from src.perception.segmentation import remove_plane_ransac, cluster_dbscan


CAD_DIR = Path("Data/CAD_Models")

CAD_PATHS = [
    CAD_DIR / "Cube.stl",
    CAD_DIR / "Hand.stl",
    CAD_DIR / "Cat.stl",
    CAD_DIR / "dolphin.stl",
]

CAD_SCALES = {
    "Cube.stl": 0.01,
    "Hand.stl": 0.01,
    "Cat.stl": 0.007,
    "dolphin.stl": 0.001,
}

# Scene generation
TABLE_POINTS = 30000
OBJ_NOISE_STD = 0.002    # 2mm
PLANE_NOISE_STD = 0.001  # 1mm
TABLE_XLIM = (-0.6, 0.6)
TABLE_YLIM = (-0.6, 0.6)
TABLE_Z = 0.0

# RANSAC plane removal
PLANE_DIST_THRESH = 0.003  # 3mm
PLANE_ITERS = 2000

# Clustering
DBSCAN_EPS = 0.02
DBSCAN_MIN_POINTS = 30

# Placement (to avoid merged clusters)
MIN_SEP_XY = 0.35  # meters (increase if DBSCAN merges objects)

def save_interactive_html(points_remaining: np.ndarray, clusters: list[np.ndarray], out_path="clusters_debug.html"):
    import plotly.graph_objects as go

    fig = go.Figure()

    # remaining (faint)
    fig.add_trace(go.Scatter3d(
        x=points_remaining[:, 0],
        y=points_remaining[:, 1],
        z=points_remaining[:, 2],
        mode="markers",
        marker=dict(size=1, opacity=0.05),
        name="remaining"
    ))

    # clusters
    for i, c in enumerate(clusters):
        fig.add_trace(go.Scatter3d(
            x=c[:, 0], y=c[:, 1], z=c[:, 2],
            mode="markers",
            marker=dict(size=2),
            name=f"cluster {i} ({len(c)})"
        ))

    fig.update_layout(
        title="Clusters after plane removal (interactive)",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(itemsizing="constant"),
    )

    fig.write_html(out_path, include_plotlyjs="cdn")
    print("Saved interactive HTML:", out_path)

def random_pose_xy(z_min=0.03, z_max=0.12) -> SE3:
    R = random_rot_matrix()
    t = np.array([
        np.random.uniform(-0.35, 0.35),
        np.random.uniform(-0.35, 0.35),
        np.random.uniform(z_min, z_max),
    ])
    return SE3(R, t)


def sample_translation_far_enough(existing_xy: list[np.ndarray], min_sep: float) -> np.ndarray:
    for _ in range(500):
        xy = np.array([
            np.random.uniform(-0.35, 0.35),
            np.random.uniform(-0.35, 0.35),
        ])
        if all(np.linalg.norm(xy - e) >= min_sep for e in existing_xy):
            return xy
    # fallback if table is too crowded
    return np.array([np.random.uniform(-0.35, 0.35), np.random.uniform(-0.35, 0.35)])


def make_scene_all_objects(cad_list: list[np.ndarray]) -> tuple[np.ndarray, list[SE3]]:
    plane = generate_plane_points(
        n=TABLE_POINTS,
        xlim=TABLE_XLIM,
        ylim=TABLE_YLIM,
        z=TABLE_Z,
        noise_std=PLANE_NOISE_STD,
    )

    poses: list[SE3] = []
    existing_xy: list[np.ndarray] = []
    obs_all = []

    for cad in cad_list:
        R = random_rot_matrix()
        xy = sample_translation_far_enough(existing_xy, MIN_SEP_XY)
        existing_xy.append(xy)
        z = np.random.uniform(0.04, 0.14)  # keep above plane
        t = np.array([xy[0], xy[1], z])
        T = SE3(R, t)
        poses.append(T)

        obs = T.transform_points(cad) + np.random.normal(0, OBJ_NOISE_STD, cad.shape)
        obs_all.append(obs)

    scene = np.vstack([plane] + obs_all)
    return scene, poses


def nn_rms(source_pts: np.ndarray, target_pts: np.ndarray) -> float:
    if len(source_pts) == 0 or len(target_pts) == 0:
        return float("inf")
    src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source_pts))
    tgt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target_pts))
    dists = np.asarray(src.compute_point_cloud_distance(tgt))
    if not np.isfinite(dists).all():
        return float("inf")
    return float(np.sqrt(np.mean(dists**2)))


def save_clusters_png(scene_wo_plane: np.ndarray, clusters: list[np.ndarray], out_path="clusters_debug.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def ds(pts, n=8000):
        if len(pts) <= n:
            return pts
        idx = np.random.choice(len(pts), n, replace=False)
        return pts[idx]

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")

    S = ds(scene_wo_plane, 12000)
    ax.scatter(S[:, 0], S[:, 1], S[:, 2], s=1, c="k", alpha=0.03, label="remaining")

    colors = ["r", "g", "b", "c", "m", "y"]
    for i, cpts in enumerate(clusters[:6]):
        C = ds(cpts, 9000)
        ax.scatter(C[:, 0], C[:, 1], C[:, 2], s=2, c=colors[i % len(colors)], label=f"cluster {i} ({len(cpts)})")

    ax.set_title("DBSCAN clusters after plane removal (all CADs in scene)")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print("Saved:", out_path)


def main():
    # ---- Load CAD models
    cad_models = []
    for p in CAD_PATHS:
        if not p.exists():
            raise FileNotFoundError(f"Missing CAD file: {p}")
        scale = CAD_SCALES.get(p.name, 0.01)

        pts = load_cad_as_pointcloud(p, scale=scale, center=True)
        extent = pts.max(axis=0) - pts.min(axis=0)

        # reject degenerate models
        if float(np.min(extent)) < 1e-4:
            print(f"[WARN] {p.name} degenerate extent={extent}, skipping.")
            continue

        print(f"{p.name}: scale={scale} extent={extent} diag={np.linalg.norm(extent):.3f} m")
        cad_models.append({"name": p.stem, "path": str(p), "pts": pts})

    if len(cad_models) < 2:
        print("Need at least 2 valid CADs.")
        return

    # ---- Build scene with ALL CADs
    scene, poses_gt = make_scene_all_objects([c["pts"] for c in cad_models])

    # ---- Plane removal
    wo_plane, plane_pts, plane_model = remove_plane_ransac(
        scene,
        distance_threshold=PLANE_DIST_THRESH,
        num_iterations=PLANE_ITERS,
    )
    print("plane_model:", plane_model)
    print("remaining after plane:", len(wo_plane))

    # ---- Cluster
    clusters = cluster_dbscan(wo_plane, eps=DBSCAN_EPS, min_points=DBSCAN_MIN_POINTS)
    print("num clusters:", len(clusters))
    for i, c in enumerate(clusters[:10]):
        print(f"  cluster {i}: {len(c)} points")

    # Save debug artifacts
    o3d.io.write_point_cloud("/tmp/wo_plane.ply", o3d.geometry.PointCloud(o3d.utility.Vector3dVector(wo_plane)))
    for i, c in enumerate(clusters[:10]):
        o3d.io.write_point_cloud(f"/tmp/cluster_{i}.ply", o3d.geometry.PointCloud(o3d.utility.Vector3dVector(c)))
    print("Wrote /tmp/wo_plane.ply and /tmp/cluster_*.ply")
    save_clusters_png(wo_plane, clusters, out_path="clusters_debug.png")

    if len(clusters) == 0:
        print("No clusters found. Try increasing DBSCAN_EPS or lowering DBSCAN_MIN_POINTS.")
        return

    save_interactive_html(wo_plane, clusters, out_path="clusters_debug.html")
    # ---- Score matrix
    scores = np.full((len(clusters), len(cad_models)), np.inf, dtype=float)
    poses = [[None for _ in cad_models] for _ in clusters]

    for i, cluster in enumerate(clusters):
        for j, cad in enumerate(cad_models):
            pose_obs_to_cad = estimate_pose_icp(cluster, cad["pts"])
            aligned = pose_obs_to_cad.transform_points(cluster)
            rms = nn_rms(aligned, cad["pts"])
            scores[i, j] = rms
            poses[i][j] = pose_obs_to_cad
            print(f"Cluster {i} vs CAD {cad['name']}: ICP NN RMS [m] = {rms:.6f}")

    print("\nScore matrix (NN RMS in meters): rows=clusters, cols=CADs")
    print("CADs:", [c["name"] for c in cad_models])
    print(np.round(scores, 6))

    MAX_RMS = 0.01      # meters
    MIN_MARGIN = 1.5    # second_best / best

    used_cads = set()
    assignments = []
    rejected = []

    for i in range(len(clusters)):
        row = scores[i].copy()
        order = np.argsort(row)

        best_j = int(order[0])
        best = float(row[best_j])

        second = float(row[int(order[1])]) if len(order) > 1 else float("inf")
        margin = (second / best) if best > 1e-12 else float("inf")

        print(f"Cluster {i}: best={cad_models[best_j]['name']} rms={best:.6f}, "
            f"second={second:.6f}, margin={margin:.2f}")

        # gate: reject ambiguous or bad fits
        if not np.isfinite(best) or best > MAX_RMS or margin < MIN_MARGIN:
            rejected.append((i, best_j, best, second, margin))
            continue

        # enforce one-to-one CAD assignment (optional)
        chosen = None
        for j in order:
            j = int(j)
            if j in used_cads:
                continue
            # if you want, you can recompute margin for the chosen one,
            # but usually best is fine here.
            chosen = j
            break

        if chosen is None:
            rejected.append((i, best_j, best, second, margin))
            continue

        used_cads.add(chosen)
        assignments.append((i, chosen, float(scores[i, chosen])))

    print("\nAccepted assignments:")
    for ci, cj, sc in assignments:
        print(f"  cluster {ci} -> {cad_models[cj]['name']} (RMS={sc:.6f} m)")

    print("\nRejected clusters (ambiguous or bad fit):")
    for (ci, best_j, best, second, margin) in rejected:
        print(f"  cluster {ci}: best={cad_models[best_j]['name']} rms={best:.6f}, "
            f"second={second:.6f}, margin={margin:.2f}")

        print("\nAssignments (cluster -> CAD):")
        for ci, cj, sc in assignments:
            print(f"  cluster {ci} -> {cad_models[cj]['name']}  (RMS={sc:.6f} m)")

    # ---- Save aligned clouds for inspection
    for ci, cj, sc in assignments:
        pose_obs_to_cad = poses[ci][cj]
        aligned = pose_obs_to_cad.transform_points(clusters[ci])
        o3d.io.write_point_cloud(
            f"/tmp/aligned_cluster_{ci}_to_{cad_models[cj]['name']}.ply",
            o3d.geometry.PointCloud(o3d.utility.Vector3dVector(aligned)),
        )

    print("Wrote /tmp/aligned_cluster_*_to_*.ply")


if __name__ == "__main__":
    main()