from __future__ import annotations
import numpy as np
import open3d as o3d

def remove_plane_ransac(
    points: np.ndarray,
    distance_threshold: float = 0.003,  # 3mm
    ransac_n: int = 3,
    num_iterations: int = 2000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Removes the dominant plane via RANSAC.

    Returns:
      points_wo_plane: (K,3)
      plane_points: (L,3)
      plane_model: (4,) coefficients [a,b,c,d] for ax+by+cz+d=0
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
    )

    inlier_mask = np.zeros(len(points), dtype=bool)
    inlier_mask[inliers] = True

    plane_pts = points[inlier_mask]
    wo_plane = points[~inlier_mask]

    return wo_plane, plane_pts, np.asarray(plane_model, dtype=float)

def cluster_dbscan(
    points: np.ndarray,
    eps: float = 0.03,          # meters (start with 1 cm)
    min_points: int = 20,
) -> list[np.ndarray]:
    """
    DBSCAN clustering on a point cloud.
    Returns list of clusters, each (Ni, 3).
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    if labels.size == 0:
        return []

    clusters = []
    for lab in sorted(set(labels)):
        if lab == -1:
            continue  # noise
        cluster_pts = points[labels == lab]
        clusters.append(cluster_pts)

    # sort big-to-small
    clusters.sort(key=lambda c: c.shape[0], reverse=True)
    return clusters


def merge_close_clusters(
    clusters: list[np.ndarray],
    merge_dist: float = 0.05,   # 5 cm
    z_overlap: float = 0.03,    # 3 cm overlap tolerance
) -> list[np.ndarray]:
    """
    Greedy merge of cluster fragments based on centroid distance + z overlap.
    Helps when one physical object is split into multiple DBSCAN clusters.
    """
    if len(clusters) <= 1:
        return clusters

    # Precompute stats
    cents = [c.mean(axis=0) for c in clusters]
    zmins = [float(c[:, 2].min()) for c in clusters]
    zmaxs = [float(c[:, 2].max()) for c in clusters]

    used = [False] * len(clusters)
    merged: list[np.ndarray] = []

    for i in range(len(clusters)):
        if used[i]:
            continue
        used[i] = True
        acc = [clusters[i]]

        changed = True
        while changed:
            changed = False
            acc_pts = np.vstack(acc)
            acc_cent = acc_pts.mean(axis=0)
            acc_zmin = float(acc_pts[:, 2].min())
            acc_zmax = float(acc_pts[:, 2].max())

            for j in range(len(clusters)):
                if used[j]:
                    continue
                d = float(np.linalg.norm(cents[j] - acc_cent))
                # check z overlap-ish
                overlap_ok = not (zmaxs[j] < acc_zmin - z_overlap or zmins[j] > acc_zmax + z_overlap)
                if d < merge_dist and overlap_ok:
                    used[j] = True
                    acc.append(clusters[j])
                    changed = True

        merged.append(np.vstack(acc))

    merged.sort(key=lambda c: c.shape[0], reverse=True)
    return merged