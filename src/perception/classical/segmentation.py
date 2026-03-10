from __future__ import annotations

import numpy as np
import open3d as o3d


def _pcd(points: np.ndarray) -> o3d.geometry.PointCloud:
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=float))
    return p


def remove_plane_ransac(
    points: np.ndarray,
    distance_threshold: float = 0.003,
    ransac_n: int = 3,
    num_iterations: int = 2000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Dominant plane removal (kept for reference / fallback).
    Returns: (wo_plane, plane_pts, plane_model[a,b,c,d])
    """
    pcd = _pcd(points)
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
    )

    mask = np.zeros(len(points), dtype=bool)
    mask[inliers] = True
    plane_pts = points[mask]
    wo_plane = points[~mask]
    return wo_plane, plane_pts, np.asarray(plane_model, dtype=float)


def remove_horizontal_plane_ransac(
    points: np.ndarray,
    distance_threshold: float = 0.004,
    ransac_n: int = 3,
    num_iterations: int = 3000,
    min_abs_nz: float = 0.93,      # tighter than before (~<=21deg tilt)
    max_tries: int = 6,
    min_inliers: int = 2000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Robustly find the *horizontal* supporting plane by extracting planes iteratively
    and selecting one whose normal is close to vertical (|nz| >= min_abs_nz).

    Returns: (wo_plane, plane_pts, plane_model_normalized)
    where plane_model is normalized so signed distance has units of meters.
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be (N,3), got {points.shape}")

    remaining = points
    remaining_idx = np.arange(points.shape[0])

    for _ in range(max_tries):
        if remaining.shape[0] < 500:
            break

        pcd = _pcd(remaining)
        model, inliers = pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations,
        )

        if len(inliers) < min_inliers:
            keep = np.ones(remaining.shape[0], dtype=bool)
            keep[inliers] = False
            remaining = remaining[keep]
            remaining_idx = remaining_idx[keep]
            continue

        a, b, c, d = [float(x) for x in model]
        n = np.array([a, b, c], dtype=np.float64)
        n_norm = np.linalg.norm(n) + 1e-12
        n = n / n_norm
        d = d / n_norm

        if abs(n[2]) >= min_abs_nz:
            # normalize model
            plane_model = np.array([n[0], n[1], n[2], d], dtype=float)

            mask = np.zeros(points.shape[0], dtype=bool)
            mask[remaining_idx[inliers]] = True

            plane_pts = points[mask]
            wo_plane = points[~mask]
            return wo_plane, plane_pts, plane_model

        # remove this plane and continue
        keep = np.ones(remaining.shape[0], dtype=bool)
        keep[inliers] = False
        remaining = remaining[keep]
        remaining_idx = remaining_idx[keep]

    # fallback
    return remove_plane_ransac(points, distance_threshold, ransac_n, num_iterations)


def cluster_dbscan(
    points: np.ndarray,
    eps: float = 0.02,
    min_points: int = 30,
) -> list[np.ndarray]:
    if points.size == 0:
        return []

    pcd = _pcd(points)
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    if labels.size == 0:
        return []

    clusters = []
    for lab in sorted(set(labels)):
        if lab == -1:
            continue
        clusters.append(points[labels == lab])

    clusters.sort(key=lambda c: c.shape[0], reverse=True)
    return clusters


def merge_close_clusters(
    clusters: list[np.ndarray],
    merge_dist: float = 0.05,
    z_overlap: float = 0.03,
) -> list[np.ndarray]:
    if len(clusters) <= 1:
        return clusters

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
                overlap_ok = not (zmaxs[j] < acc_zmin - z_overlap or zmins[j] > acc_zmax + z_overlap)
                if d < merge_dist and overlap_ok:
                    used[j] = True
                    acc.append(clusters[j])
                    changed = True

        merged.append(np.vstack(acc))

    merged.sort(key=lambda c: c.shape[0], reverse=True)
    return merged