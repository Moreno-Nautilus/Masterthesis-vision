from __future__ import annotations
import numpy as np

def save_scene_png(
    scene: np.ndarray,
    plane: np.ndarray | None,
    clusters: list[np.ndarray],
    objects_world: list[tuple[str, np.ndarray]] | None = None,
    out_path: str = "/tmp/pipeline_debug.png",
    max_points: int = 12000,
) -> None:
    """
    Save a quick matplotlib 3D scatter PNG.
    - scene: full scene points (N,3)
    - plane: plane points (M,3) or None
    - clusters: list of cluster points
    - objects_world: optional list of (name, cad_points_transformed_to_world)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)

    def ds(pts: np.ndarray, n: int) -> np.ndarray:
        if pts is None or len(pts) == 0:
            return pts
        if len(pts) <= n:
            return pts
        idx = rng.choice(len(pts), size=n, replace=False)
        return pts[idx]

    # downsample for speed
    scene_ds = ds(scene, min(max_points, len(scene)))
    plane_ds = ds(plane, min(max_points // 3, len(plane))) if plane is not None else None
    clusters_ds = [ds(c, min(max_points // 2, len(c))) for c in clusters]
    objects_ds = None
    if objects_world is not None:
        objects_ds = [(name, ds(pts, min(max_points // 2, len(pts)))) for name, pts in objects_world]

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(scene_ds[:, 0], scene_ds[:, 1], scene_ds[:, 2], s=1, c="k", alpha=0.15, label="scene")
    if plane_ds is not None and len(plane_ds) > 0:
        ax.scatter(plane_ds[:, 0], plane_ds[:, 1], plane_ds[:, 2], s=1, c="g", alpha=0.25, label="plane")

    for i, c in enumerate(clusters_ds):
        if c is None or len(c) == 0:
            continue
        ax.scatter(c[:, 0], c[:, 1], c[:, 2], s=2, alpha=0.7, label=f"cluster_{i}")

    if objects_ds is not None:
        for name, pts in objects_ds:
            if pts is None or len(pts) == 0:
                continue
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=2, alpha=0.8, label=f"CAD@world:{name}")

    ax.set_title("Pipeline debug: scene / plane / clusters / CAD aligned")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.legend(loc="upper right", fontsize=8)

    # equal-ish axis scaling
    all_pts = [scene_ds]
    if plane_ds is not None and len(plane_ds) > 0:
        all_pts.append(plane_ds)
    for c in clusters_ds:
        if c is not None and len(c) > 0:
            all_pts.append(c)
    if objects_ds is not None:
        for _, pts in objects_ds:
            if pts is not None and len(pts) > 0:
                all_pts.append(pts)

    all_pts = np.vstack(all_pts)
    mins = all_pts.min(axis=0)
    maxs = all_pts.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * float(np.max(maxs - mins))
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.view_init(elev=18, azim=35)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved PNG: {out_path}")