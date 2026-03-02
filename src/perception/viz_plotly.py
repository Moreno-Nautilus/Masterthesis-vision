from __future__ import annotations
import numpy as np

def save_scene_plotly_html(
    scene: np.ndarray,
    plane: np.ndarray | None,
    clusters: list[np.ndarray],
    objects_world: list[tuple[str, np.ndarray]] | None = None,
    out_path: str = "/tmp/pipeline_debug.html",
    max_points: int = 20000,
) -> None:
    import plotly.graph_objects as go

    rng = np.random.default_rng(0)

    def ds(pts: np.ndarray | None, n: int) -> np.ndarray | None:
        if pts is None or len(pts) == 0:
            return pts
        if len(pts) <= n:
            return pts
        idx = rng.choice(len(pts), size=n, replace=False)
        return pts[idx]

    scene_ds = ds(scene, min(max_points, len(scene)))
    plane_ds = ds(plane, min(max_points // 3, len(plane))) if plane is not None else None
    clusters_ds = [ds(c, min(max_points // 2, len(c))) for c in clusters]
    objects_ds = None
    if objects_world is not None:
        objects_ds = [(name, ds(pts, min(max_points // 2, len(pts)))) for name, pts in objects_world]

    fig = go.Figure()

    # scene
    fig.add_trace(go.Scatter3d(
        x=scene_ds[:, 0], y=scene_ds[:, 1], z=scene_ds[:, 2],
        mode="markers",
        marker=dict(size=1, opacity=0.15),
        name="scene"
    ))

    # plane
    if plane_ds is not None and len(plane_ds) > 0:
        fig.add_trace(go.Scatter3d(
            x=plane_ds[:, 0], y=plane_ds[:, 1], z=plane_ds[:, 2],
            mode="markers",
            marker=dict(size=1, opacity=0.25),
            name="plane"
        ))

    # clusters
    for i, c in enumerate(clusters_ds):
        if c is None or len(c) == 0:
            continue
        fig.add_trace(go.Scatter3d(
            x=c[:, 0], y=c[:, 1], z=c[:, 2],
            mode="markers",
            marker=dict(size=2, opacity=0.8),
            name=f"cluster_{i}"
        ))

    # aligned CADs
    if objects_ds is not None:
        for name, pts in objects_ds:
            if pts is None or len(pts) == 0:
                continue
            fig.add_trace(go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode="markers",
                marker=dict(size=2, opacity=0.9),
                name=f"CAD@world:{name}"
            ))

    fig.update_layout(
        title="Pipeline debug: scene / plane / clusters / CAD aligned",
        scene=dict(
            xaxis_title="X [m]",
            yaxis_title="Y [m]",
            zaxis_title="Z [m]",
            aspectmode="data",
        ),
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"Saved Plotly HTML: {out_path}")