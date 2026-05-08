"""Generate synthetic DINO reference crops by rendering CAD meshes from
multiple viewpoints.

Output layout matches what DINOIdentifier expects:

    <out_dir>/<object_id>/<view_idx>.png

Each output PNG is the rendered object on a black background, cropped tight
around the object silhouette and (optionally) padded.

Usage:
    python3 tools/generate_dino_reference_renders.py \
        --cad-dir Data/CAD_Models_centered \
        --out-dir Data/reference_renders \
        --views 42 \
        --image-size 512 \
        --mesh-scale 0.01

Notes
-----
- Uses Open3D's offscreen renderer when DISPLAY is unset, falls back to
  trimesh + pyrender otherwise. We default to Open3D's OffscreenRenderer
  since it's already a dependency in this repo.
- Viewpoints are sampled on a Fibonacci sphere. Camera looks at the mesh
  centroid; up is +Y unless the polar angle is too close to ±Y.
- Background is set to black so DINOIdentifier's masked-background path
  doesn't double-mask. The generated images are tight crops around the
  rendered silhouette so DINOv2's resize-to-518 keeps the object large.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import open3d as o3d


def fibonacci_sphere(n: int) -> np.ndarray:
    """n unit vectors approximately uniform on the sphere."""
    if n <= 0:
        return np.zeros((0, 3))
    indices = np.arange(n, dtype=np.float64) + 0.5
    phi = np.arccos(1.0 - 2.0 * indices / n)  # polar [0, pi]
    theta = np.pi * (1.0 + 5.0 ** 0.5) * indices  # azimuth
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.stack([x, y, z], axis=1)


def look_at(eye: np.ndarray, target: np.ndarray, up_hint: np.ndarray) -> np.ndarray:
    """Build a camera-from-world extrinsic. Open3D wants the world->camera
    transform with +Z forward (into the scene), +Y down (image convention)."""
    forward = target - eye
    forward /= np.linalg.norm(forward) + 1e-12
    # Build a stable right vector
    up = up_hint / (np.linalg.norm(up_hint) + 1e-12)
    if abs(np.dot(forward, up)) > 0.99:
        up = np.array([1.0, 0.0, 0.0])
    right = np.cross(forward, up)
    right /= np.linalg.norm(right) + 1e-12
    new_up = np.cross(right, forward)

    # World->camera basis (rows = camera axes)
    R = np.stack([right, -new_up, forward], axis=0)
    t = -R @ eye
    extr = np.eye(4)
    extr[:3, :3] = R
    extr[:3, 3] = t
    return extr


def render_object_views(
    mesh_path: Path,
    out_dir: Path,
    n_views: int,
    image_size: int,
    mesh_scale: float,
    distance_scale: float = 3.0,
    pad_px: int = 8,
) -> int:
    """Render `n_views` viewpoints of the mesh; returns count written."""
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if mesh.is_empty():
        print(f"[render] empty mesh: {mesh_path}")
        return 0

    mesh.compute_vertex_normals()
    # Match the runner's mesh-scale convention
    mesh.scale(mesh_scale, center=(0.0, 0.0, 0.0))
    mesh.translate(-mesh.get_center())

    # Apply a default colour if the mesh has no per-vertex colour, so DINO
    # gets a non-flat appearance even on plain meshes.
    if not mesh.has_vertex_colors():
        mesh.paint_uniform_color([0.65, 0.65, 0.7])

    aabb = mesh.get_axis_aligned_bounding_box()
    extent = float(np.linalg.norm(aabb.get_extent()))
    cam_dist = max(0.05, distance_scale * extent)

    # Offscreen renderer
    renderer = o3d.visualization.rendering.OffscreenRenderer(image_size, image_size)
    scene = renderer.scene
    scene.set_background([0.0, 0.0, 0.0, 1.0])
    scene.scene.set_sun_light([0.0, 0.0, -1.0], [1.0, 1.0, 1.0], 75000.0)
    scene.scene.enable_sun_light(True)

    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit"
    scene.add_geometry("mesh", mesh, mat)

    fov_deg = 60.0
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for i, dir_vec in enumerate(fibonacci_sphere(n_views)):
        eye = dir_vec * cam_dist
        # Use mesh centroid (already at origin after recentre).
        target = np.zeros(3)
        # Pick up vector — avoid singularity by switching when too colinear.
        up_hint = np.array([0.0, 0.0, 1.0])
        if abs(dir_vec @ up_hint) > 0.95:
            up_hint = np.array([0.0, 1.0, 0.0])
        renderer.setup_camera(fov_deg, target, eye, up_hint)

        img = renderer.render_to_image()
        rgb = np.asarray(img)  # (H, W, 3) uint8

        # Crop to silhouette using non-black pixels.
        non_bg = rgb.sum(axis=2) > 0
        if not non_bg.any():
            continue
        ys, xs = np.where(non_bg)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0 = max(0, y0 - pad_px); y1 = min(image_size, y1 + pad_px)
        x0 = max(0, x0 - pad_px); x1 = min(image_size, x1 + pad_px)
        crop = rgb[y0:y1, x0:x1]

        out_path = out_dir / f"view_{i:03d}.png"
        # cv2 encodes as BGR; convert.
        bgr = crop[:, :, ::-1]
        import cv2
        cv2.imwrite(str(out_path), bgr)
        written += 1

    scene.remove_geometry("mesh")
    return written


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cad-dir", default="Data/CAD_Models_centered")
    p.add_argument("--out-dir", default="Data/reference_renders")
    p.add_argument("--views", type=int, default=42,
                   help="Number of viewpoints per object (Fibonacci sphere).")
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--mesh-scale", type=float, default=0.01,
                   help="Same as runner's --mesh-scale (mm -> m typically).")
    p.add_argument("--distance-scale", type=float, default=3.0,
                   help="Camera distance as a multiple of mesh diagonal.")
    p.add_argument("--pad-px", type=int, default=8)
    p.add_argument("--object-ids", nargs="*", default=None,
                   help="Subset of object names (mesh stems) to render.")
    args = p.parse_args()

    cad_root = Path(args.cad_dir)
    if not cad_root.exists():
        raise SystemExit(f"CAD dir does not exist: {cad_root}")
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    mesh_files = []
    for ext in ("*.obj", "*.stl", "*.ply"):
        mesh_files.extend(sorted(cad_root.glob(ext)))

    if args.object_ids:
        wanted = set(args.object_ids)
        mesh_files = [m for m in mesh_files if m.stem in wanted]

    if not mesh_files:
        raise SystemExit(f"No mesh files found in {cad_root}")

    total = 0
    for mesh_path in mesh_files:
        obj_id = mesh_path.stem
        obj_out = out_root / obj_id
        n = render_object_views(
            mesh_path=mesh_path,
            out_dir=obj_out,
            n_views=args.views,
            image_size=args.image_size,
            mesh_scale=args.mesh_scale,
            distance_scale=args.distance_scale,
            pad_px=args.pad_px,
        )
        print(f"[render] {obj_id}: {n} views -> {obj_out}")
        total += n
    print(f"[render] total views written: {total}")


if __name__ == "__main__":
    main()
