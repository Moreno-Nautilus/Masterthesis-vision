from __future__ import annotations
import numpy as np
from src.utils.se3 import SE3


def occlude_points_halfspace(
    pts: np.ndarray,
    keep_ratio: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Random half-space occlusion: keeps roughly keep_ratio of points.
    keep_ratio=1.0 => no occlusion
    keep_ratio=0.5 => keep about half the points
    """
    if keep_ratio >= 1.0:
        return pts
    if keep_ratio <= 0.0:
        return pts[:0]

    n = rng.normal(size=3)
    n /= (np.linalg.norm(n) + 1e-12)
    proj = pts @ n
    thresh = np.quantile(proj, 1.0 - keep_ratio)
    return pts[proj >= thresh]


def add_outliers_uniform(
    scene: np.ndarray,
    n: int,
    bounds: tuple[float, float, float, float, float, float],
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Adds uniformly sampled random outlier points within given bounds.
    bounds = (xmin, xmax, ymin, ymax, zmin, zmax)
    """
    if n <= 0:
        return scene
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    out = np.stack(
        [
            rng.uniform(xmin, xmax, n),
            rng.uniform(ymin, ymax, n),
            rng.uniform(zmin, zmax, n),
        ],
        axis=1,
    )
    return np.vstack([scene, out])

def generate_plane_points(
    n: int = 20000,
    xlim: tuple[float, float] = (-0.5, 0.5),
    ylim: tuple[float, float] = (-0.5, 0.5),
    z: float = 0.0,
    noise_std: float = 0.001,
) -> np.ndarray:
    """
    Generate a horizontal plane z = constant with Gaussian noise.
    Units: meters.
    Returns: (n,3)
    """
    xs = np.random.uniform(xlim[0], xlim[1], size=n)
    ys = np.random.uniform(ylim[0], ylim[1], size=n)
    zs = np.full(n, z)
    pts = np.stack([xs, ys, zs], axis=1)
    pts += np.random.normal(0.0, noise_std, size=pts.shape)
    return pts

def make_synthetic_scene_single_object(
    cad_points: np.ndarray,
    pose_gt: SE3,
    table_z: float = 0.0,
    obj_noise_std: float = 0.002,
    plane_noise_std: float = 0.001,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      scene_points: (N,3) plane + object
      observed_obj: (M,3) object-only (for debugging)
    """
    observed_obj = pose_gt.transform_points(cad_points)
    observed_obj = observed_obj + np.random.normal(0.0, obj_noise_std, size=observed_obj.shape)

    plane = generate_plane_points(z=table_z, noise_std=plane_noise_std)

    scene = np.vstack([plane, observed_obj])
    return scene, observed_obj



def make_synthetic_scene_single_object_robust(
    cad_points: np.ndarray,
    pose_gt: SE3,
    table_z: float = 0.0,
    obj_noise_std: float = 0.002,
    plane_noise_std: float = 0.001,
    occlusion_keep_ratio: float = 1.0,
    n_outliers: int = 0,
    outlier_bounds: tuple[float, float, float, float, float, float] = (-0.5, 0.5, -0.5, 0.5, -0.05, 0.2),
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Same as make_synthetic_scene_single_object but adds:
      - random half-space occlusion on object points
      - uniform outlier points in scene

    Returns:
      scene_points: (N,3)
      observed_obj: (M,3) object-only after occlusion+noise
    """
    rng = np.random.default_rng(seed)

    observed_obj = pose_gt.transform_points(cad_points)
    observed_obj = observed_obj + rng.normal(0.0, obj_noise_std, size=observed_obj.shape)

    # occlusion
    observed_obj = occlude_points_halfspace(observed_obj, occlusion_keep_ratio, rng)

    # plane
    plane = generate_plane_points(z=table_z, noise_std=plane_noise_std)

    scene = np.vstack([plane, observed_obj])

    # outliers
    scene = add_outliers_uniform(scene, n_outliers, outlier_bounds, rng)

    return scene, observed_obj


def make_synthetic_scene_multi_object_robust(
    cad_library: dict[str, np.ndarray],
    poses_gt: dict[str, SE3],
    table_z: float = 0.0,
    obj_noise_std: float = 0.002,
    plane_noise_std: float = 0.001,
    occlusion_keep_ratio: float = 1.0,
    n_outliers: int = 0,
    outlier_bounds: tuple[float, float, float, float, float, float] = (-0.5, 0.5, -0.5, 0.5, -0.05, 0.2),
    seed: int | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Returns:
    scene_points: plane + all objects (+ outliers)
    observed_objs: dict[obj_id] -> (Mi,3) object points after noise+occlusion (in world)
    """
    rng = np.random.default_rng(seed)

    observed_objs: dict[str, np.ndarray] = {}
    all_obj_pts = []

    for obj_id, cad_pts in cad_library.items():
        pose = poses_gt[obj_id]
        pts = pose.transform_points(cad_pts)
        pts = pts + rng.normal(0.0, obj_noise_std, size=pts.shape)
        pts = occlude_points_halfspace(pts, occlusion_keep_ratio, rng)
        observed_objs[obj_id] = pts
        all_obj_pts.append(pts)

    plane = generate_plane_points(z=table_z, noise_std=plane_noise_std)
    scene = np.vstack([plane] + all_obj_pts)

    scene = add_outliers_uniform(scene, n_outliers, outlier_bounds, rng)
    return scene, observed_objs