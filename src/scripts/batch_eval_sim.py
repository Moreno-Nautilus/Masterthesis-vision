from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import open3d as o3d

from src.utils.se3 import SE3
from src.utils.geometry import rotation_error_deg
from src.perception.pose_icp import load_cad_as_pointcloud
from src.perception.sim_scene import make_synthetic_scene_multi_object_robust
from src.perception.pipeline import GraspPerceptionPipeline, PipelineConfig

from src.perception.segmentation import remove_plane_ransac
from src.perception.viz import save_scene_png
from src.perception.viz_plotly import save_scene_plotly_html
import itertools

def fixed_pose(tx: float, ty: float, tz: float, rng: np.random.Generator) -> SE3:
    A = rng.normal(size=(3, 3))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return SE3(Q, np.array([tx, ty, tz], dtype=float))

def cube_symmetry_rots() -> list[np.ndarray]:
    """
    24 proper rotations of the cube (det=+1), built from axis permutations and sign flips.
    """
    rots: list[np.ndarray] = []
    axes = np.eye(3)
    for perm in itertools.permutations([0, 1, 2]):
        P = axes[:, perm]
        for signs in itertools.product([-1, 1], repeat=3):
            R = P @ np.diag(signs)
            if np.isclose(np.linalg.det(R), 1.0):
                rots.append(R.astype(float))

    uniq: list[np.ndarray] = []
    for R in rots:
        if not any(np.allclose(R, U) for U in uniq):
            uniq.append(R)
    return uniq


def rot_err_symmetry_aware(R_est: np.ndarray, R_gt: np.ndarray, sym_rots: list[np.ndarray]) -> float:
    errs = [rotation_error_deg(R_est, R_gt @ S) for S in sym_rots]
    return float(np.min(errs)) if errs else float(rotation_error_deg(R_est, R_gt))

def build_cad_library() -> dict[str, np.ndarray]:
    return {
        "cube": load_cad_as_pointcloud("Data/CAD_Models/Cube.stl", scale=0.01, center=True),
        "cat": load_cad_as_pointcloud("Data/CAD_Models/Cat.stl", scale=0.0045, center=True),
        "dolphin": load_cad_as_pointcloud("Data/CAD_Models/dolphin.stl", scale=0.001, center=True),
        "hand": load_cad_as_pointcloud("Data/CAD_Models/Hand.stl", scale=0.01, center=True),
    }

def cad_xy_radius(cad_pts: np.ndarray) -> float:
    # radius of CAD in XY plane (centered CAD assumed)
    xy = cad_pts[:, :2]
    return float(np.max(np.linalg.norm(xy, axis=1)))

def build_poses_gt(
    obj_ids: list[str],
    seed: int,
    cad_r_xy: dict[str, float],
    margin: float = 0.03,  # extra clearance in meters
) -> dict[str, SE3]:
    """
    Place objects in XY so their *footprints* don't overlap:
      ||p_i - p_j||_xy >= r_i + r_j + margin
    """
    rng = np.random.default_rng(seed)
    z = 0.1
    xy_bounds = (-0.35, 0.35)

    positions: dict[str, tuple[float, float, float]] = {}

    for obj_id in obj_ids:
        ri = cad_r_xy[obj_id]
        for _ in range(2000):
            x = rng.uniform(*xy_bounds)
            y = rng.uniform(*xy_bounds)

            ok = True
            for other_id, (px, py, _pz) in positions.items():
                rj = cad_r_xy[other_id]
                if np.hypot(x - px, y - py) < (ri + rj + margin):
                    ok = False
                    break

            if ok:
                positions[obj_id] = (x, y, z)
                break
        else:
            # if it fails, expand search bounds and try again
            # (keeps everything automatic)
            xy_bounds = (xy_bounds[0] * 1.15, xy_bounds[1] * 1.15)
            # restart this object placement with expanded bounds
            # (simple approach: clear and restart all placements)
            return build_poses_gt(obj_ids=obj_ids, seed=seed + 999, cad_r_xy=cad_r_xy, margin=margin)
    poses: dict[str, SE3] = {}
    for obj_id in obj_ids:
        tx, ty, tz = positions[obj_id]
        poses[obj_id] = fixed_pose(tx, ty, tz, rng)
    return poses


def pose_errors(
    T_est_obj_to_world: SE3,
    T_gt_obj_to_world: SE3,
    obj_id: str,
    cube_syms: list[np.ndarray],
) -> tuple[float, float]:
    t_err = float(np.linalg.norm(T_est_obj_to_world.t - T_gt_obj_to_world.t))
    if obj_id == "cube":
        r_err = rot_err_symmetry_aware(T_est_obj_to_world.R, T_gt_obj_to_world.R, cube_syms)
    else:
        r_err = float(rotation_error_deg(T_est_obj_to_world.R, T_gt_obj_to_world.R))
    return t_err, r_err


def write_debug_bundle(
    out_dir: Path,
    seed: int,
    scene: np.ndarray,
    result,
    cad_library: dict[str, np.ndarray],
    poses_gt: dict[str, SE3],
) -> None:
    dbg = out_dir / f"debug_seed_{seed}"
    dbg.mkdir(parents=True, exist_ok=True)

    o3d.io.write_point_cloud(
        str(dbg / "scene.ply"),
        o3d.geometry.PointCloud(o3d.utility.Vector3dVector(scene)),
    )
    for k, c in enumerate(result.clusters):
        o3d.io.write_point_cloud(
            str(dbg / f"cluster_{k}.ply"),
            o3d.geometry.PointCloud(o3d.utility.Vector3dVector(c)),
        )

    _, plane_pts, _ = remove_plane_ransac(scene)

    objects_world: list[tuple[str, np.ndarray]] = []
    for obj in result.objects:
        cad_pts = cad_library[obj.object_id]
        cad_world = obj.T_object_to_world.transform_points(cad_pts)
        objects_world.append((f"pred_{obj.object_id}", cad_world))

    for obj_id, T_gt in poses_gt.items():
        cad_pts = cad_library[obj_id]
        cad_world = T_gt.transform_points(cad_pts)
        objects_world.append((f"gt_{obj_id}", cad_world))

    save_scene_png(
        scene=scene,
        plane=plane_pts,
        clusters=result.clusters,
        objects_world=objects_world,
        out_path=str(dbg / "debug.png"),
    )
    save_scene_plotly_html(
        scene=scene,
        plane=plane_pts,
        clusters=result.clusters,
        objects_world=objects_world,
        out_path=str(dbg / "debug.html"),
    )

    with (dbg / "summary.txt").open("w") as f:
        f.write(f"seed={seed}\n")
        f.write(f"n_clusters={len(result.clusters)} n_detected={len(result.objects)}\n")
        for o in result.objects:
            f.write(
                f"pred id={o.object_id} rms_nn={o.metrics.get('rms_nn')} margin={o.metrics.get('margin')} "
                f"icp_fit={o.metrics.get('icp_fitness')} icp_rmse={o.metrics.get('icp_inlier_rmse')}\n"
            )

    print(f"[debug] wrote bundle: {dbg}")


def main() -> None:
    save_debug = True
    max_debug_folders = 15
    debug_written = 0

    bad_t_thresh = 0.02   # 2 cm
    bad_r_thresh = 20.0   # 20 deg
    out_dir = Path("outputs/batch_eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.jsonl"

    cad_library = build_cad_library()
    cad_r_xy = {k: cad_xy_radius(v) for k, v in cad_library.items()}
    obj_ids = list(cad_library.keys())

    cfg = PipelineConfig(
        plane_distance_threshold=0.003,
        dbscan_eps=0.020,         # slightly larger
        dbscan_min_points=25,     # lower so occluded objects still form clusters
        merge_dist=0.07,          # allow merging fragments more aggressively
        merge_z_overlap=0.05,
        voxel_size=0.005,
        min_margin=1.15,
        enforce_one_to_one=True,
        max_rms_nn=0.020,         # relax a bit (was 0.015)
        min_icp_fitness=0.10,
        max_icp_inlier_rmse=0.02,
    )
    pipe = GraspPerceptionPipeline(cad_library=cad_library, cfg=cfg)

    # ---- batch settings
    seeds = list(range(50))
    occlusion_keep_ratio = 0.70   # realistic with 3 cams + overview
    n_outliers = 800

    # aggregate stats
    n_total = 0
    n_scene_success = 0
    id_hits = 0
    id_total = 0
    t_errs: list[float] = []
    r_errs: list[float] = []

    cube_syms = cube_symmetry_rots()

    # per-object accumulators
    per_obj_found = {k: 0 for k in obj_ids}
    per_obj_total = {k: 0 for k in obj_ids}
    per_obj_t = {k: [] for k in obj_ids}
    per_obj_r = {k: [] for k in obj_ids}



    with out_path.open("w") as f:
        for seed in seeds:
            poses_gt = build_poses_gt(obj_ids, seed=seed, cad_r_xy=cad_r_xy, margin=0.03)
            scene, _ = make_synthetic_scene_multi_object_robust(
                cad_library=cad_library,
                poses_gt=poses_gt,
                table_z=0.0,
                obj_noise_std=0.002,
                plane_noise_std=0.001,
                occlusion_keep_ratio=occlusion_keep_ratio,
                n_outliers=n_outliers,
                seed=seed,
            )

            result = pipe.run(scene)
            det_by_id = {o.object_id: o for o in result.objects}

            per_obj = []
            all_found = True

            for obj_id in obj_ids:
                id_total += 1
                per_obj_total[obj_id] += 1

                if obj_id not in det_by_id:
                    all_found = False
                    per_obj.append({"obj_id": obj_id, "found": False})
                    continue

                id_hits += 1
                per_obj_found[obj_id] += 1

                det = det_by_id[obj_id]
                T_est = det.T_object_to_world
                T_gt = poses_gt[obj_id]
                t_err, r_err = pose_errors(T_est, T_gt, obj_id=obj_id, cube_syms=cube_syms)
                t_errs.append(t_err)
                r_errs.append(r_err)
                per_obj_t[obj_id].append(t_err)
                per_obj_r[obj_id].append(r_err)

                per_obj.append({
                    "obj_id": obj_id,
                    "found": True,
                    "t_err_m": t_err,
                    "r_err_deg": r_err,
                    "rms_nn": float(det.metrics.get("rms_nn", np.nan)),
                    "margin": float(det.metrics.get("margin", np.nan)),
                    "icp_fitness": float(det.metrics.get("icp_fitness", np.nan)),
                    "icp_inlier_rmse": float(det.metrics.get("icp_inlier_rmse", np.nan)),
                })

            bad_pose = any(
                item.get("found")
                and item["obj_id"] != "cube"
                and (item["t_err_m"] > bad_t_thresh or item["r_err_deg"] > bad_r_thresh)
                for item in per_obj
            )

            if save_debug and debug_written < max_debug_folders and ((not all_found) or bad_pose):
                write_debug_bundle(out_dir, seed, scene, result, cad_library, poses_gt)
                debug_written += 1
            n_total += 1
            if all_found:
                n_scene_success += 1

            row = {
                "seed": seed,
                "occlusion_keep_ratio": occlusion_keep_ratio,
                "n_outliers": n_outliers,
                "n_clusters": len(result.clusters),
                "n_detected": len(result.objects),
                "scene_success_all_found": all_found,
                "timings": [{"name": t.name, "dt_s": t.dt_s} for t in result.timings],
                "per_obj": per_obj,
            }
            f.write(json.dumps(row) + "\n")

    def mean(xs: list[float]) -> float:
        return float(np.mean(xs)) if len(xs) else float("nan")

    def med(xs: list[float]) -> float:
        return float(np.median(xs)) if len(xs) else float("nan")

    print("\n=== BATCH SUMMARY ===")
    print("scenes:", n_total)
    print("scene success (all 4 found):", f"{n_scene_success}/{n_total} = {n_scene_success/n_total:.2f}")
    print("ID recall:", f"{id_hits}/{id_total} = {id_hits/id_total:.2f}")
    print("t_err mean/median [m]:", mean(t_errs), med(t_errs))
    print("r_err mean/median [deg]:", mean(r_errs), med(r_errs))
    print("wrote:", out_path)

    print("\n=== PER-OBJECT ===")
    for obj_id in obj_ids:
        found = per_obj_found[obj_id]
        total = per_obj_total[obj_id]
        print(
            f"{obj_id:>8s}: found {found}/{total} = {found/total:.2f} | "
            f"t_med={med(per_obj_t[obj_id]):.4f} m | r_med={med(per_obj_r[obj_id]):.1f} deg"
        )


if __name__ == "__main__":
    main()