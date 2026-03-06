from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import open3d as o3d

from src.utils.se3 import SE3
from src.perception.pose_icp import estimate_pose_icp
from src.perception.segmentation import (
    cluster_dbscan,
    merge_close_clusters,
)


class Logger:
    def info(self, msg: str, **kv: Any) -> None:
        print(msg + (" | " + " ".join(f"{k}={v}" for k, v in kv.items()) if kv else ""))

    def warn(self, msg: str, **kv: Any) -> None:
        print("[WARN] " + msg + (" | " + " ".join(f"{k}={v}" for k, v in kv.items()) if kv else ""))


@dataclass
class StageTiming:
    name: str
    dt_s: float


@dataclass
class DetectedObject:
    object_id: str
    point_cloud: np.ndarray
    T_object_to_world: SE3  # semantically T_base_obj (OBJ -> BASE)
    id_confidence: float
    pose_confidence: float
    metrics: dict[str, float]


@dataclass
class SceneResult:
    points_world_raw: np.ndarray
    plane_model: np.ndarray
    plane_points: np.ndarray
    points_wo_plane: np.ndarray
    clusters: list[np.ndarray]
    objects: list[DetectedObject]
    timings: list[StageTiming]


@dataclass
class PipelineConfig:
    # plane (dominant plane only; DO NOT assume Z-up)
    plane_distance_threshold: float = 0.002
    plane_ransac_n: int = 3
    plane_num_iterations: int = 4000

    # object band above plane (meters)
    h_min: float = 0.010
    h_max: float = 0.07

    # remove a second plane from the above-band (tabletop / plate)
    remove_plane2: bool = True
    plane2_distance_threshold: float = 0.0025
    plane2_num_iterations: int = 2000

    # workspace crop in the plane around the plane centroid
    use_disc_crop: bool = True
    disc_radius: float = 0.45

    # clustering (first pass)
    dbscan_eps: float = 0.03
    dbscan_min_points: int = 40
    merge_dist: float = 0.06
    merge_z_overlap: float = 0.04

    # --- refinement inside the biggest cluster (zoom in) ---
    refine_on_largest_cluster: bool = True
    refine_max_points: int = 60000
    refine_plane2_distance_threshold: float = 0.0020
    refine_plane2_num_iterations: int = 2500
    refine_dbscan_eps: float = 0.014
    refine_dbscan_min_points: int = 50

    # ICP / matching
    voxel_size: float = 0.005
    enforce_one_to_one: bool = True

    # NN RMS gate + margin
    max_rms_nn: float = 0.030
    min_margin: float = 1.5

    # soft ICP-quality gates (reject only if BOTH are bad)
    min_icp_fitness: float = 0.05
    max_icp_inlier_rmse: float = 0.04

    # --- NEW: CAD coverage scoring (fixes "cube fits to screw") ---
    cad_cover_thresh: float = 0.012          # meters; ~1.2cm
    min_cad_cover_ratio: float = 0.35        # require >=35% of CAD points explained


def _pcd(pts: np.ndarray) -> o3d.geometry.PointCloud:
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(np.asarray(pts, dtype=float))
    return p

def _snap_object_to_plane(
    T_base_obj: SE3,
    cad_obj: np.ndarray,
    plane_n: np.ndarray,
    plane_d: float,
    clearance: float = 0.001,
) -> SE3:
    """
    Shift object along plane normal so the CAD lowest point touches the plane (plus clearance).
    plane equation: n^T x + d = 0, with n normalized.
    """
    n = plane_n / (np.linalg.norm(plane_n) + 1e-12)
    cad_base = T_base_obj.transform_points(cad_obj)
    h = cad_base @ n + float(plane_d)
    h_min = float(h.min())
    # want h_min == clearance
    delta = (clearance - h_min) * n
    return SE3(T_base_obj.R, T_base_obj.t + delta)


def _remove_plane(
    points: np.ndarray,
    distance_threshold: float,
    ransac_n: int = 3,
    num_iterations: int = 2000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Segment one dominant plane from points.
    Returns (wo_plane, plane_points, plane_model_normalized[a,b,c,d]).
    """
    if points is None or len(points) < 500:
        return points, np.zeros((0, 3), dtype=float), np.array([0.0, 0.0, 1.0, 0.0], dtype=float)

    pcd = _pcd(points)
    model, inliers = pcd.segment_plane(
        distance_threshold=float(distance_threshold),
        ransac_n=int(ransac_n),
        num_iterations=int(num_iterations),
    )
    inliers = np.asarray(inliers, dtype=int)
    if inliers.size == 0:
        return points, np.zeros((0, 3), dtype=float), np.array([0.0, 0.0, 1.0, 0.0], dtype=float)

    mask = np.zeros(len(points), dtype=bool)
    mask[inliers] = True
    plane_pts = points[mask]
    wo = points[~mask]

    a, b, c, d = [float(x) for x in model]
    n = np.array([a, b, c], dtype=float)
    nn = np.linalg.norm(n) + 1e-12
    n = n / nn
    d = d / nn
    model_n = np.array([n[0], n[1], n[2], d], dtype=float)
    return wo, plane_pts, model_n


def nn_rms(source_pts: np.ndarray, target_pts: np.ndarray) -> float:
    if len(source_pts) == 0 or len(target_pts) == 0:
        return float("inf")
    src = _pcd(source_pts)
    tgt = _pcd(target_pts)
    dists = np.asarray(src.compute_point_cloud_distance(tgt))
    if not np.isfinite(dists).all():
        return float("inf")
    return float(np.sqrt(np.mean(dists**2)))


def _plane_basis_from_normal(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = n / (np.linalg.norm(n) + 1e-12)
    tmp = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(float(np.dot(tmp, n))) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0], dtype=float)
    u = np.cross(n, tmp)
    u = u / (np.linalg.norm(u) + 1e-12)
    v = np.cross(n, u)
    v = v / (np.linalg.norm(v) + 1e-12)
    return u, v


def _crop_disc_in_plane(points: np.ndarray, n: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    if len(points) == 0:
        return points
    u, v = _plane_basis_from_normal(n)
    p = points - center
    du = p @ u
    dv = p @ v
    r2 = du**2 + dv**2
    return points[r2 <= float(radius) ** 2]


def _choose_plane_sign_by_above_band(
    points: np.ndarray,
    n: np.ndarray,
    d: float,
    h_min: float,
    h_max: float,
) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Choose sign of plane normal such that the number of points in the above-band is maximized.
    Returns (n, d, signed_distances) after choosing sign.
    """
    signed0 = points @ n + d
    signed1 = -signed0

    n_above0 = int(np.count_nonzero((signed0 > h_min) & (signed0 < h_max)))
    n_above1 = int(np.count_nonzero((signed1 > h_min) & (signed1 < h_max)))

    if n_above1 > n_above0:
        return -n, -d, signed1
    return n, d, signed0


def _bbox_diag(pts: np.ndarray) -> float:
    if pts is None or len(pts) == 0:
        return float("inf")
    ext = pts.max(axis=0) - pts.min(axis=0)
    return float(np.linalg.norm(ext))


def _cad_coverage_metrics(cad_base: np.ndarray, cluster_base: np.ndarray, thresh: float) -> tuple[float, float, int]:
    """
    Compute how well the cluster explains the full CAD (coverage).
    Returns (cover_ratio, cover_rmse, inlier_count) using CAD->cluster NN distances.
    """
    if len(cad_base) == 0 or len(cluster_base) == 0:
        return 0.0, float("inf"), 0
    cad_p = _pcd(cad_base)
    clu_p = _pcd(cluster_base)
    d = np.asarray(cad_p.compute_point_cloud_distance(clu_p), dtype=float)  # per CAD point
    if d.size == 0 or (not np.isfinite(d).any()):
        return 0.0, float("inf"), 0
    inl = d < float(thresh)
    ninl = int(np.count_nonzero(inl))
    cover = float(ninl) / float(len(d))
    rmse = float(np.sqrt(np.mean(d[inl] ** 2))) if ninl > 0 else float("inf")
    return cover, rmse, ninl


class GraspPerceptionPipeline:
    def __init__(
        self,
        cad_library: dict[str, np.ndarray],
        cfg: PipelineConfig | None = None,
        logger: Logger | None = None,
    ):
        self.cad_library = cad_library
        self.cfg = cfg or PipelineConfig()
        self.log = logger or Logger()

        self._cad_diag: dict[str, float] = {}
        for k, pts in self.cad_library.items():
            self._cad_diag[k] = _bbox_diag(np.asarray(pts, dtype=float))

    def run(self, scene_points_world: np.ndarray) -> SceneResult:
        timings: list[StageTiming] = []
        points_world_raw = np.asarray(scene_points_world, dtype=float)

        # --- 1) Dominant plane
        t0 = perf_counter()
        pcd = _pcd(points_world_raw)
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=self.cfg.plane_distance_threshold,
            ransac_n=self.cfg.plane_ransac_n,
            num_iterations=self.cfg.plane_num_iterations,
        )
        plane_model = np.asarray(plane_model, dtype=float)
        inliers = np.asarray(inliers, dtype=int)

        mask = np.zeros(len(points_world_raw), dtype=bool)
        mask[inliers] = True
        plane_pts = points_world_raw[mask]
        wo_plane = points_world_raw[~mask]

        # normalize plane
        n = plane_model[:3].astype(float)
        nn = np.linalg.norm(n) + 1e-12
        n = n / nn
        d = float(plane_model[3]) / nn
        plane_model = np.array([n[0], n[1], n[2], d], dtype=float)

        up_axis = int(np.argmax(np.abs(n)))
        self.log.info(
            "plane_removal",
            n_in=len(points_world_raw),
            n_plane=len(plane_pts),
            n_left=len(wo_plane),
            n=f"[{n[0]:.3f},{n[1]:.3f},{n[2]:.3f}]",
            up_axis=f"{up_axis}(0=x,1=y,2=z)",
        )
        timings.append(StageTiming("plane_fit", perf_counter() - t0))

        # --- 2) Above-plane band (robust sign)
        h_min = float(self.cfg.h_min)
        h_max = float(self.cfg.h_max)

        n, d, signed = _choose_plane_sign_by_above_band(points_world_raw, n, d, h_min, h_max)
        plane_model = np.array([n[0], n[1], n[2], d], dtype=float)

        mask_above = (signed > h_min) & (signed < h_max)
        points_objects = points_world_raw[mask_above]

        # --- 2.5) Remove tabletop/plate plane from above-band
        if self.cfg.remove_plane2 and len(points_objects) > 0:
            before2 = len(points_objects)
            points_objects, plane2_pts, _ = _remove_plane(
                points_objects,
                distance_threshold=float(self.cfg.plane2_distance_threshold),
                ransac_n=3,
                num_iterations=int(self.cfg.plane2_num_iterations),
            )
            self.log.info("remove_plane2", n_before=before2, n_plane2=len(plane2_pts), n_after=len(points_objects))

        self.log.info(
            "above_plane_band",
            h_min=h_min,
            h_max=h_max,
            n_above=len(points_objects),
            h_q50=float(np.quantile(signed, 0.5)),
            h_q95=float(np.quantile(signed, 0.95)),
            h_q99=float(np.quantile(signed, 0.99)),
        )

        # --- 3) Disc crop around plane centroid
        if self.cfg.use_disc_crop and len(points_objects) > 0 and len(plane_pts) > 0:
            center = plane_pts.mean(axis=0)
            before = len(points_objects)
            points_objects = _crop_disc_in_plane(points_objects, n, center, radius=float(self.cfg.disc_radius))
            self.log.info("disc_crop", radius=float(self.cfg.disc_radius), n_before=before, n_after=len(points_objects))

        # --- 4) First clustering
        t0 = perf_counter()
        clusters = cluster_dbscan(
            points_objects,
            eps=float(self.cfg.dbscan_eps),
            min_points=int(self.cfg.dbscan_min_points),
        )
        clusters = merge_close_clusters(
            clusters,
            merge_dist=float(self.cfg.merge_dist),
            z_overlap=float(self.cfg.merge_z_overlap),
        )
        timings.append(StageTiming("clustering", perf_counter() - t0))
        self.log.info("clustering", n_clusters=len(clusters), sizes=[len(c) for c in clusters[:5]])

        # --- 4.5) Refinement inside the largest cluster
        if self.cfg.refine_on_largest_cluster and len(clusters) > 0:
            largest = max(clusters, key=lambda c: c.shape[0])
            self.log.info("refine_stage0", largest_n=len(largest))

            pts_roi = largest
            if len(pts_roi) > int(self.cfg.refine_max_points):
                idx = np.random.choice(len(pts_roi), int(self.cfg.refine_max_points), replace=False)
                pts_roi = pts_roi[idx]
                self.log.info("refine_subsample", n=int(self.cfg.refine_max_points))

            before = len(pts_roi)
            pts_roi_wo_plane, plane2b_pts, _ = _remove_plane(
                pts_roi,
                distance_threshold=float(self.cfg.refine_plane2_distance_threshold),
                ransac_n=3,
                num_iterations=int(self.cfg.refine_plane2_num_iterations),
            )
            self.log.info("refine_remove_plane2", n_before=before, n_plane2=len(plane2b_pts), n_after=len(pts_roi_wo_plane))

            clusters_ref = cluster_dbscan(
                pts_roi_wo_plane,
                eps=float(self.cfg.refine_dbscan_eps),
                min_points=int(self.cfg.refine_dbscan_min_points),
            )
            clusters_ref = merge_close_clusters(
                clusters_ref,
                merge_dist=float(self.cfg.merge_dist),
                z_overlap=float(self.cfg.merge_z_overlap),
            )
            self.log.info("refine_clustering", n_clusters=len(clusters_ref), sizes=[len(c) for c in clusters_ref[:5]])

            if len(clusters_ref) > 0:
                clusters = clusters_ref

        # --- 5) Identify + pose
        t0 = perf_counter()
        objects = self._identify_and_pose(clusters, plane_model)        
        timings.append(StageTiming("identify_pose", perf_counter() - t0))
        self.log.info("identify_pose", n_objects=len(objects))

        return SceneResult(
            points_world_raw=points_world_raw,
            plane_model=np.asarray(plane_model, dtype=float),
            plane_points=np.asarray(plane_pts, dtype=float),
            points_wo_plane=np.asarray(wo_plane, dtype=float),
            clusters=clusters,
            objects=objects,
            timings=timings,
        )

    def _identify_and_pose(self, clusters: list[np.ndarray], plane_model: np.ndarray) -> list[DetectedObject]:
        if len(clusters) == 0:
            return []

        cad_ids = list(self.cad_library.keys())
        cad_pts = [self.cad_library[k] for k in cad_ids]

        used_models: set[int] = set()
        out: list[DetectedObject] = []

        for i, cluster in enumerate(clusters):
            if cluster.shape[0] < max(20, int(self.cfg.dbscan_min_points)):
                continue

            cand = []
            for j, model in enumerate(cad_pts):
                T_base_obj, icp_metrics = estimate_pose_icp(
                    observed_points=cluster,   # BASE
                    cad_points=model,          # OBJ
                    voxel_size=float(self.cfg.voxel_size),
                )

                # shape similarity (cluster -> obj frame)
                T_obj_base = T_base_obj.inverse()
                cluster_in_obj = T_obj_base.transform_points(cluster)
                rms = nn_rms(cluster_in_obj, model)

                fit = float(icp_metrics.get("icp_fitness", 0.0))
                rmse = float(icp_metrics.get("icp_inlier_rmse", np.inf))

                # --- NEW: coverage score (CAD -> cluster) ---
                cad_base = T_base_obj.transform_points(model)
                cover, cover_rmse, cover_n = _cad_coverage_metrics(
                    cad_base=cad_base,
                    cluster_base=cluster,
                    thresh=float(self.cfg.cad_cover_thresh),
                )

                cand.append(
                    {
                        "j": j,
                        "obj_id": cad_ids[j],
                        "T_base_obj": T_base_obj,
                        "rms_nn": rms,
                        "icp_fitness": fit,
                        "icp_rmse": rmse,
                        "cad_cover_ratio": float(cover),
                        "cad_cover_rmse": float(cover_rmse),
                        "cad_cover_n": int(cover_n),
                        "metrics": icp_metrics,
                    }
                )

            if len(cand) == 0:
                continue

            # Prefer coverage (prevents tiny screw cluster from winning)
            cand.sort(
                key=lambda c: (
                    -float(c["cad_cover_ratio"]),
                    float(c["cad_cover_rmse"]),
                    float(c["rms_nn"]),
                )
            )

            best = cand[0]
            second = cand[1] if len(cand) > 1 else None

            nn_best = float(best["rms_nn"])
            nn_second = float(second["rms_nn"]) if second is not None else float("inf")
            margin = (nn_second / nn_best) if nn_best > 1e-12 else float("inf")

            chosen = None
            for c in cand:
                j = int(c["j"])
                if self.cfg.enforce_one_to_one and j in used_models:
                    continue

                rms = float(c["rms_nn"])
                fit = float(c["icp_fitness"])
                rmse = float(c["icp_rmse"])
                cover = float(c["cad_cover_ratio"])

                if cover < float(self.cfg.min_cad_cover_ratio):
                    continue

                if (not np.isfinite(rms)) or (rms > float(self.cfg.max_rms_nn)):
                    continue

                # keep margin gate (still useful if you later add more CADs)
                if margin < float(self.cfg.min_margin):
                    continue

                # Soft ICP gating: reject only if BOTH are bad
                if (fit < float(self.cfg.min_icp_fitness)) and (rmse > float(self.cfg.max_icp_inlier_rmse)):
                    # allow if coverage is strong (common with partial views)
                    if cover < (float(self.cfg.min_cad_cover_ratio) + 0.10):
                        continue

                chosen = c
                break

            if chosen is None:
                top = cand[:5]
                self.log.warn(
                    "reject_cluster_no_candidate",
                    cluster=i,
                    top=[t["obj_id"] for t in top],
                    top_cover=[float(t["cad_cover_ratio"]) for t in top],
                    top_cover_rmse=[float(t["cad_cover_rmse"]) for t in top],
                    top_nn=[float(t["rms_nn"]) for t in top],
                    top_icp=[float(t["icp_rmse"]) for t in top],
                )
                continue

            if self.cfg.enforce_one_to_one:
                used_models.add(int(chosen["j"]))

            obj_id = str(chosen["obj_id"])
            T_base_obj = chosen["T_base_obj"]
                        # snap translation so object rests on plane (prevents penetration)
            n = np.asarray(plane_model[:3], dtype=float)
            d = float(plane_model[3])
            T_base_obj = _snap_object_to_plane(
                T_base_obj=T_base_obj,
                cad_obj=self.cad_library[obj_id],
                plane_n=n,
                plane_d=d,
                clearance=0.001,
            )

            cad_base = T_base_obj.transform_points(self.cad_library[obj_id])
            c_err = float(np.linalg.norm(cad_base.mean(axis=0) - cluster.mean(axis=0)))
            self.log.info(
                "pose_sanity",
                cluster=i,
                centroid_err_m=c_err,
                cad_cover=f"{float(chosen['cad_cover_ratio']):.3f}",
                cad_cover_rmse=f"{float(chosen['cad_cover_rmse']):.3f}",
                cad_cover_n=int(chosen["cad_cover_n"]),
            )

            id_conf = float(np.clip((margin - float(self.cfg.min_margin)) / 2.0, 0.0, 1.0))
            # confidence should strongly track coverage now
            pose_conf = float(np.clip((float(chosen["cad_cover_ratio"]) - float(self.cfg.min_cad_cover_ratio)) / 0.40, 0.0, 1.0))

            out.append(
                DetectedObject(
                    object_id=obj_id,
                    point_cloud=cluster,
                    T_object_to_world=T_base_obj,
                    id_confidence=id_conf,
                    pose_confidence=pose_conf,
                    metrics={
                        "margin": float(margin),
                        "rms_nn": float(chosen["rms_nn"]),
                        "icp_fitness": float(chosen["icp_fitness"]),
                        "icp_inlier_rmse": float(chosen["icp_rmse"]),
                        "cad_cover_ratio": float(chosen["cad_cover_ratio"]),
                        "cad_cover_rmse": float(chosen["cad_cover_rmse"]),
                        "cad_cover_n": float(chosen["cad_cover_n"]),
                        "cad_cover_thresh": float(self.cfg.cad_cover_thresh),
                        **{k: float(v) for k, v in chosen["metrics"].items() if isinstance(v, (int, float, np.floating))},
                    },
                )
            )

            self.log.info(
                "accept",
                cluster=i,
                obj=obj_id,
                cad_cover=float(chosen["cad_cover_ratio"]),
                cad_cover_rmse=float(chosen["cad_cover_rmse"]),
                nn_rms=float(chosen["rms_nn"]),
                icp_rmse=float(chosen["icp_rmse"]),
                fitness=float(chosen["icp_fitness"]),
            )

        return out