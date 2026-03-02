from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import open3d as o3d

from src.utils.se3 import SE3
from src.perception.pose_icp import estimate_pose_icp
from src.perception.segmentation import (
    remove_plane_ransac,
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
    T_object_to_world: SE3
    id_confidence: float
    pose_confidence: float
    metrics: dict[str, float]


@dataclass
class SceneResult:
    plane_model: np.ndarray
    points_wo_plane: np.ndarray
    clusters: list[np.ndarray]
    objects: list[DetectedObject]
    timings: list[StageTiming]


@dataclass
class PipelineConfig:
    # plane
    plane_distance_threshold: float = 0.003
    plane_ransac_n: int = 3
    plane_num_iterations: int = 2000

    # clustering
    dbscan_eps: float = 0.02
    dbscan_min_points: int = 30
    merge_dist: float = 0.05
    merge_z_overlap: float = 0.03

    # matching / gating
    voxel_size: float = 0.005
    enforce_one_to_one: bool = True

    # NN RMS gate + margin (margin is computed on NN RMS)
    max_rms_nn: float = 0.015
    min_margin: float = 1.5

    # soft ICP-quality gates (reject only if BOTH are bad)
    min_icp_fitness: float = 0.10
    max_icp_inlier_rmse: float = 0.02


def nn_rms(source_pts: np.ndarray, target_pts: np.ndarray) -> float:
    if len(source_pts) == 0 or len(target_pts) == 0:
        return float("inf")
    src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source_pts))
    tgt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target_pts))
    dists = np.asarray(src.compute_point_cloud_distance(tgt))
    if not np.isfinite(dists).all():
        return float("inf")
    return float(np.sqrt(np.mean(dists**2)))


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

    def run(self, scene_points_world: np.ndarray) -> SceneResult:
        timings: list[StageTiming] = []

        # Plane removal
        t0 = perf_counter()
        wo_plane, plane_pts, plane_model = remove_plane_ransac(
            scene_points_world,
            distance_threshold=self.cfg.plane_distance_threshold,
            ransac_n=self.cfg.plane_ransac_n,
            num_iterations=self.cfg.plane_num_iterations,
        )
        timings.append(StageTiming("plane_removal", perf_counter() - t0))
        self.log.info(
            "plane_removal",
            n_in=len(scene_points_world),
            n_plane=len(plane_pts),
            n_left=len(wo_plane),
        )

        # Clustering (+ merge fragments)
        t0 = perf_counter()
        clusters = cluster_dbscan(
            wo_plane, eps=self.cfg.dbscan_eps, min_points=self.cfg.dbscan_min_points
        )
        clusters = merge_close_clusters(
            clusters, merge_dist=self.cfg.merge_dist, z_overlap=self.cfg.merge_z_overlap
        )
        timings.append(StageTiming("clustering", perf_counter() - t0))
        self.log.info(
            "clustering",
            n_clusters=len(clusters),
            sizes=[len(c) for c in clusters[:5]],
        )

        # Identify + pose
        t0 = perf_counter()
        objects = self._identify_and_pose(clusters)
        timings.append(StageTiming("identify_pose", perf_counter() - t0))
        self.log.info("identify_pose", n_objects=len(objects))

        return SceneResult(
            plane_model=np.asarray(plane_model, dtype=float),
            points_wo_plane=wo_plane,
            clusters=clusters,
            objects=objects,
            timings=timings,
        )

    def _identify_and_pose(self, clusters: list[np.ndarray]) -> list[DetectedObject]:
        if len(clusters) == 0:
            return []

        cad_ids = list(self.cad_library.keys())
        cad_pts = [self.cad_library[k] for k in cad_ids]

        # score matrix used for choosing best CAD per cluster:
        # we choose by ICP inlier RMSE (lower is better)
        scores = np.full((len(clusters), len(cad_ids)), np.inf, dtype=float)

        poses_obs_to_cad: list[list[SE3]] = [
            [SE3.identity() for _ in cad_ids] for _ in clusters
        ]
        pair_metrics: dict[tuple[int, int], dict[str, float]] = {}

        # --- compute all pairs
        for i, cluster in enumerate(clusters):
            for j, model in enumerate(cad_pts):
                T_obs_to_cad, icp_metrics = estimate_pose_icp(
                    cluster, model, voxel_size=self.cfg.voxel_size
                )
                aligned = T_obs_to_cad.transform_points(cluster)
                rms_nn = nn_rms(aligned, model)

                icp_fit = float(icp_metrics.get("icp_fitness", np.nan))
                icp_rmse = float(icp_metrics.get("icp_inlier_rmse", np.nan))

                scores[i, j] = icp_rmse
                poses_obs_to_cad[i][j] = T_obs_to_cad

                pair_metrics[(i, j)] = {
                    "rms_nn": float(rms_nn),
                    "icp_fitness": icp_fit,
                    "icp_inlier_rmse": icp_rmse,
                    "ransac_fitness": float(icp_metrics.get("ransac_fitness", np.nan)),
                    "ransac_inlier_rmse": float(icp_metrics.get("ransac_inlier_rmse", np.nan)),
                    "n_candidates": float(icp_metrics.get("n_candidates", np.nan)),
                }

        used: set[int] = set()
        out: list[DetectedObject] = []

        # --- pick best CAD per cluster + gates
        for i in range(len(clusters)):
            # sort CADs by ICP inlier RMSE (lower is better)
            order = np.argsort(scores[i])

            # margin computed on NN RMS (stable)
            nn_row = np.array([pair_metrics[(i, j)]["rms_nn"] for j in range(len(cad_ids))], dtype=float)
            nn_order = np.argsort(nn_row)
            nn_best = float(nn_row[int(nn_order[0])])
            nn_second = float(nn_row[int(nn_order[1])]) if len(nn_order) > 1 else float("inf")
            margin = (nn_second / nn_best) if nn_best > 1e-12 else float("inf")

            if margin < self.cfg.min_margin:
                self.log.warn("reject_margin", cluster=i, margin=margin, min_margin=self.cfg.min_margin)
                continue

            chosen = None
            chosen_metrics = None

            # try candidates in ICP-RMSE order until one passes gates
            for j in order:
                j = int(j)

                if self.cfg.enforce_one_to_one and j in used:
                    continue

                m = pair_metrics[(i, j)]
                fit = float(m.get("icp_fitness", 0.0))
                inlier = float(m.get("icp_inlier_rmse", np.inf))
                rms_nn = float(m.get("rms_nn", np.inf))

                # HARD gate: NN RMS
                if (not np.isfinite(rms_nn)) or rms_nn > self.cfg.max_rms_nn:
                    continue

                # SOFT gate: reject only if BOTH bad
                if (fit < self.cfg.min_icp_fitness) and (inlier > self.cfg.max_icp_inlier_rmse):
                    continue

                chosen = j
                chosen_metrics = m
                break

            if chosen is None:
                # helpful debug: report best few candidates
                top = [int(x) for x in order[:3]]
                self.log.warn(
                    "reject_cluster_no_candidate",
                    cluster=i,
                    top=[cad_ids[t] for t in top],
                    top_nn=[float(pair_metrics[(i, t)]["rms_nn"]) for t in top],
                    top_icp=[float(pair_metrics[(i, t)]["icp_inlier_rmse"]) for t in top],
                )
                continue

            used.add(chosen)

            # output pose
            T_obs_to_cad = poses_obs_to_cad[i][chosen]
            T_object_to_world = T_obs_to_cad.inverse()

            # optional sanity log
            cad_world = T_object_to_world.transform_points(self.cad_library[cad_ids[chosen]])
            cluster_world = clusters[i]
            c_err = float(np.linalg.norm(cad_world.mean(axis=0) - cluster_world.mean(axis=0)))
            self.log.info("pose_sanity", cluster=i, centroid_err_m=c_err)

            # confidences
            id_conf = float(np.clip((margin - self.cfg.min_margin) / 2.0, 0.0, 1.0))
            pose_conf = float(np.clip(1.0 - float(chosen_metrics["rms_nn"]) / self.cfg.max_rms_nn, 0.0, 1.0))

            out.append(
                DetectedObject(
                    object_id=cad_ids[chosen],
                    point_cloud=clusters[i],
                    T_object_to_world=T_object_to_world,
                    id_confidence=id_conf,
                    pose_confidence=pose_conf,
                    metrics={
                        "margin": float(margin),
                        "score_icp_inlier_rmse": float(chosen_metrics.get("icp_inlier_rmse", np.nan)),
                        **chosen_metrics,
                    },
                )
            )

            self.log.info(
                "accept",
                cluster=i,
                obj=cad_ids[chosen],
                nn_rms=float(chosen_metrics.get("rms_nn", np.nan)),
                icp_rmse=float(chosen_metrics.get("icp_inlier_rmse", np.nan)),
                fitness=float(chosen_metrics.get("icp_fitness", np.nan)),
                margin=float(margin),
            )

        return out