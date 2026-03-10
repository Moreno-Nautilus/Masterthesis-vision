from __future__ import annotations

from dataclasses import dataclass
import time
import numpy as np
import open3d as o3d

from src.perception.pose_icp_fast import estimate_pose_icp


@dataclass
class SceneObject:
    object_id: str
    T_object_to_world: object
    point_cloud: np.ndarray


@dataclass
class SceneResult:
    points_world: np.ndarray
    objects: list[SceneObject]


@dataclass
class PipelineConfig:
    # point cloud preprocessing
    voxel_size: float = 0.005

    # dominant plane removal
    plane_distance_threshold: float = 0.008
    plane_ransac_n: int = 3
    plane_num_iterations: int = 400

    # clustering
    dbscan_eps: float = 0.035
    dbscan_min_points: int = 25
    max_clusters_considered: int = 6

    # subsampling / speed guards
    max_points_plane: int = 15000
    max_points_cluster: int = 6000
    icp_max_points: int = 1200

    # acceptance / rejection
    accept_icp_rmse_max: float = 0.015
    accept_icp_fitness_min: float = 0.15
    accept_cluster_points_min: int = 120

    # duplicate suppression
    duplicate_center_dist_min: float = 0.04

    # object-count control
    # None = dynamic number of objects
    expected_num_objects: int | None = None
    max_objects_returned: int = 3

    # compatibility fields
    max_rms_nn: float = 0.020
    min_margin: float = 1.2


def _pcd(points: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    if points.size != 0:
        pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    return pcd


def _subsample(points: np.ndarray, max_pts: int) -> np.ndarray:
    if points.shape[0] <= max_pts:
        return points
    stride = max(1, int(np.ceil(points.shape[0] / max_pts)))
    return points[::stride]


class GraspPerceptionPipeline:
    def __init__(self, cad_library: dict[str, np.ndarray], cfg: PipelineConfig | None = None):
        self.cad_library = cad_library
        self.cfg = cfg or PipelineConfig()

    def _remove_plane(self, points: np.ndarray) -> np.ndarray:
        if points.shape[0] == 0:
            return points

        pts_plane = _subsample(points, self.cfg.max_points_plane)
        if pts_plane.shape[0] < self.cfg.plane_ransac_n:
            return points

        pcd = _pcd(pts_plane)

        try:
            plane_model, _ = pcd.segment_plane(
                distance_threshold=self.cfg.plane_distance_threshold,
                ransac_n=self.cfg.plane_ransac_n,
                num_iterations=self.cfg.plane_num_iterations,
            )
        except RuntimeError:
            return points

        a, b, c, d = plane_model
        normal = np.array([a, b, c], dtype=np.float64)
        denom = np.linalg.norm(normal)
        if denom < 1e-9:
            return points

        dist = np.abs(points @ normal + d) / denom
        mask = dist > self.cfg.plane_distance_threshold
        return points[mask]

    def _cluster(self, points: np.ndarray) -> list[np.ndarray]:
        if points.shape[0] == 0:
            return []

        pts = _subsample(points, self.cfg.max_points_cluster)
        if pts.shape[0] < self.cfg.dbscan_min_points:
            return []

        pcd = _pcd(pts)
        labels = np.array(
            pcd.cluster_dbscan(
                eps=self.cfg.dbscan_eps,
                min_points=self.cfg.dbscan_min_points,
                print_progress=False,
            )
        )

        clusters: list[np.ndarray] = []
        for l in np.unique(labels):
            if l < 0:
                continue
            mask = labels == l
            cluster = pts[mask]
            if cluster.shape[0] > 0:
                clusters.append(cluster)

        clusters.sort(key=lambda x: x.shape[0], reverse=True)
        return clusters[: self.cfg.max_clusters_considered]

    def _extract_pose_and_metrics(self, out):
        """
        Supports several possible return styles from estimate_pose_icp(...):
          - (T, float_score)
          - (T, dict_metrics)
          - {"T": T, ...metrics...}
          - T
        Returns:
          T, rmse, fitness
        """
        T = None
        rmse = np.inf
        fitness = 0.0

        if isinstance(out, (tuple, list)) and len(out) >= 2:
            T = out[0]
            raw_score = out[1]

            if isinstance(raw_score, (int, float, np.floating)):
                rmse = float(raw_score)

            elif isinstance(raw_score, dict):
                if "icp_inlier_rmse" in raw_score:
                    rmse = float(raw_score["icp_inlier_rmse"])
                elif "rmse" in raw_score:
                    rmse = float(raw_score["rmse"])
                elif "nn_rmse" in raw_score:
                    rmse = float(raw_score["nn_rmse"])
                elif "error" in raw_score:
                    rmse = float(raw_score["error"])
                elif "score" in raw_score:
                    rmse = float(raw_score["score"])

                if "icp_fitness" in raw_score:
                    fitness = float(raw_score["icp_fitness"])
                elif "fitness" in raw_score:
                    fitness = float(raw_score["fitness"])

        elif isinstance(out, dict):
            T = out.get("T", out.get("transform", out.get("pose", None)))

            if "icp_inlier_rmse" in out:
                rmse = float(out["icp_inlier_rmse"])
            elif "rmse" in out:
                rmse = float(out["rmse"])
            elif "nn_rmse" in out:
                rmse = float(out["nn_rmse"])
            elif "error" in out:
                rmse = float(out["error"])
            elif "score" in out:
                rmse = float(out["score"])

            if "icp_fitness" in out:
                fitness = float(out["icp_fitness"])
            elif "fitness" in out:
                fitness = float(out["fitness"])

        else:
            T = out
            rmse = 0.0
            fitness = 1.0

        return T, rmse, fitness

    def _estimate_object(self, cluster: np.ndarray) -> tuple[SceneObject | None, float]:
        if cluster.shape[0] < self.cfg.accept_cluster_points_min:
            return None, np.inf

        cluster_small = _subsample(cluster, self.cfg.icp_max_points)

        best_obj: SceneObject | None = None
        best_score = np.inf

        for obj_id, cad in self.cad_library.items():
            if cad.shape[0] == 0:
                continue

            cad_small = _subsample(cad, self.cfg.icp_max_points)

            try:
                out = estimate_pose_icp(cluster_small, cad_small)
            except Exception:
                continue

            T, rmse, fitness = self._extract_pose_and_metrics(out)

            if T is None:
                continue

            # reject weak fits
            if rmse > self.cfg.accept_icp_rmse_max:
                continue
            if fitness < self.cfg.accept_icp_fitness_min:
                continue

            # lower is better
            score = rmse - 0.01 * fitness

            if score < best_score:
                best_score = score
                best_obj = SceneObject(
                    object_id=obj_id,
                    T_object_to_world=T,
                    point_cloud=cluster_small,
                )

        return best_obj, best_score

    def _too_close_to_selected(self, obj: SceneObject, selected: list[SceneObject]) -> bool:
        """
        Duplicate suppression: if two detections are very close in 3D, keep only the better one.
        """
        try:
            t = np.asarray(obj.T_object_to_world.t, dtype=float).reshape(3)
        except Exception:
            return False

        for other in selected:
            try:
                t_other = np.asarray(other.T_object_to_world.t, dtype=float).reshape(3)
            except Exception:
                continue

            if np.linalg.norm(t - t_other) < self.cfg.duplicate_center_dist_min:
                return True

        return False

    def _target_object_count(self) -> int:
        if self.cfg.expected_num_objects is not None:
            return max(0, int(self.cfg.expected_num_objects))
        return max(0, int(self.cfg.max_objects_returned))

    def run(self, points_world: np.ndarray) -> SceneResult:
        points_world = np.asarray(points_world, dtype=np.float32).reshape(-1, 3)

        if points_world.size == 0:
            return SceneResult(points_world=np.zeros((0, 3), dtype=np.float32), objects=[])

        t0 = time.perf_counter()

        pcd = _pcd(points_world)
        pcd = pcd.voxel_down_sample(self.cfg.voxel_size)
        points = np.asarray(pcd.points, dtype=np.float32)

        t1 = time.perf_counter()

        if points.size == 0:
            return SceneResult(points_world=np.zeros((0, 3), dtype=np.float32), objects=[])

        points = self._remove_plane(points)

        t2 = time.perf_counter()

        if points.size == 0:
            return SceneResult(points_world=np.zeros((0, 3), dtype=np.float32), objects=[])

        clusters = self._cluster(points)

        t3 = time.perf_counter()

        candidates: list[tuple[float, SceneObject]] = []

        for cluster in clusters:
            obj, score = self._estimate_object(cluster)
            if obj is not None:
                candidates.append((score, obj))

        candidates.sort(key=lambda x: x[0])

        target_count = self._target_object_count()
        objects: list[SceneObject] = []

        for score, obj in candidates:
            if self._too_close_to_selected(obj, objects):
                continue
            objects.append(obj)
            if len(objects) >= target_count:
                break

        t4 = time.perf_counter()

        print(
            f"[TIMING pipeline_fast] "
            f"voxel={(t1 - t0) * 1000:.1f} ms | "
            f"plane={(t2 - t1) * 1000:.1f} ms | "
            f"cluster={(t3 - t2) * 1000:.1f} ms | "
            f"icp={(t4 - t3) * 1000:.1f} ms | "
            f"pts_in={points_world.shape[0]} pts_after={points.shape[0]} "
            f"clusters={len(clusters)} accepted={len(objects)} candidates={len(candidates)}"
        )

        return SceneResult(points_world=points, objects=objects)