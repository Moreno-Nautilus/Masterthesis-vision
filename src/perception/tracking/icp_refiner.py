"""
ICP-based pose refinement for 6DoF tracking.

"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


class ICPVariant(Enum):
    POINT_TO_POINT = "point_to_point"
    POINT_TO_PLANE = "point_to_plane"


@dataclass
class ICPConfig:
    """Configuration for ICP pose refinement."""
    
    # ICP variant
    variant: ICPVariant = ICPVariant.POINT_TO_POINT
    
    # Convergence criteria
    max_iterations: int = 30
    relative_fitness: float = 1e-6
    relative_rmse: float = 1e-6

    max_correspondence_distance: float = 0.02  # 2cm

    # Voxel downsampling for speed (meters, None = no downsampling)
    voxel_size: Optional[float] = 0.002  # 2mm
    
    # Outlier removal
    remove_statistical_outliers: bool = False
    nb_neighbors: int = 20
    std_ratio: float = 2.0

    mask_morph_close_kernel: int = 0
    mask_interior_erosion: int = 0


@dataclass
class ICPResult:
    """Result from ICP pose refinement."""
    T_refined: np.ndarray  # (4, 4) refined pose
    fitness: float  # Overlap ratio [0, 1], higher is better
    inlier_rmse: float  # RMSE of inlier correspondences (meters)
    num_inliers: int
    converged: bool
    elapsed_ms: float


class ICPRefiner:
    """
    Point cloud registration for pose refinement.
    """
    
    def __init__(self, cfg: Optional[ICPConfig] = None):
        self.cfg = cfg or ICPConfig()
        self._o3d = None
        self._model_cloud = None
        self._model_cloud_down = None
        
    def _lazy_import(self):
        """Lazy import Open3D."""
        if self._o3d is not None:
            return
        try:
            import open3d as o3d
            self._o3d = o3d
        except ImportError:
            raise ImportError("Open3D not installed. Run: pip install open3d")
    
    def set_model_cloud(
        self,
        vertices: np.ndarray,
        normals: Optional[np.ndarray] = None,
    ) -> None:
        """
        Set the model point cloud (from mesh).
        
        Args:
            vertices: (N, 3) float32 points in object frame
            normals: (N, 3) float32 normals, optional (computed if not given)
        """
        self._lazy_import()
        o3d = self._o3d

        # Wrap the model vertices in an Open3D cloud.
        self._model_cloud = o3d.geometry.PointCloud()
        self._model_cloud.points = o3d.utility.Vector3dVector(vertices.astype(np.float64))

        # Use supplied normals, or estimate them (needed for point-to-plane).
        if normals is not None:
            self._model_cloud.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
        else:
            self._model_cloud.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
            )

        # Pre-compute downsampled version for speed
        if self.cfg.voxel_size is not None:
            self._model_cloud_down = self._model_cloud.voxel_down_sample(self.cfg.voxel_size)
        else:
            self._model_cloud_down = self._model_cloud
            
        print(f"[ICPRefiner] Model cloud set: {len(self._model_cloud.points)} pts "
              f"-> {len(self._model_cloud_down.points)} pts (downsampled)")
    
   
    def set_model_from_mesh(self, mesh_path: str, num_points: int = 5000, scale: float = 0.01) -> None:
        """Load model cloud by sampling points from mesh."""
        self._lazy_import()
        o3d = self._o3d
        
        mesh = o3d.io.read_triangle_mesh(mesh_path)

        # Center the mesh (same as FP does)
        mesh.translate(-mesh.get_center())

        # Scale from mm to meters (same as FP)
        mesh.scale(scale, center=(0, 0, 0))

        mesh.compute_vertex_normals()

        # Sample a uniform point cloud off the mesh surface.
        cloud = mesh.sample_points_uniformly(number_of_points=num_points)

        # Re-center on the sampled points' centroid.
        pts = np.asarray(cloud.points)
        center_of_mass = pts.mean(axis=0)
        cloud.translate(-center_of_mass)

        vertices = np.asarray(cloud.points, dtype=np.float32)
        normals = np.asarray(cloud.normals, dtype=np.float32)
        
        self.set_model_cloud(vertices, normals=normals)

    def _depth_to_cloud(self, depth, rgb, mask, K):
        # Back-project masked depth pixels into a camera-frame point cloud.
        o3d = self._o3d
        H, W = depth.shape

        mask_bool = mask.astype(bool, copy=False)

        # Optionally clean the mask: close small holes / erode the border inward.
        close_k = int(self.cfg.mask_morph_close_kernel)
        erode_k = int(self.cfg.mask_interior_erosion)
        if close_k > 0 or erode_k > 0:
            import cv2
            m_u8 = mask_bool.astype(np.uint8)
            if close_k > 0:
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (close_k, close_k),
                )
                m_u8 = cv2.morphologyEx(m_u8, cv2.MORPH_CLOSE, kernel)
            if erode_k > 0:
                ekernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (erode_k, erode_k),
                )
                m_u8 = cv2.erode(m_u8, ekernel)
            mask_bool = m_u8.astype(bool, copy=False)

        # Crop to mask bbox first so we don't allocate full-image meshgrids or
        # index across the whole frame. The bbox is tight by construction.
        rows = np.flatnonzero(mask_bool.any(axis=1))
        if rows.size == 0:
            return o3d.geometry.PointCloud()
        cols = np.flatnonzero(mask_bool.any(axis=0))
        y0, y1 = int(rows[0]), int(rows[-1]) + 1
        x0, x1 = int(cols[0]), int(cols[-1]) + 1

        depth_c = depth[y0:y1, x0:x1]
        mask_c = mask_bool[y0:y1, x0:x1]
        h, w = depth_c.shape

        # Cache meshgrid per (cropped) image size — bbox sizes vary, so we
        # build the local grid on demand. It's small and cheap.
        uu = np.arange(x0, x1)
        vv = np.arange(y0, y1)
        grid_u, grid_v = np.meshgrid(uu, vv)

        # Pull the masked pixels' depth and image coordinates.
        z = depth_c[mask_c]
        u = grid_u[mask_c]
        v = grid_v[mask_c]

        # Keep only finite depths in a sane range.
        valid = (z > 0.01) & (z < 2.0) & np.isfinite(z)
        z, u, v = z[valid], u[valid], v[valid]

        if len(z) < 10:
            return o3d.geometry.PointCloud()

        # Pinhole un-projection to 3D camera coordinates.
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        points = np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z], axis=1).astype(np.float64)

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)

        # Skip outlier removal during tracking (Cutie mask is clean)
        if self.cfg.remove_statistical_outliers:
            cloud, _ = cloud.remove_statistical_outlier(
                nb_neighbors=self.cfg.nb_neighbors, std_ratio=self.cfg.std_ratio,
            )

        if self.cfg.voxel_size is not None:
            cloud = cloud.voxel_down_sample(self.cfg.voxel_size)

        # Only estimate normals if ICP variant needs them
        need_normals = self.cfg.variant == ICPVariant.POINT_TO_PLANE
        if need_normals and len(cloud.points) >= 10:
            normal_radius = max(0.01, 3.0 * (self.cfg.voxel_size or 0.002))
            cloud.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30)
            )
            cloud.orient_normals_towards_camera_location(camera_location=np.array([0.0, 0.0, 0.0]))

        return cloud

    def refine(
            self,
            depth: np.ndarray,
            rgb: np.ndarray,
            mask: np.ndarray,
            K: np.ndarray,
            T_init: np.ndarray,
        ) -> ICPResult:
            if self._model_cloud is None:
                raise RuntimeError("Call set_model_cloud() first")

            self._lazy_import()
            o3d = self._o3d

            t0 = time.time()

            # Extract observed point cloud from depth + mask
            observed_cloud = self._depth_to_cloud(depth, rgb, mask, K)

            # Too few points to register reliably → return the init pose unchanged.
            if len(observed_cloud.points) < 50:
                return ICPResult(
                    T_refined=T_init.copy(),
                    fitness=0.0,
                    inlier_rmse=float('inf'),
                    num_inliers=0,
                    converged=False,
                    elapsed_ms=(time.time() - t0) * 1000,
                )
            
            # Register the model cloud onto the observed cloud, seeded from T_init.
            source_model = self._model_cloud_down
            T_init64 = T_init.astype(np.float64)

            criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=self.cfg.max_iterations,
                relative_fitness=self.cfg.relative_fitness,
                relative_rmse=self.cfg.relative_rmse,
            )

            if self.cfg.variant == ICPVariant.POINT_TO_POINT:
                result = o3d.pipelines.registration.registration_icp(
                    source=source_model,
                    target=observed_cloud,
                    max_correspondence_distance=self.cfg.max_correspondence_distance,
                    init=T_init64,
                    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    criteria=criteria,
                )

            elif self.cfg.variant == ICPVariant.POINT_TO_PLANE:
                result = o3d.pipelines.registration.registration_icp(
                    source=source_model,
                    target=observed_cloud,
                    max_correspondence_distance=self.cfg.max_correspondence_distance,
                    init=T_init64,
                    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                    criteria=criteria,
                )

            else:
                raise ValueError(f"Unknown ICP variant: {self.cfg.variant}")

            # `init` was applied internally; result.transformation is already
            # the full T_refined.
            T_refined = np.asarray(result.transformation, dtype=np.float64)

            # Call it converged when overlap is high and inlier error is small.
            elapsed_ms = (time.time() - t0) * 1000
            converged = result.fitness > 0.3 and result.inlier_rmse < 0.005

            return ICPResult(
                T_refined=T_refined.astype(np.float32),
                fitness=float(result.fitness),
                inlier_rmse=float(result.inlier_rmse),
                num_inliers=len(result.correspondence_set),
                converged=converged,
                elapsed_ms=elapsed_ms,
            )
    
