"""
ICP-based pose refinement for 6DoF tracking.

Given a tracked mask and depth image, refines the object pose using
point cloud registration.

Primary: Colored ICP (uses both geometry and color)
Fallback: FilterReg, TEASER++ (if accuracy needs improvement)

Install: pip install open3d
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import copy


class ICPVariant(Enum):
    POINT_TO_POINT = "point_to_point"
    POINT_TO_PLANE = "point_to_plane"
    COLORED = "colored"
    GENERALIZED = "generalized"  # GICP


@dataclass
class ICPConfig:
    """Configuration for ICP pose refinement."""
    
    # ICP variant
    variant: ICPVariant = ICPVariant.COLORED
    
    # Convergence criteria
    max_iterations: int = 30
    relative_fitness: float = 1e-6
    relative_rmse: float = 1e-6
    
    # Max correspondence distance (meters)
    # Larger = more robust to initial misalignment, but slower
    max_correspondence_distance: float = 0.02  # 2cm
    
    # For colored ICP: weight of color vs geometry [0, 1]
    # 0 = geometry only, 1 = color only
    lambda_geometric: float = 0.968  # Default from Open3D
    
    # Voxel downsampling for speed (meters, None = no downsampling)
    voxel_size: Optional[float] = 0.002  # 2mm
    
    # Outlier removal
    remove_statistical_outliers: bool = False
    nb_neighbors: int = 20
    std_ratio: float = 2.0


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
    
    Usage:
        refiner = ICPRefiner(ICPConfig())
        refiner.set_model_cloud(mesh_vertices, mesh_colors)  # Once per object
        
        for frame in video:
            result = refiner.refine(
                depth=depth_image,
                rgb=rgb_image,
                mask=tracked_mask,
                K=camera_intrinsics,
                T_init=previous_pose,
            )
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
        colors: Optional[np.ndarray] = None,
        normals: Optional[np.ndarray] = None,
    ) -> None:
        """
        Set the model point cloud (from mesh).
        
        Args:
            vertices: (N, 3) float32 points in object frame
            colors: (N, 3) float32 RGB colors [0, 1], optional
            normals: (N, 3) float32 normals, optional (computed if not given)
        """
        self._lazy_import()
        o3d = self._o3d
        
        self._model_cloud = o3d.geometry.PointCloud()
        self._model_cloud.points = o3d.utility.Vector3dVector(vertices.astype(np.float64))
        
        if colors is not None:
            self._model_cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
            
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
        
        # Sample points
        cloud = mesh.sample_points_uniformly(number_of_points=num_points)
        
        # NEW: Re-center based on center of mass of sampled points
        pts = np.asarray(cloud.points)
        center_of_mass = pts.mean(axis=0)
        cloud.translate(-center_of_mass)
        
        vertices = np.asarray(cloud.points, dtype=np.float32)
        normals = np.asarray(cloud.normals, dtype=np.float32)
        colors = np.asarray(cloud.colors, dtype=np.float32) if cloud.has_colors() else None
        
        self.set_model_cloud(vertices, colors, normals)

    def _depth_to_cloud(self, depth, rgb, mask, K):
        o3d = self._o3d
        H, W = depth.shape

        # Cache meshgrid per image size
        if not hasattr(self, '_grid_cache') or self._grid_cache_shape != (H, W):
            uu = np.arange(W)
            vv = np.arange(H)
            self._grid_u, self._grid_v = np.meshgrid(uu, vv)
            self._grid_cache_shape = (H, W)

        mask_bool = mask.astype(bool)
        z = depth[mask_bool]
        u = self._grid_u[mask_bool]
        v = self._grid_v[mask_bool]

        valid = (z > 0.01) & (z < 2.0) & np.isfinite(z)
        z, u, v = z[valid], u[valid], v[valid]

        if len(z) < 10:
            return o3d.geometry.PointCloud()

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        points = np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z], axis=1).astype(np.float64)

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)

        # Only add colors if needed (colored ICP)
        need_colors = self.cfg.variant == ICPVariant.COLORED
        if need_colors:
            rgb_masked = rgb[mask_bool][valid]
            cloud.colors = o3d.utility.Vector3dVector(rgb_masked.astype(np.float64) / 255.0)

        # Skip outlier removal during tracking (Cutie mask is clean)
        if self.cfg.remove_statistical_outliers:
            cloud, _ = cloud.remove_statistical_outlier(
                nb_neighbors=self.cfg.nb_neighbors, std_ratio=self.cfg.std_ratio,
            )

        if self.cfg.voxel_size is not None:
            cloud = cloud.voxel_down_sample(self.cfg.voxel_size)

        # Only estimate normals if ICP variant needs them
        need_normals = self.cfg.variant in (ICPVariant.POINT_TO_PLANE, ICPVariant.COLORED, ICPVariant.GENERALIZED)
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
            
            if len(observed_cloud.points) < 50:
                return ICPResult(
                    T_refined=T_init.copy(),
                    fitness=0.0,
                    inlier_rmse=float('inf'),
                    num_inliers=0,
                    converged=False,
                    elapsed_ms=(time.time() - t0) * 1000,
                )
            
            # Transform model cloud by initial pose - MAKE A COPY
            model_transformed = o3d.geometry.PointCloud(self._model_cloud_down)
            model_transformed.transform(T_init.astype(np.float64))

           # Only compute normals if the ICP variant actually uses them
            need_normals = self.cfg.variant in (
                ICPVariant.POINT_TO_PLANE,
                ICPVariant.COLORED,
                ICPVariant.GENERALIZED,
            )
            if need_normals:
                model_transformed.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
                )
                model_transformed.orient_normals_towards_camera_location(
                    camera_location=np.array([0.0, 0.0, 0.0])
                )
                observed_cloud.orient_normals_towards_camera_location(
                    camera_location=np.array([0.0, 0.0, 0.0])
                )
            # Build convergence criteria
            criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=self.cfg.max_iterations,
                relative_fitness=self.cfg.relative_fitness,
                relative_rmse=self.cfg.relative_rmse,
            )
            
            # Run ICP based on variant
            if self.cfg.variant == ICPVariant.POINT_TO_POINT:
                result = o3d.pipelines.registration.registration_icp(
                    source=model_transformed,
                    target=observed_cloud,
                    max_correspondence_distance=self.cfg.max_correspondence_distance,
                    init=np.eye(4),
                    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    criteria=criteria,
                )
                
            elif self.cfg.variant == ICPVariant.POINT_TO_PLANE:
                result = o3d.pipelines.registration.registration_icp(
                    source=model_transformed,
                    target=observed_cloud,
                    max_correspondence_distance=self.cfg.max_correspondence_distance,
                    init=np.eye(4),
                    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                    criteria=criteria,
                )
                
            elif self.cfg.variant == ICPVariant.COLORED:
                result = o3d.pipelines.registration.registration_colored_icp(
                    source=model_transformed,
                    target=observed_cloud,
                    max_correspondence_distance=self.cfg.max_correspondence_distance,
                    init=np.eye(4),
                    estimation_method=o3d.pipelines.registration.TransformationEstimationForColoredICP(
                        lambda_geometric=self.cfg.lambda_geometric,
                    ),
                    criteria=criteria,
                )
                
            elif self.cfg.variant == ICPVariant.GENERALIZED:
                result = o3d.pipelines.registration.registration_generalized_icp(
                    source=model_transformed,
                    target=observed_cloud,
                    max_correspondence_distance=self.cfg.max_correspondence_distance,
                    init=np.eye(4),
                    criteria=criteria,
                )
            else:
                raise ValueError(f"Unknown ICP variant: {self.cfg.variant}")
            
            # Compose final transformation: T_refined = T_icp @ T_init
            T_icp = np.asarray(result.transformation, dtype=np.float64)
            T_refined = T_icp @ T_init.astype(np.float64)
            
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
    
