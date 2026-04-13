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
    remove_statistical_outliers: bool = True
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
        
        vertices = np.asarray(cloud.points, dtype=np.float32)
        normals = np.asarray(cloud.normals, dtype=np.float32)
        colors = np.asarray(cloud.colors, dtype=np.float32) if cloud.has_colors() else None
        
        self.set_model_cloud(vertices, colors, normals)

    # CLAUDE DEPTH TO CLOUD  
    # def _depth_to_cloud(
    #     self,
    #     depth: np.ndarray,
    #     rgb: np.ndarray,
    #     mask: np.ndarray,
    #     K: np.ndarray,
    # ) -> "open3d.geometry.PointCloud":
    #     """Convert masked depth image to colored point cloud."""
    #     o3d = self._o3d
        
    #     H, W = depth.shape
        
    #     # Create pixel coordinate grid
    #     u = np.arange(W)
    #     v = np.arange(H)
    #     u, v = np.meshgrid(u, v)
        
    #     # Apply mask
    #     mask_bool = mask.astype(bool)
    #     z = depth[mask_bool]
    #     u = u[mask_bool]
    #     v = v[mask_bool]
        
    #     # Filter invalid depth
    #     valid = (z > 0.01) & (z < 2.0) & np.isfinite(z)
    #     z = z[valid]
    #     u = u[valid]
    #     v = v[valid]
        
    #     if len(z) < 10:
    #         # Not enough points
    #         return o3d.geometry.PointCloud()
        
    #     # Unproject to 3D
    #     fx, fy = K[0, 0], K[1, 1]
    #     cx, cy = K[0, 2], K[1, 2]
        
    #     x = (u - cx) * z / fx
    #     y = (v - cy) * z / fy
        
    #     points = np.stack([x, y, z], axis=1).astype(np.float64)
        
    #     # Get colors
    #     rgb_masked = rgb[mask_bool][valid]
    #     colors = rgb_masked.astype(np.float64) / 255.0
        
    #     # Create cloud
    #     cloud = o3d.geometry.PointCloud()
    #     cloud.points = o3d.utility.Vector3dVector(points)
    #     cloud.colors = o3d.utility.Vector3dVector(colors)
        
    #     # Estimate normals
    #     cloud.estimate_normals(
    #         search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
    #     )
        
    #     # Optional: outlier removal
    #     if self.cfg.remove_statistical_outliers:
    #         cloud, _ = cloud.remove_statistical_outlier(
    #             nb_neighbors=self.cfg.nb_neighbors,
    #             std_ratio=self.cfg.std_ratio,
    #         )
            
    #     # Optional: downsample
    #     if self.cfg.voxel_size is not None:
    #         cloud = cloud.voxel_down_sample(self.cfg.voxel_size)
            
    #     return cloud
    # GPT DEPTH TO CLOUD
    def _depth_to_cloud(
        self,
        depth: np.ndarray,
        rgb: np.ndarray,
        mask: np.ndarray,
        K: np.ndarray,
    ) -> "open3d.geometry.PointCloud":
        o3d = self._o3d

        H, W = depth.shape

        u = np.arange(W)
        v = np.arange(H)
        u, v = np.meshgrid(u, v)

        mask_bool = mask.astype(bool)
        z = depth[mask_bool]
        u = u[mask_bool]
        v = v[mask_bool]

        valid = (z > 0.01) & (z < 2.0) & np.isfinite(z)
        z = z[valid]
        u = u[valid]
        v = v[valid]

        if len(z) < 10:
            return o3d.geometry.PointCloud()

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        points = np.stack([x, y, z], axis=1).astype(np.float64)

        rgb_masked = rgb[mask_bool][valid]
        colors = rgb_masked.astype(np.float64) / 255.0

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        cloud.colors = o3d.utility.Vector3dVector(colors)

        # 1) clean first
        if self.cfg.remove_statistical_outliers:
            cloud, _ = cloud.remove_statistical_outlier(
                nb_neighbors=self.cfg.nb_neighbors,
                std_ratio=self.cfg.std_ratio,
            )

        if self.cfg.voxel_size is not None:
            cloud = cloud.voxel_down_sample(self.cfg.voxel_size)

        # 2) THEN estimate normals on the final cloud
        if len(cloud.points) >= 10:
            normal_radius = max(0.01, 3.0 * (self.cfg.voxel_size or 0.002))
            cloud.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=normal_radius,
                    max_nn=30,
                )
            )
            cloud.orient_normals_towards_camera_location(
                camera_location=np.array([0.0, 0.0, 0.0])
            )

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

        model_transformed.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
        )
        model_transformed.orient_normals_towards_camera_location(camera_location=np.array([0.0, 0.0, 0.0]))

        # Orient model normals toward camera (camera is at origin in camera frame)
        observed_cloud.orient_normals_towards_camera_location(camera_location=np.array([0.0, 0.0, 0.0]))
        # # Fix degenerate normals - renormalize
        # obs_normals_arr = np.asarray(observed_cloud.normals)
        # magnitudes = np.linalg.norm(obs_normals_arr, axis=1, keepdims=True)
        # # Replace near-zero normals with [0, 0, -1] (pointing toward camera)
        # bad_mask = magnitudes.flatten() < 0.1
        # obs_normals_arr[bad_mask] = [0.0, 0.0, -1.0]
        # # Renormalize all
        # magnitudes = np.linalg.norm(obs_normals_arr, axis=1, keepdims=True)
        # magnitudes[magnitudes < 1e-6] = 1.0  # Avoid division by zero
        # obs_normals_arr = obs_normals_arr / magnitudes
        # observed_cloud.normals = o3d.utility.Vector3dVector(obs_normals_arr)

        # Debug prints
        obs_pts = np.asarray(observed_cloud.points)
        trans_pts = np.asarray(model_transformed.points)
        print(f"[ICP DEBUG] observed: {len(obs_pts)} pts, bounds: {obs_pts.min(axis=0)} to {obs_pts.max(axis=0)}")
        print(f"[ICP DEBUG] model: {len(trans_pts)} pts, bounds: {trans_pts.min(axis=0)} to {trans_pts.max(axis=0)}")
        print(f"[ICP DEBUG] observed has_normals: {observed_cloud.has_normals()}, model has_normals: {model_transformed.has_normals()}")
        obs_normals = np.asarray(observed_cloud.normals)
        model_normals = np.asarray(model_transformed.normals)
        print(f"[ICP DEBUG] observed normals z-mean: {obs_normals[:, 2].mean():.3f}")
        print(f"[ICP DEBUG] model normals z-mean: {model_normals[:, 2].mean():.3f}")
        print(f"[ICP DEBUG] observed normals sample: {obs_normals[:3]}")
        print(f"[ICP DEBUG] model normals sample: {model_normals[:3]}")
        from scipy.spatial import cKDTree
        tree = cKDTree(trans_pts)
        distances, _ = tree.query(obs_pts, k=1)
        print(f"[ICP DEBUG] min distance between clouds: {distances.min():.4f}m, max: {distances.max():.4f}m, mean: {distances.mean():.4f}m")

        obs_norms_mag = np.linalg.norm(obs_normals, axis=1)
        model_norms_mag = np.linalg.norm(model_normals, axis=1)
        print(f"[ICP DEBUG] observed normals magnitude: min={obs_norms_mag.min():.4f}, max={obs_norms_mag.max():.4f}")
        print(f"[ICP DEBUG] model normals magnitude: min={model_norms_mag.min():.4f}, max={model_norms_mag.max():.4f}")
        print(f"[ICP DEBUG] target (model) has {len(model_transformed.normals)} normals")
        print(f"[ICP DEBUG] source (observed) has {len(observed_cloud.normals)} normals")
        o3d.io.write_point_cloud("/tmp/observed.pcd", observed_cloud)
        o3d.io.write_point_cloud("/tmp/model.pcd", model_transformed)
        print("[ICP DEBUG] Saved clouds to /tmp/observed.pcd and /tmp/model.pcd")
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
        
        print(f"[ICP DEBUG] result: fitness={result.fitness:.4f}, rmse={result.inlier_rmse:.6f}, correspondences={len(result.correspondence_set)}")
        
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
# =============================================================================
# FilterReg fallback (more robust, ~25ms)
# =============================================================================

@dataclass
class FilterRegConfig:
    """Configuration for FilterReg registration."""
    max_iterations: int = 50
    tol: float = 1e-5
    sigma2: Optional[float] = None  # Noise variance, None = estimate
    w: float = 0.0  # Outlier weight [0, 1]


class FilterRegRefiner:
    """
    FilterReg: Gaussian mixture model based registration.
    More robust to outliers and partial overlap than ICP.
    
    Install: pip install probreg
    """
    
    def __init__(self, cfg: Optional[FilterRegConfig] = None):
        self.cfg = cfg or FilterRegConfig()
        self._probreg = None
        self._model_cloud = None
        
    def _lazy_import(self):
        if self._probreg is not None:
            return
        try:
            import probreg
            self._probreg = probreg
        except ImportError:
            raise ImportError("probreg not installed. Run: pip install probreg")
            
    # Similar interface to ICPRefiner...
    # Implementation would follow same pattern


# =============================================================================
# TEASER++ fallback (global registration, ~30ms)
# =============================================================================

@dataclass  
class TeaserConfig:
    """Configuration for TEASER++ registration."""
    noise_bound: float = 0.01  # 1cm
    cbar2: float = 1.0
    rotation_gnc_factor: float = 1.4
    rotation_max_iterations: int = 100
    rotation_cost_threshold: float = 1e-6


class TeaserRefiner:
    """
    TEASER++: Certifiably robust point cloud registration.
    Handles up to 99% outliers. Good for re-initialization.
    
    Install: pip install teaserpp-python
    """
    
    def __init__(self, cfg: Optional[TeaserConfig] = None):
        self.cfg = cfg or TeaserConfig()
        self._teaser = None
        self._model_cloud = None
        
    def _lazy_import(self):
        if self._teaser is not None:
            return
        try:
            import teaserpp_python
            self._teaser = teaserpp_python
        except ImportError:
            raise ImportError("TEASER++ not installed. Run: pip install teaserpp-python")
            
   