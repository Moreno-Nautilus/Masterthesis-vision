from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R

@dataclass
class FoundationPoseConfig:
    repo_root: str = "external/FoundationPose"
    weights_dir: str = "external/FoundationPose/weights"
    debug_dir: str = "outputs/foundationpose/fp_internal_debug"
    debug: int = 0
    est_refine_iter: int = 5
    mesh_scale: float = 0.01  # STL is in mm, convert to meters


def generate_pose_perturbations(
    T_prior: np.ndarray,
    n_trans: int = 3,
    n_rot: int = 3,
    max_trans_m: float = 0.05,
    max_rot_deg: float = 15.0,
) -> np.ndarray:
    """
    Generate pose perturbations around a prior pose.
    
    Args:
        T_prior: (4,4) prior pose matrix
        n_trans: number of translation samples per axis (total = n_trans^3)
        n_rot: number of rotation samples per axis (total = n_rot^3)
        max_trans_m: max translation perturbation in meters
        max_rot_deg: max rotation perturbation in degrees
        
    Returns:
        poses: (N, 4, 4) array of candidate poses
    """
    # Generate translation offsets
    if n_trans > 1:
        trans_offsets = np.linspace(-max_trans_m, max_trans_m, n_trans)
    else:
        trans_offsets = np.array([0.0])
    
    # Generate rotation offsets (in degrees)
    if n_rot > 1:
        rot_offsets = np.linspace(-max_rot_deg, max_rot_deg, n_rot)
    else:
        rot_offsets = np.array([0.0])
    
    poses = []
    
    # Get prior rotation and translation
    R_prior = T_prior[:3, :3]
    t_prior = T_prior[:3, 3]
    
    for dx in trans_offsets:
        for dy in trans_offsets:
            for dz in trans_offsets:
                for rx in rot_offsets:
                    for ry in rot_offsets:
                        for rz in rot_offsets:
                            # Create delta rotation
                            dR = R.from_euler('xyz', [rx, ry, rz], degrees=True).as_matrix()
                            
                            # Apply perturbation
                            R_new = dR @ R_prior
                            t_new = t_prior + np.array([dx, dy, dz])
                            
                            # Build pose matrix
                            T_new = np.eye(4, dtype=np.float32)
                            T_new[:3, :3] = R_new
                            T_new[:3, 3] = t_new
                            poses.append(T_new)
    
    return np.array(poses, dtype=np.float32)


def generate_pose_perturbations_compact(
    T_prior: np.ndarray,
    n_samples: int = 27,
    max_trans_m: float = 0.04,
    max_rot_deg: float = 12.0,
) -> np.ndarray:
    """
    Generate a compact set of pose perturbations using random sampling.
    More efficient than grid sampling for larger candidate counts.
    
    Args:
        T_prior: (4,4) prior pose matrix
        n_samples: total number of candidates to generate
        max_trans_m: max translation perturbation in meters
        max_rot_deg: max rotation perturbation in degrees
        
    Returns:
        poses: (N, 4, 4) array of candidate poses (includes original)
    """
    poses = [T_prior.copy()]  # Always include the original
    
    R_prior = T_prior[:3, :3]
    t_prior = T_prior[:3, 3]
    
    # Generate random perturbations
    np.random.seed(42)  # Reproducible for debugging
    for _ in range(n_samples - 1):
        # Random translation offset
        dt = np.random.uniform(-max_trans_m, max_trans_m, 3).astype(np.float32)
        
        # Random rotation offset
        dr_euler = np.random.uniform(-max_rot_deg, max_rot_deg, 3)
        dR = R.from_euler('xyz', dr_euler, degrees=True).as_matrix()
        
        # Apply perturbation
        T_new = np.eye(4, dtype=np.float32)
        T_new[:3, :3] = (dR @ R_prior).astype(np.float32)
        T_new[:3, 3] = t_prior + dt
        poses.append(T_new)
    
    return np.array(poses, dtype=np.float32)


@dataclass
class FoundationPoseResult:
    object_id: str
    mesh_path: str
    T_object_camera: np.ndarray   # 4x4 homogeneous transform (6D pose representation)
    mask_area: int
    debug_dir: str


class FoundationPoseWrapper:
    """
    Thin lazy wrapper around FoundationPose registration.

    Current scope:
    - model-based pose initialization from RGB + depth + K + mask + mesh
    - no tracking yet

    Notes
    -----
    We intentionally keep imports lazy so the rest of the repo can still run
    even when FoundationPose dependencies are only available in its own env.
    """

    def __init__(self, cfg: FoundationPoseConfig | None = None) -> None:
        self.cfg = cfg or FoundationPoseConfig()

        self.repo_root = Path(self.cfg.repo_root).resolve()
        self.weights_dir = Path(self.cfg.weights_dir).resolve()
        self.debug_dir = Path(self.cfg.debug_dir).resolve()
        self.debug_dir.mkdir(parents=True, exist_ok=True)

        self._imports_ready = False

        self._trimesh = None
        self._dr = None
        self._FoundationPose = None
        self._ScorePredictor = None
        self._PoseRefinePredictor = None

        self._mesh_path_loaded: str | None = None
        self._est = None
        self._mesh = None
        self._glctx = None
        self._scorer = None
        self._refiner = None

    def _ensure_repo_on_path(self) -> None:
        if not self.repo_root.exists():
            raise FileNotFoundError(f"FoundationPose repo_root does not exist: {self.repo_root}")

        repo_root_str = str(self.repo_root)
        if repo_root_str not in sys.path:
            sys.path.insert(0, repo_root_str)

    def _prepare_env(self) -> None:
        if not self.weights_dir.exists():
            raise FileNotFoundError(f"FoundationPose weights_dir does not exist: {self.weights_dir}")

        # Helpful for code that assumes cwd-relative weights lookup
        os.environ.setdefault("FOUNDATIONPOSE_WEIGHTS_DIR", str(self.weights_dir))
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", os.environ.get("TORCH_CUDA_ARCH_LIST", ""))

    def _lazy_imports(self) -> None:
        if self._imports_ready:
            return

        self._ensure_repo_on_path()
        self._prepare_env()

        import trimesh
        import nvdiffrast.torch as dr
        import learning.training.predict_score as ps
        #print(f"[FP DEBUG import] predict_score file = {ps.__file__}")

        from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor

        self._trimesh = trimesh
        self._dr = dr
        self._FoundationPose = FoundationPose
        self._ScorePredictor = ScorePredictor
        self._PoseRefinePredictor = PoseRefinePredictor

        self._imports_ready = True

    @staticmethod
    def _sanitize_rgb(rgb: np.ndarray) -> np.ndarray:
        rgb = np.asarray(rgb)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"Expected rgb shape (H, W, 3), got {rgb.shape}")
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(rgb)

    @staticmethod
    def _sanitize_depth(depth: np.ndarray) -> np.ndarray:
        depth = np.asarray(depth, dtype=np.float32)
        if depth.ndim == 3:
            depth = depth[..., 0]
        depth = depth.astype(np.float32, copy=False)
        depth[~np.isfinite(depth)] = 0.0
        depth[depth < 0.0] = 0.0
        return np.ascontiguousarray(depth)

    @staticmethod
    def _sanitize_K(K: np.ndarray) -> np.ndarray:
        K = np.asarray(K, dtype=np.float32)
        if K.shape != (3, 3):
            raise ValueError(f"Expected K shape (3, 3), got {K.shape}")
        return np.ascontiguousarray(K)

    @staticmethod
    def _sanitize_mask(mask: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
        mask = np.asarray(mask).astype(bool)
        if mask.shape != hw:
            raise ValueError(f"Mask shape {mask.shape} does not match image shape {hw}")
        if int(mask.sum()) <= 0:
            raise ValueError("FoundationPose received an empty mask")
        return np.ascontiguousarray(mask)

    def _build_estimator(self, *, object_id: str, mesh_path: str) -> None:
        self._lazy_imports()

        mesh_path = str(Path(mesh_path).resolve())
        if not Path(mesh_path).exists():
            raise FileNotFoundError(f"Mesh file does not exist: {mesh_path}")

        if self._est is not None and self._mesh_path_loaded == mesh_path:
            return

        mesh = self._trimesh.load(mesh_path)
        # Center mesh at origin (required for correct FP pose)
        mesh.vertices -= mesh.centroid
        mesh_scale = self.cfg.mesh_scale

        # Scale mesh to meters if needed
        if self.cfg.mesh_scale != 1.0:
            mesh.apply_scale(self.cfg.mesh_scale)

        # trimesh usually computes these lazily; force availability
        _ = mesh.vertex_normals

        # trimesh usually computes these lazily; force availability
        _ = mesh.vertex_normals

        object_debug_dir = self.debug_dir / object_id
        object_debug_dir.mkdir(parents=True, exist_ok=True)

        scorer = self._ScorePredictor()
        refiner = self._PoseRefinePredictor()

        import torch

        if hasattr(scorer, "model") and scorer.model is not None:
            scorer.model = scorer.model.float().cuda().eval()
        # if hasattr(refiner, "model") and refiner.model is not None:
        #     refiner.model = refiner.model.float().cuda().eval()

        # DEBUG: print model param dtype/device once when estimator is built
        if hasattr(scorer, "model") and scorer.model is not None:
            p = next(scorer.model.parameters())
            #print(f"[FP DEBUG] scorer model dtype={p.dtype}, device={p.device}")
        if hasattr(refiner, "model") and refiner.model is not None:
            p = next(refiner.model.parameters())
            #print(f"[FP DEBUG] refiner model dtype={p.dtype}, device={p.device}")

        glctx = self._dr.RasterizeCudaContext()

        est = self._FoundationPose(
            model_pts=np.asarray(mesh.vertices, dtype=np.float32),
            model_normals=np.asarray(mesh.vertex_normals, dtype=np.float32),
            mesh=mesh,
            scorer=scorer,
            refiner=None,
            debug_dir=str(object_debug_dir),
            debug=int(self.cfg.debug),
            glctx=glctx,
        
        )

        self._mesh_path_loaded = mesh_path
        self._mesh = mesh
        self._scorer = scorer
        self._refiner = refiner
        self._glctx = glctx
        self._est = est

  
    def track_pose(
        self,
        *,
        object_id: str,
        mesh_path: str,
        rgb: np.ndarray,
        depth: np.ndarray,
        K: np.ndarray,
        T_object_camera_init: np.ndarray,
    ) -> FoundationPoseResult:
        import time
        import torch
        t0 = time.time()
        
        # Build estimator if needed
        self._build_estimator(object_id=object_id, mesh_path=mesh_path)
        
        # Ensure contiguous arrays
        rgb = np.ascontiguousarray(rgb)
        depth = np.ascontiguousarray(depth, dtype=np.float32)
        K = np.ascontiguousarray(K, dtype=np.float32).reshape(3, 3)
        
        # Ensure inputs are the right type
        T_init = np.asarray(T_object_camera_init, dtype=np.float32).reshape(4, 4)
        
        # Generate candidate poses around the prior
        candidates = generate_pose_perturbations_compact(
            T_init,
            n_samples=16,
            max_trans_m=0.04,
            max_rot_deg=10.0,
        )

        
        # print(f"[MINI-REGISTER] Generated {len(candidates)} candidates around prior")
        
        # Convert candidates to the uncentered convention expected by FP scorer
        tf_to_center = self._est.get_tf_to_centered_mesh()
        if torch.is_tensor(tf_to_center):
            tf_to_center = tf_to_center.cpu().numpy()
        tf_to_center = np.ascontiguousarray(tf_to_center, dtype=np.float32)
        tf_to_center_inv = np.linalg.inv(tf_to_center)
        
        # Uncenter for the scorer
        candidates_uncentered = np.ascontiguousarray(np.array([
            c @ tf_to_center_inv for c in candidates
        ], dtype=np.float32))
        
        # Run scorer on all candidates
        scores, _ = self._est.scorer.predict(
            mesh=self._est.mesh,
            mesh_tensors=self._est.mesh_tensors,
            rgb=rgb,
            depth=depth,
            K=K,
            ob_in_cams=candidates_uncentered,
            normal_map=None,
            glctx=self._est.glctx,
            mesh_diameter=self._est.diameter,
            get_vis=False,
        )
        
        if torch.is_tensor(scores):
            scores = scores.cpu().numpy()
        # Find best scoring pose
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        best_pose_uncentered = np.ascontiguousarray(candidates_uncentered[best_idx], dtype=np.float32)
        
        # Convert back to centered convention
        best_pose_centered = np.ascontiguousarray(best_pose_uncentered @ tf_to_center, dtype=np.float32)
        
        # Update pose_last for potential future tracking
        self._est.pose_last = torch.from_numpy(best_pose_uncentered.copy()).cuda().float()
        
        elapsed = (time.time() - t0) * 1000
        # print(f"[MINI-REGISTER] Best score={best_score:.2f} idx={best_idx} time={elapsed:.1f}ms")
        # print(f"[MINI-REGISTER] t_in={T_init[:3, 3]} -> t_out={best_pose_centered[:3, 3]}")
        
        return FoundationPoseResult(
            object_id=object_id,
            mesh_path=str(Path(mesh_path).resolve()),
            T_object_camera=best_pose_centered,
            mask_area=0,
            debug_dir=str((self.debug_dir / object_id).resolve()),
        )
    # =============================================================================
    # ALTERNATIVE: FASTER VERSION WITH FEWER CANDIDATES
    # =============================================================================

    def track_pose_fast(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        K: np.ndarray,
        T_object_camera_init: np.ndarray,
        mask: np.ndarray | None = None,
     ) -> np.ndarray:
        """
        Faster tracking with fewer candidates - for when speed is critical.
        Uses only 8 candidates (original + 7 small perturbations).
        """
        import time
        t0 = time.time()
        
        T_init = np.asarray(T_object_camera_init, dtype=np.float32).reshape(4, 4)
        K = np.asarray(K, dtype=np.float32).reshape(3, 3)
        
        # Minimal candidate set: original + axis-aligned perturbations
        candidates = [T_init.copy()]
        
        R_prior = T_init[:3, :3]
        t_prior = T_init[:3, 3]
        
        # Add 6 axis-aligned translation perturbations (±2cm each axis)
        delta = 0.02  # 2cm
        for axis in range(3):
            for sign in [-1, 1]:
                T_new = T_init.copy()
                T_new[axis, 3] += sign * delta
                candidates.append(T_new)
        
        # Add 1 small rotation perturbation (for safety)
        dR = R.from_euler('xyz', [5, 0, 0], degrees=True).as_matrix()
        T_rot = T_init.copy()
        T_rot[:3, :3] = dR @ R_prior
        candidates.append(T_rot)
        
        candidates = np.array(candidates, dtype=np.float32)
        
        # Uncenter for scorer
        tf_to_center = self._est.get_tf_to_centered_mesh().cpu().numpy()
        tf_to_center_inv = np.linalg.inv(tf_to_center)
        candidates_uncentered = np.array([c @ tf_to_center_inv for c in candidates], dtype=np.float32)
        
        from estimater import depth2xyzmap
        xyz_map = depth2xyzmap(depth, K)
        
        scores, _ = self._est.scorer.predict(
            mesh=self._est.mesh,
            mesh_tensors=self._est.mesh_tensors,
            rgb=rgb,
            depth=depth,
            K=K,
            ob_in_cams=candidates_uncentered,
            normal_map=None,
            glctx=self._est.glctx,
            mesh_diameter=self._est.diameter,
            get_vis=False,
        )
        
        if torch.is_tensor(scores):
            scores = scores.cpu().numpy()
        best_idx = np.argmax(scores)
        best_pose_uncentered = candidates_uncentered[best_idx]
        best_pose_centered = best_pose_uncentered @ tf_to_center
        
        self._est.pose_last = torch.from_numpy(best_pose_uncentered).cuda().float()
        
        elapsed = (time.time() - t0) * 1000
        # print(f"[MINI-REGISTER FAST] {len(candidates)} candidates, best={best_idx}, time={elapsed:.1f}ms")
        
        return best_pose_centered.astype(np.float32)

    # def track_pose(
    #     self,
    #     *,
    #     object_id: str,
    #     mesh_path: str,
    #     rgb: np.ndarray,
    #     depth: np.ndarray,
    #     K: np.ndarray,
    #     T_object_camera_init: np.ndarray,
    # ):
    #     try:
    #         self._build_estimator(object_id=object_id, mesh_path=mesh_path)
    #     except Exception as e:
    #         raise RuntimeError(
    #             f"FoundationPose _build_estimator() failed for {object_id}: {e}"
    #         ) from None

    #     rgb = self._sanitize_rgb(rgb)
    #     depth = self._sanitize_depth(depth).astype(np.float32)
    #     K = self._sanitize_K(K).astype(np.float32)

    #     try:
    #         import torch
    #         T_init = np.asarray(T_object_camera_init, dtype=np.float32).reshape(4, 4)
    #         tf_to_center = self._est.get_tf_to_centered_mesh().cpu().numpy()
    #         pose_uncentered = T_init @ np.linalg.inv(tf_to_center)
    #         self._est.pose_last = torch.from_numpy(pose_uncentered).cuda().float()

    #         print(f"[TRACK DEBUG] T_init t={T_init[:3,3]}")
    #         print(f"[TRACK DEBUG] tf_to_center t={tf_to_center[:3,3]}")
    #         print(f"[TRACK DEBUG] pose_uncentered t={pose_uncentered[:3,3]}")
    #         print(f"[TRACK DEBUG] pose_last before={self._est.pose_last}")
    #         print(f"[REFINER DEBUG] self._est.diameter={self._est.diameter:.6f}")
    #         print(f"[REFINER DEBUG] mesh vertices min={self._est.mesh.vertices.min():.6f} max={self._est.mesh.vertices.max():.6f}")
    #         pose_raw = self._est.track_one(
    #             rgb=rgb,
    #             depth=depth,
    #             K=K,
    #             iteration= 2,
    #         )

    #         if isinstance(pose_raw, torch.Tensor):
    #             pose = pose_raw.detach().cpu().numpy()
    #         else:
    #             pose = np.asarray(pose_raw, dtype=np.float32)

    #     except Exception as e:
    #         raise RuntimeError(
    #             f"FoundationPose track_one() failed for {object_id}: {e}"
    #         ) from None

    #     pose = np.asarray(pose, dtype=np.float32).reshape(4, 4)

    #     return FoundationPoseResult(
    #         object_id=object_id,
    #         mesh_path=str(Path(mesh_path).resolve()),
    #         T_object_camera=pose,
    #         mask_area=0,
    #         debug_dir=str((self.debug_dir / object_id).resolve()),
    #     )

    def estimate_pose(
        self,
        *,
        object_id: str,
        mesh_path: str,
        rgb: np.ndarray,
        depth: np.ndarray,
        K: np.ndarray,
        mask: np.ndarray,
        est_refine_iter: int | None = None,
    ) -> FoundationPoseResult:
        try:
            self._build_estimator(object_id=object_id, mesh_path=mesh_path)
        except Exception as e:
            raise RuntimeError(
                f"FoundationPose _build_estimator() failed for {object_id} "
                f"(mesh={mesh_path}): {e}"
            ) from None

        rgb = self._sanitize_rgb(rgb)
        depth = self._sanitize_depth(depth).astype(np.float32)
        K = self._sanitize_K(K).astype(np.float32)
        mask = self._sanitize_mask(mask, rgb.shape[:2])
        mask_bool = mask.astype(bool)
        d = depth[mask_bool]
        valid = d[np.isfinite(d) & (d > 0)]
        
        try:
            import torch

            if hasattr(self._scorer, "model") and self._scorer.model is not None:
                self._scorer.model = self._scorer.model.float().cuda().eval()
            if hasattr(self._refiner, "model") and self._refiner.model is not None:
                self._refiner.model = self._refiner.model.float().cuda().eval()
            # RIGHT BEFORE self._est.register():
            # print(f"[DEBUG depth check] has_nan={np.isnan(depth).any()} has_inf={np.isinf(depth).any()} min={np.nanmin(depth):.3f} max={np.nanmax(depth):.3f}")
            pose = self._est.register(
                K=K,
                rgb=rgb,
                depth=depth,
                ob_mask=mask,
                iteration=0,
            )
            # print("FP raw translation:", pose[:3, 3])
        except Exception as e:
            raise RuntimeError(
                f"FoundationPose register() failed for {object_id} "
                f"(mask_area={int(mask.sum())}): {e}"
            ) from None

        if pose is None:
            raise RuntimeError(
                f"FoundationPose register() returned None for {object_id} "
                f"(mask_area={int(mask.sum())})"
            )
        pose = np.asarray(pose, dtype=np.float32).reshape(4, 4)

        t_raw = pose[:3, 3].copy()


        return FoundationPoseResult(
            object_id=object_id,
            mesh_path=str(Path(mesh_path).resolve()),
            T_object_camera=pose,
            mask_area=int(mask.sum()),
            debug_dir=str((self.debug_dir / object_id).resolve()),
        )
    #     # pose = np.asarray(pose, dtype=np.float32).reshape(4, 4)
    #     # pose_inv = np.linalg.inv(pose)

    #     # t_raw = pose[:3, 3].copy()
    #     # t_inv = pose_inv[:3, 3].copy()
    #     # z_raw = float(t_raw[2])
    #     # z_inv = float(t_inv[2])

    #     # # Pick physically plausible candidate.
    #     # # Your cube should be roughly 0.30-0.50 m away, so prefer:
    #     # #   1) positive z
    #     # #   2) z in a sane range [0.10, 1.50]
    #     # #   3) smaller |x|, |y| if both are sane
    #     # raw_ok = 0.10 <= z_raw <= 1.50
    #     # inv_ok = 0.10 <= z_inv <= 1.50

    #     # if raw_ok and not inv_ok:
    #     #     pose_out = pose
    #     #     chosen = "raw"
    #     # elif inv_ok and not raw_ok:
    #     #     pose_out = pose_inv
    #     #     chosen = "inv"
    #     # elif raw_ok and inv_ok:
    #     #     raw_xy = float(np.linalg.norm(t_raw[:2]))
    #     #     inv_xy = float(np.linalg.norm(t_inv[:2]))
    #     #     if raw_xy <= inv_xy:
    #     #         pose_out = pose
    #     #         chosen = "raw"
    #     #     else:
    #     #         pose_out = pose_inv
    #     #         chosen = "inv"
    #     # else:
    #     #     # fallback: choose positive z if available, otherwise raw
    #     #     if z_raw > 0.0 and z_inv <= 0.0:
    #     #         pose_out = pose
    #     #         chosen = "raw_fallback"
    #     #     elif z_inv > 0.0 and z_raw <= 0.0:
    #     #         pose_out = pose_inv
    #     #         chosen = "inv_fallback"
    #     #     else:
    #     #         pose_out = pose
    #     #         chosen = "raw_fallback"

    #     # t_out = pose_out[:3, 3]
    #     # print(
    #     #     f"[estimate_pose()] {object_id} "
    #     #     f"t_raw=[{t_raw[0]:.3f}, {t_raw[1]:.3f}, {t_raw[2]:.3f}] "
    #     #     f"t_inv=[{t_inv[0]:.3f}, {t_inv[1]:.3f}, {t_inv[2]:.3f}] "
    #     #     f"chosen={chosen} "
    #     #     f"t_out=[{t_out[0]:.3f}, {t_out[1]:.3f}, {t_out[2]:.3f}]"
    #     # )

    #     # return FoundationPoseResult(
    #     #     object_id=object_id,
    #     #     mesh_path=str(Path(mesh_path).resolve()),
    #     #     T_object_camera=pose_out,
    #     #     mask_area=int(mask.sum()),
    #     #     debug_dir=str((self.debug_dir / object_id).resolve()),
    #     # )