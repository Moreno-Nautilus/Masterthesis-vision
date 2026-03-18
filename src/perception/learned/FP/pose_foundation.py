from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import sys

import numpy as np


@dataclass
class FoundationPoseConfig:
    repo_root: str = "external/FoundationPose"
    weights_dir: str = "external/FoundationPose/weights"
    debug_dir: str = "outputs/foundationpose/fp_internal_debug"
    debug: int = 2
    est_refine_iter: int = 5
    mesh_scale: float = 0.01  # STL is in mm, convert to meters



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
        print(f"[FoundationPoseWrapper] mesh_scale={self.cfg.mesh_scale}")

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
        print("DEBUG mesh extents BEFORE:", mesh.extents)
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
        if hasattr(refiner, "model") and refiner.model is not None:
            refiner.model = refiner.model.float().cuda().eval()

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
    ):
        try:
            self._build_estimator(object_id=object_id, mesh_path=mesh_path)
        except Exception as e:
            raise RuntimeError(
                f"FoundationPose _build_estimator() failed for {object_id}: {e}"
            ) from None

        rgb = self._sanitize_rgb(rgb)
        depth = self._sanitize_depth(depth).astype(np.float32)
        K = self._sanitize_K(K).astype(np.float32)

        try:
            import torch

            # IMPORTANT:
            # Do NOT overwrite self._est.pose_last here.
            # After estimate_pose/register(), FoundationPose already keeps internal
            # tracker state in its own convention. Re-seeding it from T_init appears
            # to be what causes the immediate same-frame jump.

            pose_raw = self._est.track_one(
                rgb=rgb,
                depth=depth,
                K=K,
                iteration=2,
            )

            if isinstance(pose_raw, torch.Tensor):
                pose = pose_raw.detach().cpu().numpy()
            else:
                pose = np.asarray(pose_raw, dtype=np.float32)

        except Exception as e:
            raise RuntimeError(
                f"FoundationPose track_one() failed for {object_id}: {e}"
            ) from None

        pose = np.asarray(pose, dtype=np.float32).reshape(4, 4)

        return FoundationPoseResult(
            object_id=object_id,
            mesh_path=str(Path(mesh_path).resolve()),
            T_object_camera=pose,
            mask_area=0,
            debug_dir=str((self.debug_dir / object_id).resolve()),
        )

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
        if valid.size > 0:
            print(
                f"[estimate_pose depth] {object_id} "
                f"mask_area={int(mask.sum())} "
                f"depth_min={float(valid.min()):.3f} "
                f"depth_med={float(np.median(valid)):.3f} "
                f"depth_max={float(valid.max()):.3f}"
            )
        else:
            print(f"[estimate_pose depth] {object_id} no valid depth in mask")

        try:
            import torch

            if hasattr(self._scorer, "model") and self._scorer.model is not None:
                self._scorer.model = self._scorer.model.float().cuda().eval()
            if hasattr(self._refiner, "model") and self._refiner.model is not None:
                self._refiner.model = self._refiner.model.float().cuda().eval()
            # RIGHT BEFORE self._est.register():
            print(f"[DEBUG depth check] has_nan={np.isnan(depth).any()} has_inf={np.isinf(depth).any()} min={np.nanmin(depth):.3f} max={np.nanmax(depth):.3f}")
            pose = self._est.register(
                K=K,
                rgb=rgb,
                depth=depth,
                ob_mask=mask,
                iteration=0,
            )
            print("FP raw translation:", pose[:3, 3])
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
        print(
            f"[estimate_pose RAW] {object_id} "
            f"t_raw=[{t_raw[0]:.3f}, {t_raw[1]:.3f}, {t_raw[2]:.3f}]"
        )

        return FoundationPoseResult(
            object_id=object_id,
            mesh_path=str(Path(mesh_path).resolve()),
            T_object_camera=pose,
            mask_area=int(mask.sum()),
            debug_dir=str((self.debug_dir / object_id).resolve()),
        )
        # pose = np.asarray(pose, dtype=np.float32).reshape(4, 4)
        # pose_inv = np.linalg.inv(pose)

        # t_raw = pose[:3, 3].copy()
        # t_inv = pose_inv[:3, 3].copy()
        # z_raw = float(t_raw[2])
        # z_inv = float(t_inv[2])

        # # Pick physically plausible candidate.
        # # Your cube should be roughly 0.30-0.50 m away, so prefer:
        # #   1) positive z
        # #   2) z in a sane range [0.10, 1.50]
        # #   3) smaller |x|, |y| if both are sane
        # raw_ok = 0.10 <= z_raw <= 1.50
        # inv_ok = 0.10 <= z_inv <= 1.50

        # if raw_ok and not inv_ok:
        #     pose_out = pose
        #     chosen = "raw"
        # elif inv_ok and not raw_ok:
        #     pose_out = pose_inv
        #     chosen = "inv"
        # elif raw_ok and inv_ok:
        #     raw_xy = float(np.linalg.norm(t_raw[:2]))
        #     inv_xy = float(np.linalg.norm(t_inv[:2]))
        #     if raw_xy <= inv_xy:
        #         pose_out = pose
        #         chosen = "raw"
        #     else:
        #         pose_out = pose_inv
        #         chosen = "inv"
        # else:
        #     # fallback: choose positive z if available, otherwise raw
        #     if z_raw > 0.0 and z_inv <= 0.0:
        #         pose_out = pose
        #         chosen = "raw_fallback"
        #     elif z_inv > 0.0 and z_raw <= 0.0:
        #         pose_out = pose_inv
        #         chosen = "inv_fallback"
        #     else:
        #         pose_out = pose
        #         chosen = "raw_fallback"

        # t_out = pose_out[:3, 3]
        # print(
        #     f"[estimate_pose()] {object_id} "
        #     f"t_raw=[{t_raw[0]:.3f}, {t_raw[1]:.3f}, {t_raw[2]:.3f}] "
        #     f"t_inv=[{t_inv[0]:.3f}, {t_inv[1]:.3f}, {t_inv[2]:.3f}] "
        #     f"chosen={chosen} "
        #     f"t_out=[{t_out[0]:.3f}, {t_out[1]:.3f}, {t_out[2]:.3f}]"
        # )

        # return FoundationPoseResult(
        #     object_id=object_id,
        #     mesh_path=str(Path(mesh_path).resolve()),
        #     T_object_camera=pose_out,
        #     mask_area=int(mask.sum()),
        #     debug_dir=str((self.debug_dir / object_id).resolve()),
        # )