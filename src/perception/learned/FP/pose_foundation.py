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
        depth = np.asarray(depth)
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

        # trimesh usually computes these lazily; force availability
        _ = mesh.vertex_normals

        object_debug_dir = self.debug_dir / object_id
        object_debug_dir.mkdir(parents=True, exist_ok=True)

        scorer = self._ScorePredictor()
        refiner = self._PoseRefinePredictor()
        glctx = self._dr.RasterizeCudaContext()

        est = self._FoundationPose(
            model_pts=np.asarray(mesh.vertices),
            model_normals=np.asarray(mesh.vertex_normals),
            mesh=mesh,
            scorer=scorer,
            refiner=refiner,
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
        self._build_estimator(object_id=object_id, mesh_path=mesh_path)

        rgb = self._sanitize_rgb(rgb)
        depth = self._sanitize_depth(depth)
        K = self._sanitize_K(K)
        mask = self._sanitize_mask(mask, rgb.shape[:2])

        pose = self._est.register(
            K=K,
            rgb=rgb,
            depth=depth,
            ob_mask=mask,
            iteration=int(est_refine_iter if est_refine_iter is not None else self.cfg.est_refine_iter),
        )

        pose = np.asarray(pose, dtype=np.float32).reshape(4, 4)

        return FoundationPoseResult(
            object_id=object_id,
            mesh_path=str(Path(mesh_path).resolve()),
            T_object_camera=pose,
            mask_area=int(mask.sum()),
            debug_dir=str((self.debug_dir / object_id).resolve()),
        )