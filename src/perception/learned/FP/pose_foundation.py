from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import os
import sys

import numpy as np


@dataclass
class FoundationPoseConfig:
    repo_root: str = "external/FoundationPose"
    weights_dir: str = "external/FoundationPose/weights"
    debug_dir: str = "outputs/foundationpose/fp_internal_debug"
    debug: int = 0
    mesh_scale: float = 0.01  # STL is in mm, convert to meters


@dataclass
class FoundationPoseResult:
    object_id: str
    mesh_path: str
    T_object_camera: np.ndarray   # 4x4 homogeneous transform (6D pose representation)
    mask_area: int
    debug_dir: str


class FoundationPoseWrapper:
    """Lazy wrapper for FoundationPose registration."""

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

        self._mesh_path_loaded: str | None = None
        self._est = None
        self._mesh = None
        self._glctx = None
        self._scorer = None

        # All nvdiffrast / warp / CUDA work for this wrapper MUST run on a single
        # dedicated thread. nvdiffrast's RasterizeCudaContext and warp kernels are
        # bound to the thread that created them; touching one from a different
        # thread corrupts the CUDA context (error 700, illegal memory access at
        # cudaStreamSynchronize). The ROS MultiThreadedExecutor runs _tick (and
        # therefore register()) on whichever of its worker threads is free, which
        # migrates between ticks. Marshalling _build_estimator() and register()
        # through this 1-worker pool guarantees the context is created AND used on
        # the same thread for the lifetime of the process.
        self._gpu_exec = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="fp-gpu"
        )

    def _ensure_repo_on_path(self) -> None:
        if not self.repo_root.exists():
            raise FileNotFoundError(f"FoundationPose repo_root does not exist: {self.repo_root}")

        repo_root_str = str(self.repo_root)
        if repo_root_str not in sys.path:
            sys.path.insert(0, repo_root_str)

    def _prepare_env(self) -> None:
        if not self.weights_dir.exists():
            raise FileNotFoundError(f"FoundationPose weights_dir does not exist: {self.weights_dir}")

        os.environ.setdefault("FOUNDATIONPOSE_WEIGHTS_DIR", str(self.weights_dir))
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", os.environ.get("TORCH_CUDA_ARCH_LIST", ""))

    def _lazy_imports(self) -> None:
        if self._imports_ready:
            return

        self._ensure_repo_on_path()
        self._prepare_env()

        import trimesh
        import nvdiffrast.torch as dr
        from estimater import FoundationPose, ScorePredictor

        self._trimesh = trimesh
        self._dr = dr
        self._FoundationPose = FoundationPose
        self._ScorePredictor = ScorePredictor

        self._imports_ready = True

    def preload_mesh(self, mesh_path: str, object_id: str | None = None) -> None:
        """Build the estimator once so first init is faster."""
        if object_id is None:
            object_id = Path(mesh_path).stem
        
        mesh_file = Path(mesh_path)
        if not mesh_file.exists():
            print(f"[FoundationPose] Failed to pre-cache {object_id}: Mesh file does not exist: {mesh_path}")
            return
        
        try:
            # Build on the dedicated GPU thread so the nvdiffrast context is
            # created there (see __init__).
            self._gpu_exec.submit(
                self._build_estimator, object_id=object_id, mesh_path=mesh_path
            ).result()
            print(f"[FoundationPose] Pre-cached mesh for {object_id}")
        except Exception as e:
            print(f"[FoundationPose] Failed to pre-cache {object_id}: {e}")

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
        mesh.vertices -= mesh.centroid

        if self.cfg.mesh_scale != 1.0:
            mesh.apply_scale(self.cfg.mesh_scale)

        _ = mesh.vertex_normals

        object_debug_dir = self.debug_dir / object_id
        object_debug_dir.mkdir(parents=True, exist_ok=True)

        model_pts = np.asarray(mesh.vertices, dtype=np.float32)
        model_normals = np.asarray(mesh.vertex_normals, dtype=np.float32)

        if self._est is None:
            # Build the estimator ONCE. The ScorePredictor / PoseRefinePredictor
            # networks and the nvdiffrast context are mesh-agnostic; only the mesh
            # data changes per object. Rebuilding a whole FoundationPose (reloading
            # both networks + a new RasterizeCudaContext) for every object churned
            # and fragmented GPU memory, intermittently triggering CUDA error 700
            # (illegal memory access) once all three cameras pushed ~25 objects
            # through every init cycle.
            scorer = self._ScorePredictor()
            if hasattr(scorer, "model") and scorer.model is not None:
                scorer.model = scorer.model.float().cuda().eval()

            self._glctx = self._dr.RasterizeCudaContext()

            self._est = self._FoundationPose(
                model_pts=model_pts,
                model_normals=model_normals,
                mesh=mesh,
                scorer=scorer,
                refiner=None,
                debug_dir=str(object_debug_dir),
                debug=int(self.cfg.debug),
                glctx=self._glctx,
            )
            self._scorer = scorer
        else:
            # Swap only the mesh; reuse the networks, context and rotation grid.
            # We always use identity symmetry here, so the rotation grid built at
            # construction stays valid for every object.
            self._est.reset_object(
                model_pts=model_pts,
                model_normals=model_normals,
                mesh=mesh,
            )
            self._est.debug_dir = str(object_debug_dir)

        self._mesh_path_loaded = mesh_path
        self._mesh = mesh

    def estimate_pose(
        self,
        *,
        object_id: str,
        mesh_path: str,
        rgb: np.ndarray,
        depth: np.ndarray,
        K: np.ndarray,
        mask: np.ndarray,
    ) -> FoundationPoseResult:
        # CPU-side sanitization is safe on the caller thread; the GPU work
        # (estimator build + nvdiffrast/warp register) is marshalled onto the
        # dedicated thread so the CUDA context is never touched cross-thread.
        rgb = self._sanitize_rgb(rgb)
        depth = self._sanitize_depth(depth).astype(np.float32)
        K = self._sanitize_K(K).astype(np.float32)
        mask = self._sanitize_mask(mask, rgb.shape[:2])

        return self._gpu_exec.submit(
            self._register_on_gpu_thread,
            object_id=object_id,
            mesh_path=mesh_path,
            rgb=rgb,
            depth=depth,
            K=K,
            mask=mask,
        ).result()

    def _register_on_gpu_thread(
        self,
        *,
        object_id: str,
        mesh_path: str,
        rgb: np.ndarray,
        depth: np.ndarray,
        K: np.ndarray,
        mask: np.ndarray,
    ) -> FoundationPoseResult:
        """Runs ONLY on self._gpu_exec's single worker thread (see __init__)."""
        try:
            self._build_estimator(object_id=object_id, mesh_path=mesh_path)
        except Exception as e:
            raise RuntimeError(
                f"FoundationPose _build_estimator() failed for {object_id} "
                f"(mesh={mesh_path}): {e}"
            ) from None

        try:
            if hasattr(self._scorer, "model") and self._scorer.model is not None:
                self._scorer.model = self._scorer.model.float().cuda().eval()
            pose = self._est.register(
                K=K,
                rgb=rgb,
                depth=depth,
                ob_mask=mask,
                iteration=0,
            )
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

        return FoundationPoseResult(
            object_id=object_id,
            mesh_path=str(Path(mesh_path).resolve()),
            T_object_camera=pose,
            mask_area=int(mask.sum()),
            debug_dir=str((self.debug_dir / object_id).resolve()),
        )
