from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


MUSE_EMBEDDING_MODE = "two_stream"
MUSE_SIMILARITY = "tanimoto"
MUSE_RELATIVE_SCORE_MODE = "softmax_tau"
MUSE_STREAM_ALPHA = 0.5
MUSE_JOINT_SCORE_ALPHA = 0.8
MUSE_TAU = 0.02
MUSE_OBJECTNESS_PRIOR_GAMMA = 0.1


@dataclass
class DINOIdentifierConfig:
    model_name: str = "dinov2_vitb14"
    device: str = "cuda"
    input_size: int = 518
    use_masked_background: bool = True
    background_value: int = 0
    normalize_embeddings: bool = True
    reference_dir: str = "Data/reference_crops"
    allowed_exts: tuple[str, ...] = (".png", ".jpg", ".jpeg")

    # MUSE two-stream patch pooling exponent.
    gem_p: float = 1.5
    # When False, suppresses all status/info prints from this module.
    verbose: bool = False


@dataclass
class ReferenceEmbedding:
    object_id: str
    image_path: str
    embedding: np.ndarray


@dataclass
class DINOResult:
    object_id: str
    score: float
    embedding: np.ndarray
    scores_by_object: dict[str, float] = field(default_factory=dict)


class DINOIdentifier:
    def __init__(self, cfg: DINOIdentifierConfig | None = None) -> None:
        self.cfg = cfg or DINOIdentifierConfig()
        self.device = torch.device(
            self.cfg.device if self.cfg.device == "cuda" and torch.cuda.is_available() else "cpu"
        )

        self.model = self._build_model()

        self.reference_bank: list[ReferenceEmbedding] = []
        self.reference_matrix: np.ndarray | None = None
        self.reference_object_ids: list[str] = []

    def _build_model(self) -> torch.nn.Module:
        model = torch.hub.load("facebookresearch/dinov2", self.cfg.model_name)
        model = model.to(self.device)
        model.eval()
        return model

    @staticmethod
    def _ensure_rgb(img: np.ndarray) -> np.ndarray:
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"Expected (H, W, 3) RGB image, got {img.shape}")
        return np.ascontiguousarray(img.astype(np.uint8))

    def _apply_mask(self, rgb: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
        if mask is None or not self.cfg.use_masked_background:
            return rgb

        if mask.shape != rgb.shape[:2]:
            raise ValueError(f"Mask shape {mask.shape} does not match image shape {rgb.shape[:2]}")

        out = rgb.copy()
        out[~mask.astype(bool)] = self.cfg.background_value
        return out

    def _center_crop_square(self, rgb: np.ndarray) -> np.ndarray:
        h, w = rgb.shape[:2]
        s = min(h, w)
        y0 = (h - s) // 2
        x0 = (w - s) // 2
        return rgb[y0:y0 + s, x0:x0 + s]

    def _preprocess(self, rgb: np.ndarray) -> torch.Tensor:
        rgb = self._center_crop_square(rgb)
        rgb = cv2.resize(rgb, (self.cfg.input_size, self.cfg.input_size), interpolation=cv2.INTER_AREA)

        x = rgb.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        x = (x - mean) / std
        x = np.transpose(x, (2, 0, 1))  # HWC -> CHW
        x = torch.from_numpy(x).unsqueeze(0).to(self.device)
        return x

    @staticmethod
    def _gem_pool(patch_tokens: torch.Tensor, p: float = 1.5, eps: float = 1e-6) -> torch.Tensor:
        """Generalised Mean pooling over patch tokens"""
        x = patch_tokens.clamp(min=eps)
        return x.pow(p).mean(dim=1).pow(1.0 / p)

    def embed_image(self, rgb: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        """Compute the MUSE two-stream embedding for one RGB crop."""
        rgb = self._ensure_rgb(rgb)
        rgb = self._apply_mask(rgb, mask)

        x = self._preprocess(rgb)

        with torch.inference_mode():
            feats = self.model.forward_features(x)
        if not isinstance(feats, dict):
            raise RuntimeError(
                f"forward_features returned {type(feats)}; expected dict for patch tokens"
            )

        cls_tok = feats["x_norm_clstoken"].reshape(1, -1)
        patch_toks = feats["x_norm_patchtokens"]  # (1, N, D)
        gem = self._gem_pool(patch_toks, p=float(self.cfg.gem_p)).reshape(1, -1)

        cls_n = F.normalize(cls_tok, dim=1).squeeze(0)
        gem_n = F.normalize(gem, dim=1).squeeze(0)
        out = torch.stack([cls_n, gem_n], dim=0)  # (2, D)
        return out.detach().cpu().numpy()

    def build_reference_bank_from_folder(
        self,
        cache_path: str | None = None,
        extra_dirs: list[str] | None = None,
    ) -> None:
        roots = [Path(self.cfg.reference_dir)]
        if extra_dirs:
            roots.extend(Path(d) for d in extra_dirs if d)

        # Drop dirs that don't exist; raise only if none remain.
        existing_roots = [r for r in roots if r.exists()]
        if not existing_roots:
            raise FileNotFoundError(
                f"None of the reference directories exist: {[str(r) for r in roots]}"
            )
        roots = existing_roots

        primary_root = roots[0]

        # Default cache path next to primary reference dir. 
        if cache_path is None:
            roots_tag = "_".join(sorted(r.name for r in roots))
            # gem_p affects patch-pool output; size affects raw token grid.
            extra_tag = (
                f"gem{float(self.cfg.gem_p):g}_in{int(self.cfg.input_size)}"
            )
            tag = (
                f"{self.cfg.model_name}__{MUSE_EMBEDDING_MODE}"
                f"__{roots_tag}__{extra_tag}"
            )
            cache_path = str(primary_root / f"_embedding_cache__{tag}.npz")

        # Try to load from cache
        if Path(cache_path).exists():
            try:
                cached = np.load(cache_path, allow_pickle=True)
                cached_paths = list(cached["image_paths"])
                cached_object_ids = list(cached["object_ids"])
                cached_embeddings = cached["embeddings"]
                expected_dim = int(getattr(self.model, "embed_dim", cached_embeddings.shape[-1]))
                cache_shape_ok = (
                    cached_embeddings.ndim == 3
                    and cached_embeddings.shape[1] == 2
                    and cached_embeddings.shape[2] == expected_dim
                )

                # Verify cache matches current folder structure (across all roots).
                current_paths = []
                for r in roots:
                    for object_dir in sorted(p for p in r.iterdir() if p.is_dir()):
                        for img_path in sorted(object_dir.iterdir()):
                            if img_path.suffix.lower() in self.cfg.allowed_exts:
                                current_paths.append(str(img_path))

                if cached_paths == current_paths and cache_shape_ok:
                    # Cache is valid, use it
                    bank = []
                    for i, (obj_id, img_path, emb) in enumerate(
                        zip(cached_object_ids, cached_paths, cached_embeddings)
                    ):
                        bank.append(
                            ReferenceEmbedding(
                                object_id=obj_id,
                                image_path=img_path,
                                embedding=emb,
                            )
                        )
                    self.reference_bank = bank
                    self.reference_matrix = cached_embeddings
                    self.reference_object_ids = cached_object_ids
                    if self.cfg.verbose:
                        print(f"[DINO] Loaded {len(bank)} embeddings from cache")
                    return
                if self.cfg.verbose and cached_paths == current_paths and not cache_shape_ok:
                    print(
                        f"[DINO] Cache shape mismatch, rebuilding: "
                        f"{cached_embeddings.shape} expected (*, 2, {expected_dim})"
                    )
            except Exception as e:
                if self.cfg.verbose:
                    print(f"[DINO] Cache load failed, rebuilding: {e}")

        # Build from scratch (iterate across all reference roots).
        bank: list[ReferenceEmbedding] = []

        for r in roots:
          for object_dir in sorted(p for p in r.iterdir() if p.is_dir()):
            object_id = object_dir.name
            for img_path in sorted(object_dir.iterdir()):
                if img_path.suffix.lower() not in self.cfg.allowed_exts:
                    continue

                bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                if bgr is None:
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

                emb = self.embed_image(rgb)
                bank.append(
                    ReferenceEmbedding(
                        object_id=object_id,
                        image_path=str(img_path),
                        embedding=emb,
                    )
                )

        if not bank:
            raise RuntimeError(
                f"No valid reference images found under: {[str(r) for r in roots]}"
            )

        self.reference_bank = bank
        self.reference_matrix = np.stack([r.embedding for r in bank], axis=0)
        self.reference_object_ids = [r.object_id for r in bank]

        # Save cache
        try:
            np.savez(
                cache_path,
                image_paths=np.array([r.image_path for r in bank], dtype=object),
                object_ids=np.array([r.object_id for r in bank], dtype=object),
                embeddings=self.reference_matrix,
            )
            if self.cfg.verbose:
                print(f"[DINO] Saved {len(bank)} embeddings to cache: {cache_path}")
        except Exception as e:
            if self.cfg.verbose:
                print(f"[DINO] Failed to save cache: {e}")




    @staticmethod
    def _pairwise_tanimoto(query: np.ndarray, bank: np.ndarray) -> np.ndarray:
        """Compute (N,) Tanimoto similarity row between query (1, D) and bank."""
        qr = (query @ bank.T).reshape(-1)
        qq = float(np.sum(query * query))
        rr = np.sum(bank * bank, axis=1)
        denom = qq + rr - qr
        return qr / np.maximum(denom, 1e-12)

    def _aggregate_per_class(
        self,
        sims: np.ndarray,
        top_k: int = 3,
    ) -> dict[str, float]:
        scores_by_object: dict[str, list[float]] = {}
        for sim, obj_id in zip(sims, self.reference_object_ids):
            scores_by_object.setdefault(obj_id, []).append(float(sim))
        agg: dict[str, float] = {}
        for obj_id, vals in scores_by_object.items():
            vals_sorted = sorted(vals, reverse=True)
            k = min(top_k, len(vals_sorted))
            agg[obj_id] = float(np.mean(vals_sorted[:k]))
        return agg

    def classify_embedding(
        self,
        embedding: np.ndarray,
        objectness_prior: float | dict[str, float] | None = None,
    ) -> DINOResult:
        """Classify a MUSE two-stream query embedding against the reference bank."""
        if self.reference_matrix is None or len(self.reference_bank) == 0:
            raise RuntimeError("Reference bank is empty. Build or set it first.")

        embedding = np.asarray(embedding, dtype=np.float32)
        if (
            embedding.ndim != 2
            or embedding.shape[0] != 2
            or self.reference_matrix.ndim != 3
            or self.reference_matrix.shape[1] != 2
        ):
            raise ValueError(
                "MUSE DINO classification expects query shape (2, D) and "
                "reference bank shape (N, 2, D)."
            )

        # Compute S_cls and S_patch independently and blend (MUSE Eq.4).
        q_cls = embedding[0:1, :]
        q_patch = embedding[1:2, :]
        bank_cls = self.reference_matrix[:, 0, :]
        bank_patch = self.reference_matrix[:, 1, :]
        if self.cfg.normalize_embeddings:
            q_cls = q_cls / (np.linalg.norm(q_cls, axis=1, keepdims=True) + 1e-12)
            q_patch = q_patch / (np.linalg.norm(q_patch, axis=1, keepdims=True) + 1e-12)
        sims_cls = self._pairwise_tanimoto(q_cls, bank_cls)
        sims_patch = self._pairwise_tanimoto(q_patch, bank_patch)
        sims = (
            MUSE_STREAM_ALPHA * sims_cls
            + (1.0 - MUSE_STREAM_ALPHA) * sims_patch
        )
        # Keep a representative embedding for downstream consumers
        # (memory bank, debug viz). Use the CLS row.
        embedding_out = embedding[0]

        agg = self._aggregate_per_class(sims, top_k=3)

        # ── MUSE Eq.5/Eq.6: joint absolute + relative score ────────────────
        beta = MUSE_JOINT_SCORE_ALPHA  # paper "beta"
        if beta > 0.0 and len(agg) >= 2:
            tau = max(1e-6, MUSE_TAU)
            names = list(agg.keys())
            logits = np.array([agg[k] for k in names], dtype=np.float64) / tau
            logits -= logits.max()  # numerical stability
            probs = np.exp(logits)
            probs /= probs.sum() + 1e-12
            relative = {k: float(p) for k, p in zip(names, probs)}
            joint = {k: beta * agg[k] + (1.0 - beta) * relative[k] for k in agg}
            agg_for_decision = joint
        else:
            agg_for_decision = dict(agg)

        # ── MUSE Eq.8/Eq.10: per-proposal objectness prior ─────────────────
        if isinstance(objectness_prior, (int, float)):
            gamma = MUSE_OBJECTNESS_PRIOR_GAMMA
            if gamma > 0.0:
                p_scaled = max(0.0, float(objectness_prior)) ** gamma
                agg_for_decision = {
                    k: v * p_scaled for k, v in agg_for_decision.items()
                }
        elif isinstance(objectness_prior, dict) and objectness_prior:
            # Legacy per-class multiplicative weighting.
            agg_for_decision = {
                k: v * float(objectness_prior.get(k, 1.0))
                for k, v in agg_for_decision.items()
            }

        best_obj = max(agg_for_decision, key=agg_for_decision.get)
        best_score = agg_for_decision[best_obj]

        return DINOResult(
            object_id=best_obj,
            score=float(best_score),
            embedding=embedding_out,
            scores_by_object=agg_for_decision,
        )
