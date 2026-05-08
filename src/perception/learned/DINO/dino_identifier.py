from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import os
import time

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

    # Bundle 5 / MUSE-style options. All defaults preserve the original
    # CLS-token + cosine behaviour.
    #
    # embedding_mode:
    #   "cls"        — original DINOv2 CLS token (unchanged behaviour).
    #   "patch_gem"  — Generalised-Mean-pooled patch tokens (more spatial
    #                  detail; better at distinguishing similar objects).
    #   "concat"     — concatenate L2-normalised CLS + GeM patch features.
    #                  Closest to the MUSE descriptor.
    embedding_mode: str = "cls"
    gem_p: float = 3.0  # GeM pooling exponent. Higher = closer to max-pool.
    # similarity:
    #   "cosine"   — original.
    #   "tanimoto" — generalised Tanimoto / Jaccard for normalised vectors.
    #                Penalises descriptors that share magnitude only.
    similarity: str = "cosine"
    # Joint absolute + relative score: blend raw cosine with how much the
    # top-1 dominates the rest of the bank. 0 = pure raw score (default).
    joint_score_alpha: float = 0.0


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
        # self.model.eval()

        self.reference_bank: list[ReferenceEmbedding] = []
        self.reference_matrix: np.ndarray | None = None
        self.reference_object_ids: list[str] = []
        self.debug_dir = "/home/moreno/MasterThesis/outputs/DINODEBUG"
        self.debug_enabled = False  # Set False to disable
        if self.debug_enabled:
            os.makedirs(self.debug_dir, exist_ok=True)
            os.makedirs(f"{self.debug_dir}/query_crops", exist_ok=True)
            os.makedirs(f"{self.debug_dir}/matches", exist_ok=True)

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
    def _gem_pool(patch_tokens: torch.Tensor, p: float = 3.0, eps: float = 1e-6) -> torch.Tensor:
        """Generalised Mean pooling over patch tokens.

        patch_tokens: (B, N_patches, D). Returns (B, D).
        """
        x = patch_tokens.clamp(min=eps)
        return x.pow(p).mean(dim=1).pow(1.0 / p)

    def embed_image(self, rgb: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        rgb = self._ensure_rgb(rgb)
        rgb = self._apply_mask(rgb, mask)

        x = self._preprocess(rgb)

        mode = self.cfg.embedding_mode
        if mode == "cls":
            # Original path — single forward, take CLS.
            with torch.inference_mode():
                feat = self.model(x)
            if isinstance(feat, dict):
                if "x_norm_clstoken" not in feat:
                    raise RuntimeError(f"Unexpected dict output keys: {list(feat.keys())}")
                feat = feat["x_norm_clstoken"]
            feat = feat.reshape(1, -1)
        elif mode in ("patch_gem", "concat"):
            # Need patch tokens — use forward_features which returns the dict.
            with torch.inference_mode():
                feats = self.model.forward_features(x)
            if not isinstance(feats, dict):
                raise RuntimeError(
                    f"forward_features returned {type(feats)}; expected dict for patch tokens"
                )
            cls_tok = feats["x_norm_clstoken"].reshape(1, -1)
            patch_toks = feats["x_norm_patchtokens"]  # (1, N, D)
            gem = self._gem_pool(patch_toks, p=float(self.cfg.gem_p)).reshape(1, -1)
            if mode == "patch_gem":
                feat = gem
            else:  # concat
                # L2-normalise each component before concat so neither dominates.
                cls_n = F.normalize(cls_tok, dim=1)
                gem_n = F.normalize(gem, dim=1)
                feat = torch.cat([cls_n, gem_n], dim=1)
        else:
            raise ValueError(f"Unknown embedding_mode: {self.cfg.embedding_mode!r}")

        if self.cfg.normalize_embeddings:
            feat = F.normalize(feat, dim=1)

        return feat.squeeze(0).detach().cpu().numpy()

    def build_reference_bank_from_folder(self, cache_path: str | None = None) -> None:
        root = Path(self.cfg.reference_dir)
        if not root.exists():
            raise FileNotFoundError(f"Reference directory does not exist: {root}")

        # Default cache path next to reference dir. Tag with model + mode so
        # switching DINO model or embedding mode doesn't reuse a stale cache
        # of incompatible dimensionality.
        if cache_path is None:
            tag = f"{self.cfg.model_name}__{self.cfg.embedding_mode}"
            cache_path = str(root / f"_embedding_cache__{tag}.npz")

        # Try to load from cache
        if Path(cache_path).exists():
            try:
                cached = np.load(cache_path, allow_pickle=True)
                cached_paths = list(cached["image_paths"])
                cached_object_ids = list(cached["object_ids"])
                cached_embeddings = cached["embeddings"]

                # Verify cache matches current folder structure
                current_paths = []
                for object_dir in sorted(p for p in root.iterdir() if p.is_dir()):
                    for img_path in sorted(object_dir.iterdir()):
                        if img_path.suffix.lower() in self.cfg.allowed_exts:
                            current_paths.append(str(img_path))

                if cached_paths == current_paths:
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
                    print(f"[DINO] Loaded {len(bank)} embeddings from cache")
                    return
            except Exception as e:
                print(f"[DINO] Cache load failed, rebuilding: {e}")

        # Build from scratch
        bank: list[ReferenceEmbedding] = []

        for object_dir in sorted(p for p in root.iterdir() if p.is_dir()):
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
            raise RuntimeError(f"No valid reference images found in {root}")

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
            print(f"[DINO] Saved {len(bank)} embeddings to cache: {cache_path}")
        except Exception as e:
            print(f"[DINO] Failed to save cache: {e}")




    def set_reference_bank(self, refs: Iterable[ReferenceEmbedding]) -> None:
        refs = list(refs)
        if not refs:
            raise ValueError("Reference bank is empty")

        self.reference_bank = refs
        self.reference_matrix = np.stack([r.embedding for r in refs], axis=0)
        self.reference_object_ids = [r.object_id for r in refs]
        

    @staticmethod
    def _pairwise_similarity(query: np.ndarray, bank: np.ndarray, kind: str) -> np.ndarray:
        """Compute (1, N) similarity row between query and bank rows.

        Both query (1, D) and bank (N, D) are assumed L2-normalised when
        the caller normalises embeddings. For Tanimoto we recompute on the
        raw values to handle the |q|^2 + |r|^2 - q.r denominator.
        """
        if kind == "cosine":
            return (query @ bank.T).reshape(-1)
        if kind == "tanimoto":
            qr = (query @ bank.T).reshape(-1)
            qq = float(np.sum(query * query))
            rr = np.sum(bank * bank, axis=1)
            denom = qq + rr - qr
            return qr / np.maximum(denom, 1e-12)
        raise ValueError(f"Unknown similarity kind: {kind!r}")

    def classify_embedding(
        self,
        embedding: np.ndarray,
        objectness_prior: dict[str, float] | None = None,
    ) -> DINOResult:
        """Classify a query embedding against the reference bank.

        objectness_prior: optional per-object Bayesian prior weight. Posterior
        for object o becomes: score(o) * prior(o). Useful when the proposal
        stage (e.g. SAM stability score, mask-area plausibility) carries
        information about which class the candidate is more likely to be.
        Missing keys default to 1.0 (uniform).
        """
        if self.reference_matrix is None or len(self.reference_bank) == 0:
            raise RuntimeError("Reference bank is empty. Build or set it first.")

        embedding = np.asarray(embedding, dtype=np.float32).reshape(1, -1)

        if self.cfg.normalize_embeddings:
            norm = np.linalg.norm(embedding, axis=1, keepdims=True) + 1e-12
            embedding = embedding / norm

        sims = self._pairwise_similarity(embedding, self.reference_matrix, self.cfg.similarity)

        scores_by_object: dict[str, list[float]] = {}
        for sim, obj_id in zip(sims, self.reference_object_ids):
            scores_by_object.setdefault(obj_id, []).append(float(sim))

        top_k = 3
        agg: dict[str, float] = {}
        for obj_id, vals in scores_by_object.items():
            vals_sorted = sorted(vals, reverse=True)
            k = min(top_k, len(vals_sorted))
            agg[obj_id] = float(np.mean(vals_sorted[:k]))

        # Joint absolute + relative score (MUSE-style). Blends the raw aggregate
        # with how much the top class dominates the rest. alpha=0 -> raw scores.
        alpha = float(self.cfg.joint_score_alpha)
        if alpha > 0.0 and len(agg) >= 2:
            sorted_vals = sorted(agg.values(), reverse=True)
            other_mean = float(np.mean(sorted_vals[1:]))
            relative = {k: (v - other_mean) for k, v in agg.items()}
            joint = {k: (1.0 - alpha) * agg[k] + alpha * relative[k] for k in agg}
            agg_for_decision = joint
        else:
            agg_for_decision = agg

        # Bayesian objectness prior: multiplicative weighting on the decision score.
        if objectness_prior:
            agg_for_decision = {
                k: v * float(objectness_prior.get(k, 1.0))
                for k, v in agg_for_decision.items()
            }

        best_obj = max(agg_for_decision, key=agg_for_decision.get)
        best_score = agg_for_decision[best_obj]

        return DINOResult(
            object_id=best_obj,
            score=float(best_score),
            embedding=embedding.squeeze(0),
            scores_by_object=agg_for_decision,
        )


    def classify_crop(self, rgb: np.ndarray, mask: np.ndarray | None = None) -> DINOResult:
        # Apply mask if enabled
        print(f"[DINO DEBUG] classify_crop called, rgb shape: {rgb.shape}")

        rgb_processed = self._apply_mask(rgb, mask)
        
        # Debug: save the crop before preprocessing
        if self.debug_enabled:
            ts = int(time.time() * 1000)
            cv2.imwrite(
                f"{self.debug_dir}/query_crops/crop_{ts}.png",
                cv2.cvtColor(rgb_processed, cv2.COLOR_RGB2BGR)
            )
        
        emb = self.embed_image(rgb, mask=mask)
        result = self.classify_embedding(emb)
        
        # Debug: save match visualization
        if self.debug_enabled and result.score > 0.5:
            self._save_match_debug(rgb_processed, result, ts)
        
        return result

    def _save_match_debug(self, query_crop: np.ndarray, result: DINOResult, ts: int) -> None:
        """Save side-by-side of query crop and best matching reference."""
        # Find best matching reference image
        best_ref = None
        for ref in self.reference_bank:
            if ref.object_id == result.object_id:
                if best_ref is None:
                    best_ref = ref
                else:
                    # Check if this ref has higher similarity
                    sim = np.dot(result.embedding, ref.embedding)
                    best_sim = np.dot(result.embedding, best_ref.embedding)
                    if sim > best_sim:
                        best_ref = ref
        
        if best_ref is None:
            return
        
        # Load reference image
        ref_bgr = cv2.imread(best_ref.image_path)
        if ref_bgr is None:
            return
        ref_rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)
        
        # Resize both to same height for comparison
        h = 200
        query_resized = cv2.resize(query_crop, (h, h))
        ref_resized = cv2.resize(ref_rgb, (h, h))
        
        # Concatenate side by side
        combined = np.hstack([query_resized, ref_resized])
        
        # Add text
        cv2.putText(combined, f"{result.object_id}: {result.score:.3f}", 
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.imwrite(
            f"{self.debug_dir}/matches/match_{ts}_{result.object_id}_{result.score:.2f}.png",
            cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
        )