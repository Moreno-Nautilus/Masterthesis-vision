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
    #   "cls"         — original DINOv2 CLS token (unchanged behaviour).
    #   "patch_gem"   — Generalised-Mean-pooled patch tokens (more spatial
    #                   detail; better at distinguishing similar objects).
    #   "concat"      — concatenate L2-normalised CLS + GeM patch features.
    #   "two_stream"  — MUSE Eq.4: separate Tanimoto/cosine on CLS-only and
    #                   GeM-patch-only banks, blended by stream_alpha.
    #   "patch_match" — B2/M1: true per-patch token matching for
    #                   disambiguation. Stores the full (N_patches, D) grid
    #                   per reference; scores via mutual-NN + Lowe-ratio
    #                   inlier counting (SuperGlue-style spatial matching).
    embedding_mode: str = "cls"
    gem_p: float = 3.0  # GeM pooling exponent. Higher = closer to max-pool.
    # similarity:
    #   "cosine"   — original.
    #   "tanimoto" — generalised Tanimoto / Jaccard for normalised vectors.
    #                Penalises descriptors that share magnitude only.
    similarity: str = "cosine"
    # Joint absolute + relative score. In MUSE Eq.6 this is `beta`:
    #   S_joint = beta * S_abs + (1 - beta) * S_rel.
    # alpha=0 -> pure raw score (legacy default).
    joint_score_alpha: float = 0.0
    # M-new1 / B2 / M1: per-stream blend weight (paper's "alpha" in Eq.4).
    # Only consulted when embedding_mode == "two_stream":
    #   S_int = stream_alpha * S_cls + (1 - stream_alpha) * S_patch.
    stream_alpha: float = 0.5
    # M3: how to derive the relative score from absolute per-class scores.
    #   "mean_sub"     — legacy: (score - mean(scores_of_other_classes)).
    #   "softmax_tau"  — MUSE Eq.5: softmax(S_abs / tau) over classes.
    relative_score_mode: str = "mean_sub"
    tau: float = 0.02  # MUSE temperature for the softmax-tau relative score.
    # M4: power-scaling exponent on the per-proposal objectness prior
    # (MUSE Eq.8). 0 disables the prior entirely; the paper uses 0.1.
    objectness_prior_gamma: float = 0.0
    # M-new5: PCA-whiten the reference bank to this many components before
    # matching. 0 disables; recommended values: 64 / 128. The same projection
    # is applied to every query embedding.
    bank_pca_dim: int = 0

    # B2 / M1: knobs for the "patch_match" embedding mode.
    #   lowe_ratio:    Lowe's ratio test, keep matches with
    #                  best_sim / second_best_sim >= ratio. <=0 disables.
    #   min_cos:       minimum cosine similarity for an inlier match.
    #   use_mutual_nn: require the match to be mutual nearest-neighbour
    #                  in both directions (query<->reference).
    #   bg_threshold:  pixel-sum threshold used to derive the foreground
    #                  patch mask from a pre-masked (black-background) RGB.
    patch_match_lowe_ratio: float = 1.05
    patch_match_min_cos: float = 0.5
    patch_match_use_mutual_nn: bool = True
    patch_match_bg_threshold: int = 10


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
        # Shape depends on embedding_mode:
        #   single-stream modes -> (N, D)
        #   "two_stream"        -> (N, 2, D)        rows = [CLS, GeM]
        #   "patch_match"       -> (N, N_patches, D) per-ref patch grids
        self.reference_matrix: np.ndarray | None = None
        self.reference_object_ids: list[str] = []
        # B2 / M1: foreground patch mask per reference, shape (N, N_patches).
        # Only populated when embedding_mode == "patch_match".
        self.reference_patch_masks: np.ndarray | None = None
        # M-new5: bank whitening state (None when --dino-bank-pca-dim == 0).
        self._pca_mean: np.ndarray | None = None  # (D,)
        self._pca_components: np.ndarray | None = None  # (k, D)
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

    def _patch_size(self) -> int:
        """Return the DINOv2 patch side length (14 for ViT-B/L/14, 16 otherwise)."""
        name = (self.cfg.model_name or "").lower()
        if "14" in name:
            return 14
        if "16" in name:
            return 16
        ps = getattr(self.model, "patch_size", None)
        if ps:
            return int(ps)
        return 14

    def _patch_grid_size(self) -> int:
        return int(self.cfg.input_size) // int(self._patch_size())

    def _center_crop_resize_uint8(self, rgb: np.ndarray) -> np.ndarray:
        """Square center-crop + resize matching _preprocess. Returns uint8."""
        rgb = self._center_crop_square(rgb)
        return cv2.resize(
            rgb, (self.cfg.input_size, self.cfg.input_size),
            interpolation=cv2.INTER_AREA,
        )

    def _patch_mask_from_input_mask(self, mask: np.ndarray) -> np.ndarray:
        """Downsample a HxW binary mask onto the patch grid (max-pool).

        Mirrors the center-crop + resize chain used in _preprocess so the
        returned (N_patches,) bool vector aligns with the patch tokens.
        """
        m = mask.astype(np.uint8)
        m = self._center_crop_square(m)
        m = cv2.resize(
            m, (self.cfg.input_size, self.cfg.input_size),
            interpolation=cv2.INTER_NEAREST,
        )
        ps = self._patch_size()
        g = self._patch_grid_size()
        # max-pool over ps x ps blocks
        grid = m.reshape(g, ps, g, ps).max(axis=(1, 3))
        return (grid > 0).reshape(-1).astype(bool)

    def _patch_mask_from_masked_rgb(self, rgb: np.ndarray) -> np.ndarray:
        """Derive a foreground patch mask from a pre-masked (black-bg) RGB."""
        rgb_r = self._center_crop_resize_uint8(self._ensure_rgb(rgb))
        thr = int(max(0, self.cfg.patch_match_bg_threshold))
        fg = (rgb_r.astype(np.int32).sum(axis=2) > thr)
        ps = self._patch_size()
        g = self._patch_grid_size()
        grid = fg.reshape(g, ps, g, ps).any(axis=(1, 3))
        return grid.reshape(-1)

    def embed_image(self, rgb: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        """Compute the embedding for one RGB crop.

        Return shape depends on embedding_mode:
          - "cls" / "patch_gem"             -> (D,)
          - "concat"                        -> (2D,)
          - "two_stream"                    -> (2, D)         rows = [CLS, GeM]
          - "patch_match"                   -> (N_patches, D) L2-normalised
                                              patch tokens. The matching
                                              foreground patch mask is derived
                                              separately via
                                              `_patch_mask_from_input_mask` /
                                              `_patch_mask_from_masked_rgb`.
        """
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
            if self.cfg.normalize_embeddings:
                feat = F.normalize(feat, dim=1)
            return feat.squeeze(0).detach().cpu().numpy()

        if mode == "patch_match":
            with torch.inference_mode():
                feats = self.model.forward_features(x)
            if not isinstance(feats, dict):
                raise RuntimeError(
                    f"forward_features returned {type(feats)}; expected dict"
                )
            patch_toks = feats["x_norm_patchtokens"]  # (1, N, D)
            patch_toks = F.normalize(patch_toks, dim=2)
            return patch_toks.squeeze(0).detach().cpu().numpy()  # (N, D)

        if mode in ("patch_gem", "concat", "two_stream"):
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
                if self.cfg.normalize_embeddings:
                    feat = F.normalize(feat, dim=1)
                return feat.squeeze(0).detach().cpu().numpy()

            if mode == "concat":
                cls_n = F.normalize(cls_tok, dim=1)
                gem_n = F.normalize(gem, dim=1)
                feat = torch.cat([cls_n, gem_n], dim=1)
                if self.cfg.normalize_embeddings:
                    feat = F.normalize(feat, dim=1)
                return feat.squeeze(0).detach().cpu().numpy()

            # two_stream — return (2, D), each row L2-normalised independently.
            cls_n = F.normalize(cls_tok, dim=1).squeeze(0)
            gem_n = F.normalize(gem, dim=1).squeeze(0)
            out = torch.stack([cls_n, gem_n], dim=0)  # (2, D)
            return out.detach().cpu().numpy()

        raise ValueError(f"Unknown embedding_mode: {self.cfg.embedding_mode!r}")

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

        # Default cache path next to primary reference dir. Tag with model + mode
        # AND a digest of all root paths so caches don't get reused across
        # different reference-source combinations.
        if cache_path is None:
            roots_tag = "_".join(sorted(r.name for r in roots))
            # gem_p affects patch-pool output; size affects raw token grid.
            extra_tag = (
                f"gem{float(self.cfg.gem_p):g}_in{int(self.cfg.input_size)}"
            )
            tag = (
                f"{self.cfg.model_name}__{self.cfg.embedding_mode}"
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

                # Verify cache matches current folder structure (across all roots).
                current_paths = []
                for r in roots:
                    for object_dir in sorted(p for p in r.iterdir() if p.is_dir()):
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
                    # B2 / M1: restore patch masks when present (patch_match cache).
                    pm = None
                    try:
                        if "patch_masks" in cached.files:
                            pm = np.asarray(cached["patch_masks"], dtype=bool)
                    except Exception:
                        pm = None
                    self.reference_patch_masks = pm
                    self._maybe_fit_bank_pca()
                    print(f"[DINO] Loaded {len(bank)} embeddings from cache")
                    return
            except Exception as e:
                print(f"[DINO] Cache load failed, rebuilding: {e}")

        # Build from scratch (iterate across all reference roots).
        bank: list[ReferenceEmbedding] = []
        patch_masks: list[np.ndarray] = []
        is_patch_match = (self.cfg.embedding_mode == "patch_match")

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
                if is_patch_match:
                    # Reference crops on disk are pre-masked (black bg);
                    # derive the foreground patch mask from the image itself.
                    patch_masks.append(self._patch_mask_from_masked_rgb(rgb))

        if not bank:
            raise RuntimeError(
                f"No valid reference images found under: {[str(r) for r in roots]}"
            )

        self.reference_bank = bank
        # Two-stream embeddings have shape (2, D) per sample; np.stack pushes
        # the new axis to front, giving (N, 2, D). Single-stream stays (N, D).
        # patch_match per-sample shape is (N_p, D) -> stacked (N, N_p, D).
        self.reference_matrix = np.stack([r.embedding for r in bank], axis=0)
        self.reference_object_ids = [r.object_id for r in bank]
        self.reference_patch_masks = (
            np.stack(patch_masks, axis=0).astype(bool) if is_patch_match else None
        )
        self._maybe_fit_bank_pca()

        # Save cache
        try:
            extra = {}
            if self.reference_patch_masks is not None:
                extra["patch_masks"] = self.reference_patch_masks
            np.savez(
                cache_path,
                image_paths=np.array([r.image_path for r in bank], dtype=object),
                object_ids=np.array([r.object_id for r in bank], dtype=object),
                embeddings=self.reference_matrix,
                **extra,
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
        self._maybe_fit_bank_pca()
        

    @staticmethod
    def _pairwise_similarity(query: np.ndarray, bank: np.ndarray, kind: str) -> np.ndarray:
        """Compute (N,) similarity row between query (1, D) and bank (N, D).

        Both are assumed L2-normalised when the caller normalises embeddings.
        For Tanimoto we recompute on the raw values to handle the
        |q|^2 + |r|^2 - q.r denominator.
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

    # ── M-new5: PCA / whitening on the reference bank ──────────────────────
    def _maybe_fit_bank_pca(self) -> None:
        """Fit a PCA-whitening projection on the current reference matrix.

        Only used when `bank_pca_dim > 0`. After fitting, the stored matrix
        is the projected version (the original raw embeddings are dropped
        from `reference_matrix`; ReferenceEmbedding entries keep the raw
        vectors for cache integrity). Two-stream mode fits one PCA *per
        stream* and stores the projected (N, 2, k) matrix.
        """
        k = int(self.cfg.bank_pca_dim)
        if k <= 0 or self.reference_matrix is None:
            self._pca_mean = None
            self._pca_components = None
            return
        mat = self.reference_matrix
        if mat.ndim == 2:
            self._pca_mean, self._pca_components = self._fit_one_pca(mat, k)
            self.reference_matrix = self._apply_pca(mat, self._pca_mean, self._pca_components)
        elif mat.ndim == 3 and mat.shape[1] == 2:
            mean_a, comp_a = self._fit_one_pca(mat[:, 0, :], k)
            mean_b, comp_b = self._fit_one_pca(mat[:, 1, :], k)
            self._pca_mean = np.stack([mean_a, mean_b], axis=0)        # (2, D)
            self._pca_components = np.stack([comp_a, comp_b], axis=0)  # (2, k, D)
            proj_a = self._apply_pca(mat[:, 0, :], mean_a, comp_a)
            proj_b = self._apply_pca(mat[:, 1, :], mean_b, comp_b)
            self.reference_matrix = np.stack([proj_a, proj_b], axis=1)  # (N, 2, k)
        else:
            print(f"[DINO] PCA skipped: unsupported bank shape {mat.shape}")
            self._pca_mean = None
            self._pca_components = None

    @staticmethod
    def _fit_one_pca(mat: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (mean (D,), whitened_components (k, D)).

        True PCA whitening: the top-k right-singular vectors are pre-scaled
        by 1/σ so that ``centred @ comp.T`` yields per-axis unit-variance
        outputs. A small ε is added to σ for numerical stability on
        near-zero modes. The 1/sqrt(N-1) sample-variance factor is omitted
        because the downstream comparison is cosine similarity (which is
        invariant to a global scale, and we L2-renorm after projection).
        """
        mat = np.asarray(mat, dtype=np.float32)
        mean = mat.mean(axis=0)
        centred = mat - mean
        # SVD on the centred data: rows of Vt are the principal directions,
        # s are the singular values (sqrt of eigenvalues of the covariance,
        # up to the omitted 1/sqrt(N-1) factor).
        _, s, vt = np.linalg.svd(centred, full_matrices=False)
        k_eff = max(1, min(k, vt.shape[0]))
        vt_k = vt[:k_eff]
        s_k = s[:k_eff]
        eps = 1e-6 * (float(s.max()) if s.size > 0 else 1.0)
        scale = 1.0 / (s_k + eps)  # (k,)
        comp = vt_k * scale[:, None]  # (k, D), whitened
        return mean.astype(np.float32), comp.astype(np.float32)

    @staticmethod
    def _apply_pca(
        mat: np.ndarray,
        mean: np.ndarray,
        components: np.ndarray,
    ) -> np.ndarray:
        centred = mat - mean
        proj = centred @ components.T  # (N, k)
        # L2-renormalise so cosine on the projected space remains meaningful.
        n = np.linalg.norm(proj, axis=1, keepdims=True) + 1e-12
        return (proj / n).astype(np.float32)

    def _maybe_project_query(self, query: np.ndarray) -> np.ndarray:
        """Apply the fitted PCA projection (if any) to a query embedding."""
        if self._pca_mean is None or self._pca_components is None:
            return query
        if query.ndim == 1 and self._pca_components.ndim == 2:
            return self._apply_pca(query.reshape(1, -1), self._pca_mean, self._pca_components).squeeze(0)
        if query.ndim == 2 and self._pca_components.ndim == 3:
            # two_stream: query (2, D), components (2, k, D), mean (2, D)
            out = np.stack(
                [
                    self._apply_pca(
                        query[i].reshape(1, -1),
                        self._pca_mean[i],
                        self._pca_components[i],
                    ).squeeze(0)
                    for i in range(query.shape[0])
                ],
                axis=0,
            )
            return out
        return query

    # ── B2 / M1: per-patch token matching ──────────────────────────────────
    def _score_query_vs_ref_patches(
        self,
        q_patches: np.ndarray,        # (n_q, D), L2-normalised
        r_patches: np.ndarray,        # (n_r, D), L2-normalised
    ) -> float:
        """Inlier-fraction score for one (query, reference) pair.

        Builds the (n_q, n_r) cosine matrix, then applies
        mutual-NN + Lowe-ratio + min-cosine filters and returns
        `inliers / max(1, n_q)`.
        """
        if q_patches.shape[0] == 0 or r_patches.shape[0] == 0:
            return 0.0

        sim = q_patches @ r_patches.T  # (n_q, n_r); cosine since inputs are L2-normed
        if sim.size == 0:
            return 0.0

        n_q = sim.shape[0]
        n_r = sim.shape[1]
        fwd = np.argmax(sim, axis=1)          # (n_q,) best ref-patch per query
        best_sim = sim[np.arange(n_q), fwd]   # (n_q,)

        min_cos = float(self.cfg.patch_match_min_cos)
        keep = best_sim >= min_cos

        if self.cfg.patch_match_use_mutual_nn and n_r > 0:
            bwd = np.argmax(sim, axis=0)      # (n_r,) best query-patch per ref-patch
            keep &= (bwd[fwd] == np.arange(n_q))

        ratio = float(self.cfg.patch_match_lowe_ratio)
        if ratio > 0.0 and n_r >= 2:
            sim_masked = sim.copy()
            sim_masked[np.arange(n_q), fwd] = -np.inf
            second = np.max(sim_masked, axis=1)
            # Replace non-positive second-best with a tiny epsilon to avoid
            # divide-by-zero giving infinite "distinctiveness".
            second = np.maximum(second, 1e-6)
            keep &= (best_sim / second) >= ratio

        inliers = int(np.count_nonzero(keep))
        return float(inliers) / float(max(1, n_q))

    def _patch_match_sims(self, query_patches: np.ndarray, query_mask: np.ndarray | None) -> np.ndarray:
        """Compute per-reference inlier-fraction scores for a patch_match query.

        query_patches: (N_pq, D)  L2-normalised patch tokens of the query.
        query_mask:    (N_pq,) bool — True for foreground patches, or None to
                       use all patches.
        Returns:       (N_refs,) per-reference scalar score in [0, 1].
        """
        if self.reference_matrix is None or self.reference_matrix.ndim != 3:
            raise RuntimeError(
                "patch_match: reference_matrix must be (N_refs, N_p, D)."
            )

        n_refs = self.reference_matrix.shape[0]
        if query_mask is None:
            q_fg = query_patches
        else:
            q_fg = query_patches[query_mask.astype(bool)]
            if q_fg.shape[0] == 0:
                # Fall back to all patches if mask is empty (e.g. degenerate crop).
                q_fg = query_patches

        out = np.zeros(n_refs, dtype=np.float32)
        for i in range(n_refs):
            r_patches = self.reference_matrix[i]
            if self.reference_patch_masks is not None:
                r_mask = self.reference_patch_masks[i].astype(bool)
                r_fg = r_patches[r_mask] if r_mask.any() else r_patches
            else:
                r_fg = r_patches
            out[i] = self._score_query_vs_ref_patches(q_fg, r_fg)
        return out

    # ── per-class score aggregation helpers ────────────────────────────────
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
        query_patch_mask: np.ndarray | None = None,
    ) -> DINOResult:
        """Classify a query embedding against the reference bank.

        objectness_prior:
            * float    — MUSE Eq.10: per-proposal scalar applied as
                         `prior^gamma * S_joint` to every class score.
                         `gamma` comes from `cfg.objectness_prior_gamma`.
                         Note: this uniformly scales class scores, so it
                         does not change argmax — but it scales the final
                         score used for downstream proposal ranking.
            * dict     — legacy per-object multiplicative weight (changes argmax).
            * None     — no prior.

        query_patch_mask:
            Only used when embedding_mode == "patch_match". Bool array of
            shape (N_patches,) selecting foreground patches of the query.
            When None, all query patches are used.
        """
        if self.reference_matrix is None or len(self.reference_bank) == 0:
            raise RuntimeError("Reference bank is empty. Build or set it first.")

        embedding = np.asarray(embedding, dtype=np.float32)

        patch_match = (
            self.cfg.embedding_mode == "patch_match"
            and embedding.ndim == 2
            and self.reference_matrix.ndim == 3
        )

        two_stream = (
            self.cfg.embedding_mode == "two_stream"
            and embedding.ndim == 2
            and embedding.shape[0] == 2
            and self.reference_matrix.ndim == 3
        )

        # PCA projection (no-op when not configured). patch_match buckets
        # disable PCA because the projection is fit on aggregate embeddings
        # rather than per-patch tokens; mixing the two would silently corrupt
        # the matching geometry.
        if not patch_match:
            embedding = self._maybe_project_query(embedding)

        if patch_match:
            # Each row of `embedding` is one L2-normalised patch token.
            sims = self._patch_match_sims(embedding, query_patch_mask)
            # Keep a representative embedding for downstream consumers
            # (memory bank, debug viz). Use the mean foreground patch.
            if query_patch_mask is not None and query_patch_mask.any():
                emb_for_pool = embedding[query_patch_mask.astype(bool)]
            else:
                emb_for_pool = embedding
            embedding_out = emb_for_pool.mean(axis=0)
            n = float(np.linalg.norm(embedding_out)) + 1e-12
            embedding_out = (embedding_out / n).astype(np.float32)
        elif two_stream:
            # Compute S_cls and S_patch independently and blend (MUSE Eq.4).
            q_cls = embedding[0:1, :]
            q_patch = embedding[1:2, :]
            bank_cls = self.reference_matrix[:, 0, :]
            bank_patch = self.reference_matrix[:, 1, :]
            if self.cfg.normalize_embeddings:
                q_cls = q_cls / (np.linalg.norm(q_cls, axis=1, keepdims=True) + 1e-12)
                q_patch = q_patch / (np.linalg.norm(q_patch, axis=1, keepdims=True) + 1e-12)
            sims_cls = self._pairwise_similarity(q_cls, bank_cls, self.cfg.similarity)
            sims_patch = self._pairwise_similarity(q_patch, bank_patch, self.cfg.similarity)
            sa = float(self.cfg.stream_alpha)
            sims = sa * sims_cls + (1.0 - sa) * sims_patch
            # Keep a representative embedding for downstream consumers
            # (memory bank, debug viz). Use the CLS row.
            embedding_out = embedding[0]
        else:
            embedding = embedding.reshape(1, -1)
            if self.cfg.normalize_embeddings:
                norm = np.linalg.norm(embedding, axis=1, keepdims=True) + 1e-12
                embedding = embedding / norm
            sims = self._pairwise_similarity(
                embedding, self.reference_matrix, self.cfg.similarity
            )
            embedding_out = embedding.squeeze(0)

        agg = self._aggregate_per_class(sims, top_k=3)

        # ── MUSE Eq.5/Eq.6: joint absolute + relative score ────────────────
        beta = float(self.cfg.joint_score_alpha)  # paper "beta"
        if beta > 0.0 and len(agg) >= 2:
            mode = self.cfg.relative_score_mode
            if mode == "softmax_tau":
                tau = max(1e-6, float(self.cfg.tau))
                names = list(agg.keys())
                logits = np.array([agg[k] for k in names], dtype=np.float64) / tau
                logits -= logits.max()  # numerical stability
                probs = np.exp(logits)
                probs /= probs.sum() + 1e-12
                relative = {k: float(p) for k, p in zip(names, probs)}
            elif mode == "mean_sub":
                sorted_vals = sorted(agg.values(), reverse=True)
                other_mean = float(np.mean(sorted_vals[1:]))
                relative = {k: (v - other_mean) for k, v in agg.items()}
            else:
                raise ValueError(f"Unknown relative_score_mode: {mode!r}")
            joint = {k: beta * agg[k] + (1.0 - beta) * relative[k] for k in agg}
            agg_for_decision = joint
        else:
            agg_for_decision = dict(agg)

        # ── MUSE Eq.8/Eq.10: per-proposal objectness prior ─────────────────
        if isinstance(objectness_prior, (int, float)):
            gamma = float(self.cfg.objectness_prior_gamma)
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
        # Find best matching reference image. patch_match references are
        # (N_patches, D); fall back to picking the first ref of the class.
        is_patch_match = (self.cfg.embedding_mode == "patch_match")
        best_ref = None
        for ref in self.reference_bank:
            if ref.object_id == result.object_id:
                if best_ref is None:
                    best_ref = ref
                elif not is_patch_match:
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