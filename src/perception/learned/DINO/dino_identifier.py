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
        self.model.eval()

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
        rgb = self._ensure_rgb(rgb)
        rgb = self._center_crop_square(rgb)
        rgb = cv2.resize(rgb, (self.cfg.input_size, self.cfg.input_size), interpolation=cv2.INTER_AREA)

        x = rgb.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        x = (x - mean) / std
        x = np.transpose(x, (2, 0, 1))  # HWC -> CHW
        x = torch.from_numpy(x).unsqueeze(0).to(self.device)
        return x

    def embed_image(self, rgb: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        rgb = self._ensure_rgb(rgb)
        rgb = self._apply_mask(rgb, mask)

        x = self._preprocess(rgb)

        with torch.inference_mode():
            feat = self.model(x)

        if isinstance(feat, dict):
            if "x_norm_clstoken" in feat:
                feat = feat["x_norm_clstoken"]
            else:
                raise RuntimeError(f"Unexpected dict output keys: {list(feat.keys())}")

        feat = feat.reshape(1, -1)
        if self.cfg.normalize_embeddings:
            feat = F.normalize(feat, dim=1)

        return feat.squeeze(0).detach().cpu().numpy()

    # def build_reference_bank_from_folder(self) -> None:
    #     root = Path(self.cfg.reference_dir)
    #     if not root.exists():
    #         raise FileNotFoundError(f"Reference directory does not exist: {root}")

    #     bank: list[ReferenceEmbedding] = []

    #     for object_dir in sorted(p for p in root.iterdir() if p.is_dir()):
    #         object_id = object_dir.name
    #         for img_path in sorted(object_dir.iterdir()):
    #             if img_path.suffix.lower() not in self.cfg.allowed_exts:
    #                 continue

    #             bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    #             if bgr is None:
    #                 continue
    #             rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    #             emb = self.embed_image(rgb)
    #             bank.append(
    #                 ReferenceEmbedding(
    #                     object_id=object_id,
    #                     image_path=str(img_path),
    #                     embedding=emb,
    #                 )
    #             )

    #     if not bank:
    #         raise RuntimeError(f"No valid reference images found in {root}")

    #     self.reference_bank = bank
    #     self.reference_matrix = np.stack([r.embedding for r in bank], axis=0)
    #     self.reference_object_ids = [r.object_id for r in bank]

    def build_reference_bank_from_folder(self, cache_path: str | None = None) -> None:
        root = Path(self.cfg.reference_dir)
        if not root.exists():
            raise FileNotFoundError(f"Reference directory does not exist: {root}")

        # Default cache path next to reference dir
        if cache_path is None:
            cache_path = str(root / "_embedding_cache.npz")

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
        

    def classify_embedding(self, embedding: np.ndarray) -> DINOResult:
        if self.reference_matrix is None or len(self.reference_bank) == 0:
            raise RuntimeError("Reference bank is empty. Build or set it first.")

        embedding = np.asarray(embedding, dtype=np.float32).reshape(1, -1)

        if self.cfg.normalize_embeddings:
            norm = np.linalg.norm(embedding, axis=1, keepdims=True) + 1e-12
            embedding = embedding / norm

        sims = (embedding @ self.reference_matrix.T).reshape(-1)

        scores_by_object: dict[str, list[float]] = {}
        for sim, obj_id in zip(sims, self.reference_object_ids):
            scores_by_object.setdefault(obj_id, []).append(float(sim))

        top_k = 3
        agg = {}
        for obj_id, vals in scores_by_object.items():
            vals_sorted = sorted(vals, reverse=True)
            k = min(top_k, len(vals_sorted))
            agg[obj_id] = float(np.mean(vals_sorted[:k]))

        best_obj = max(agg, key=agg.get)
        best_score = agg[best_obj]

        return DINOResult(
            object_id=best_obj,
            score=float(best_score),
            embedding=embedding.squeeze(0),
            scores_by_object=agg,
        )

    # def classify_embedding(self, embedding: np.ndarray) -> DINOResult:
    #     if self.reference_matrix is None or len(self.reference_bank) == 0:
    #         raise RuntimeError("Reference bank is empty. Build or set it first.")

    #     embedding = np.asarray(embedding, dtype=np.float32).reshape(1, -1)

    #     if self.cfg.normalize_embeddings:
    #         norm = np.linalg.norm(embedding, axis=1, keepdims=True) + 1e-12
    #         embedding = embedding / norm

    #     sims = (embedding @ self.reference_matrix.T).reshape(-1)

    #     scores_by_object: dict[str, list[float]] = {}
    #     for sim, obj_id in zip(sims, self.reference_object_ids):
    #         scores_by_object.setdefault(obj_id, []).append(float(sim))

    #     # aggregate by max similarity per object
    #     agg = {k: max(v) for k, v in scores_by_object.items()}
    #     best_obj = max(agg, key=agg.get)
    #     best_score = agg[best_obj]

    #     return DINOResult(
    #         object_id=best_obj,
    #         score=float(best_score),
    #         embedding=embedding.squeeze(0),
    #         scores_by_object=agg,
    #     )

    # def classify_crop(self, rgb: np.ndarray, mask: np.ndarray | None = None) -> DINOResult:
    #     emb = self.embed_image(rgb, mask=mask)
    #     return self.classify_embedding(emb)

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