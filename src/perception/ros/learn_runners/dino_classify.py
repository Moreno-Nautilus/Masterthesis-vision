"""Batched DINOv2 crop classification against the reference bank.

Extracted from run_pipeline_track_multicam_realsense.py.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from src.perception.learned.DINO.dino_identifier import DINOIdentifier, DINOResult


def batch_dino_classify(
    dino: DINOIdentifier,
    crops_rgb: list[np.ndarray],
    crops_mask: list[np.ndarray | None],
    objectness_priors: list[float | None] | None = None,
) -> list[DINOResult]:
    """Embed `crops_rgb` in one forward pass, then classify each via the ref bank."""

    if not crops_rgb:
        return []
    if objectness_priors is not None and len(objectness_priors) != len(crops_rgb):
        raise ValueError(
            f"objectness_priors len {len(objectness_priors)} != crops {len(crops_rgb)}"
        )

    tensors = []
    for rgb, mask in zip(crops_rgb, crops_mask):
        rgb_proc = dino._ensure_rgb(rgb)
        rgb_masked = dino._apply_mask(rgb_proc, mask)
        t = dino._preprocess(rgb_masked)
        tensors.append(t)

    batch = torch.cat(tensors, dim=0)

    with torch.inference_mode():
        out = dino.model.forward_features(batch)
    if not isinstance(out, dict):
        raise RuntimeError(
            f"forward_features returned {type(out)}; expected dict"
        )
    cls_tok = out["x_norm_clstoken"].reshape(batch.shape[0], -1)
    patch_toks = out["x_norm_patchtokens"]  # (B, N, D)
    gem = dino._gem_pool(patch_toks, p=float(dino.cfg.gem_p)).reshape(
        batch.shape[0], -1
    )
    cls_n = F.normalize(cls_tok, dim=1)
    gem_n = F.normalize(gem, dim=1)
    embeddings = torch.stack([cls_n, gem_n], dim=1).detach().cpu().numpy()

    results: list[DINOResult] = []
    for i, emb in enumerate(embeddings):
        prior = None
        if objectness_priors is not None:
            p = objectness_priors[i]
            if p is not None:
                prior = float(p)
        results.append(
            dino.classify_embedding(
                emb,
                objectness_prior=prior,
            )
        )
    return results
