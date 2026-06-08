"""Standalone SAM2 'cursed build' stress test.

Builds N fresh SAM2 models under different torch numerical-backend settings and
counts how many come up CURSED (fail warmup = produce empty/garbage masks on the
synthetic dummy). math-SDP is always on (it's hardcoded in _predict_boxes_once),
so this measures the RESIDUAL curse rate on top of it, and whether TF32 / cuDNN-
determinism / fp32 change it.

Run inside the container:
  python scripts/sam_curse_stress.py [N]
"""
from __future__ import annotations

import sys

import torch

sys.path.insert(0, "/workspace/MasterThesis")
from src.perception.learned.SAM.sam_segmentation import (  # noqa: E402
    SAMSegmenter,
    SAMSegmenterConfig,
)

CKPT = "external/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
CFG = "configs/sam2.1/sam2.1_hiera_b+.yaml"


def make_cfg(use_bf16: bool) -> SAMSegmenterConfig:
    return SAMSegmenterConfig(
        repo_root="external/sam2",
        checkpoint=CKPT,
        model_cfg=CFG,
        device="cuda",
        max_image_side=1536,
        min_mask_area=10,
        min_bbox_side_px=2,
        use_bfloat16=use_bf16,
        attach_rgb_crops=False,
    )


def set_backends(tf32: bool, deterministic: bool) -> None:
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def trial(n: int, use_bf16: bool) -> int:
    cold = 0
    pattern = []
    for _ in range(n):
        seg = SAMSegmenter(make_cfg(use_bf16))
        warm = seg.warmup(max_iters=6)
        pattern.append("." if warm else "X")
        if not warm:
            cold += 1
        del seg
        torch.cuda.empty_cache()
    print(f"    builds: {''.join(pattern)}  ('.'=warm 'X'=cursed)")
    return cold


def run(name: str, n: int, use_bf16: bool, tf32: bool, deterministic: bool) -> None:
    set_backends(tf32=tf32, deterministic=deterministic)
    print(f"[{name}]")
    cold = trial(n, use_bf16)
    print(f"  => {cold}/{n} cursed\n")


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    print(f"torch {torch.__version__} | cuda {torch.version.cuda} | "
          f"device {torch.cuda.get_device_name(0)}\n")

    # Absorb the documented 'first build in a process' curse so we measure
    # steady-state, not the one-off first-build artifact.
    print("(absorbing first-build curse with a throwaway...)")
    _tw = SAMSegmenter(make_cfg(True))
    _tw.warmup(max_iters=6)
    del _tw
    torch.cuda.empty_cache()
    print()

    run("A  bf16, tf32 ON,  cudnn default", n, use_bf16=True,  tf32=True,  deterministic=False)
    run("B  bf16, tf32 OFF", n, use_bf16=True,  tf32=False, deterministic=False)
    run("C  bf16, cudnn DETERMINISTIC",   n, use_bf16=True,  tf32=True,  deterministic=True)
    run("D  fp32, tf32 ON,  cudnn default", n, use_bf16=False, tf32=True,  deterministic=False)


if __name__ == "__main__":
    main()
