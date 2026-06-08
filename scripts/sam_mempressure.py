"""Does GPU memory pressure trigger the SAM 'cursed build'?

Occupies increasing amounts of VRAM, then stresses SAM build+warmup at each level.
If the cursed rate climbs as free memory shrinks, the curse is memory-pressure /
fragmentation driven — which is what the full pipeline (many resident models)
creates and the clean standalone does not.
"""
import sys

sys.path.insert(0, "/workspace/MasterThesis")
import torch  # noqa: E402
from src.perception.learned.SAM.sam_segmentation import (  # noqa: E402
    SAMSegmenter,
    SAMSegmenterConfig,
)


def cfg():
    return SAMSegmenterConfig(
        repo_root="external/sam2",
        checkpoint="external/sam2/checkpoints/sam2.1_hiera_base_plus.pt",
        model_cfg="configs/sam2.1/sam2.1_hiera_b+.yaml", device="cuda",
        max_image_side=1536, min_mask_area=10, min_bbox_side_px=2,
        use_bfloat16=True, attach_rgb_crops=False,
    )


def stress(n, tag):
    cold = 0
    pat = []
    for _ in range(n):
        s = SAMSegmenter(cfg())
        w = s.warmup(max_iters=6)
        pat.append("." if w else "X")
        cold += 0 if w else 1
        del s
        torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info()
    print(f"[{tag}] free={free/1e9:.1f}/{total/1e9:.1f}GB  "
          f"builds: {''.join(pat)}  => {cold}/{n} cursed", flush=True)


total = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU total {total:.1f} GB\n", flush=True)
blocks = []
for gb in [0, 6, 10, 14, 17, 19, 20, 21]:
    cur = sum(b.numel() * 2 for b in blocks) / 1e9
    if gb > cur:
        try:
            blocks.append(torch.empty(int((gb - cur) * 1e9 / 2),
                                      dtype=torch.float16, device="cuda"))
        except RuntimeError as e:
            print(f"(could not reserve {gb}GB: {str(e)[:60]})", flush=True)
            break
    stress(8, f"~{gb}GB held")
