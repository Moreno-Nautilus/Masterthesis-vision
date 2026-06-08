"""Does SAM get 'cursed' only when it shares the GPU with nvdiffrast/warp?

Step 2.1 showed SAM builds 48/48 clean in isolation. This loads a FoundationPose
estimator first (which initialises nvdiffrast's CudaRaster context + warp — the
raw-CUDA tenants), then runs the same SAM build-and-warmup stress. If the cursed
rate jumps from ~0, the curse is a GPU-coexistence effect, not a SAM bug.

  python scripts/sam_curse_coexist.py [N] [mesh.obj]
"""
from __future__ import annotations

import sys

import numpy as np
import torch

sys.path.insert(0, "/workspace/MasterThesis")
from src.perception.learned.SAM.sam_segmentation import (  # noqa: E402
    SAMSegmenter,
    SAMSegmenterConfig,
)
from src.perception.learned.FP.pose_foundation import (  # noqa: E402
    FoundationPoseWrapper,
    FoundationPoseConfig,
)

CKPT = "external/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
CFG = "configs/sam2.1/sam2.1_hiera_b+.yaml"


def make_cfg() -> SAMSegmenterConfig:
    return SAMSegmenterConfig(
        repo_root="external/sam2", checkpoint=CKPT, model_cfg=CFG, device="cuda",
        max_image_side=1536, min_mask_area=10, min_bbox_side_px=2,
        use_bfloat16=True, attach_rgb_crops=False,
    )


def stress(n: int, tag: str) -> int:
    cold = 0
    pat = []
    for _ in range(n):
        seg = SAMSegmenter(make_cfg())
        warm = seg.warmup(max_iters=6)
        pat.append("." if warm else "X")
        cold += 0 if warm else 1
        del seg
        torch.cuda.empty_cache()
    print(f"[{tag}] builds: {''.join(pat)}  => {cold}/{n} cursed")
    return cold


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    mesh = sys.argv[2] if len(sys.argv) > 2 else "Data/CAD_Models_centered/cooling_f.obj"
    print(f"torch {torch.__version__} | cuda {torch.version.cuda} | "
          f"{torch.cuda.get_device_name(0)}\n")

    print("=== baseline: SAM alone (no FP loaded) ===")
    stress(n, "SAM-ALONE")

    print("\n=== loading FoundationPose (nvdiffrast CudaRaster + warp) ===")
    fp = FoundationPoseWrapper(FoundationPoseConfig(
        repo_root="external/FoundationPose",
        weights_dir="external/FoundationPose/weights",
        mesh_scale=0.001,
    ))
    fp.preload_mesh(mesh_path=mesh, object_id="probe")
    used = torch.cuda.memory_allocated() / 1e9
    print(f"  FP loaded. torch-allocated ~{used:.2f} GB\n")

    print("=== SAM stress WITH FP/nvdiffrast/warp resident ===")
    stress(n, "SAM+FP")

    print("\n=== run an FP register so nvdiffrast actually rasterizes, then stress SAM ===")
    try:
        rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        depth = np.full((480, 640), 0.5, dtype=np.float32)
        K = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=np.float32)
        mask = np.zeros((480, 640), dtype=bool)
        mask[200:280, 280:360] = True
        fp.estimate_pose(object_id="probe", mesh_path=mesh, rgb=rgb, depth=depth, K=K, mask=mask)
        print("  FP register ran.")
    except Exception as e:
        print(f"  FP register raised (ok for this probe): {type(e).__name__}: {e}")
    print()
    stress(n, "SAM+FP-after-render")


if __name__ == "__main__":
    main()
