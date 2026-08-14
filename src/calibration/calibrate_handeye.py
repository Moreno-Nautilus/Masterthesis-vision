"""Stage B of dual-arm hand-eye calibration: solve T_flange_cam from the
data Stage A (capture_handeye_data.py) already captured.

Never touches ROS/hardware -- purely offline. Loads whatever
sample_*.json files already exist under
outputs/calibration_debug/handeye/<cam_id>/ (see capture_handeye_data.py's
docstring for how they get there) and solves via one of two methods:

  --method direct (the original approach)
    Per-camera independent closed-form cv2.calibrateHandEye (AX=XB) --
    handeye_flange_cam_realsense.py's _solve_handeye, unchanged. Fast, no
    tuning knobs, no prior. Each arm solved completely independently of
    the other.

  --method joint (default -- the newer bundle-adjustment refinement)
    Jointly refines both arms' T_flange_cam against every checkerboard
    corner's reprojection error at once, via the joint_handeye_calib
    submodule, with two soft priors the closed-form solve has no way to
    use: the two arms share the same physical camera mount (pulled toward
    each other), and a CAD-derived nominal offset
    (config/camera_extrinsics_realsense.yaml's realsense_nominal entry,
    used as both initial guess and prior). See
    docs/joint_handeye_calibration.md for the full math/troubleshooting
    writeup -- still accurate, only the entry point moved.

Both write through io_extrinsics.update_extrinsics_yaml_preserving_header
into config/camera_extrinsics_realsense.yaml when --write is passed
(backs up the previous file to .yaml.bak first); without --write, results
are only printed.

Run (inside the 'vision' container):
    # Dry run, default (joint) method, both cameras:
    python3 -m src.calibration.calibrate_handeye -v

    # Looks sane? write it:
    python3 -m src.calibration.calibrate_handeye --write

    # Original closed-form solve instead:
    python3 -m src.calibration.calibrate_handeye --method direct --write

    # Just one camera:
    python3 -m src.calibration.calibrate_handeye --cam-ids realsense_2 --write
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from src.calibration.handeye_flange_cam_realsense import (
    CHESS_COLS,
    CHESS_ROWS,
    DEBUG_DIR,
    OUT_YAML,
    SQUARE_SIZE_M,
    HandEyeSample,
    _load_samples_from_dir,
    _solve_handeye,
)
from src.calibration.io_extrinsics import update_extrinsics_yaml_preserving_header
from src.utils.se3 import SE3

# The bundle-adjustment core lives in its own git submodule/repo (see
# src/calibration/joint_handeye_calib/README.md for why it's standalone) --
# add its root to sys.path rather than requiring `pip install -e` for a
# simple in-repo import. Only needed for --method joint, so imported lazily
# inside _solve_joint() rather than at module load time.
_SUBMODULE_ROOT = Path(__file__).resolve().parent / "joint_handeye_calib"

ARM_CAM_IDS = ["realsense_1", "realsense_2"]
DEFAULT_MIN_SAMPLES = 5


def _backup_and_write(results: dict[str, SE3]) -> None:
    out_path = Path(OUT_YAML)
    if out_path.exists():
        backup = out_path.with_suffix(".yaml.bak")
        backup.write_text(out_path.read_text())
        print(f"\nBacked up existing YAML to: {backup}")
    update_extrinsics_yaml_preserving_header(out_path, results)
    print(f"Wrote T_flange_cam for {sorted(results)} into: {out_path}")


def _solve_direct(cam_ids: list[str], debug_root: str, min_samples: int) -> dict[str, SE3]:
    results: dict[str, SE3] = {}
    for cam_id in cam_ids:
        debug_dir = Path(debug_root) / cam_id
        samples: list[HandEyeSample] = _load_samples_from_dir(debug_dir) if debug_dir.exists() else []
        print(f"[{cam_id}] {len(samples)} samples loaded from {debug_dir}")
        if len(samples) < min_samples:
            print(f"[{cam_id}] skipping -- need >= {min_samples} samples, have {len(samples)}")
            continue

        T_flange_cam, residuals_deg, residuals_m = _solve_handeye(samples)
        print(f"\n--- {cam_id} hand-eye result (direct/closed-form) ---")
        print(T_flange_cam)
        print(
            f"AX=XB residuals: rotation mean={residuals_deg.mean():.4f}deg "
            f"max={residuals_deg.max():.4f}deg | translation mean={residuals_m.mean():.6f}m "
            f"max={residuals_m.max():.6f}m"
        )
        results[cam_id] = T_flange_cam

    if not results:
        raise SystemExit(
            f"No camera had >= {min_samples} samples under {debug_root} -- run "
            f"capture_handeye_data.py first."
        )
    return results


def _solve_joint(cam_ids: list[str], debug_root: str, args: argparse.Namespace) -> dict[str, SE3]:
    if str(_SUBMODULE_ROOT) not in sys.path:
        sys.path.insert(0, str(_SUBMODULE_ROOT))
    from joint_handeye_calib import (  # noqa: E402
        ArmData,
        PriorWeights,
        load_samples_from_dir,
        make_checkerboard_object_points,
        nominal_flange_cam,
        solve_joint_handeye,
    )

    object_points = make_checkerboard_object_points(CHESS_COLS, CHESS_ROWS, SQUARE_SIZE_M)
    nominal = None if args.no_nominal else nominal_flange_cam()
    init_guess = nominal if nominal is not None else SE3.identity()

    arms = {}
    for cam_id in cam_ids:
        debug_dir = Path(debug_root) / cam_id
        samples = load_samples_from_dir(debug_dir) if debug_dir.exists() else []
        if not samples:
            print(f"[{cam_id}] no samples under {debug_dir} -- skipping")
            continue
        n_with_px = sum(s.has_reprojection_data() for s in samples)
        print(f"[{cam_id}] {len(samples)} samples loaded ({n_with_px} with pixel reprojection data)")
        arms[cam_id] = ArmData(samples=samples, object_points=object_points, initial_T_flange_cam=init_guess)

    if not arms:
        raise SystemExit(f"No samples found for any of {cam_ids} under {debug_root} -- run capture_handeye_data.py first.")
    if len(arms) == 1:
        print(f"Only one arm ({next(iter(arms))}) has data -- solving it alone (no shared-mount prior to apply).")

    weights = PriorWeights(
        shared_mount_sigma_r=np.deg2rad(args.shared_mount_sigma_deg),
        shared_mount_sigma_t=args.shared_mount_sigma_m,
        nominal_sigma_r=np.deg2rad(args.nominal_sigma_deg),
        nominal_sigma_t=args.nominal_sigma_m,
    )
    result = solve_joint_handeye(
        arms, nominal_T_flange_cam=nominal, weights=weights,
        loss=args.loss, f_scale=args.f_scale, verbose=min(args.verbose, 2),
    )

    print("\n=== Result (joint bundle adjustment) ===")
    for cam_id in arms:
        print(f"\n[{cam_id}] T_flange_cam:\n{result.T_flange_cam[cam_id]}")
        if result.reprojection_errors_px[cam_id].size:
            errs = result.reprojection_errors_px[cam_id]
            print(f"  reprojection error (px): mean={errs.mean():.3f} max={errs.max():.3f} n={errs.size}")
        if result.pose_errors_deg_m[cam_id].size:
            rot = result.pose_errors_deg_m[cam_id][:, 0]
            trans = result.pose_errors_deg_m[cam_id][:, 1]
            print(
                f"  pose residual (legacy samples): rot mean={rot.mean():.4f}deg max={rot.max():.4f}deg  "
                f"trans mean={trans.mean() * 1000:.3f}mm max={trans.max() * 1000:.3f}mm  n={trans.size}"
            )
        if cam_id in result.nominal_diff_deg_m:
            rot_deg, trans_m = result.nominal_diff_deg_m[cam_id]
            print(f"  vs. CAD nominal: rot={rot_deg:.3f}deg  trans={trans_m * 1000:.3f}mm")

    for (a, b), (rot_deg, trans_m) in result.pairwise_mount_diff_deg_m.items():
        print(f"\n[{a} vs {b}] flange-cam disagreement: rot={rot_deg:.3f}deg  trans={trans_m * 1000:.3f}mm")

    return result.T_flange_cam


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", choices=("direct", "joint"), default="joint")
    parser.add_argument("--cam-ids", nargs="+", default=ARM_CAM_IDS, help="which cam_ids to solve")
    parser.add_argument("--debug-root", default=DEBUG_DIR, help="parent dir of each cam_id's sample_*.json folder")
    parser.add_argument(
        "--write", action="store_true",
        help=f"write the solved T_flange_cam values into {OUT_YAML} (backing up the previous file first). "
             "Without this flag, results are only printed.",
    )
    parser.add_argument("-v", "--verbose", action="count", default=0)

    direct_group = parser.add_argument_group("--method direct")
    direct_group.add_argument("--min-samples", type=int, default=DEFAULT_MIN_SAMPLES)

    joint_group = parser.add_argument_group("--method joint")
    joint_group.add_argument("--no-nominal", action="store_true", help="disable the CAD nominal init/prior entirely")
    joint_group.add_argument("--shared-mount-sigma-deg", type=float, default=2.0)
    joint_group.add_argument("--shared-mount-sigma-m", type=float, default=0.01)
    joint_group.add_argument("--nominal-sigma-deg", type=float, default=5.0)
    joint_group.add_argument("--nominal-sigma-m", type=float, default=0.02)
    joint_group.add_argument(
        "--loss", default="linear", choices=["linear", "soft_l1", "cauchy", "huber", "arctan"],
        help="scipy least_squares loss. Defaults to plain linear -- robust losses can stall badly when "
             "the initial guess (CAD nominal) is several degrees off an arm's actual as-built mount; see "
             "docs/joint_handeye_calibration.md's Troubleshooting section before changing this.",
    )
    joint_group.add_argument("--f-scale", type=float, default=2.0)

    args = parser.parse_args()

    if args.method == "direct":
        results = _solve_direct(args.cam_ids, args.debug_root, args.min_samples)
    else:
        results = _solve_joint(args.cam_ids, args.debug_root, args)

    if not args.write:
        print(f"\n(dry run -- pass --write to update {OUT_YAML})")
        return

    _backup_and_write(results)


if __name__ == "__main__":
    main()
