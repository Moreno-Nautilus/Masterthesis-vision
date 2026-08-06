"""Joint hand-eye calibration for both wrist RealSense cameras (Stage 1 replacement).

Replaces the per-arm closed-form cv2.calibrateHandEye solve in
handeye_flange_cam_realsense.py with a reprojection-error bundle adjustment (see
src/calibration/joint_handeye_calib/), run over BOTH arms at once. Compared to the
per-arm closed-form solve, this:

  - uses every checkerboard corner's reprojection error, not just the solved
    per-sample board pose, when a sample has corners_px+K (any capture from the
    updated handeye_flange_cam_realsense.py qualifies; older captures fall back to a
    pose-level residual automatically -- no data loss, no recapture needed);
  - ties both arms' T_flange_cam together with a soft prior (they share the same
    physical camera mount, see joint_handeye_calib/problem.py), so a well-conditioned
    arm's data helps constrain a less-well-conditioned one;
  - uses the CAD-derived nominal offset (joint_handeye_calib/nominal.py) as both the
    initial guess and a second soft prior, instead of requiring calibrateHandEye's
    default (identity) initialization.

Works fine with only ONE arm's data present (e.g. currently realsense_1 has no
captured samples yet, only realsense_2 does) -- it just solves that one arm with the
CAD-nominal prior only; the cross-arm prior appears automatically once both arms have
data.

Run (inside the 'vision' container, matching handeye_flange_cam_realsense.py):
    python3 -m src.calibration.joint_calibrate_dual_realsense --write
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# The bundle-adjustment core lives in its own git submodule/repo (see
# src/calibration/joint_handeye_calib/README.md for why it's standalone) -- add its
# root to sys.path rather than requiring `pip install -e` for a simple in-repo import.
_SUBMODULE_ROOT = Path(__file__).resolve().parent / "joint_handeye_calib"
if str(_SUBMODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SUBMODULE_ROOT))

from joint_handeye_calib import (  # noqa: E402
    ArmData,
    PriorWeights,
    SE3,
    load_samples_from_dir,
    make_checkerboard_object_points,
    nominal_flange_cam,
    solve_joint_handeye,
)

from src.calibration.handeye_flange_cam_realsense import (
    CHESS_COLS,
    CHESS_ROWS,
    DEBUG_DIR,
    OUT_YAML,
    SQUARE_SIZE_M,
)
from src.calibration.io_extrinsics import update_extrinsics_yaml_preserving_header

ARM_CAM_IDS = ["realsense_1", "realsense_2"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cam-ids", nargs="+", default=ARM_CAM_IDS, help="which cam_ids to include")
    parser.add_argument("--debug-root", default=DEBUG_DIR, help="parent dir of each cam_id's sample_*.json folder")
    parser.add_argument("--no-nominal", action="store_true", help="disable the CAD nominal init/prior entirely")
    parser.add_argument("--shared-mount-sigma-deg", type=float, default=2.0)
    parser.add_argument("--shared-mount-sigma-m", type=float, default=0.01)
    parser.add_argument("--nominal-sigma-deg", type=float, default=5.0)
    parser.add_argument("--nominal-sigma-m", type=float, default=0.02)
    parser.add_argument(
        "--loss", default="linear", choices=["linear", "soft_l1", "cauchy", "huber", "arctan"],
        help="scipy least_squares loss. Defaults to plain linear -- robust losses can stall badly when "
             "the initial guess (CAD nominal) is several degrees off an arm's actual as-built mount; see "
             "joint_handeye_calib.problem.solve_joint_handeye's docstring. Verify convergence (printed "
             "pose/reprojection error should be sub-degree/sub-mm-ish, not tens of degrees) before "
             "trusting a different --loss.",
    )
    parser.add_argument("--f-scale", type=float, default=2.0)
    parser.add_argument(
        "--write", action="store_true",
        help=f"write the solved T_flange_cam values into {OUT_YAML} (backing up the previous file first). "
             "Without this flag, results are only printed.",
    )
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args()

    object_points = make_checkerboard_object_points(CHESS_COLS, CHESS_ROWS, SQUARE_SIZE_M)
    nominal = None if args.no_nominal else nominal_flange_cam()
    init_guess = nominal if nominal is not None else SE3.identity()

    arms = {}
    for cam_id in args.cam_ids:
        debug_dir = Path(args.debug_root) / cam_id
        samples = load_samples_from_dir(debug_dir) if debug_dir.exists() else []
        if not samples:
            print(f"[{cam_id}] no samples under {debug_dir} -- skipping")
            continue
        n_with_px = sum(s.has_reprojection_data() for s in samples)
        print(f"[{cam_id}] {len(samples)} samples loaded ({n_with_px} with pixel reprojection data)")
        arms[cam_id] = ArmData(samples=samples, object_points=object_points, initial_T_flange_cam=init_guess)

    if not arms:
        raise SystemExit(f"No samples found for any of {args.cam_ids} under {args.debug_root}. Capture some first.")
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

    print("\n=== Result ===")
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

    if not args.write:
        print("\n(dry run -- pass --write to update", OUT_YAML, ")")
        return

    out_path = Path(OUT_YAML)
    if out_path.exists():
        backup = out_path.with_suffix(".yaml.bak")
        backup.write_text(out_path.read_text())
        print(f"\nBacked up existing YAML to: {backup}")

    update_extrinsics_yaml_preserving_header(out_path, result.T_flange_cam)
    print(f"Wrote T_flange_cam for {list(arms)} into: {out_path}")


if __name__ == "__main__":
    main()
