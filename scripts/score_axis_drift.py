#!/usr/bin/env python3
"""Score tracked shaft-axis drift from the (trusted) init pose.

The thin symmetric screws' shaft-axis DIRECTION is only weakly observable from
depth, so the fused ICP can walk the orientation away from init during tracking
(see memory rotation-axis-observability). The init pose (grid+chamfer, ~2-5mm
translation, ~11-13deg axis floor) is the only clean rotation reference, so we
measure how far each TRACKED frame's shaft axis deviates from the INIT axis.

This is the validation tool for --fused-track-rot-slew-limit-deg /
--fused-track-rot-lowpass (rotation damping): run BASELINE and DAMPED, then

    python3 scripts/score_axis_drift.py \
        baseline=outputs/logs/live_baseline_q.csv \
        damped=outputs/logs/live_damped_q.csv

GOTCHAS baked in:
  * base-frame Z is NOT vertical, so we do NOT measure tilt from [0,0,1]; we
    measure deviation from the init shaft axis (per object, per run).
  * init_pose_log.csv is append-only across sessions and screw placement differs
    between sessions, so init rows are matched to each track run BY TIMESTAMP
    (--init-window-s before the track start). Check the printed init timestamps.
  * the model shaft axis = the object mesh's principal (PCA) axis (~object +Z);
    its arbitrary sign cancels because the same vector is used for init & track.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation as SciRot


# ── small helpers ────────────────────────────────────────────────────────────
def _to_float(s: str) -> Optional[float]:
    try:
        v = float(s)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _percentile(values: list[float], q: float) -> Optional[float]:
    return float(np.percentile(values, q)) if values else None


def _median(values: list[float]) -> Optional[float]:
    return float(np.median(values)) if values else None


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


def _axis_dev_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Deviation between two axis DIRECTIONS, folded to [0, 90] (a line, not a
    ray) so a benign end-for-end flip of a symmetric screw isn't penalised."""
    c = abs(float(np.clip(np.dot(_unit(a), _unit(b)), -1.0, 1.0)))
    return float(np.degrees(np.arccos(c)))


# ── mesh shaft axis (object local) ───────────────────────────────────────────
def mesh_shaft_axis(obj_path: Path) -> np.ndarray:
    """Principal axis of the mesh vertices (~object +Z). Lightweight .obj 'v'
    parser so this needs no trimesh/open3d."""
    pts = []
    with obj_path.open() as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                pts.append((float(parts[1]), float(parts[2]), float(parts[3])))
    if len(pts) < 3:
        raise ValueError(f"{obj_path}: too few vertices for PCA")
    P = np.asarray(pts, dtype=np.float64)
    P = P - P.mean(axis=0, keepdims=True)
    # principal direction = right-singular vector of largest singular value
    _, _, vt = np.linalg.svd(P, full_matrices=False)
    return _unit(vt[0])


# ── init reference axes ──────────────────────────────────────────────────────
def load_init_axes(
    init_log: Path,
    shaft_obj: dict[str, np.ndarray],
    t_start: float,
    t_end: float,
    window_s: float,
) -> dict[str, dict]:
    """Per object: mean init shaft axis (base frame) from the cameras' init
    poses whose timestamp falls in [t_start - window_s, t_end]. Also returns the
    cross-camera spread (the irreducible init floor) and the rows used."""
    rows_by_obj: dict[str, list[dict]] = {}
    with init_log.open() as f:
        for r in csv.DictReader(f):
            obj = r.get("obj_id", "")
            ts = _to_float(r.get("timestamp", ""))
            if obj not in shaft_obj or ts is None:
                continue
            if not (t_start - window_s <= ts <= t_end):
                continue
            rows_by_obj.setdefault(obj, []).append({
                "ts": ts,
                "cam_id": r.get("cam_id", ""),
                "rpy": [_to_float(r.get(k, "")) for k in ("roll", "pitch", "yaw")],
                "accepted": str(r.get("accepted", "")).strip().lower() == "true",
            })

    out: dict[str, dict] = {}
    for obj, rows in rows_by_obj.items():
        accepted = [r for r in rows if r["accepted"]] or rows  # fall back to all
        # latest row per camera (re-init within the window keeps the freshest)
        latest: dict[str, dict] = {}
        for r in sorted(accepted, key=lambda r: r["ts"]):
            if None in r["rpy"]:
                continue
            latest[r["cam_id"]] = r
        axes = []
        for r in latest.values():
            R = SciRot.from_euler("xyz", r["rpy"], degrees=True).as_matrix()
            axes.append(_unit(R @ shaft_obj[obj]))
        if not axes:
            continue
        ref = _unit(axes[0])
        aligned = [a if np.dot(a, ref) >= 0 else -a for a in axes]  # sign-align
        mean_axis = _unit(np.mean(aligned, axis=0))
        spread = [_axis_dev_deg(a, mean_axis) for a in aligned]
        out[obj] = {
            "axis": mean_axis,
            "n_cams": len(latest),
            "cross_cam_p90_deg": _percentile(spread, 90) or 0.0,
            "cross_cam_max_deg": max(spread) if spread else 0.0,
            "ts": sorted(r["ts"] for r in latest.values()),
        }
    return out


# ── track CSV scoring ────────────────────────────────────────────────────────
def score_track(
    label: str,
    path: Path,
    shaft_obj: dict[str, np.ndarray],
    init_window_s: float,
    init_log: Path,
    threshold_deg: float,
) -> dict:
    rows = []
    with path.open() as f:
        for r in csv.DictReader(f):
            rows.append(r)
    if not rows:
        raise ValueError(f"{path}: empty")

    stamps = [s for s in (_to_float(r.get("stamp_s", "")) for r in rows) if s is not None]
    t_start, t_end = (min(stamps), max(stamps)) if stamps else (0.0, 0.0)
    init_axes = load_init_axes(init_log, shaft_obj, t_start, t_end, init_window_s)

    # per (object, track): list of folded axis deviations from init
    per_track: dict[tuple[str, str], list[float]] = {}
    skipped_no_quat = 0
    skipped_no_init = 0
    for r in rows:
        obj = r.get("object_id", "")
        if obj not in shaft_obj:
            continue
        if obj not in init_axes:
            skipped_no_init += 1
            continue
        q = [_to_float(r.get(k, "")) for k in ("qx", "qy", "qz", "qw")]
        if None in q or abs(np.linalg.norm(q)) < 1e-6:
            skipped_no_quat += 1
            continue
        R = SciRot.from_quat(q).as_matrix()  # xyzw
        shaft_base = R @ shaft_obj[obj]
        dev = _axis_dev_deg(shaft_base, init_axes[obj]["axis"])
        per_track.setdefault((obj, r.get("track_id", "")), []).append(dev)

    return {
        "label": label,
        "path": path,
        "init_axes": init_axes,
        "per_track": per_track,
        "skipped_no_quat": skipped_no_quat,
        "skipped_no_init": skipped_no_init,
        "threshold_deg": threshold_deg,
    }


def _fmt(v: Optional[float]) -> str:
    return f"{v:6.1f}" if v is not None else "   n/a"


def print_report(results: list[dict]) -> None:
    thr = results[0]["threshold_deg"]

    for res in results:
        print(f"\n=== init reference ({res['label']}: {res['path'].name}) ===")
        if not res["init_axes"]:
            print("  !! NO init rows matched this run's time window — "
                  "check --init-window-s / --init-log (results below are empty).")
        for obj, info in sorted(res["init_axes"].items()):
            ts = info["ts"]
            span = f"{ts[0]:.0f}..{ts[-1]:.0f}" if ts else "n/a"
            print(f"  {obj:<14s} cams={info['n_cams']}  "
                  f"cross-cam spread p90={info['cross_cam_p90_deg']:.1f}deg "
                  f"max={info['cross_cam_max_deg']:.1f}deg  init_ts={span}")

    # per-track table
    header = (f"{'label':<10s} {'object':<14s} {'track':<22s} "
              f"{'n':>5s} {'med':>7s} {'p90':>7s} {'<' + str(int(thr)) + 'deg':>8s}")
    print("\n" + header)
    print("-" * len(header))
    for res in results:
        for (obj, track), devs in sorted(res["per_track"].items()):
            med = _median(devs)
            p90 = _percentile(devs, 90)
            frac = 100.0 * sum(d <= thr for d in devs) / len(devs) if devs else None
            frac_s = f"{frac:6.1f}%" if frac is not None else "   n/a"
            print(f"{res['label']:<10s} {obj:<14s} {track:<22s} "
                  f"{len(devs):>5d} {_fmt(med)} {_fmt(p90)} {frac_s:>8s}")
        skips = []
        if res["skipped_no_quat"]:
            skips.append(f"{res['skipped_no_quat']} no-quat (held/lost)")
        if res["skipped_no_init"]:
            skips.append(f"{res['skipped_no_init']} no-init-ref")
        if skips:
            print(f"{res['label']:<10s} (skipped: {', '.join(skips)})")

    # per-object baseline-vs-rest comparison (pooled over tracks of that object)
    objs = sorted({o for res in results for (o, _) in res["per_track"]})
    print("\n=== pooled per object (all tracks) ===")
    head = (f"{'object':<14s} {'label':<10s} "
            f"{'n':>6s} {'med':>7s} {'p90':>7s} {'<' + str(int(thr)) + 'deg':>8s}")
    print(head)
    print("-" * len(head))
    for obj in objs:
        for res in results:
            pooled = [d for (o, _), ds in res["per_track"].items() if o == obj for d in ds]
            if not pooled:
                continue
            frac = 100.0 * sum(d <= thr for d in pooled) / len(pooled)
            print(f"{obj:<14s} {res['label']:<10s} "
                  f"{len(pooled):>6d} {_fmt(_median(pooled))} "
                  f"{_fmt(_percentile(pooled, 90))} {frac:6.1f}%")
        print()


def parse_spec(spec: str) -> tuple[str, Path]:
    if "=" in spec:
        label, path = spec.split("=", 1)
        return label, Path(path)
    p = Path(spec)
    return p.stem, p


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("specs", nargs="+",
                    help="One or more `label=track_pose.csv` (or bare path) logs.")
    ap.add_argument("--init-log", type=Path, default=Path("init_pose_log.csv"),
                    help="init pose log (default: ./init_pose_log.csv).")
    ap.add_argument("--cad-dir", type=Path, default=Path("Data/CAD_Models_centered"),
                    help="dir with <object>.obj meshes for the shaft-axis PCA.")
    ap.add_argument("--init-window-s", type=float, default=900.0,
                    help="match init rows within this many seconds BEFORE each "
                         "track run's start (init is append-only across sessions).")
    ap.add_argument("--threshold-deg", type=float, default=20.0,
                    help="acceptance threshold for the within-N-deg fraction.")
    args = ap.parse_args()

    if not args.init_log.exists():
        sys.exit(f"init log not found: {args.init_log} "
                 f"(re-run the tracker with --log-init-poses)")

    # shaft axis per object that appears in any track CSV
    specs = [parse_spec(s) for s in args.specs]
    needed = set()
    for _, path in specs:
        with path.open() as f:
            for r in csv.DictReader(f):
                if r.get("object_id"):
                    needed.add(r["object_id"])
    shaft_obj: dict[str, np.ndarray] = {}
    for obj in sorted(needed):
        obj_path = args.cad_dir / f"{obj}.obj"
        if obj_path.exists():
            shaft_obj[obj] = mesh_shaft_axis(obj_path)
        else:
            print(f"  [WARN] no mesh for '{obj}' at {obj_path}; skipping it.")

    results = [
        score_track(label, path, shaft_obj, args.init_window_s,
                    args.init_log, args.threshold_deg)
        for label, path in specs
    ]
    print_report(results)


if __name__ == "__main__":
    main()
