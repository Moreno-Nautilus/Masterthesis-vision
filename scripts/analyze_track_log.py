#!/usr/bin/env python3
"""Compare track-mode pose logs across tracking backends.

Reads one or more CSVs written by run_pipeline_track_multicam's
``--log-track-poses`` (and the init_only baseline, same schema) and prints a
side-by-side summary geared at the two things that matter for finalising
tracking: **real-time stable lock** (tick rate, accept rate, loss/recovery,
jitter) and, secondarily, **pose quality** (chamfer / fitness / rmse).

Dependency-free (stdlib only) so it runs on the host where the CSVs land.
Pass ``--plots-dir DIR`` to also dump per-log chamfer / jump / XY-trajectory
PNGs when matplotlib happens to be importable.

Usage:
    python3 scripts/analyze_track_log.py \
        default=outputs/logs/track_default.csv \
        fast_cutie=outputs/logs/track_fastcutie.csv \
        projected_icp=outputs/logs/track_projicp.csv \
        init_ref=outputs/logs/multicam_init_final_baseline.csv

    # bare paths are allowed; the label defaults to the file stem
    python3 scripts/analyze_track_log.py outputs/logs/track_default.csv
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# pose_status values the pipeline emits, worst -> best.
_LOST_STATES = {"lost", "stale"}
_GOOD_STATES = {"fresh", "held"}


def _to_float(s: str) -> Optional[float]:
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    return v if math.isfinite(v) else None


def _to_bool(s: str) -> bool:
    return str(s).strip().lower() in ("true", "1", "yes")


def _percentile(values: list[float], q: float) -> Optional[float]:
    """Linear-interpolated percentile (q in [0,100]); no numpy."""
    xs = sorted(v for v in values if v is not None)
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    pos = (q / 100.0) * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def _median(values: list[float]) -> Optional[float]:
    return _percentile(values, 50.0)


@dataclass
class TrackStats:
    track_id: str
    object_id: str = ""
    n_rows: int = 0
    n_accepted: int = 0
    status_counts: Counter = field(default_factory=Counter)
    chamfer_mm: list[float] = field(default_factory=list)
    fitness: list[float] = field(default_factory=list)
    rmse_mm: list[float] = field(default_factory=list)
    dt_mm: list[float] = field(default_factory=list)      # accepted only
    drot_deg: list[float] = field(default_factory=list)   # accepted only
    n_loss_events: int = 0       # transitions into a lost/stale state
    n_recoveries: int = 0        # lost/stale -> good
    longest_lost_streak: int = 0
    _prev_status: Optional[str] = None
    _cur_lost_streak: int = 0
    # XY trajectory of accepted poses, for optional plotting.
    traj: list[tuple[int, float, float, float]] = field(default_factory=list)


@dataclass
class LogStats:
    label: str
    path: str
    n_rows: int = 0
    frame_stamps: dict[int, float] = field(default_factory=dict)  # frame -> stamp_s
    tracks: dict[str, TrackStats] = field(default_factory=dict)

    def tick_dts(self) -> list[float]:
        items = sorted(self.frame_stamps.items())
        dts = []
        for (_, t0), (_, t1) in zip(items, items[1:]):
            dt = t1 - t0
            if dt > 0:
                dts.append(dt)
        return dts


def load_log(label: str, path: Path) -> LogStats:
    log = LogStats(label=label, path=str(path))
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            log.n_rows += 1
            frame = row.get("frame")
            stamp = _to_float(row.get("stamp_s", ""))
            try:
                frame_i = int(frame)
            except (TypeError, ValueError):
                frame_i = log.n_rows
            if stamp is not None:
                # first stamp seen for a frame wins (rows of a frame share it)
                log.frame_stamps.setdefault(frame_i, stamp)

            tid = (row.get("track_id") or "").strip() or "?"
            ts = log.tracks.get(tid)
            if ts is None:
                ts = TrackStats(track_id=tid, object_id=(row.get("object_id") or "").strip())
                log.tracks[tid] = ts

            ts.n_rows += 1
            accepted = _to_bool(row.get("accepted", ""))
            if accepted:
                ts.n_accepted += 1

            status = (row.get("pose_status") or "").strip()
            ts.status_counts[status or "?"] += 1

            # loss / recovery state machine
            is_lost = status in _LOST_STATES
            was_lost = (ts._prev_status in _LOST_STATES) if ts._prev_status else False
            if is_lost and not was_lost:
                ts.n_loss_events += 1
            if (not is_lost) and was_lost and status in _GOOD_STATES:
                ts.n_recoveries += 1
            if is_lost:
                ts._cur_lost_streak += 1
                ts.longest_lost_streak = max(ts.longest_lost_streak, ts._cur_lost_streak)
            else:
                ts._cur_lost_streak = 0
            ts._prev_status = status

            for key, bucket in (
                ("chamfer_mm", ts.chamfer_mm),
                ("fitness", ts.fitness),
                ("rmse_mm", ts.rmse_mm),
            ):
                v = _to_float(row.get(key, ""))
                if v is not None:
                    bucket.append(v)

            if accepted:
                dt = _to_float(row.get("dt_mm", ""))
                if dt is not None:
                    ts.dt_mm.append(dt)
                dr = _to_float(row.get("drot_deg", ""))
                if dr is not None:
                    ts.drot_deg.append(dr)
                tx = _to_float(row.get("tx", ""))
                ty = _to_float(row.get("ty", ""))
                tz = _to_float(row.get("tz", ""))
                if None not in (tx, ty, tz):
                    ts.traj.append((frame_i, tx, ty, tz))

    return log


def _fmt(v: Optional[float], digits: int = 1) -> str:
    return "-" if v is None else f"{v:.{digits}f}"


def summarize_log(log: LogStats) -> dict[str, str]:
    """One aggregate column of metrics for a log (pooled across its tracks)."""
    dts = log.tick_dts()
    med_dt = _median(dts)
    p95_dt = _percentile(dts, 95.0)
    hz = (1.0 / med_dt) if med_dt and med_dt > 0 else None

    n_rows = sum(t.n_rows for t in log.tracks.values())
    n_acc = sum(t.n_accepted for t in log.tracks.values())
    accept_rate = (100.0 * n_acc / n_rows) if n_rows else None

    chamfer = [v for t in log.tracks.values() for v in t.chamfer_mm]
    fitness = [v for t in log.tracks.values() for v in t.fitness]
    rmse = [v for t in log.tracks.values() for v in t.rmse_mm]
    dt_mm = [v for t in log.tracks.values() for v in t.dt_mm]
    drot = [v for t in log.tracks.values() for v in t.drot_deg]

    n_loss = sum(t.n_loss_events for t in log.tracks.values())
    n_rec = sum(t.n_recoveries for t in log.tracks.values())
    worst_streak = max((t.longest_lost_streak for t in log.tracks.values()), default=0)
    duration = (max(log.frame_stamps.values()) - min(log.frame_stamps.values())
                if log.frame_stamps else 0.0)

    return {
        "tracks": str(len(log.tracks)),
        "ticks": str(len(log.frame_stamps)),
        "duration_s": _fmt(duration, 1),
        "tick_hz (med)": _fmt(hz, 1),
        "tick_dt_ms med/p95": f"{_fmt((med_dt or 0)*1000,0)}/{_fmt((p95_dt or 0)*1000,0)}",
        "accept_rate_%": _fmt(accept_rate, 1),
        "loss_events": str(n_loss),
        "recoveries": str(n_rec),
        "worst_lost_streak": str(worst_streak),
        "chamfer_mm med/p95": f"{_fmt(_median(chamfer))}/{_fmt(_percentile(chamfer,95))}",
        "trans_jump_mm med/p95": f"{_fmt(_median(dt_mm))}/{_fmt(_percentile(dt_mm,95))}",
        "rot_jump_deg med/p95": f"{_fmt(_median(drot))}/{_fmt(_percentile(drot,95))}",
        "fitness med": _fmt(_median(fitness), 3),
        "rmse_mm med": _fmt(_median(rmse), 1),
    }


# Row order for the comparison table.
_METRIC_ORDER = [
    "tracks", "ticks", "duration_s",
    "tick_hz (med)", "tick_dt_ms med/p95",
    "accept_rate_%", "loss_events", "recoveries", "worst_lost_streak",
    "chamfer_mm med/p95", "trans_jump_mm med/p95", "rot_jump_deg med/p95",
    "fitness med", "rmse_mm med",
]


def print_comparison(logs: list[LogStats]) -> None:
    cols = {log.label: summarize_log(log) for log in logs}
    labels = [log.label for log in logs]
    metric_w = max(len(m) for m in _METRIC_ORDER)
    col_w = max(14, *(len(l) for l in labels))

    header = "metric".ljust(metric_w) + "  " + "  ".join(l.ljust(col_w) for l in labels)
    print("\n" + header)
    print("-" * len(header))
    for metric in _METRIC_ORDER:
        line = metric.ljust(metric_w) + "  " + "  ".join(
            cols[l].get(metric, "-").ljust(col_w) for l in labels
        )
        print(line)
    print()
    print("Legend: tick_hz = effective pipeline rate (real-time lock). "
          "trans/rot_jump = per-frame pose change on accepted frames (jitter; "
          "lower = steadier lock). chamfer = model-to-observation fit (quality "
          "proxy vs the init baseline). loss_events/worst_lost_streak gauge "
          "robustness.")


def _track_motion_stats(traj: list[tuple[int, float, float, float]]) -> dict[str, float]:
    """Span (bbox diagonal), net first->last displacement, and path length of a
    track's accepted positions, all in mm."""
    if len(traj) < 2:
        return {"span_mm": 0.0, "net_mm": 0.0, "path_mm": 0.0}
    xs = [p[1] for p in traj]
    ys = [p[2] for p in traj]
    zs = [p[3] for p in traj]
    span = math.sqrt((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2
                     + (max(zs) - min(zs)) ** 2)
    net = math.sqrt((xs[-1] - xs[0]) ** 2 + (ys[-1] - ys[0]) ** 2
                    + (zs[-1] - zs[0]) ** 2)
    path = 0.0
    for a, b in zip(traj, traj[1:]):
        path += math.sqrt((b[1] - a[1]) ** 2 + (b[2] - a[2]) ** 2
                          + (b[3] - a[3]) ** 2)
    return {"span_mm": span * 1000, "net_mm": net * 1000, "path_mm": path * 1000}


def print_per_track_motion(logs: list[LogStats]) -> None:
    """Per-track motion summary — exposes frozen/stuck trackers that the pooled
    summary table can hide (high accept rate + 'fresh' status but ~zero
    displacement). The CSV alone can't tell a genuinely static object from a
    stuck tracker, so a low-span track is flagged for the human to judge."""
    print("\nPer-track motion (accepted poses):")
    hdr = (f"  {'log':14s} {'track_id':22s} {'n_acc':>5s} {'%fresh':>6s} "
           f"{'span_mm':>8s} {'net_mm':>8s} {'path_mm':>8s}  note")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for log in logs:
        for tid, t in sorted(log.tracks.items()):
            m = _track_motion_stats(t.traj)
            n_fresh = t.status_counts.get("fresh", 0)
            pct_fresh = (100.0 * n_fresh / t.n_rows) if t.n_rows else 0.0
            note = ""
            if t.n_accepted >= 20 and m["span_mm"] < 15.0:
                note = "near-static / possibly stuck"
            print(f"  {log.label:14s} {tid:22s} {t.n_accepted:5d} {pct_fresh:5.0f}% "
                  f"{m['span_mm']:8.1f} {m['net_mm']:8.1f} {m['path_mm']:8.1f}  {note}")
    print("\n  span_mm = bbox diagonal of accepted positions; net_mm = first->last; "
          "path_mm = summed step length.")
    print("  A track you physically moved that shows span_mm < ~15mm is a stuck "
          "tracker, not a steady one.")


def maybe_plot(logs: list[LogStats], plots_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"[plots] matplotlib unavailable, skipping: {e}")
        return

    plots_dir.mkdir(parents=True, exist_ok=True)

    # chamfer-over-frame, one line per (log, track)
    fig, ax = plt.subplots(figsize=(10, 5))
    for log in logs:
        for t in log.tracks.values():
            if t.chamfer_mm:
                ax.plot(range(len(t.chamfer_mm)), t.chamfer_mm,
                        label=f"{log.label}:{t.track_id}", alpha=0.7, linewidth=1)
    ax.set_xlabel("accepted-frame index"); ax.set_ylabel("chamfer (mm)")
    ax.set_title("Pose quality over time"); ax.legend(fontsize=7)
    fig.savefig(plots_dir / "chamfer_over_time.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # per-frame jump (jitter), one line per (log, track)
    fig, ax = plt.subplots(figsize=(10, 5))
    for log in logs:
        for t in log.tracks.values():
            if t.dt_mm:
                ax.plot(range(len(t.dt_mm)), t.dt_mm,
                        label=f"{log.label}:{t.track_id}", alpha=0.7, linewidth=1)
    ax.set_xlabel("accepted-frame index"); ax.set_ylabel("translation jump (mm)")
    ax.set_title("Per-frame pose jump (lower = steadier lock)"); ax.legend(fontsize=7)
    fig.savefig(plots_dir / "trans_jump_over_time.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # XY trajectory
    fig, ax = plt.subplots(figsize=(7, 7))
    for log in logs:
        for t in log.tracks.values():
            if t.traj:
                xs = [p[1] for p in t.traj]
                ys = [p[2] for p in t.traj]
                ax.plot(xs, ys, label=f"{log.label}:{t.track_id}", alpha=0.7,
                        marker=".", markersize=2, linewidth=1)
    ax.set_xlabel("base X (m)"); ax.set_ylabel("base Y (m)")
    ax.set_title("Tracked trajectory (base frame, top-down)")
    ax.set_aspect("equal", adjustable="datalim"); ax.legend(fontsize=7)
    fig.savefig(plots_dir / "trajectory_xy.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"[plots] wrote chamfer_over_time.png, trans_jump_over_time.png, "
          f"trajectory_xy.png -> {plots_dir}")


def parse_spec(spec: str) -> tuple[str, Path]:
    """`label=path` or bare `path` (label defaults to file stem)."""
    if "=" in spec:
        label, path = spec.split("=", 1)
        if label and path:
            return label, Path(path)
    p = Path(spec)
    return p.stem, p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+",
                    help="One or more `label=path.csv` (or bare path.csv) track logs.")
    ap.add_argument("--plots-dir", default=None,
                    help="If set, write comparison PNGs here (needs matplotlib).")
    args = ap.parse_args()

    loaded: list[LogStats] = []
    for spec in args.logs:
        label, path = parse_spec(spec)
        if not path.is_file():
            ap.error(f"not a file: {path}")
        loaded.append(load_log(label, path))

    print_comparison(loaded)
    print_per_track_motion(loaded)

    if args.plots_dir:
        maybe_plot(loaded, Path(args.plots_dir))


if __name__ == "__main__":
    main()
