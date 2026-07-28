"""Append-only JSON run logs for the dual-arm auto-calibration routine
(src/calibration/autocalibrate_dual_realsense.py).

Three logs, one per calibrated entity, each keyed by cam_id/entity and
appended to (not overwritten) on every run so calibration history and
quality metrics accumulate over time instead of only keeping the latest
result (config/camera_extrinsics_realsense.yaml / config/base_board_pose.yaml
themselves still only hold the latest value -- these logs are the history +
QA trail alongside them):

  outputs/calibration_logs/camera_transforms.json       -- T_flange_cam per
      RealSense (hand-eye) + T_base_cam for the ZED, one entry per run.
  outputs/calibration_logs/checkerboard_transforms.json -- T_base_board per
      run (Stage 2 of the RealSense routine).
  outputs/calibration_logs/flange_transforms.json       -- which saved flange
      poses (config/flange_poses/<arm>.json indices) were used as the
      hand-eye vs. board-pose split for a given run, so a later audit can
      tell exactly which physical robot motions produced a given result.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

LOG_DIR = Path("outputs/calibration_logs")
CAMERA_LOG = LOG_DIR / "camera_transforms.json"
CHECKERBOARD_LOG = LOG_DIR / "checkerboard_transforms.json"
FLANGE_LOG = LOG_DIR / "flange_transforms.json"


def _append_entry(path: Path, entry: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    if path.exists():
        entries = json.loads(path.read_text())
    entry = dict(entry)
    entry.setdefault("logged_at_unix_s", time.time())
    entries.append(entry)
    path.write_text(json.dumps(entries, indent=2))


def log_camera_transform(entry: dict[str, Any]) -> None:
    """entry should identify cam_id, the solved R/t, and quality metrics
    (e.g. ax_xb_residual_rot_deg_mean/max, ax_xb_residual_t_m_mean/max for a
    RealSense hand-eye solve; reproj_err_px_mean, translation_std_m,
    rotation_std_deg for the ZED static solve)."""
    _append_entry(CAMERA_LOG, entry)


def log_checkerboard_transform(entry: dict[str, Any]) -> None:
    """entry should identify active_robot, the solved T_base_board R/t, and
    quality metrics (translation_std_m, rotation_std_deg, reproj_err_px_mean)."""
    _append_entry(CHECKERBOARD_LOG, entry)


def log_flange_transform_usage(entry: dict[str, Any]) -> None:
    """entry should identify arm_key, which saved capture indices were
    consumed for hand-eye vs. board-pose, and each capture's own T_armBase_flange
    (so this log is self-contained even if config/flange_poses/<arm>.json is
    later overwritten by a fresh capture session)."""
    _append_entry(FLANGE_LOG, entry)
