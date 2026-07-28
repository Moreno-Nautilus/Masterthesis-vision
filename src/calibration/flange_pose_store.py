"""Permanent on-disk storage for the flange poses captured by
capture_flange_poses_dual.py and consumed by autocalibrate_dual_realsense.py.

Separated from either script so both can import the same schema without a
circular dependency: capture writes, autocalibrate reads (and both scripts'
--help/docstrings point here for the file format).

One JSON file per arm (config/flange_poses/<arm_key>.json), written
incrementally -- each captured pose is appended and the whole file rewritten
immediately (see save_pose_set below), the same "never lose a sample to a
later crash" rationale as HandEyeSample/BoardPoseSample's per-sample JSON in
handeye_flange_cam_realsense.py / board_pose_from_flange_realsense.py. Unlike
those (debug/scratch dirs under outputs/), this is permanent, versioned
config data -- config/flange_poses/ is meant to be committed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.utils.se3 import SE3

FLANGE_POSES_DIR = Path("config/flange_poses")

# Every entry in ARM_KEYS must have a matching robot_bases.yaml key (see
# src/utils/robot_bases.py) so a captured pose can be related to the other
# arm's frame if ever needed -- ARM_KEYS["left"]["robot_base_key"] ==
# "robot_a", mirroring DynamicCameraTopics.robot_base_key in
# multicam_grabber_realsense.py / run_pipeline_track_multicam_realsense.py.
ARM_KEYS = {
    "left": {
        "cam_id": "realsense_1",
        "base_frame": "lbr_one_link_0",
        "flange_frame": "lbr_one_link_ee",
        "flange_pose_topic": "/left/ee_pose",
        "group_name": "arm_one",
        "robot_base_key": "robot_a",
    },
    "right": {
        "cam_id": "realsense_2",
        "base_frame": "lbr_two_link_0",
        "flange_frame": "lbr_two_link_ee",
        "flange_pose_topic": "/right/ee_pose",
        "group_name": "arm_two",
        "robot_base_key": "robot_b",
    },
}


@dataclass
class FlangePoseCapture:
    idx: int
    T_armBase_flange: SE3
    captured_at_unix_s: float
    note: str = ""


@dataclass
class FlangePoseSet:
    arm_key: str
    cam_id: str
    base_frame: str
    flange_frame: str
    captures: list[FlangePoseCapture] = field(default_factory=list)


def _pose_set_path(arm_key: str) -> Path:
    if arm_key not in ARM_KEYS:
        raise ValueError(f"Unknown arm_key={arm_key!r}, expected one of {sorted(ARM_KEYS)}")
    return FLANGE_POSES_DIR / f"{arm_key}.json"


def _capture_to_dict(c: FlangePoseCapture) -> dict:
    return {
        "idx": c.idx,
        "captured_at_unix_s": c.captured_at_unix_s,
        "note": c.note,
        "T_armBase_flange": {
            "R": c.T_armBase_flange.R.tolist(),
            "t": c.T_armBase_flange.t.tolist(),
        },
    }


def _capture_from_dict(d: dict) -> FlangePoseCapture:
    return FlangePoseCapture(
        idx=d["idx"],
        captured_at_unix_s=d["captured_at_unix_s"],
        note=d.get("note", ""),
        T_armBase_flange=SE3(
            np.array(d["T_armBase_flange"]["R"]), np.array(d["T_armBase_flange"]["t"])
        ),
    )


def save_pose_set(pose_set: FlangePoseSet) -> Path:
    """Overwrites config/flange_poses/<arm_key>.json with the full current set.

    Called after every single capture (not just at the end of a run) so a
    crash mid-session loses at most the in-memory state of the CURRENT
    capture, never previously accepted ones.
    """
    path = _pose_set_path(pose_set.arm_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "arm_key": pose_set.arm_key,
        "cam_id": pose_set.cam_id,
        "base_frame": pose_set.base_frame,
        "flange_frame": pose_set.flange_frame,
        "captures": [_capture_to_dict(c) for c in pose_set.captures],
    }
    path.write_text(json.dumps(data, indent=2))
    return path


def load_pose_set(arm_key: str) -> FlangePoseSet:
    path = _pose_set_path(arm_key)
    if not path.exists():
        raise FileNotFoundError(
            f"No saved flange poses for arm_key={arm_key!r} at {path} -- "
            f"run capture_flange_poses_dual.py first."
        )
    data = json.loads(path.read_text())
    return FlangePoseSet(
        arm_key=data["arm_key"],
        cam_id=data["cam_id"],
        base_frame=data["base_frame"],
        flange_frame=data["flange_frame"],
        captures=[_capture_from_dict(c) for c in data["captures"]],
    )
