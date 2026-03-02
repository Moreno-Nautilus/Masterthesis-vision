from __future__ import annotations

from pathlib import Path
import numpy as np
import yaml

from src.utils.se3 import SE3


def load_extrinsics_yaml(path: str | Path) -> dict[str, SE3]:
    """
    YAML format:
    cam_id:
      R: [r00,r01,r02, r10,r11,r12, r20,r21,r22]   # row-major
      t: [tx,ty,tz]
    """
    path = Path(path)
    data = yaml.safe_load(path.read_text())

    out: dict[str, SE3] = {}
    for cam_id, d in data.items():
        R = np.array(d["R"], dtype=float).reshape(3, 3)
        t = np.array(d["t"], dtype=float).reshape(3,)
        out[cam_id] = SE3(R, t)
    return out