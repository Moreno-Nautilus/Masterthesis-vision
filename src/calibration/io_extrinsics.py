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

def save_extrinsics_yaml(path: str | Path, extr: dict[str, SE3]) -> None:
    """
    Writes YAML format:
    cam_id:
      R: [r00,r01,r02, r10,r11,r12, r20,r21,r22]   # row-major
      t: [tx,ty,tz]
    """
    path = Path(path)
    data = {}
    for cam_id, T in extr.items():
        R = np.asarray(T.R, dtype=float).reshape(3, 3)
        t = np.asarray(T.t, dtype=float).reshape(3,)
        data[cam_id] = {
            "R": R.reshape(-1).tolist(),
            "t": t.tolist(),
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=True))