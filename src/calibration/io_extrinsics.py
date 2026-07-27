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

    # Rebuild one SE3 per camera from its flat row-major R + translation t.
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
    # Flatten each camera's SE3 into the row-major R + t YAML layout.
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


def update_extrinsics_yaml_preserving_header(
    path: str | Path,
    updates: dict[str, SE3],
) -> None:
    """
    Rewrites only the given cam_id block(s) of an existing extrinsics YAML,
    leaving every other cam_id entry and the file's leading '#' comment
    header untouched. Use this instead of save_extrinsics_yaml() whenever the
    target file mixes cam_ids with different owners/semantics (e.g.
    config/camera_extrinsics_realsense.yaml, where zed2i_1 is a normal
    static entry but realsense_1/realsense_2 are camera-to-flange offsets) --
    save_extrinsics_yaml() would silently drop the untouched entries and any
    inline documentation comments.

    Any cam_id in `updates` that doesn't already exist in the file is
    appended as a new block.
    """
    path = Path(path)

    header_lines: list[str] = []
    for line in path.read_text().splitlines(keepends=True):
        stripped = line.strip()
        if stripped == "" or stripped.startswith("#"):
            header_lines.append(line)
        else:
            break
    header = "".join(header_lines)

    existing = load_extrinsics_yaml(path) if path.exists() else {}
    merged = dict(existing)
    merged.update(updates)

    data = {}
    for cam_id, T in merged.items():
        R = np.asarray(T.R, dtype=float).reshape(3, 3)
        t = np.asarray(T.t, dtype=float).reshape(3,)
        data[cam_id] = {
            "R": R.reshape(-1).tolist(),
            "t": t.tolist(),
        }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + yaml.safe_dump(data, sort_keys=True))