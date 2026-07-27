from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass


@dataclass
class RealsenseDevice:
    name: str
    serial: str


def list_connected_realsense_devices() -> list[RealsenseDevice]:
    """Runs `rs-enumerate-devices -s` (host-installed librealsense2-utils)
    and parses the one-row-per-device summary table into a list.

    Raises RuntimeError if the CLI tool isn't on PATH or no devices are
    found -- callers should treat that as "nothing connected", not silently
    proceed with zero cameras.
    """
    exe = shutil.which("rs-enumerate-devices")
    if exe is None:
        raise RuntimeError(
            "rs-enumerate-devices not found on PATH. This calibration script "
            "must be run where librealsense2-utils is installed (the host, "
            "not necessarily the 'vision' container) -- see "
            "docs/getting_started_realsense.md section 3."
        )

    proc = subprocess.run([exe, "-s"], capture_output=True, text=True, timeout=15)
    if proc.returncode != 0:
        raise RuntimeError(f"rs-enumerate-devices failed: {proc.stderr.strip()}")

    devices: list[RealsenseDevice] = []
    for line in proc.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        name, serial = parts[0], parts[1]
        if not serial or not serial.isdigit():
            continue
        if "intel" not in name.lower() and "realsense" not in name.lower() and "d4" not in name.lower():
            continue
        devices.append(RealsenseDevice(name=name, serial=serial))

    if not devices:
        raise RuntimeError(
            "No RealSense devices detected by rs-enumerate-devices. Check USB3 "
            "connections (docs/getting_started_realsense.md section 7)."
        )

    return devices


def match_devices_to_cam_ids(
    devices: list[RealsenseDevice],
    known_serials: dict[str, str],
) -> dict[str, RealsenseDevice]:
    """known_serials: {cam_id: serial}. Returns {cam_id: device} for every
    known cam_id whose serial is currently connected; cam_ids whose serial
    isn't plugged in are simply omitted (not an error -- callers decide
    whether that's fatal)."""
    by_serial = {d.serial: d for d in devices}
    out: dict[str, RealsenseDevice] = {}
    for cam_id, serial in known_serials.items():
        if serial in by_serial:
            out[cam_id] = by_serial[serial]
    return out


if __name__ == "__main__":
    for d in list_connected_realsense_devices():
        print(f"{d.name}  serial={d.serial}")
