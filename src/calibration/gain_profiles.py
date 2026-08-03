"""Loader for the named, human-editable gain profiles used by
cartesian_impedance_dual_arm.py and admittance_dual_arm.py (e.g. "holding"
vs "insertion") -- config data, not hardcoded in either client, following
this repo's config/ convention (e.g. config/base_board_pose.yaml).

Both profile files are flat: {profile_name: {param: value, ...}}. Callers
build the actual GainSettings/AdmittanceController kwargs from the returned
dict themselves -- this module only loads and validates presence.
"""

from __future__ import annotations

from pathlib import Path

import yaml

IMPEDANCE_GAIN_PROFILES_PATH = Path("config/impedance_gain_profiles.yaml")
ADMITTANCE_GAIN_PROFILES_PATH = Path("config/admittance_gain_profiles.yaml")


def _load_profiles(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"No gain profiles file at {path}")
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping of profile_name -> params, got {type(data)}")
    return data


def load_impedance_profile(name: str, path: Path = IMPEDANCE_GAIN_PROFILES_PATH) -> dict:
    profiles = _load_profiles(path)
    if name not in profiles:
        raise KeyError(f"No impedance gain profile {name!r} in {path} -- have {sorted(profiles)}")
    return profiles[name]


def load_admittance_profile(name: str, path: Path = ADMITTANCE_GAIN_PROFILES_PATH) -> dict:
    profiles = _load_profiles(path)
    if name not in profiles:
        raise KeyError(f"No admittance gain profile {name!r} in {path} -- have {sorted(profiles)}")
    return profiles[name]
