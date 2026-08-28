"""Config resolution for the multicam pipeline runners.

Every knob has its default in ``config.yaml`` (this directory). ``--config``
names an overlay YAML merged on top of it — the launch-script presets in
``presets/`` are exactly that. If the resolved config names a
``tracking_profile``, ``profiles/<name>.yaml`` is merged next. Any remaining
``--key value`` / ``--flag`` / ``--no-flag`` args (ad-hoc tweaks) are folded on
last and win.

Replaces the ~350-line argparse block that used to live in the runner.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

BASE_CONFIG = Path(__file__).with_name("config.yaml")
PROFILES_DIR = Path(__file__).with_name("profiles")

# store_false / enable-disable spellings the retired argparse understood, mapped
# to (config_key, value). Everything else is handled generically below.
_CLI_ALIASES = {
    "no-debug-frame-publish":     ("debug_frame_publish", False),
    "debug-frame-publish":        ("debug_frame_publish", True),
    "no-restart-on-dead-init":    ("restart_on_dead_init", False),
    "no-gdino-use-items-prompt":  ("gdino_use_items_prompt", False),
    "gdino-use-items-prompt":     ("gdino_use_items_prompt", True),
    "enable-fused-kalman":        ("disable_fused_kalman", False),
    "disable-fused-kalman":       ("disable_fused_kalman", True),
    "enable-axis-jump-gate":      ("disable_axis_jump_gate", False),
    "disable-axis-jump-gate":     ("disable_axis_jump_gate", True),
}


def _load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"config {path!r}: top level must be a mapping")
    return data


def _coerce(raw: str, ref):
    """Cast a CLI string to the type of its YAML counterpart (``ref``)."""
    if isinstance(ref, bool):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(ref, int):
        return int(raw)
    if isinstance(ref, float):
        return float(raw)
    if ref is None:
        for cast in (int, float):
            try:
                return cast(raw)
            except ValueError:
                pass
    return raw


def _apply_cli_overrides(cfg: dict, argv: list) -> None:
    """Fold leftover ``--key value`` / ``--flag`` / ``--no-flag`` tokens onto cfg.

    Mirrors the retired argparse so the launch-script presets and one-off runs
    keep working: ``--x-y v`` sets ``cfg['x_y']`` (typed from its YAML value), a
    bare ``--flag`` is True, ``--no-flag`` is False, and ``--tracking-profile
    NAME`` merges ``profiles/NAME.yaml`` in place (as _TrackingProfileAction did).
    """
    i = 0
    while i < len(argv):
        tok = argv[i]
        if not tok.startswith("--"):
            raise SystemExit(f"pipeline config: unexpected argument {tok!r}")
        name = tok[2:]
        if name in _CLI_ALIASES:
            key, val = _CLI_ALIASES[name]
            cfg[key] = val
            i += 1
            continue
        key = name.replace("-", "_")
        if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
            cfg[key] = _coerce(argv[i + 1], cfg.get(key))
            i += 2
        else:
            cfg[key] = True
            i += 1
        if key == "tracking_profile" and cfg.get("tracking_profile"):
            prof = cfg["tracking_profile"]
            cfg.update(_load_yaml(str(PROFILES_DIR / f"{prof}.yaml")))
            cfg["tracking_profile"] = prof


def parse_args() -> argparse.Namespace:
    """Resolve the pipeline config: base YAML + optional overlay + profile + CLI."""
    p = argparse.ArgumentParser(
        description="FoundationPose multi-camera tracking pipeline"
    )
    p.add_argument(
        "--config",
        default=None,
        help="YAML overlay merged onto config.yaml (see presets/ for the "
             "launch-script presets). Omit to run config.yaml as-is.",
    )
    known, rest = p.parse_known_args()

    # Layers, lowest precedence first: base config.yaml < tracking profile <
    # --config overlay (presets/) < CLI overrides. The profile sits *under* the
    # overlay so a preset can still tighten a value the profile relaxed, exactly
    # as the old "--tracking-profile fast_cutie <more flags>" ordering did.
    cfg = _load_yaml(str(BASE_CONFIG))
    overlay = {}
    if known.config:
        try:
            overlay = _load_yaml(known.config)
        except FileNotFoundError:
            p.error(f"--config {known.config}: file not found")

    profile = overlay.get("tracking_profile", cfg.get("tracking_profile"))
    if profile:
        try:
            cfg.update(_load_yaml(str(PROFILES_DIR / f"{profile}.yaml")))
        except FileNotFoundError:
            p.error(f"tracking_profile {profile!r}: no profiles/{profile}.yaml")

    cfg.update(overlay)
    if profile:
        cfg["tracking_profile"] = profile
    _apply_cli_overrides(cfg, rest)

    args = argparse.Namespace(**cfg)

    if getattr(args, "run_mode", None) == "track" and args.tracking_profile is None:
        p.error("tracking_profile 'fast_cutie' is required for run_mode 'track' "
                "(set it in the config or pass --tracking-profile fast_cutie)")
    if not (1 <= args.min_active_cameras <= 3):
        p.error(f"min_active_cameras must be between 1 and 3, got {args.min_active_cameras}")
    return args
