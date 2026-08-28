"""Assembly-name resolution and stable Fabrica part_id assignment.

Extracted from run_pipeline_track_multicam_realsense.py.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


# Assembly subfolder name (under Data/CAD_Models*, Data/ZED_screens, Data/reference_renders)
# for each known object_id prefix. Objects with no matching prefix (e.g. blue_cube,
# screwdriver_1) live directly under the Data root and have no assembly.
ASSEMBLY_PREFIXES = {
    "cooling": "cooling_manifold",
    "pb": "plumbers_block",
}


def resolve_assembly_name(object_id: str) -> str:
    """Map an object_id (e.g. 'cooling_screw', 'pb_pipe') to its assembly name.

    Returns "" if object_id does not belong to a known assembly.
    """
    prefix = str(object_id).split("_", 1)[0]
    return ASSEMBLY_PREFIXES.get(prefix, "")


class PartIdAssigner:
    """Maps (assembly_name, object_id) detections to stable Fabrica part_ids.

    Fabrica lists assembly parts as assembly/0, assembly/1, ... with repeated
    object_ids where the same part occurs multiple times (e.g. plumbers_block
    slot 1 and slot 4 are both pb_screw). The config (assembly_part_ids.json)
    gives, per assembly, the object_id at each slot index (== part_id). Since
    duplicate-object slots are physically interchangeable, each new tracked
    instance of that object_id simply claims the lowest not-yet-claimed slot;
    the claim is keyed by track_id so it stays stable for the life of that
    track (including across re-init, which reuses track_id via centroid
    matching in _resolve_track_id_for_new_detection).
    """

    def __init__(self, config_path: str):
        self._slots_by_assembly: dict[str, list[str]] = {}
        path = Path(config_path)
        if path.is_file():
            with path.open("r") as f:
                raw = json.load(f)
            self._slots_by_assembly = {k: list(v) for k, v in raw.items()}
        # track_id -> (assembly_name, part_id), so claims are scoped per assembly.
        self._claim_by_track: dict[str, tuple[str, int]] = {}

    def resolve(self, assembly_name: str, object_id: str, track_id: str) -> int:
        """Return the stable part_id for track_id, claiming a free slot on first use.

        Returns -1 if assembly_name is unknown or has no matching slot for object_id.
        """
        if track_id in self._claim_by_track:
            claimed_assembly, claimed_part_id = self._claim_by_track[track_id]
            if claimed_assembly == assembly_name:
                return claimed_part_id

        slots = self._slots_by_assembly.get(assembly_name, [])
        claimed_in_assembly = {
            part_id for (a, part_id) in self._claim_by_track.values() if a == assembly_name
        }
        for part_id, slot_object_id in enumerate(slots):
            if slot_object_id == object_id and part_id not in claimed_in_assembly:
                self._claim_by_track[track_id] = (assembly_name, part_id)
                return part_id
        return -1

    def part_id_for_track(self, track_id: str) -> Optional[tuple[str, int]]:
        """Return the (assembly_name, part_id) already claimed by track_id, if any."""
        return self._claim_by_track.get(track_id)

    def release(self, track_id: str) -> None:
        """Drop a stale track_id's slot claim so it can be reused."""
        self._claim_by_track.pop(track_id, None)
