#!/usr/bin/env bash
# Apply the local thesis modifications to the pinned third-party submodules.
# Run this once after `git submodule update --init`.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATCH="$REPO_ROOT/patches/FoundationPose_thesis_changes.patch"
FP="$REPO_ROOT/external/FoundationPose"

if [ ! -d "$FP/.git" ]; then
  echo "error: $FP is not initialized. Run: git submodule update --init external/FoundationPose" >&2
  exit 1
fi

# Skip if already applied (idempotent).
if git -C "$FP" apply --reverse --check "$PATCH" >/dev/null 2>&1; then
  echo "FoundationPose patch already applied — nothing to do."
  exit 0
fi

git -C "$FP" apply "$PATCH"
echo "Applied FoundationPose thesis patch to external/FoundationPose."
