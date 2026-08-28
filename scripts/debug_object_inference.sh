#!/usr/bin/env bash
#
# Runs both halves of the object-inference debug tool back to back:
#
#   1. tools/bagviz/run_object_inference_debug.py -- fresh Grounding DINO /
#      SAM2 / FoundationPose inference on a frame captured by
#      capture_pipeline_snapshots.py. Needs the full GPU inference stack,
#      so this runs inside the `vision` docker container (normally
#      headless -- it never opens a window, only computes + saves).
#   2. tools/bagviz/view_object_inference_debug.py -- the Open3D viewer for
#      what step 1 saved. Needs only numpy/open3d/pyyaml, so this runs on
#      the host in the `bagviz` conda env.
#
# See docs/bagviz_quickstart.md for what each stage actually does and
# docs/bagviz_quickstart.md#running-fresh-inference-on-a-captured-frame-not-the-bags-cached-detections
# for the filtering caveats.
#
# Usage:
#   scripts/debug_object_inference.sh --run-dir <run-dir> [--frame N] [--cameras a,b,c] \
#       [--skip-inference | --skip-view] [-- <extra run_object_inference_debug.py args>]
#
# Examples:
#   scripts/debug_object_inference.sh --run-dir outputs/bagviz/<run>
#   scripts/debug_object_inference.sh --run-dir outputs/bagviz/<run> --frame 1
#   scripts/debug_object_inference.sh --run-dir outputs/bagviz/<run> -- --gdino-text-threshold 0.4
#   scripts/debug_object_inference.sh --run-dir outputs/bagviz/<run> --skip-inference   # just re-view
#
# <run-dir> must be a path RELATIVE to the repo root (e.g. outputs/bagviz/<run>),
# not absolute -- it gets resolved against /workspace/MasterThesis inside the
# container and against this repo's root on the host, which only line up for
# a relative path.
#
# Anything after a literal `--` is forwarded only to stage 1
# (run_object_inference_debug.py) -- e.g. --device, --gdino-*, --sam-*,
# --cad-dir, --mesh-scale, --max-depth-m. Stage 2's own flags
# (--use-config-extrinsics, --extrinsics-yaml, --extrinsics-realsense-yaml,
# --dry-run) aren't reachable through this wrapper -- call
# tools.bagviz.view_object_inference_debug directly for those.
#
# Env overrides:
#   CONTAINER=other-container     docker container to run inference in (default: vision)
#   CONDA_ENV_NAME=other-env      host conda env for the viewer (default: bagviz)

set -euo pipefail

CONTAINER="${CONTAINER:-vision}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-bagviz}"

RUN_DIR=""
FRAME=0
CAMERAS=""
SKIP_VIEW=0
SKIP_INFERENCE=0
EXTRA_INFER_ARGS=()

usage() {
    echo "Usage: $0 --run-dir <run-dir> [--frame N] [--cameras a,b,c] [--skip-inference | --skip-view] [-- <extra args>]" >&2
}

while [ $# -gt 0 ]; do
    case "$1" in
        --run-dir)        RUN_DIR="$2"; shift 2 ;;
        --frame)          FRAME="$2"; shift 2 ;;
        --cameras)        CAMERAS="$2"; shift 2 ;;
        --skip-view)      SKIP_VIEW=1; shift ;;
        --skip-inference) SKIP_INFERENCE=1; shift ;;
        --)               shift; EXTRA_INFER_ARGS=("$@"); break ;;
        -h|--help)        usage; exit 0 ;;
        *)                echo "Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

if [ -z "$RUN_DIR" ]; then
    usage
    exit 1
fi

if [ "$SKIP_INFERENCE" -eq 1 ] && [ "$SKIP_VIEW" -eq 1 ]; then
    echo "--skip-inference and --skip-view together leave nothing to do." >&2
    exit 1
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

COMMON_ARGS=(--run-dir "$RUN_DIR" --frame "$FRAME")
if [ -n "$CAMERAS" ]; then
    COMMON_ARGS+=(--cameras "$CAMERAS")
fi

if [ "$SKIP_INFERENCE" -eq 0 ]; then
    echo "[*] stage 1/2: fresh inference in the '$CONTAINER' container ..."
    INFER_ARGS=("${COMMON_ARGS[@]}" "${EXTRA_INFER_ARGS[@]}")
    INFER_CMD="source /opt/thesis-venv/bin/activate && cd /workspace/MasterThesis && python -m tools.bagviz.run_object_inference_debug $(printf '%q ' "${INFER_ARGS[@]}")"
    docker exec -it "$CONTAINER" bash -lc "$INFER_CMD"
fi

if [ "$SKIP_VIEW" -eq 0 ]; then
    echo "[*] stage 2/2: viewer (host, conda env '$CONDA_ENV_NAME') ..."
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV_NAME"
    python3 -m tools.bagviz.view_object_inference_debug "${COMMON_ARGS[@]}"
fi
