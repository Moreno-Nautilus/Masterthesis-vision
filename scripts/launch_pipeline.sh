#!/usr/bin/env bash
#
# Bundle 2 — restart the docker container and enter it with everything sourced.
# If pipeline args are given, runs the pipeline runner with those args.
# Use "baseline" to run the locked multicam init baseline.
# Otherwise drops into an interactive shell so you can paste your runner cmd.
#
# Usage:
#   scripts/launch_pipeline.sh                   # interactive shell, sources only
#   scripts/launch_pipeline.sh baseline          # runs the locked init baseline
#   scripts/launch_pipeline.sh --debug-logging   # runs pipeline runner with given args
#
# Override the container name via env var:
#   CONTAINER=other-container scripts/launch_pipeline.sh

set -euo pipefail

CONTAINER="${CONTAINER:-vision}"

SRC='export FASTDDS_BUILTIN_TRANSPORTS=UDPv4 && source /opt/ros-thesis-venv/bin/activate && source /workspace/Masterthesis-vision/install/setup.bash && cd /workspace/Masterthesis-vision'

BASELINE_ARGS=(
    --num-cameras 3
    --mask-source gdino_sam
    --gdino-device cpu
    --gdino-box-threshold 0.30
    --gdino-text-threshold 0.20
    --gdino-max-boxes 20
    --sam-max-image-side 1536
    --reference-source real
    --dino-min-crop-side 112
    --icp-grid-n-rot 45
    --icp-grid-prescreen
    --icp-grid-cross-cam-chamfer
    --fusion-match-max-centroid-dist-m 0.07
    --depth-fill-holes-kernel 3
    --run-mode init_only
    --tracking-backend cutie
    --tracking-profile default
    --debug-logging
    --debug-verbose-logs
    --log-init-poses
    --track-pose-log-path outputs/logs/multicam_init_final_baseline.csv
)

BASELINE_MODE=0
if [ "${1:-}" = "baseline" ]; then
    BASELINE_MODE=1
    shift
    set -- "${BASELINE_ARGS[@]}" "$@"
fi

echo "[*] restarting container: $CONTAINER"
docker stop "$CONTAINER" >/dev/null
docker start "$CONTAINER" >/dev/null

if (( $# == 0 )); then
    exec docker exec -it "$CONTAINER" bash -lc "$SRC && exec bash"
elif (( BASELINE_MODE == 1 )); then
    exec docker exec -it "$CONTAINER" bash -lc \
        "$SRC && mkdir -p outputs/logs && set -o pipefail && python3 -m src.perception.ros.learn_runners.run_pipeline_track_multicam \"\$@\" 2>&1 | tee outputs/logs/multicam_init_final_baseline.log" \
        -- "$@"
else
    exec docker exec -it "$CONTAINER" bash -lc \
        "$SRC && exec python3 -m src.perception.ros.learn_runners.run_pipeline_track_multicam \"\$@\"" \
        -- "$@"
fi
