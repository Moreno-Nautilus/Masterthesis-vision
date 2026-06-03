#!/usr/bin/env bash
#
# Bundle 2 — restart the docker container and enter it with everything sourced.
# If pipeline args are given, runs the pipeline runner with those args.
# Otherwise drops into an interactive shell so you can paste your runner cmd.
#
# Usage:
#   scripts/launch_pipeline.sh                   # interactive shell, sources only
#   scripts/launch_pipeline.sh --debug-logging   # runs pipeline runner with given args
#
# Override the container name via env var:
#   CONTAINER=other-container scripts/launch_pipeline.sh

set -euo pipefail

CONTAINER="${CONTAINER:-thesis-newcuda}"

SRC='export FASTDDS_BUILTIN_TRANSPORTS=UDPv4 && source /opt/ros-thesis-venv/bin/activate && source /workspace/MasterThesis/install/setup.bash && cd /workspace/MasterThesis'

echo "[*] restarting container: $CONTAINER"
docker stop "$CONTAINER" >/dev/null
docker start "$CONTAINER" >/dev/null

if (( $# == 0 )); then
    exec docker exec -it "$CONTAINER" bash -lc "$SRC && exec bash"
else
    exec docker exec -it "$CONTAINER" bash -lc \
        "$SRC && exec python3 -m src.perception.ros.learn_runners.run_pipeline_track_multicam \"\$@\"" \
        -- "$@"
fi
