#!/usr/bin/env bash
#
# Supervised launcher for the multicam pipeline.
#
# A rare (~1/6) "bad SAM session" produces 0 masks on every camera for the whole
# process lifetime and is only cured by a restart (see gdinoeager1). The pipeline
# detects this and exits with code 42; this wrapper relaunches on 42 (and ONLY
# on 42, so real crashes / Ctrl-C / clean exits are NOT looped).
#
# Usage — pass the normal runner flags:
#   scripts/run_pipeline_supervised.sh \
#     --num-cameras 3 --mask-source gdino_sam ... \
#     --track-pose-log-path outputs/logs/run.csv
#
#   LOG=outputs/logs/run.log MAX_RESTARTS=10 scripts/run_pipeline_supervised.sh ...
#
set -uo pipefail

MAX_RESTARTS="${MAX_RESTARTS:-10}"
LOG="${LOG:-outputs/logs/multicam_init_supervised.log}"

# Stop the loop on Ctrl-C instead of relaunching.
trap 'echo "[SUPERVISOR] interrupted"; exit 130' INT TERM

n=0
while :; do
    n=$((n + 1))
    run_log="${LOG%.log}.${n}.log"
    echo "[SUPERVISOR] launch #${n} (log: ${run_log})"
    python -m src.perception.ros.learn_runners.run_pipeline_track_multicam "$@" 2>&1 | tee "${run_log}"
    code=${PIPESTATUS[0]}
    echo "[SUPERVISOR] launch #${n} exited code=${code}"

    if [ "${code}" -ne 42 ]; then
        # 0 = clean, 130 = Ctrl-C, anything else = real crash → do NOT loop.
        echo "[SUPERVISOR] exit ${code} is not a bad-SAM-session (42) — stopping."
        exit "${code}"
    fi

    echo "[SUPERVISOR] bad SAM session (code 42) — relaunching to get a fresh CUDA state."
    if [ "${n}" -ge "${MAX_RESTARTS}" ]; then
        echo "[SUPERVISOR] reached MAX_RESTARTS=${MAX_RESTARTS} — giving up."
        exit 1
    fi
    sleep 3
done
