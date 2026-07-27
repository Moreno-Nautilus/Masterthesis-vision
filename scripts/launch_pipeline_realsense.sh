#!/usr/bin/env bash
#
# Bundle 2 (RealSense variant) — restart the docker container and enter it
# with everything sourced. If pipeline args are given, runs the RealSense
# pipeline runner (run_pipeline_track_multicam_realsense) with those args.
# Use a named mode to run a pinned preset.
# Otherwise drops into an interactive shell so you can paste your runner cmd.
#
# Camera set is fixed at 1 ZED (zed2i_1) + 2 end-effector-mounted RealSense
# cameras. This is a separate pipeline from launch_pipeline.sh (3-ZED) —
# that script is untouched.
#
# Usage:
#   scripts/launch_pipeline_realsense.sh                   # interactive shell, sources only
#   scripts/launch_pipeline_realsense.sh init-only         # locked trio init baseline
#   scripts/launch_pipeline_realsense.sh fast-track        # fast tracking preset
#   scripts/launch_pipeline_realsense.sh accurate-track    # settled-axis accuracy preset
#   scripts/launch_pipeline_realsense.sh baseline          # alias for init-only
#   scripts/launch_pipeline_realsense.sh --debug-logging   # runs pipeline runner with given args
#
# Pass --disable-debug-frames after a track preset to skip building/publishing
# fp_debug_msgs/DebugFrame (and therefore the /perception/fp/*_overlay/* topics),
# e.g. scripts/launch_pipeline_realsense.sh fast-track --disable-debug-frames
#
# Override the container name via env var:
#   CONTAINER=other-container scripts/launch_pipeline_realsense.sh

set -euo pipefail

CONTAINER="${CONTAINER:-vision}"

SRC='export FASTDDS_BUILTIN_TRANSPORTS=UDPv4 && source /opt/thesis-venv/bin/activate && source /workspace/MasterThesis/install/setup.bash && cd /workspace/MasterThesis'

COMMON_ARGS=(
    --num-cameras 3
    --flange-pose-topic /iiwa/ee_pose
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
)

INIT_ONLY_ARGS=(
    "${COMMON_ARGS[@]}"
    --run-mode init_only
    --debug-logging
    --debug-verbose-logs
    --log-init-poses
)

TRACK_BASE_ARGS=(
    "${COMMON_ARGS[@]}"
    --run-mode track
    --tracking-profile fast_cutie
    --sam-fp32
    --icp-variant point_to_point
    --track-icp-num-points 800
    --fused-track-icp-max-iteration 8
    --fused-track-icp-max-correspondence-dist-m 0.15
    --chamfer-every-n-frames 1
    --fused-track-max-translation-speed-mps 1.2
    --fused-track-min-translation-jump-m 0.10
    --fused-track-max-dt-s 0.80
    --fused-gate-max-centroid-dist-m 0.25
    --fused-track-centroid-recovery
    --fused-track-centroid-recovery-min-cameras 1
    --fused-track-centroid-recovery-cluster-dist-m 0.12
    --fused-track-centroid-recovery-max-seed-jump-m 0.75
    --fused-track-max-rotation-speed-degps 1200
    --fused-track-min-rotation-jump-deg 90
    --fused-track-hold-window-frames 5
    --fused-track-max-lost-frames 20
    --track-pose-mask-margin-px 250
    --timer-period-s 0.05
)

FAST_TRACK_ARGS=(
    "${TRACK_BASE_ARGS[@]}"
    --log-track-poses
    --track-pose-log-path outputs/logs/live_fast_track_realsense_q.csv
)

ACCURATE_TRACK_ARGS=(
    "${TRACK_BASE_ARGS[@]}"
    --fused-track-rot-slew-limit-deg 10.0
    --fused-track-rot-lowpass 0.2
    --fused-track-rot-reseed
    --fused-track-rot-reseed-chamfer-m 0.006
    --fused-track-rot-reseed-max-chamfer-m 0.080
    --fused-track-rot-reseed-n-rot 45
    --fused-track-rot-reseed-icp-iters 10
    --fused-track-pca-axis
    --fused-track-pca-axis-min-deg 8
    --fused-track-pca-axis-max-deg 45
    --fused-track-pca-axis-min-elongation 5.0
    --fused-track-pca-axis-min-points 80
    --fused-track-pca-axis-blend 0.5
    --log-init-poses
    --log-track-poses
    --track-pose-log-path outputs/logs/live_accurate_track_realsense_q.csv
)

usage() {
    sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
}

DISABLE_DEBUG_FRAMES=0
ARGS=()
for arg in "$@"; do
    if [ "$arg" = "--disable-debug-frames" ]; then
        DISABLE_DEBUG_FRAMES=1
    else
        ARGS+=("$arg")
    fi
done
set -- "${ARGS[@]}"

MODE_NAME=""
MODE_LOG=""
case "${1:-}" in
    -h|--help|help)
        usage
        exit 0
        ;;
    baseline|init-only|init_only)
        MODE_NAME="init-only"
        MODE_LOG="outputs/logs/multicam_init_realsense_baseline.log"
        shift
        set -- "${INIT_ONLY_ARGS[@]}" "$@"
        ;;
    fast-track|fast_track|fast)
        MODE_NAME="fast-track"
        MODE_LOG="outputs/logs/live_fast_track_realsense.log"
        shift
        set -- "${FAST_TRACK_ARGS[@]}" "$@"
        ;;
    accurate-track|accurate_track|accurate)
        MODE_NAME="accurate-track"
        MODE_LOG="outputs/logs/live_accurate_track_realsense.log"
        shift
        set -- "${ACCURATE_TRACK_ARGS[@]}" "$@"
        ;;
esac

if (( DISABLE_DEBUG_FRAMES )); then
    set -- "$@" --no-debug-frame-publish
fi

echo "[*] restarting container: $CONTAINER"
docker stop "$CONTAINER" >/dev/null
docker start "$CONTAINER" >/dev/null

if (( $# == 0 )); then
    exec docker exec -it "$CONTAINER" bash -lc "$SRC && exec bash"
elif [ -n "$MODE_NAME" ]; then
    echo "[*] launching pipeline preset: $MODE_NAME (realsense trio)"
    exec docker exec -it "$CONTAINER" bash -lc \
        "$SRC && mkdir -p outputs/logs && set -o pipefail && python3 -m src.perception.ros.learn_runners.run_pipeline_track_multicam_realsense \"\$@\" 2>&1 | tee \"$MODE_LOG\"" \
        -- "$@"
else
    exec docker exec -it "$CONTAINER" bash -lc \
        "$SRC && exec python3 -m src.perception.ros.learn_runners.run_pipeline_track_multicam_realsense \"\$@\"" \
        -- "$@"
fi
