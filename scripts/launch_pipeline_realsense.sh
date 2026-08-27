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
# Named presets (init-only/fast-track/accurate-track/baseline) run inside a
# tmux session with three windows so the tracker's start/stop/reset services
# are one keypress away without reopening a shell:
#   window 0 "run"       — the pipeline itself (this is what you were seeing before)
#   window 1 "keys"      — tracking_keyboard_control.py (s=start, x=stop, r=reset)
#   window 2 "cam-scene" — publish_camera_scene_objects.py --dual-arm: publishes
#                          the ZED camera + holder meshes as CollisionObjects on
#                          /planning_scene, in the same base_link frame as the
#                          tracked-part CollisionObjects the pipeline itself
#                          publishes. This is a plain topic publisher (no
#                          robot_state_publisher, no move_group) so it never
#                          duplicates the robot on a shared ROS network — for
#                          that reason it's also fine to run alongside a real
#                          MoveIt elsewhere on the network. The interactive
#                          mock-robot RViz viewer (scripts/view_scene.sh) is a
#                          separate, local-only visualization workflow and is
#                          untouched by this.
# Switch windows: Ctrl+b then 0/1/2 (or n/p). Detach without killing: Ctrl+b d.
#   scripts/launch_pipeline_realsense.sh stop              # kill that tmux session
#   scripts/launch_pipeline_realsense.sh attach            # attach if already running
#
# fp_debug_msgs/DebugFrame (and therefore the /perception/fp/*_overlay/*
# topics) is OFF by default -- pass --enable-debug-frames after a mode name
# to build/publish it, e.g.
#   scripts/launch_pipeline_realsense.sh fast-track --enable-debug-frames
#
# Override the container name or tmux session name via env vars:
#   CONTAINER=other-container scripts/launch_pipeline_realsense.sh
#   SESSION=other-session scripts/launch_pipeline_realsense.sh fast-track

set -euo pipefail

CONTAINER="${CONTAINER:-vision}"
SESSION="${SESSION:-mv_pipeline_realsense}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SRC='export FASTDDS_BUILTIN_TRANSPORTS=UDPv4 && source /opt/thesis-venv/bin/activate && source /workspace/MasterThesis/install/setup.bash && cd /workspace/MasterThesis'

COMMON_ARGS=(
    --num-cameras 3
    # Dual-arm bringup has no "world" link (see docs/visualization.md §2) --
    # base_link matches publish_camera_scene_objects.py --dual-arm's own
    # default, so tracked-part and camera-holder CollisionObjects land in the
    # same frame.
    --planning-scene-frame-id base_link
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
    sed -n '2,49p' "$0" | sed 's/^# \{0,1\}//'
}

ENABLE_DEBUG_FRAMES=0
ARGS=()
for arg in "$@"; do
    if [ "$arg" = "--enable-debug-frames" ]; then
        ENABLE_DEBUG_FRAMES=1
    else
        ARGS+=("$arg")
    fi
done
set -- "${ARGS[@]}"

# Snapshot before the mode-preset case below rewrites $@ -- tells us whether
# this invocation actually runs the pipeline (a mode name or raw runner
# args) vs. the bare/interactive-shell form, which must stay untouched.
HAD_ARGS=$#

MODE_NAME=""
MODE_LOG=""
case "${1:-}" in
    -h|--help|help)
        usage
        exit 0
        ;;
    stop)
        if tmux has-session -t "$SESSION" 2>/dev/null; then
            tmux kill-session -t "$SESSION"
            echo "[*] killed tmux session: $SESSION"
        else
            echo "[*] no tmux session named $SESSION"
        fi
        exit 0
        ;;
    attach)
        exec tmux attach -t "$SESSION"
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

if (( HAD_ARGS > 0 )) && (( ! ENABLE_DEBUG_FRAMES )); then
    set -- "$@" --no-debug-frame-publish
fi

# Plain CollisionObject publisher for the fixed ZED camera + holder meshes
# (no robot_state_publisher, no move_group -- see src/calibration/
# publish_camera_scene_objects.py) -- safe to run alongside the pipeline on a
# shared ROS network without duplicating the robot. --dual-arm both
# re-expresses the camera poses into base_link and defaults --frame-id to
# base_link, matching COMMON_ARGS' --planning-scene-frame-id above.
CAM_SCENE_CMD="$SRC && exec python3 -m src.calibration.publish_camera_scene_objects --dual-arm"

echo "[*] restarting container: $CONTAINER"
docker stop "$CONTAINER" >/dev/null
docker start "$CONTAINER" >/dev/null

if (( $# == 0 )); then
    exec docker exec -it "$CONTAINER" bash -lc "$SRC && exec bash"
elif [ -n "$MODE_NAME" ]; then
    echo "[*] launching pipeline preset: $MODE_NAME (realsense trio, tmux session: $SESSION)"

    # Quote each pipeline arg for safe re-use inside the tmux send-keys string below.
    QUOTED_ARGS=""
    for a in "$@"; do
        QUOTED_ARGS+=" $(printf '%q' "$a")"
    done
    RUN_CMD="$SRC && mkdir -p outputs/logs && set -o pipefail && python3 -m src.perception.ros.learn_runners.run_pipeline_track_multicam_realsense$QUOTED_ARGS 2>&1 | tee \"$MODE_LOG\""
    KEYS_CMD="$SRC && python3 -m src.perception.ros.tracking_keyboard_control"

    if tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "[*] session $SESSION already running — attaching (run 'scripts/launch_pipeline_realsense.sh stop' first for a clean restart)"
        exec tmux attach -t "$SESSION"
    fi

    tmux new-session -d -s "$SESSION" -n run -c "$REPO_DIR"
    tmux send-keys -t "$SESSION:run" "docker exec -it \"$CONTAINER\" bash -lc $(printf '%q' "$RUN_CMD")" Enter

    tmux new-window -t "$SESSION" -n keys -c "$REPO_DIR"
    tmux send-keys -t "$SESSION:keys" "docker exec -it \"$CONTAINER\" bash -lc $(printf '%q' "$KEYS_CMD")" Enter

    tmux new-window -t "$SESSION" -n cam-scene -c "$REPO_DIR"
    tmux send-keys -t "$SESSION:cam-scene" "docker exec -it \"$CONTAINER\" bash -lc $(printf '%q' "$CAM_SCENE_CMD")" Enter

    tmux select-window -t "$SESSION:run"
    exec tmux attach -t "$SESSION"
else
    # Custom args, no tmux: start the camera-scene publisher detached in the
    # background (docker restart above already ensures no stale copy is left
    # running from a previous invocation), then run the pipeline itself in
    # the foreground as before.
    docker exec -d "$CONTAINER" bash -lc "$CAM_SCENE_CMD"
    exec docker exec -it "$CONTAINER" bash -lc \
        "$SRC && exec python3 -m src.perception.ros.learn_runners.run_pipeline_track_multicam_realsense \"\$@\"" \
        -- "$@"
fi
