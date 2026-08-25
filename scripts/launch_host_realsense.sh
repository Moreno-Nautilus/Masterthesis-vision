#!/usr/bin/env bash
#
# Bundle 1 (RealSense variant) — host-side stack in a single tmux session.
#   window 0: cams       — ros2 launch mv_launch zed_realsense_trio.launch.py
#   window 1: viz1       — visualize_pipeline for zed2i_1
#   window 2: viz2       — visualize_pipeline for realsense_1
#   window 3: viz3       — visualize_pipeline for realsense_2
#   window 4: axes       — python3 -m debug_pose_axes
#   window 5: foxglove   — ros2 launch foxglove_bridge foxglove_bridge_launch.xml
#                          (only created when --foxglove is passed)
#
# Camera set is fixed at 1 ZED (zed2i_1, static) + 2 end-effector-mounted
# RealSense cameras (realsense_1, realsense_2, dynamic extrinsics). This is
# a separate pipeline from launch_host.sh (3-ZED) — that script is untouched.
#
# Usage:
#   scripts/launch_host_realsense.sh              # start session (and attach)
#   scripts/launch_host_realsense.sh stop         # kill the session
#   scripts/launch_host_realsense.sh attach       # attach if already running
#   scripts/launch_host_realsense.sh --foxglove   # also start the foxglove_bridge window
#
# foxglove_bridge is OFF by default -- its topic_whitelist is '.*', so once a
# client subscribes through it, it can pull full-res images/IMU/point cloud
# over the network for anything on the graph. Pass --foxglove when you
# actually need live visualization; omit it for normal pipeline runs.
#
# RealSense serials default to the known units (realsense_1=260322275185,
# realsense_2=260522275434). Override if you swap hardware:
#   RS1_SERIAL=123456789 RS2_SERIAL=987654321 scripts/launch_host_realsense.sh
#
# ZED serial defaults to the current physical unit (33137761). Briefly
# swapped to 39725782 on 2026-08-14 to rule out a bad unit as the cause of
# a bad ZED calibration -- didn't help, reverted same day (see
# docs/camera_bandwidth_optimization.md). Override the same way:
#   CAM1_SERIAL=12345678 scripts/launch_host_realsense.sh
#
# Switch windows in tmux: Ctrl+b then 0/1/2/3/4/5 (or n/p for next/prev).
# Detach without killing: Ctrl+b d.

set -euo pipefail

SESSION="${SESSION:-mv_host_realsense}"
RS1_SERIAL="${RS1_SERIAL:-260322275185}"
RS2_SERIAL="${RS2_SERIAL:-260522275434}"
CAM1_SERIAL="${CAM1_SERIAL:-33137761}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SRC_HOST="export FASTDDS_BUILTIN_TRANSPORTS=UDPv4 && source /opt/ros/humble/setup.bash && source \"\$HOME/franka_ros2_ws/install/setup.bash\" && source \"$REPO_DIR/install/setup.bash\" && if [ -f \"\$HOME/franka_ros2_ws/install/setup.bash\" ]; then source \"\$HOME/franka_ros2_ws/install/setup.bash\"; fi"
SRC_ROS="$SRC_HOST"

CAM_CMD="ros2 launch mv_launch zed_realsense_trio.launch.py cam1_serial:='$CAM1_SERIAL' rs1_serial:='$RS1_SERIAL' rs2_serial:='$RS2_SERIAL'"
FOXGLOVE_CMD='ros2 launch foxglove_bridge foxglove_bridge_launch.xml address:=0.0.0.0 port:=8765'

VIZ1_CMD='python3 -m src.perception.ros.learn_runners.visualize_pipeline --node-name foundationpose_external_visualizer_zed2i_1 --cam-id zed2i_1 --rgb-topic /zed2i_1/zed_node/rgb/color/rect/image --camera-info-topic /zed2i_1/zed_node/rgb/color/rect/image/camera_info --debug-topic /perception/fp/debug_frame/zed2i_1 --raw-out-topic /perception/fp/rgb_raw/zed2i_1_external --sam-out-topic /perception/fp/sam_overlay/zed2i_1_external --dino-out-topic /perception/fp/dino_overlay/zed2i_1_external --pose-out-topic /perception/fp/pose_overlay/zed2i_1_external --track-out-topic /perception/fp/track_overlay/zed2i_1_external --output-scale 0.5 --max-sync-dt-s 999'

VIZ2_CMD='python3 -m src.perception.ros.learn_runners.visualize_pipeline --node-name foundationpose_external_visualizer_realsense_1 --cam-id realsense_1 --rgb-topic /realsense_1/camera/color/image_rect --camera-info-topic /realsense_1/camera/color/camera_info --debug-topic /perception/fp/debug_frame/realsense_1 --raw-out-topic /perception/fp/rgb_raw/realsense_1_external --sam-out-topic /perception/fp/sam_overlay/realsense_1_external --dino-out-topic /perception/fp/dino_overlay/realsense_1_external --pose-out-topic /perception/fp/pose_overlay/realsense_1_external --track-out-topic /perception/fp/track_overlay/realsense_1_external --output-scale 0.5 --max-sync-dt-s 999'

VIZ3_CMD='python3 -m src.perception.ros.learn_runners.visualize_pipeline --node-name foundationpose_external_visualizer_realsense_2 --cam-id realsense_2 --rgb-topic /realsense_2/camera/color/image_rect --camera-info-topic /realsense_2/camera/color/camera_info --debug-topic /perception/fp/debug_frame/realsense_2 --raw-out-topic /perception/fp/rgb_raw/realsense_2_external --sam-out-topic /perception/fp/sam_overlay/realsense_2_external --dino-out-topic /perception/fp/dino_overlay/realsense_2_external --pose-out-topic /perception/fp/pose_overlay/realsense_2_external --track-out-topic /perception/fp/track_overlay/realsense_2_external --output-scale 0.5 --max-sync-dt-s 999'

AXES_CMD='python3 -m debug_pose_axes'

FOXGLOVE=0
ARGS=()
for arg in "$@"; do
    if [ "$arg" = "--foxglove" ]; then
        FOXGLOVE=1
    else
        ARGS+=("$arg")
    fi
done
set -- "${ARGS[@]}"

case "${1:-}" in
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
esac

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "[*] session $SESSION already running — attaching"
    exec tmux attach -t "$SESSION"
fi

tmux new-session -d -s "$SESSION" -n cams -c "$REPO_DIR"
tmux send-keys -t "$SESSION:cams" "$SRC_HOST && $CAM_CMD" Enter

tmux new-window -t "$SESSION" -n viz1 -c "$REPO_DIR"
tmux send-keys -t "$SESSION:viz1" "$SRC_HOST && $VIZ1_CMD" Enter

tmux new-window -t "$SESSION" -n viz2 -c "$REPO_DIR"
tmux send-keys -t "$SESSION:viz2" "$SRC_HOST && $VIZ2_CMD" Enter

tmux new-window -t "$SESSION" -n viz3 -c "$REPO_DIR"
tmux send-keys -t "$SESSION:viz3" "$SRC_HOST && $VIZ3_CMD" Enter

tmux new-window -t "$SESSION" -n axes -c "$REPO_DIR"
tmux send-keys -t "$SESSION:axes" "$SRC_HOST && $AXES_CMD" Enter

if (( FOXGLOVE )); then
    tmux new-window -t "$SESSION" -n foxglove -c "$REPO_DIR"
    tmux send-keys -t "$SESSION:foxglove" "$SRC_ROS && $FOXGLOVE_CMD" Enter
fi

tmux select-window -t "$SESSION:cams"
exec tmux attach -t "$SESSION"
