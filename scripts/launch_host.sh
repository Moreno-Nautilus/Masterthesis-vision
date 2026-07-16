#!/usr/bin/env bash
#
# Bundle 1 — host-side stack in a single tmux session.
#   window 0: cams       — ros2 launch mv_launch zed2i_pair.launch.py
#   window 1: foxglove   — ros2 launch foxglove_bridge foxglove_bridge_launch.xml
#   window 2: viz1       — visualize_pipeline for zed2i_1
#   window 3: viz2       — visualize_pipeline for zed2i_2
#   window 4: viz3       — visualize_pipeline for zed2i_3
#   window 5: axes       — python3 -m debug_pose_axes
#
# Usage:
#   scripts/launch_host.sh          # start session (and attach)
#   scripts/launch_host.sh stop     # kill the session
#   scripts/launch_host.sh attach   # attach if already running
#
# Set NUM_CAMERAS=2 to skip the viz window for the 3rd camera (zed2i_3):
#   NUM_CAMERAS=2 scripts/launch_host.sh
#
# Switch windows in tmux: Ctrl+b then 0/1/2/3/4 (or n/p for next/prev).
# Detach without killing: Ctrl+b d.

set -euo pipefail

SESSION="${SESSION:-mv_host}"
NUM_CAMERAS="${NUM_CAMERAS:-3}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SRC_HOST="export FASTDDS_BUILTIN_TRANSPORTS=UDPv4 && source /opt/ros/humble/setup.bash && source \"\$HOME/franka_ros2_ws/install/setup.bash\" && source \"$REPO_DIR/install/setup.bash\" && if [ -f \"\$HOME/franka_ros2_ws/install/setup.bash\" ]; then source \"\$HOME/franka_ros2_ws/install/setup.bash\"; fi"
SRC_ROS="$SRC_HOST"

CAM_CMD='ros2 launch mv_launch zed2i_pair.launch.py'
FOXGLOVE_CMD='ros2 launch foxglove_bridge foxglove_bridge_launch.xml address:=0.0.0.0 port:=8765'

VIZ1_CMD='python3 -m src.perception.ros.learn_runners.visualize_pipeline --node-name foundationpose_external_visualizer_zed2i_1 --cam-id zed2i_1 --rgb-topic /zed2i_1/zed_node/rgb/color/rect/image --camera-info-topic /zed2i_1/zed_node/rgb/color/rect/image/camera_info --debug-topic /perception/fp/debug_frame/zed2i_1 --raw-out-topic /perception/fp/rgb_raw/zed2i_1_external --sam-out-topic /perception/fp/sam_overlay/zed2i_1_external --dino-out-topic /perception/fp/dino_overlay/zed2i_1_external --pose-out-topic /perception/fp/pose_overlay/zed2i_1_external --track-out-topic /perception/fp/track_overlay/zed2i_1_external --output-scale 0.5 --max-sync-dt-s 999'

VIZ2_CMD='python3 -m src.perception.ros.learn_runners.visualize_pipeline --node-name foundationpose_external_visualizer_zed2i_2 --cam-id zed2i_2 --rgb-topic /zed2i_2/zed_node/rgb/color/rect/image --camera-info-topic /zed2i_2/zed_node/rgb/color/rect/image/camera_info --debug-topic /perception/fp/debug_frame/zed2i_2 --raw-out-topic /perception/fp/rgb_raw/zed2i_2_external --sam-out-topic /perception/fp/sam_overlay/zed2i_2_external --dino-out-topic /perception/fp/dino_overlay/zed2i_2_external --pose-out-topic /perception/fp/pose_overlay/zed2i_2_external --track-out-topic /perception/fp/track_overlay/zed2i_2_external --output-scale 0.5 --max-sync-dt-s 999'

VIZ3_CMD='python3 -m src.perception.ros.learn_runners.visualize_pipeline --node-name foundationpose_external_visualizer_zed2i_3 --cam-id zed2i_3 --rgb-topic /zed2i_3/zed_node/rgb/color/rect/image --camera-info-topic /zed2i_3/zed_node/rgb/color/rect/image/camera_info --debug-topic /perception/fp/debug_frame/zed2i_3 --raw-out-topic /perception/fp/rgb_raw/zed2i_3_external --sam-out-topic /perception/fp/sam_overlay/zed2i_3_external --dino-out-topic /perception/fp/dino_overlay/zed2i_3_external --pose-out-topic /perception/fp/pose_overlay/zed2i_3_external --track-out-topic /perception/fp/track_overlay/zed2i_3_external --output-scale 0.5 --max-sync-dt-s 999'

AXES_CMD='python3 -m debug_pose_axes'

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

# Create session detached, then feed each window the command via send-keys.
tmux new-session -d -s "$SESSION" -n cams -c "$REPO_DIR"
tmux send-keys -t "$SESSION:cams" "$SRC_HOST && $CAM_CMD" Enter

tmux new-window -t "$SESSION" -n foxglove -c "$REPO_DIR"
tmux send-keys -t "$SESSION:foxglove" "$SRC_ROS && $FOXGLOVE_CMD" Enter

tmux new-window -t "$SESSION" -n viz1 -c "$REPO_DIR"
tmux send-keys -t "$SESSION:viz1" "$SRC_HOST && $VIZ1_CMD" Enter

tmux new-window -t "$SESSION" -n viz2 -c "$REPO_DIR"
tmux send-keys -t "$SESSION:viz2" "$SRC_HOST && $VIZ2_CMD" Enter

if [ "$NUM_CAMERAS" -ge 3 ]; then
    tmux new-window -t "$SESSION" -n viz3 -c "$REPO_DIR"
    tmux send-keys -t "$SESSION:viz3" "$SRC_HOST && $VIZ3_CMD" Enter
fi

tmux new-window -t "$SESSION" -n axes -c "$REPO_DIR"
tmux send-keys -t "$SESSION:axes" "$SRC_HOST && $AXES_CMD" Enter

tmux select-window -t "$SESSION:cams"
exec tmux attach -t "$SESSION"
