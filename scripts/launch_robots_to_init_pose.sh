#!/usr/bin/env bash
#
# Brings up the dual-arm rig and drives both arms to the saved
# config/robot_init_pose.yaml pose, all from one terminal, no tmux/docker:
#   1. hardware.launch.py    (bringup)  -- backgrounded, logged
#   2. move_group.launch.py  (MoveIt)   -- backgrounded, logged
#   3. move_to_init_pose.py             -- runs in the foreground here; it
#      waits on its own (--timeout-s) for MoveGroup + a plannable state
#      before commanding any motion, so steps 1-3 can all start right away --
#      no need to wait for bringup/moveit yourself first. That plannable
#      state additionally needs the KUKA FRI application streaming from each
#      pendant (a manual, per-robot-controller step) -- start those once
#      step 1's log shows "Awaiting robot heartbeat" (or sooner).
#
# Ctrl+C (or normal exit) tears down bringup + MoveIt together.
#
# Usage:
#   scripts/launch_robots_to_init_pose.sh
#
# Env overrides:
#   ARM=left|right|both    (default: both)   -- which arm(s) to bring up + move
#   USE_GRIPPER=true|false (default: true)   -- must match both launch files
#   TIMEOUT_S=180           (default: 180)   -- move_to_init_pose.py --timeout-s
#   CONFIG=config/robot_init_pose.yaml       -- move_to_init_pose.py --config
#   LOG_DIR=outputs/robot_init_pose_logs     -- where bringup/moveit logs go

set -euo pipefail

ARM="${ARM:-both}"
USE_GRIPPER="${USE_GRIPPER:-true}"
TIMEOUT_S="${TIMEOUT_S:-180}"
CONFIG="${CONFIG:-config/robot_init_pose.yaml}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${LOG_DIR:-$REPO_DIR/outputs/robot_init_pose_logs}"

case "$ARM" in
    left)  LAUNCH_ARMS_ARG="arms:=lbr_one" ;;
    right) LAUNCH_ARMS_ARG="arms:=lbr_two" ;;
    both)  LAUNCH_ARMS_ARG="" ;;
    *) echo "ARM must be left|right|both (got: $ARM)" >&2; exit 1 ;;
esac

mkdir -p "$LOG_DIR"
BRINGUP_LOG="$LOG_DIR/bringup.log"
MOVEIT_LOG="$LOG_DIR/moveit.log"

export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
# ROS's own setup.bash files reference unset variables internally -- relax
# nounset just for sourcing them, restore it right after.
set +u
source /opt/ros/humble/setup.bash
source "$HOME/franka_ros2_ws/install/setup.bash"
if [ -f "$REPO_DIR/install/setup.bash" ]; then
    source "$REPO_DIR/install/setup.bash"
fi
set -u

echo "[*] starting bringup: hardware.launch.py (log: $BRINGUP_LOG)"
# shellcheck disable=SC2086
ros2 launch lbr_dual_arm_bringup hardware.launch.py use_gripper:="$USE_GRIPPER" \
    > "$BRINGUP_LOG" 2>&1 &
BRINGUP_PID=$!

sleep 5
echo "[*] starting MoveIt: move_group.launch.py (log: $MOVEIT_LOG)"
# shellcheck disable=SC2086
ros2 launch lbr_dual_arm_bringup move_group.launch.py mode:=hardware rviz:=true use_gripper:="$USE_GRIPPER" \
    > "$MOVEIT_LOG" 2>&1 &
MOVEIT_PID=$!

cleanup() {
    echo
    echo "[*] shutting down MoveIt + bringup..."
    kill "$MOVEIT_PID" "$BRINGUP_PID" 2>/dev/null || true
    wait "$MOVEIT_PID" "$BRINGUP_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "[*] Start the KUKA FRI application on each pendant now (if not already) --"
echo "    the init-pose move below will wait for it. Tail logs with:"
echo "      tail -f \"$BRINGUP_LOG\""
echo "      tail -f \"$MOVEIT_LOG\""
echo

cd "$REPO_DIR"
python3 -m src.calibration.move_to_init_pose --arm "$ARM" --config "$CONFIG" --timeout-s "$TIMEOUT_S"

echo "[*] init pose reached. Bringup + MoveIt still running -- Ctrl+C to stop them."
wait "$MOVEIT_PID" "$BRINGUP_PID"
