#!/usr/bin/env bash
#
# Cartesian-impedance twin of launch_robots_to_init_pose.sh: brings up the
# dual-arm rig in torque mode and drives both arms to the saved
# config/robot_init_pose.yaml pose through the compliant controllers,
# all from one terminal, no tmux/docker:
#   1. cartesian_impedance.launch.py (bringup) -- backgrounded, logged.
#      Brings up cartesian_impedance_lbr_one/_two (torque mode) instead of
#      joint_trajectory_controller -- see
#      lbr_dual_arm_bringup/launch/cartesian_impedance.launch.py and
#      src/calibration/cartesian_impedance_dual_arm.py.
#   2. move_group.launch.py (MoveIt)          -- backgrounded, logged. Same
#      launch file as the position-mode script (mode:=hardware) -- MoveIt is
#      only used for PLANNING here (plan_joint_trajectory()); execution goes
#      straight to the impedance controllers, not through MoveGroup.
#   3. move_to_init_pose.py --control-mode cartesian_impedance -- runs in the
#      foreground here; it waits on its own (--timeout-s) for MoveGroup + a
#      plannable state before commanding any motion, so steps 1-3 can all
#      start right away -- no need to wait for bringup/moveit yourself first.
#      That plannable state additionally needs the KUKA FRI application
#      streaming from each pendant (a manual, per-robot-controller step) --
#      start those once step 1's log shows the controllers spawned.
#
# Pendant settings differ from position mode -- see
# docs/calibration_control_modes.md's "Pendant settings" table: send period
# 1 ms (1000 Hz), FRI control mode JOINT_IMPEDANCE_CONTROL, client command
# mode TORQUE.
#
# Ctrl+C (or normal exit) tears down bringup + MoveIt together.
#
# Usage:
#   scripts/launch_robots_to_init_pose_impedance.sh
#
# Env overrides:
#   USE_GRIPPER=true|false (default: true)   -- must match both launch files
#   TIMEOUT_S=180           (default: 180)   -- move_to_init_pose.py --timeout-s
#   CONFIG=config/robot_init_pose.yaml       -- move_to_init_pose.py --config
#   LOG_DIR=outputs/robot_init_pose_impedance_logs -- where bringup/moveit logs go
#
# Note: unlike launch_robots_to_init_pose.sh, there is no ARM env override
# here -- cartesian_impedance.launch.py always brings both
# cartesian_impedance_lbr_one/_two up together (it has no `arms` launch
# argument). Use move_to_init_pose.py's own --arm directly if you need to
# restrict which arm actually moves once this script's arms are up.

set -euo pipefail

USE_GRIPPER="${USE_GRIPPER:-true}"
TIMEOUT_S="${TIMEOUT_S:-180}"
CONFIG="${CONFIG:-config/robot_init_pose.yaml}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${LOG_DIR:-$REPO_DIR/outputs/robot_init_pose_impedance_logs}"

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

echo "[*] starting bringup: cartesian_impedance.launch.py (log: $BRINGUP_LOG)"
ros2 launch lbr_dual_arm_bringup cartesian_impedance.launch.py use_gripper:="$USE_GRIPPER" \
    > "$BRINGUP_LOG" 2>&1 &
BRINGUP_PID=$!

sleep 5
echo "[*] starting MoveIt: move_group.launch.py (log: $MOVEIT_LOG)"
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
echo "    send period 1 ms, FRI control mode JOINT_IMPEDANCE_CONTROL, client"
echo "    command mode TORQUE (see docs/calibration_control_modes.md)."
echo "    The init-pose move below will wait for it. Tail logs with:"
echo "      tail -f \"$BRINGUP_LOG\""
echo "      tail -f \"$MOVEIT_LOG\""
echo

cd "$REPO_DIR"
python3 -m src.calibration.move_to_init_pose --arm both --control-mode cartesian_impedance \
    --config "$CONFIG" --timeout-s "$TIMEOUT_S"

echo "[*] init pose reached. Bringup + MoveIt still running -- Ctrl+C to stop them."
wait "$MOVEIT_PID" "$BRINGUP_PID"
