#!/usr/bin/env bash
#
# Brings up the dual-arm rig and drives both arms to the saved
# config/robot_init_pose.yaml pose, all from one terminal, no tmux/docker:
#   1. bringup launch file      -- backgrounded, logged. Which one depends on
#      CONTROL_MODE:
#        cartesian_impedance (default) -- cartesian_impedance.launch.py
#          (cartesian_impedance_lbr_one/_two, torque mode -- see
#          src/calibration/cartesian_impedance_dual_arm.py). Always brings
#          both arms' controllers up -- it has no `arms:=` argument, so ARM
#          (below) only restricts which arm move_to_init_pose.py commands,
#          not which arm's controller comes up.
#        position                     -- hardware.launch.py
#          (joint_trajectory_controller). Honors ARM's `arms:=` restriction
#          at bring-up too.
#   2. move_group.launch.py     -- backgrounded, logged. Same launch file
#      either way (mode:=hardware); control_mode:=cartesian_impedance is
#      passed through so it loads moveit_cartesian_impedance_controllers.yaml
#      instead of the plain joint_trajectory_controller config. In
#      cartesian_impedance mode MoveIt is only used for PLANNING
#      (plan_joint_trajectory()) -- execution goes straight to the impedance
#      controllers, not through MoveGroup.
#   3. move_to_init_pose.py --control-mode "$CONTROL_MODE" -- runs in the
#      foreground here; it waits on its own (--timeout-s) for MoveGroup + a
#      plannable state before commanding any motion, so steps 1-3 can all
#      start right away -- no need to wait for bringup/moveit yourself first.
#      That plannable state additionally needs the KUKA FRI application
#      streaming from each pendant (a manual, per-robot-controller step) --
#      start those once step 1's log shows "Awaiting robot heartbeat" (or, in
#      cartesian_impedance mode, once the controllers spawn -- see the log).
#
# Pendant settings differ by control mode -- see
# docs/calibration_control_modes.md's "Pendant settings" table:
#   position             -- send period 10 ms (100 Hz), FRI control mode
#                            "position", client command mode POSITION.
#   cartesian_impedance  -- send period 1 ms (1000 Hz), FRI control mode
#                            JOINT_IMPEDANCE_CONTROL, client command mode
#                            TORQUE.
#
# Ctrl+C (or normal exit) tears down bringup + MoveIt together.
#
# Usage:
#   scripts/launch_robots_to_init_pose.sh
#   CONTROL_MODE=position scripts/launch_robots_to_init_pose.sh
#
# Env overrides:
#   CONTROL_MODE=cartesian_impedance|position (default: cartesian_impedance)
#   ARM=left|right|both    (default: both)   -- which arm(s) to move; also
#                                                restricts bring-up in
#                                                position mode only (see
#                                                above)
#   USE_GRIPPER=true|false (default: true)   -- must match both launch files
#   TIMEOUT_S=180           (default: 180)   -- move_to_init_pose.py --timeout-s
#   CONFIG=config/robot_init_pose.yaml       -- move_to_init_pose.py --config
#   LOG_DIR=outputs/robot_init_pose_logs     -- where bringup/moveit logs go

set -euo pipefail

CONTROL_MODE="${CONTROL_MODE:-cartesian_impedance}"
ARM="${ARM:-both}"
USE_GRIPPER="${USE_GRIPPER:-true}"
TIMEOUT_S="${TIMEOUT_S:-180}"
CONFIG="${CONFIG:-config/robot_init_pose.yaml}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${LOG_DIR:-$REPO_DIR/outputs/robot_init_pose_logs}"

case "$CONTROL_MODE" in
    cartesian_impedance|position) ;;
    *) echo "CONTROL_MODE must be cartesian_impedance|position (got: $CONTROL_MODE)" >&2; exit 1 ;;
esac

case "$ARM" in
    left)  BRINGUP_ARMS_ARG="arms:=lbr_one" ;;
    right) BRINGUP_ARMS_ARG="arms:=lbr_two" ;;
    both)  BRINGUP_ARMS_ARG="" ;;
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

if [ "$CONTROL_MODE" = "cartesian_impedance" ]; then
    # cartesian_impedance.launch.py has no `arms:=` argument -- it always
    # brings both arms' controllers up together. ARM still restricts which
    # arm move_to_init_pose.py commands below.
    echo "[*] starting bringup: cartesian_impedance.launch.py (log: $BRINGUP_LOG)"
    ros2 launch lbr_dual_arm_bringup cartesian_impedance.launch.py use_gripper:="$USE_GRIPPER" \
        > "$BRINGUP_LOG" 2>&1 &
    BRINGUP_PID=$!
    BRINGUP_SETTLE_S=2
    MOVE_GROUP_CONTROL_MODE_ARG="control_mode:=cartesian_impedance"
else
    echo "[*] starting bringup: hardware.launch.py (log: $BRINGUP_LOG)"
    # shellcheck disable=SC2086
    ros2 launch lbr_dual_arm_bringup hardware.launch.py use_gripper:="$USE_GRIPPER" $BRINGUP_ARMS_ARG \
        > "$BRINGUP_LOG" 2>&1 &
    BRINGUP_PID=$!
    BRINGUP_SETTLE_S=5
    MOVE_GROUP_CONTROL_MODE_ARG=""
fi

sleep "$BRINGUP_SETTLE_S"
echo "[*] starting MoveIt: move_group.launch.py (log: $MOVEIT_LOG)"
# shellcheck disable=SC2086
ros2 launch lbr_dual_arm_bringup move_group.launch.py mode:=hardware rviz:=true use_gripper:="$USE_GRIPPER" \
    $MOVE_GROUP_CONTROL_MODE_ARG > "$MOVEIT_LOG" 2>&1 &
MOVEIT_PID=$!

cleanup() {
    echo
    echo "[*] shutting down MoveIt + bringup..."
    kill "$MOVEIT_PID" "$BRINGUP_PID" 2>/dev/null || true
    wait "$MOVEIT_PID" "$BRINGUP_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "[*] Start the KUKA FRI application on each pendant now (if not already) --"
if [ "$CONTROL_MODE" = "cartesian_impedance" ]; then
    echo "    send period 1 ms, FRI control mode JOINT_IMPEDANCE_CONTROL, client"
    echo "    command mode TORQUE (see docs/calibration_control_modes.md)."
else
    echo "    send period 10 ms, FRI control mode position, client command mode"
    echo "    POSITION (see docs/calibration_control_modes.md)."
fi
echo "    The init-pose move below will wait for it. Tail logs with:"
echo "      tail -f \"$BRINGUP_LOG\""
echo "      tail -f \"$MOVEIT_LOG\""
echo

cd "$REPO_DIR"
python3 -m src.calibration.move_to_init_pose --arm "$ARM" --control-mode "$CONTROL_MODE" \
    --config "$CONFIG" --timeout-s "$TIMEOUT_S"

echo "[*] init pose reached. Bringup + MoveIt still running -- Ctrl+C to stop them."
wait "$MOVEIT_PID" "$BRINGUP_PID"
