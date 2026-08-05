#!/usr/bin/env bash
#
# Wrapper around scripts/launch_gripper_moveit_debug.launch.py that resolves
# its own absolute path first, since `ros2 launch` always treats a bare path
# as a package name and rejects it.
#
# Usage:
#   source /opt/ros/humble/setup.bash
#   source ~/franka_ros2_ws/install/setup.bash   # wherever the lbr_fri_ros2_stack workspace lives
#   scripts/view_gripper_moveit_debug.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec ros2 launch "$SCRIPT_DIR/launch_gripper_moveit_debug.launch.py" "$@"
