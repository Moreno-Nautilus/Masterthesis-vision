#!/usr/bin/env bash
#
# ZED calibration stage of src/calibration/autocalibrate_dual_realsense.py:
# calibrate the single static ZED (zed2i_1) against the checkerboard pose
# the board-pose stage just computed (config/base_board_pose.yaml).
#
# This is a thin wrapper, not a separate calibration implementation --
# src/calibration/base_to_cams_calib_3.py was generalized to accept an
# arbitrary --cam-ids list (still defaulting to the original 3-ZED trio when
# called with no args, so nothing else that relies on it changes behavior).
# This script just pins that same, unmodified solve to the one ZED this
# RealSense-trio rig actually has, so the 3-ZED script stays reusable as-is
# if a future rig adds more static ZEDs back.
#
# Run from INSIDE the 'vision' container (same environment
# autocalibrate_dual_realsense.py itself runs in -- it shells out to this
# script directly, no docker exec here). Can also be run standalone any time
# after config/base_board_pose.yaml holds a real (non-placeholder) pose:
#
#   scripts/calibrate_zed_from_board_pose.sh
#   scripts/calibrate_zed_from_board_pose.sh --cam-ids zed2i_1 zed2i_2   # override

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [ "$#" -eq 0 ]; then
    set -- --cam-ids zed2i_1
fi

exec python3 -m src.calibration.base_to_cams_calib_3 "$@"
