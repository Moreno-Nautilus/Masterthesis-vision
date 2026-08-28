#!/usr/bin/env bash
#
# Offline sanity-check dump of a recorded rosbag's perception-pipeline debug
# output: per-camera RGB/depth frames, whatever dino/sam/pose/track overlays
# the bag already contains, reconstructed masks, raw + segmented point
# clouds, and a redrawn coordinate-axes overlay + poses.yaml for the
# detected objects. See tools/bagviz/capture_pipeline_snapshots.py for
# details and the full flag list.
#
# Runs in the dedicated `bagviz` conda env (opencv/open3d/numpy/pyyaml);
# rclpy + fp_debug_msgs come from this machine's normal ROS 2 Humble +
# workspace sourcing in ~/.bashrc, not from the conda env itself.
#
# Usage:
#   scripts/visualize_bag_pipeline.sh <path-to-bag> [extra capture_pipeline_snapshots.py args...]
#   scripts/visualize_bag_pipeline.sh ~/Desktop/rosbag_20260807_173538
#   scripts/visualize_bag_pipeline.sh ~/Desktop/rosbag_20260807_173538 --num-frames 5 --cameras zed2i_1

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <path-to-bag> [extra args...]" >&2
    exit 1
fi

BAG="$1"
shift

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate bagviz

exec python3 -m tools.bagviz.capture_pipeline_snapshots --bag "$BAG" "$@"
