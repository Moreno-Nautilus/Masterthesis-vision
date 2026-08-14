"""Standalone recapture of ONLY the checkerboard board-pose sample set --
gathers exactly `--num-samples` (default 5, pooled across whichever arm(s)
are used) checkerboard detections, then overwrites config/base_board_pose.yaml.

Two ways to get those detections (--source):
  live     (default) -- interactive manual positioning, same prompt-driven
      capture loop as capture_handeye_data.py (position the arm, press
      Enter, capture a FRESH image live).
  handeye  -- no robot motion, no camera needed at all: reuses the
      images/detections ALREADY saved by capture_handeye_data.py under
      outputs/calibration_debug/handeye/<cam_id>/sample_*.json (each one
      already has T_base_flange + T_cam_board from Stage A) and just
      recomputes T_base_board = T_base_flange @ T_flange_cam @ T_cam_board
      from that existing data. Use this when you already ran Stage A and
      the checkerboard hasn't moved since -- see that flag's own section
      below.

This is the manual/interactive alternative to
autocalibrate_dual_realsense.py's board-pose stage, which instead
auto-replays previously-saved flange poses and perturbs via Cartesian IK to
fill the gap. Use this script when you just want to redo the 5-sample
board-pose average -- e.g. the checkerboard moved and you don't want to
re-run the full autocalibrate pipeline (which also drives via saved flange
poses and runs the ZED stage). Scope is deliberately narrow: this script
NEVER touches config/flange_poses/<arm>.json (Stage A's hand-eye flange-pose
captures) and NEVER runs ZED calibration -- for that, use
autocalibrate_dual_realsense.py (optionally --zed-only) or
scripts/calibrate_zed_from_board_pose.sh directly, once this script has
written a fresh config/base_board_pose.yaml.

Requires hand-eye already solved (config/camera_extrinsics_realsense.yaml
has a real, non-identity T_flange_cam) for whichever camera(s) --arm
selects -- same prerequisite autocalibrate_dual_realsense.py checks. Neither
--source trusts or re-derives T_flange_cam itself; both just read it.

--source {live,handeye} (default: live)
    live -- as described above: drives/jogs the real arm(s), captures fresh
        images. Needs the robot bring-up + camera stack running (see the
        "Run" section below).
    handeye -- purely offline recompute from Stage A's already-saved
        detections. --controller/--gain-profile/--no-debug-topic are
        ignored (no robot motion, nothing to jog, no debug topic to
        publish). Ignores --num-samples too: pools ALL of the requested
        arm(s)' saved hand-eye samples (not just the last N) since more,
        already-accepted samples only make the average more robust --
        pass --arm to restrict which arm's samples are used if you only
        trust one camera's set. No ROS node, no rclpy.init() even -- this
        is plain file I/O + linear algebra, runs anywhere with the repo
        checked out. Copies each source sample's debug PNG over (for
        traceability back to which Stage A capture it came from) if that
        PNG is still on disk.

--arm {left,right,both} (default: both)
    Which arm(s)' wrist RealSense may contribute a sample this run. Applies
    to both --source values. --source live: a board-pose sample only needs
    ONE camera and the checkerboard is a single shared fixed object, so
    samples from both arms are pooled into one running total toward
    --num-samples: with --arm both, this script captures with left first,
    then tops up with right only if left didn't reach the target (type
    'q' + Enter to move on from an arm early). --source handeye: restricts
    which arm's saved hand-eye sample directory gets read.

--controller {moveit,admittance,handguided} (default: moveit)
    --source live only -- ignored (with a note) for --source handeye. How
    the arm gets positioned before each capture -- identical semantics to
    capture_handeye_data.py's --mode interactive. moveit/handguided need no
    active control loop from this script (jog via RViz or physically
    hand-guide, respectively); admittance runs AdmittanceDualArmNode scoped
    to whichever arm is currently being captured, in a background thread
    alongside the prompt.

--num-samples N (default: 5)
    --source live only -- ignored (with a note) for --source handeye, which
    always pools every saved sample instead (see --source above). Total
    POOLED sample target -- matches autocalibrate_dual_realsense.py's
    --target-board-samples default. Existing on-disk samples count toward
    this when --append is given.

Backup-before-overwrite: unless --append, archives (moves) any existing
outputs/calibration_debug/board_pose/dual/ and config/base_board_pose.yaml
to timestamped `..._bak_<UTC-timestamp>` paths before capturing anything
new, and auto-restores them if the run is interrupted before a fresh
config/base_board_pose.yaml was actually written (same policy as
capture_handeye_data.py -- see docs/calibration_cheatsheet.md).

Run (inside the 'vision' container), with the checkerboard placed and the
arm(s) you intend to use positionable (hardware.launch.py + move_group.launch.py
for --controller moveit, additionally admittance.launch.py for --controller
admittance, or calibration.launch.py for --controller handguided), and the
host camera stack (scripts/launch_host_realsense.sh) up:

    # Default: pool up to 5 samples, starting with the left arm, RViz-jogged.
    python3 -m src.calibration.capture_board_pose_data

    # Only the left arm's camera, admittance-guided:
    python3 -m src.calibration.capture_board_pose_data --arm left --controller admittance

    # Extend the existing pooled set instead of starting over (checkerboard hasn't moved):
    python3 -m src.calibration.capture_board_pose_data --append --num-samples 8

Or, with --source handeye, no robot/camera stack needed at all -- reuses
whatever Stage A already captured (run from anywhere the repo is checked out):

    python3 -m src.calibration.capture_board_pose_data --source handeye
    python3 -m src.calibration.capture_board_pose_data --source handeye --arm right
"""

from __future__ import annotations

import argparse
import shutil
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import PoseStamped
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

from src.calibration.admittance_dual_arm import (
    AdmittanceDualArmNode,
    apply_gain_profile,
    log_active_gains,
)
from src.calibration.board_pose_from_flange_realsense import (
    BoardPoseSample,
    _average_se3,
    _load_samples_from_dir as _load_board_samples_from_dir,
    _rotation_matrix_to_rpy_deg,
    _save_sample_json as _save_board_sample_json,
)
from src.calibration.calibration_log import log_checkerboard_transform
from src.calibration.capture_handeye_data import (
    HANDEYE_DEBUG_DIR,
    _archive_if_exists,
    _restore_untouched_backups,
)
from src.calibration.flange_pose_store import ARM_KEYS
from src.calibration.handeye_flange_cam_realsense import (
    MAX_REPROJ_ERR_PX,
    RGB_INFO_MAX_DT_S,
    HandEyeSample,
    _camera_topics,
    _draw_chessboard,
    _img_to_numpy_bgr,
    _K_from_camerainfo,
    _load_samples_from_dir as _load_handeye_samples_from_dir,
    _rgb_numpy_to_imgmsg,
    _solve_board_pose,
    _stamp_to_sec,
)
from src.calibration.io_extrinsics import load_extrinsics_yaml
from src.perception.ros.multicam_grabber_realsense import _pose_msg_to_se3
from src.perception.ros.qos_profiles import qos_profile_sensor_data_low_latency
from src.utils.robot_bases import get_active_robot_base, load_robot_bases
from src.utils.se3 import SE3

CONTROLLERS = ("moveit", "admittance", "handguided")
SOURCES = ("live", "handeye")

FLANGE_POSE_MAX_AGE_S = 0.25
DEFAULT_NUM_SAMPLES = 5   # matches autocalibrate_dual_realsense.py's DEFAULT_TARGET_BOARD_SAMPLES
SETTLE_WAIT_S = 2.0

CAMERA_EXTRINSICS_YAML = Path("config/camera_extrinsics_realsense.yaml")
BASE_BOARD_YAML = Path("config/base_board_pose.yaml")
BOARD_POSE_DEBUG_DIR = Path("outputs/calibration_debug/board_pose/dual")
ROBOT_BASES_YAML = "config/robot_bases.yaml"


class _CaptureNode(Node):
    """One arm's live wrist-RealSense image + flange-pose subscriptions.

    Deliberately not a copy of the multi-arm _DualArmCalibNode pattern used
    by autocalibrate_dual_realsense.py -- captures here happen one arm at a
    time (see module docstring), same as capture_handeye_data.py's own
    per-arm _CaptureNode, which this mirrors (minus the joint_state
    subscription -- this script never replays via MoveIt, so joint
    positions are never needed)."""

    def __init__(self, arm_key: str, publish_debug: bool):
        arm = ARM_KEYS[arm_key]
        super().__init__(f"capture_board_pose_data_{arm_key}")
        self.arm_key = arm_key
        self.cam_id = arm["cam_id"]
        rgb_topic, info_topic = _camera_topics(self.cam_id)

        self.img_msg: Optional[Image] = None
        self.img_t: Optional[float] = None
        self.info_t: Optional[float] = None
        self.K: Optional[np.ndarray] = None

        self.flange_pose: Optional[SE3] = None
        self.flange_pose_wall_t: float = 0.0

        self.create_subscription(Image, rgb_topic, self._on_img, qos_profile_sensor_data_low_latency)
        self.create_subscription(CameraInfo, info_topic, self._on_info, qos_profile_sensor_data_low_latency)
        self.create_subscription(
            PoseStamped, arm["flange_pose_topic"], self._on_flange, qos_profile_sensor_data_low_latency
        )

        self._debug_pub = None
        if publish_debug:
            self._debug_pub = self.create_publisher(
                Image, f"/calibration/capture_board_pose_data/{arm_key}/debug_image", 1
            )

        self.get_logger().info(
            f"[{arm_key}] cam={self.cam_id} rgb={rgb_topic} info={info_topic} "
            f"flange_pose={arm['flange_pose_topic']}"
        )

    def _on_img(self, msg: Image) -> None:
        self.img_msg = msg
        self.img_t = _stamp_to_sec(msg.header.stamp)

    def _on_info(self, msg: CameraInfo) -> None:
        self.info_t = _stamp_to_sec(msg.header.stamp)
        self.K = _K_from_camerainfo(msg)

    def _on_flange(self, msg: PoseStamped) -> None:
        self.flange_pose = _pose_msg_to_se3(msg)
        self.flange_pose_wall_t = time.time()

    def has_fresh_flange_pose(self) -> bool:
        if self.flange_pose is None:
            return False
        return (time.time() - self.flange_pose_wall_t) <= FLANGE_POSE_MAX_AGE_S

    def has_fresh_rgb_info(self) -> bool:
        if self.img_msg is None or self.K is None or self.img_t is None or self.info_t is None:
            return False
        return abs(self.img_t - self.info_t) <= RGB_INFO_MAX_DT_S

    def publish_debug(self, img_bgr: np.ndarray) -> None:
        if self._debug_pub is None:
            return
        self._debug_pub.publish(_rgb_numpy_to_imgmsg(img_bgr, self.cam_id, self.get_clock().now().to_msg()))


def _require_handeye_done(arms: list[str]) -> dict[str, SE3]:
    """Loads config/camera_extrinsics_realsense.yaml and confirms every
    requested arm's camera already has a real (non-identity) T_flange_cam --
    same check autocalibrate_dual_realsense.py does before its board-pose
    stage, scoped to only the camera(s) --arm actually needs."""
    if not CAMERA_EXTRINSICS_YAML.exists():
        raise RuntimeError(
            f"{CAMERA_EXTRINSICS_YAML} doesn't exist yet -- run capture_handeye_data.py then "
            f"calibrate_handeye.py --write first (see docs/calibration_cheatsheet.md)."
        )
    extrinsics = load_extrinsics_yaml(CAMERA_EXTRINSICS_YAML)
    cam_ids = [ARM_KEYS[a]["cam_id"] for a in arms]
    missing_or_identity = []
    for cam_id in cam_ids:
        T = extrinsics.get(cam_id)
        if T is None or (np.allclose(T.R, np.eye(3)) and np.allclose(T.t, 0.0)):
            missing_or_identity.append(cam_id)
    if missing_or_identity:
        raise RuntimeError(
            f"{CAMERA_EXTRINSICS_YAML} has no real (non-identity) T_flange_cam for "
            f"{missing_or_identity} -- run capture_handeye_data.py then calibrate_handeye.py "
            f"--write for {missing_or_identity} first (see docs/calibration_cheatsheet.md)."
        )
    return {cam_id: extrinsics[cam_id] for cam_id in cam_ids}


def _try_capture_board(
    node: _CaptureNode, T_flange_cam: SE3, T_robotA_armBase: SE3, idx: int,
) -> Optional[tuple[BoardPoseSample, np.ndarray]]:
    if not node.has_fresh_rgb_info():
        print("  [skip] no fresh RGB/CameraInfo yet")
        return None
    if not node.has_fresh_flange_pose():
        print(
            f"  [skip] no fresh flange pose (< {FLANGE_POSE_MAX_AGE_S}s old) on "
            f"{ARM_KEYS[node.arm_key]['flange_pose_topic']} -- is lbr_dual_arm_bringup running?"
        )
        return None

    img = _img_to_numpy_bgr(node.img_msg)
    K = node.K.copy()
    T_armBase_flange = node.flange_pose

    try:
        T_cam_board, corners, reproj_err = _solve_board_pose(img, K)
    except RuntimeError as e:
        print(f"  [skip] checkerboard not visible to {node.cam_id}: {e}")
        node.publish_debug(img)
        return None

    if reproj_err > MAX_REPROJ_ERR_PX:
        print(f"  [skip] reprojection error too high: {reproj_err:.3f}px > {MAX_REPROJ_ERR_PX}px")
        return None

    vis = _draw_chessboard(img, corners, f"{node.arm_key} board idx={idx} reproj={reproj_err:.3f}px")
    node.publish_debug(vis)

    T_armBase_board = T_armBase_flange @ T_flange_cam @ T_cam_board
    T_robotA_board = T_robotA_armBase @ T_armBase_board
    sample = BoardPoseSample(idx=idx, T_base_board=T_robotA_board, reproj_px=reproj_err)
    return sample, vis


def _prompt_capture_loop_single_arm(
    capture_node: _CaptureNode,
    remaining: int,
    start_idx: int,
    debug_dir: Path,
    T_flange_cam: SE3,
    T_robotA_armBase: SE3,
    controller_label: str,
    pooled_total: int,
    already: int,
) -> tuple[list[BoardPoseSample], int]:
    arm_key = capture_node.arm_key
    print(f"\n=== {controller_label} board-pose capture: arm={arm_key} (cam={capture_node.cam_id}) ===")
    print(f"Need up to {remaining} more sample(s) from this arm ({already}/{pooled_total} pooled so far).")
    print(f"Foxglove: Image panel -> /calibration/capture_board_pose_data/{arm_key}/debug_image")
    print("Keep the checkerboard FIXED for the whole run (across both arms, if using --arm both).")
    print("Position this arm so its wrist RealSense sees the board, let it settle, then press Enter")
    print("to capture. Type 'q' + Enter to move on early (to the next arm, or to finish).\n")

    samples: list[BoardPoseSample] = []
    idx = start_idx
    while len(samples) < remaining:
        pooled_so_far = already + len(samples)
        user_in = input(
            f"[{pooled_so_far}/{pooled_total}] arm={arm_key} Press Enter to capture (or 'q' to move on): "
        )
        if user_in.strip().lower() == "q":
            break

        time.sleep(SETTLE_WAIT_S)

        result = _try_capture_board(capture_node, T_flange_cam, T_robotA_armBase, idx)
        if result is None:
            continue
        sample, vis = result
        samples.append(sample)

        cv2.imwrite(str(debug_dir / f"sample_{idx:02d}_{arm_key}.png"), vis)
        _save_board_sample_json(debug_dir, sample)
        print(
            f"  [ok] reproj={sample.reproj_px:.3f}px  T_robotA_board.t={sample.T_base_board.t}  "
            f"saved -> {debug_dir}/sample_{idx:02d}.json ({already + len(samples)}/{pooled_total} pooled)"
        )
        idx += 1

    return samples, idx


def _average_and_write(samples: list[BoardPoseSample]) -> None:
    active_robot, T_robotA_activeRobot = get_active_robot_base(ROBOT_BASES_YAML)
    T_robotA_board_avg, t_std, rot_std_deg = _average_se3([s.T_base_board for s in samples])
    reproj_mean = float(np.mean([s.reproj_px for s in samples]))

    print(f"\nT_board averaged over {len(samples)} pooled sample(s) (robot_a frame):")
    print(T_robotA_board_avg)
    print(f"translation std: {t_std:.6f}m  rotation std: {rot_std_deg:.6f}deg  mean reproj: {reproj_mean:.3f}px")

    # active_robot's own frame -- matches board_pose_from_flange_realsense.py /
    # autocalibrate_dual_realsense.py's convention: base_board_pose.yaml's
    # primary entry is in active_robot's frame.
    T_activeRobot_board = T_robotA_activeRobot.inverse() @ T_robotA_board_avg
    roll, pitch, yaw = _rotation_matrix_to_rpy_deg(T_activeRobot_board.R)
    roll_a, pitch_a, yaw_a = _rotation_matrix_to_rpy_deg(T_robotA_board_avg.R)

    out_path = BASE_BOARD_YAML
    if out_path.exists():
        backup = out_path.with_suffix(".yaml.bak")
        backup.write_text(out_path.read_text())
        print(f"Backed up existing YAML to: {backup}")

    out_data = {
        "base_board": {
            "translation_xyz_m": [round(float(v), 6) for v in T_activeRobot_board.t],
            "rotation_rpy_deg": [round(roll, 4), round(pitch, 4), round(yaw, 4)],
        },
        "base_board_robot_a_frame": {
            "translation_xyz_m": [round(float(v), 6) for v in T_robotA_board_avg.t],
            "rotation_rpy_deg": [round(roll_a, 4), round(pitch_a, 4), round(yaw_a, 4)],
        },
    }
    header = (
        f"# Checkerboard pose in the robot base frame ({active_robot}'s lbr_link_0),\n"
        f"# computed by src/calibration/capture_board_pose_data.py from\n"
        f"# {len(samples)} samples (pooled across whichever wrist RealSense(s) were used).\n"
        f"#\n"
        f"# base_board_robot_a_frame is the SAME checkerboard pose, re-expressed in\n"
        f"# robot_a's base frame (the global reference -- see config/robot_bases.yaml).\n"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(header + yaml.safe_dump(out_data, sort_keys=False))
    print(f"Wrote computed board pose to: {out_path}")

    log_checkerboard_transform({
        "active_robot": active_robot,
        "num_samples": len(samples),
        "T_base_board": {"R": T_activeRobot_board.R.tolist(), "t": T_activeRobot_board.t.tolist()},
        "T_robotA_board": {"R": T_robotA_board_avg.R.tolist(), "t": T_robotA_board_avg.t.tolist()},
        "translation_std_m": t_std,
        "rotation_std_deg": rot_std_deg,
        "reproj_err_px_mean": reproj_mean,
    })


def _run_interactive(arms: list[str], args: argparse.Namespace) -> None:
    target = args.num_samples
    debug_dir = BOARD_POSE_DEBUG_DIR
    debug_dir.mkdir(parents=True, exist_ok=True)

    samples: list[BoardPoseSample] = []
    if args.append:
        samples = _load_board_samples_from_dir(debug_dir)
        if samples:
            print(f"Appending to {len(samples)} existing pooled board-pose sample(s) under {debug_dir}")

    T_flange_cam_by_cam = _require_handeye_done(arms)
    robot_bases_map = load_robot_bases(ROBOT_BASES_YAML)
    next_idx = max((s.idx for s in samples), default=-1) + 1

    for arm_key in arms:
        if len(samples) >= target:
            print(f"\nAlready have {len(samples)}/{target} pooled sample(s) -- skipping arm={arm_key}.")
            continue

        arm = ARM_KEYS[arm_key]
        capture_node = _CaptureNode(arm_key, publish_debug=not args.no_debug_topic)
        nodes = [capture_node]

        if args.controller == "admittance":
            print(
                f"Bringing up the admittance control loop for arm={arm_key} only (needs "
                "`ros2 launch lbr_dual_arm_bringup admittance.launch.py` already running)..."
            )
            admittance_node = AdmittanceDualArmNode(arm_keys=(arm_key,))
            if args.gain_profile is not None:
                apply_gain_profile(admittance_node, (arm_key,), args.gain_profile)
            log_active_gains(admittance_node, (arm_key,))
            nodes.append(admittance_node)
            print(
                f"Only arm={arm_key} is software-compliant this session -- push it directly. "
                "The other arm's position controller just holds its last commanded pose."
            )
        else:
            if args.gain_profile is not None:
                print("NOTE: --gain-profile only applies to --controller admittance; ignoring.")
            if args.controller == "moveit":
                print(
                    "Prerequisite: hardware.launch.py + move_group.launch.py already running -- "
                    "jog via RViz's MotionPlanning panel (Plan & Execute)."
                )
            else:  # handguided
                print(
                    "Prerequisite: `ros2 launch lbr_dual_arm_bringup calibration.launch.py` already "
                    "running (gravity-compensation mode) -- physically hand-guide the arm."
                )

        executor = MultiThreadedExecutor()
        for n in nodes:
            executor.add_node(n)
        spin_thread = threading.Thread(target=executor.spin, daemon=True)
        spin_thread.start()

        try:
            remaining = target - len(samples)
            new_samples, next_idx = _prompt_capture_loop_single_arm(
                capture_node, remaining, next_idx, debug_dir,
                T_flange_cam_by_cam[arm["cam_id"]], robot_bases_map[arm["robot_base_key"]],
                controller_label=args.controller, pooled_total=target, already=len(samples),
            )
            samples.extend(new_samples)
        finally:
            executor.shutdown()
            for n in nodes:
                n.destroy_node()

    print(f"\nDone capturing. {len(samples)} pooled board-pose sample(s) saved under {debug_dir}.")
    if len(samples) < 2:
        raise RuntimeError(
            f"Only {len(samples)} pooled sample(s) -- need >= 2 to average a board pose. "
            f"Re-run (with --append) to add more."
        )

    _average_and_write(samples)


def _run_from_handeye(arms: list[str], args: argparse.Namespace) -> None:
    """No robot motion, no camera: recomputes T_base_board directly from
    Stage A's already-saved outputs/calibration_debug/handeye/<cam_id>/
    sample_*.json (each already has T_base_flange + T_cam_board -- see
    module docstring's --source handeye section)."""
    if args.controller != "moveit" or args.gain_profile is not None:
        print("NOTE: --controller/--gain-profile are ignored with --source handeye (no robot motion).")
    if args.no_debug_topic:
        print("NOTE: --no-debug-topic has no effect with --source handeye (no debug topic published).")

    debug_dir = BOARD_POSE_DEBUG_DIR
    debug_dir.mkdir(parents=True, exist_ok=True)

    samples: list[BoardPoseSample] = []
    if args.append:
        samples = _load_board_samples_from_dir(debug_dir)
        if samples:
            print(f"Appending to {len(samples)} existing pooled board-pose sample(s) under {debug_dir}")

    T_flange_cam_by_cam = _require_handeye_done(arms)
    robot_bases_map = load_robot_bases(ROBOT_BASES_YAML)
    next_idx = max((s.idx for s in samples), default=-1) + 1

    for arm_key in arms:
        arm = ARM_KEYS[arm_key]
        handeye_dir = HANDEYE_DEBUG_DIR / arm["cam_id"]
        handeye_samples = _load_handeye_samples_from_dir(handeye_dir)
        if not handeye_samples:
            print(f"  [skip] no saved hand-eye samples under {handeye_dir} for arm={arm_key}.")
            continue

        T_flange_cam = T_flange_cam_by_cam[arm["cam_id"]]
        T_robotA_armBase = robot_bases_map[arm["robot_base_key"]]
        print(
            f"\narm={arm_key}: pooling all {len(handeye_samples)} saved hand-eye sample(s) "
            f"under {handeye_dir} (idx={[hs.idx for hs in handeye_samples]})"
        )

        for hs in handeye_samples:
            T_armBase_board = hs.T_base_flange @ T_flange_cam @ hs.T_cam_board
            T_robotA_board = T_robotA_armBase @ T_armBase_board
            sample = BoardPoseSample(idx=next_idx, T_base_board=T_robotA_board, reproj_px=hs.reproj_px)
            _save_board_sample_json(debug_dir, sample)
            src_png = handeye_dir / f"sample_{hs.idx:02d}.png"
            if src_png.exists():
                shutil.copy2(src_png, debug_dir / f"sample_{next_idx:02d}_{arm_key}.png")
            samples.append(sample)
            next_idx += 1

    print(f"\nDone. {len(samples)} pooled board-pose sample(s) sourced from existing hand-eye captures.")
    if len(samples) < 2:
        raise RuntimeError(
            f"Only {len(samples)} pooled sample(s) across {arms} -- need >= 2 to average a board "
            f"pose. Run capture_handeye_data.py for more arm(s)/samples first."
        )

    _average_and_write(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--source", choices=SOURCES, default="live",
        help="'live': jog the real arm(s) and capture fresh images (needs the robot/camera stack "
             "running). 'handeye': no robot motion -- recompute from Stage A's already-saved "
             "outputs/calibration_debug/handeye/<cam_id>/ detections instead (see module docstring).",
    )
    parser.add_argument("--arm", choices=("left", "right", "both"), default="both")
    parser.add_argument("--controller", choices=CONTROLLERS, default="moveit")
    parser.add_argument(
        "--num-samples", type=int, default=DEFAULT_NUM_SAMPLES,
        help="Pooled total across whichever arm(s) are used (see module docstring).",
    )
    parser.add_argument(
        "--append", action="store_true",
        help="Extend the existing pooled sample set under outputs/calibration_debug/board_pose/dual/ "
             "instead of archiving it first (only valid if the checkerboard has NOT moved since the "
             "archived data was captured).",
    )
    parser.add_argument(
        "--gain-profile", default=None, choices=("holding", "insertion"),
        help="--controller admittance only -- see config/admittance_gain_profiles.yaml.",
    )
    parser.add_argument("--no-debug-topic", action="store_true")
    args = parser.parse_args()

    arms = ["left", "right"] if args.arm == "both" else [args.arm]

    # (canonical_path, backup_path) pairs -- restored automatically if the
    # run is aborted before a fresh config/base_board_pose.yaml is actually
    # written (see _restore_untouched_backups; same policy as
    # capture_handeye_data.py's --mode interactive).
    archived: list[tuple[Path, Path]] = []
    if not args.append:
        backup = _archive_if_exists(BOARD_POSE_DEBUG_DIR)
        if backup:
            print(f"Archived existing {BOARD_POSE_DEBUG_DIR} -> {backup}")
            archived.append((BOARD_POSE_DEBUG_DIR, backup))
        backup = _archive_if_exists(BASE_BOARD_YAML)
        if backup:
            print(f"Archived existing {BASE_BOARD_YAML} -> {backup}")
            archived.append((BASE_BOARD_YAML, backup))

    if args.source == "handeye":
        # No robot/camera involved -- plain file I/O + linear algebra, no
        # rclpy node needed at all.
        try:
            _run_from_handeye(arms, args)
        except BaseException:
            _restore_untouched_backups(archived)
            raise
        return

    rclpy.init()
    try:
        _run_interactive(arms, args)
    except BaseException:
        _restore_untouched_backups(archived)
        raise
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
