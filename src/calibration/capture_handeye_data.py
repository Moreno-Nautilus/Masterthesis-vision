"""Stage A of dual-arm hand-eye calibration: capture flange poses AND
checkerboard images/detections, with the positioning method, arm(s), and
capture mode all picked via flags -- not separate scripts/launch files.

Replaces capture_flange_poses_dual.py / _handguided.py / _admittance.py
(three ~85%-identical scripts, one per controller) with a single entry
point. See docs/calibration_cheatsheet.md and docs/calibration_control_modes.md
for the full walkthrough; this docstring covers the flag semantics.

--arm {left,right,both} (default: both)
    "both" runs the capture loop for left, then right, WITHIN THIS ONE
    PROCESS -- one command instead of two terminals. For --controller
    admittance this stays one-arm-at-a-time under the hood regardless
    (running two AdmittanceController instances concurrently was measured
    to roughly halve the achievable control-loop rate for the arm actually
    being guided -- see admittance_dual_arm.py); for moveit/handguided
    there's no such hardware constraint, but the interaction is still
    sequential per arm for a consistent UX (one checkerboard debug view,
    one prompt at a time). "both" is a process/UX consolidation, not a
    claim of true simultaneous motion.

--controller {moveit,admittance,handguided} (default: moveit)
    How the arm gets positioned before each --mode interactive capture:
      moveit      -- no active control loop here; jog via RViz's
                      MotionPlanning panel (Plan & Execute). Needs
                      `ros2 launch lbr_dual_arm_bringup hardware.launch.py`
                      + `move_group.launch.py` already up.
      admittance  -- this script runs AdmittanceDualArmNode itself (see
                      admittance_dual_arm.py), scoped to ONLY the arm
                      currently being captured, in a background thread
                      alongside the interactive prompt. Push the arm
                      directly. Needs
                      `ros2 launch lbr_dual_arm_bringup admittance.launch.py`.
      handguided  -- no active control loop (compliance lives in the
                      gravity_compensation_* hardware controller); push the
                      arm directly. Needs
                      `ros2 launch lbr_dual_arm_bringup calibration.launch.py`.
    Ignored (with a note printed) when --mode replay -- replay always drives
    via MoveIt joint-space goals regardless.

--mode {interactive,replay,augment} (default: interactive)
    interactive -- operator positions the arm (per --controller above),
        presses Enter, script captures. This is how you build up the
        initial pose set.
    replay -- no manual positioning. Drives the arm via
        DualArmMoveitClient.move_to_joint() (joint-space, exact replay) to
        each ALREADY-SAVED config/flange_poses/<arm>.json configuration
        (both_arms_flange simultaneously when --arm both and both arms have
        an equal number of saved captures, single-arm otherwise) and
        re-captures a fresh checkerboard detection at each stop. This is
        for "the checkerboard moved a little since the last session, but
        the robot poses that used to see it should still roughly work" --
        it recaptures the image/detection half of the data without
        re-doing the manual positioning, and without touching
        config/flange_poses/<arm>.json (the poses themselves didn't
        change). Needs `move_group.launch.py` up and an existing
        interactive-mode session's joint_positions to replay from. Needs
        `ros2 launch lbr_dual_arm_bringup hardware.launch.py` +
        `move_group.launch.py` up (same as --controller moveit).
    augment -- STANDARD way to grow a small interactive-mode pose set into a
        well-conditioned one, no manual positioning. APPEND-ONLY: every
        already-saved pose for the arm is kept in config/flange_poses/<arm>.json
        no matter what happens this run -- nothing is ever removed,
        renumbered, or randomly downsampled, even if --target-samples is
        below the known count or a known pose stops detecting the
        checkerboard. Per arm: (1) replay every already-saved pose exactly
        like --mode replay and re-detect the checkerboard at each; a hit
        refreshes that pose's debug hand-eye sample in place (same idx --
        the checkerboard may have moved since first capture) and becomes an
        anchor for step 2, a miss just leaves that pose/sample untouched;
        (2) if the known count is still below --target-samples, repeatedly
        perturb the MOST RECENTLY accepted anchor pose (the last known-good
        one first, then whichever perturbation was just accepted -- NOT a
        random pick across the whole anchor pool, so consecutive IK targets
        stay close together and don't force large joint-space swings /
        near-limit detours on this redundant 7-DOF arm) by a small random
        Cartesian offset (+/-3cm per translation axis, +/-7deg per
        rotation axis -- see pose_augmentation.py), drive there via IK
        seeded from the arm's current joint state
        (DualArmMoveitClient.move_to_seeded -- see that method's docstring
        for why: MoveGroup's own Cartesian-goal IK sampler seeds each
        attempt from a RANDOM state, which can pick a completely different
        elbow/null-space solution for this redundant 7-DOF arm even for a
        tiny Cartesian offset, producing a large, visually pointless
        joint-space move; seeding from the current state instead keeps the
        IK solution -- and therefore the executed motion -- close to where
        the arm already is), and attempt a detection; a hit is appended
        with a fresh idx (added to the anchor pool too), a miss is
        discarded and a new perturbation is drawn, until --target-samples
        total is reached (or the attempt budget runs out) -- if the known
        count already meets or exceeds --target-samples, this step is
        skipped entirely. Both config/flange_poses/<arm>.json and
        outputs/calibration_debug/handeye/<cam_id>/ are saved after every
        appended sample (never lose a sample to a crash); the pre-run
        content is snapshotted (copied, not moved) up front purely as an
        extra backup, since the live files are never at risk of going
        missing. Same prerequisites as replay. Works with however many arms
        are actually connected -- pass --arm left/right/both as
        appropriate; each arm's augmentation is independent.

What gets saved, every accepted sample, both modes:
  1. config/flange_poses/<arm>.json (flange_pose_store.FlangePoseCapture --
     T_armBase_flange + joint_positions). --mode interactive only; replay
     doesn't rewrite this since the pose itself is unchanged.
  2. outputs/calibration_debug/handeye/<cam_id>/sample_NN.json + .png -- the
     HandEyeSample schema from handeye_flange_cam_realsense.py (T_base_flange,
     T_cam_board, reproj_px, corners_px, K). Both modes. This is what makes
     Stage B (calibrate_handeye.py) able to run completely offline, with no
     robot or camera involved -- it only ever reads this directory.
  NOTE: unlike the old capture_flange_poses_dual.py (RViz-jog, no active
  controller), joint_positions is now ALWAYS saved regardless of
  --controller -- joint_state_broadcaster publishes independently of which
  controller is active (see docs/hand_guided_calibration.md's original
  finding), so there's no reason the moveit-jogged path shouldn't have this
  too. Every capture from this script is replay-mode-eligible.

Backup-before-overwrite: unless --append, a fresh --mode interactive
session archives any existing config/flange_poses/<arm>.json and
outputs/calibration_debug/handeye/<cam_id>/ for the arm(s)/cam(s) about to
be captured to a timestamped `..._bak_<UTC-timestamp>` path before writing
anything new (never deletes data, never silently mixes it either --
AX=XB assumes every sample shares the same physical checkerboard pose, so
old-board-position and new-board-position samples must never end up in the
same active sample set). --mode replay archives the recaptured cam(s)'
sample directory unconditionally (that's the whole point of replay mode --
recapturing after the board moved) and never touches
config/flange_poses/<arm>.json (read-only input in this mode). If a replay
recapture fails at some index (checkerboard not visible, bad reprojection),
that index is simply absent from the fresh sample set rather than falling
back to the archived (now stale) one.

Run (inside the 'vision' container):
    # Default: both arms, MoveIt/RViz-jogged, 7 poses each.
    python3 -m src.calibration.capture_handeye_data

    # One arm, admittance-guided:
    python3 -m src.calibration.capture_handeye_data --arm left --controller admittance

    # Gravity-compensation hand-guiding, both arms, 10 samples each:
    python3 -m src.calibration.capture_handeye_data --controller handguided --num-samples 10

    # Checkerboard moved slightly -- recapture at the previously saved poses:
    python3 -m src.calibration.capture_handeye_data --mode replay

    # Standard: grow an existing pose set to 10 well-conditioned samples/arm:
    python3 -m src.calibration.capture_handeye_data --mode augment
    python3 -m src.calibration.capture_handeye_data --arm left --mode augment --target-samples 10
"""

from __future__ import annotations

import argparse
import shutil
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, JointState

from src.calibration.admittance_dual_arm import (
    AdmittanceDualArmNode,
    apply_gain_profile,
    log_active_gains,
)
from src.calibration.flange_pose_store import (
    ARM_KEYS,
    FLANGE_POSES_DIR,
    FlangePoseCapture,
    FlangePoseSet,
    load_pose_set,
    save_pose_set,
)
from src.calibration.handeye_flange_cam_realsense import (
    MAX_REPROJ_ERR_PX,
    RGB_INFO_MAX_DT_S,
    HandEyeSample,
    _camera_topics,
    _draw_chessboard,
    _img_to_numpy_bgr,
    _K_from_camerainfo,
    _rgb_numpy_to_imgmsg,
    _save_sample_json,
    _solve_board_pose,
    _stamp_to_sec,
)
from src.calibration.moveit_dual_arm import ArmTarget, DEFAULT_MOVE_GROUP_NAMESPACE, DualArmMoveitClient, JointTarget
from src.calibration.pose_augmentation import PoseAugmentationConfig, sample_augmented_pose
from src.perception.ros.multicam_grabber_realsense import _pose_msg_to_se3
from src.perception.ros.qos_profiles import qos_profile_sensor_data_low_latency
from src.utils.se3 import SE3

CONTROLLERS = ("moveit", "admittance", "handguided")
MODES = ("interactive", "replay", "augment")

FLANGE_POSE_MAX_AGE_S = 0.25
JOINT_STATE_MAX_AGE_S = 0.25
RECOMMENDED_SAMPLES = 7
DEFAULT_TARGET_SAMPLES = 10   # --mode augment default (see docstring)
HANDEYE_DEBUG_DIR = Path("outputs/calibration_debug/handeye")

# joint_state_broadcaster's namespace -- matches lbr_dual_arm_bringup's
# default robot_name across every bring-up mode (hardware/calibration/admittance).
JOINT_STATES_TOPIC = "/lbr_dual_arm/joint_states"

# Interactive: how long to let a background executor thread process fresh
# messages after a capture request before checking freshness. Replay: how
# long to let the arm settle after a MoveGroup goal completes before
# capturing.
SETTLE_WAIT_S = 2.0
REPLAY_SETTLE_S = 1.5


def _arm_joint_prefix(arm_key: str) -> str:
    # ARM_KEYS["left"]["base_frame"] == "lbr_one_link_0" -> "lbr_one_A"
    # (matches joint names lbr_one_A1..A7).
    base_frame = ARM_KEYS[arm_key]["base_frame"]
    robot_prefix = base_frame.rsplit("_link_0", 1)[0]
    return f"{robot_prefix}_A"


def _archive_if_exists(path: Path) -> Optional[Path]:
    """Moves an existing file/dir aside to a timestamped backup path and
    returns the backup path, or returns None if nothing existed to move.

    Only for call sites where discarding the live data is the explicit,
    immediate intent (--mode interactive without --append: the user is
    deliberately starting that arm's captures over). Everywhere else, prefer
    _snapshot_if_exists (copy, not move) so the live file is never at risk
    of being left missing -- see _restore_untouched_backups for the
    generic safety net around uses of this function specifically."""
    if not path.exists():
        return None
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = f"_bak_{ts}" if path.is_dir() else f".bak_{ts}"
    backup = path.with_name(path.name + suffix) if path.is_dir() else path.with_name(path.stem + f"_bak_{ts}" + path.suffix)
    shutil.move(str(path), str(backup))
    return backup


def _snapshot_if_exists(path: Path) -> Optional[Path]:
    """Copies (never moves) an existing file/dir aside to a timestamped
    backup path, leaving the original in place, and returns the backup path
    (or None if nothing existed to copy). Used where the live data must
    never go missing mid-run, e.g. --mode augment's append-only rewrite."""
    if not path.exists():
        return None
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if path.is_dir():
        backup = path.with_name(path.name + f"_bak_{ts}")
        shutil.copytree(path, backup)
    else:
        backup = path.with_name(path.stem + f"_bak_{ts}" + path.suffix)
        shutil.copy2(path, backup)
    return backup


def _restore_untouched_backups(archived: list[tuple[Path, Path]]) -> None:
    """Moves each _archive_if_exists backup back to its canonical path if
    that path ended up missing or empty -- i.e. this run was interrupted
    before the arm/cam it belongs to was ever reached, so archiving it
    turned out to be premature and must be undone rather than leaving a
    silently-missing config/flange_poses/*.json or an empty debug dir.

    Added after a run aborted (Ctrl-C) while still processing the left arm
    left config/flange_poses/right.json archived-and-never-recreated, even
    though right hadn't been touched yet -- both arms get archived up front
    in main() before either is actually processed."""
    for canonical, backup in archived:
        if not backup.exists():
            continue
        empty_dir = canonical.is_dir() and not any(canonical.iterdir())
        if canonical.exists() and not empty_dir:
            continue  # this run actually wrote something here -- leave it
        if canonical.exists():
            shutil.rmtree(canonical) if canonical.is_dir() else canonical.unlink()
        shutil.move(str(backup), str(canonical))
        print(f"Restored {canonical} <- {backup} (never reached this run)")


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------


class _CaptureNode(Node):
    """One arm's live image/flange-pose/joint-state subscriptions, shared by
    all three --controller values -- joint_state_broadcaster and
    flange_pose_publisher run independently of which controller is active."""

    def __init__(self, arm_key: str, publish_debug: bool):
        arm = ARM_KEYS[arm_key]
        super().__init__(f"capture_handeye_data_{arm_key}")
        self.arm_key = arm_key
        self.cam_id = arm["cam_id"]
        self.joint_prefix = _arm_joint_prefix(arm_key)
        rgb_topic, info_topic = _camera_topics(self.cam_id)

        self.img_msg: Optional[Image] = None
        self.img_t: Optional[float] = None
        self.info_t: Optional[float] = None
        self.K: Optional[np.ndarray] = None

        self.flange_pose: Optional[SE3] = None
        self.flange_pose_wall_t: float = 0.0

        self.joint_positions: dict[str, float] = {}
        self.joint_state_wall_t: float = 0.0

        self.create_subscription(Image, rgb_topic, self._on_img, qos_profile_sensor_data_low_latency)
        self.create_subscription(CameraInfo, info_topic, self._on_info, qos_profile_sensor_data_low_latency)
        self.create_subscription(
            PoseStamped, arm["flange_pose_topic"], self._on_flange, qos_profile_sensor_data_low_latency
        )
        self.create_subscription(JointState, JOINT_STATES_TOPIC, self._on_joint_state, 10)

        self._debug_pub = None
        if publish_debug:
            self._debug_pub = self.create_publisher(
                Image, f"/calibration/capture_handeye_data/{arm_key}/debug_image", 1
            )

        self.get_logger().info(
            f"[{arm_key}] cam={self.cam_id} rgb={rgb_topic} info={info_topic} "
            f"flange_pose={arm['flange_pose_topic']} joint_states={JOINT_STATES_TOPIC} "
            f"(joint_prefix={self.joint_prefix}*)"
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

    def _on_joint_state(self, msg: JointState) -> None:
        this_arm = {
            name: pos for name, pos in zip(msg.name, msg.position) if name.startswith(self.joint_prefix)
        }
        if this_arm:
            self.joint_positions = this_arm
            self.joint_state_wall_t = time.time()

    def has_fresh_flange_pose(self) -> bool:
        if self.flange_pose is None:
            return False
        return (time.time() - self.flange_pose_wall_t) <= FLANGE_POSE_MAX_AGE_S

    def has_fresh_joint_state(self) -> bool:
        if not self.joint_positions:
            return False
        return (time.time() - self.joint_state_wall_t) <= JOINT_STATE_MAX_AGE_S

    def has_fresh_rgb_info(self) -> bool:
        if self.img_msg is None or self.K is None or self.img_t is None or self.info_t is None:
            return False
        return abs(self.img_t - self.info_t) <= RGB_INFO_MAX_DT_S

    def publish_debug(self, img_bgr: np.ndarray) -> None:
        if self._debug_pub is None:
            return
        self._debug_pub.publish(_rgb_numpy_to_imgmsg(img_bgr, self.cam_id, self.get_clock().now().to_msg()))


def _try_capture(
    node: _CaptureNode, arm_key: str, idx: int
) -> Optional[tuple[FlangePoseCapture, HandEyeSample, np.ndarray]]:
    if not node.has_fresh_rgb_info():
        print("  [skip] no fresh RGB/CameraInfo yet")
        return None
    if not node.has_fresh_flange_pose():
        print(
            f"  [skip] no fresh flange pose (< {FLANGE_POSE_MAX_AGE_S}s old) on "
            f"{ARM_KEYS[arm_key]['flange_pose_topic']} -- is lbr_dual_arm_bringup running?"
        )
        return None
    if not node.has_fresh_joint_state():
        print(
            f"  [skip] no fresh joint state (< {JOINT_STATE_MAX_AGE_S}s old) on "
            f"{JOINT_STATES_TOPIC} with prefix {node.joint_prefix}*"
        )
        return None

    img = _img_to_numpy_bgr(node.img_msg)
    K = node.K.copy()
    T_armBase_flange = node.flange_pose
    joint_positions = dict(node.joint_positions)

    try:
        T_cam_board, corners, reproj_err = _solve_board_pose(img, K)
    except RuntimeError as e:
        print(f"  [skip] checkerboard not visible to {node.cam_id}: {e}")
        node.publish_debug(img)
        return None

    if reproj_err > MAX_REPROJ_ERR_PX:
        print(f"  [skip] reprojection error too high: {reproj_err:.3f}px > {MAX_REPROJ_ERR_PX}px")
        return None

    vis = _draw_chessboard(img, corners, f"{arm_key} sample {idx} reproj={reproj_err:.3f}px")
    node.publish_debug(vis)

    flange_capture = FlangePoseCapture(
        idx=idx,
        T_armBase_flange=T_armBase_flange,
        captured_at_unix_s=time.time(),
        note=f"board_reproj_px={reproj_err:.3f}",
        joint_positions=joint_positions,
    )
    handeye_sample = HandEyeSample(
        idx=idx, T_base_flange=T_armBase_flange, T_cam_board=T_cam_board, reproj_px=reproj_err,
        corners_px=corners, K=K,
    )
    return flange_capture, handeye_sample, vis


def _prompt_capture_loop(
    capture_node: _CaptureNode,
    arm_key: str,
    num_samples: int,
    existing: list[FlangePoseCapture],
    cam_debug_dir: Path,
    controller_label: str,
) -> list[FlangePoseCapture]:
    arm = ARM_KEYS[arm_key]
    captures = list(existing)
    start_idx = len(captures)

    print(f"\n=== {controller_label} capture: arm={arm_key} (cam={arm['cam_id']}) ===")
    print(f"Need {num_samples} accepted sample(s) this run.")
    print(f"Foxglove: Image panel -> /calibration/capture_handeye_data/{arm_key}/debug_image")
    print(f"          Raw Messages panel -> {arm['flange_pose_topic']}")
    print("For each sample: position THIS arm so the checkerboard is fully visible to its")
    print("wrist RealSense, let it settle, then press Enter here to capture. Vary orientation,")
    print("not just position, across samples. Type 'q' + Enter to stop early once you have")
    print("at least 1 sample.\n")

    n_this_run = 0
    while n_this_run < num_samples:
        idx = start_idx + n_this_run
        user_in = input(
            f"[{n_this_run}/{num_samples}] arm={arm_key} Press Enter to capture (or 'q' to finish early): "
        )
        if user_in.strip().lower() == "q":
            if not captures:
                print("Need at least 1 sample. Keep going.")
                continue
            break

        time.sleep(SETTLE_WAIT_S)

        result = _try_capture(capture_node, arm_key, idx)
        if result is None:
            continue
        flange_capture, handeye_sample, vis = result
        captures.append(flange_capture)
        n_this_run += 1

        cv2.imwrite(str(cam_debug_dir / f"sample_{idx:02d}.png"), vis)
        _save_sample_json(cam_debug_dir, handeye_sample)

        pose_set = FlangePoseSet(
            arm_key=arm_key, cam_id=arm["cam_id"], base_frame=arm["base_frame"],
            flange_frame=arm["flange_frame"], captures=captures,
        )
        out_path = save_pose_set(pose_set)
        print(
            f"  [ok] reproj={handeye_sample.reproj_px:.3f}px  t={flange_capture.T_armBase_flange.t}  "
            f"saved -> {out_path}, {cam_debug_dir}/sample_{idx:02d}.json ({len(captures)} total)"
        )

    print(f"\nDone. {len(captures)} flange pose(s) + hand-eye sample(s) saved for arm={arm_key}.")
    if len(captures) < 5:
        print(
            f"NOTE: fewer than 5 samples ({len(captures)}) -- calibrate_handeye.py needs >= 5 "
            f"well-conditioned (rotationally varied) samples per arm to solve reliably."
        )
    return captures


def _run_interactive(arms: list[str], args: argparse.Namespace) -> None:
    for arm_key in arms:
        arm = ARM_KEYS[arm_key]
        cam_debug_dir = HANDEYE_DEBUG_DIR / arm["cam_id"]
        cam_debug_dir.mkdir(parents=True, exist_ok=True)

        existing: list[FlangePoseCapture] = []
        if args.append:
            try:
                existing = load_pose_set(arm_key).captures
                print(f"Appending to {len(existing)} existing saved capture(s) for arm={arm_key}")
            except FileNotFoundError:
                pass

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
            _prompt_capture_loop(
                capture_node, arm_key, args.num_samples, existing, cam_debug_dir,
                controller_label=args.controller,
            )
        finally:
            executor.shutdown()
            for n in nodes:
                n.destroy_node()


# ---------------------------------------------------------------------------
# Replay mode
# ---------------------------------------------------------------------------


@dataclass
class _ArmCamState:
    arm_key: str
    cam_id: str
    flange_pose_topic: str
    img_msg: Optional[Image] = None
    img_t: Optional[float] = None
    info_t: Optional[float] = None
    K: Optional[np.ndarray] = None
    flange_pose: Optional[SE3] = None
    flange_pose_wall_t: float = 0.0

    def has_fresh_rgb_info(self) -> bool:
        if self.img_msg is None or self.K is None or self.img_t is None or self.info_t is None:
            return False
        return abs(self.img_t - self.info_t) <= RGB_INFO_MAX_DT_S

    def has_fresh_flange_pose(self) -> bool:
        if self.flange_pose is None:
            return False
        return (time.time() - self.flange_pose_wall_t) <= FLANGE_POSE_MAX_AGE_S


class _ReplayNode(Node):
    """Subscribes each requested arm's wrist RealSense + ee_pose, hosts the
    MoveGroup client used to drive back to previously-saved joint configs."""

    def __init__(self, arms: list[str], publish_debug: bool, robot_namespace: str):
        super().__init__("capture_handeye_data_replay")
        self.arms: dict[str, _ArmCamState] = {}
        self._debug_pubs: dict[str, object] = {}

        for arm_key in arms:
            arm = ARM_KEYS[arm_key]
            state = _ArmCamState(arm_key=arm_key, cam_id=arm["cam_id"], flange_pose_topic=arm["flange_pose_topic"])
            self.arms[arm_key] = state

            rgb_topic, info_topic = _camera_topics(arm["cam_id"])
            self.create_subscription(
                Image, rgb_topic, lambda msg, k=arm_key: self._on_img(k, msg), qos_profile_sensor_data_low_latency
            )
            self.create_subscription(
                CameraInfo, info_topic, lambda msg, k=arm_key: self._on_info(k, msg),
                qos_profile_sensor_data_low_latency,
            )
            self.create_subscription(
                PoseStamped, arm["flange_pose_topic"], lambda msg, k=arm_key: self._on_flange(k, msg),
                qos_profile_sensor_data_low_latency,
            )
            if publish_debug:
                self._debug_pubs[arm_key] = self.create_publisher(
                    Image, f"/calibration/capture_handeye_data/{arm_key}/debug_image", 1
                )
            self.get_logger().info(f"[{arm_key}] cam={arm['cam_id']} rgb={rgb_topic} info={info_topic}")

        self.moveit = DualArmMoveitClient(self, namespace=robot_namespace)

    def _on_img(self, arm_key: str, msg: Image) -> None:
        st = self.arms[arm_key]
        st.img_msg = msg
        st.img_t = _stamp_to_sec(msg.header.stamp)

    def _on_info(self, arm_key: str, msg: CameraInfo) -> None:
        st = self.arms[arm_key]
        st.info_t = _stamp_to_sec(msg.header.stamp)
        st.K = _K_from_camerainfo(msg)

    def _on_flange(self, arm_key: str, msg: PoseStamped) -> None:
        st = self.arms[arm_key]
        st.flange_pose = _pose_msg_to_se3(msg)
        st.flange_pose_wall_t = time.time()

    def publish_debug(self, arm_key: str, img_bgr: np.ndarray) -> None:
        pub = self._debug_pubs.get(arm_key)
        if pub is None:
            return
        pub.publish(_rgb_numpy_to_imgmsg(img_bgr, self.arms[arm_key].cam_id, self.get_clock().now().to_msg()))

    def spin_briefly(self, duration_s: float) -> None:
        deadline = time.time() + duration_s
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)


def _recapture_at(node: _ReplayNode, arm_key: str, cap: FlangePoseCapture, debug_dir: Path) -> None:
    st = node.arms[arm_key]
    if not st.has_fresh_rgb_info():
        print(f"  [skip:{arm_key}] no fresh RGB/CameraInfo")
        return
    if not st.has_fresh_flange_pose():
        print(f"  [skip:{arm_key}] no fresh flange pose on {st.flange_pose_topic}")
        return

    img = _img_to_numpy_bgr(st.img_msg)
    K = st.K.copy()
    try:
        T_cam_board, corners, reproj_err = _solve_board_pose(img, K)
    except RuntimeError as e:
        print(f"  [skip:{arm_key}] checkerboard not visible: {e}")
        node.publish_debug(arm_key, img)
        return
    if reproj_err > MAX_REPROJ_ERR_PX:
        print(f"  [skip:{arm_key}] reprojection error too high: {reproj_err:.3f}px")
        return

    vis = _draw_chessboard(img, corners, f"{arm_key} idx={cap.idx} reproj={reproj_err:.3f}px (replay)")
    node.publish_debug(arm_key, vis)
    cv2.imwrite(str(debug_dir / f"sample_{cap.idx:02d}.png"), vis)

    sample = HandEyeSample(
        idx=cap.idx, T_base_flange=st.flange_pose, T_cam_board=T_cam_board, reproj_px=reproj_err,
        corners_px=corners, K=K,
    )
    _save_sample_json(debug_dir, sample)
    print(f"  [ok:{arm_key}] reproj={reproj_err:.3f}px  saved -> {debug_dir}/sample_{cap.idx:02d}.json")


def _replay_single(node: _ReplayNode, arm_key: str, cap: FlangePoseCapture, debug_dir: Path) -> None:
    print(f"\nmoving {arm_key} to saved idx={cap.idx}...")
    target = JointTarget(
        group_name=ARM_KEYS[arm_key]["group_name"], joint_positions=cap.joint_positions,
        label=f"{arm_key} idx={cap.idx}",
    )
    ok, _record = node.moveit.move_to_joint([target])
    if not ok:
        print(f"  [skip:{arm_key}] MoveGroup failed to reach saved idx={cap.idx}")
        return
    node.spin_briefly(REPLAY_SETTLE_S)
    _recapture_at(node, arm_key, cap, debug_dir)


def _replay_pair(
    node: _ReplayNode, left_cap: FlangePoseCapture, right_cap: FlangePoseCapture, debug_dirs: dict[str, Path]
) -> None:
    print(f"\nmoving both arms simultaneously to saved idx left={left_cap.idx} right={right_cap.idx}...")
    targets = [
        JointTarget(
            group_name=ARM_KEYS["left"]["group_name"], joint_positions=left_cap.joint_positions,
            label=f"left idx={left_cap.idx}",
        ),
        JointTarget(
            group_name=ARM_KEYS["right"]["group_name"], joint_positions=right_cap.joint_positions,
            label=f"right idx={right_cap.idx}",
        ),
    ]
    ok, _record = node.moveit.move_to_joint(targets, group_name="both_arms_flange")
    if not ok:
        print("  [skip] MoveGroup failed to reach this pose pair -- skipping recapture for both arms.")
        return
    node.spin_briefly(REPLAY_SETTLE_S)
    _recapture_at(node, "left", left_cap, debug_dirs["left"])
    _recapture_at(node, "right", right_cap, debug_dirs["right"])


def _run_replay(arms: list[str], args: argparse.Namespace) -> None:
    per_arm_captures: dict[str, list[FlangePoseCapture]] = {}
    for arm_key in arms:
        pose_set = load_pose_set(arm_key)
        missing = [c.idx for c in pose_set.captures if not c.joint_positions]
        if missing:
            raise RuntimeError(
                f"{arm_key}: capture idx={missing} have no saved joint_positions -- --mode replay "
                f"drives to saved poses via joint-space MoveGroup goals, so every saved capture "
                f"needs joint_positions. Run --mode interactive first for this arm."
            )
        per_arm_captures[arm_key] = pose_set.captures

    debug_dirs: dict[str, Path] = {}
    for arm_key in arms:
        d = HANDEYE_DEBUG_DIR / ARM_KEYS[arm_key]["cam_id"]
        d.mkdir(parents=True, exist_ok=True)
        debug_dirs[arm_key] = d

    node = _ReplayNode(arms, publish_debug=not args.no_debug_topic, robot_namespace=args.robot_namespace)
    print("Waiting for /move_action (MoveGroup) action server...")
    if not node.moveit.wait_for_server(timeout_s=30.0):
        raise RuntimeError(
            "MoveGroup action server not available -- is lbr_dual_arm_bringup's "
            "move_group.launch.py running?"
        )

    try:
        if len(arms) == 2:
            left_caps, right_caps = per_arm_captures["left"], per_arm_captures["right"]
            n_pairs = min(len(left_caps), len(right_caps))
            if len(left_caps) != len(right_caps):
                print(
                    f"NOTE: left has {len(left_caps)} saved captures, right has {len(right_caps)} -- "
                    f"pairing the first {n_pairs} simultaneously; the rest replay single-arm after."
                )
            for i in range(n_pairs):
                _replay_pair(node, left_caps[i], right_caps[i], debug_dirs)
            for arm_key, caps in (("left", left_caps[n_pairs:]), ("right", right_caps[n_pairs:])):
                for cap in caps:
                    _replay_single(node, arm_key, cap, debug_dirs[arm_key])
        else:
            arm_key = arms[0]
            for cap in per_arm_captures[arm_key]:
                _replay_single(node, arm_key, cap, debug_dirs[arm_key])
        print("\nDone recapturing.")
    finally:
        node.destroy_node()


# ---------------------------------------------------------------------------
# Augment mode
# ---------------------------------------------------------------------------


def _detect_at_live_pose(
    node: "_ReplayNode", arm_key: str, idx: int, note_prefix: str,
) -> Optional[tuple[FlangePoseCapture, HandEyeSample, np.ndarray]]:
    """Attempts one checkerboard detection at the arm's CURRENT live pose
    (already moved there by the caller) -- shared by both halves of --mode
    augment (replaying known poses, and the newly-perturbed ones), unlike
    _try_capture (interactive mode, needs joint_state freshness too) or
    _recapture_at (replay mode, doesn't return anything to the caller)."""
    st = node.arms[arm_key]
    if not st.has_fresh_rgb_info():
        print("  [skip] no fresh RGB/CameraInfo")
        return None
    if not st.has_fresh_flange_pose():
        print(f"  [skip] no fresh flange pose on {st.flange_pose_topic}")
        return None

    img = _img_to_numpy_bgr(st.img_msg)
    K = st.K.copy()
    try:
        T_cam_board, corners, reproj_err = _solve_board_pose(img, K)
    except RuntimeError as e:
        print(f"  [skip] checkerboard not visible to {st.cam_id}: {e}")
        node.publish_debug(arm_key, img)
        return None
    if reproj_err > MAX_REPROJ_ERR_PX:
        print(f"  [skip] reprojection error too high: {reproj_err:.3f}px > {MAX_REPROJ_ERR_PX}px")
        return None

    vis = _draw_chessboard(img, corners, f"{arm_key} {note_prefix} idx={idx} reproj={reproj_err:.3f}px")
    node.publish_debug(arm_key, vis)

    joint_prefix = _arm_joint_prefix(arm_key)
    joint_positions = {
        name: pos for name, pos in node.moveit.current_joint_positions().items()
        if name.startswith(joint_prefix)
    }

    flange_capture = FlangePoseCapture(
        idx=idx, T_armBase_flange=st.flange_pose, captured_at_unix_s=time.time(),
        note=f"{note_prefix} board_reproj_px={reproj_err:.3f}", joint_positions=joint_positions,
    )
    handeye_sample = HandEyeSample(
        idx=idx, T_base_flange=st.flange_pose, T_cam_board=T_cam_board, reproj_px=reproj_err,
        corners_px=corners, K=K,
    )
    return flange_capture, handeye_sample, vis


def _run_augment(arms: list[str], args: argparse.Namespace) -> None:
    """Append-only: existing captures (config/flange_poses/<arm>.json) and
    their debug hand-eye samples (outputs/calibration_debug/handeye/<cam>/)
    are NEVER removed, renumbered, or randomly downsampled here, regardless
    of whether a known pose still detects the checkerboard this run or
    whether the known count already exceeds --target-samples. A known pose
    that still detects the board gets its debug sample refreshed in place
    (same idx -- the checkerboard may have moved since it was first
    captured); one that doesn't is left exactly as it was. New poses are
    appended with fresh idx values on top. This is why the archiving below
    is a copy (_snapshot_if_exists), not a move: the live files are never at
    risk of going missing, so there's nothing to restore on a crash/Ctrl-C
    (see the right-arm postmortem 2026-08-14 this replaced -- the old
    archive-then-rebuild design left config/flange_poses/right.json
    deleted-and-never-recreated when the run was interrupted while still on
    the left arm)."""
    per_arm_existing: dict[str, list[FlangePoseCapture]] = {}
    for arm_key in arms:
        pose_set = load_pose_set(arm_key)
        if not pose_set.captures:
            raise RuntimeError(
                f"{arm_key}: no saved poses to augment from -- run --mode interactive first "
                f"(need at least 1 known-good pose to perturb around)."
            )
        missing = [c.idx for c in pose_set.captures if not c.joint_positions]
        if missing:
            raise RuntimeError(
                f"{arm_key}: capture idx={missing} have no saved joint_positions -- --mode augment "
                f"replays known poses via joint-space MoveGroup goals first, so every saved capture "
                f"needs joint_positions."
            )
        per_arm_existing[arm_key] = pose_set.captures

    for arm_key in arms:
        flange_path = FLANGE_POSES_DIR / f"{arm_key}.json"
        backup = _snapshot_if_exists(flange_path)
        if backup:
            print(f"Snapshotted existing {flange_path} -> {backup} (original left in place)")
        cam_dir = HANDEYE_DEBUG_DIR / ARM_KEYS[arm_key]["cam_id"]
        backup = _snapshot_if_exists(cam_dir)
        if backup:
            print(f"Snapshotted existing {cam_dir} -> {backup} (original left in place)")
        cam_dir.mkdir(parents=True, exist_ok=True)

    node = _ReplayNode(arms, publish_debug=not args.no_debug_topic, robot_namespace=args.robot_namespace)
    print("Waiting for /move_action (MoveGroup) action server...")
    if not node.moveit.wait_for_server(timeout_s=30.0):
        raise RuntimeError(
            "MoveGroup action server not available -- is lbr_dual_arm_bringup's "
            "move_group.launch.py running?"
        )

    cfg = PoseAugmentationConfig()
    rng = np.random.default_rng()

    try:
        for arm_key in arms:
            arm = ARM_KEYS[arm_key]
            cam_dir = HANDEYE_DEBUG_DIR / arm["cam_id"]
            known_captures = list(per_arm_existing[arm_key])
            next_idx = max((c.idx for c in known_captures), default=-1) + 1

            print(f"\n=== augment: arm={arm_key} (cam={arm['cam_id']}) -- target {args.target_samples} total samples ===")
            print(
                f"Step 1/2: verifying {len(known_captures)} known pose(s) still see the checkerboard "
                f"(append-only -- none are removed from the saved set regardless of outcome)..."
            )
            verified_bases: list[SE3] = []
            for cap in known_captures:
                target = JointTarget(
                    group_name=arm["group_name"], joint_positions=cap.joint_positions,
                    label=f"{arm_key} known idx={cap.idx}",
                )
                ok, _record = node.moveit.move_to_joint([target])
                if not ok:
                    print(f"  [warn] MoveGroup failed to reach known idx={cap.idx} -- kept as-is")
                    continue
                node.spin_briefly(REPLAY_SETTLE_S)
                result = _detect_at_live_pose(node, arm_key, idx=cap.idx, note_prefix="known-verify")
                if result is None:
                    print(f"  [warn] known idx={cap.idx} no longer detects the checkerboard -- kept as-is")
                    continue
                # Still visible -- refresh this idx's debug sample in place
                # (same idx, so it overwrites only itself) in case the
                # checkerboard moved since it was first captured. The saved
                # flange pose/joint_positions are untouched either way.
                _flange_cap, handeye_sample, vis = result
                handeye_sample.idx = cap.idx
                cv2.imwrite(str(cam_dir / f"sample_{cap.idx:02d}.png"), vis)
                _save_sample_json(cam_dir, handeye_sample)
                verified_bases.append(cap.T_armBase_flange)

            print(f"  {len(verified_bases)}/{len(known_captures)} known pose(s) still detect the checkerboard.")
            if not verified_bases:
                raise RuntimeError(
                    f"{arm_key}: none of the known poses still see the checkerboard -- can't perturb "
                    f"around zero successful poses. Re-run --mode interactive for this arm."
                )

            all_captures = list(known_captures)
            n_needed = max(0, args.target_samples - len(known_captures))
            if n_needed == 0:
                print(
                    f"Step 2/2: already have {len(known_captures)} known pose(s) >= target "
                    f"{args.target_samples} -- append-only mode never discards existing poses, "
                    f"so no perturbation needed."
                )
            else:
                print(f"Step 2/2: perturbing to append up to {n_needed} new sample(s)...")
                attempts = 0
                n_added = 0
                max_attempts = 20 * max(1, n_needed)
                while n_added < n_needed and attempts < max_attempts:
                    attempts += 1
                    candidate = sample_augmented_pose(verified_bases, rng, cfg)
                    arm_target = ArmTarget(
                        group_name=arm["group_name"], base_frame=arm["base_frame"],
                        tip_link=arm["flange_frame"], T_armBase_flange=candidate,
                    )
                    if not node.moveit.move_to_seeded(arm_target, label=f"{arm_key} augmented"):
                        continue
                    node.spin_briefly(REPLAY_SETTLE_S)
                    result = _detect_at_live_pose(node, arm_key, idx=next_idx, note_prefix="augmented")
                    if result is None:
                        continue
                    flange_capture, handeye_sample, vis = result
                    cv2.imwrite(str(cam_dir / f"sample_{next_idx:02d}.png"), vis)
                    _save_sample_json(cam_dir, handeye_sample)
                    all_captures.append(flange_capture)
                    verified_bases.append(flange_capture.T_armBase_flange)
                    # Save as soon as a new sample is appended -- crash-safe,
                    # and always a strict superset of what was on disk before.
                    save_pose_set(FlangePoseSet(
                        arm_key=arm_key, cam_id=arm["cam_id"], base_frame=arm["base_frame"],
                        flange_frame=arm["flange_frame"], captures=all_captures,
                    ))
                    n_added += 1
                    print(
                        f"  [ok] appended sample idx={next_idx} ({len(all_captures)}/{args.target_samples} "
                        f"total, reproj={handeye_sample.reproj_px:.3f}px, {attempts} attempt(s) so far) -- saved"
                    )
                    next_idx += 1

                if n_added < n_needed:
                    print(
                        f"WARNING: only appended {n_added}/{n_needed} new samples for arm={arm_key} "
                        f"after {attempts} perturbation attempts -- proceeding with {len(all_captures)} total."
                    )

            # Persist even when nothing new was appended, so any Step 1
            # in-place refreshes are reflected in the canonical file too.
            save_pose_set(FlangePoseSet(
                arm_key=arm_key, cam_id=arm["cam_id"], base_frame=arm["base_frame"],
                flange_frame=arm["flange_frame"], captures=all_captures,
            ))
            print(f"Done: {len(all_captures)} sample(s) saved -> {FLANGE_POSES_DIR / f'{arm_key}.json'}, {cam_dir}")

        print("\nDone augmenting.")
    finally:
        node.destroy_node()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", choices=("left", "right", "both"), default="both")
    parser.add_argument("--controller", choices=CONTROLLERS, default="moveit")
    parser.add_argument("--mode", choices=MODES, default="interactive")
    parser.add_argument("--num-samples", type=int, default=RECOMMENDED_SAMPLES, help="per arm, --mode interactive only")
    parser.add_argument(
        "--target-samples", type=int, default=DEFAULT_TARGET_SAMPLES,
        help="per arm, --mode augment only -- minimum total samples to reach by appending perturbed "
             "ones (known + newly appended). Never removes samples if the known count already meets "
             "or exceeds this.",
    )
    parser.add_argument(
        "--append", action="store_true",
        help="--mode interactive: extend existing data instead of archiving it first (only valid if "
             "the checkerboard has NOT moved since the archived data was captured). No effect in "
             "--mode replay/augment.",
    )
    parser.add_argument(
        "--gain-profile", default=None, choices=("holding", "insertion"),
        help="--controller admittance only -- see config/admittance_gain_profiles.yaml.",
    )
    parser.add_argument("--robot-namespace", default=DEFAULT_MOVE_GROUP_NAMESPACE, help="--mode replay/augment only")
    parser.add_argument("--no-debug-topic", action="store_true")
    args = parser.parse_args()

    arms = ["left", "right"] if args.arm == "both" else [args.arm]

    if args.mode in ("replay", "augment") and args.controller != "moveit":
        print(
            f"NOTE: --mode {args.mode} always drives via MoveIt (joint-space and/or Cartesian IK) "
            f"-- ignoring --controller {args.controller}."
        )

    # (canonical_path, backup_path) pairs from _archive_if_exists (move
    # semantics) below -- restored automatically if the run is aborted
    # before the canonical path is touched again (see
    # _restore_untouched_backups). --mode augment archives internally via
    # _snapshot_if_exists (copy, not move) instead, so it has nothing to
    # register here -- its live files are never at risk.
    archived: list[tuple[Path, Path]] = []

    if args.mode == "interactive":
        if not args.append:
            for arm_key in arms:
                flange_path = FLANGE_POSES_DIR / f"{arm_key}.json"
                backup = _archive_if_exists(flange_path)
                if backup:
                    print(f"Archived existing {flange_path} -> {backup}")
                    archived.append((flange_path, backup))
                cam_dir = HANDEYE_DEBUG_DIR / ARM_KEYS[arm_key]["cam_id"]
                backup = _archive_if_exists(cam_dir)
                if backup:
                    print(f"Archived existing {cam_dir} -> {backup}")
                    archived.append((cam_dir, backup))
    elif args.mode == "replay":
        if args.append:
            print(
                "NOTE: --append has no effect in --mode replay -- the previous hand-eye samples for "
                "the recaptured arm(s)/cam(s) are archived unconditionally below."
            )
        for arm_key in arms:
            cam_dir = HANDEYE_DEBUG_DIR / ARM_KEYS[arm_key]["cam_id"]
            backup = _archive_if_exists(cam_dir)
            if backup:
                print(f"Archived existing {cam_dir} -> {backup}")
                archived.append((cam_dir, backup))
    else:  # augment
        if args.append:
            print("NOTE: --append has no effect in --mode augment -- it always rewrites the full resulting set.")
        # _run_augment does its own (non-destructive) archiving of
        # flange_poses AND the debug dir, since it needs to read the
        # existing content before snapshotting it.

    rclpy.init()
    try:
        if args.mode == "interactive":
            _run_interactive(arms, args)
        elif args.mode == "replay":
            _run_replay(arms, args)
        else:
            _run_augment(arms, args)
    except BaseException:
        _restore_untouched_backups(archived)
        raise
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
