"""Admittance-guided dual-arm flange pose capture -- the DEFAULT Step 1
capture method (see docs/calibration_control_modes.md and
docs/calibration_cheatsheet.md). Admittance twin of
capture_flange_poses_dual_handguided.py, which remains a supported
alternative (gravity-compensation hand-guiding), as does RViz jogging
(capture_flange_poses_dual.py).

capture_flange_poses_dual_handguided.py expects the rig in gravity-
compensation mode (calibration.launch.py, torque command interface): the
arm floats at near-zero commanded torque and you push it directly. This
script instead expects the rig in software-admittance mode
(admittance.launch.py, POSITION command interface -- see
src/calibration/admittance_dual_arm.py's module docstring): the arm stays
on the position interface throughout, and this script itself runs the
AdmittanceDualArmNode control loop (measured external torque -> task-space
force -> task-space velocity -> integrated joint-position setpoint) in a
background thread for the whole capture session, so pushing on the arm
being captured makes it yield exactly like hand-guiding, without ever
switching the rig into torque mode.

ONLY the arm passed via --arm gets an admittance controller instantiated
-- NOT both arms. Each AdmittanceController does its own Jacobian
pseudo-inverse solve per incoming LBRState message; running two of them
concurrently was measured to roughly halve the achievable control-loop
rate for the arm actually being hand-guided, which on real hardware showed
up as the arm feeling rigid/unresponsive even with the node confirmed
running. The other arm's position controller just holds its last
commanded pose (controller_manager's own fixed-rate loop keeps streaming
that to FRI regardless -- no dropout risk from this node going quiet on
it). This is why the routine is sequential and one-arm-at-a-time: capture
left fully, Ctrl-C, then capture right -- safe to do as two entirely
separate launches, not just two calls within one process.

Everything else matches capture_flange_poses_dual_handguided.py exactly:
same checkerboard quality gate, same incremental-save-every-capture
reliability, same joint_positions field saved alongside the Cartesian
T_armBase_flange pose (see that script's docstring for why: this is a
redundant 7-DOF arm, so the Cartesian pose alone under-determines the elbow
configuration that was actually captured).

Why a background thread: AdmittanceDualArmNode has to keep processing
LBRState -> command/joint_position at whatever rate the state broadcaster
publishes, or the position command it last published simply holds -- since
the command interface is a rigid position controller underneath, an admittance
loop that stops spinning stops yielding entirely (unlike gravity
compensation, where the compliance lives in the hardware controller, not in
a python loop). The interactive prompt below blocks the main thread on
input() between captures, so the admittance node and the capture node are
both added to one MultiThreadedExecutor spun in a daemon thread, decoupling
"keep the arms compliant" from "wait for the operator to press Enter".

Run (inside the 'vision' container), with
`ros2 launch lbr_dual_arm_bringup admittance.launch.py use_gripper:=true`
already up, checkerboard placed and visible to whichever arm you're currently
capturing:

    python3 -m src.calibration.capture_flange_poses_dual_admittance --arm left
    python3 -m src.calibration.capture_flange_poses_dual_admittance --arm right

Optionally pick a starting compliance profile (see
config/admittance_gain_profiles.yaml -- "holding" is the default applied by
AdmittanceDualArmNode itself; "insertion" yields more readily / at a lower
force threshold, closer to "freely positionable"):

    python3 -m src.calibration.capture_flange_poses_dual_admittance --arm left --gain-profile insertion
"""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, JointState

from src.calibration.admittance_dual_arm import (
    AdmittanceDualArmNode,
    apply_gain_profile,
    log_active_gains,
)
from src.calibration.flange_pose_store import (
    ARM_KEYS,
    FlangePoseCapture,
    FlangePoseSet,
    load_pose_set,
    save_pose_set,
)
from src.calibration.handeye_flange_cam_realsense import (
    MAX_REPROJ_ERR_PX,
    RGB_INFO_MAX_DT_S,
    _camera_topics,
    _draw_chessboard,
    _img_to_numpy_bgr,
    _K_from_camerainfo,
    _rgb_numpy_to_imgmsg,
    _solve_board_pose,
    _stamp_to_sec,
)
from src.perception.ros.multicam_grabber_realsense import _pose_msg_to_se3

FLANGE_POSE_MAX_AGE_S = 0.25
JOINT_STATE_MAX_AGE_S = 0.25
RECOMMENDED_SAMPLES = 7
DEBUG_DIR = "outputs/calibration_debug/capture_flange_poses"

# joint_state_broadcaster publishes under the ros2_control namespace --
# admittance.launch.py defaults robot_name to "lbr_dual_arm" specifically to
# match this (see that launch file's docstring).
JOINT_STATES_TOPIC = "/lbr_dual_arm/joint_states"

# How long to let the background executor thread process fresh
# messages/commands after a capture request before checking freshness --
# same 2s budget capture_flange_poses_dual_handguided.py spends spinning
# inline; here the equivalent work happens continuously in the background
# thread, so this is a plain sleep.
SETTLE_WAIT_S = 2.0


def _arm_joint_prefix(arm_key: str) -> str:
    # ARM_KEYS["left"]["base_frame"] == "lbr_one_link_0" -> "lbr_one_A"
    # (matches joint names lbr_one_A1..A7).
    base_frame = ARM_KEYS[arm_key]["base_frame"]
    robot_prefix = base_frame.rsplit("_link_0", 1)[0]
    return f"{robot_prefix}_A"


class _CaptureNode(Node):
    def __init__(self, arm_key: str, publish_debug: bool):
        arm = ARM_KEYS[arm_key]
        super().__init__(f"capture_flange_poses_dual_admittance_{arm_key}")
        self.arm_key = arm_key
        self.cam_id = arm["cam_id"]
        self.joint_prefix = _arm_joint_prefix(arm_key)
        rgb_topic, info_topic = _camera_topics(self.cam_id)

        self.img_msg: Optional[Image] = None
        self.img_t: Optional[float] = None
        self.info_t: Optional[float] = None
        self.K: Optional[np.ndarray] = None

        self.flange_pose = None
        self.flange_pose_wall_t: float = 0.0

        self.joint_positions: dict[str, float] = {}
        self.joint_state_wall_t: float = 0.0

        self.create_subscription(Image, rgb_topic, self._on_img, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, info_topic, self._on_info, qos_profile_sensor_data)
        self.create_subscription(
            PoseStamped, arm["flange_pose_topic"], self._on_flange, qos_profile_sensor_data
        )
        self.create_subscription(JointState, JOINT_STATES_TOPIC, self._on_joint_state, 10)

        self._debug_pub = None
        if publish_debug:
            self._debug_pub = self.create_publisher(
                Image, f"/calibration/capture_flange_poses/{arm_key}/debug_image", 1
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
            name: pos
            for name, pos in zip(msg.name, msg.position)
            if name.startswith(self.joint_prefix)
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
        self._debug_pub.publish(
            _rgb_numpy_to_imgmsg(img_bgr, self.cam_id, self.get_clock().now().to_msg())
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--arm", required=True, choices=sorted(ARM_KEYS))
    parser.add_argument("--num-samples", type=int, default=RECOMMENDED_SAMPLES)
    parser.add_argument(
        "--append", action="store_true",
        help="Append to (rather than overwrite) any previously saved captures for this arm.",
    )
    parser.add_argument("--no-debug-topic", action="store_true")
    parser.add_argument(
        "--gain-profile", default=None, choices=("holding", "insertion"),
        help="Admittance compliance profile from config/admittance_gain_profiles.yaml, "
             "applied to BOTH arms at startup. Defaults to AdmittanceDualArmNode's own "
             "startup default ('holding') if omitted. 'insertion' yields more readily "
             "(lower force deadband, higher gains) -- closer to 'freely positionable'.",
    )
    args = parser.parse_args()

    arm = ARM_KEYS[args.arm]

    existing: list[FlangePoseCapture] = []
    if args.append:
        try:
            existing = load_pose_set(args.arm).captures
            print(f"Appending to {len(existing)} existing saved capture(s) for arm={args.arm}")
        except FileNotFoundError:
            pass

    rclpy.init()

    # Only the arm being captured runs an admittance control loop -- NOT
    # both arms. Running both arms' AdmittanceController (each its own
    # Jacobian pseudo-inverse solve) on every incoming LBRState was
    # measured to roughly halve the achievable control-loop rate for the
    # arm actually being hand-guided, which showed up on real hardware as
    # the arm feeling rigid/unresponsive ("stuck") even though the node
    # was confirmed running and publishing. The other arm's position
    # controller just holds its last commanded pose (controller_manager's
    # own fixed-rate loop keeps streaming that to FRI regardless of
    # whether this node publishes anything for it -- no dropout risk),
    # so this is the standard routine: capture left fully, Ctrl-C, then
    # capture right -- one arm at a time, safe to do as two entirely
    # separate launches.
    admittance_arm_keys = (args.arm,)
    print(f"Bringing up the admittance control loop for arm={args.arm} only (needs "
          "`ros2 launch lbr_dual_arm_bringup admittance.launch.py` already running)...")
    admittance_node = AdmittanceDualArmNode(arm_keys=admittance_arm_keys)
    if args.gain_profile is not None:
        apply_gain_profile(admittance_node, admittance_arm_keys, args.gain_profile)
    log_active_gains(admittance_node, admittance_arm_keys)

    capture_node = _CaptureNode(args.arm, publish_debug=not args.no_debug_topic)

    # Both nodes' callbacks need to keep firing continuously -- the
    # admittance node's to keep yielding to pushes, the capture node's to
    # keep tracking fresh images/poses -- even while the interactive prompt
    # below blocks the main thread on input(). Spin them together in a
    # background daemon thread instead of the main-thread spin_once loop
    # capture_flange_poses_dual_handguided.py uses (that pattern only works
    # because gravity compensation's compliance lives in the hardware
    # controller, not a python control loop that would stall).
    executor = MultiThreadedExecutor()
    executor.add_node(admittance_node)
    executor.add_node(capture_node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    print("")
    print(f"=== Admittance-guided flange pose capture: arm={args.arm} (cam={arm['cam_id']}) ===")
    print(f"Need {args.num_samples} accepted sample(s) this run.")
    print("")
    print(f"Only arm={args.arm} is software-compliant this session (position interface,")
    print("admittance law running in this process) -- push it and it should yield smoothly")
    print("in that direction, holding wherever you stop pushing (no spring-back to a nominal")
    print("pose). The other arm's position controller just holds its last commanded pose --")
    print("capture that arm in a separate run: --arm " + ("right" if args.arm == "left" else "left"))
    print("")
    print("Suggested Foxglove layout while collecting samples:")
    print(f"  - Image panel -> /calibration/capture_flange_poses/{args.arm}/debug_image")
    print(f"  - Raw Messages panel -> {arm['flange_pose_topic']}")
    print("")
    print("For each sample: PHYSICALLY push THIS arm to a pose where the checkerboard is")
    print("fully visible to its wrist RealSense, let go and let it settle, then press Enter")
    print("here to capture. Both the flange pose AND the current joint configuration are")
    print("saved -- the checkerboard detection here is just a quality gate to confirm the")
    print("board is in view. Vary orientation, not just position, across samples.")
    print("Type 'q' + Enter to stop early once you have at least 1 sample.")
    print("")

    captures: list[FlangePoseCapture] = list(existing)
    start_idx = len(captures)

    try:
        n_this_run = 0
        while n_this_run < args.num_samples:
            idx = start_idx + n_this_run
            user_in = input(
                f"[{n_this_run}/{args.num_samples}] arm={args.arm} "
                f"Press Enter to capture (or 'q' to finish early): "
            )
            if user_in.strip().lower() == "q":
                if not captures:
                    print("Need at least 1 sample. Keep going.")
                    continue
                break

            time.sleep(SETTLE_WAIT_S)

            if not capture_node.has_fresh_rgb_info():
                print("  [skip] no fresh RGB/CameraInfo yet")
                continue
            if not capture_node.has_fresh_flange_pose():
                print(
                    f"  [skip] no fresh flange pose (< {FLANGE_POSE_MAX_AGE_S}s old) on "
                    f"{arm['flange_pose_topic']} -- is flange_pose_publisher running?"
                )
                continue
            if not capture_node.has_fresh_joint_state():
                print(
                    f"  [skip] no fresh joint state (< {JOINT_STATE_MAX_AGE_S}s old) on "
                    f"{JOINT_STATES_TOPIC} with prefix {capture_node.joint_prefix}* -- is "
                    f"lbr_dual_arm_bringup admittance.launch.py running?"
                )
                continue

            img = _img_to_numpy_bgr(capture_node.img_msg)
            K = capture_node.K.copy()
            T_armBase_flange = capture_node.flange_pose
            joint_positions = dict(capture_node.joint_positions)

            try:
                _, corners, reproj_err = _solve_board_pose(img, K)
            except RuntimeError as e:
                print(f"  [skip] checkerboard not visible to {arm['cam_id']}: {e}")
                capture_node.publish_debug(img)
                continue

            if reproj_err > MAX_REPROJ_ERR_PX:
                print(f"  [skip] reprojection error too high: {reproj_err:.3f}px > {MAX_REPROJ_ERR_PX}px")
                continue

            vis = _draw_chessboard(img, corners, f"{args.arm} sample {idx} reproj={reproj_err:.3f}px")
            capture_node.publish_debug(vis)

            debug_dir = Path(DEBUG_DIR) / args.arm
            debug_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(debug_dir / f"sample_{idx:02d}.png"), vis)

            capture = FlangePoseCapture(
                idx=idx,
                T_armBase_flange=T_armBase_flange,
                captured_at_unix_s=time.time(),
                note=f"board_reproj_px={reproj_err:.3f} admittance_guided=true",
                joint_positions=joint_positions,
            )
            captures.append(capture)
            n_this_run += 1

            # Save after EVERY capture (not just at the end) -- same
            # "preliminary results are immediately saved along the way"
            # rationale as the other capture scripts: a crash/Ctrl-C
            # mid-session loses nothing already accepted.
            pose_set = FlangePoseSet(
                arm_key=args.arm,
                cam_id=arm["cam_id"],
                base_frame=arm["base_frame"],
                flange_frame=arm["flange_frame"],
                captures=captures,
            )
            out_path = save_pose_set(pose_set)

            print(
                f"  [ok] reproj={reproj_err:.3f}px  T_armBase_flange.t={T_armBase_flange.t}  "
                f"joints={ {k: round(v, 4) for k, v in joint_positions.items()} }  "
                f"saved -> {out_path} ({len(captures)} total)"
            )

        print(f"\nDone. {len(captures)} flange pose(s) saved for arm={args.arm} "
              f"in config/flange_poses/{args.arm}.json")
        if len(captures) < 7:
            print(
                f"NOTE: fewer than the recommended 7 samples ({len(captures)}) -- "
                f"autocalibrate_dual_realsense.py needs >= 5 for the hand-eye solve "
                f"plus >= 1 for the board-pose solve for this arm."
            )
    finally:
        executor.shutdown()
        capture_node.destroy_node()
        admittance_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
