"""Hand-guided dual-arm flange pose capture -- gravity-compensation twin of
capture_flange_poses_dual.py.

capture_flange_poses_dual.py jogs each arm with the MoveIt RViz
MotionPlanning panel (Plan & Execute -- i.e. MoveGroup goals) between
captures. This script instead expects the arm to be physically hand-guided:
bring the rig up in gravity-compensation mode (both arms compliant, no
active position/velocity control) via

    ros2 launch lbr_dual_arm_bringup calibration.launch.py

(see that launch file and dual_arm_gravity_compensation_controllers.yaml --
this spawns gravity_compensation_lbr_one/_lbr_two instead of
joint_trajectory_controller, so both arms are hand-guidable immediately, no
extra controller-switching step). Then physically push whichever arm you're
capturing to a pose where its wrist RealSense sees the checkerboard, let go,
and press Enter here to capture -- exactly like capture_flange_poses_dual.py
otherwise (same checkerboard quality gate, same incremental-save-every-
capture reliability, same config/flange_poses/<arm>.json output).

The one schema difference: each saved FlangePoseCapture also carries
joint_positions (that arm's 7 raw joint angles at capture time, read from
/lbr_dual_arm/joint_states -- published by joint_state_broadcaster
regardless of which controller is active), not just the Cartesian
T_armBase_flange pose. On this redundant 7-DOF arm, T_armBase_flange alone
under-determines the elbow configuration that was actually captured -- IK
re-solved from the Cartesian pose alone could land on a different joint
configuration. Saving joint_positions lets a later MoveIt-based
reconstruction target the exact recorded configuration directly (a
JointConstraint goal), not a re-derived one.

NOTE: gravity_compensation's torque term is a KDL ChainDynParam::JntToGravity()
computed from robot_description -- it accounts for the Y-gripper's mass
(y_gripper.xacro has proper <inertial> tags) but NOT for whatever RealSense
camera + mount is bolted on beyond that (no inertial tags exist for it
anywhere in this URDF). Expect a small residual pull/droop near the
wrist-mounted camera while hand-guiding; this is a known, currently
uncalibrated gap, not a bug.

Run (inside the 'vision' container), with
`ros2 launch lbr_dual_arm_bringup calibration.launch.py` already
up, checkerboard placed and visible to whichever arm you're currently
capturing:

    python3 -m src.calibration.capture_flange_poses_dual_handguided --arm left
    python3 -m src.calibration.capture_flange_poses_dual_handguided --arm right
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, JointState

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
# calibration.launch.py defaults robot_name to "lbr_dual_arm" specifically to
# match this (see that launch file's docstring).
JOINT_STATES_TOPIC = "/lbr_dual_arm/joint_states"


def _arm_joint_prefix(arm_key: str) -> str:
    # ARM_KEYS["left"]["base_frame"] == "lbr_one_link_0" -> "lbr_one_A"
    # (matches joint names lbr_one_A1..A7 -- see dual_arm_controllers.yaml /
    # dual_arm_gravity_compensation_controllers.yaml). Derived rather than
    # added to ARM_KEYS to keep flange_pose_store.py's schema untouched
    # beyond the new joint_positions field.
    base_frame = ARM_KEYS[arm_key]["base_frame"]
    robot_prefix = base_frame.rsplit("_link_0", 1)[0]
    return f"{robot_prefix}_A"


class _CaptureNode(Node):
    def __init__(self, arm_key: str, publish_debug: bool):
        arm = ARM_KEYS[arm_key]
        super().__init__(f"capture_flange_poses_dual_handguided_{arm_key}")
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
    node = _CaptureNode(args.arm, publish_debug=not args.no_debug_topic)

    print("")
    print(f"=== Hand-guided flange pose capture: arm={args.arm} (cam={arm['cam_id']}) ===")
    print(f"Need {args.num_samples} accepted sample(s) this run.")
    print("")
    print("Prerequisite: `ros2 launch lbr_dual_arm_bringup calibration.launch.py`")
    print("already running (both arms in gravity-compensation mode -- no MoveIt needed here).")
    print("")
    print("Suggested Foxglove layout while collecting samples:")
    print(f"  - Image panel -> /calibration/capture_flange_poses/{args.arm}/debug_image")
    print(f"  - Raw Messages panel -> {arm['flange_pose_topic']}")
    print("")
    print("For each sample: PHYSICALLY hand-guide THIS arm (it's gravity-compensated, should")
    print("move freely -- expect a small residual pull near the wrist camera, see this")
    print("script's module docstring) to a pose where the checkerboard is fully visible to")
    print("its wrist RealSense, let go and let it settle, then press Enter here to capture.")
    print("Both the flange pose AND the current joint configuration are saved -- the")
    print("checkerboard detection here is just a quality gate to confirm the board is in view.")
    print("Vary orientation, not just position, across samples.")
    print("Type 'q' + Enter to stop early once you have at least 1 sample.")
    print("")

    captures: list[FlangePoseCapture] = list(existing)
    start_idx = len(captures)

    try:
        n_this_run = 0
        while n_this_run < args.num_samples:
            rclpy.spin_once(node, timeout_sec=0.0)
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

            t_deadline = time.time() + 2.0
            while time.time() < t_deadline:
                rclpy.spin_once(node, timeout_sec=0.05)

            if not node.has_fresh_rgb_info():
                print("  [skip] no fresh RGB/CameraInfo yet")
                continue
            if not node.has_fresh_flange_pose():
                print(
                    f"  [skip] no fresh flange pose (< {FLANGE_POSE_MAX_AGE_S}s old) on "
                    f"{arm['flange_pose_topic']} -- is flange_pose_publisher running?"
                )
                continue
            if not node.has_fresh_joint_state():
                print(
                    f"  [skip] no fresh joint state (< {JOINT_STATE_MAX_AGE_S}s old) on "
                    f"{JOINT_STATES_TOPIC} with prefix {node.joint_prefix}* -- is "
                    f"lbr_dual_arm_bringup calibration.launch.py running?"
                )
                continue

            img = _img_to_numpy_bgr(node.img_msg)
            K = node.K.copy()
            T_armBase_flange = node.flange_pose
            joint_positions = dict(node.joint_positions)

            try:
                _, corners, reproj_err = _solve_board_pose(img, K)
            except RuntimeError as e:
                print(f"  [skip] checkerboard not visible to {arm['cam_id']}: {e}")
                node.publish_debug(img)
                continue

            if reproj_err > MAX_REPROJ_ERR_PX:
                print(f"  [skip] reprojection error too high: {reproj_err:.3f}px > {MAX_REPROJ_ERR_PX}px")
                continue

            vis = _draw_chessboard(img, corners, f"{args.arm} sample {idx} reproj={reproj_err:.3f}px")
            node.publish_debug(vis)

            debug_dir = Path(DEBUG_DIR) / args.arm
            debug_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(debug_dir / f"sample_{idx:02d}.png"), vis)

            capture = FlangePoseCapture(
                idx=idx,
                T_armBase_flange=T_armBase_flange,
                captured_at_unix_s=time.time(),
                note=f"board_reproj_px={reproj_err:.3f} hand_guided=true",
                joint_positions=joint_positions,
            )
            captures.append(capture)
            n_this_run += 1

            # Save after EVERY capture (not just at the end) -- same
            # "preliminary results are immediately saved along the way"
            # rationale as capture_flange_poses_dual.py: a crash/Ctrl-C
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
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
