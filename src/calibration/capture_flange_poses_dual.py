"""Manual dual-arm flange pose capture -- Step 1 of the "turned up a notch"
RealSense calibration routine (see docs/getting_started_realsense.md section 4).

Unlike the original handeye_flange_cam_realsense.py, this script does NOT
solve any calibration itself. It only jogs-and-captures: for each arm, jog it
with the MoveIt RViz MotionPlanning panel to a pose where its own wrist
RealSense sees the checkerboard, then press Enter here to record that arm's
CURRENT flange pose (read from /left/ee_pose or /right/ee_pose) permanently
to config/flange_poses/<left|right>.json (see flange_pose_store.py for the
schema). A live checkerboard detection + reprojection-error gate on that
arm's own camera is used only as a QUALITY CHECK before accepting the
capture (reject if the board isn't actually visible/well-posed) -- the
checkerboard pose itself is discarded, only the flange pose is kept.

autocalibrate_dual_realsense.py (Step 2) is the script that actually drives
the arms back to these saved poses and solves the calibration.

Recommended: 7 poses per arm (14 total). Of those, autocalibrate_dual_realsense.py
by default uses the first 5 per arm for the hand-eye (AX=XB) solve and the
last 2 per arm for the checkerboard-in-base-frame solve -- see that script's
docstring. Vary orientation, not just position, across the 5 hand-eye poses
per arm (same AX=XB conditioning requirement as the original script).

Both arms can be captured in the same run, in any order/interleaving --
useful since dual_arm_bringup + MoveIt let you jog either arm independently
and moveit's `both_arms` group means simultaneous positioning is possible
too, but nothing here requires moving both at once: jog one, capture, jog
the other, capture, repeat.

Run (inside the 'vision' container), with lbr_dual_arm_bringup hardware.launch.py
and its move_group.launch.py already up, checkerboard placed and visible to
whichever arm you're currently capturing:

    python3 -m src.calibration.capture_flange_poses_dual --arm left
    python3 -m src.calibration.capture_flange_poses_dual --arm right
    # or capture both in one run, switching --arm has no effect on a running
    # process -- just run one terminal per arm, or capture serially:
    python3 -m src.calibration.capture_flange_poses_dual --arm left --num-samples 7
    python3 -m src.calibration.capture_flange_poses_dual --arm right --num-samples 7
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
from sensor_msgs.msg import CameraInfo, Image

from src.perception.ros.qos_profiles import qos_profile_sensor_data_low_latency

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
RECOMMENDED_SAMPLES = 7
DEBUG_DIR = "outputs/calibration_debug/capture_flange_poses"


class _CaptureNode(Node):
    def __init__(self, arm_key: str, publish_debug: bool):
        arm = ARM_KEYS[arm_key]
        super().__init__(f"capture_flange_poses_dual_{arm_key}")
        self.arm_key = arm_key
        self.cam_id = arm["cam_id"]
        rgb_topic, info_topic = _camera_topics(self.cam_id)

        self.img_msg: Optional[Image] = None
        self.img_t: Optional[float] = None
        self.info_t: Optional[float] = None
        self.K: Optional[np.ndarray] = None

        self.flange_pose = None
        self.flange_pose_wall_t: float = 0.0

        self.create_subscription(Image, rgb_topic, self._on_img, qos_profile_sensor_data_low_latency)
        self.create_subscription(CameraInfo, info_topic, self._on_info, qos_profile_sensor_data_low_latency)
        self.create_subscription(
            PoseStamped, arm["flange_pose_topic"], self._on_flange, qos_profile_sensor_data_low_latency
        )

        self._debug_pub = None
        if publish_debug:
            self._debug_pub = self.create_publisher(
                Image, f"/calibration/capture_flange_poses/{arm_key}/debug_image", 1
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
    print(f"=== Flange pose capture: arm={args.arm} (cam={arm['cam_id']}) ===")
    print(f"Need {args.num_samples} accepted sample(s) this run.")
    print("")
    print("Suggested Foxglove layout while collecting samples:")
    print(f"  - Image panel -> /calibration/capture_flange_poses/{args.arm}/debug_image")
    print(f"  - Raw Messages panel -> {arm['flange_pose_topic']}")
    print("")
    print("For each sample: jog THIS arm with the MoveIt RViz MotionPlanning panel")
    print("(Plan & Execute) to a pose where the checkerboard is fully visible to its")
    print("wrist RealSense, let it settle, then press Enter here to capture.")
    print("Only this arm's flange pose is saved -- the checkerboard detection here")
    print("is just a quality gate to confirm the board is actually in view.")
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
                    f"{arm['flange_pose_topic']} -- is lbr_dual_arm_bringup running?"
                )
                continue

            img = _img_to_numpy_bgr(node.img_msg)
            K = node.K.copy()
            T_armBase_flange = node.flange_pose

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
                note=f"board_reproj_px={reproj_err:.3f}",
            )
            captures.append(capture)
            n_this_run += 1

            # Save after EVERY capture (not just at the end) -- this is the
            # "preliminary results are immediately saved along the way"
            # requirement: a crash/Ctrl-C mid-session loses nothing already
            # accepted.
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
