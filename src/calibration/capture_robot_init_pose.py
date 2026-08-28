"""Snapshots the robot's current joint positions (per arm) to
config/robot_init_pose.yaml.

Reads /lbr_dual_arm/joint_states (moveit_dual_arm.JOINT_STATES_TOPIC).
Joint-space only -- move_to_init_pose.py drives back to these exact values
via MoveGroup joint-space goals (JointTarget/move_to_joint(), no IK), so no
Cartesian flange pose is needed or stored here (a stored Cartesian pose
would be ambiguous for this redundant 7-DOF arm anyway -- re-solving IK for
it can land on a different elbow/null-space configuration than the one
actually captured).

Usage:
    python -m src.calibration.capture_robot_init_pose
"""

from __future__ import annotations

import time
from pathlib import Path

import rclpy
import yaml
from rclpy.node import Node
from sensor_msgs.msg import JointState

from src.calibration.moveit_dual_arm import JOINT_STATES_TOPIC

OUTPUT_PATH = Path("config/robot_init_pose.yaml")
WAIT_TIMEOUT_S = 10.0

ARM_JOINT_PREFIXES = {"left": "lbr_one_", "right": "lbr_two_"}


class _SnapshotNode(Node):
    def __init__(self):
        super().__init__("capture_robot_init_pose")
        self.joint_positions: dict[str, float] = {}
        self.create_subscription(JointState, JOINT_STATES_TOPIC, self._on_joint_state, 10)
        self.get_logger().info(f"joint_states={JOINT_STATES_TOPIC}")

    def _on_joint_state(self, msg: JointState) -> None:
        self.joint_positions = dict(zip(msg.name, msg.position))

    def joints_ready(self) -> bool:
        if not self.joint_positions:
            return False
        return all(
            any(name.startswith(prefix) for name in self.joint_positions)
            for prefix in ARM_JOINT_PREFIXES.values()
        )


def main() -> None:
    rclpy.init()
    node = _SnapshotNode()
    deadline = time.monotonic() + WAIT_TIMEOUT_S
    try:
        while time.monotonic() < deadline and not node.joints_ready():
            rclpy.spin_once(node, timeout_sec=0.1)

        if not node.joints_ready():
            raise RuntimeError(
                f"Timed out after {WAIT_TIMEOUT_S}s waiting for joint_states on both arms -- "
                f"is lbr_dual_arm_bringup hardware.launch.py running?"
            )

        data: dict = {"captured_at_unix_s": time.time()}
        for arm_key, prefix in ARM_JOINT_PREFIXES.items():
            data[arm_key] = {
                "joint_positions": {
                    name: pos for name, pos in node.joint_positions.items() if name.startswith(prefix)
                },
            }

        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.write_text(yaml.safe_dump(data, sort_keys=False))
        print(f"Wrote {OUTPUT_PATH}")
        for arm_key in ARM_JOINT_PREFIXES:
            print(f"  [{arm_key}] joints={list(data[arm_key]['joint_positions'])}")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
