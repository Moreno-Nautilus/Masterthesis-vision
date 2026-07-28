"""Local debugging tool: keyboard control for the tracker's start/stop/reset services.

Run alongside run_pipeline_track_multicam(_realsense).py in its own terminal.
Reads single keypresses (no Enter needed) and calls the tracker node's
std_srvs services so you can pause/resume/reset tracking without touching
the pipeline terminal or reloading any model.

    s  -> start tracking   (set_tracking_active(data=True))
    x  -> stop tracking    (set_tracking_active(data=False))
    r  -> reset tracking   (reset_tracking)
    q  -> quit this tool (does not stop the pipeline)
"""
from __future__ import annotations

import argparse
import sys
import termios
import tty

import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool, Trigger

KEY_BINDINGS = {
    "s": "start",
    "x": "stop",
    "r": "reset",
    "q": "quit",
}

HELP_TEXT = """
Tracking keyboard control — press a key (no Enter needed):
  s  start tracking   (set_tracking_active: true)
  x  stop tracking    (set_tracking_active: false)
  r  reset tracking   (reset_tracking, clears state without stopping)
  q  quit this tool (pipeline keeps running)
"""


def read_single_key() -> str:
    """Blocking single-character read from stdin, raw mode (no Enter needed)."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


class TrackingKeyboardControl(Node):
    def __init__(self, node_name: str) -> None:
        super().__init__("tracking_keyboard_control")
        self._set_active_client = self.create_client(
            SetBool, f"/{node_name}/set_tracking_active"
        )
        self._reset_client = self.create_client(
            Trigger, f"/{node_name}/reset_tracking"
        )

    def _wait_for(self, client, label: str, timeout_s: float = 2.0) -> bool:
        if not client.wait_for_service(timeout_sec=timeout_s):
            self.get_logger().warn(
                f"Service {client.srv_name} not available "
                f"(is the {label} tracker node running?)"
            )
            return False
        return True

    def set_active(self, active: bool) -> None:
        if not self._wait_for(self._set_active_client, "target"):
            return
        req = SetBool.Request(data=active)
        future = self._set_active_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        if future.done() and future.result() is not None:
            print(f"[keyboard-control] {future.result().message}")
        else:
            self.get_logger().warn("set_tracking_active call did not complete")

    def reset(self) -> None:
        if not self._wait_for(self._reset_client, "target"):
            return
        future = self._reset_client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        if future.done() and future.result() is not None:
            print(f"[keyboard-control] {future.result().message}")
        else:
            self.get_logger().warn("reset_tracking call did not complete")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--node-name",
        default="foundationpose_tracker",
        help="Name of the tracker node whose services to call.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not sys.stdin.isatty():
        print(
            "tracking_keyboard_control needs an interactive TTY for raw keypresses "
            "(run it directly at a terminal, not through a pipe or non-interactive shell).",
            file=sys.stderr,
        )
        sys.exit(1)

    rclpy.init()
    node = TrackingKeyboardControl(args.node_name)
    print(HELP_TEXT, flush=True)
    try:
        while rclpy.ok():
            key = read_single_key()
            action = KEY_BINDINGS.get(key)
            if action == "start":
                node.set_active(True)
            elif action == "stop":
                node.set_active(False)
            elif action == "reset":
                node.reset()
            elif action == "quit" or key == "\x03":  # q or Ctrl-C
                break
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
