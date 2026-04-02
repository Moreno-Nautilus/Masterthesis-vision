from __future__ import annotations

import time
import threading
from typing import Callable, Dict, Optional

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

from std_srvs.srv import Trigger
from builtin_interfaces.msg import Duration

# Standard ros2_control service type
from controller_manager_msgs.srv import SwitchController


class AssemblySupervisor(Node):
    """
    Supervisor for:
      - switching between grasp and insert controllers
      - exposing manual wrapper services for each stage
      - optionally running the full 2-screw sequence

    Important:
      - This is intentionally parameterized so you can fill in exact controller
        and service names later when you're back on the robot.
      - It assumes BOTH underlying skill nodes can stay running, while only one
        low-level arm controller is active at a time.
    """

    MODE_UNKNOWN = "unknown"
    MODE_GRASP = "grasp"
    MODE_INSERT = "insert"

    def __init__(self) -> None:
        super().__init__("assembly_supervisor")
        self.cb_group = ReentrantCallbackGroup()

        # ------------------------------------------------------------------
        # Parameters you will likely change later
        # ------------------------------------------------------------------
        self.declare_parameter("controller_manager_service", "/controller_manager/switch_controller")

        self.declare_parameter("grasp_controller", "cartesian_impedance_controller")
        self.declare_parameter("insert_controller", "joint_trajectory_controller")

        # Some setups also have other conflicting arm controllers that should be
        # explicitly deactivated when switching.
        self.declare_parameter("extra_deactivate_for_grasp", [])
        self.declare_parameter("extra_deactivate_for_insert", [])

        # Wrapped grasp services
        self.declare_parameter("srv_go_pregrasp", "/franka_grasp_bridge/go_pregrasp")
        self.declare_parameter("srv_go_grasp", "/franka_grasp_bridge/go_grasp")
        self.declare_parameter("srv_go_lift", "/franka_grasp_bridge/go_lift")
        self.declare_parameter("srv_go_neutral", "/franka_grasp_bridge/go_neutral")

        # Wrapped insert services hole 1
        self.declare_parameter("srv_preinsert_hole1", "/assembly/preinsert")
        self.declare_parameter("srv_insert_hole1", "/assembly/insert")
        self.declare_parameter("srv_seat_hole1", "/assembly/seat")
        self.declare_parameter("srv_retreat_hole1", "/assembly/retreat")

        # Wrapped insert services hole 2
        self.declare_parameter("srv_preinsert_hole2", "/assembly/preinsert_hole2")
        self.declare_parameter("srv_insert_hole2", "/assembly/insert_hole2")
        self.declare_parameter("srv_seat_hole2", "/assembly/seat_hole2")
        self.declare_parameter("srv_retreat_hole2", "/assembly/retreat_hole2")

        # Behavior
        self.declare_parameter("auto_switch_on_stage_calls", True)
        self.declare_parameter("downstream_wait_for_service_timeout_s", 2.0)
        self.declare_parameter("switch_timeout_s", 5.0)
        self.declare_parameter("settle_after_switch_s", 0.5)
        self.declare_parameter("post_stage_sleep_s", 0.0)

        # Optional: if you want to force a neutral before switching modes later
        self.declare_parameter("always_go_neutral_before_switch", False)

        self.controller_manager_service = str(self.get_parameter("controller_manager_service").value)
        self.grasp_controller = str(self.get_parameter("grasp_controller").value)
        self.insert_controller = str(self.get_parameter("insert_controller").value)

        self.extra_deactivate_for_grasp = list(self.get_parameter("extra_deactivate_for_grasp").value)
        self.extra_deactivate_for_insert = list(self.get_parameter("extra_deactivate_for_insert").value)

        self.auto_switch_on_stage_calls = bool(self.get_parameter("auto_switch_on_stage_calls").value)
        self.downstream_wait_for_service_timeout_s = float(
            self.get_parameter("downstream_wait_for_service_timeout_s").value
        )
        self.switch_timeout_s = float(self.get_parameter("switch_timeout_s").value)
        self.settle_after_switch_s = float(self.get_parameter("settle_after_switch_s").value)
        self.post_stage_sleep_s = float(self.get_parameter("post_stage_sleep_s").value)
        self.always_go_neutral_before_switch = bool(
            self.get_parameter("always_go_neutral_before_switch").value
        )

        # ------------------------------------------------------------------
        # State
        # ------------------------------------------------------------------
        self.current_mode = self.MODE_UNKNOWN

        # ------------------------------------------------------------------
        # Clients
        # ------------------------------------------------------------------
        self.switch_client = self.create_client(
            SwitchController,
            self.controller_manager_service,
            callback_group=self.cb_group,
        )

        # Route table: local wrapper service -> required mode + downstream service
        self.routes: Dict[str, Dict[str, Optional[str]]] = {
            # manual grasp stages
            "go_pregrasp": {
                "required_mode": self.MODE_GRASP,
                "downstream_service": str(self.get_parameter("srv_go_pregrasp").value),
            },
            "go_grasp": {
                "required_mode": self.MODE_GRASP,
                "downstream_service": str(self.get_parameter("srv_go_grasp").value),
            },
            "go_lift": {
                "required_mode": self.MODE_GRASP,
                "downstream_service": str(self.get_parameter("srv_go_lift").value),
            },
            "go_neutral": {
                "required_mode": self.MODE_GRASP,
                "downstream_service": str(self.get_parameter("srv_go_neutral").value),
            },

            # hole 1
            "preinsert_hole1": {
                "required_mode": self.MODE_INSERT,
                "downstream_service": str(self.get_parameter("srv_preinsert_hole1").value),
            },
            "insert_hole1": {
                "required_mode": self.MODE_INSERT,
                "downstream_service": str(self.get_parameter("srv_insert_hole1").value),
            },
            "seat_hole1": {
                "required_mode": self.MODE_INSERT,
                "downstream_service": str(self.get_parameter("srv_seat_hole1").value),
            },
            "retreat_hole1": {
                "required_mode": self.MODE_INSERT,
                "downstream_service": str(self.get_parameter("srv_retreat_hole1").value),
            },

            # hole 2
            "preinsert_hole2": {
                "required_mode": self.MODE_INSERT,
                "downstream_service": str(self.get_parameter("srv_preinsert_hole2").value),
            },
            "insert_hole2": {
                "required_mode": self.MODE_INSERT,
                "downstream_service": str(self.get_parameter("srv_insert_hole2").value),
            },
            "seat_hole2": {
                "required_mode": self.MODE_INSERT,
                "downstream_service": str(self.get_parameter("srv_seat_hole2").value),
            },
            "retreat_hole2": {
                "required_mode": self.MODE_INSERT,
                "downstream_service": str(self.get_parameter("srv_retreat_hole2").value),
            },
        }

        self.trigger_clients: Dict[str, any] = {}
        for route in self.routes.values():
            srv_name = route["downstream_service"]
            if srv_name not in self.trigger_clients:
                self.trigger_clients[srv_name] = self.create_client(
                    Trigger,
                    srv_name,
                    callback_group=self.cb_group,
                )

        # ------------------------------------------------------------------
        # Supervisor services
        # ------------------------------------------------------------------
        self.create_service(
            Trigger,
            "~/switch_to_grasp",
            self._switch_to_grasp_cb,
            callback_group=self.cb_group,
        )
        self.create_service(
            Trigger,
            "~/switch_to_insert",
            self._switch_to_insert_cb,
            callback_group=self.cb_group,
        )
        self.create_service(
            Trigger,
            "~/run_full_sequence",
            self._run_full_sequence_cb,
            callback_group=self.cb_group,
        )
        self.create_service(
            Trigger,
            "~/report_mode",
            self._report_mode_cb,
            callback_group=self.cb_group,
        )

        # Create one wrapper service per manual stage
        for public_name in self.routes.keys():
            self.create_service(
                Trigger,
                f"~/{public_name}",
                self._make_stage_callback(public_name),
                callback_group=self.cb_group,
            )

        self.get_logger().info("AssemblySupervisor started")
        self.get_logger().info(
            f"grasp_controller={self.grasp_controller} | "
            f"insert_controller={self.insert_controller} | "
            f"auto_switch_on_stage_calls={self.auto_switch_on_stage_calls}"
        )

    # ----------------------------------------------------------------------
    # Generic helpers
    # ----------------------------------------------------------------------

    def _wait_future(self, future, timeout_s: float):
        done = threading.Event()
        future.add_done_callback(lambda _: done.set())
        if not done.wait(timeout_s):
            return None
        return future.result()

    def _duration_msg(self, seconds: float) -> Duration:
        sec = int(seconds)
        nanosec = int((seconds - sec) * 1e9)
        return Duration(sec=sec, nanosec=nanosec)

    def _call_trigger_service(self, service_name: str) -> tuple[bool, str]:
        client = self.trigger_clients[service_name]

        if not client.wait_for_service(timeout_sec=self.downstream_wait_for_service_timeout_s):
            return False, f"Service not available: {service_name}"

        req = Trigger.Request()
        fut = client.call_async(req)
        res = self._wait_future(fut, 60.0)
        if res is None:
            return False, f"Timeout waiting for {service_name}"

        if not res.success:
            return False, f"{service_name} failed: {res.message}"

        if self.post_stage_sleep_s > 0.0:
            time.sleep(self.post_stage_sleep_s)

        return True, res.message

    def _switch_controllers(self, activate: list[str], deactivate: list[str]) -> tuple[bool, str]:
        if not self.switch_client.wait_for_service(timeout_sec=self.downstream_wait_for_service_timeout_s):
            return False, f"Controller manager service not available: {self.controller_manager_service}"

        req = SwitchController.Request()

        # Standard fields
        req.activate_controllers = activate
        req.deactivate_controllers = deactivate

        # STRICT is standard in ros2_control SwitchController
        if hasattr(SwitchController.Request, "STRICT"):
            req.strictness = SwitchController.Request.STRICT
        else:
            # fallback if constant exposure differs
            req.strictness = 2

        # ros2_control variants sometimes differ here
        if hasattr(req, "activate_asap"):
            req.activate_asap = True
        elif hasattr(req, "start_asap"):
            req.start_asap = True

        # timeout is normally builtin_interfaces/Duration
        if hasattr(req, "timeout"):
            req.timeout = self._duration_msg(self.switch_timeout_s)

        fut = self.switch_client.call_async(req)
        res = self._wait_future(fut, self.switch_timeout_s + 2.0)
        if res is None:
            return False, "Timeout waiting for controller switch"

        if not getattr(res, "ok", False):
            return False, (
                f"Controller switch failed | activate={activate} | deactivate={deactivate}"
            )

        if self.settle_after_switch_s > 0.0:
            time.sleep(self.settle_after_switch_s)

        return True, "controller switch ok"

    def _ensure_mode(self, target_mode: str) -> tuple[bool, str]:
        if target_mode == self.current_mode:
            return True, f"Already in mode '{target_mode}'"

        if self.always_go_neutral_before_switch and target_mode != self.MODE_UNKNOWN:
            # best-effort neutral only if grasp side service exists and we are currently in grasp mode
            if self.current_mode == self.MODE_GRASP:
                neutral_srv = self.routes["go_neutral"]["downstream_service"]
                ok, msg = self._call_trigger_service(neutral_srv)
                if not ok:
                    return False, f"Failed neutral before switch: {msg}"

        if target_mode == self.MODE_GRASP:
            activate = [self.grasp_controller]
            deactivate = [self.insert_controller] + self.extra_deactivate_for_grasp
        elif target_mode == self.MODE_INSERT:
            activate = [self.insert_controller]
            deactivate = [self.grasp_controller] + self.extra_deactivate_for_insert
        else:
            return False, f"Unknown target_mode='{target_mode}'"

        # remove duplicates and self-activation duplicates
        deactivate = [x for x in deactivate if x and x not in activate]
        activate = [x for x in activate if x]

        ok, msg = self._switch_controllers(activate, deactivate)
        if not ok:
            return False, msg

        self.current_mode = target_mode
        self.get_logger().info(f"Switched to mode: {self.current_mode}")
        return True, msg

    def _run_stage_by_name(self, stage_name: str) -> tuple[bool, str]:
        if stage_name not in self.routes:
            return False, f"Unknown stage '{stage_name}'"

        route = self.routes[stage_name]
        required_mode = route["required_mode"]
        downstream_service = route["downstream_service"]

        if self.auto_switch_on_stage_calls:
            ok, msg = self._ensure_mode(required_mode)
            if not ok:
                return False, f"Mode switch failed before '{stage_name}': {msg}"
        else:
            if self.current_mode != required_mode:
                return False, (
                    f"Stage '{stage_name}' requires mode '{required_mode}', "
                    f"but current_mode is '{self.current_mode}'"
                )

        self.get_logger().info(f"Running stage '{stage_name}' -> {downstream_service}")
        return self._call_trigger_service(downstream_service)

    def _make_stage_callback(self, stage_name: str) -> Callable:
        def _cb(request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
            del request
            ok, msg = self._run_stage_by_name(stage_name)
            response.success = ok
            response.message = msg
            return response
        return _cb

    # ----------------------------------------------------------------------
    # Supervisor service callbacks
    # ----------------------------------------------------------------------

    def _switch_to_grasp_cb(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        del request
        ok, msg = self._ensure_mode(self.MODE_GRASP)
        response.success = ok
        response.message = msg
        return response

    def _switch_to_insert_cb(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        del request
        ok, msg = self._ensure_mode(self.MODE_INSERT)
        response.success = ok
        response.message = msg
        return response

    def _report_mode_cb(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        del request
        response.success = True
        response.message = f"current_mode={self.current_mode}"
        return response

    def _run_full_sequence_cb(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        del request

        sequence = [
            # screw 1 pickup
            "go_pregrasp",
            "go_grasp",
            "go_lift",

            # hole 1 insert
            "preinsert_hole1",
            "insert_hole1",
            "seat_hole1",
            "retreat_hole1",

            # screw 2 pickup
            "go_pregrasp",
            "go_grasp",
            "go_lift",

            # hole 2 insert
            "preinsert_hole2",
            "insert_hole2",
            "seat_hole2",
            "retreat_hole2",
        ]

        for stage_name in sequence:
            ok, msg = self._run_stage_by_name(stage_name)
            if not ok:
                response.success = False
                response.message = f"Sequence failed at '{stage_name}': {msg}"
                return response

        response.success = True
        response.message = "Full 2-screw sequence completed"
        return response


def main() -> None:
    rclpy.init()
    node = AssemblySupervisor()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()