"""Software admittance control for the dual-LBR rig -- position-interface,
no torque command interface involved anywhere in this module (confirmed
correct even though it doesn't send torque commands; see the discussion
this module's design followed). Additive alongside moveit_dual_arm.py and
cartesian_impedance_dual_arm.py, not a replacement.

Reuses lbr_fri_ros2_stack's existing, working control law --
lbr_demos_advanced_py.admittance_controller.AdmittanceController -- rather
than reimplementing the Jacobian-pseudo-inverse force->velocity->position
law. That class has no restoring/goal-tracking term: with zero external
force it holds wherever it currently is (dq decays via its own internal
exponential smoothing, does not spring back to a nominal pose) -- so
"gains" here are purely about compliance/responsiveness (how much
displacement per unit external force, and how much of that force gets
through the deadband), not pose-holding. Requires the `optas` Python
package (added to Dockerfile.thesisnewcuda's pip install list) -- not
previously a dependency of this repo, needed only by the reused
AdmittanceController for its Jacobian.

One AdmittanceController instance per arm, both built from the SAME
dual-arm robot_description (each instance's own base_link/end_effector_link
correctly isolates its own kinematic chain -- see ARM_KEYS below), talking
to lbr_dual_arm_bringup/launch/admittance.launch.py's per-arm
lbr_state_broadcaster_lbr_{one,two} (state topic, LBRState incl.
external_torque) and lbr_joint_position_command_controller_lbr_{one,two}
(command/joint_position topic, LBRJointPositionCommand) -- NOT
dual_arm_controllers.yaml's joint_trajectory_controller.

Gains (f_ext_th/dq_gains/dx_gains) are ordinary ROS parameters on this
node, one triplet per arm (f_ext_th.<arm_key> etc.), defaulting at startup
to config/admittance_gain_profiles.yaml's "holding" profile -- the
"reasonable by default" values. The ONLY way to change them afterward is
this node's own standard `~/set_parameters` service (automatic once
parameters are declared, no custom .srv needed) -- there is no public
Python method that mutates them directly, mirroring
cartesian_impedance_dual_arm.py's set_gains() (same service-only
constraint, just via the generic parameter service instead of the
controller's own). The validating callback
(_on_set_parameters) refuses a change and returns
successful=False unless that arm is currently stationary, checked via the
admittance law's own internal `_dq` (the joint velocity it is CURRENTLY
commanding) -- exactly "is this admittance loop presently moving the arm",
no separate velocity source needed. gain_profiles.py's
load_admittance_profile() is a pure helper for building the values to send
in a set_parameters request; it does not itself change anything.
"""

from __future__ import annotations

import numpy as np
import rclpy
from lbr_fri_idl.msg import LBRJointPositionCommand, LBRState
from rcl_interfaces.msg import ParameterValue, SetParametersResult
from rcl_interfaces.srv import GetParameters
from rclpy.node import Node
from rclpy.parameter import Parameter

from lbr_demos_advanced_py.admittance_controller import AdmittanceController
from src.calibration import gain_profiles
from src.calibration.moveit_dual_arm import DEFAULT_MOVE_GROUP_NAMESPACE

ARM_KEYS = {
    "left": {
        "base_link": "lbr_one_link_0",
        "end_effector_link": "lbr_one_link_ee",
        "state_topic": "lbr_state_broadcaster_lbr_one/state",
        "command_topic": "lbr_joint_position_command_controller_lbr_one/command/joint_position",
    },
    "right": {
        "base_link": "lbr_two_link_0",
        "end_effector_link": "lbr_two_link_ee",
        "state_topic": "lbr_state_broadcaster_lbr_two/state",
        "command_topic": "lbr_joint_position_command_controller_lbr_two/command/joint_position",
    },
}

DEFAULT_NAMESPACE = DEFAULT_MOVE_GROUP_NAMESPACE  # "lbr_dual_arm"

# Startup default profile -- config/admittance_gain_profiles.yaml's
# "reasonable by default" values, applied to every arm at construction.
DEFAULT_GAIN_PROFILE = "holding"

# Per-gain-name parameter array length, and the set_parameters-callback
# stationary guard, checked against the admittance law's own current
# joint-velocity command (controller._dq norm) -- same stationary-only
# constraint as cartesian_impedance_dual_arm.py's set_gains(), same
# threshold units (rad/s).
GAIN_PARAM_LENGTHS = {"f_ext_th": 6, "dq_gains": 7, "dx_gains": 6}
GAIN_CHANGE_DQ_NORM_THRESHOLD = 0.01


class AdmittanceDualArmNode(Node):
    def __init__(
        self,
        node_name: str = "admittance_dual_arm",
        namespace: str = DEFAULT_NAMESPACE,
        arm_keys: tuple[str, ...] = ("left", "right"),
    ) -> None:
        super().__init__(node_name)
        self._namespace = namespace

        robot_description = self._retrieve_parameter(
            f"/{namespace}/robot_state_publisher/get_parameters", "robot_description"
        ).string_value
        update_rate = self._retrieve_parameter(
            f"/{namespace}/controller_manager/get_parameters", "update_rate"
        ).integer_value
        self._dt = 1.0 / float(update_rate)

        default_gains = gain_profiles.load_admittance_profile(DEFAULT_GAIN_PROFILE)

        self._controllers: dict[str, AdmittanceController] = {}
        self._command_pubs = {}
        for arm_key in arm_keys:
            info = ARM_KEYS[arm_key]
            self._controllers[arm_key] = AdmittanceController(
                robot_description=robot_description,
                base_link=info["base_link"],
                end_effector_link=info["end_effector_link"],
            )
            self._command_pubs[arm_key] = self.create_publisher(
                LBRJointPositionCommand, f"/{namespace}/{info['command_topic']}", 1
            )
            self.create_subscription(
                LBRState,
                f"/{namespace}/{info['state_topic']}",
                lambda msg, arm_key=arm_key: self._on_lbr_state(arm_key, msg),
                1,
            )
            self.get_logger().info(
                f"admittance_dual_arm: arm_key={arm_key!r} listening on "
                f"{info['state_topic']}, commanding {info['command_topic']}"
            )

            # Declare the live-adjustable gain parameters, seeded from
            # DEFAULT_GAIN_PROFILE -- the "reasonable by default" values --
            # and apply them immediately (no stationary check needed here,
            # nothing has moved yet). Once declared, this node's standard
            # ~/set_parameters service is the ONLY way to change them (see
            # _on_set_parameters below); there is no other public method.
            for gain_name in GAIN_PARAM_LENGTHS:
                self.declare_parameter(f"{gain_name}.{arm_key}", default_gains[gain_name])
            self._apply_gains(arm_key, default_gains)

        self.add_on_set_parameters_callback(self._on_set_parameters)

    def _retrieve_parameter(self, service: str, name: str) -> ParameterValue:
        client = self.create_client(GetParameters, service)
        while not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f"Waiting for '{service}' service...")
        future = client.call_async(GetParameters.Request(names=[name]))
        rclpy.spin_until_future_complete(self, future)
        result = future.result()
        if result is None:
            raise RuntimeError(f"Failed to retrieve parameter '{name}' from '{service}'.")
        return result.values[0]

    def _on_lbr_state(self, arm_key: str, lbr_state: LBRState) -> None:
        command = self._controllers[arm_key](lbr_state, self._dt)
        self._command_pubs[arm_key].publish(command)

    def _apply_gains(self, arm_key: str, gains: dict) -> None:
        # AdmittanceController has no public setter for these -- reaching
        # into its own attributes directly is the minimal-footprint way to
        # reuse it with runtime-switchable gains without patching
        # lbr_demos_advanced_py. Constructor semantics (matching
        # admittance_controller.py exactly): f_ext_th stored as a plain
        # vector; dq_gains/dx_gains stored pre-diagonalized.
        controller = self._controllers[arm_key]
        if "f_ext_th" in gains:
            controller._f_ext_th = np.array(gains["f_ext_th"], dtype=float)
        if "dq_gains" in gains:
            controller._dq_gains = np.diag(np.array(gains["dq_gains"], dtype=float))
        if "dx_gains" in gains:
            controller._dx_gains = np.diag(np.array(gains["dx_gains"], dtype=float))

    def _on_set_parameters(self, params: list[Parameter]) -> SetParametersResult:
        """The only path that can change f_ext_th/dq_gains/dx_gains after
        construction -- triggered exclusively by this node's standard
        ~/set_parameters service (e.g. gain_profiles.load_admittance_profile()
        results pushed there by a caller). Validates array lengths and, per
        the agreed design, refuses (successful=False) unless that arm's
        admittance loop is currently stationary -- callers must not retry
        this into succeeding while the arm is moving."""
        by_arm: dict[str, dict] = {}
        for p in params:
            for gain_name, expected_len in GAIN_PARAM_LENGTHS.items():
                prefix = f"{gain_name}."
                if not p.name.startswith(prefix):
                    continue
                arm_key = p.name[len(prefix):]
                if arm_key not in ARM_KEYS:
                    return SetParametersResult(
                        successful=False,
                        reason=f"Unknown arm_key in parameter {p.name!r} -- expected one of {sorted(ARM_KEYS)}.",
                    )
                values = p.value
                if len(values) != expected_len:
                    return SetParametersResult(
                        successful=False,
                        reason=f"{p.name} must have {expected_len} values, got {len(values)}.",
                    )
                by_arm.setdefault(arm_key, {})[gain_name] = list(values)

        for arm_key in by_arm:
            controller = self._controllers[arm_key]
            current_dq_norm = float(np.linalg.norm(controller._dq))
            if current_dq_norm >= GAIN_CHANGE_DQ_NORM_THRESHOLD:
                return SetParametersResult(
                    successful=False,
                    reason=(
                        f"Refusing to change gains for arm_key={arm_key!r}: the "
                        f"admittance loop is still commanding joint motion (|dq|="
                        f"{current_dq_norm:.4f} rad/s >= {GAIN_CHANGE_DQ_NORM_THRESHOLD}). "
                        f"Gain changes are only safe while the arm is stationary."
                    ),
                )

        for arm_key, gains in by_arm.items():
            self._apply_gains(arm_key, gains)

        return SetParametersResult(successful=True)
