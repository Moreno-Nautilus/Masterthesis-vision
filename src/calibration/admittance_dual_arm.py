"""Software admittance control for the dual-LBR rig -- position-interface,
no torque command interface involved anywhere in this module (confirmed
correct even though it doesn't send torque commands; see the discussion
this module's design followed). Additive alongside moveit_dual_arm.py and
cartesian_impedance_dual_arm.py, not a replacement.

Two ways to run this: standalone, via this module's own main()
(`python3 -m src.calibration.admittance_dual_arm`) for general hand-guiding;
or embedded, as the default Step 1 calibration-image-gathering control
mode -- capture_flange_poses_dual_admittance.py constructs
AdmittanceDualArmNode itself and spins it in a background thread alongside
its own interactive capture loop, rather than importing this main(). Both
paths share apply_gain_profile()/log_active_gains() below.

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
"reasonable by default" values (rotational axes retuned 2026-08-03 -- see
that file's comments -- after hand-guiding felt too stiff rotationally).
The ONLY way to change them afterward is through this node's registered
parameter-set callback (_on_set_parameters) -- reachable either via the
standard `~/set_parameters` *service* (an external caller, e.g. a
different process/node) or, for an in-process caller that already holds
the node object, this module's own apply_gain_profile() helper, which just
calls the node's plain set_parameters() Python method -- rclpy invokes the
exact same registered callback either way, no bypass. Mirrors
cartesian_impedance_dual_arm.py's set_gains() (same
stationary-guard constraint, just via the generic parameter mechanism
instead of the controller's own). The validating callback refuses a change
and returns successful=False unless that arm is currently stationary,
checked via the admittance law's own internal `_dq` (the joint velocity it
is CURRENTLY commanding) -- exactly "is this admittance loop presently
moving the arm", no separate velocity source needed. gain_profiles.py's
load_admittance_profile() is a pure helper for building the values to send;
it does not itself change anything.
"""

from __future__ import annotations

import argparse

import numpy as np
import rclpy
from lbr_fri_idl.msg import LBRJointPositionCommand, LBRState
from rcl_interfaces.msg import ParameterValue, SetParametersResult
from rcl_interfaces.srv import GetParameters
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from urdf_parser_py.urdf import URDF

from lbr_demos_advanced_py.admittance_controller import AdmittanceController
from src.calibration import gain_profiles
from src.calibration.moveit_dual_arm import DEFAULT_MOVE_GROUP_NAMESPACE


def _single_arm_urdf(robot_description: str, base_link: str, end_effector_link: str) -> str:
    """Prune the shared dual-arm robot_description down to just the serial
    chain base_link -> end_effector_link, so optas.RobotModel's ndof (and the
    Jacobian function's expected input length) is that arm's own 7 joints,
    not the combined dual-arm + gripper model's ~18. AdmittanceController
    (lbr_demos_advanced_py, written for a single-arm rig) sizes every
    internal array off self._robot.ndof and feeds it lbr_state's own 7-long
    measured_joint_position/external_torque directly -- handing it the full
    dual-arm URDF makes ndof 18 while the message stays 7-long, crashing the
    Jacobian call with a shape mismatch (confirmed: optas' param_joints does
    NOT shrink the Jacobian function's required input length, only a
    genuinely smaller URDF does).
    """
    full = URDF.from_xml_string(robot_description)
    chain = full.get_chain(base_link, end_effector_link, joints=True, links=True, fixed=True)
    link_names, joint_names = chain[0::2], chain[1::2]

    pruned = URDF(name=f"{full.name}_{base_link}_{end_effector_link}")
    for link_name in link_names:
        pruned.add_link(full.link_map[link_name])
    for joint_name in joint_names:
        pruned.add_joint(full.joint_map[joint_name])
    return pruned.to_xml_string()

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
        # One MutuallyExclusiveCallbackGroup PER ARM (rather than every
        # subscription defaulting to this node's single shared group) --
        # otherwise both arms' LBRState callbacks would serialize against
        # each other on a single group, so a slow callback for one arm
        # (e.g. optas' first, uncompiled Jacobian evaluation) could delay
        # the other arm's command right behind it in the queue. Only
        # actually buys concurrency when spun via a MultiThreadedExecutor
        # (this module's own main() below, and
        # capture_flange_poses_dual_admittance.py, both do) -- a plain
        # rclpy.spin() uses a single-threaded executor internally and would
        # still process one callback at a time regardless of grouping.
        self._callback_groups: dict[str, MutuallyExclusiveCallbackGroup] = {
            arm_key: MutuallyExclusiveCallbackGroup() for arm_key in arm_keys
        }
        for arm_key in arm_keys:
            info = ARM_KEYS[arm_key]
            self._controllers[arm_key] = AdmittanceController(
                robot_description=_single_arm_urdf(
                    robot_description, info["base_link"], info["end_effector_link"]
                ),
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
                callback_group=self._callback_groups[arm_key],
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


def apply_gain_profile(
    node: AdmittanceDualArmNode, arm_keys: tuple[str, ...], profile_name: str
) -> None:
    """Override the startup ("holding") profile applied by
    AdmittanceDualArmNode.__init__ with a named profile from
    config/admittance_gain_profiles.yaml, for every arm_key. Calling the
    node's own set_parameters() (rather than a set_parameters *service*
    call) still goes through the exact same _on_set_parameters
    validation/stationary-guard as an external caller -- it's the same
    registered callback either way (rclpy.node.Node.set_parameters()
    invokes add_on_set_parameters_callback callbacks directly, no service
    round-trip needed for an in-process caller). Safe to call right after
    construction: nothing has moved yet, so every arm reads as stationary.
    Shared by this module's own main() and
    capture_flange_poses_dual_admittance.py so both pick up retuned
    profiles without duplicating this logic.
    """
    profile = gain_profiles.load_admittance_profile(profile_name)
    params = [
        Parameter(f"{gain_name}.{arm_key}", value=list(profile[gain_name]))
        for arm_key in arm_keys
        for gain_name in GAIN_PARAM_LENGTHS
    ]
    results = node.set_parameters(params)
    for param, result in zip(params, results):
        if not result.successful:
            raise RuntimeError(
                f"Failed to apply gain-profile {profile_name!r} param {param.name!r}: "
                f"{result.reason}"
            )


def log_active_gains(node: AdmittanceDualArmNode, arm_keys: tuple[str, ...]) -> None:
    node.get_logger().info("*** Active admittance gains (per arm):")
    for arm_key in arm_keys:
        values = {
            gain_name: node.get_parameter(f"{gain_name}.{arm_key}").value
            for gain_name in GAIN_PARAM_LENGTHS
        }
        node.get_logger().info(f"*   {arm_key}: {values}")


def main(args: list[str] | None = None) -> None:
    """Standalone entry point -- general-purpose admittance hand-guiding for
    the dual-arm rig, independent of any calibration capture script (for
    that, see capture_flange_poses_dual_admittance.py, which runs this same
    node internally alongside its own capture loop instead of calling this
    main()). Requires `ros2 launch lbr_dual_arm_bringup admittance.launch.py`
    already running.

        python3 -m src.calibration.admittance_dual_arm
        python3 -m src.calibration.admittance_dual_arm --arm left
        python3 -m src.calibration.admittance_dual_arm --gain-profile insertion

    --arm restricts the admittance loop to a single arm (see
    capture_flange_poses_dual_admittance.py, which always does this --
    running both arms' Jacobian solves concurrently roughly halves the
    achievable control-loop rate for the arm actually being hand-guided).
    Omit it to hand-guide both arms at once, same as before.
    """
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument(
        "--arm", default=None, choices=sorted(ARM_KEYS),
        help="Restrict the admittance loop to just this arm. Defaults to both arms.",
    )
    parser.add_argument(
        "--gain-profile", default=None, choices=("holding", "insertion"),
        help="Compliance profile from config/admittance_gain_profiles.yaml, applied to "
             "the active arm(s) at startup. Defaults to AdmittanceDualArmNode's own "
             "startup default ('holding') if omitted.",
    )
    parsed = parser.parse_args(args=args)

    arm_keys = (parsed.arm,) if parsed.arm is not None else tuple(ARM_KEYS)

    rclpy.init()
    node = AdmittanceDualArmNode(arm_keys=arm_keys)
    # MultiThreadedExecutor, not plain rclpy.spin() -- required for the two
    # arms' per-arm callback groups (see AdmittanceDualArmNode.__init__) to
    # actually run concurrently instead of falling back to one callback at
    # a time.
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        if parsed.gain_profile is not None:
            apply_gain_profile(node, arm_keys, parsed.gain_profile)
        log_active_gains(node, arm_keys)
        executor.spin()  # one control step per incoming LBRState message, per arm
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
