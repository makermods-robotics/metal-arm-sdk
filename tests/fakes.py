"""Simulated Damiao MIT motors behind MockBackend (the package's _sim_motor), plus test helpers.

Shared tests use only the generic surface (same names as the other SDK's fakes):
FakeMotor(motor_id, model=..., position=..., temperature=..., enabled=..., online=..., track=...,
faulted=..., warning=..., params_supported=...), its ``log`` ("status_query", "clear", "enable",
"disable", "zero", ("cmd", kp)), ``frames`` ((position, kp, tau) per command), ``last_kp`` and
``handle(can_id, data) -> replies``; FakeBus; respond(); two_joint_config(); make_arm();
classify(); FAULT_CODE / FAULT_TEXT; FULL_TURN_SUPPORTED / WARNING_SUPPORTED. Everything else is
Damiao detail (late, duplicate, drop, registers, flash, power_cycle, ids_apply_at_once,
make_joint, wired, register_reply).
"""

from metal_arm import protocol as p
from metal_arm._sim_motor import (FAULT_CODE, FAULT_TEXT, FULL_TURN_SUPPORTED, WARNING_SUPPORTED,  # noqa: F401
                                  classify, feedback_bytes, register_reply, respond)
from metal_arm._sim_motor import SimBus as FakeBus
from metal_arm._sim_motor import SimMotor as FakeMotor


def make_joint(motor_id, model="jlo", name="", **kwargs):
    """A JointConfig for one motor."""
    from metal_arm.config import JointConfig
    defaults = dict(lo=-1.0, hi=1.0, kp=20.0, kd=0.6)
    defaults.update(kwargs)
    return JointConfig(motor_id=motor_id, model=model, name=name, **defaults)


def wired(*fakes, host_id=p.DEFAULT_MASTER_OFFSET, **joint_kwargs):
    """FakeBus with these motors and a Motor for each, dispatched the way the arm does it
    (motor.route then handle_frame). Returns (motors, bus); one fake returns (motor, bus)."""
    from metal_arm.motor import Motor, route
    bus = FakeBus(fakes)
    motors = {f.motor_id: Motor(make_joint(f.motor_id, f.model, **joint_kwargs), bus, host_id)
              for f in fakes}

    def dispatch(can_id, data):
        m = motors.get(route(can_id, data, host_id))
        if m is not None:
            m.handle_frame(can_id, data)

    bus.set_recv_callback(dispatch)
    if len(fakes) == 1:
        return motors[fakes[0].motor_id], bus
    return motors, bus


def two_joint_config(**overrides):
    """The shared two-joint test arm (same limits/gains/protection as the other SDK's), both
    joints on the default model so FakeMotor(n) scales like the joint."""
    from metal_arm.config import ArmConfig, JointConfig
    cfg = dict(control_rate_hz=200, max_velocity=50.0, joints=[
        JointConfig(motor_id=1, name="j1", lo=-2.0, hi=2.0, kp=60, kd=2.0),
        JointConfig(motor_id=2, name="gripper", lo=-1.6, hi=-0.04, kp=20, kd=0.5, can_timeout_ms=500,
                    max_temperature_c=55, enable_temperature_c=45, return_to_rest=False,
                    release_on_fault=True),
    ])
    cfg.update(overrides)
    return ArmConfig(**cfg)


def make_arm(motors=None, **overrides):
    """(arm, bus, {motor_id: FakeMotor}) for two_joint_config(**overrides); default motors sit at
    0.3 and -0.5 rad."""
    from metal_arm.arm import Arm
    motors = motors or [FakeMotor(1, position=0.3), FakeMotor(2, position=-0.5)]
    bus = FakeBus(motors)
    arm = Arm(two_joint_config(**overrides), bus)
    return arm, bus, {m.motor_id: m for m in motors}
