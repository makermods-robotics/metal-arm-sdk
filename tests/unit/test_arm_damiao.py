"""Arm behaviour that needs the Damiao motor: an id change that the arm keeps routing afterwards."""

from dataclasses import replace

import pytest

from fakes import FakeMotor, make_arm, two_joint_config
from metal_arm.arm import Arm, ArmState


@pytest.fixture(autouse=True)
def fast_param_probe(monkeypatch):
    monkeypatch.setattr(Arm, "PARAM_TIMEOUT", 0.02)


@pytest.mark.parametrize("explicit_reply_id", [False, True])
def test_set_motor_id_reroutes_the_joint_to_the_new_id(explicit_reply_id):
    joints = two_joint_config().joints
    if explicit_reply_id:
        joints = [replace(joints[0], reply_id=0x11), joints[1]]
    arm, bus, fakes = make_arm(joints=joints, required_motor_ids=[1, 2])
    arm.connect()
    arm.set_motor_id(1, 9)
    assert (fakes[1].send_id, fakes[1].reply_id) == (9, 0x19)
    assert arm.config.joints[0].motor_id == 9 and arm.config.required_motor_ids == [9, 2]
    assert arm.config.joints[0].reply_id == (0x19 if explicit_reply_id else None)
    assert arm.motors[0].motor_id == 9 and arm.motors[0].reply_id == 0x19
    assert arm.refresh() == [True, True]
    assert arm.get_joint_positions() == pytest.approx([0.3, -0.5], abs=1e-3)
    arm.enable(start_loop=False)
    assert arm.state is ArmState.ENABLED and fakes[1].enabled
    arm.disable()
    arm.disconnect()


def test_set_motor_id_while_disconnected_opens_and_closes_the_bus():
    arm, bus, fakes = make_arm([FakeMotor(1, position=0.3), FakeMotor(2, position=-0.5)])
    arm.set_motor_id(2, 5)
    assert not bus.is_open and fakes[2].send_id == 5
    assert [m.motor_id for m in arm.motors] == [1, 5]
    arm.connect()
    assert arm.get_joint_positions() == pytest.approx([0.3, -0.5], abs=1e-3)
    arm.disconnect()


def test_enable_and_disable_with_the_late_duplicating_gripper():
    # The gripper withholds each reply until another frame is on the bus, then sends it twice.
    arm, _, fakes = make_arm([FakeMotor(1, position=0.3), FakeMotor(2, position=-0.5, late=True, duplicate=True)])
    arm.connect()
    arm.enable(start_loop=False)
    assert arm.state is ArmState.ENABLED and fakes[1].enabled and fakes[2].enabled
    assert arm.get_faults() == [0, 0]   # status "enabled" is not a fault code
    arm.disable()
    assert arm.unconfirmed_disable == [] and not fakes[1].enabled and not fakes[2].enabled
