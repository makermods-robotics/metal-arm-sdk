import math

import pytest

from fakes import FakeBus, FakeMotor, feedback_bytes, make_joint, register_reply, wired
from metal_arm import motor as m
from metal_arm import protocol as p
from metal_arm.errors import ConnectError, ParamTimeout, StateError
from metal_arm.motor import Motor, route
from metal_arm.transport.base import CanId


def test_seam_constants():
    assert m.MODELS == ("jlo", "j2", "jhi") and m.DEFAULT_MODEL in m.MODELS
    assert m.KP_RANGE == (0.0, 500.0) and m.KD_RANGE == (0.0, 5.0)
    assert m.ID_RANGE == (1, 15) and m.DEFAULT_HOST_ID == 0x10


def test_reply_id_defaults_to_host_plus_id_or_joint_setting():
    bus = FakeBus([])
    assert Motor(make_joint(3), bus, 0x10).reply_id == 0x13
    assert Motor(make_joint(3, reply_id=0x23), bus, 0x10).reply_id == 0x23
    with pytest.raises(ValueError):
        Motor(make_joint(3, model="RS00"), bus, 0x10)
    with pytest.raises(ValueError):
        Motor(make_joint(16), bus, 0x10)


def test_every_command_is_a_standard_frame_to_the_send_id():
    motor, bus = wired(FakeMotor(4))
    motor.enable()
    motor.send_command(0.1, 0, 10, 1, 0)
    motor.disable()
    motor.set_zero()
    motor.clear_fault()
    assert bus.sent_extended == [False] * 5
    assert [cid for cid, _ in bus.sent] == [4] * 5
    assert [d[7] for _, d in bus.sent if d[:7] == p.COMMAND_PREFIX] == [0xFC, 0xFD, 0xFE, 0xFB]


def test_feedback_from_command_replies_and_cache():
    fake = FakeMotor(2, model="j2", enabled=True, temperature=41, rotor_temperature=48)
    motor, _ = wired(fake)
    assert motor.feedback is None and motor.feedback_age == math.inf and motor.status_age == math.inf
    motor.send_command(0.5, 0, 50, 1, 0)
    fb = motor.feedback
    assert fb.position == pytest.approx(0.5, abs=1e-3)
    assert (fb.temperature, fb.temperatures) == (48, {"mos": 41, "rotor": 48})
    assert fb.enabled is True and fb.warning is False and fb.status == 1
    assert motor.feedback_sequence == 1 and motor.feedback_age < 1
    assert motor.status == m.MotorStatus(2, False, 1, "", True)
    motor.reset_cache()
    assert motor.feedback is None and motor.status is None


def test_read_state_is_fresh_and_read_only():
    fake = FakeMotor(3, position=0.7, enabled=True)
    motor, bus = wired(fake)
    fb = motor.read_state()
    assert fb.position == pytest.approx(0.7, abs=1e-3)
    assert fake.enabled and "clear" not in fake.log
    assert bus.sent[0] == p.encode_refresh(3)


def test_check_reports_without_clearing_and_latches():
    fake = FakeMotor(7, model="jhi", fault=p.STATUS_COMM_LOST)
    motor, _ = wired(fake)
    status = motor.check()
    assert status.has_fault and status.code == 0xD and status.enabled is False
    assert status.text == "fault 0xD (communication lost)"
    assert motor.fault_latched == "fault 0xD (communication lost)"
    assert fake.fault == 0xD and "clear" not in fake.log
    motor.read_state()                        # never refused: the refresh clears nothing
    assert fake.fault == 0xD
    assert motor.health_problem() == "fault 0xD (communication lost)"


def test_clear_fault_sends_fb_and_resets_latch():
    fake = FakeMotor(1, fault=0x9)
    motor, bus = wired(fake)
    motor.check()
    motor.clear_fault()
    assert bus.sent[-1] == (1, p.COMMAND_PREFIX + bytes([0xFB]))
    assert motor.fault_latched is None and fake.fault == 0
    assert not motor.check().has_fault


def test_confirms_from_status_nibble():
    fake = FakeMotor(1)
    motor, _ = wired(fake)
    assert not motor.confirms("enabled") and not motor.confirms("disabled")
    motor.enable()
    assert motor.confirms("enabled") and not motor.confirms("disabled")
    motor.disable()
    assert motor.confirms("disabled")
    fake.fault = 0xE
    motor.check()
    assert motor.confirms("disabled") and not motor.confirms("enabled")
    with pytest.raises(ValueError):
        motor.confirms("on")


def test_disable_confirmed():
    fake = FakeMotor(1, enabled=True)
    motor, _ = wired(fake)
    assert motor.disable_confirmed() and not fake.enabled
    offline, _ = wired(FakeMotor(2, online=False))
    assert not offline.disable_confirmed(timeout=0.005, attempts=1)


def test_health_problem_on_persistent_disabled_status():
    fake = FakeMotor(1, enabled=True)
    motor, _ = wired(fake)
    motor.enable()
    assert motor.health_problem() is None
    fake.enabled = False
    motor.check()
    problems = [motor.health_problem() for _ in range(Motor.STATUS_DISABLED_TICKS)]
    assert problems[:-1] == [None] * (Motor.STATUS_DISABLED_TICKS - 1)
    assert "disabled" in problems[-1]


def test_missing_motor_times_out():
    motor, _ = wired(FakeMotor(1, online=False))
    with pytest.raises(ParamTimeout):
        motor.read_state(timeout=0.005, attempts=2)
    with pytest.raises(ParamTimeout):
        motor.check(timeout=0.005, attempts=1)
    with pytest.raises(ParamTimeout):
        motor.read_param("ctrl_mode", timeout=0.005)


def test_a_dropped_request_is_resent():
    fake = FakeMotor(2, position=0.3, drop=1)
    motor, bus = wired(fake)
    assert motor.read_state(timeout=0.01).position == pytest.approx(0.3, abs=1e-3)
    assert bus.sent.count(p.encode_refresh(2)) == 2


def test_replies_on_another_reply_id_are_ignored():
    motor, bus = wired(FakeMotor(1))
    motor.handle_frame(0x12, feedback_bytes(1, pos=1.0))    # right nibble, wrong master id
    motor.handle_frame(0x11, feedback_bytes(2, pos=1.0))    # right master id, wrong nibble
    motor.handle_frame(CanId(0x11, extended=True), feedback_bytes(1, pos=1.0))
    assert motor.feedback is None


# ---- the Metal gripper's late / duplicate replies ----

def test_late_reply_is_flushed_out():
    fake = FakeMotor(7, model="jhi", position=0.4, late=True)
    motor, bus = wired(fake)
    assert motor.read_state().position == pytest.approx(0.4, abs=1e-3)
    flushes = [f for f in bus.sent if f == p.encode_refresh(6)]
    assert flushes, "a held reply needs another frame on the bus"


@pytest.mark.parametrize("duplicate", [False, True])
def test_stale_held_reply_is_superseded_by_the_fresh_one(duplicate):
    fake = FakeMotor(7, model="jhi", position=0.9, late=True, duplicate=duplicate)
    fake.held = [(0x17, feedback_bytes(7, pos=0.1, model="jhi"))]   # answer to an earlier request
    motor, _ = wired(fake)
    for position in (0.9, 0.7, 0.5):
        fake.position = position
        assert motor.read_state().position == pytest.approx(position, abs=1e-3)


def test_punctual_motor_needs_one_trailing_flush_only():
    motor, bus = wired(FakeMotor(3))
    motor.read_state()
    assert bus.sent == [p.encode_refresh(3), p.encode_refresh(2)]


def test_flush_refreshes_the_neighbour_and_updates_it():
    fakes = [FakeMotor(6, model="jhi", position=-0.2), FakeMotor(7, model="jhi", position=0.4, late=True)]
    motors, _ = wired(*fakes)
    assert motors[7].read_state().position == pytest.approx(0.4, abs=1e-3)
    assert motors[6].feedback.position == pytest.approx(-0.2, abs=1e-3)


def test_stale_register_echo_is_skipped():
    fake = FakeMotor(7, model="jhi", late=True)
    fake.held = [(0x17, register_reply(7, p.OP_WRITE, p.Reg.TIMEOUT, (200).to_bytes(4, "little")))]
    motor, _ = wired(fake)
    assert motor.write_param("timeout", 10_000) == 10_000
    assert fake.watchdog_ticks == 10_000


def test_register_write_that_does_not_stick_fails_closed():
    fake = FakeMotor(1)
    motor, bus = wired(fake)
    original = fake._param_frame

    def stubborn(data):
        replies = original(data)
        fake.registers[p.Reg.TIMEOUT] = 0
        return replies

    fake._param_frame = stubborn
    with pytest.raises(StateError):
        motor.write_param("timeout", 100)


# ---- registers, validation, watchdog ----

def test_read_and_write_params_by_name_and_index():
    fake = FakeMotor(5, model="jhi")
    motor, _ = wired(fake)
    assert motor.read_param("ctrl_mode") == 1
    assert motor.read_param(p.Reg.VMAX) == pytest.approx(30.0)
    assert motor.read_param("PMAX") == pytest.approx(6.28)
    assert motor.write_param("kp_apr", 12.5) == 12.5
    assert "save" not in fake.log
    assert set(Motor.PARAMS) == set(p.REGISTERS)
    assert Motor.PARAMS["timeout"] == m.ParamSpec(9, "u32", p.REGISTERS["timeout"].doc)
    with pytest.raises(ValueError):
        motor.read_param("nope")


def test_write_param_save_requires_disabled_and_stores_flash():
    fake = FakeMotor(1)
    motor, _ = wired(fake)
    motor.write_param("timeout", 40, save=True)
    assert fake.flash[p.Reg.TIMEOUT] == 40 and "save" in fake.log
    fake.enabled = True
    with pytest.raises(StateError):
        motor.write_param("timeout", 60, save=True)
    assert fake.flash[p.Reg.TIMEOUT] == 40 and fake.watchdog_ticks == 40   # refused before the RAM write


def test_validate_accepts_matching_motor():
    motor, _ = wired(FakeMotor(2, model="j2"))
    motor.validate()


def test_validate_rejects_non_mit_mode():
    motor, _ = wired(FakeMotor(7, model="jhi", registers={p.Reg.CTRL_MODE: p.MODE_FORCE_POS}))
    with pytest.raises(ConnectError, match="not 1 \\(MIT\\)"):
        motor.validate()


def test_validate_rejects_scaling_mismatch_and_names_the_fix():
    fake = FakeMotor(2, model="j2")
    bus = FakeBus([fake])
    motor = Motor(make_joint(2, "jlo", name="shoulder_lift"), bus, 0x10)
    bus.set_recv_callback(motor.handle_frame)
    with pytest.raises(ConnectError, match="model: j2"):
        motor.validate()
    fake.registers[p.Reg.TMAX] = 7.0
    with pytest.raises(ConnectError, match="registers 21-23"):
        motor.validate()


def test_validate_offline_is_a_connect_error():
    motor, _ = wired(FakeMotor(1, online=False))
    with pytest.raises(ConnectError):
        motor.validate(timeout=0.005)


def test_watchdog_is_set_in_ram_and_restored():
    fake = FakeMotor(7, model="jhi", registers={p.Reg.TIMEOUT: 0})
    motor, _ = wired(fake)
    token = motor.arm_watchdog(500)
    assert token == 0 and fake.watchdog_ticks == 10_000
    assert fake.flash[p.Reg.TIMEOUT] == 0 and "save" not in fake.log
    motor.restore_watchdog(token)
    assert fake.watchdog_ticks == 0
    motor.restore_watchdog(None)
    with pytest.raises(ValueError):
        motor.arm_watchdog(0)


# ---- set_id ----

def test_set_id_writes_saves_and_verifies():
    fake = FakeMotor(3)
    motor, bus = wired(fake)
    bus.set_recv_callback(motor.handle_frame)
    motor.set_id(9)
    assert (motor.motor_id, motor.reply_id) == (9, 0x19)
    assert (fake.send_id, fake.reply_id) == (9, 0x19)
    assert fake.flash[p.Reg.ESC_ID] == 9 and fake.flash[p.Reg.MST_ID] == 0x19
    writes = [e for e in fake.log if isinstance(e, tuple) and e[0] == "write"]
    assert writes == [("write", p.Reg.MST_ID), ("write", p.Reg.ESC_ID)]
    assert "save" in fake.log
    assert motor.read_state().motor_id == 9
    assert bus.recv_callback == motor.handle_frame   # the backend callback is restored


@pytest.mark.parametrize("echo", ["new", "none"])
def test_set_id_when_the_id_write_echo_names_the_new_id_or_is_missing(echo):
    fake = FakeMotor(7, esc_echo=echo)
    motor, bus = wired(fake)
    bus.set_recv_callback(motor.handle_frame)
    motor.set_id(8)
    assert (motor.motor_id, motor.reply_id) == (8, 0x18)
    assert fake.flash[p.Reg.ESC_ID] == 8 and fake.flash[p.Reg.MST_ID] == 0x18


def test_set_id_when_ids_apply_only_after_power_cycle():
    fake = FakeMotor(3, ids_apply_at_once=False)
    motor, bus = wired(fake)
    bus.set_recv_callback(motor.handle_frame)
    with pytest.raises(StateError, match="power-cycle"):
        motor.set_id(9, timeout=0.02)
    assert fake.flash[p.Reg.ESC_ID] == 9
    fake.power_cycle()
    assert (fake.send_id, fake.reply_id) == (9, 0x19)


def test_set_id_refuses_an_id_in_use():
    fakes = [FakeMotor(3), FakeMotor(4)]
    motors, bus = wired(*fakes)
    with pytest.raises(StateError, match="already used"):
        motors[3].set_id(4, timeout=0.02)
    assert not any(isinstance(e, tuple) and e[0] == "write" for e in fakes[0].log)
    assert motors[3].motor_id == 3


def test_set_id_refuses_an_enabled_motor_and_bad_ids():
    fake = FakeMotor(3, enabled=True)
    motor, _ = wired(fake)
    with pytest.raises(StateError, match="disable"):
        motor.set_id(9, timeout=0.02)
    assert fake.send_id == 3
    with pytest.raises(ValueError):
        motor.set_id(16)
    motor.set_id(3)   # no-op


def test_set_id_on_the_late_gripper():
    fake = FakeMotor(7, model="jhi", late=True, duplicate=True)
    motor, bus = wired(fake)
    bus.set_recv_callback(motor.handle_frame)
    motor.set_id(8)
    assert (fake.send_id, fake.flash[p.Reg.ESC_ID]) == (8, 8)


# ---- routing ----

def test_route():
    assert route(0x17, feedback_bytes(7, status=1), 0x10) == 7
    assert route(0x13, register_reply(3, p.OP_READ, 10, bytes(4)), 0x10) == 3
    assert route(0x23, feedback_bytes(3), 0x20) == 3
    assert route(0x55, feedback_bytes(3), 0x10) == 3     # custom reply id: nibble
    assert route(0x07, feedback_bytes(7), 0x10) is None   # a command to motor 7
    assert route(0x7FF, p.encode_refresh(7)[1], 0x10) is None
    assert route(0x307, feedback_bytes(7), 0x10) is None
    assert route(CanId(0x17, extended=True), feedback_bytes(7), 0x10) is None
    assert route(0x17, feedback_bytes(7)[:5], 0x10) is None
    assert route(0x17, bytes([0x27]) + feedback_bytes(7)[1:], 0x10) is None


def test_poll_status_is_a_read_only_refresh():
    fake = FakeMotor(2, faulted=True)
    motor, bus = wired(fake)
    motor.poll_status()
    assert bus.sent == [p.encode_refresh(2)]
    assert fake.log == ["status_query"] and fake.faulted
    assert motor.status.has_fault and motor.status.text == "fault 0xC (rotor overtemperature)"


def test_fake_classify():
    from fakes import classify
    assert classify(*p.encode_refresh(3)) == ("status_query", 3)
    assert classify(*p.encode_read_register(3, 9)) == ("param_read", 3)
    assert classify(*p.encode_write_register(3, 9, 1)) == ("param_write", 3)
    assert classify(*p.encode_save(3)) == ("param_save", 3)
    assert classify(*p.encode_clear_error(3)) == ("clear", 3)
    assert classify(*p.encode_enable(3)) == ("enable", 3)
    assert classify(*p.encode_mit(3, 0, 0, 1, 0, 0, p.MODELS["jlo"])) == ("command", 3)
