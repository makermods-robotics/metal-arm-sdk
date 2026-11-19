import math
import struct

import pytest

from fakes import feedback_bytes
from metal_arm import protocol as p
from metal_arm.transport.base import CanId

# Golden MIT frames generated once with the LeRobot Damiao encoder
# (DamiaoMotorsBus._encode_mit_packet, makermods-robotics fork, METAL_* motor types).
# (model, kp, kd, position_deg, velocity_deg_s, torque, payload)
MIT_GOLDEN = [
    ("jlo", 100.0, 5.0, 0.0, 0.0, 0.0, "7fff7ff333fff7ff"),
    ("jlo", 100.0, 5.0, 45.0, 30.0, 1.5, "900186a333fff865"),
    ("j2", 100.0, 5.0, -90.0, -10.0, -20.0, "5ffb7db333fff6aa"),
    ("jhi", 20.0, 0.6, 120.0, 200.0, 0.3, "aaaf8ed0a31eb81e"),
    ("jhi", 0.0, 0.0, -400.0, -9999.0, -99.0, "0000000000000000"),   # clamped low
    ("jlo", 500.0, 5.0, 359.0, 572.0, 30.0, "ffb4ffbfffffffff"),    # clamped high
]

# Golden decodes from DamiaoMotorsBus._decode_motor_state: (model, payload, status, id,
# position_deg, velocity_deg_s, torque, t_mos, t_rotor).
STATE_GOLDEN = [
    ("jlo", "1180007ff7ff1e20", 1, 1, 0.005490463040242452, -0.13991643348738003, -0.0073260073260073, 30, 32),
    ("j2", "02a3d71234567890", 0, 2, 100.7554872514522, -491.5264308411677, -54.94505494505495, 120, 144),
    ("jhi", "d7ffff0000002d2f", 0xD, 7, 359.817495342157, -1718.8733853924696, -20.0, 45, 47),
]


@pytest.mark.parametrize("model, kp, kd, pos, vel, tau, payload", MIT_GOLDEN)
def test_mit_encoder_matches_lerobot_bytes(model, kp, kd, pos, vel, tau, payload):
    cid, data = p.encode_mit(3, math.radians(pos), math.radians(vel), kp, kd, tau, p.MODELS[model])
    assert cid == 3
    assert data.hex() == payload


@pytest.mark.parametrize("model, payload, status, motor_id, pos, vel, tau, t_mos, t_rotor", STATE_GOLDEN)
def test_state_decoder_matches_lerobot(model, payload, status, motor_id, pos, vel, tau, t_mos, t_rotor):
    msg = p.decode(0x10 + motor_id, bytes.fromhex(payload), p.MODELS[model])
    assert isinstance(msg, p.StateFrame)
    assert (msg.motor_id, msg.status) == (motor_id, status)
    assert math.degrees(msg.position) == pytest.approx(pos, abs=1e-9)
    assert math.degrees(msg.velocity) == pytest.approx(vel, abs=1e-9)
    assert msg.torque == pytest.approx(tau, abs=1e-9)
    assert (msg.t_mos, msg.t_rotor) == (t_mos, t_rotor)


@pytest.mark.parametrize("encode, last", [
    (p.encode_enable, 0xFC), (p.encode_disable, 0xFD), (p.encode_set_zero, 0xFE), (p.encode_clear_error, 0xFB),
])
def test_commands_are_seven_ff_and_a_command_byte_to_the_send_id(encode, last):
    assert encode(7) == (7, bytes([0xFF] * 7 + [last]))


def test_command_mode_offsets():
    assert p.encode_clear_error(7, p.MODE_FORCE_POS)[0] == 0x307
    assert p.encode_enable(2, p.MODE_POS_VEL)[0] == 0x102


@pytest.mark.parametrize("bad", [0, 16, 0x7FF])
def test_motor_id_range(bad):
    with pytest.raises(ValueError):
        p.encode_enable(bad)
    with pytest.raises(ValueError):
        p.encode_refresh(bad)


def test_refresh_and_register_frames():
    assert p.encode_refresh(7) == (0x7FF, bytes.fromhex("0700cc0000000000"))
    assert p.encode_read_register(3, p.Reg.CTRL_MODE) == (0x7FF, bytes.fromhex("0300330a00000000"))
    assert p.encode_write_register(7, p.Reg.TIMEOUT, 10_000) == (0x7FF, bytes.fromhex("0700550910270000"))
    assert p.encode_write_register(1, p.Reg.PMAX, 6.28) == (0x7FF, bytes.fromhex("01005515") + struct.pack("<f", 6.28))
    assert p.encode_save(5) == (0x7FF, bytes.fromhex("0500aa0100000000"))


def test_register_dtypes_follow_dm_can():
    u32 = {7, 8, 9, 10, 13, 14, 15, 16, 35, 36}
    for spec in p.REGISTERS.values():
        assert spec.dtype == ("u32" if spec.index in u32 else "f"), spec
    assert p.register_dtype(59) == "f"
    with pytest.raises(ValueError):
        p.encode_write_register(1, p.Reg.PMAX, math.nan)
    with pytest.raises(ValueError):
        p.pack_value(1, "u8")


def test_mit_rejects_non_finite():
    with pytest.raises(ValueError):
        p.encode_mit(1, math.nan, 0, 0, 0, 0, p.MODELS["jlo"])
    with pytest.raises(ValueError):
        p.encode_mit(1, 0, 0, math.inf, 0, 0, p.MODELS["jlo"])


def test_mit_frame_never_reads_as_a_command():
    model = p.MODELS["jlo"]
    _, data = p.encode_mit(1, model.p_max, model.v_max, p.KP_MAX, p.KD_MAX, model.t_max, model)
    assert data[:7] != p.COMMAND_PREFIX
    assert data == bytes.fromhex("fffeffffffffffff")


def test_register_reply_decodes_and_echoes_header():
    data = bytes.fromhex("0700330a01000000")
    msg = p.decode(0x17, data, p.MODELS["jhi"])
    assert msg == p.RegisterReply(7, p.OP_READ, p.Reg.CTRL_MODE, bytes([1, 0, 0, 0]))
    assert msg.value() == 1
    pmax = p.decode(0x17, bytes.fromhex("07003315") + struct.pack("<f", 6.28), p.MODELS["jhi"])
    assert pmax.value() == pytest.approx(6.28)


def test_disabled_state_frame_that_looks_like_a_register_reply_is_a_register_reply():
    # status 0, id 7, position bytes 00 55: identical to a write echo header.
    data = bytes.fromhex("0700550910270000")
    assert isinstance(p.decode(0x17, data, p.MODELS["jhi"]), p.RegisterReply)


def test_foreign_frames_are_ignored():
    model = p.MODELS["jlo"]
    data = feedback_bytes(1, status=1)
    assert p.decode(CanId(0x11, extended=True), data, model) is None
    assert p.decode(0x01, data, model) is None           # someone's command to motor 1
    assert p.decode(0x7FF, data, model) is None          # someone's register request
    assert p.decode(0x101, data, model) is None          # pos-vel command
    assert p.decode(0x11, data[:7], model) is None       # short frame
    assert p.decode(0x11, bytes([0x21]) + data[1:], model) is None   # unknown status 2


def test_state_round_trip_and_status_text():
    model = p.MODELS["j2"]
    msg = p.decode(0x12, feedback_bytes(2, pos=-1.0, vel=2.0, tau=50.0, temp=41, rotor_temp=55,
                                        status=0xC, model="j2"), model)
    assert msg.position == pytest.approx(-1.0, abs=2e-4)
    assert msg.velocity == pytest.approx(2.0, abs=6e-3)
    assert msg.torque == pytest.approx(50.0, abs=0.06)
    assert (msg.t_mos, msg.t_rotor, msg.status) == (41, 55, 0xC)
    assert p.status_text(0xC) == "fault 0xC (rotor overtemperature)"
    assert p.status_text(1) == "enabled" and p.status_text(0) == "disabled"
    assert p.status_text(None) == "status unknown"
    assert p.status_text(0x5) == "fault 0x5 (unknown fault)"
    assert p.is_fault_status(0xD) and not p.is_fault_status(1)


def test_matching_model():
    assert p.matching_model(6.28, 10.0, 120.0) == "j2"
    assert p.matching_model(*struct.unpack("<3f", struct.pack("<3f", 6.28, 30.0, 20.0))) == "jhi"
    assert p.matching_model(12.5, 30.0, 10.0) is None
