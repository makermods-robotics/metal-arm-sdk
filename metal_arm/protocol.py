"""Damiao MIT wire protocol (classic CAN, standard 11-bit ids, 1 Mbit/s): pure encode/decode, no I/O.

Frames to a motor go to its send id (ESC_ID, 1..15 here). Its replies come back on its master id
(MST_ID, conventionally 0x10 + send id). Register access and the read-only state refresh go to the
shared id 0x7FF with the target send id in bytes 0-1.

Sources: Damiao DM_CAN reference driver and motor manual (register map, 0x33/0x55/0xAA, 0xCC
refresh, status nibble); the vendor's Metal C++ SDK (status codes, temperatures in bytes 6/7).
The MIT packing matches the LeRobot Damiao encoder byte for byte (tests/unit/test_protocol.py).
"""

import math
import struct
from dataclasses import dataclass
from typing import NamedTuple, Optional, Union

PARAM_CAN_ID = 0x7FF          # register read/write/save and state refresh
ID_MIN, ID_MAX = 1, 15        # the state frame carries only the low nibble of the id
DEFAULT_MASTER_OFFSET = 0x10  # master (reply) id = 0x10 + send id on every Metal motor
KP_MIN, KP_MAX = 0.0, 500.0
KD_MIN, KD_MAX = 0.0, 5.0

# Command bytes (byte 7 after seven 0xFF bytes, sent to the send id + the mode offset).
CMD_ENABLE = 0xFC
CMD_DISABLE = 0xFD
CMD_SET_ZERO = 0xFE
CMD_CLEAR_ERROR = 0xFB
COMMAND_PREFIX = bytes([0xFF] * 7)

# Register opcodes (byte 2 of a frame to 0x7FF) and the refresh request.
OP_READ = 0x33
OP_WRITE = 0x55
OP_SAVE = 0xAA
OP_REFRESH = 0xCC
REGISTER_OPS = (OP_READ, OP_WRITE, OP_SAVE)

# Control modes (CTRL_MODE register). Commands in mode m go to send id + MODE_OFFSETS[m].
MODE_MIT = 1
MODE_POS_VEL = 2
MODE_VEL = 3
MODE_FORCE_POS = 4
MODE_OFFSETS = {MODE_MIT: 0x000, MODE_POS_VEL: 0x100, MODE_VEL: 0x200, MODE_FORCE_POS: 0x300}

# Status nibble (byte 0 high nibble of every state frame).
STATUS_DISABLED = 0x0
STATUS_ENABLED = 0x1
STATUS_COMM_LOST = 0xD
STATUS_NAMES = {
    0x0: "disabled",
    0x1: "enabled",
    0x8: "overvoltage",
    0x9: "undervoltage",
    0xA: "overcurrent",
    0xB: "MOS overtemperature",
    0xC: "rotor overtemperature",
    0xD: "communication lost",
    0xE: "overload",
}

TIMEOUT_TICKS_PER_MS = 20  # TIMEOUT register counts 50 us ticks; 0 disables the watchdog


def is_fault_status(status: int) -> bool:
    return status not in (STATUS_DISABLED, STATUS_ENABLED)


def status_text(status: Optional[int]) -> str:
    if status is None:
        return "status unknown"
    if not is_fault_status(status):
        return STATUS_NAMES[status]
    return f"fault 0x{status:X} ({STATUS_NAMES.get(status, 'unknown fault')})"


@dataclass(frozen=True)
class MotorModel:
    """MIT mapping ranges, which must equal the motor's PMAX/VMAX/TMAX registers (21/22/23)."""
    p_max: float   # rad
    v_max: float   # rad/s
    t_max: float   # N·m


# Metal Arm motors, source: LeRobot damiao tables.py METAL_JLO / METAL_J2 / METAL_JHI.
MODELS = {
    "jlo": MotorModel(6.28, 10.0, 30.0),    # shoulder_pan, elbow_flex, wrist_flex
    "j2": MotorModel(6.28, 10.0, 120.0),    # shoulder_lift
    "jhi": MotorModel(6.28, 30.0, 20.0),    # wrist_yaw, wrist_roll, gripper
}


class Reg:
    MST_ID = 7       # u32 master (reply) id
    ESC_ID = 8       # u32 send id
    TIMEOUT = 9      # u32 communication watchdog, 50 us ticks; 0 = off
    CTRL_MODE = 10   # u32 1 MIT, 2 pos-vel, 3 vel, 4 force-position
    PMAX = 21        # f32 rad
    VMAX = 22        # f32 rad/s
    TMAX = 23        # f32 N·m


class RegisterSpec(NamedTuple):
    index: int
    dtype: str   # "f" (float32) or "u32"
    doc: str


# Damiao register map (DM_CAN DM_variable). u32 registers: 7-10, 13-16, 35-36; the rest float32.
REGISTERS = {
    "uv_value": RegisterSpec(0, "f", "undervoltage threshold (V)"),
    "kt_value": RegisterSpec(1, "f", "torque constant"),
    "ot_value": RegisterSpec(2, "f", "overtemperature threshold (°C)"),
    "oc_value": RegisterSpec(3, "f", "overcurrent threshold"),
    "acc": RegisterSpec(4, "f", "acceleration"),
    "dec": RegisterSpec(5, "f", "deceleration"),
    "max_spd": RegisterSpec(6, "f", "maximum speed"),
    "mst_id": RegisterSpec(7, "u32", "master (reply) CAN id"),
    "esc_id": RegisterSpec(8, "u32", "send CAN id"),
    "timeout": RegisterSpec(9, "u32", "communication watchdog in 50 us ticks; 0 = off"),
    "ctrl_mode": RegisterSpec(10, "u32", "control mode: 1 MIT, 2 pos-vel, 3 vel, 4 force-position"),
    "damp": RegisterSpec(11, "f", "damping"),
    "inertia": RegisterSpec(12, "f", "inertia"),
    "hw_ver": RegisterSpec(13, "u32", "hardware version"),
    "sw_ver": RegisterSpec(14, "u32", "firmware version"),
    "sn": RegisterSpec(15, "u32", "serial number"),
    "npp": RegisterSpec(16, "u32", "motor pole pairs"),
    "rs": RegisterSpec(17, "f", "phase resistance"),
    "ls": RegisterSpec(18, "f", "phase inductance"),
    "flux": RegisterSpec(19, "f", "flux linkage"),
    "gr": RegisterSpec(20, "f", "gear ratio"),
    "pmax": RegisterSpec(21, "f", "MIT position mapping range (rad)"),
    "vmax": RegisterSpec(22, "f", "MIT velocity mapping range (rad/s)"),
    "tmax": RegisterSpec(23, "f", "MIT torque mapping range (N·m)"),
    "i_bw": RegisterSpec(24, "f", "current loop bandwidth"),
    "kp_asr": RegisterSpec(25, "f", "speed loop Kp"),
    "ki_asr": RegisterSpec(26, "f", "speed loop Ki"),
    "kp_apr": RegisterSpec(27, "f", "position loop Kp"),
    "ki_apr": RegisterSpec(28, "f", "position loop Ki"),
    "ov_value": RegisterSpec(29, "f", "overvoltage threshold (V)"),
    "gref": RegisterSpec(30, "f", "gear efficiency"),
    "deta": RegisterSpec(31, "f", "speed loop damping"),
    "v_bw": RegisterSpec(32, "f", "speed loop filter bandwidth"),
    "iq_c1": RegisterSpec(33, "f", "current loop gain"),
    "vl_c1": RegisterSpec(34, "f", "speed loop gain"),
    "can_br": RegisterSpec(35, "u32", "CAN bitrate code (4 = 1 Mbit/s)"),
    "sub_ver": RegisterSpec(36, "u32", "sub version"),
    "u_off": RegisterSpec(50, "f", "U phase offset"),
    "v_off": RegisterSpec(51, "f", "V phase offset"),
    "k1": RegisterSpec(52, "f", "compensation factor 1"),
    "k2": RegisterSpec(53, "f", "compensation factor 2"),
    "m_off": RegisterSpec(54, "f", "angle offset"),
    "dir": RegisterSpec(55, "f", "direction"),
    "p_m": RegisterSpec(80, "f", "motor position (rad)"),
    "xout": RegisterSpec(81, "f", "output shaft position (rad)"),
}
REGISTER_DTYPES = {spec.index: spec.dtype for spec in REGISTERS.values()}
_FORMATS = {"f": "<f", "u32": "<I"}


def register_dtype(index: int) -> str:
    """Known registers use their own type; unknown indices are read as float32 (DM_CAN default)."""
    return REGISTER_DTYPES.get(index, "f")


def pack_value(value, dtype: str) -> bytes:
    if dtype not in _FORMATS:
        raise ValueError(f"unknown register dtype {dtype!r} (choices: f, u32)")
    if dtype == "f":
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("register values must be finite")
        return struct.pack("<f", value)
    return struct.pack("<I", int(value))


def unpack_value(raw: bytes, dtype: str):
    if dtype not in _FORMATS:
        raise ValueError(f"unknown register dtype {dtype!r} (choices: f, u32)")
    return struct.unpack(_FORMATS[dtype], bytes(raw))[0]


def _id(motor_id: int) -> int:
    if not ID_MIN <= motor_id <= ID_MAX:
        raise ValueError(f"Damiao motor id must be in [{ID_MIN}, {ID_MAX}], got {motor_id}")
    return motor_id


def _register_rid(rid: int) -> int:
    if not 0 <= rid <= 0xFF:
        raise ValueError(f"register index must be in [0, 255], got {rid}")
    return rid


# ---------------------------------------------------------------- encoders (can_id, payload)

def encode_command(motor_id: int, command: int, mode: int = MODE_MIT) -> tuple[int, bytes]:
    return _id(motor_id) + MODE_OFFSETS[mode], COMMAND_PREFIX + bytes([command])


def encode_enable(motor_id: int, mode: int = MODE_MIT) -> tuple[int, bytes]:
    return encode_command(motor_id, CMD_ENABLE, mode)


def encode_disable(motor_id: int, mode: int = MODE_MIT) -> tuple[int, bytes]:
    return encode_command(motor_id, CMD_DISABLE, mode)


def encode_set_zero(motor_id: int, mode: int = MODE_MIT) -> tuple[int, bytes]:
    return encode_command(motor_id, CMD_SET_ZERO, mode)


def encode_clear_error(motor_id: int, mode: int = MODE_MIT) -> tuple[int, bytes]:
    return encode_command(motor_id, CMD_CLEAR_ERROR, mode)


def encode_refresh(motor_id: int) -> tuple[int, bytes]:
    """Read-only state request: the motor answers with a state frame, torque state unchanged."""
    _id(motor_id)
    return PARAM_CAN_ID, bytes([motor_id & 0xFF, motor_id >> 8, OP_REFRESH, 0, 0, 0, 0, 0])


def encode_read_register(motor_id: int, rid: int) -> tuple[int, bytes]:
    _id(motor_id)
    return PARAM_CAN_ID, bytes([motor_id & 0xFF, motor_id >> 8, OP_READ, _register_rid(rid), 0, 0, 0, 0])


def encode_write_register(motor_id: int, rid: int, value, dtype: Optional[str] = None) -> tuple[int, bytes]:
    """Write to RAM only; the value is lost at power-off unless saved with encode_save()."""
    _id(motor_id)
    raw = pack_value(value, dtype or register_dtype(rid))
    return PARAM_CAN_ID, bytes([motor_id & 0xFF, motor_id >> 8, OP_WRITE, _register_rid(rid)]) + raw


def encode_save(motor_id: int) -> tuple[int, bytes]:
    """Store every register to flash (DM_CAN save_motor_param). Only honoured while disabled."""
    _id(motor_id)
    return PARAM_CAN_ID, bytes([motor_id & 0xFF, motor_id >> 8, OP_SAVE, 0x01, 0, 0, 0, 0])


def float_to_uint(value: float, lo: float, hi: float, bits: int) -> int:
    """Clamp then truncate, as the Damiao reference encoder does."""
    if not math.isfinite(value):
        raise ValueError("MIT commands must be finite")
    value = min(hi, max(lo, value))
    return int((value - lo) / (hi - lo) * ((1 << bits) - 1))


def uint_to_float(value: int, lo: float, hi: float, bits: int) -> float:
    return value / ((1 << bits) - 1) * (hi - lo) + lo


def encode_mit(motor_id: int, pos: float, vel: float, kp: float, kd: float, tau: float,
               model: MotorModel) -> tuple[int, bytes]:
    """MIT impedance command: tau_out = kp*(pos - q) + kd*(vel - dq) + tau. Motor coordinates, SI."""
    q = float_to_uint(pos, -model.p_max, model.p_max, 16)
    v = float_to_uint(vel, -model.v_max, model.v_max, 12)
    k = float_to_uint(kp, KP_MIN, KP_MAX, 12)
    d = float_to_uint(kd, KD_MIN, KD_MAX, 12)
    t = float_to_uint(tau, -model.t_max, model.t_max, 12)
    payload = bytearray([q >> 8, q & 0xFF, v >> 4, ((v & 0xF) << 4) | (k >> 8), k & 0xFF,
                         d >> 4, ((d & 0xF) << 4) | (t >> 8), t & 0xFF])
    if payload[:7] == COMMAND_PREFIX:
        # Seven 0xFF bytes would read as a command (enable/disable/zero/clear). One position LSB
        # (0.19 mrad) lower keeps the frame an MIT command.
        payload[1] = 0xFE
    return _id(motor_id), bytes(payload)


# ---------------------------------------------------------------- decoders

@dataclass(frozen=True)
class StateFrame:
    motor_id: int        # low nibble of the send id
    status: int          # see STATUS_NAMES
    position: float      # rad, motor coordinates
    velocity: float      # rad/s
    torque: float        # N·m
    t_mos: float         # °C
    t_rotor: float       # °C


@dataclass(frozen=True)
class RegisterReply:
    motor_id: int        # send id echoed in bytes 0-1
    op: int              # OP_READ / OP_WRITE / OP_SAVE
    rid: int
    raw: bytes           # bytes 4-7

    def value(self, dtype: Optional[str] = None):
        return unpack_value(self.raw, dtype or register_dtype(self.rid))


def is_register_reply(data: bytes) -> bool:
    """Register replies echo the request header [id_lo, id_hi, op, rid]. A disabled state frame whose
    position high byte is 0 and low byte an opcode looks the same; it is taken as a register reply
    and that one sample is dropped."""
    return (len(data) == 8 and data[1] == 0 and data[2] in REGISTER_OPS
            and ID_MIN <= data[0] <= ID_MAX)


def is_command_id(can_id: int) -> bool:
    """A frame addressed to a motor (any control mode) or to 0x7FF: a host's request, never a reply."""
    return can_id == PARAM_CAN_ID or (can_id >> 8 in (0, 1, 2, 3) and ID_MIN <= can_id & 0xFF <= ID_MAX)


def decode(can_id: int, data: bytes, model: MotorModel) -> Optional[Union[StateFrame, RegisterReply]]:
    """Decode one received frame. Anything that is not a Damiao reply returns None."""
    if getattr(can_id, "extended", False) or not 0 <= can_id <= 0x7FF or len(data) != 8:
        return None
    if is_command_id(can_id):
        return None
    if is_register_reply(data):
        return RegisterReply(data[0], data[2], data[3], bytes(data[4:8]))
    status = data[0] >> 4
    if status not in STATUS_NAMES:
        return None
    q = (data[1] << 8) | data[2]
    v = (data[3] << 4) | (data[4] >> 4)
    t = ((data[4] & 0xF) << 8) | data[5]
    return StateFrame(
        motor_id=data[0] & 0xF,
        status=status,
        position=uint_to_float(q, -model.p_max, model.p_max, 16),
        velocity=uint_to_float(v, -model.v_max, model.v_max, 12),
        torque=uint_to_float(t, -model.t_max, model.t_max, 12),
        t_mos=float(data[6]),
        t_rotor=float(data[7]),
    )


def encode_state(motor_id: int, status: int, pos: float, vel: float, tau: float,
                 t_mos: int, t_rotor: int, model: MotorModel) -> bytes:
    """A motor's state frame (used by simulators and tests)."""
    q = float_to_uint(pos, -model.p_max, model.p_max, 16)
    v = float_to_uint(vel, -model.v_max, model.v_max, 12)
    t = float_to_uint(tau, -model.t_max, model.t_max, 12)
    return bytes([(status << 4) | (motor_id & 0xF), q >> 8, q & 0xFF, v >> 4,
                  ((v & 0xF) << 4) | (t >> 8), t & 0xFF, int(t_mos) & 0xFF, int(t_rotor) & 0xFF])


def matching_model(p_max: float, v_max: float, t_max: float) -> Optional[str]:
    """Name of the model whose scaling equals these register values, if any."""
    for name, model in MODELS.items():
        if all(math.isclose(a, b, rel_tol=1e-5) for a, b in
               zip((p_max, v_max, t_max), (model.p_max, model.v_max, model.t_max))):
            return name
    return None
