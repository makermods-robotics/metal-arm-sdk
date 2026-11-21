"""Simulated Damiao MIT motors behind MockBackend, for tests and ``testing``.

Each SimMotor keeps RAM registers and a flash copy, a status nibble, two temperatures and a
position that follows MIT commands while enabled (kp > 0). Options model the field behaviour the
driver must survive: `late` holds every reply until the next frame appears on the bus (the
gripper), `duplicate` releases a held reply twice, `drop` ignores that many requests, `fault`
reports a status code, and `ids_apply_at_once` chooses whether an ESC_ID/MST_ID write takes
effect immediately or only after save + power_cycle(). ``log`` records what it received
("status_query", "clear", "enable", "disable", "zero", ("cmd", kp), ("read", reg), ("write", reg),
"save"), ``frames`` every MIT command as (position, kp, tau).
"""

from typing import Optional

from . import protocol as p
from .transport.mock import MockBackend


def feedback_bytes(motor_id, pos=0.0, vel=0.0, tau=0.0, temp=30.0, status=0, rotor_temp=None,
                   model="jlo"):
    """A Damiao state frame payload (temperatures in whole °C)."""
    params = p.MODELS[model] if isinstance(model, str) else model
    return p.encode_state(motor_id, status, pos, vel, tau, round(temp),
                          round(temp if rotor_temp is None else rotor_temp), params)


def register_reply(motor_id, op, rid, raw: bytes) -> bytes:
    return bytes([motor_id & 0xFF, motor_id >> 8, op, rid]) + bytes(raw)


FAULT_CODE = 0xC                       # what ``faulted = True`` reports
FAULT_TEXT = "rotor overtemperature"    # how the SDK describes FAULT_CODE

# What the protocol can show:
FULL_TURN_SUPPORTED = False   # position range is +-PMAX (6.28 rad): a whole-turn offset is unrepresentable
WARNING_SUPPORTED = False     # Damiao feedback has no warning flag


def classify(can_id: int, data: bytes) -> tuple[str, Optional[int]]:
    """(kind, motor id) of a host->motor frame: "command", "enable", "disable", "zero", "clear",
    "status_query", "param_read", "param_write", "param_save" or "other"."""
    data = bytes(data)
    if can_id == p.PARAM_CAN_ID:
        if len(data) < 4:
            return "other", None
        target = data[0] | data[1] << 8
        kind = {p.OP_REFRESH: "status_query", p.OP_READ: "param_read", p.OP_WRITE: "param_write",
                p.OP_SAVE: "param_save"}.get(data[2], "other")
        return kind, target
    motor_id = can_id & 0xFF
    if can_id >> 8 not in (0, 1, 2, 3):
        return "other", None
    if data[:7] == p.COMMAND_PREFIX:
        name = {p.CMD_ENABLE: "enable", p.CMD_DISABLE: "disable", p.CMD_SET_ZERO: "zero",
                p.CMD_CLEAR_ERROR: "clear"}.get(data[7], "other")
        return name, motor_id
    return "command", motor_id


class SimMotor:
    def __init__(self, motor_id: int, model: str = "jlo", *, position: float = 0.0,
                 velocity: float = 0.0, torque: float = 0.0, temperature: float = 30.0,
                 rotor_temperature: Optional[float] = None, enabled: bool = False,
                 online: bool = True, track: bool = True, faulted: bool = False,
                 warning: bool = False, fault: Optional[int] = None, late: bool = False,
                 duplicate: bool = False, drop: int = 0, ids_apply_at_once: bool = True,
                 save_reply: bool = False, esc_echo: str = "old", registers: Optional[dict] = None,
                 params_supported: bool = True):
        self.motor_id = motor_id
        self.model = model
        self.position = position
        self.velocity = velocity
        self.torque = torque
        self.temperature = temperature              # MOS °C
        self.rotor_temperature = rotor_temperature  # defaults to MOS
        self.enabled = enabled
        self.online = online
        self.track = track                # an enabled motor jumps to the commanded position (kp > 0)
        self.warning = warning            # accepted for the shared surface; Damiao reports none
        self.params_supported = params_supported   # accepted for the shared surface; registers always answer
        self.fault = (FAULT_CODE if faulted else 0) if fault is None else fault   # status code 0x8..0xE
        self.late = late                  # hold each reply until the next frame on the bus
        self.duplicate = duplicate        # release a held reply twice
        self.drop = drop                  # ignore this many upcoming requests
        self.ids_apply_at_once = ids_apply_at_once   # ESC_ID/MST_ID writes act before a power cycle
        self.save_reply = save_reply      # echo the 0xAA save request
        self.esc_echo = esc_echo          # id named in the ESC_ID write echo: "old", "new" or "none"
        self.log: list = []
        self.frames: list = []            # every MIT frame: (position, kp, tau)
        self.commands: list = []          # every MIT frame: (position, velocity, kp, kd, tau)
        self.last_kp = 0.0
        self.held: list = []
        params = p.MODELS[model]
        defaults = {p.Reg.MST_ID: p.DEFAULT_MASTER_OFFSET + motor_id, p.Reg.ESC_ID: motor_id,
                    p.Reg.TIMEOUT: 0, p.Reg.CTRL_MODE: p.MODE_MIT, p.Reg.PMAX: params.p_max,
                    p.Reg.VMAX: params.v_max, p.Reg.TMAX: params.t_max, 14: 0x1234}
        self.registers = {**defaults, **(registers or {})}
        self.flash = dict(self.registers)
        self.send_id = int(self.registers[p.Reg.ESC_ID])
        self.reply_id = int(self.registers[p.Reg.MST_ID])

    @property
    def faulted(self) -> bool:
        return bool(self.fault)

    @faulted.setter
    def faulted(self, value: bool) -> None:
        self.fault = FAULT_CODE if value else 0

    @property
    def params(self) -> p.MotorModel:
        return p.MODELS[self.model]

    @property
    def status(self) -> int:
        if self.fault:
            return self.fault
        return p.STATUS_ENABLED if self.enabled else p.STATUS_DISABLED

    @property
    def watchdog_ticks(self) -> int:
        return int(self.registers[p.Reg.TIMEOUT])

    def feedback(self):
        rotor = self.temperature if self.rotor_temperature is None else self.rotor_temperature
        return self.reply_id, feedback_bytes(self.send_id, self.position, self.velocity, self.torque,
                                             self.temperature, self.status, rotor, self.params)

    def power_cycle(self) -> None:
        """Reload flash: unsaved register writes are lost, ids from flash take effect."""
        self.registers = dict(self.flash)
        self.send_id = int(self.registers[p.Reg.ESC_ID])
        self.reply_id = int(self.registers[p.Reg.MST_ID])
        self.enabled, self.fault, self.held = False, 0, []

    def addressed(self, can_id: int, data: bytes) -> bool:
        if can_id == p.PARAM_CAN_ID:
            return len(data) >= 2 and (data[0] | data[1] << 8) == self.send_id
        return can_id >> 8 in (0, 1, 2, 3) and can_id & 0xFF == self.send_id

    def handle(self, can_id: int, data: bytes) -> list:
        """Replies to one frame addressed to this motor."""
        if not self.online:
            return []
        if self.drop:
            self.drop -= 1
            self.log.append("dropped")
            return []
        data = bytes(data)
        if can_id == p.PARAM_CAN_ID:
            return self._param_frame(data)
        if data[:7] == p.COMMAND_PREFIX:
            return self._command(data[7])
        return self._mit(data)

    def _command(self, command: int) -> list:
        if command == p.CMD_ENABLE:
            self.log.append("enable")
            self.enabled = not self.fault
        elif command == p.CMD_DISABLE:
            self.log.append("disable")
            self.enabled = False
        elif command == p.CMD_SET_ZERO:
            self.log.append("zero")
            self.position = 0.0
        elif command == p.CMD_CLEAR_ERROR:
            self.log.append("clear")
            self.fault = 0
        return [self.feedback()]

    def _mit(self, data: bytes) -> list:
        params = self.params
        q = (data[0] << 8) | data[1]
        v = (data[2] << 4) | (data[3] >> 4)
        kp = (((data[3] & 0xF) << 8) | data[4]) * p.KP_MAX / 4095
        kd = ((data[5] << 4) | (data[6] >> 4)) * p.KD_MAX / 4095
        t = ((data[6] & 0xF) << 8) | data[7]
        position = p.uint_to_float(q, -params.p_max, params.p_max, 16)
        velocity = p.uint_to_float(v, -params.v_max, params.v_max, 12)
        tau = p.uint_to_float(t, -params.t_max, params.t_max, 12)
        self.last_kp = kp
        self.log.append(("cmd", round(kp, 1)))
        self.frames.append((position, kp, tau))
        self.commands.append((position, velocity, kp, kd, tau))
        if self.enabled and self.track and kp > 0:
            self.position = position
        return [self.feedback()]

    def _param_frame(self, data: bytes) -> list:
        op, rid = data[2], data[3]
        if op == p.OP_REFRESH:
            self.log.append("status_query")
            return [self.feedback()]
        if op == p.OP_READ:
            self.log.append(("read", rid))
            raw = p.pack_value(self.registers.get(rid, 0), p.register_dtype(rid))
            return [(self.reply_id, register_reply(self.send_id, op, rid, raw))]
        if op == p.OP_WRITE:
            self.log.append(("write", rid))
            self.registers[rid] = p.unpack_value(data[4:8], p.register_dtype(rid))
            reply = [(self.reply_id, register_reply(self.send_id, op, rid, data[4:8]))]
            if self.ids_apply_at_once and rid == p.Reg.MST_ID:
                self.reply_id = int(self.registers[rid])
                reply = [(self.reply_id, reply[0][1])]
            if self.ids_apply_at_once and rid == p.Reg.ESC_ID:
                self.send_id = int(self.registers[rid])
                if self.esc_echo == "none":
                    return []
                if self.esc_echo == "new":
                    reply = [(self.reply_id, register_reply(self.send_id, op, rid, data[4:8]))]
            return reply
        if op == p.OP_SAVE:
            self.log.append("save")
            if not self.enabled:
                self.flash = dict(self.registers)
            return [(self.reply_id, bytes(data))] if self.save_reply else []
        return []


def respond(motors: dict, can_id: int, data: bytes) -> list:
    """Replies of the simulated motors to one frame on the bus. Any frame releases held (late)
    replies first; the motor that owns the frame's send id then answers (or holds its answer)."""
    if getattr(can_id, "extended", False):
        return []
    out = []
    for m in motors.values():
        if m.held:
            for frame in m.held:
                out.extend([frame, frame] if m.duplicate else [frame])
            m.held = []
    for m in motors.values():
        if m.addressed(can_id, data):
            replies = m.handle(can_id, data)
            if m.late:
                m.held.extend(replies)
            else:
                out.extend(replies)
    return out


class SimBus(MockBackend):
    """MockBackend whose responder routes each frame to the SimMotor that owns its send id."""

    def __init__(self, motors):
        super().__init__()
        self.motors = {m.motor_id: m for m in motors}
        self.responder = lambda can_id, data: respond(self.motors, can_id, data)


def watchdog_timeout(motor: SimMotor) -> None:
    """What the motor's watchdog does once commands stop: torque off, status "communication lost"."""
    motor.enabled, motor.fault = False, p.STATUS_COMM_LOST


def for_joint(joint, host_id: int, position: float = 0.0, **options) -> SimMotor:
    """A SimMotor answering like the motor of ``joint`` (a JointConfig) at ``position`` (motor rad),
    replying on the joint's reply id (default ``host_id`` + motor id)."""
    registers = {p.Reg.MST_ID: joint.reply_id or host_id + joint.motor_id, **options.pop("registers", {})}
    return SimMotor(joint.motor_id, joint.model, position=position, registers=registers, **options)
