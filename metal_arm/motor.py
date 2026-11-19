"""Damiao motor behind the Motor seam: one send id = one Motor. Commands are async, queries are
sync, feedback is timestamped. Everything Damiao-specific that the arm layer needs lives here
(reply routing, status nibble, register checks, watchdog register, id change); protocol.py stays a
pure codec.

Every Damiao state frame (each MIT reply) carries the motor's status; the read-only refresh
(0xCC, sent by poll_status/check/read_state) never clears anything. A fault status (overvoltage,
comm lost, ...) is latched until the caller explicitly runs clear_fault().

Late replies. The Metal gripper motor (id 7) often emits its reply to request N only when the
next frame appears on the bus, sometimes delivers that held reply twice, and occasionally never
answers (measured on two arms and two adapters). Every sync query
here therefore (1) matches replies by content, (2) after FLUSH_AFTER_S of silence from this motor
puts a harmless flush frame on the bus (a refresh of a neighbouring id) to release a held reply,
(3) for state reads, which have no content to match, sends one trailing flush and keeps the newest
reply, so a stale reply released by the request itself is superseded by the fresh one, and
(4) re-sends the request on a miss.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, NamedTuple, Optional, Union

from . import protocol
from .errors import ConnectError, ParamTimeout, StateError
from .protocol import Reg, RegisterReply, StateFrame
from .transport.base import CanBackend

if TYPE_CHECKING:
    from .config import JointConfig

log = logging.getLogger(__package__)

MODELS: tuple[str, ...] = tuple(protocol.MODELS)
DEFAULT_MODEL = "jlo"
KP_RANGE = (protocol.KP_MIN, protocol.KP_MAX)
KD_RANGE = (protocol.KD_MIN, protocol.KD_MAX)
ID_RANGE = (protocol.ID_MIN, protocol.ID_MAX)
DEFAULT_HOST_ID = protocol.DEFAULT_MASTER_OFFSET   # reply id = host_id + motor id unless set per joint

FLUSH_AFTER_S = 0.003       # a punctual reply arrives well inside this; a held one needs a frame
FLUSHES_PER_ATTEMPT = 3
SAVE_SETTLE_S = 0.1         # flash write time before the motor is asked anything again


def route(can_id: int, data: bytes, host_id: int = DEFAULT_HOST_ID) -> Optional[int]:
    """Motor id a received frame belongs to, or None. Register replies carry the send id in bytes
    0-1; state frames carry its low nibble in byte 0. Frames addressed to motors (another host's
    commands) are never replies. Motor.handle_frame additionally checks the exact reply id."""
    if getattr(can_id, "extended", False) or not 0 <= can_id <= 0x7FF or len(data) != 8:
        return None
    if protocol.is_command_id(can_id):
        return None
    if protocol.is_register_reply(data):
        return data[0]
    if data[0] >> 4 not in protocol.STATUS_NAMES:
        return None
    nibble = data[0] & 0xF
    offset = can_id - host_id
    if ID_RANGE[0] <= offset <= ID_RANGE[1] and offset & 0xF == nibble:
        return offset
    return nibble if ID_RANGE[0] <= nibble <= ID_RANGE[1] else None


@dataclass(frozen=True)
class MotorFeedback:
    """One feedback sample, motor coordinates."""
    motor_id: int
    position: float                # rad
    velocity: float                # rad/s
    torque: float                  # N·m
    temperature: float             # °C, hottest sensor
    temperatures: dict = field(default_factory=dict)   # {"mos": .., "rotor": ..}
    enabled: Optional[bool] = None
    warning: bool = False          # Damiao reports no warning level
    status: int = 0                # Damiao status nibble, see protocol.STATUS_NAMES


@dataclass(frozen=True)
class MotorStatus:
    motor_id: int
    has_fault: bool
    code: int                      # Damiao status nibble
    text: str                      # "" when healthy
    enabled: Optional[bool] = None


class ParamSpec(NamedTuple):
    index: int
    dtype: str
    doc: str


class _Waiter:
    """Collects replies to one sync query; fed from the RX thread."""

    def __init__(self, match: Callable[[int, bytes], object]):
        self.match = match
        self.replies: list = []
        self.event = threading.Event()
        self._lock = threading.Lock()

    def feed(self, can_id: int, data: bytes) -> None:
        msg = self.match(can_id, data)
        if msg is not None:
            with self._lock:
                self.replies.append(msg)
            self.event.set()

    def since(self, index: int) -> list:
        with self._lock:
            return self.replies[index:]


class Motor:
    STATUS_DISABLED_TICKS = 5   # consecutive health checks an enabled motor may report "disabled"
    PARAMS: dict[str, ParamSpec] = {name: ParamSpec(*spec) for name, spec in protocol.REGISTERS.items()}

    def __init__(self, joint: "JointConfig", backend: CanBackend, host_id: int = DEFAULT_HOST_ID):
        if joint.model not in protocol.MODELS:
            raise ValueError(f"motor {joint.motor_id}: unknown model {joint.model!r} (supported: {MODELS})")
        self.joint = joint
        self.motor_id = joint.motor_id
        protocol.encode_refresh(self.motor_id)   # validates the id range
        self._backend = backend
        self._host_id = host_id
        self.params = protocol.MODELS[joint.model]
        self.reply_id = getattr(joint, "reply_id", None) or host_id + self.motor_id
        # Refreshing a neighbouring id flushes a held reply without asking this motor again.
        self.flush_id = self.motor_id - 1 if self.motor_id > ID_RANGE[0] else self.motor_id + 1
        self._feedback: Optional[MotorFeedback] = None
        self._fb_time: Optional[float] = None
        self._feedback_sequence = 0
        self._status: Optional[MotorStatus] = None
        self._latched: Optional[str] = None
        self._disabled_ticks = 0
        self._query_lock = threading.RLock()
        self._waiter: Optional[_Waiter] = None
        self._tapping = False

    @property
    def torque_limit(self) -> float:
        """Largest torque the MIT frame can carry for this model, N·m."""
        return self.params.t_max

    # -- receive path (invoked by the RX dispatcher) --
    def handle_frame(self, can_id: int, data: bytes) -> None:
        data = bytes(data)
        if can_id != self.reply_id or getattr(can_id, "extended", False):
            return
        msg = self._decode_own(can_id, data)
        if isinstance(msg, StateFrame):
            self._store(msg)
        waiter = self._waiter
        if waiter is not None and not self._tapping:
            waiter.feed(can_id, data)

    def _decode_own(self, can_id: int, data: bytes):
        msg = protocol.decode(can_id, data, self.params)
        if isinstance(msg, StateFrame) and msg.motor_id == self.motor_id & 0xF:
            return msg
        if isinstance(msg, RegisterReply) and msg.motor_id == self.motor_id:
            return msg
        return None

    def _store(self, frame: StateFrame) -> None:
        fault = protocol.is_fault_status(frame.status)
        self._feedback = MotorFeedback(
            motor_id=self.motor_id, position=frame.position, velocity=frame.velocity,
            torque=frame.torque, temperature=max(frame.t_mos, frame.t_rotor),
            temperatures={"mos": frame.t_mos, "rotor": frame.t_rotor},
            enabled=frame.status == protocol.STATUS_ENABLED, warning=False, status=frame.status)
        self._status = MotorStatus(self.motor_id, fault, frame.status,
                                   protocol.status_text(frame.status) if fault else "",
                                   frame.status == protocol.STATUS_ENABLED)
        self._fb_time = time.monotonic()
        self._feedback_sequence += 1
        if fault and self._latched is None:
            self._latched = protocol.status_text(frame.status)

    # -- cached state --
    @property
    def feedback(self) -> Optional[MotorFeedback]:
        return self._feedback

    @property
    def feedback_age(self) -> float:
        return math.inf if self._fb_time is None else time.monotonic() - self._fb_time

    @property
    def feedback_sequence(self) -> int:
        """Monotonic count used to distinguish a fresh reply from cached feedback."""
        return self._feedback_sequence

    @property
    def status(self) -> Optional[MotorStatus]:
        return self._status

    @property
    def status_age(self) -> float:
        return self.feedback_age   # every state frame carries the status

    @property
    def fault_latched(self) -> Optional[str]:
        """Reason a fault was latched, or None."""
        return self._latched

    def reset_cache(self) -> None:
        self._feedback = None
        self._fb_time = None
        self._status = None
        self._latched = None
        self._disabled_ticks = 0

    def _send(self, frame) -> None:
        self._backend.send(*frame, extended=False)

    # -- async commands --
    def enable(self) -> None:
        self._disabled_ticks = 0
        self._send(protocol.encode_enable(self.motor_id))

    def disable(self) -> None:
        self._send(protocol.encode_disable(self.motor_id))

    def send_command(self, pos: float, vel: float, kp: float, kd: float, tau: float) -> None:
        self._send(protocol.encode_mit(self.motor_id, pos, vel, kp, kd, tau, self.params))

    def poll_status(self) -> None:
        """Send the read-only refresh; its state frame updates status (as every MIT reply does)."""
        self._send(protocol.encode_refresh(self.motor_id))

    # -- sync query engine --
    def _query(self, frame, match: Callable[[int, bytes], object], want: Optional[Callable] = None,
               timeout: float = 0.05, attempts: int = 3, newest: bool = False, what: str = "request"):
        """Send `frame` and return the reply that answers it (see the module docstring).

        `match` returns the decoded reply or None; `want` narrows a match to the content this
        request must produce, so a late duplicate of an earlier answer is skipped. A match that is
        not wanted is returned only if nothing better arrives (the caller's check then fails
        closed). `newest` keeps the newest wanted reply after one trailing flush.
        """
        if timeout <= 0 or attempts < 1:
            raise ValueError("timeout must be positive and attempts >= 1")
        flush = protocol.encode_refresh(self.flush_id)
        unwanted = None
        with self._query_lock:
            for _ in range(attempts):
                waiter = _Waiter(match)
                self._waiter = waiter
                try:
                    best = self._attempt(waiter, frame, flush, want, timeout, newest)
                finally:
                    self._waiter = None
                if best is not None and (want is None or want(best)):
                    return best
                unwanted = best or unwanted
        if unwanted is not None:
            return unwanted
        raise ParamTimeout(f"motor {self.motor_id} did not answer {what} "
                           f"({attempts} attempts of {timeout:g} s)")

    def _attempt(self, waiter: _Waiter, frame, flush, want, timeout: float, newest: bool):
        self._send(frame)
        now = quiet_since = time.monotonic()
        deadline = now + timeout
        flushes, seen = 0, 0
        best = unwanted = None
        trailing_until = None
        while True:
            new = waiter.since(seen)
            seen += len(new)
            if new:
                quiet_since = now
            for msg in new:
                if want is None or want(msg):
                    best = msg
                else:
                    unwanted = msg
            if best is not None:
                if not newest:
                    return best
                if trailing_until is None:
                    self._send(flush)
                    trailing_until = now + FLUSH_AFTER_S
            if trailing_until is not None and now >= trailing_until:
                return best
            if now >= deadline:
                return best or unwanted
            if best is None and flushes < FLUSHES_PER_ATTEMPT and now - quiet_since >= FLUSH_AFTER_S:
                self._send(flush)
                flushes += 1
                quiet_since = now
            waiter.event.wait(max(0.0, min(FLUSH_AFTER_S, deadline - now)))
            waiter.event.clear()
            now = time.monotonic()

    def _match_state(self, can_id: int, data: bytes):
        msg = self._decode_own(can_id, data) if can_id == self.reply_id else None
        return msg if isinstance(msg, StateFrame) else None

    def _refresh(self, timeout: float, attempts: int) -> StateFrame:
        return self._query(protocol.encode_refresh(self.motor_id), self._match_state,
                           timeout=timeout, attempts=attempts, newest=True, what="a state refresh")

    def _register(self, op: int, rid: int, value=None, dtype: str = "f", timeout: float = 0.12,
                  attempts: int = 2, send_id: Optional[int] = None,
                  reply_ids: Optional[tuple] = None, echo_ids: Optional[tuple] = None) -> RegisterReply:
        """One register transaction. reply_ids: accepted master ids (None = this motor's, () = any).
        echo_ids: motor ids accepted inside the reply (None = the one addressed)."""
        send_id = self.motor_id if send_id is None else send_id
        echo_ids = (send_id,) if echo_ids is None else echo_ids
        reply_ids = (self.reply_id,) if reply_ids is None else reply_ids
        if op == protocol.OP_READ:
            frame = protocol.encode_read_register(send_id, rid)
        else:
            frame = protocol.encode_write_register(send_id, rid, value, dtype)

        def match(can_id, data):
            if reply_ids and can_id not in reply_ids:
                return None
            if protocol.is_command_id(can_id) or not protocol.is_register_reply(data):
                return None
            reply = RegisterReply(data[0], data[2], data[3], bytes(data[4:8]))
            return reply if (reply.op, reply.rid) == (op, rid) and reply.motor_id in echo_ids else None

        # A write echoes its value: a late duplicate of an earlier echo is skipped.
        want = None if op == protocol.OP_READ else (lambda r: r.raw == frame[1][4:8])
        verb = "read" if op == protocol.OP_READ else "write"
        return self._query(frame, match, want, timeout, attempts, what=f"register {rid} {verb}")

    # -- sync queries --
    def check(self, timeout: float = 0.05, attempts: int = 3) -> MotorStatus:
        """Fresh status via the read-only refresh: never clears a fault, never changes torque."""
        self._refresh(timeout, attempts)
        return self._status

    def read_state(self, timeout: float = 0.05, attempts: int = 3) -> MotorFeedback:
        """Fresh feedback (not cached). The Damiao refresh clears nothing, so this never needs to
        refuse; a fault status in the reply is latched as usual."""
        self._refresh(timeout, attempts)
        return self._feedback

    def validate(self, timeout: float = 0.12) -> None:
        """Connect-time checks: MIT control mode and MIT scaling registers equal to the model."""
        label = self.joint.label
        try:
            mode = self.read_param("ctrl_mode", timeout=timeout)
            scaling = [self.read_param(name, timeout=timeout) for name in ("pmax", "vmax", "tmax")]
        except ParamTimeout as e:
            raise ConnectError(f"{label}: no reply to a register read ({e}); check power and wiring") from e
        if mode != protocol.MODE_MIT:
            raise ConnectError(
                f"{label}: control mode is {mode}, not 1 (MIT). Another program left it there; run "
                f"`<cli> params write {self.motor_id} ctrl_mode 1` or power-cycle the arm")
        expected = (self.params.p_max, self.params.v_max, self.params.t_max)
        if not all(math.isclose(a, b, rel_tol=1e-5) for a, b in zip(scaling, expected)):
            found = protocol.matching_model(*scaling)
            fix = (f"set `model: {found}` for this joint in the profile" if found else
                   "the motor was reconfigured; restore PMAX/VMAX/TMAX (registers 21-23) with the "
                   "Damiao tool or use a matching profile")
            raise ConnectError(
                f"{label}: MIT scaling PMAX/VMAX/TMAX = {', '.join(f'{v:g}' for v in scaling)} does not "
                f"match model {self.joint.model!r} ({', '.join(f'{v:g}' for v in expected)}); {fix}")

    def confirms(self, expected: str) -> bool:
        """Does the latest feedback confirm "enabled" / "disabled"? Enabled is status 1; disabled is
        any status but 1 (a faulted motor carries no torque)."""
        if expected not in ("enabled", "disabled"):
            raise ValueError(f"expected 'enabled' or 'disabled', got {expected!r}")
        fb = self._feedback
        if fb is None:
            return False
        return (fb.status == protocol.STATUS_ENABLED) == (expected == "enabled")

    def health_problem(self) -> Optional[str]:
        """Per-tick check while enabled: a fault status at once; status "disabled" (the motor
        dropped torque, e.g. after a supply dip reboot) once it persists STATUS_DISABLED_TICKS ticks."""
        if self._latched is not None:
            return self._latched
        fb = self._feedback
        if fb is None:
            return None
        self._disabled_ticks = self._disabled_ticks + 1 if fb.status == protocol.STATUS_DISABLED else 0
        if self._disabled_ticks >= self.STATUS_DISABLED_TICKS:
            return f"motor reports disabled while it should be enabled ({self._disabled_ticks} ticks)"
        return None

    def preload(self, pos: float, timeout: float = 0.05) -> bool:
        """Zero-gain MIT command at ``pos`` (motor coordinates); True once its state reply arrives."""
        if not math.isfinite(pos):
            return False
        try:
            self._query(protocol.encode_mit(self.motor_id, pos, 0.0, 0.0, 0.0, 0.0, self.params),
                        self._match_state, timeout=timeout, attempts=2, newest=True, what="the neutral preload")
        except ParamTimeout:
            return False
        return True

    def enable_confirmed(self, timeout: float = 0.05) -> bool:
        """Enable; True once a reply reports status enabled."""
        self._disabled_ticks = 0
        try:
            reply = self._query(protocol.encode_enable(self.motor_id), self._match_state,
                                lambda m: m.status == protocol.STATUS_ENABLED, timeout, attempts=2,
                                what="enable")
        except ParamTimeout:
            return False
        return reply.status == protocol.STATUS_ENABLED

    def disable_confirmed(self, timeout: float = 0.05, attempts: int = 3) -> bool:
        """Disable and confirm a non-enabled status in the reply."""
        try:
            self._query(protocol.encode_disable(self.motor_id), self._match_state,
                        lambda m: m.status != protocol.STATUS_ENABLED, timeout, attempts,
                        what="disable")
        except ParamTimeout:
            return False
        return self.confirms("disabled")

    # -- operator actions --
    def clear_fault(self) -> None:
        """Explicit operator action: clear the motor's error (0xFB) and the latched evidence."""
        self._latched = None
        self._send(protocol.encode_clear_error(self.motor_id))

    def set_zero(self) -> None:
        """Make the current position the motor's zero (0xFE). Persists in the motor."""
        self._send(protocol.encode_set_zero(self.motor_id))

    def arm_watchdog(self, timeout_ms: int, timeout: float = 0.12) -> Optional[int]:
        """Set and verify the communication watchdog (register 9, 50 us ticks) in RAM only, never
        flash. Returns the previous tick count as the token."""
        if not 0 < timeout_ms <= 200_000:
            raise ValueError("timeout_ms must be in (0, 200000]; 0 would disable the watchdog")
        ticks = int(round(timeout_ms * protocol.TIMEOUT_TICKS_PER_MS))
        original = self.read_param("timeout", timeout=timeout)
        try:
            readback = self.write_param("timeout", ticks, timeout=timeout)
            if readback != ticks:
                raise ConnectError(f"{self.joint.label}: watchdog read back {readback} ticks, expected {ticks}")
        except BaseException:
            try:
                self.restore_watchdog(original, timeout)
            except Exception as e:
                log.error("could not restore %s watchdog: %s", self.joint.label, e)
            raise
        return original

    def restore_watchdog(self, token, timeout: float = 0.12) -> None:
        if token is None:
            return
        readback = self.write_param("timeout", int(token), timeout=timeout)
        if readback != int(token):
            raise StateError(f"{self.joint.label}: watchdog restore read back {readback}, expected {token}")

    @contextmanager
    def _tap(self):
        """Feed every received frame (not only this motor's) to the active waiter: an id change
        must hear the new id. Chains the backend's existing callback."""
        backend = self._backend
        previous = backend.recv_callback

        def tap(can_id, data):
            waiter = self._waiter
            if waiter is not None:
                waiter.feed(can_id, bytes(data))
            if previous is not None:
                previous(can_id, data)

        with self._query_lock:
            self._tapping = True
            backend.set_recv_callback(tap)
            try:
                yield
            finally:
                backend.set_recv_callback(previous)
                self._tapping = False

    def _answers_at(self, send_id: int, reply_ids: tuple, timeout: float) -> Optional[int]:
        """ESC_ID register as read through `send_id`, or None when nothing answers."""
        try:
            return self._register(protocol.OP_READ, Reg.ESC_ID, timeout=timeout, attempts=2,
                                  send_id=send_id, reply_ids=reply_ids).value("u32")
        except ParamTimeout:
            return None

    def _id_in_use(self, new_id: int, timeout: float) -> bool:
        def any_state(can_id, data):
            if protocol.is_command_id(can_id) or protocol.is_register_reply(data):
                return None
            return data if data[0] & 0xF == new_id and data[0] >> 4 in protocol.STATUS_NAMES else None
        try:
            self._query(protocol.encode_refresh(new_id), any_state, timeout=timeout, attempts=2)
            return True
        except ParamTimeout:
            pass
        return self._answers_at(new_id, (), timeout) is not None   # () = any reply id

    def set_id(self, new_id: int, timeout: float = 0.5) -> None:
        """Change the send id to new_id and the master id to host_id + new_id, saved to flash.

        Procedure (Damiao register map, DM_CAN change_motor_param/save_motor_param): refuse if
        anything answers at new_id; require the motor disabled (flash save is ignored while
        enabled); write MST_ID then ESC_ID through the old id. The firmware applies both at once,
        so each echo is accepted on the old or the new master id, and the ESC_ID echo may name
        either id or not arrive at all. Then find the id the motor now answers at and save (0xAA)
        through it; verify by reading ESC_ID and MST_ID through the new
        id. A motor that keeps answering at the old id has the change saved but not applied:
        StateError asks for a power cycle.
        """
        lo, hi = ID_RANGE
        if not lo <= new_id <= hi:
            raise ValueError(f"motor id must be in [{lo}, {hi}], got {new_id}")
        if new_id == self.motor_id:
            return
        old_id, old_reply = self.motor_id, self.reply_id
        new_reply = self._host_id + new_id
        if protocol.is_command_id(new_reply) or not 0 < new_reply < 0x7FF:
            raise ValueError(f"master id 0x{new_reply:X} would collide with command ids")
        step = min(timeout, 0.12)
        replies = (old_reply, new_reply)
        with self._tap():
            if self._id_in_use(new_id, step):
                raise StateError(f"id {new_id} is already used by a motor on this bus; "
                                 "pick a free id or disconnect that motor")
            status = self.check(step)
            if status.enabled:
                raise StateError(f"{self.joint.label}: disable the motor before changing its id")
            reply = self._register(protocol.OP_WRITE, Reg.MST_ID, new_reply, "u32", step, 2,
                                    send_id=old_id, reply_ids=replies)
            if reply.value("u32") != new_reply:
                raise StateError(f"master id echoed {reply.value('u32')}, expected {new_reply}; "
                                 "nothing was saved, power-cycle the motor to undo")
            try:
                self._register(protocol.OP_WRITE, Reg.ESC_ID, new_id, "u32", step, 1,
                               send_id=old_id, reply_ids=replies, echo_ids=(old_id, new_id))
            except ParamTimeout:
                pass   # the read through the new id below decides
            if self._answers_at(new_id, (new_reply,), step) == new_id:
                save_id = new_id
            elif self._answers_at(old_id, replies, step) == new_id:
                save_id = old_id
            else:
                raise StateError(f"{self.joint.label}: motor stopped answering after the id write; "
                                 "nothing was saved, power-cycle the motor to undo")
            self._send(protocol.encode_save(save_id))
            time.sleep(SAVE_SETTLE_S)
            esc = self._answers_at(new_id, (new_reply,), timeout)
            if esc != new_id:
                raise StateError(
                    f"{self.joint.label}: ids written and saved, but the motor does not answer at id "
                    f"{new_id} yet; power-cycle the arm and check with a scan")
            mst = self._register(protocol.OP_READ, Reg.MST_ID, timeout=timeout, send_id=new_id,
                                 reply_ids=(new_reply,)).value("u32")
            if mst != new_reply:
                raise StateError(f"{self.joint.label}: master id reads 0x{mst:X}, expected 0x{new_reply:X}")
        self.motor_id, self.reply_id = new_id, new_reply
        self.flush_id = new_id - 1 if new_id > lo else new_id + 1
        self.reset_cache()

    # -- engineering --
    def _param(self, name_or_index: Union[str, int], dtype: Optional[str]) -> tuple[int, str]:
        if isinstance(name_or_index, str):
            key = name_or_index.lower()
            if key not in self.PARAMS:
                raise ValueError(f"unknown parameter {name_or_index!r} (known: {sorted(self.PARAMS)})")
            spec = self.PARAMS[key]
            return spec.index, dtype or spec.dtype
        index = int(name_or_index)
        return index, dtype or protocol.register_dtype(index)

    def read_param(self, name_or_index: Union[str, int], dtype: Optional[str] = None, timeout: float = 0.12):
        index, dtype = self._param(name_or_index, dtype)
        return self._register(protocol.OP_READ, index, timeout=timeout).value(dtype)

    def write_param(self, name_or_index: Union[str, int], value, dtype: Optional[str] = None,
                    timeout: float = 0.12, save: bool = False):
        """Write a register (RAM), verify the echo and a read-back, and return the read-back value.
        save=True then stores all registers to flash (motor must be disabled; flash wears out)."""
        index, dtype = self._param(name_or_index, dtype)
        packed = protocol.pack_value(value, dtype)
        if save and self.check(min(timeout, 0.05)).enabled:
            raise StateError(f"motor {self.motor_id}: disable it before saving registers to flash")
        echo = self._register(protocol.OP_WRITE, index, value, dtype, timeout)
        readback = self._register(protocol.OP_READ, index, timeout=timeout)
        if echo.raw != packed or readback.raw != packed:
            raise StateError(f"motor {self.motor_id} register {index} kept "
                             f"{readback.value(dtype)!r}, not {protocol.unpack_value(packed, dtype)!r}")
        if save:
            self._send(protocol.encode_save(self.motor_id))
            time.sleep(SAVE_SETTLE_S)
        return readback.value(dtype)
