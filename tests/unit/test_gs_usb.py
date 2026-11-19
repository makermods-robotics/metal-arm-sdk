"""GsUsbBackend against a fake candleLight USB device: no real USB device is ever opened."""

import struct
import threading
import time
from collections import deque

import pytest

usb_core = pytest.importorskip("usb.core")

from fakes import FakeBus, FakeMotor, make_joint  # noqa: E402
from metal_arm.errors import ConnectError  # noqa: E402
from metal_arm.motor import Motor, route  # noqa: E402
from metal_arm.transport import gs_usb  # noqa: E402
from metal_arm.transport.base import create_backend  # noqa: E402


@pytest.fixture(autouse=True)
def linux(monkeypatch):
    """The kernel-driver detach only happens on Linux; test that path on every platform."""
    monkeypatch.setattr(gs_usb.sys, "platform", "linux")

BT_CONST_48MHZ = struct.pack("<10I", 0x1F, 48_000_000, 1, 16, 1, 8, 4, 1, 1024, 1)
FRAME = struct.Struct("<IIBBBB8s")


def rx_frame(can_id, data, extended=False, echo=0xFFFFFFFF, flags=0, channel=0):
    ident = can_id | (gs_usb.CAN_EFF_FLAG if extended else 0)
    return FRAME.pack(echo, ident, len(data), channel, flags, 0, bytes(data).ljust(8, b"\x00"))


class Ctx:
    def __init__(self):
        self.claimed, self.released, self.disposed = [], [], 0

    def managed_claim_interface(self, dev, interface):
        self.claimed.append(interface)

    def managed_release_interface(self, dev, interface):
        self.released.append(interface)

    def dispose(self, dev):
        self.disposed += 1


class FakeUsbDevice:
    """A candleLight adapter: echoes acknowledged frames and can host a simulated CAN bus."""

    idVendor, idProduct = 0x1D50, 0x606F

    def __init__(self, serial="ABC123", bus=1, port_numbers=(2, 1), ack=True, can_bus=None):
        self.serial_number = serial
        self.product = "candleLight USB to CAN adapter"
        self.manufacturer = "bytewerk"
        self.bus, self.address, self.port_numbers = bus, 7, port_numbers
        self._ctx = Ctx()
        self.ack = ack
        self.can_bus = can_bus          # FakeBus: frames written reach the simulated motors
        self.kernel_driver = True
        self.detached = []
        self.controls = []
        self.writes = []
        self.resets = 0
        self.rx = deque()
        self.read_error = None
        self.write_error = None
        self.ctrl_error = None
        self.configured = True
        self.configurations_set = 0

    def reset(self):
        self.resets += 1

    def is_kernel_driver_active(self, interface):
        return self.kernel_driver

    def detach_kernel_driver(self, interface):
        self.detached.append(interface)
        self.kernel_driver = False

    def get_active_configuration(self):
        if not self.configured:
            raise usb_core.USBError("not configured")
        return object()

    def set_configuration(self):
        self.configurations_set += 1
        self.configured = True

    def ctrl_transfer(self, bm, req, value, index, data_or_length, timeout):
        self.controls.append((bm, req, value, index, data_or_length, timeout))
        if self.ctrl_error:
            raise self.ctrl_error
        if bm == 0xC1 and req == gs_usb.BREQ_BT_CONST:
            return bytearray(BT_CONST_48MHZ)
        return len(data_or_length)

    def write(self, endpoint, data, timeout):
        if self.write_error:
            raise self.write_error
        self.writes.append((endpoint, bytes(data), timeout))
        echo, ident, dlc, _, _, _, payload = FRAME.unpack(bytes(data))
        if self.ack:
            self.rx.append(FRAME.pack(echo, ident, dlc, 0, 0, 0, payload))
            if self.can_bus is not None:
                for cid, reply in self.can_bus.responder(ident & 0x7FF, payload[:dlc]) or []:
                    self.rx.append(rx_frame(cid, reply))
        return len(data)

    def read(self, endpoint, size, timeout):
        if self.read_error:
            raise self.read_error
        try:
            return bytearray(self.rx.popleft())
        except IndexError:
            time.sleep(0.0005)
            raise usb_core.USBTimeoutError("timeout")

    def ctrl(self, req):
        return [c for c in self.controls if c[1] == req]


@pytest.fixture
def opened():
    backends = []

    def make(dev=None, **kwargs):
        dev = dev or FakeUsbDevice()
        backend = gs_usb.GsUsbBackend("gs_usb:" + dev.serial_number, device=dev, **kwargs)
        received = []
        backend.set_recv_callback(lambda cid, data: received.append((cid, bytes(data))))
        backend.open()
        backends.append(backend)
        return backend, dev, received

    yield make
    for b in backends:
        b.close()


def wait_for(predicate, timeout=1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.001)
    return False


def test_bit_timing_for_candlelight_48mhz_matches_gs_usb_table():
    assert gs_usb.bit_timing(48_000_000, 1_000_000, 1, 16, 1, 8, 4, 1, 1024, 1) == (1, 12, 2, 1, 3)
    assert gs_usb.bit_timing(48_000_000, 500_000, 1, 16, 1, 8, 4, 1, 1024, 1) == (1, 12, 2, 1, 6)
    prop, seg1, seg2, _, brp = gs_usb.bit_timing(80_000_000, 1_000_000, 1, 16, 1, 8, 4, 1, 1024, 1)
    quanta = 1 + prop + seg1 + seg2
    assert 80_000_000 // (brp * quanta) == 1_000_000 and (1 + prop + seg1) / quanta == 0.875
    with pytest.raises(ConnectError):
        gs_usb.bit_timing(48_000_000, 1_000_000, 1, 4, 1, 2, 1, 1, 1, 1)


def test_parse_port():
    assert gs_usb.parse_port("gs_usb:ABC") == "ABC"
    assert gs_usb.parse_port("ABC") == "ABC"
    assert gs_usb.parse_port(None) is None and gs_usb.parse_port("gs_usb:") is None
    with pytest.raises(ValueError):
        gs_usb.parse_port("gs_usb:A B")


def test_open_detaches_kernel_driver_and_starts_without_usb_reset(opened):
    backend, dev, _ = opened()
    assert dev.detached == [0] and dev._ctx.claimed == [0]
    assert dev.resets == 0
    assert [c[1] for c in dev.controls] == [gs_usb.BREQ_BT_CONST, gs_usb.BREQ_BITTIMING, gs_usb.BREQ_MODE]
    assert dev.ctrl(gs_usb.BREQ_BITTIMING)[0][4] == struct.pack("<5I", 1, 12, 2, 1, 3)
    assert dev.ctrl(gs_usb.BREQ_MODE)[0][4] == struct.pack("<II", 1, 0)
    assert all(c[5] == 100 for c in dev.controls)   # every transfer bounded


def test_unconfigured_device_is_configured(opened):
    dev = FakeUsbDevice()
    dev.configured = False
    opened(dev)
    assert dev.configurations_set == 1


def test_send_layout_standard_and_extended(opened):
    backend, dev, _ = opened()
    backend.send(0x7FF, bytes([7, 0, 0xCC, 0, 0, 0, 0, 0]), extended=False)
    backend.send(0x1234567, b"\x01\x02", extended=True)
    (ep1, f1, t1), (ep2, f2, _) = dev.writes
    assert ep1 == gs_usb.EP_OUT and t1 >= 1
    assert FRAME.unpack(f1)[1:4] == (0x7FF, 8, 0) and FRAME.unpack(f1)[6] == bytes([7, 0, 0xCC, 0, 0, 0, 0, 0])
    assert FRAME.unpack(f2)[1] == 0x1234567 | gs_usb.CAN_EFF_FLAG and FRAME.unpack(f2)[2] == 2
    assert FRAME.unpack(f1)[0] != FRAME.unpack(f2)[0] != gs_usb.ECHO_RX
    with pytest.raises(ValueError):
        backend.send(1, bytes(9), extended=False)


def test_receive_delivers_frames_and_drops_echoes_errors_and_other_channels(opened):
    backend, dev, received = opened()
    dev.rx.extend([
        rx_frame(0x17, bytes(range(8))),
        rx_frame(0x1ABCDEF, b"\xAA", extended=True),
        rx_frame(0x11, bytes(8), echo=3),                                  # TX echo
        FRAME.pack(gs_usb.ECHO_RX, gs_usb.CAN_ERR_FLAG | 4, 8, 0, 0, 0, bytes(8)),
        rx_frame(0x12, bytes(8), channel=1),
        rx_frame(0x13, bytes(8), flags=0x02),                              # CAN FD
        rx_frame(0x14, b"\x01", flags=gs_usb.FLAG_OVERFLOW),
    ])
    assert wait_for(lambda: len(received) == 3)
    time.sleep(0.01)
    assert received[0] == (0x17, bytes(range(8))) and received[0][0].extended is False
    assert received[1] == (0x1ABCDEF, b"\xAA") and received[1][0].extended is True
    assert received[2] == (0x14, b"\x01")
    assert backend.error_frames == 1 and backend.rx_overflows == 1


def test_unacknowledged_frames_stop_further_writes(opened):
    backend, dev, _ = opened(FakeUsbDevice(ack=False), io_timeout=0.02)
    for _ in range(gs_usb.MAX_IN_FLIGHT):
        backend.send(1, bytes(8), extended=False)
    started = time.monotonic()
    with pytest.raises(TimeoutError, match="never acknowledged"):
        backend.send(1, bytes(8), extended=False)
    assert time.monotonic() - started < 0.5
    assert len(dev.writes) == gs_usb.MAX_IN_FLIGHT


def test_echoes_free_the_in_flight_budget(opened):
    backend, dev, _ = opened(io_timeout=0.05)
    for _ in range(5 * gs_usb.MAX_IN_FLIGHT):
        backend.send(1, bytes(8), extended=False)
    assert len(dev.writes) == 5 * gs_usb.MAX_IN_FLIGHT


def test_usb_write_failure_is_a_clear_error(opened):
    backend, dev, _ = opened()
    dev.write_error = usb_core.USBTimeoutError("timeout")
    with pytest.raises(TimeoutError, match="replugging"):
        backend.send(1, bytes(8), extended=False)
    dev.write_error = None
    for _ in range(gs_usb.MAX_IN_FLIGHT + 1):   # the failed frame does not hold a slot
        backend.send(1, bytes(8), extended=False)


def test_read_failure_stops_receive_and_is_reported(opened):
    backend, dev, _ = opened()
    dev.read_error = usb_core.USBError("No such device")
    assert wait_for(lambda: backend.error is not None)
    assert "No such device" in backend.error


def test_close_stops_channel_releases_and_disposes(opened):
    backend, dev, _ = opened()
    backend.close()
    assert dev.ctrl(gs_usb.BREQ_MODE)[-1][4] == struct.pack("<II", 0, 0)
    assert dev._ctx.released == [0] and dev._ctx.disposed >= 1
    assert dev.kernel_driver is False and dev.resets == 0   # left detached, never reset
    backend.close()
    with pytest.raises(ConnectError):
        backend.send(1, bytes(8), extended=False)


def test_open_failure_cleans_up_and_explains():
    dev = FakeUsbDevice()
    dev.ctrl_error = usb_core.USBError("Pipe error")
    backend = gs_usb.GsUsbBackend("gs_usb:ABC123", device=dev)
    with pytest.raises(ConnectError, match="replug"):
        backend.open()
    assert dev._ctx.released == [0] and dev._ctx.disposed >= 1
    assert dev.resets == 0


def test_list_adapters_reads_descriptors_only():
    devices = [FakeUsbDevice("B2", bus=1, port_numbers=(2, 3)), FakeUsbDevice("A1", bus=3, port_numbers=(1,))]
    adapters = gs_usb.list_adapters(devices)
    assert adapters == [
        {"serial": "A1", "port": "gs_usb:A1", "usb_path": "3-1", "product": "candleLight USB to CAN adapter",
         "manufacturer": "bytewerk", "vid_pid": "1d50:606f"},
        {"serial": "B2", "port": "gs_usb:B2", "usb_path": "1-2.3", "product": "candleLight USB to CAN adapter",
         "manufacturer": "bytewerk", "vid_pid": "1d50:606f"},
    ]
    for dev in devices:
        assert dev.controls == [] and dev._ctx.claimed == [] and dev._ctx.disposed == 1


def test_selects_the_adapter_by_serial(monkeypatch):
    a, b = FakeUsbDevice("A1"), FakeUsbDevice("B2")
    monkeypatch.setattr(gs_usb, "_find_devices", lambda: [a, b])
    backend = gs_usb.GsUsbBackend("gs_usb:B2")
    backend.open()
    try:
        assert b._ctx.claimed == [0] and a._ctx.claimed == [] and a._ctx.disposed == 1
    finally:
        backend.close()
    with pytest.raises(ConnectError, match="exactly one"):
        gs_usb.GsUsbBackend(None).open()
    with pytest.raises(ConnectError, match="no gs_usb adapter with serial 'C3'"):
        gs_usb.GsUsbBackend("C3").open()
    monkeypatch.setattr(gs_usb, "_find_devices", lambda: [a])
    only = gs_usb.GsUsbBackend()
    only.open()
    only.close()


def test_create_backend_knows_gs_usb():
    backend = create_backend("gs_usb", port="gs_usb:XYZ")
    assert isinstance(backend, gs_usb.GsUsbBackend) and backend.port == "gs_usb:XYZ"


def test_motor_over_gs_usb_with_a_late_gripper(opened):
    fakes = [FakeMotor(6, model="jhi", position=-0.3), FakeMotor(7, model="jhi", position=0.8, late=True)]
    bus = FakeBus(fakes)
    backend, dev, _ = opened(FakeUsbDevice(can_bus=bus))
    motors = {f.motor_id: Motor(make_joint(f.motor_id, f.model), backend) for f in fakes}
    lock = threading.Lock()

    def dispatch(can_id, data):
        motor = motors.get(route(can_id, data))
        if motor is not None:
            with lock:
                motor.handle_frame(can_id, data)

    backend.set_recv_callback(dispatch)
    assert motors[7].read_state(timeout=0.1).position == pytest.approx(0.8, abs=1e-3)
    motors[7].validate()
    token = motors[7].arm_watchdog(500)
    assert fakes[1].watchdog_ticks == 10_000
    motors[7].restore_watchdog(token)
    assert motors[6].read_state().position == pytest.approx(-0.3, abs=1e-3)
