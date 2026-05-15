"""Paxini GEN3 high-speed board tactile sensor driver for the Midas hand.

Uses the AA56 auto-push stream exclusively. The board pushes all configured
finger data in one packed frame; the reader thread consumes frames as fast as
the hardware delivers them (~83.3 Hz max). A separate publish thread caps how
often read_latest() sees a new value (default 60 Hz).

Typical usage::

    config = PaxiniConfig(port="/dev/ttyUSB1")
    with PaxiniHandSensor(config) as sensor:
        while True:
            data = sensor.read_latest()        # dict[str, ndarray (N, 3)]
            fz   = sensor.read_tactile_fz()    # dict[str, ndarray (N,)]
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import serial

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Protocol constants
# ---------------------------------------------------------------------------

_BAUDRATE = 921_600
_FRAME_HEAD_AUTO_PUSH = b"\xAA\x56"
_FRAME_HEAD_REQUEST = b"\x55\xAA"

_ENABLE_AUTO_PUSH = bytes.fromhex("55AA00101700010001D8")
_DISABLE_AUTO_PUSH = bytes.fromhex("55AA00101700010000D9")

# Register 0x0016: controls what the board includes in each auto-push frame.
# 0x03 = resultant force (6 bytes/finger) + distributed force (N*3 bytes/finger).
_ADDR_DATA_TYPE = 0x0016
_DATA_TYPE_RESULTANT_AND_DISTRIBUTED = 0x03

# 1 LSB = 0.1 N per Paxini GEN3 spec.
_FORCE_SCALE_N = 0.1

# The high-speed board's hardware-limited auto-push ceiling is ~83.3 Hz.
# 60 Hz is the default publish rate to leave headroom below that ceiling.
_DEFAULT_PUBLISH_HZ = 60.0

_DEFAULT_DISCARD_STARTUP_FRAMES = 5
_DEFAULT_MEDIAN_WINDOW = 3
_DEFAULT_SERIAL_SETTLE_S = 0.75
_DEFAULT_RESPONSE_TIMEOUT_S = 1.0

# ---------------------------------------------------------------------------
# Per-finger hardware defaults for the 4-finger Midas hand.
# device_address = module_number + 1  (Paxini spec §3.1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FingerDefault:
    device_address: int
    force_points: int


_FINGER_DEFAULTS: dict[str, _FingerDefault] = {
    "thumb":  _FingerDefault(device_address=1, force_points=127),
    "index":  _FingerDefault(device_address=2, force_points=52),
    "middle": _FingerDefault(device_address=3, force_points=52),
    "ring":   _FingerDefault(device_address=4, force_points=52),
}

_DEFAULT_FINGERS: list[str] = list(_FINGER_DEFAULTS)


# ---------------------------------------------------------------------------
# Public config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PaxiniConfig:
    """Configuration for PaxiniHandSensor.

    Args:
        port: Serial port, e.g. ``"/dev/ttyUSB1"``.
        fingers: Ordered list of finger names to read. Defaults to all four
            fingers (thumb, index, middle, ring). Every name in this list must
            be physically connected; connect() raises if any are missing.
        baudrate: Serial baud rate. 921600 matches the Paxini board default.
        publish_rate_hz: Rate at which read_latest() sees a new value. The
            board streams at up to ~83.3 Hz; this caps delivery independently.
        scale_n: Force scale factor (N per LSB). 0.1 matches Paxini GEN3 spec.
        signed_z: Parse Fz as signed int8. Default False keeps Fz unsigned,
            matching the Paxini communication protocol document.
        discard_startup_frames: AA56 frames to throw away after enabling the
            stream before treating data as valid.
        median_window: Rolling median window over consecutive frames for noise
            reduction. Set 1 to disable.
        response_timeout_s: Seconds to wait for one complete AA56 frame.
        serial_settle_s: Seconds to wait after opening the port before writing.
        dtr: USB serial DTR line state. False is the Paxini board default.
        rts: USB serial RTS line state. True is the Paxini board default.
    """

    port: str
    fingers: list[str] = field(default_factory=lambda: list(_DEFAULT_FINGERS))
    baudrate: int = _BAUDRATE
    publish_rate_hz: float = _DEFAULT_PUBLISH_HZ
    scale_n: float = _FORCE_SCALE_N
    signed_z: bool = False
    discard_startup_frames: int = _DEFAULT_DISCARD_STARTUP_FRAMES
    median_window: int = _DEFAULT_MEDIAN_WINDOW
    response_timeout_s: float = _DEFAULT_RESPONSE_TIMEOUT_S
    serial_settle_s: float = _DEFAULT_SERIAL_SETTLE_S
    dtr: bool = False
    rts: bool = True


# ---------------------------------------------------------------------------
# Internal finger descriptor
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Finger:
    name: str
    device_address: int
    force_points: int


def _build_fingers(names: list[str]) -> list[_Finger]:
    fingers = []
    for name in names:
        if name not in _FINGER_DEFAULTS:
            raise ValueError(
                f"Unknown finger {name!r}. Valid names: {list(_FINGER_DEFAULTS)}"
            )
        d = _FINGER_DEFAULTS[name]
        fingers.append(_Finger(name, d.device_address, d.force_points))
    return fingers


# ---------------------------------------------------------------------------
# Protocol helpers (all private)
# ---------------------------------------------------------------------------


def _lrc(data: bytes) -> int:
    return (-sum(data)) & 0xFF


def _build_adapter_write_frame(address: int, payload: bytes) -> bytes:
    frame = (
        _FRAME_HEAD_REQUEST
        + bytes([0x00, 0x10])
        + address.to_bytes(2, "little")
        + len(payload).to_bytes(2, "little")
        + payload
    )
    return frame + bytes([_lrc(frame)])


def _write_no_ack(ser: serial.Serial, address: int, payload: bytes) -> None:
    """Write an adapter register without waiting for an acknowledgement."""
    frame = _build_adapter_write_frame(address, payload)
    ser.write(frame)
    logger.debug("TX adapter no-ack %s", frame.hex(" ").upper())


def _drain(
    ser: serial.Serial,
    *,
    quiet_s: float = 0.15,
    timeout_s: float = 1.0,
) -> None:
    """Discard all pending serial input until the line is quiet."""
    deadline = time.monotonic() + timeout_s
    quiet_deadline = time.monotonic() + quiet_s
    while time.monotonic() < deadline:
        if ser.in_waiting:
            ser.read(ser.in_waiting)
            quiet_deadline = time.monotonic() + quiet_s
        elif time.monotonic() >= quiet_deadline:
            break
        time.sleep(0.005)


def _read_frame(
    ser: serial.Serial,
    timeout_s: float,
    buffer: bytearray,
) -> bytes:
    """Block until one valid AA56 auto-push frame is available.

    ``buffer`` is a persistent bytearray shared across calls so that partial
    frames left over from the previous read are not discarded.
    """
    deadline = time.monotonic() + timeout_s
    max_body = 8192

    while time.monotonic() < deadline:
        # Try to find and extract a complete frame from what's already buffered.
        while True:
            head = buffer.find(_FRAME_HEAD_AUTO_PUSH)
            if head < 0:
                keep = buffer[-1:] if buffer.endswith(_FRAME_HEAD_AUTO_PUSH[:1]) else b""
                buffer.clear()
                buffer.extend(keep)
                break
            if head > 0:
                del buffer[:head]

            if len(buffer) < 5:
                break

            valid_frame_len = int.from_bytes(buffer[3:5], "little")
            if valid_frame_len < 1 or valid_frame_len > max_body:
                del buffer[0]
                continue

            total_len = 5 + valid_frame_len + 1
            if len(buffer) < total_len:
                break

            frame = bytes(buffer[:total_len])
            if frame[-1] != _lrc(frame[:-1]):
                logger.debug(
                    "LRC mismatch, discarding byte; expected=0x%02X got=0x%02X",
                    _lrc(frame[:-1]),
                    frame[-1],
                )
                del buffer[0]
                continue

            del buffer[:total_len]
            logger.debug("RX auto-push frame (%d bytes)", total_len)
            return frame

        # No complete frame yet — read more bytes from the serial port.
        waiting = ser.in_waiting
        if waiting:
            buffer.extend(ser.read(waiting))
        else:
            time.sleep(0.001)

    raise TimeoutError(
        f"No valid AA56 auto-push frame received within {timeout_s:.1f}s"
    )


def _parse_frame_payload(frame: bytes) -> bytes:
    """Return the payload bytes between the header and the LRC."""
    valid_frame_len = int.from_bytes(frame[3:5], "little")
    total_len = 5 + valid_frame_len + 1
    status = frame[5]
    if status != 0:
        raise RuntimeError(f"Auto-push status byte is 0x{status:02X} (non-zero indicates board error)")
    return frame[6 : total_len - 1]


def _parse_force_vectors(
    data: bytes,
    n_points: int,
    *,
    scale_n: float,
    signed_z: bool,
) -> np.ndarray:
    expected = n_points * 3
    if len(data) < expected:
        raise ValueError(f"Expected {expected} force bytes, got {len(data)}")
    raw = (
        np.frombuffer(data[:expected], dtype=np.uint8)
        .reshape(n_points, 3)
        .astype(np.int16)
    )
    vectors = np.empty((n_points, 3), dtype=np.float64)
    vectors[:, 0] = np.where(raw[:, 0] <= 127, raw[:, 0], raw[:, 0] - 256)
    vectors[:, 1] = np.where(raw[:, 1] <= 127, raw[:, 1], raw[:, 1] - 256)
    if signed_z:
        vectors[:, 2] = np.where(raw[:, 2] <= 127, raw[:, 2], raw[:, 2] - 256)
    else:
        vectors[:, 2] = raw[:, 2]
    vectors *= scale_n
    return vectors


def _parse_digits(
    payload: bytes,
    fingers: list[_Finger],
    *,
    scale_n: float,
    signed_z: bool,
) -> dict[str, np.ndarray]:
    """Parse one auto-push payload into per-finger force matrices.

    Each finger block in the payload is:
        6 bytes  — resultant force (3 × int16 Fx/Fy/Fz), skipped here
        N × 3 bytes — distributed force points
    """
    offset = 0
    result: dict[str, np.ndarray] = {}
    for finger in fingers:
        resultant_len = 6
        data_len = finger.force_points * 3
        block_len = resultant_len + data_len
        if len(payload) < offset + block_len:
            raise ValueError(
                f"Auto-push payload too short for finger {finger.name!r}: "
                f"payload has {len(payload)} bytes but need at least "
                f"{offset + block_len}. "
                f"Verify that {finger.name!r} is physically connected to the board."
            )
        force_data = payload[offset + resultant_len : offset + block_len]
        result[finger.name] = _parse_force_vectors(
            force_data, finger.force_points, scale_n=scale_n, signed_z=signed_z
        )
        offset += block_len
    return result


def _median_filter(window: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for name in window[-1]:
        arrays = [w[name] for w in window if name in w]
        if arrays:
            result[name] = np.median(np.stack(arrays, axis=0), axis=0)
    return result


# ---------------------------------------------------------------------------
# Public sensor class
# ---------------------------------------------------------------------------


class PaxiniHandSensor:
    """Paxini high-speed board driver using the AA56 auto-push stream.

    The reader thread consumes frames as fast as the board delivers them
    (~83.3 Hz hardware ceiling). A separate publish thread controls how often
    read_latest() reflects a new value (default 60 Hz).

    Usage::

        config = PaxiniConfig(port="/dev/ttyUSB1")

        sensor = PaxiniHandSensor(config)
        sensor.connect()
        data = sensor.read_latest()   # dict[str, ndarray shape (N, 3)]
        fz   = sensor.read_tactile_fz()  # dict[str, ndarray shape (N,)]
        sensor.disconnect()

        # Or as a context manager:
        with PaxiniHandSensor(config) as sensor:
            data = sensor.read_latest()
    """

    def __init__(self, config: PaxiniConfig) -> None:
        self.config = config
        self._fingers = _build_fingers(config.fingers)

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._read_thread: Optional[threading.Thread] = None
        self._publish_thread: Optional[threading.Thread] = None
        self._serial: Optional[serial.Serial] = None
        self._buffer = bytearray()

        self._raw_latest: Optional[dict[str, np.ndarray]] = None
        self._raw_seq = 0
        self._pub_seq = 0
        self._latest: Optional[dict[str, np.ndarray]] = None
        self._connected = False
        self._error: Optional[str] = None

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        """Open the serial port, configure the board, validate all fingers, and start streaming.

        Raises:
            RuntimeError: If already connected, or if any configured finger is
                not detected on the board.
            ValueError: If a finger name in config.fingers is not recognised.
            serial.SerialException: If the serial port cannot be opened.
        """
        if self._connected:
            raise RuntimeError("Already connected. Call disconnect() first.")

        ser = serial.Serial(
            port=self.config.port,
            baudrate=self.config.baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=1,
            write_timeout=0.5,
            inter_byte_timeout=0.001,
            exclusive=True,
            xonxoff=False,
            rtscts=False,
        )
        ser.dtr = self.config.dtr
        ser.rts = self.config.rts

        time.sleep(self.config.serial_settle_s)
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        logger.info(
            "Opened %s at %d baud (DTR=%s RTS=%s)",
            self.config.port,
            self.config.baudrate,
            self.config.dtr,
            self.config.rts,
        )

        # Stop any running stream, set data type, then re-enable.
        # Use no-ack writes to avoid CDC driver timing issues with ACK responses.
        ser.write(_DISABLE_AUTO_PUSH)
        time.sleep(0.2)
        _drain(ser)
        _write_no_ack(ser, _ADDR_DATA_TYPE, bytes([_DATA_TYPE_RESULTANT_AND_DISTRIBUTED]))
        time.sleep(0.05)
        _drain(ser)
        ser.reset_input_buffer()
        self._buffer.clear()

        ser.write(_ENABLE_AUTO_PUSH)
        logger.debug("TX enable auto-push")

        # Discard transient startup frames before treating data as valid.
        for _ in range(max(0, self.config.discard_startup_frames)):
            _read_frame(ser, self.config.response_timeout_s, self._buffer)
        if self.config.discard_startup_frames > 0:
            logger.info("Discarded %d startup frames", self.config.discard_startup_frames)

        # Validate: parse one complete frame to confirm every configured finger
        # is present. If a finger is missing the payload will be too short and
        # _parse_digits raises a clear per-finger error.
        frame = _read_frame(ser, self.config.response_timeout_s, self._buffer)
        try:
            _parse_digits(
                _parse_frame_payload(frame),
                self._fingers,
                scale_n=self.config.scale_n,
                signed_z=self.config.signed_z,
            )
        except (ValueError, RuntimeError) as exc:
            try:
                ser.write(_DISABLE_AUTO_PUSH)
            except serial.SerialException:
                pass
            ser.close()
            raise RuntimeError(
                f"Finger validation failed: {exc}"
            ) from exc

        logger.info(
            "Connected. Fingers: %s",
            ", ".join(f"{f.name}({f.force_points}pts)" for f in self._fingers),
        )

        self._serial = ser
        self._connected = True
        self._stop.clear()
        self._error = None

        self._publish_thread = threading.Thread(
            target=self._publish_loop, name="paxini-publisher", daemon=True
        )
        self._read_thread = threading.Thread(
            target=self._read_loop, name="paxini-reader", daemon=True
        )
        self._publish_thread.start()
        self._read_thread.start()

    def disconnect(self) -> None:
        """Stop streaming and close the serial port."""
        self._connected = False
        self._stop.set()
        if self._read_thread:
            self._read_thread.join(timeout=2.0)
            self._read_thread = None
        if self._publish_thread:
            self._publish_thread.join(timeout=2.0)
            self._publish_thread = None
        if self._serial and self._serial.is_open:
            try:
                self._serial.write(_DISABLE_AUTO_PUSH)
            except serial.SerialException:
                pass
            self._serial.close()
        self._serial = None

    def close(self) -> None:
        """Alias for disconnect(), for compatibility with MidasHand.close()."""
        self.disconnect()

    def read_latest(self) -> dict[str, np.ndarray]:
        """Return the latest tactile frame.

        Returns:
            dict mapping finger name to an ``(N, 3)`` float64 array with
            columns ``[Fx, Fy, Fz]`` in Newtons.

        Raises:
            RuntimeError: If not connected, or if no frame has arrived yet.
        """
        if not self._connected:
            raise RuntimeError(
                "PaxiniHandSensor is not connected. Call connect() first."
            )
        with self._lock:
            if self._latest is None:
                raise RuntimeError(
                    "No tactile data available yet. The stream may still be starting up."
                )
            return self._latest

    def read_tactile_fx(self) -> dict[str, np.ndarray]:
        """Return Fx for each finger as ``dict[name → (N,) array]`` in Newtons."""
        return {k: v[:, 0] for k, v in self.read_latest().items()}

    def read_tactile_fy(self) -> dict[str, np.ndarray]:
        """Return Fy for each finger as ``dict[name → (N,) array]`` in Newtons."""
        return {k: v[:, 1] for k, v in self.read_latest().items()}

    def read_tactile_fz(self) -> dict[str, np.ndarray]:
        """Return Fz for each finger as ``dict[name → (N,) array]`` in Newtons."""
        return {k: v[:, 2] for k, v in self.read_latest().items()}

    # ------------------------------------------------------------------
    # Internal threads
    # ------------------------------------------------------------------

    def _read_loop(self) -> None:
        """Read AA56 frames as fast as the board delivers them (~83.3 Hz max)."""
        median_buf: deque[dict[str, np.ndarray]] = deque(
            maxlen=max(1, self.config.median_window)
        )
        try:
            while not self._stop.is_set():
                frame = _read_frame(
                    self._serial,
                    self.config.response_timeout_s,
                    self._buffer,
                )
                sample = _parse_digits(
                    _parse_frame_payload(frame),
                    self._fingers,
                    scale_n=self.config.scale_n,
                    signed_z=self.config.signed_z,
                )
                median_buf.append(sample)
                if self.config.median_window > 1 and len(median_buf) >= self.config.median_window:
                    sample = _median_filter(list(median_buf))
                with self._lock:
                    self._raw_latest = sample
                    self._raw_seq += 1
        except Exception as exc:
            if not self._stop.is_set():
                logger.exception("Paxini reader stopped unexpectedly")
                with self._lock:
                    self._error = str(exc)
                self._connected = False

    def _publish_loop(self) -> None:
        """Promote the latest raw sample to read_latest() at publish_rate_hz."""
        interval_s = 1.0 / self.config.publish_rate_hz
        next_s = time.monotonic()
        while not self._stop.is_set():
            with self._lock:
                if self._raw_latest is not None and self._raw_seq != self._pub_seq:
                    self._latest = self._raw_latest
                    self._pub_seq = self._raw_seq
            next_s += interval_s
            sleep_s = next_s - time.monotonic()
            if sleep_s <= 0:
                next_s = time.monotonic()
                continue
            self._stop.wait(sleep_s)

    def __enter__(self) -> "PaxiniHandSensor":
        self.connect()
        return self

    def __exit__(self, *args: object) -> None:
        self.disconnect()
