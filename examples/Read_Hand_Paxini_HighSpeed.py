"""Read hand Paxini force vectors through the high-speed board.

By default this script uses the high-speed board's auto-push stream:

    Enable stream:
        55 AA 00 10 17 00 01 00 01 D8

    Stream frame:
        AA 56 <reserved> <valid_frame_len:2> <status>
        <6-byte thumb resultant> <thumb force bytes>
        <6-byte index resultant> <index force bytes>
        <6-byte middle resultant> <middle force bytes>
        <6-byte ring resultant> <ring force bytes> <lrc>

The high-speed board request/response format is available with
``--mode request``:

    Request:
        55 AA 00 <03 read | 10 write> <register:2> <data_len:2> [data] <lrc>

    Response:
        AA 55 00 <03 read | 10 write> <register:2> <data_len:2> [data] <lrc>

The native sensor UART request/response format is available with
``--mode native-request``. This is the protocol used by the single-sensor
USB/UART adapter, not the high-speed board:

    Request:
        55 AA <frame_len:2> <device> 00 FB <start_addr:4> <data_len:2> <lrc>

    Response:
        AA 55 <frame_len:2> <device> 00 FB <start_addr:4>
        <returned_len:2> <status> <returned_data> <lrc>

The device address is module number + 1. Defaults here match the current hand:
thumb module 0 -> device address 1, index module 1 -> device address 2,
middle module 2 -> device address 3, ring module 3 -> device address 4.

Run:
    python examples/Read_Hand_Paxini_HighSpeed.py --port /dev/ttyUSB0

Use ``--calibrate`` only with ``--mode native-request`` and unloaded sensors.
It matches ``Read_Single_Sensor_Usb.py`` by writing ``0x01`` to native Paxini
calibration register ``0x0003`` for each configured device address. The
high-speed-board stream/request protocols do not document a sensor calibration
command. Use ``--software-zero`` explicitly if you only want local baseline
subtraction for plotting.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import serial
import serial.tools.list_ports


BAUDRATE = 921_600

FRAME_HEAD_REQUEST = b"\x55\xAA"
FRAME_HEAD_RESPONSE = b"\xAA\x55"
FRAME_HEAD_AUTO_PUSH = b"\xAA\x56"
RESERVED = 0x00
FUNC_WRITE_CONFIG = 0x79
FUNC_READ_APPLICATION = 0xFB

ENABLE_AUTO_PUSH_FRAME = bytes.fromhex("55AA00101700010001D8")
DISABLE_AUTO_PUSH_FRAME = bytes.fromhex("55AA00101700010000D9")
READ_VERSION_FRAME = bytes.fromhex("55AA000300000F00EF")

ADDR_CALIBRATION = 0x0003
ADDR_DISTRIBUTED_FORCE = 0x040E
FORCE_SCALE_N = 0.1
ASSET_DIR = Path(__file__).resolve().parents[1] / "midas_hand_api" / "assets"

BOARD_DISTRIBUTED_FORCE_ADDRESSES = {
    "thumb": 0x1000,
    "index": 0x1800,
    "middle": 0x2000,
    "ring": 0x2800,
    "pinky": 0x3000,
    "palm": 0x3800,
}
BOARD_SENSOR_COMBINATION_ADDRESS = 0x0010
BOARD_AUTO_POST_BACK_DATA_TYPE_ADDRESS = 0x0016
BOARD_FORCE_POINT_COUNT_ADDRESS = 0x0030
BOARD_FORCE_POINT_COUNT_LENGTH = 0x38
AUTO_PUSH_DATA_TYPE_RESULTANT_AND_DISTRIBUTED = 0x03
BOARD_DIGIT_MODULE_START = {
    "thumb": 0,
    "index": 4,
    "middle": 8,
    "ring": 12,
    "pinky": 16,
    "palm": 20,
}
BOARD_DIGIT_MODULE_STRIDE = {
    "thumb": 0x0200,
    "index": 0x0200,
    "middle": 0x0200,
    "ring": 0x0200,
    "pinky": 0x0200,
    "palm": 0x0100,
}

THUMB_DEVICE_ADDRESS = 1
INDEX_DEVICE_ADDRESS = 2
MIDDLE_DEVICE_ADDRESS = 3
RING_DEVICE_ADDRESS = 4
THUMB_FORCE_POINTS = 127
FINGER_FORCE_POINTS = 52
INDEX_FORCE_POINTS = FINGER_FORCE_POINTS
MIDDLE_FORCE_POINTS = FINGER_FORCE_POINTS
RING_FORCE_POINTS = FINGER_FORCE_POINTS
THUMB_COORDS_PATH = ASSET_DIR / "paxini_fingertip_26mm_127pts.csv"
FINGER_COORDS_PATH = ASSET_DIR / "paxini_fingertip_15mm_52pts.csv"

DEFAULT_READ_INTERVAL_S = 0.03
DEFAULT_UPDATE_MS = 100
DEFAULT_PUBLISH_RATE_HZ = 60.0
DEFAULT_SERIAL_SETTLE_S = 0.75
DEFAULT_MAX_RESPONSE_BODY_LENGTH = 4096
DEFAULT_READ_CHUNK_SIZE = 32
DEFAULT_CALIBRATION_SAMPLES = 20
DEFAULT_DISCARD_STARTUP_FRAMES = 5
DEFAULT_MEDIAN_WINDOW = 3

logger = logging.getLogger(__name__)


Dash = None
dcc = None
html = None
Input = None
Output = None
go = None


@dataclass(frozen=True)
class PaxiniResponse:
    device_address: int
    function: int
    address: int
    returned_length: int
    status: int
    data: bytes


@dataclass(frozen=True)
class DigitConfig:
    name: str
    device_address: int
    force_points: int
    coordinate_path: Optional[Path] = None


@dataclass(frozen=True)
class DigitSample:
    timestamp_s: float
    vectors_n: np.ndarray


@dataclass(frozen=True)
class BoardLayout:
    sensor_combination: bytes
    force_point_counts: list[int]


@dataclass(frozen=True)
class HandSample:
    timestamp_s: float
    digits: dict[str, DigitSample]


def load_dash() -> None:
    """Load optional Plotly/Dash dependencies only when the app starts."""

    global Dash, dcc, html, Input, Output, go
    if Dash is not None:
        return
    try:
        from dash import Dash as Dash_
        from dash import dcc as dcc_
        from dash import html as html_
        from dash.dependencies import Input as Input_
        from dash.dependencies import Output as Output_
        import plotly.graph_objects as go_
    except ImportError as exc:
        raise SystemExit(
            "This live viewer needs optional visualization dependencies:\n"
            '  python -m pip install -e ".[viz]"\n'
            "or:\n"
            "  python -m pip install dash plotly"
        ) from exc

    Dash = Dash_
    dcc = dcc_
    html = html_
    Input = Input_
    Output = Output_
    go = go_


def calculate_lrc(data: bytes) -> int:
    """Return the 8-bit two's-complement LRC over all preceding bytes."""

    return (-sum(data)) & 0xFF


def format_bytes(data: bytes, max_bytes: int = 80) -> str:
    if len(data) <= max_bytes:
        return data.hex(" ").upper()
    head_len = max_bytes // 2
    tail_len = max_bytes - head_len
    return (
        f"{data[:head_len].hex(' ').upper()} ... "
        f"{data[-tail_len:].hex(' ').upper()} ({len(data)} bytes)"
    )


def drain_serial_input(
    ser: serial.Serial,
    *,
    quiet_s: float = 0.15,
    timeout_s: float = 1.0,
) -> bytes:
    drained = bytearray()
    deadline = time.monotonic() + timeout_s
    quiet_deadline = time.monotonic() + quiet_s
    while time.monotonic() < deadline:
        waiting = ser.in_waiting
        if waiting:
            chunk = ser.read(waiting)
            drained.extend(chunk)
            quiet_deadline = time.monotonic() + quiet_s
        elif time.monotonic() >= quiet_deadline:
            break
        time.sleep(0.005)
    if drained:
        logger.debug("Drained serial input %s", format_bytes(bytes(drained)))
    return bytes(drained)


def parse_int(value: str) -> int:
    return int(value, 0)


def parse_path(value: str) -> Path:
    return Path(value).expanduser()


def load_coordinate_csv(path: Path) -> np.ndarray:
    coordinates = []
    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            try:
                coordinates.append(
                    (
                        float(row["X"]),
                        float(row["Y"]),
                        float(row["Z"]),
                    )
                )
            except KeyError as exc:
                raise ValueError(
                    f"Coordinate file {path} must contain X, Y, and Z columns"
                ) from exc
    if not coordinates:
        raise ValueError(f"Coordinate file {path} is empty")
    return np.asarray(coordinates, dtype=np.float64)


def load_digit_coordinates(digits: list[DigitConfig]) -> dict[str, np.ndarray]:
    coordinates: dict[str, np.ndarray] = {}
    cache: dict[Path, np.ndarray] = {}
    for digit in digits:
        if digit.coordinate_path is None:
            continue
        path = digit.coordinate_path
        if path not in cache:
            cache[path] = load_coordinate_csv(path)
        digit_coordinates = cache[path]
        if len(digit_coordinates) != digit.force_points:
            raise ValueError(
                f"{digit.name} coordinate count mismatch: {path} has "
                f"{len(digit_coordinates)} points, but {digit.force_points} "
                "force points are configured"
            )
        coordinates[digit.name] = digit_coordinates
    return coordinates


def build_read_frame(device_address: int, address: int, data_length: int) -> bytes:
    if not 1 <= device_address <= 255:
        raise ValueError("device_address must be in [1, 255]")
    if data_length < 1 or data_length > 0xFFFF:
        raise ValueError("data_length must be in [1, 65535]")

    body = (
        bytes([device_address, RESERVED, FUNC_READ_APPLICATION])
        + address.to_bytes(4, "little")
        + data_length.to_bytes(2, "little")
    )
    frame = FRAME_HEAD_REQUEST + len(body).to_bytes(2, "little") + body
    return frame + bytes([calculate_lrc(frame)])


def build_write_frame(
    device_address: int,
    address: int,
    payload: bytes,
) -> bytes:
    if not 1 <= device_address <= 255:
        raise ValueError("device_address must be in [1, 255]")
    if not payload:
        raise ValueError("payload cannot be empty")
    if len(payload) > 0xFFFF:
        raise ValueError("payload is too large")

    body = (
        bytes([device_address, RESERVED, FUNC_WRITE_CONFIG])
        + address.to_bytes(4, "little")
        + len(payload).to_bytes(2, "little")
        + payload
    )
    frame = FRAME_HEAD_REQUEST + len(body).to_bytes(2, "little") + body
    return frame + bytes([calculate_lrc(frame)])


def build_adapter_write_frame(address: int, payload: bytes) -> bytes:
    """Build the high-speed-board adapter command style used for auto-push."""

    if not payload:
        raise ValueError("payload cannot be empty")
    frame = (
        FRAME_HEAD_REQUEST
        + bytes([RESERVED, 0x10])
        + address.to_bytes(2, "little")
        + len(payload).to_bytes(2, "little")
        + payload
    )
    return frame + bytes([calculate_lrc(frame)])


def build_adapter_read_frame(address: int, data_length: int) -> bytes:
    """Build the high-speed-board adapter read command style."""

    frame = (
        FRAME_HEAD_REQUEST
        + bytes([RESERVED, 0x03])
        + address.to_bytes(2, "little")
        + data_length.to_bytes(2, "little")
    )
    return frame + bytes([calculate_lrc(frame)])


def read_response(
    ser: serial.Serial,
    timeout_s: float,
    max_body_length: int = DEFAULT_MAX_RESPONSE_BODY_LENGTH,
) -> bytes:
    response = b""
    raw_response = b""
    expected_total_length: Optional[int] = None
    deadline = time.monotonic() + timeout_s

    while time.monotonic() < deadline:
        waiting = ser.in_waiting
        if waiting:
            chunk = ser.read(waiting)
            raw_response += chunk
            response += chunk
            logger.debug("RX chunk %s", format_bytes(chunk))

            head_index = response.find(FRAME_HEAD_RESPONSE)
            if head_index < 0:
                response = response[-1:] if response.endswith(FRAME_HEAD_RESPONSE[:1]) else b""
                continue
            if head_index > 0:
                response = response[head_index:]

            if len(response) >= 4 and expected_total_length is None:
                body_length = int.from_bytes(response[2:4], "little")
                if body_length < 10 or body_length > max_body_length:
                    logger.debug(
                        "Discarding invalid response candidate: body_length=%s raw=%s",
                        body_length,
                        response.hex(" ").upper(),
                    )
                    response = response[1:]
                    expected_total_length = None
                    continue
                expected_total_length = 4 + body_length + 1

            if expected_total_length is not None and len(response) >= expected_total_length:
                frame = response[:expected_total_length]
                logger.debug("RX %s", format_bytes(frame))
                return frame

        time.sleep(0.001)

    if not raw_response:
        raise TimeoutError("No Paxini response received")
    raise TimeoutError(
        f"Incomplete Paxini response: got {len(raw_response)} raw bytes, "
        f"expected {expected_total_length or 'unknown'}; "
        f"raw={raw_response.hex(' ').upper()}"
    )


def read_adapter_response(
    ser: serial.Serial,
    timeout_s: float,
    *,
    expected_function: Optional[int] = None,
    expected_address: Optional[int] = None,
    max_data_length: int = DEFAULT_MAX_RESPONSE_BODY_LENGTH,
) -> bytes:
    response = bytearray()
    raw_response = bytearray()
    deadline = time.monotonic() + timeout_s

    while time.monotonic() < deadline:
        while True:
            head_index = response.find(FRAME_HEAD_RESPONSE)
            if head_index < 0:
                keep = response[-1:] if response.endswith(FRAME_HEAD_RESPONSE[:1]) else b""
                response.clear()
                response.extend(keep)
                break
            if head_index > 0:
                del response[:head_index]

            if len(response) < 9:
                break

            data_length = int.from_bytes(response[6:8], "little")
            if data_length > max_data_length:
                logger.debug(
                    "Discarding invalid adapter candidate: data_length=%s raw=%s",
                    data_length,
                    bytes(response).hex(" ").upper(),
                )
                del response[0]
                continue
            expected_total_length = 8 + data_length + 1
            if len(response) < expected_total_length:
                break

            frame = bytes(response[:expected_total_length])
            expected_lrc = calculate_lrc(frame[:-1])
            if frame[-1] != expected_lrc:
                logger.debug(
                    "Discarding adapter candidate with LRC mismatch: "
                    "calculated 0x%02X, got 0x%02X; raw=%s",
                    expected_lrc,
                    frame[-1],
                    format_bytes(frame),
                )
                del response[0]
                continue

            function = frame[3]
            address = int.from_bytes(frame[4:6], "little")
            if (
                expected_function is not None
                and function != expected_function
                or expected_address is not None
                and address != expected_address
            ):
                logger.debug(
                    "Discarding stale adapter response: function=0x%02X address=0x%04X raw=%s",
                    function,
                    address,
                    format_bytes(frame),
                )
                del response[:expected_total_length]
                continue

            del response[:expected_total_length]
            logger.debug("RX adapter %s", format_bytes(frame))
            return frame

        waiting = ser.in_waiting
        if waiting:
            chunk = ser.read(waiting)
            raw_response.extend(chunk)
            response.extend(chunk)
            logger.debug("RX chunk %s", format_bytes(chunk))

        time.sleep(0.001)

    if not raw_response:
        raise TimeoutError("No adapter response received")
    raise TimeoutError(
        f"Incomplete adapter response: got {len(raw_response)} raw bytes; "
        f"raw={bytes(raw_response).hex(' ').upper()}"
    )


def read_adapter_version(
    ser: serial.Serial,
    response_timeout_s: float,
    retries: int = 3,
) -> Optional[str]:
    last_error: Optional[Exception] = None
    for attempt in range(1, max(1, retries) + 1):
        ser.reset_input_buffer()
        bytes_sent = ser.write(READ_VERSION_FRAME)
        if bytes_sent != len(READ_VERSION_FRAME):
            raise IOError(
                f"Serial write incomplete: sent {bytes_sent}/{len(READ_VERSION_FRAME)} bytes"
            )
        logger.debug(
            "TX adapter version %s attempt %s/%s",
            READ_VERSION_FRAME.hex(" ").upper(),
            attempt,
            max(1, retries),
        )

        try:
            response = read_adapter_response(
                ser,
                response_timeout_s,
                expected_function=0x03,
                expected_address=0x0000,
            )
            break
        except TimeoutError as exc:
            last_error = exc
            if attempt >= max(1, retries):
                raise
            logger.debug("Adapter version probe attempt %s failed: %s", attempt, exc)
            time.sleep(0.1)
    else:
        raise TimeoutError("No adapter version response received") from last_error

    if response[3] != 0x03:
        raise ValueError(f"Unexpected version response function: 0x{response[3]:02X}")
    data = response[8:-1]
    version = data.decode("ascii", errors="ignore").strip()
    logger.info("Adapter version: %s", version or data.hex(" ").upper())
    drain_serial_input(ser)
    return version or None


def read_adapter_register(
    ser: serial.Serial,
    address: int,
    data_length: int,
    response_timeout_s: float,
    *,
    reset_input: bool,
) -> bytes:
    if data_length < 1 or data_length > 512:
        raise ValueError("High-speed-board request reads must be in [1, 512] bytes")

    frame = build_adapter_read_frame(address, data_length)
    if reset_input:
        ser.reset_input_buffer()
    bytes_sent = ser.write(frame)
    if bytes_sent != len(frame):
        raise IOError(f"Serial write incomplete: sent {bytes_sent}/{len(frame)} bytes")
    logger.debug("TX adapter read %s", frame.hex(" ").upper())

    response = read_adapter_response(
        ser,
        response_timeout_s,
        expected_function=0x03,
        expected_address=address,
    )
    if response[3] != 0x03:
        raise ValueError(f"Unexpected adapter read response function: 0x{response[3]:02X}")
    returned_address = int.from_bytes(response[4:6], "little")
    if returned_address != address:
        raise ValueError(
            f"Adapter read address mismatch: expected 0x{address:04X}, "
            f"got 0x{returned_address:04X}"
        )
    returned_length = int.from_bytes(response[6:8], "little")
    data = response[8:-1]
    if len(data) != returned_length:
        raise ValueError(
            f"Adapter read length mismatch: header says {returned_length}, got {len(data)}"
        )
    return data


def read_board_layout(
    ser: serial.Serial,
    response_timeout_s: float,
) -> BoardLayout:
    sensor_combination = read_adapter_register(
        ser,
        BOARD_SENSOR_COMBINATION_ADDRESS,
        4,
        response_timeout_s,
        reset_input=True,
    )
    count_data = read_adapter_register(
        ser,
        BOARD_FORCE_POINT_COUNT_ADDRESS,
        BOARD_FORCE_POINT_COUNT_LENGTH,
        response_timeout_s,
        reset_input=True,
    )
    force_point_counts = [
        int.from_bytes(count_data[offset : offset + 2], "little")
        for offset in range(0, len(count_data), 2)
    ]
    logger.info(
        "Board sensor combination: %s; force point counts: %s",
        sensor_combination.hex(" ").upper(),
        force_point_counts,
    )
    return BoardLayout(sensor_combination, force_point_counts)


def read_auto_push_frame(
    ser: serial.Serial,
    timeout_s: float,
    max_body_length: int = DEFAULT_MAX_RESPONSE_BODY_LENGTH,
    buffer: Optional[bytearray] = None,
) -> bytes:
    response = buffer if buffer is not None else bytearray()
    raw_response = bytearray()
    expected_total_length: Optional[int] = None
    deadline = time.monotonic() + timeout_s

    while time.monotonic() < deadline:
        while True:
            head_index = response.find(FRAME_HEAD_AUTO_PUSH)
            if head_index < 0:
                keep = response[-1:] if response.endswith(FRAME_HEAD_AUTO_PUSH[:1]) else b""
                response.clear()
                response.extend(keep)
                expected_total_length = None
                break
            if head_index > 0:
                del response[:head_index]
                expected_total_length = None

            if len(response) < 5:
                break

            valid_frame_length = int.from_bytes(response[3:5], "little")
            if valid_frame_length < 1 or valid_frame_length > max_body_length:
                logger.debug(
                    "Discarding invalid auto-push candidate: valid_frame_length=%s raw=%s",
                    valid_frame_length,
                    bytes(response).hex(" ").upper(),
                )
                del response[0]
                expected_total_length = None
                continue

            expected_total_length = 5 + valid_frame_length + 1
            if len(response) < expected_total_length:
                break

            frame = bytes(response[:expected_total_length])
            expected_lrc = calculate_lrc(frame[:-1])
            actual_lrc = frame[-1]
            if expected_lrc != actual_lrc:
                logger.debug(
                    "Discarding auto-push candidate with LRC mismatch: "
                    "calculated 0x%02X, got 0x%02X; raw=%s",
                    expected_lrc,
                    actual_lrc,
                    format_bytes(frame),
                )
                del response[0]
                expected_total_length = None
                continue

            del response[:expected_total_length]
            logger.debug("RX auto-push %s", format_bytes(frame))
            return frame

        waiting = ser.in_waiting
        if waiting:
            chunk = ser.read(waiting)
            raw_response.extend(chunk)
            response.extend(chunk)
            logger.debug("RX chunk %s", format_bytes(chunk))

        time.sleep(0.001)

    if not raw_response:
        raise TimeoutError("No Paxini auto-push frame received")
    raise TimeoutError(
        f"Incomplete Paxini auto-push frame: got {len(raw_response)} raw bytes, "
        f"expected {expected_total_length or 'unknown'}; "
        f"raw={bytes(raw_response).hex(' ').upper()}"
    )


def parse_auto_push_frame(frame: bytes) -> bytes:
    if len(frame) < 7:
        raise ValueError(f"Auto-push frame is too short: {len(frame)} bytes")
    if frame[:2] != FRAME_HEAD_AUTO_PUSH:
        raise ValueError(
            f"Unexpected auto-push header {frame[:2].hex(' ').upper()}, "
            f"expected {FRAME_HEAD_AUTO_PUSH.hex(' ').upper()}"
        )

    valid_frame_length = int.from_bytes(frame[3:5], "little")
    expected_total_length = 5 + valid_frame_length + 1
    if len(frame) < expected_total_length:
        raise ValueError(
            f"Incomplete auto-push frame: got {len(frame)} bytes, "
            f"expected {expected_total_length}"
        )

    expected_lrc = calculate_lrc(frame[: expected_total_length - 1])
    actual_lrc = frame[expected_total_length - 1]
    if expected_lrc != actual_lrc:
        raise ValueError(
            f"Auto-push LRC mismatch: calculated 0x{expected_lrc:02X}, "
            f"got 0x{actual_lrc:02X}"
        )

    status = frame[5]
    if status != 0:
        raise RuntimeError(f"Auto-push status is 0x{status:02X}")
    return frame[6 : expected_total_length - 1]


def parse_response(frame: bytes) -> PaxiniResponse:
    if len(frame) < 15:
        raise ValueError(f"Response frame is too short: {len(frame)} bytes")
    if frame[:2] != FRAME_HEAD_RESPONSE:
        raise ValueError(
            f"Unexpected response header {frame[:2].hex(' ').upper()}, "
            f"expected {FRAME_HEAD_RESPONSE.hex(' ').upper()}"
        )

    body_length = int.from_bytes(frame[2:4], "little")
    lrc_index = 4 + body_length
    if lrc_index >= len(frame):
        raise ValueError(
            f"Incomplete response: body length {body_length}, frame length {len(frame)}"
        )

    expected_lrc = calculate_lrc(frame[:lrc_index])
    actual_lrc = frame[lrc_index]
    if expected_lrc != actual_lrc:
        raise ValueError(
            f"LRC mismatch: calculated 0x{expected_lrc:02X}, got 0x{actual_lrc:02X}"
        )

    body = frame[4:lrc_index]
    if len(body) < 10:
        raise ValueError(f"Response body is too short: {len(body)} bytes")

    returned_length = int.from_bytes(body[7:9], "little")
    data = body[10:]
    if len(data) != returned_length:
        raise ValueError(
            f"Returned data length mismatch: header says {returned_length}, got {len(data)}"
        )

    return PaxiniResponse(
        device_address=body[0],
        function=body[2],
        address=int.from_bytes(body[3:7], "little"),
        returned_length=returned_length,
        status=body[9],
        data=data,
    )


def request_read(
    ser: serial.Serial,
    device_address: int,
    address: int,
    data_length: int,
    response_timeout_s: float,
    *,
    reset_input: bool,
    max_body_length: int,
) -> PaxiniResponse:
    frame = build_read_frame(device_address, address, data_length)
    if reset_input:
        ser.reset_input_buffer()
    bytes_sent = ser.write(frame)
    if bytes_sent != len(frame):
        raise IOError(f"Serial write incomplete: sent {bytes_sent}/{len(frame)} bytes")
    logger.debug("TX %s", frame.hex(" ").upper())

    response = parse_response(
        read_response(
            ser,
            response_timeout_s,
            max_body_length=max_body_length,
        )
    )
    if response.device_address != device_address:
        raise ValueError(
            f"Response device address mismatch: expected {device_address}, "
            f"got {response.device_address}"
        )
    if response.function != FUNC_READ_APPLICATION:
        raise ValueError(f"Unexpected response function: 0x{response.function:02X}")
    if response.address != address:
        raise ValueError(
            f"Response address mismatch: expected 0x{address:08X}, got 0x{response.address:08X}"
        )
    if response.status != 0:
        raise RuntimeError(
            f"Read failed for device {device_address} with status 0x{response.status:02X}"
        )
    return response


def request_write(
    ser: serial.Serial,
    device_address: int,
    address: int,
    payload: bytes,
    response_timeout_s: float,
    *,
    reset_input: bool,
    max_body_length: int,
) -> PaxiniResponse:
    frame = build_write_frame(device_address, address, payload)
    if reset_input:
        ser.reset_input_buffer()
    bytes_sent = ser.write(frame)
    if bytes_sent != len(frame):
        raise IOError(f"Serial write incomplete: sent {bytes_sent}/{len(frame)} bytes")
    logger.debug("TX %s", frame.hex(" ").upper())

    response = parse_response(
        read_response(
            ser,
            response_timeout_s,
            max_body_length=max_body_length,
        )
    )
    if response.device_address != device_address:
        raise ValueError(
            f"Response device address mismatch: expected {device_address}, "
            f"got {response.device_address}"
        )
    if response.function != FUNC_WRITE_CONFIG:
        raise ValueError(f"Unexpected response function: 0x{response.function:02X}")
    if response.address != address:
        raise ValueError(
            f"Response address mismatch: expected 0x{address:08X}, got 0x{response.address:08X}"
        )
    if response.status != 0:
        raise RuntimeError(
            f"Write failed for device {device_address} with status 0x{response.status:02X}"
        )
    return response


def write_adapter_register(
    ser: serial.Serial,
    address: int,
    payload: bytes,
    response_timeout_s: float,
    *,
    reset_input: bool,
) -> bytes:
    frame = build_adapter_write_frame(address, payload)
    if reset_input:
        ser.reset_input_buffer()
    bytes_sent = ser.write(frame)
    if bytes_sent != len(frame):
        raise IOError(f"Serial write incomplete: sent {bytes_sent}/{len(frame)} bytes")
    logger.debug("TX adapter %s", frame.hex(" ").upper())

    response = read_adapter_response(
        ser,
        response_timeout_s,
        expected_function=0x10,
        expected_address=address,
    )
    if response[3] != 0x10:
        raise ValueError(f"Unexpected adapter response function: 0x{response[3]:02X}")
    if int.from_bytes(response[4:6], "little") != address:
        raise ValueError(
            f"Adapter response address mismatch: expected 0x{address:04X}, "
            f"got 0x{int.from_bytes(response[4:6], 'little'):04X}"
        )
    status = response[8] if len(response) > 8 else 0
    if status != 0:
        raise RuntimeError(f"Adapter write failed with status 0x{status:02X}")
    return response


def write_adapter_register_no_ack(
    ser: serial.Serial,
    address: int,
    payload: bytes,
) -> None:
    frame = build_adapter_write_frame(address, payload)
    bytes_sent = ser.write(frame)
    if bytes_sent != len(frame):
        raise IOError(f"Serial write incomplete: sent {bytes_sent}/{len(frame)} bytes")
    logger.debug("TX adapter no-ack %s", frame.hex(" ").upper())


def calibrate_stream_adapter(
    ser: serial.Serial,
    response_timeout_s: float,
    calibration_settle_s: float,
) -> None:
    logger.info("Calibrating high-speed-board stream sensors")
    write_adapter_register(
        ser,
        ADDR_CALIBRATION,
        b"\x01",
        response_timeout_s,
        reset_input=True,
    )
    time.sleep(calibration_settle_s)


def calibrate_request_digits(
    ser: serial.Serial,
    digits: list[DigitConfig],
    response_timeout_s: float,
    calibration_settle_s: float,
    max_body_length: int,
) -> None:
    logger.info("Calibrating request-mode sensors")
    for digit in digits:
        request_write(
            ser,
            digit.device_address,
            ADDR_CALIBRATION,
            b"\x01",
            response_timeout_s,
            reset_input=True,
            max_body_length=max_body_length,
        )
    time.sleep(calibration_settle_s)


def parse_force_vectors(
    data: bytes,
    expected_points: int,
    *,
    scale_n: float,
    signed_z: bool,
) -> np.ndarray:
    expected_len = expected_points * 3
    if len(data) < expected_len:
        raise ValueError(f"Expected {expected_len} force bytes, got {len(data)}")

    raw = np.frombuffer(data[:expected_len], dtype=np.uint8).reshape(expected_points, 3)
    widened = raw.astype(np.int16)
    vectors = np.empty((expected_points, 3), dtype=np.float64)
    vectors[:, 0] = np.where(widened[:, 0] <= 127, widened[:, 0], widened[:, 0] - 256)
    vectors[:, 1] = np.where(widened[:, 1] <= 127, widened[:, 1], widened[:, 1] - 256)
    if signed_z:
        vectors[:, 2] = np.where(widened[:, 2] <= 127, widened[:, 2], widened[:, 2] - 256)
    else:
        vectors[:, 2] = widened[:, 2]
    vectors *= scale_n
    return vectors


def parse_auto_push_digits(
    payload: bytes,
    digits: list[DigitConfig],
    *,
    scale_n: float,
    signed_z: bool,
) -> dict[str, DigitSample]:
    offset = 0
    timestamp_s = time.time()
    samples = {}
    for digit in digits:
        metadata_length = 6
        data_length = digit.force_points * 3
        block_length = metadata_length + data_length
        if len(payload) < offset + block_length:
            raise ValueError(
                f"Auto-push payload too short for {digit.name}: got {len(payload)} bytes, "
                f"need at least {offset + block_length}"
            )
        data = payload[offset + metadata_length : offset + block_length]
        samples[digit.name] = DigitSample(
            timestamp_s=timestamp_s,
            vectors_n=parse_force_vectors(
                data,
                digit.force_points,
                scale_n=scale_n,
                signed_z=signed_z,
            ),
        )
        offset += block_length
    return samples


def subtract_zero_offsets(
    samples: dict[str, DigitSample],
    zero_offsets: dict[str, np.ndarray],
) -> dict[str, DigitSample]:
    if not zero_offsets:
        return samples

    corrected = {}
    for name, sample in samples.items():
        offset = zero_offsets.get(name)
        vectors = sample.vectors_n
        if offset is not None and offset.shape == vectors.shape:
            vectors = vectors - offset
        corrected[name] = DigitSample(sample.timestamp_s, vectors)
    return corrected


def median_zero_offsets(samples: list[dict[str, DigitSample]]) -> dict[str, np.ndarray]:
    if not samples:
        return {}

    offsets = {}
    for name in samples[-1]:
        arrays = [sample[name].vectors_n for sample in samples if name in sample]
        if arrays:
            offsets[name] = np.median(np.stack(arrays, axis=0), axis=0)
    return offsets


def median_filter_samples(
    samples: list[dict[str, DigitSample]],
) -> dict[str, DigitSample]:
    if not samples:
        return {}

    timestamp_s = samples[-1][next(iter(samples[-1]))].timestamp_s
    filtered = {}
    for name in samples[-1]:
        arrays = [sample[name].vectors_n for sample in samples if name in sample]
        if arrays:
            filtered[name] = DigitSample(
                timestamp_s=timestamp_s,
                vectors_n=np.median(np.stack(arrays, axis=0), axis=0),
            )
    return filtered


def read_digit_vectors(
    ser: serial.Serial,
    digit: DigitConfig,
    force_address: int,
    response_timeout_s: float,
    *,
    reset_input: bool,
    scale_n: float,
    signed_z: bool,
    max_body_length: int,
    read_chunk_size: int,
) -> np.ndarray:
    expected_length = digit.force_points * 3
    chunks = []
    offset = 0

    while offset < expected_length:
        request_length = (
            expected_length - offset
            if read_chunk_size <= 0
            else min(read_chunk_size, expected_length - offset)
        )
        response = request_read(
            ser,
            device_address=digit.device_address,
            address=force_address + offset,
            data_length=request_length,
            response_timeout_s=response_timeout_s,
            reset_input=reset_input and offset == 0,
            max_body_length=max_body_length,
        )
        chunks.append(response.data)
        offset += len(response.data)
        if read_chunk_size <= 0:
            break

    data = b"".join(chunks)
    return parse_force_vectors(
        data,
        digit.force_points,
        scale_n=scale_n,
        signed_z=signed_z,
    )


def board_force_address_for_digit(
    digit: DigitConfig,
    layout: Optional[BoardLayout] = None,
) -> int:
    try:
        base_address = BOARD_DISTRIBUTED_FORCE_ADDRESSES[digit.name]
    except KeyError as exc:
        raise ValueError(
            f"No high-speed-board distributed-force address configured for digit "
            f"{digit.name!r}"
        ) from exc
    if layout is None:
        return base_address

    module_start = BOARD_DIGIT_MODULE_START.get(digit.name)
    if module_start is None:
        return base_address
    module_count = 8 if digit.name == "palm" else 4
    counts = layout.force_point_counts[module_start : module_start + module_count]
    if not counts:
        return base_address

    # The high-speed board reserves fixed submodule address windows. For
    # example, index modules are 0x1800, 0x1A00, 0x1C00, 0x1E00. Point-count
    # entries tell us which submodule window is actually populated.
    stride = BOARD_DIGIT_MODULE_STRIDE.get(digit.name, 0)
    matches = [index for index, count in enumerate(counts) if count == digit.force_points]
    if len(matches) == 1 and stride:
        resolved = base_address + matches[0] * stride
        if resolved != base_address:
            logger.info(
                "%s: using board address 0x%04X from counts %s",
                digit.name,
                resolved,
                counts,
            )
        return resolved

    nonzero = [index for index, count in enumerate(counts) if count > 0]
    if len(nonzero) == 1 and stride and counts[nonzero[0]] >= digit.force_points:
        resolved = base_address + nonzero[0] * stride
        if resolved != base_address:
            logger.info(
                "%s: using first connected board address 0x%04X from counts %s",
                digit.name,
                resolved,
                counts,
            )
        return resolved

    logger.info("%s: using board base address 0x%04X from counts %s", digit.name, base_address, counts)
    return base_address


def read_board_digit_vectors(
    ser: serial.Serial,
    digit: DigitConfig,
    response_timeout_s: float,
    *,
    reset_input: bool,
    scale_n: float,
    signed_z: bool,
    layout: Optional[BoardLayout],
    board_address: Optional[int] = None,
) -> np.ndarray:
    expected_length = digit.force_points * 3
    base_address = (
        board_address
        if board_address is not None
        else board_force_address_for_digit(digit, layout)
    )
    data = bytearray()
    offset = 0

    while offset < expected_length:
        chunk_length = min(512, expected_length - offset)
        chunk = read_adapter_register(
            ser,
            base_address + offset,
            chunk_length,
            response_timeout_s,
            reset_input=reset_input and offset == 0,
        )
        if not chunk:
            raise ValueError(
                f"No high-speed-board data returned for {digit.name} at "
                f"0x{base_address + offset:04X}"
            )
        data.extend(chunk)
        offset += len(chunk)

    return parse_force_vectors(
        bytes(data[:expected_length]),
        digit.force_points,
        scale_n=scale_n,
        signed_z=signed_z,
    )


def choose_port() -> str:
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        raise RuntimeError("No serial ports found. Check USB cable, power, and adapter.")

    print("\nAvailable serial ports:")
    for index, port in enumerate(ports, 1):
        print(f"  {index}. {port.device} - {port.description}")

    try:
        choice = int(input(f"\nSelect serial port (1-{len(ports)}): "))
        return ports[choice - 1].device
    except (ValueError, IndexError):
        selected = ports[0].device
        print(f"Invalid selection. Using first port: {selected}")
        return selected


class HandPaxiniReader:
    def __init__(
        self,
        port: str,
        baudrate: int,
        digits: list[DigitConfig],
        force_address: int,
        read_interval_s: float,
        response_timeout_s: float,
        serial_settle_s: float,
        dtr: bool,
        rts: bool,
        reset_input: bool,
        scale_n: float,
        signed_z: bool,
        max_body_length: int,
        read_chunk_size: int,
        mode: str,
        calibrate: bool,
        calibration_settle_s: float,
        calibration_samples: int,
        software_zero: bool,
        discard_startup_frames: int,
        median_window: int,
        publish_rate_hz: float,
        probe_adapter: bool,
    ) -> None:
        self.port = port
        self.baudrate = baudrate
        self.digits = digits
        self.force_address = force_address
        self.read_interval_s = read_interval_s
        self.response_timeout_s = response_timeout_s
        self.serial_settle_s = max(0.0, serial_settle_s)
        self.dtr = dtr
        self.rts = rts
        self.reset_input = reset_input
        self.scale_n = scale_n
        self.signed_z = signed_z
        self.max_body_length = max_body_length
        self.read_chunk_size = read_chunk_size
        self.mode = mode
        self.calibrate = calibrate
        self.calibration_settle_s = calibration_settle_s
        self.calibration_samples = calibration_samples
        self.software_zero = software_zero
        self.discard_startup_frames = discard_startup_frames
        self.median_window = max(1, median_window)
        self.publish_rate_hz = max(0.1, publish_rate_hz)
        self.probe_adapter = probe_adapter
        self.zero_offsets: dict[str, np.ndarray] = {}

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._read_thread: Optional[threading.Thread] = None
        self._publish_thread: Optional[threading.Thread] = None
        self._serial: Optional[serial.Serial] = None
        self._auto_push_buffer = bytearray()
        self._raw_latest: Optional[HandSample] = None
        self._raw_sequence = 0
        self._published_sequence = 0
        self._stats_start_s = time.monotonic()
        self.acquired_count = 0
        self.published_count = 0
        self.latest: Optional[HandSample] = None
        self.error: Optional[str] = None
        self.history = deque(maxlen=300)

    def start(self) -> None:
        self._stats_start_s = time.monotonic()
        self._publish_thread = threading.Thread(
            target=self._publish_loop,
            name="paxini-publisher",
            daemon=True,
        )
        self._read_thread = threading.Thread(
            target=self._run,
            name="paxini-reader",
            daemon=True,
        )
        self._publish_thread.start()
        self._read_thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._read_thread:
            self._read_thread.join(timeout=2.0)
        if self._publish_thread:
            self._publish_thread.join(timeout=2.0)
        if self._serial and self._serial.is_open:
            self._serial.close()

    def snapshot(self) -> tuple[Optional[HandSample], Optional[str], list[HandSample], dict[str, float]]:
        with self._lock:
            elapsed_s = max(time.monotonic() - self._stats_start_s, 1e-9)
            stats = {
                "rx_hz": self.acquired_count / elapsed_s,
                "publish_hz": self.published_count / elapsed_s,
            }
            return self.latest, self.error, list(self.history), stats

    def _record_sample(self, sample: HandSample) -> None:
        with self._lock:
            self._raw_latest = sample
            self._raw_sequence += 1
            self.acquired_count += 1
            self.error = None

    def _publish_loop(self) -> None:
        interval_s = 1.0 / self.publish_rate_hz
        next_publish_s = time.monotonic()
        while not self._stop.is_set():
            with self._lock:
                if (
                    self._raw_latest is not None
                    and self._raw_sequence != self._published_sequence
                ):
                    self.latest = self._raw_latest
                    self._published_sequence = self._raw_sequence
                    self.published_count += 1
                    self.history.append(self.latest)

            next_publish_s += interval_s
            sleep_s = next_publish_s - time.monotonic()
            if sleep_s <= 0:
                next_publish_s = time.monotonic()
                continue
            self._stop.wait(sleep_s)

    def _run(self) -> None:
        try:
            self._serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
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
            self._serial.dtr = self.dtr
            self._serial.rts = self.rts
            time.sleep(self.serial_settle_s)
            self._serial.reset_input_buffer()
            self._serial.reset_output_buffer()
            logger.info(
                "Opened %s at %s baud with DTR=%s RTS=%s",
                self.port,
                self.baudrate,
                self.dtr,
                self.rts,
            )

            if self.mode == "stream":
                if self.calibrate:
                    logger.warning(
                        "--calibrate is only supported in native-request mode. "
                        "High-speed-board stream mode has no documented sensor "
                        "calibration command; use --software-zero explicitly for "
                        "local baseline subtraction."
                    )

                if self.probe_adapter:
                    read_adapter_version(self._serial, self.response_timeout_s)
                    write_adapter_register(
                        self._serial,
                        BOARD_AUTO_POST_BACK_DATA_TYPE_ADDRESS,
                        bytes([AUTO_PUSH_DATA_TYPE_RESULTANT_AND_DISTRIBUTED]),
                        self.response_timeout_s,
                        reset_input=True,
                    )
                    logger.info("Configured auto-push stream for resultant + distributed force")
                else:
                    self._serial.write(DISABLE_AUTO_PUSH_FRAME)
                    time.sleep(0.2)
                    drain_serial_input(self._serial)
                    write_adapter_register_no_ack(
                        self._serial,
                        BOARD_AUTO_POST_BACK_DATA_TYPE_ADDRESS,
                        bytes([AUTO_PUSH_DATA_TYPE_RESULTANT_AND_DISTRIBUTED]),
                    )
                    time.sleep(0.05)
                    drain_serial_input(self._serial)
                    logger.info(
                        "Sent auto-push stream config for resultant + distributed force"
                    )
                self._serial.reset_input_buffer()
                self._auto_push_buffer.clear()
                bytes_sent = self._serial.write(ENABLE_AUTO_PUSH_FRAME)
                if bytes_sent != len(ENABLE_AUTO_PUSH_FRAME):
                    raise IOError(
                        f"Serial write incomplete: sent {bytes_sent}/"
                        f"{len(ENABLE_AUTO_PUSH_FRAME)} bytes"
                )
                logger.debug("TX enable auto-push %s", ENABLE_AUTO_PUSH_FRAME.hex(" ").upper())

                for _ in range(max(0, self.discard_startup_frames)):
                    read_auto_push_frame(
                        self._serial,
                        self.response_timeout_s,
                        max_body_length=self.max_body_length,
                        buffer=self._auto_push_buffer,
                    )
                if self.discard_startup_frames > 0:
                    logger.info("Discarded %s startup stream frames", self.discard_startup_frames)

                if self.software_zero:
                    logger.info(
                        "Collecting %s unloaded stream frames for software zero baseline",
                        self.calibration_samples,
                    )
                    baseline_samples = []
                    for _ in range(max(1, self.calibration_samples)):
                        frame = read_auto_push_frame(
                            self._serial,
                            self.response_timeout_s,
                            max_body_length=self.max_body_length,
                            buffer=self._auto_push_buffer,
                        )
                        baseline_samples.append(
                            parse_auto_push_digits(
                                parse_auto_push_frame(frame),
                                self.digits,
                                scale_n=self.scale_n,
                                signed_z=self.signed_z,
                            )
                        )
                    self.zero_offsets = median_zero_offsets(baseline_samples)
                    time.sleep(self.calibration_settle_s)
                    logger.info("Software zero baseline captured")

                filter_buffer = deque(maxlen=self.median_window)
                while not self._stop.is_set():
                    frame = read_auto_push_frame(
                        self._serial,
                        self.response_timeout_s,
                        max_body_length=self.max_body_length,
                        buffer=self._auto_push_buffer,
                    )
                    digit_samples = parse_auto_push_digits(
                        parse_auto_push_frame(frame),
                        self.digits,
                        scale_n=self.scale_n,
                        signed_z=self.signed_z,
                    )
                    digit_samples = subtract_zero_offsets(digit_samples, self.zero_offsets)
                    filter_buffer.append(digit_samples)
                    if self.median_window > 1 and len(filter_buffer) >= self.median_window:
                        digit_samples = median_filter_samples(list(filter_buffer))
                    self._record_sample(
                        HandSample(timestamp_s=time.time(), digits=digit_samples)
                    )
            elif self.mode == "board-request":
                if self.calibrate:
                    logger.warning(
                        "--calibrate is only supported in native-request mode. "
                        "High-speed-board request mode has no documented sensor "
                        "calibration command."
                    )

                read_adapter_version(self._serial, self.response_timeout_s)
                board_layout = read_board_layout(self._serial, self.response_timeout_s)
                board_addresses = {
                    digit.name: board_force_address_for_digit(digit, board_layout)
                    for digit in self.digits
                }
                filter_buffer = deque(maxlen=self.median_window)
                while not self._stop.is_set():
                    timestamp_s = time.time()
                    digit_samples = {}
                    for digit in self.digits:
                        vectors = read_board_digit_vectors(
                            self._serial,
                            digit,
                            self.response_timeout_s,
                            reset_input=self.reset_input,
                            scale_n=self.scale_n,
                            signed_z=self.signed_z,
                            layout=board_layout,
                            board_address=board_addresses[digit.name],
                        )
                        digit_samples[digit.name] = DigitSample(timestamp_s, vectors)

                    digit_samples = subtract_zero_offsets(digit_samples, self.zero_offsets)
                    filter_buffer.append(digit_samples)
                    if self.median_window > 1 and len(filter_buffer) >= self.median_window:
                        digit_samples = median_filter_samples(list(filter_buffer))

                    self._record_sample(
                        HandSample(timestamp_s=timestamp_s, digits=digit_samples)
                    )
                    time.sleep(self.read_interval_s)
            elif self.mode == "native-request":
                if self.calibrate:
                    calibrate_request_digits(
                        self._serial,
                        self.digits,
                        self.response_timeout_s,
                        self.calibration_settle_s,
                        self.max_body_length,
                    )

                while not self._stop.is_set():
                    timestamp_s = time.time()
                    digit_samples = {}
                    for digit in self.digits:
                        vectors = read_digit_vectors(
                            self._serial,
                            digit,
                            self.force_address,
                            self.response_timeout_s,
                            reset_input=self.reset_input,
                            scale_n=self.scale_n,
                            signed_z=self.signed_z,
                            max_body_length=self.max_body_length,
                            read_chunk_size=self.read_chunk_size,
                        )
                        digit_samples[digit.name] = DigitSample(timestamp_s, vectors)

                    self._record_sample(
                        HandSample(timestamp_s=timestamp_s, digits=digit_samples)
                    )
                    time.sleep(self.read_interval_s)
            else:
                raise ValueError(f"Unsupported mode {self.mode!r}")
        except Exception as exc:
            logger.exception("Paxini reader stopped")
            with self._lock:
                self.error = str(exc)
        finally:
            if self.mode == "stream" and self._serial and self._serial.is_open:
                try:
                    self._serial.write(DISABLE_AUTO_PUSH_FRAME)
                    logger.debug(
                        "TX disable auto-push %s",
                        DISABLE_AUTO_PUSH_FRAME.hex(" ").upper(),
                    )
                except serial.SerialException:
                    pass
            if self._serial and self._serial.is_open:
                self._serial.close()


def grid_positions(point_count: int) -> tuple[np.ndarray, np.ndarray]:
    rows = max(1, int(math.floor(math.sqrt(point_count))))
    cols = int(math.ceil(point_count / rows))
    x = np.arange(point_count) % cols
    y = np.arange(point_count) // cols
    return x.astype(float), y.astype(float)


def empty_figure(title: str, message: str = "Waiting for tactile data..."):
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 16},
    )
    fig.update_layout(
        title=title,
        template="plotly_white",
        margin={"l": 35, "r": 20, "t": 45, "b": 35},
    )
    return fig


def component_values(vectors: np.ndarray, component: str) -> tuple[np.ndarray, np.ndarray]:
    component_index = {"Fx": 0, "Fy": 1, "Fz": 2, "|F|": 3}[component]
    magnitudes = np.linalg.norm(vectors, axis=1)
    values = magnitudes if component_index == 3 else vectors[:, component_index]
    return values, magnitudes


def make_digit_figure(
    name: str,
    sample: Optional[DigitSample],
    component: str,
    coordinates_mm: Optional[np.ndarray],
    plot_mode: str,
    show_vectors: bool,
):
    if sample is None:
        return empty_figure(f"{name.title()} Force Vectors")

    vectors = sample.vectors_n
    values, magnitudes = component_values(vectors, component)

    if (
        coordinates_mm is not None
        and len(coordinates_mm) == len(vectors)
        and plot_mode == "2d"
    ):
        point_ids = np.arange(1, len(vectors) + 1)
        fx = vectors[:, 0]
        fy = vectors[:, 1]
        fz = vectors[:, 2]
        fig = go.Figure()
        fig.add_trace(
            go.Scattergl(
                x=coordinates_mm[:, 0],
                y=coordinates_mm[:, 1],
                mode="markers",
                marker={
                    "size": 11 if len(vectors) > 80 else 13,
                    "color": values,
                    "colorscale": "Viridis",
                    "colorbar": {"title": f"{component} (N)"},
                    "line": {"color": "#111827", "width": 1},
                },
                customdata=np.column_stack(
                    [
                        point_ids,
                        coordinates_mm[:, 0],
                        coordinates_mm[:, 1],
                        coordinates_mm[:, 2],
                        fx,
                        fy,
                        fz,
                        magnitudes,
                    ]
                ),
                hovertemplate=(
                    "point=%{customdata[0]:.0f}<br>"
                    "X width=%{customdata[1]:.2f} mm<br>"
                    "Y length=%{customdata[2]:.2f} mm<br>"
                    "Z height=%{customdata[3]:.2f} mm<br>"
                    "Fx=%{customdata[4]:.2f} N<br>"
                    "Fy=%{customdata[5]:.2f} N<br>"
                    "Fz=%{customdata[6]:.2f} N<br>"
                    "|F|=%{customdata[7]:.2f} N<extra></extra>"
                ),
                name="force points",
            )
        )
        if show_vectors:
            coordinate_span = max(
                float(np.ptp(coordinates_mm[:, 0])),
                float(np.ptp(coordinates_mm[:, 1])),
                1.0,
            )
            max_xy = max(float(np.max(np.hypot(fx, fy))), 1.0)
            scale = 0.12 * coordinate_span / max_xy
            line_x = []
            line_y = []
            for point, vector in zip(coordinates_mm, vectors[:, :2] * scale):
                line_x.extend([point[0], point[0] + vector[0], None])
                line_y.extend([point[1], point[1] + vector[1], None])
            fig.add_trace(
                go.Scattergl(
                    x=line_x,
                    y=line_y,
                    mode="lines",
                    line={"color": "#D62728", "width": 2},
                    hoverinfo="skip",
                    name="Fx/Fy direction",
                )
            )
        fig.update_layout(
            title=f"{name.title()} {len(vectors)} Force Points",
            template="plotly_white",
            xaxis_title="X width (mm)",
            yaxis_title="Y length (mm)",
            margin={"l": 45, "r": 10, "t": 45, "b": 40},
            showlegend=False,
            uirevision=name,
            transition={"duration": 0},
        )
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        return fig

    if (
        coordinates_mm is not None
        and len(coordinates_mm) == len(vectors)
        and plot_mode == "3d"
    ):
        point_ids = np.arange(1, len(vectors) + 1)
        fx = vectors[:, 0]
        fy = vectors[:, 1]
        fz = vectors[:, 2]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter3d(
                x=coordinates_mm[:, 0],
                y=coordinates_mm[:, 1],
                z=coordinates_mm[:, 2],
                mode="markers",
                marker={
                    "size": 4 if len(vectors) > 80 else 5,
                    "color": values,
                    "colorscale": "Viridis",
                    "colorbar": {"title": f"{component} (N)"},
                    "line": {"color": "#111827", "width": 1},
                },
                customdata=np.column_stack(
                    [
                        point_ids,
                        coordinates_mm[:, 0],
                        coordinates_mm[:, 1],
                        coordinates_mm[:, 2],
                        fx,
                        fy,
                        fz,
                        magnitudes,
                    ]
                ),
                hovertemplate=(
                    "point=%{customdata[0]:.0f}<br>"
                    "X width=%{customdata[1]:.2f} mm<br>"
                    "Y length=%{customdata[2]:.2f} mm<br>"
                    "Z height=%{customdata[3]:.2f} mm<br>"
                    "Fx=%{customdata[4]:.2f} N<br>"
                    "Fy=%{customdata[5]:.2f} N<br>"
                    "Fz=%{customdata[6]:.2f} N<br>"
                    "|F|=%{customdata[7]:.2f} N<extra></extra>"
                ),
                name="force points",
            )
        )
        if show_vectors:
            coordinate_span = max(
                float(np.ptp(coordinates_mm[:, 0])),
                float(np.ptp(coordinates_mm[:, 1])),
                float(np.ptp(coordinates_mm[:, 2])),
                1.0,
            )
            max_force = max(float(np.max(magnitudes)), 1.0)
            vector_scale = 0.16 * coordinate_span / max_force
            line_x = []
            line_y = []
            line_z = []
            for point, vector in zip(coordinates_mm, vectors * vector_scale):
                line_x.extend([point[0], point[0] + vector[0], None])
                line_y.extend([point[1], point[1] + vector[1], None])
                line_z.extend([point[2], point[2] + vector[2], None])
            fig.add_trace(
                go.Scatter3d(
                    x=line_x,
                    y=line_y,
                    z=line_z,
                    mode="lines",
                    line={"color": "#D62728", "width": 3},
                    hoverinfo="skip",
                    name="force direction",
                )
            )
        fig.update_layout(
            title=f"{name.title()} {len(vectors)} Force Points",
            template="plotly_white",
            margin={"l": 0, "r": 0, "t": 45, "b": 0},
            showlegend=False,
            scene={
                "xaxis_title": "X width (mm)",
                "yaxis_title": "Y length (mm)",
                "zaxis_title": "Z height (mm)",
                "aspectmode": "data",
                "camera": {"eye": {"x": 1.5, "y": -1.8, "z": 1.2}},
            },
            uirevision=name,
            transition={"duration": 0},
        )
        return fig

    x, y = grid_positions(len(vectors))

    fx = vectors[:, 0]
    fy = vectors[:, 1]
    max_xy = max(float(np.max(np.hypot(fx, fy))), 1.0)
    scale = 0.35 / max_xy
    line_x = []
    line_y = []
    for x0, y0, dx, dy in zip(x, y, fx * scale, fy * scale):
        line_x.extend([x0, x0 + dx, None])
        line_y.extend([y0, y0 - dy, None])

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=x,
            y=y,
            mode="markers",
            marker={
                "size": 12,
                "color": values,
                "colorscale": "Viridis",
                "colorbar": {"title": f"{component} (N)"},
                "line": {"color": "#111827", "width": 1},
            },
            customdata=np.column_stack(
                [
                    np.arange(1, len(vectors) + 1),
                    vectors[:, 0],
                    vectors[:, 1],
                    vectors[:, 2],
                    magnitudes,
                ]
            ),
            hovertemplate=(
                "point=%{customdata[0]:.0f}<br>"
                "Fx=%{customdata[1]:.2f} N<br>"
                "Fy=%{customdata[2]:.2f} N<br>"
                "Fz=%{customdata[3]:.2f} N<br>"
                "|F|=%{customdata[4]:.2f} N<extra></extra>"
            ),
            name="force points",
        )
    )
    if show_vectors:
        fig.add_trace(
            go.Scattergl(
                x=line_x,
                y=line_y,
                mode="lines",
                line={"color": "#D62728", "width": 2},
                hoverinfo="skip",
                name="Fx/Fy direction",
            )
        )
    fig.update_layout(
        title=f"{name.title()} {len(vectors)} Force Vectors",
        template="plotly_white",
        xaxis_title="point grid column",
        yaxis_title="point grid row",
        yaxis_autorange="reversed",
        margin={"l": 45, "r": 20, "t": 50, "b": 45},
        showlegend=False,
        uirevision=name,
        transition={"duration": 0},
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def make_history_figure(history: list[HandSample], digit_order: list[str]):
    if not history:
        return empty_figure("Total Force History")

    t0 = history[0].timestamp_s
    times = [sample.timestamp_s - t0 for sample in history]

    fig = go.Figure()
    digit_names = [name for name in digit_order if name in history[-1].digits]
    digit_names.extend(name for name in history[-1].digits if name not in digit_names)
    for digit_name in digit_names:
        values = []
        for sample in history:
            digit = sample.digits.get(digit_name)
            if digit is None:
                values.append(np.nan)
            else:
                values.append(float(np.sum(np.linalg.norm(digit.vectors_n, axis=1))))
        fig.add_trace(go.Scattergl(x=times, y=values, mode="lines", name=digit_name))

    fig.update_layout(
        title="Total Distributed Force History",
        template="plotly_white",
        xaxis_title="time (s)",
        yaxis_title="sum |F| (N)",
        margin={"l": 45, "r": 20, "t": 50, "b": 45},
        uirevision="history",
        transition={"duration": 0},
    )
    return fig


def make_app(
    reader: HandPaxiniReader,
    update_ms: int,
    digit_coordinates: dict[str, np.ndarray],
    plot_mode: str,
    show_vectors: bool,
):
    app = Dash(__name__)
    digit_graphs = [
        dcc.Graph(id=f"digit-{digit.name}", style={"height": "44vh"})
        for digit in reader.digits
    ]
    app.layout = html.Div(
        [
            html.Div(
                [
                    html.H2("Paxini Hand High-Speed Board"),
                    html.Div(id="status"),
                ],
                style={"padding": "12px 16px"},
            ),
            html.Div(
                [
                    dcc.Dropdown(
                        id="component",
                        options=[
                            {"label": name, "value": name}
                            for name in ("Fz", "|F|", "Fx", "Fy")
                        ],
                        value="Fz",
                        clearable=False,
                        style={"width": "160px"},
                    ),
                ],
                style={"padding": "0 16px 8px 16px"},
            ),
            html.Div(
                digit_graphs,
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(auto-fit, minmax(360px, 1fr))",
                    "gap": "8px",
                },
            ),
            dcc.Graph(id="history", style={"height": "28vh"}),
            dcc.Interval(id="tick", interval=update_ms, n_intervals=0),
        ],
        style={"fontFamily": "Arial, sans-serif"},
    )

    outputs = [Output("status", "children")]
    outputs.extend(Output(f"digit-{digit.name}", "figure") for digit in reader.digits)
    outputs.append(Output("history", "figure"))

    @app.callback(
        outputs,
        Input("tick", "n_intervals"),
        Input("component", "value"),
    )
    def update(_n_intervals: int, component: str):
        sample, error, history, stats = reader.snapshot()
        if error:
            status = f"Error: {error}"
        elif sample:
            parts = []
            for digit in reader.digits:
                digit_sample = sample.digits.get(digit.name)
                point_count = 0 if digit_sample is None else len(digit_sample.vectors_n)
                geometry = "coords" if digit.name in digit_coordinates else "grid"
                parts.append(
                    f"{digit.name}=addr{digit.device_address}/{point_count}pts/{geometry}"
                )
            status = (
                f"mode={reader.mode} port={reader.port} baud={reader.baudrate} "
                f"zero={'on' if reader.zero_offsets else 'off'} "
                f"hwcal={'on' if reader.calibrate and reader.mode == 'native-request' else 'off'} "
                f"median={reader.median_window} "
                f"rx={stats['rx_hz']:.1f}Hz pub={stats['publish_hz']:.1f}Hz "
                f"plot={plot_mode}{'+vec' if show_vectors else ''} "
                f"force_address=0x{reader.force_address:08X} "
                + " ".join(parts)
            )
        else:
            status = "Waiting for sensor data..."

        figures = []
        for digit in reader.digits:
            digit_sample = sample.digits.get(digit.name) if sample else None
            figures.append(
                make_digit_figure(
                    digit.name,
                    digit_sample,
                    component,
                    digit_coordinates.get(digit.name),
                    plot_mode,
                    show_vectors,
                )
            )
        return [status, *figures, make_history_figure(history, [d.name for d in reader.digits])]

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default=None, help="Serial port, e.g. /dev/ttyUSB0")
    parser.add_argument("--baudrate", type=int, default=BAUDRATE)
    parser.add_argument(
        "--mode",
        choices=("stream", "request", "board-request", "native-request"),
        default="stream",
        help=(
            "stream uses high-speed-board AA56 auto-push frames; request/"
            "board-request uses high-speed-board AA55 register reads; "
            "native-request uses direct Paxini sensor UART frames."
        ),
    )
    parser.add_argument("--thumb-address", type=int, default=THUMB_DEVICE_ADDRESS)
    parser.add_argument("--index-address", type=int, default=INDEX_DEVICE_ADDRESS)
    parser.add_argument("--middle-address", type=int, default=MIDDLE_DEVICE_ADDRESS)
    parser.add_argument("--ring-address", type=int, default=RING_DEVICE_ADDRESS)
    parser.add_argument("--thumb-points", type=int, default=THUMB_FORCE_POINTS)
    parser.add_argument("--index-points", type=int, default=INDEX_FORCE_POINTS)
    parser.add_argument("--middle-points", type=int, default=MIDDLE_FORCE_POINTS)
    parser.add_argument("--ring-points", type=int, default=RING_FORCE_POINTS)
    parser.add_argument(
        "--thumb-coords",
        type=parse_path,
        default=THUMB_COORDS_PATH,
        help="CSV with thumb force-point X/Y/Z coordinates.",
    )
    parser.add_argument(
        "--index-coords",
        type=parse_path,
        default=FINGER_COORDS_PATH,
        help="CSV with index force-point X/Y/Z coordinates.",
    )
    parser.add_argument(
        "--middle-coords",
        type=parse_path,
        default=FINGER_COORDS_PATH,
        help="CSV with middle force-point X/Y/Z coordinates.",
    )
    parser.add_argument(
        "--ring-coords",
        type=parse_path,
        default=FINGER_COORDS_PATH,
        help="CSV with ring force-point X/Y/Z coordinates.",
    )
    parser.add_argument(
        "--force-address",
        type=parse_int,
        default=ADDR_DISTRIBUTED_FORCE,
        help=(
            "Native-request distributed-force start address. Accepts decimal "
            "or hex, e.g. 0x040E. High-speed-board request mode uses board "
            "register addresses from the communication-box manual."
        ),
    )
    parser.add_argument("--scale", type=float, default=FORCE_SCALE_N)
    parser.add_argument(
        "--signed-z",
        action="store_true",
        help="Parse Fz as signed int8. Default keeps Fz unsigned, matching existing examples.",
    )
    parser.add_argument("--read-interval", type=float, default=DEFAULT_READ_INTERVAL_S)
    parser.add_argument("--response-timeout", type=float, default=1.0)
    parser.add_argument(
        "--serial-settle",
        type=float,
        default=DEFAULT_SERIAL_SETTLE_S,
        help="Seconds to wait after opening the USB serial port before probing the adapter.",
    )
    parser.add_argument(
        "--dtr",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Set USB serial DTR after opening. Paxini adapter usually responds with --no-dtr.",
    )
    parser.add_argument(
        "--rts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Set USB serial RTS after opening. Paxini adapter usually responds with --rts.",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help=(
            "Calibrate unloaded sensors using native Paxini hardware calibration. "
            "Only supported with --mode native-request."
        ),
    )
    parser.add_argument(
        "--skip-calibration",
        action="store_true",
        help="Compatibility flag; calibration is skipped unless --calibrate is set.",
    )
    parser.add_argument(
        "--calibration-settle",
        type=float,
        default=0.2,
        help="Seconds to wait after the calibration command before reading.",
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=DEFAULT_CALIBRATION_SAMPLES,
        help="Stream-mode unloaded frames to median for --software-zero.",
    )
    parser.add_argument(
        "--software-zero",
        action="store_true",
        help="Capture and subtract a stream-mode software zero baseline after startup.",
    )
    parser.add_argument(
        "--discard-startup-frames",
        type=int,
        default=DEFAULT_DISCARD_STARTUP_FRAMES,
        help="AA56 frames to discard after enabling stream before plotting/calibration.",
    )
    parser.add_argument(
        "--median-window",
        type=int,
        default=DEFAULT_MEDIAN_WINDOW,
        help="Rolling median window for stream samples. Use 1 to disable.",
    )
    parser.add_argument(
        "--publish-rate-hz",
        type=float,
        default=DEFAULT_PUBLISH_RATE_HZ,
        help=(
            "Rate for publishing the latest acquired sample to the web layer. "
            "Sensor acquisition still runs as fast as frames arrive."
        ),
    )
    parser.add_argument(
        "--max-body-length",
        type=int,
        default=DEFAULT_MAX_RESPONSE_BODY_LENGTH,
        help="Largest plausible response body before the reader discards and resyncs.",
    )
    parser.add_argument(
        "--read-chunk-size",
        type=int,
        default=DEFAULT_READ_CHUNK_SIZE,
        help="Read distributed force bytes in chunks. Use 0 to request each digit in one frame.",
    )
    parser.add_argument(
        "--update-ms",
        type=int,
        default=DEFAULT_UPDATE_MS,
        help="Dash/browser refresh interval in milliseconds.",
    )
    parser.add_argument(
        "--plot-mode",
        choices=("2d", "3d"),
        default="2d",
        help="2d uses the physical X/Y coordinate projection; 3d shows X/Y/Z points.",
    )
    parser.add_argument(
        "--show-vectors",
        action="store_true",
        help="Draw force direction line segments. Disabled by default for smoother plotting.",
    )
    parser.add_argument(
        "--probe-adapter",
        action="store_true",
        help=(
            "In stream mode, read the adapter version and wait for config-write ACKs "
            "before enabling AA56 streaming. Disabled by default because some CDC "
            "startup states deliver delayed AA55 responses."
        ),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--web-port", type=int, default=8050)
    parser.add_argument(
        "--keep-input-buffer",
        action="store_true",
        help="Do not clear stale serial input before each request.",
    )
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )
    logger.setLevel(logging.DEBUG if args.debug else logging.INFO)
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    load_dash()

    digits = [
        DigitConfig("thumb", args.thumb_address, args.thumb_points, args.thumb_coords),
        DigitConfig("index", args.index_address, args.index_points, args.index_coords),
        DigitConfig("middle", args.middle_address, args.middle_points, args.middle_coords),
        DigitConfig("ring", args.ring_address, args.ring_points, args.ring_coords),
    ]
    digit_coordinates = load_digit_coordinates(digits)
    logger.info(
        "Loaded tactile coordinate maps: %s",
        ", ".join(
            f"{name}={len(coordinates)}pts"
            for name, coordinates in digit_coordinates.items()
        )
        or "none",
    )
    port = args.port or choose_port()
    mode = "board-request" if args.mode == "request" else args.mode
    reader = HandPaxiniReader(
        port=port,
        baudrate=args.baudrate,
        digits=digits,
        force_address=args.force_address,
        read_interval_s=args.read_interval,
        response_timeout_s=args.response_timeout,
        serial_settle_s=args.serial_settle,
        dtr=args.dtr,
        rts=args.rts,
        reset_input=not args.keep_input_buffer,
        scale_n=args.scale,
        signed_z=args.signed_z,
        max_body_length=args.max_body_length,
        read_chunk_size=args.read_chunk_size,
        mode=mode,
        calibrate=args.calibrate and not args.skip_calibration,
        calibration_settle_s=args.calibration_settle,
        calibration_samples=args.calibration_samples,
        software_zero=args.software_zero,
        discard_startup_frames=args.discard_startup_frames,
        median_window=args.median_window,
        publish_rate_hz=args.publish_rate_hz,
        probe_adapter=args.probe_adapter,
    )

    reader.start()
    app = make_app(
        reader,
        args.update_ms,
        digit_coordinates,
        args.plot_mode,
        args.show_vectors,
    )
    try:
        print(f"Opening Dash viewer at http://{args.host}:{args.web_port}")
        app.run(host=args.host, port=args.web_port, debug=False)
    finally:
        reader.stop()


if __name__ == "__main__":
    main()
