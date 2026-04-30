"""Read one Paxini PX-6AX GEN3 sensor through the USB/serial adapter.

This example talks to a single sensor using Paxini's native UART-style frame:

    55 AA <length:2> <device> 00 <function> <address:4> <length:2> [data] <lrc>

The Paxini docs describe the UART device address as ``module number + 1``. If
the sensor module ID is set to 2 in Paxini's tools, this script uses device
address 3 on the wire.
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from typing import Optional

import serial
import serial.tools.list_ports


BAUDRATE = 921_600
DEFAULT_SENSOR_ID = 2
DEFAULT_FORCE_POINTS = 52  # DP-S2015-Elite has 52 triaxial force points.

FRAME_HEAD_REQUEST = b"\x55\xAA"
FRAME_HEAD_RESPONSE = b"\xAA\x55"
RESERVED = 0x00

FUNC_WRITE_CONFIG = 0x79
FUNC_READ_APPLICATION = 0xFB  # 0x80 | 0x7B

ADDR_CALIBRATION = 0x0003
ADDR_RESULTANT_FORCE = 0x03F0  # 1008: Fx, Fy, Fz resultant force.
ADDR_DISTRIBUTED_FORCE = 0x040E  # 1038: first distributed force point.

FORCE_SCALE_N = 0.1


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PaxiniResponse:
    """Parsed response frame from the single-sensor USB/serial adapter."""

    device_address: int
    function: int
    address: int
    declared_length: int
    status: int
    data: bytes


def calculate_lrc(data: bytes) -> int:
    """Return the Paxini LRC checksum for all bytes before the checksum."""

    checksum = sum(data) & 0xFF
    return ((~checksum) + 1) & 0xFF


def signed_u8(value: int) -> int:
    """Interpret one byte as signed int8."""

    return value if value <= 127 else value - 256


def device_address_from_sensor_id(sensor_id: int) -> int:
    """Paxini UART device address is module/sensor ID plus one."""

    if sensor_id < 0 or sensor_id > 254:
        raise ValueError("sensor_id must be in [0, 254]")
    return sensor_id + 1


def build_frame(
    device_address: int,
    function: int,
    address: int,
    data_length: int,
    payload: bytes = b"",
) -> bytes:
    """Build a request frame for read or write operations."""

    body = (
        bytes([device_address, RESERVED, function])
        + address.to_bytes(4, "little")
        + data_length.to_bytes(2, "little")
        + payload
    )
    frame = FRAME_HEAD_REQUEST + len(body).to_bytes(2, "little") + body
    return frame + bytes([calculate_lrc(frame)])


def build_calibration_frame(device_address: int) -> bytes:
    """Build the one-byte calibration write command."""

    return build_frame(
        device_address=device_address,
        function=FUNC_WRITE_CONFIG,
        address=ADDR_CALIBRATION,
        data_length=1,
        payload=b"\x01",
    )


def build_read_frame(device_address: int, address: int, data_length: int) -> bytes:
    """Build an application-area read command."""

    return build_frame(
        device_address=device_address,
        function=FUNC_READ_APPLICATION,
        address=address,
        data_length=data_length,
    )


def write_all(ser: serial.Serial, frame: bytes) -> None:
    bytes_sent = ser.write(frame)
    if bytes_sent != len(frame):
        raise IOError(f"Serial write incomplete: sent {bytes_sent}/{len(frame)} bytes")
    logger.debug("Sent frame: %s", frame.hex(" ").upper())


def read_response(ser: serial.Serial, timeout_s: float = 1.5) -> bytes:
    """Read one complete response frame using its declared body length."""

    response = b""
    deadline = time.monotonic() + timeout_s
    expected_total_length: Optional[int] = None

    while time.monotonic() < deadline:
        waiting = ser.in_waiting
        if waiting:
            response += ser.read(waiting)

            head_index = response.find(FRAME_HEAD_RESPONSE)
            if head_index > 0:
                response = response[head_index:]

            if len(response) >= 4 and expected_total_length is None:
                body_length = int.from_bytes(response[2:4], "little")
                expected_total_length = 4 + body_length + 1

            if expected_total_length is not None and len(response) >= expected_total_length:
                frame = response[:expected_total_length]
                logger.debug("Received frame: %s", frame.hex(" ").upper())
                return frame

        time.sleep(0.001)

    if not response:
        raise TimeoutError("No response received from Paxini sensor")
    raise TimeoutError(
        f"Incomplete Paxini response: got {len(response)} bytes, "
        f"expected {expected_total_length or 'unknown'}"
    )


def parse_response(frame: bytes) -> PaxiniResponse:
    """Parse and validate a Paxini response frame."""

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

    declared_length = int.from_bytes(body[7:9], "little")
    data = body[10:]
    return PaxiniResponse(
        device_address=body[0],
        function=body[2],
        address=int.from_bytes(body[3:7], "little"),
        declared_length=declared_length,
        status=body[9],
        data=data,
    )


def request(ser: serial.Serial, frame: bytes, response_timeout_s: float = 1.5) -> PaxiniResponse:
    write_all(ser, frame)
    return parse_response(read_response(ser, response_timeout_s))


def calibrate_sensor(ser: serial.Serial, device_address: int) -> None:
    response = request(ser, build_calibration_frame(device_address))
    if response.status != 0:
        raise RuntimeError(f"Calibration failed with status 0x{response.status:02X}")
    print("Calibration command accepted.")


def read_resultant_force(ser: serial.Serial, device_address: int) -> tuple[float, float, float]:
    response = request(
        ser,
        build_read_frame(device_address, ADDR_RESULTANT_FORCE, 3),
    )
    if response.status != 0:
        raise RuntimeError(f"Resultant force read failed with status 0x{response.status:02X}")
    if len(response.data) < 3:
        raise ValueError(f"Expected 3 resultant-force bytes, got {len(response.data)}")

    fx = signed_u8(response.data[0]) * FORCE_SCALE_N
    fy = signed_u8(response.data[1]) * FORCE_SCALE_N
    fz = response.data[2] * FORCE_SCALE_N
    return fx, fy, fz


def read_distributed_force(
    ser: serial.Serial,
    device_address: int,
    force_points: int,
) -> list[tuple[float, float, float]]:
    data_length = force_points * 3
    response = request(
        ser,
        build_read_frame(device_address, ADDR_DISTRIBUTED_FORCE, data_length),
    )
    if response.status != 0:
        raise RuntimeError(
            f"Distributed force read failed with status 0x{response.status:02X}"
        )
    if len(response.data) < data_length:
        logger.warning(
            "Distributed force response is shorter than expected: got %s/%s bytes",
            len(response.data),
            data_length,
        )

    points = []
    usable_length = (len(response.data) // 3) * 3
    for offset in range(0, usable_length, 3):
        fx_raw, fy_raw, fz_raw = response.data[offset : offset + 3]
        points.append(
            (
                signed_u8(fx_raw) * FORCE_SCALE_N,
                signed_u8(fy_raw) * FORCE_SCALE_N,
                fz_raw * FORCE_SCALE_N,
            )
        )
    return points


def print_force_points(points: list[tuple[float, float, float]], limit: Optional[int] = 12) -> None:
    shown = points if limit is None else points[:limit]
    for index, (fx, fy, fz) in enumerate(shown):
        print(f"  point {index:02d}: Fx={fx:6.1f} N  Fy={fy:6.1f} N  Fz={fz:6.1f} N")
    if limit is not None and len(points) > limit:
        print(f"  ... {len(points) - limit} more points")


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default=None, help="Serial port, e.g. /dev/ttyUSB0")
    parser.add_argument("--baudrate", type=int, default=BAUDRATE)
    parser.add_argument(
        "--sensor-id",
        type=int,
        default=DEFAULT_SENSOR_ID,
        help="Paxini module/sensor ID. Device address on wire is sensor_id + 1.",
    )
    parser.add_argument(
        "--force-points",
        type=int,
        default=DEFAULT_FORCE_POINTS,
        help="Number of distributed triaxial force points to read.",
    )
    parser.add_argument("--skip-calibration", action="store_true")
    parser.add_argument("--interval", type=float, default=0.1)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.debug:
        logger.setLevel(logging.DEBUG)

    device_address = device_address_from_sensor_id(args.sensor_id)
    port = args.port or choose_port()

    print(
        f"\nOpening {port} at {args.baudrate} baud "
        f"(sensor_id={args.sensor_id}, device_address={device_address})"
    )

    ser: Optional[serial.Serial] = None
    try:
        ser = serial.Serial(
            port=port,
            baudrate=args.baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=1,
            write_timeout=0.5,
            inter_byte_timeout=0.001,
            xonxoff=False,
            rtscts=False,
        )

        if not args.skip_calibration:
            print("\nCalibrating sensor...")
            calibrate_sensor(ser, device_address)

        print("\nReading tactile data. Press Ctrl+C to stop.")
        while True:
            resultant = read_resultant_force(ser, device_address)
            distributed = read_distributed_force(
                ser,
                device_address,
                force_points=args.force_points,
            )

            print(
                f"\nResultant: Fx={resultant[0]:6.1f} N  "
                f"Fy={resultant[1]:6.1f} N  Fz={resultant[2]:6.1f} N"
            )
            print_force_points(distributed)
            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        if ser and ser.is_open:
            ser.close()
            print("Serial port closed.")


if __name__ == "__main__":
    main()
