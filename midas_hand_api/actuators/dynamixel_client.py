"""Low-level Dynamixel SDK wrapper.

This follows the same broad pattern as LEAP Hand's Python API: one client owns a
serial port, sync writers batch motor commands, and sync readers cache the last
successful read so transient packet failures do not immediately crash callers.
"""

from __future__ import annotations

import atexit
import glob
import logging
import pathlib
import shlex
import subprocess
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from . import control_table as ct


Number = Union[int, float]
DXL_ALERT_BIT = 0x80
DEFAULT_BAUDRATE = 4_000_000
_USB_SERIAL_SYSFS = pathlib.Path("/sys/bus/usb-serial/devices")
_LOW_LATENCY_TIMER_MS = 1
_ANSI_BOLD_YELLOW = "\033[1;33m"
_ANSI_RESET = "\033[0m"


def signed_to_unsigned(value: int, size: int) -> int:
    """Return the unsigned two's-complement representation for ``size`` bytes."""

    if value < 0:
        value = (1 << (8 * size)) + value
    return value


def unsigned_to_signed(value: int, size: int) -> int:
    """Interpret ``value`` as a signed two's-complement integer."""

    bit_size = 8 * size
    if value & (1 << (bit_size - 1)):
        value -= 1 << bit_size
    return value


def discover_ports() -> List[str]:
    """Return likely Dynamixel serial ports in stable-to-fallback order."""

    patterns = (
        "/dev/serial/by-id/*",
        "/dev/ttyUSB*",
        "/dev/ttyACM*",
        "/dev/tty.usbserial*",
        "/dev/tty.usbmodem*",
    )
    ports: List[str] = []
    for pattern in patterns:
        ports.extend(sorted(glob.glob(pattern)))
    return list(dict.fromkeys(ports))


def _latency_timer_path(
    port_name: str,
    sysfs_root: pathlib.Path = _USB_SERIAL_SYSFS,
) -> Optional[pathlib.Path]:
    tty_name = pathlib.Path(port_name).resolve().name
    path = sysfs_root / tty_name / "latency_timer"
    return path if path.exists() else None


def _set_low_latency_timer(
    port_name: str,
    target_ms: int = _LOW_LATENCY_TIMER_MS,
    sysfs_root: pathlib.Path = _USB_SERIAL_SYSFS,
) -> None:
    path = _latency_timer_path(port_name, sysfs_root=sysfs_root)
    if path is None:
        return

    command = f"echo {int(target_ms)} | sudo tee {shlex.quote(str(path))}"
    try:
        current_ms = int(path.read_text(encoding="utf-8").strip())
    except OSError as exc:
        logging.warning(
            "Could not read Dynamixel USB serial latency timer for %s at %s: %s. "
            "To set it manually, run: %s",
            port_name,
            path,
            exc,
            command,
        )
        return
    except ValueError:
        logging.warning(
            "Could not parse Dynamixel USB serial latency timer for %s at %s. "
            "To set it manually, run: %s",
            port_name,
            path,
            command,
        )
        return

    if current_ms <= target_ms:
        return

    try:
        path.write_text(f"{int(target_ms)}\n", encoding="utf-8")
    except OSError as exc:
        if _try_sudo_set_latency_timer(path, target_ms):
            logging.info(
                "Set Dynamixel USB serial latency timer for %s to %d ms using sudo",
                port_name,
                target_ms,
            )
            return
        logging.warning(
            "%s Dynamixel USB serial latency timer for %s is %d ms; could not set "
            "it to %d ms automatically: %s. For full-rate sync reads, run: %s. "
            "For a persistent fix, add a udev rule matching this adapter and "
            'setting ATTR{latency_timer}="%d".',
            _highlight_warning("ACTION NEEDED:"),
            port_name,
            current_ms,
            target_ms,
            exc,
            command,
            target_ms,
        )
        return

    logging.info(
        "Set Dynamixel USB serial latency timer for %s from %d ms to %d ms",
        port_name,
        current_ms,
        target_ms,
    )


def _try_sudo_set_latency_timer(path: pathlib.Path, target_ms: int) -> bool:
    if not _can_prompt_for_sudo():
        return False

    logging.warning(
        "%s Dynamixel USB serial latency timer needs sudo access. "
        "You may be prompted for your password now.",
        _highlight_warning("ACTION NEEDED:"),
    )
    try:
        result = subprocess.run(
            ["sudo", "tee", str(path)],
            input=f"{int(target_ms)}\n",
            text=True,
            stdout=subprocess.DEVNULL,
            check=False,
        )
    except OSError as exc:
        logging.warning("Could not run sudo to set %s: %s", path, exc)
        return False

    if result.returncode != 0:
        logging.warning(
            "sudo command failed while setting Dynamixel USB serial latency timer "
            "at %s",
            path,
        )
        return False

    try:
        return int(path.read_text(encoding="utf-8").strip()) <= target_ms
    except (OSError, ValueError):
        return False


def _can_prompt_for_sudo() -> bool:
    return sys.stdin.isatty() and sys.stderr.isatty()


def _highlight_warning(message: str) -> str:
    if sys.stderr.isatty():
        return f"{_ANSI_BOLD_YELLOW}{message}{_ANSI_RESET}"
    return message


def _cleanup_open_clients() -> None:
    for client in list(DynamixelClient.OPEN_CLIENTS):
        try:
            client.disconnect(disable_torque=True)
        except Exception:  # pragma: no cover - best-effort shutdown
            logging.exception("Failed to disconnect Dynamixel client during cleanup")


class DynamixelClient:
    """Client for Protocol 2 Dynamixel motors."""

    OPEN_CLIENTS = set()

    def __init__(
        self,
        motor_ids: Sequence[int],
        port: str = "/dev/ttyUSB0",
        baudrate: int = DEFAULT_BAUDRATE,
        lazy_connect: bool = False,
        pos_scale: float = 2.0 * 3.141592653589793 / 4096.0,
        vel_scale: float = 0.229 * 2.0 * 3.141592653589793 / 60.0,
        cur_scale: float = 1.0,
    ) -> None:
        import dynamixel_sdk

        self.dxl = dynamixel_sdk
        self.motor_ids = list(motor_ids)
        self.port_name = port
        self.baudrate = int(baudrate)
        self.lazy_connect = lazy_connect
        self.pos_scale = float(pos_scale)
        self.vel_scale = float(vel_scale)
        self.cur_scale = float(cur_scale)

        self.port_handler = self.dxl.PortHandler(port)
        self.packet_handler = self.dxl.PacketHandler(ct.PROTOCOL_VERSION)
        self._sync_writers: Dict[Tuple[int, int], object] = {}
        self._pos_reader = DynamixelReader(
            self, self.motor_ids, ct.ADDR_PRESENT_POSITION, ct.LEN_PRESENT_POSITION
        )
        self._vel_reader = DynamixelReader(
            self, self.motor_ids, ct.ADDR_PRESENT_VELOCITY, ct.LEN_PRESENT_VELOCITY
        )
        self._cur_reader = DynamixelReader(
            self, self.motor_ids, ct.ADDR_PRESENT_CURRENT, ct.LEN_PRESENT_CURRENT
        )
        self._pos_vel_reader = DynamixelReader(
            self, self.motor_ids, ct.ADDR_PRESENT_VELOCITY, ct.LEN_PRESENT_POS_VEL
        )
        self._pos_vel_cur_reader = DynamixelReader(
            self, self.motor_ids, ct.ADDR_PRESENT_CURRENT, ct.LEN_PRESENT_POS_VEL_CUR
        )
        self.OPEN_CLIENTS.add(self)

    @property
    def is_connected(self) -> bool:
        return bool(self.port_handler.is_open)

    def connect(self) -> None:
        if self.is_connected:
            return
        if not self.port_handler.openPort():
            raise OSError(
                f"Could not open port '{self.port_name}'. Check the port is correct, "
                "or omit port to auto-detect."
            )
        _set_low_latency_timer(self.port_name)
        if not self.port_handler.setBaudRate(self.baudrate):
            self.port_handler.closePort()
            raise OSError(f"Failed to set Dynamixel baudrate {self.baudrate}")

    def disconnect(self, disable_torque: bool = True) -> None:
        if not self.is_connected:
            self.OPEN_CLIENTS.discard(self)
            return
        if disable_torque:
            try:
                self.set_torque_enabled(self.motor_ids, False, retries=3, verify=True)
            except OSError:
                logging.warning("Could not disable torque during disconnect; closing port anyway")
        self.port_handler.closePort()
        self.OPEN_CLIENTS.discard(self)

    def check_connected(self) -> None:
        if self.lazy_connect and not self.is_connected:
            self.connect()
        if not self.is_connected:
            raise OSError("Must call connect() first.")
        # The SDK sets PortHandler.is_using True at the start of every txPacket
        # and only clears it once the matching read returns. A Ctrl-C (or any
        # exception) raised mid-transaction unwinds the stack before that clear,
        # leaving the flag stuck True so every later packet fails with
        # COMM_PORT_BUSY ("Port is in use!") -- e.g. the torque-disable on
        # shutdown. This client is single-threaded, so the flag can never be
        # legitimately True at a transaction boundary; reset it defensively.
        self.port_handler.is_using = False

    def handle_packet_result(
        self,
        comm_result: int,
        dxl_error: Optional[int] = None,
        dxl_id: Optional[int] = None,
        context: Optional[str] = None,
    ) -> bool:
        error_message = None
        if comm_result != self.dxl.COMM_SUCCESS:
            error_message = self.packet_handler.getTxRxResult(comm_result)
        elif dxl_error:
            alert_only = (dxl_error & DXL_ALERT_BIT) and (
                dxl_error & ~DXL_ALERT_BIT
            ) == 0
            if alert_only:
                message = self.packet_handler.getRxPacketError(dxl_error)
                if dxl_id is not None:
                    message = f"[Motor ID {dxl_id}] {message}"
                if context:
                    message = f"{context}: {message}"
                logging.warning(message)
                return True
            error_message = self.packet_handler.getRxPacketError(dxl_error)

        if not error_message:
            return True

        if dxl_id is not None:
            error_message = f"[Motor ID {dxl_id}] {error_message}"
        if context:
            error_message = f"{context}: {error_message}"
        logging.error(error_message)
        return False

    def ping(self, motor_ids: Optional[Sequence[int]] = None) -> Dict[int, int]:
        """Ping motors and return ``{motor_id: model_number}`` for responders."""

        self.check_connected()
        found: Dict[int, int] = {}
        for motor_id in motor_ids if motor_ids is not None else self.motor_ids:
            model, comm_result, dxl_error = self.packet_handler.ping(
                self.port_handler, int(motor_id)
            )
            if comm_result == self.dxl.COMM_SUCCESS:
                found[int(motor_id)] = int(model)
                if dxl_error:
                    error = self.packet_handler.getRxPacketError(dxl_error)
                    logging.warning("ping: [Motor ID %s] %s", motor_id, error)
            else:
                self.handle_packet_result(comm_result, dxl_error, int(motor_id), "ping")
        return found

    def set_torque_enabled(
        self,
        motor_ids: Sequence[int],
        enabled: bool,
        retries: int = 1,
        retry_interval: float = 0.1,
        verify: bool = False,
    ) -> None:
        remaining = list(motor_ids)
        while remaining:
            remaining = self.write_byte(remaining, int(enabled), ct.ADDR_TORQUE_ENABLE)
            if verify:
                time.sleep(retry_interval)
                states = self.read_byte_data(remaining, ct.ADDR_TORQUE_ENABLE)
                expected = int(enabled)
                remaining = [
                    int(motor_id)
                    for motor_id in remaining
                    if states.get(int(motor_id)) != expected
                ]
            if not remaining or retries == 0:
                break
            time.sleep(retry_interval)
            retries -= 1
        if remaining:
            raise OSError(f"Could not set torque={enabled} for IDs {remaining}")

    def write_byte(self, motor_ids: Sequence[int], value: int, address: int) -> List[int]:
        self.check_connected()
        errored = []
        for motor_id in motor_ids:
            comm_result, dxl_error = self.packet_handler.write1ByteTxRx(
                self.port_handler, int(motor_id), int(address), int(value)
            )
            if not self.handle_packet_result(
                comm_result, dxl_error, int(motor_id), "write_byte"
            ):
                errored.append(int(motor_id))
        return errored

    def read_byte_data(self, motor_ids: Sequence[int], address: int) -> Dict[int, int]:
        self.check_connected()
        values: Dict[int, int] = {}
        for motor_id in motor_ids:
            value, comm_result, dxl_error = self.packet_handler.read1ByteTxRx(
                self.port_handler, int(motor_id), int(address)
            )
            if self.handle_packet_result(
                comm_result, dxl_error, int(motor_id), "read_byte"
            ):
                values[int(motor_id)] = int(value)
        return values

    def read_hardware_error_status(
        self, motor_ids: Optional[Sequence[int]] = None
    ) -> Dict[int, int]:
        return self.read_byte_data(
            list(motor_ids) if motor_ids is not None else self.motor_ids,
            ct.ADDR_HARDWARE_ERROR_STATUS,
        )

    def sync_write(
        self,
        motor_ids: Sequence[int],
        values: Sequence[Number],
        address: int,
        size: int,
    ) -> None:
        self.check_connected()
        if len(motor_ids) != len(values):
            raise ValueError("motor_ids and values must have the same length")

        key = (int(address), int(size))
        if key not in self._sync_writers:
            self._sync_writers[key] = self.dxl.GroupSyncWrite(
                self.port_handler, self.packet_handler, int(address), int(size)
            )
        sync_writer = self._sync_writers[key]

        errored = []
        for motor_id, value in zip(motor_ids, values):
            raw = signed_to_unsigned(int(round(value)), int(size))
            param = raw.to_bytes(int(size), byteorder="little", signed=False)
            if not sync_writer.addParam(int(motor_id), param):
                errored.append(int(motor_id))
        if errored:
            sync_writer.clearParam()
            raise OSError(f"Could not add sync-write params for IDs {errored}")

        comm_result = sync_writer.txPacket()
        sync_writer.clearParam()
        if not self.handle_packet_result(comm_result, context="sync_write"):
            raise OSError("Dynamixel sync_write failed")

    def write_desired_pos(
        self, motor_ids: Sequence[int], positions_rad: Sequence[Number]
    ) -> None:
        raw_positions = [float(pos) / self.pos_scale for pos in positions_rad]
        self.sync_write(
            motor_ids, raw_positions, ct.ADDR_GOAL_POSITION, ct.LEN_GOAL_POSITION
        )

    def read_pos(self) -> np.ndarray:
        return self._pos_reader.read_signed(4) * self.pos_scale

    def read_vel(self) -> np.ndarray:
        return self._vel_reader.read_signed(4) * self.vel_scale

    def read_cur(self) -> np.ndarray:
        return self._cur_reader.read_signed(2) * self.cur_scale

    def read_pos_vel(self) -> Tuple[np.ndarray, np.ndarray]:
        vel_raw, pos_raw = self._pos_vel_reader.read_fields(
            (
                (ct.ADDR_PRESENT_VELOCITY, ct.LEN_PRESENT_VELOCITY, 4),
                (ct.ADDR_PRESENT_POSITION, ct.LEN_PRESENT_POSITION, 4),
            )
        )
        vel = vel_raw * self.vel_scale
        pos = pos_raw * self.pos_scale
        return pos, vel

    def read_pos_vel_cur(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        cur_raw, vel_raw, pos_raw = self._pos_vel_cur_reader.read_fields(
            (
                (ct.ADDR_PRESENT_CURRENT, ct.LEN_PRESENT_CURRENT, 2),
                (ct.ADDR_PRESENT_VELOCITY, ct.LEN_PRESENT_VELOCITY, 4),
                (ct.ADDR_PRESENT_POSITION, ct.LEN_PRESENT_POSITION, 4),
            )
        )
        cur = cur_raw * self.cur_scale
        vel = vel_raw * self.vel_scale
        pos = pos_raw * self.pos_scale
        return pos, vel, cur

    def __enter__(self) -> "DynamixelClient":
        self.connect()
        return self

    def __exit__(self, *args: object) -> None:
        self.disconnect()

    def __del__(self) -> None:
        try:
            self.disconnect()
        except Exception:
            pass


class DynamixelReader:
    """Cached GroupSyncRead helper."""

    def __init__(
        self, client: DynamixelClient, motor_ids: Sequence[int], address: int, size: int
    ) -> None:
        self.client = client
        self.motor_ids = list(motor_ids)
        self.address = int(address)
        self.size = int(size)
        self._data = np.zeros(len(self.motor_ids), dtype=np.int64)
        self._field_cache: Dict[Tuple[int, int, int], np.ndarray] = {}
        self.operation = self.client.dxl.GroupSyncRead(
            client.port_handler, client.packet_handler, self.address, self.size
        )
        for motor_id in self.motor_ids:
            if not self.operation.addParam(int(motor_id)):
                raise OSError(f"Could not add motor ID {motor_id} to sync reader")

    def read_block(self, retries: int = 1) -> np.ndarray:
        self.client.check_connected()
        success = False
        while retries >= 0 and not success:
            if hasattr(self.operation, "fastSyncRead"):
                comm_result = self.operation.fastSyncRead()
            else:
                comm_result = self.operation.txRxPacket()
            success = self.client.handle_packet_result(comm_result, context="read")
            retries -= 1

        if not success:
            return self._data.copy()

        for index, motor_id in enumerate(self.motor_ids):
            if not self.operation.isAvailable(int(motor_id), self.address, self.size):
                logging.error("Sync read data unavailable for motor ID %s", motor_id)
                continue
            self._data[index] = int(
                self.operation.getData(int(motor_id), self.address, self.size)
            )
        return self._data.copy()

    def read_signed(self, size: int, retries: int = 1) -> np.ndarray:
        data = self.read_block(retries)
        return np.asarray([unsigned_to_signed(int(value), size) for value in data])

    def read_fields(
        self,
        fields: Sequence[Tuple[int, int, int]],
        retries: int = 1,
    ) -> Tuple[np.ndarray, ...]:
        """Read multiple signed fields from one sync-read packet.

        Each field is ``(address, length, signed_size)``. The field address and
        length must be inside the range configured for this reader.
        """

        self.client.check_connected()
        success = False
        while retries >= 0 and not success:
            if hasattr(self.operation, "fastSyncRead"):
                comm_result = self.operation.fastSyncRead()
            else:
                comm_result = self.operation.txRxPacket()
            success = self.client.handle_packet_result(comm_result, context="read")
            retries -= 1

        if not success:
            return tuple(
                self._field_cache.get(
                    (int(address), int(length), int(signed_size)),
                    np.zeros(len(self.motor_ids), dtype=np.int64),
                ).copy()
                for address, length, signed_size in fields
            )

        outputs = []
        for address, length, signed_size in fields:
            values = np.zeros(len(self.motor_ids), dtype=np.int64)
            for index, motor_id in enumerate(self.motor_ids):
                if not self.operation.isAvailable(int(motor_id), int(address), int(length)):
                    logging.error("Sync read data unavailable for motor ID %s", motor_id)
                    continue
                raw = int(self.operation.getData(int(motor_id), int(address), int(length)))
                values[index] = unsigned_to_signed(raw, int(signed_size))
            key = (int(address), int(length), int(signed_size))
            self._field_cache[key] = values
            outputs.append(values.copy())
        return tuple(outputs)


atexit.register(_cleanup_open_clients)
