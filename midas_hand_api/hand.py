"""High-level API for commanding a Dynamixel-based Midas hand."""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Optional, Sequence, Tuple, Union

import numpy as np

from . import kinematics
from .actuators import control_table as ct
from .actuators.dynamixel_client import DynamixelClient, discover_ports
from .config import HandConfig
from .tactile import PaxiniSensor, TactileFrame


class MidasHand:
    """Convenience wrapper around a group of Dynamixel hand motors."""

    def __init__(
        self,
        config: Optional[HandConfig] = None,
        autoconnect: bool = True,
        tactile_sensor: Optional[PaxiniSensor] = None,
    ):
        self.config = config or HandConfig()
        self.config.validate()
        self.motor_ids = list(self.config.motor_ids)
        self.port = self.config.port
        self.dxl_client: Optional[DynamixelClient] = None
        self.tactile_sensor = tactile_sensor
        if autoconnect:
            self.connect()

    def connect(self) -> None:
        """Connect to the configured port, or auto-select a responding bus."""

        ports = [self.port] if self.port else discover_ports()
        if not ports:
            ports = ["/dev/ttyUSB0"]

        last_error: Optional[Exception] = None
        checked_ports = []
        for port in ports:
            if port is None:
                continue
            try:
                if self.port is None and not _port_has_responders(
                    port, self.config.baudrate, self.motor_ids
                ):
                    checked_ports.append(port)
                    continue
                client = DynamixelClient(
                    self.motor_ids,
                    port=port,
                    baudrate=self.config.baudrate,
                    pos_scale=self.config.position_scale,
                    vel_scale=self.config.velocity_scale,
                    cur_scale=self.config.current_unit_ma,
                )
                client.connect()
                self.dxl_client = client
                self.port = port
                self.config = replace(self.config, port=port)
                return
            except Exception as exc:
                last_error = exc
        if self.config.port is None and last_error is None:
            raise OSError(
                "No configured motor IDs responded on candidate ports "
                f"{checked_ports}"
            )
        raise OSError(f"Could not connect to Midas hand. Last error: {last_error}")

    @property
    def is_connected(self) -> bool:
        return bool(self.dxl_client and self.dxl_client.is_connected)

    @property
    def has_tactile(self) -> bool:
        return self.tactile_sensor is not None

    def configure(self, enable_torque: bool = True) -> None:
        """Apply operating mode, gains, and current caps to all motors."""

        client = self._client()
        client.set_torque_enabled(self.motor_ids, False)
        client.sync_write(
            self.motor_ids,
            [self.config.operating_mode] * len(self.motor_ids),
            ct.ADDR_OPERATING_MODE,
            ct.LEN_OPERATING_MODE,
        )
        client.sync_write(
            self.motor_ids,
            [self.config.position_p_gain] * len(self.motor_ids),
            ct.ADDR_POSITION_P_GAIN,
            ct.LEN_GAIN,
        )
        client.sync_write(
            self.motor_ids,
            [self.config.position_i_gain] * len(self.motor_ids),
            ct.ADDR_POSITION_I_GAIN,
            ct.LEN_GAIN,
        )
        client.sync_write(
            self.motor_ids,
            [self.config.position_d_gain] * len(self.motor_ids),
            ct.ADDR_POSITION_D_GAIN,
            ct.LEN_GAIN,
        )
        client.sync_write(
            self.motor_ids,
            [self.config.goal_current_limit] * len(self.motor_ids),
            ct.ADDR_GOAL_CURRENT,
            ct.LEN_GOAL_CURRENT,
        )
        if enable_torque:
            client.set_torque_enabled(self.motor_ids, True)

    def ping(self) -> dict[int, int]:
        """Return motor IDs that respond, mapped to model numbers."""

        return self._client().ping(self.motor_ids)

    def read_hardware_error_status(self) -> dict[int, int]:
        """Return Hardware Error Status(70) for active motors."""

        return self._client().read_hardware_error_status(self.motor_ids)

    def verify_models(self) -> dict[int, int]:
        """Return responding motors whose model number differs from config."""

        if self.config.expected_model_number is None:
            return {}
        return {
            motor_id: model_number
            for motor_id, model_number in self.ping().items()
            if model_number != self.config.expected_model_number
        }

    def set_positions(self, positions_rad: Sequence[float], clip: bool = True) -> None:
        """Command joint positions in radians, using config signs and offsets."""

        if len(positions_rad) != len(self.motor_ids):
            raise ValueError(f"Expected {len(self.motor_ids)} positions")
        commanded = np.asarray(positions_rad, dtype=np.float64)
        if clip:
            commanded = self.clip_positions(commanded)
        raw_space = (
            commanded * self.config.joint_signs_array
        ) + self.config.home_offsets_array
        self._client().write_desired_pos(self.motor_ids, raw_space)

    def set_positions_blocking(
        self,
        positions_rad: Sequence[float],
        clip: bool = True,
        tolerance_rad: float = 0.03,
        velocity_threshold_rad_s: Optional[float] = None,
        timeout_s: float = 5.0,
        poll_interval_s: float = 0.02,
        contact_current_ma: Optional[float] = None,
    ) -> None:
        """Command positions and block until active motors reach them.

        This is intended for homing, resets, demos, and scripted motion. Learned
        controllers should usually call :meth:`set_positions` at a fixed rate.
        If ``contact_current_ma`` is set, the wait exits early when any motor
        exceeds that current, which is useful for contact-aware hand motions.
        """

        commanded = np.asarray(positions_rad, dtype=np.float64)
        if len(commanded) != len(self.motor_ids):
            raise ValueError(f"Expected {len(self.motor_ids)} positions")
        if clip:
            commanded = self.clip_positions(commanded)

        self.set_positions(commanded, clip=False)
        if not self.wait_until_reached(
            commanded,
            tolerance_rad=tolerance_rad,
            velocity_threshold_rad_s=velocity_threshold_rad_s,
            timeout_s=timeout_s,
            poll_interval_s=poll_interval_s,
            contact_current_ma=contact_current_ma,
        ):
            raise TimeoutError(
                f"Motors did not reach target within {timeout_s:.2f}s"
            )

    def wait_until_reached(
        self,
        positions_rad: Sequence[float],
        tolerance_rad: float = 0.03,
        velocity_threshold_rad_s: Optional[float] = None,
        timeout_s: float = 5.0,
        poll_interval_s: float = 0.02,
        contact_current_ma: Optional[float] = None,
    ) -> bool:
        """Wait until active motors are near ``positions_rad``.

        Returns ``True`` when all active motors are within ``tolerance_rad`` and,
        if requested, below ``velocity_threshold_rad_s``. Returns ``False`` on
        timeout or contact-current early exit.
        """

        target = np.asarray(positions_rad, dtype=np.float64)
        if len(target) != len(self.motor_ids):
            raise ValueError(f"Expected {len(self.motor_ids)} positions")

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            pos, vel, cur = self._read_motor_pos_vel_cur()
            position_reached = np.all(np.abs(target - pos) <= tolerance_rad)
            velocity_settled = (
                True
                if velocity_threshold_rad_s is None
                else np.all(np.abs(vel) <= velocity_threshold_rad_s)
            )
            if position_reached and velocity_settled:
                return True
            if contact_current_ma is not None and np.any(
                np.abs(cur) >= contact_current_ma
            ):
                return False
            time.sleep(poll_interval_s)
        return False

    def set_normalized(self, values: Sequence[float]) -> None:
        """Command motor positions from normalized ``[-1, 1]`` values."""

        if len(values) != len(self.motor_ids):
            raise ValueError(f"Expected {len(self.motor_ids)} normalized values")
        values_array = np.asarray(values, dtype=np.float64)
        positions = scale(
            values_array,
            self.config.joint_lower_limits_array,
            self.config.joint_upper_limits_array,
        )
        self.set_positions(positions, clip=True)

    def read_pos(self) -> np.ndarray:
        """Return calibrated motor positions in radians for active actuators."""

        return self._read_motor_pos()

    def read_vel(self) -> np.ndarray:
        """Return calibrated motor velocities in rad/s for active actuators."""

        return self._read_motor_vel()

    def read_cur(self) -> np.ndarray:
        """Return motor currents in mA for the 13 active motors."""
        return self._client().read_cur()

    def read_pos_vel(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return calibrated motor positions and velocities."""
        return self._read_motor_pos_vel()

    def read_pos_vel_cur(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return calibrated motor positions, velocities, and currents."""
        return self._read_motor_pos_vel_cur()

    def read_joint_pos(self) -> np.ndarray:
        """Return expanded joint positions in radians for all 16 DOF.

        Active joints are read from hardware; passive DIP joints are estimated
        via the placeholder coupling in :mod:`midas_hand_api.kinematics`.
        """
        return self._expand_positions(self._read_motor_pos())

    def read_joint_vel(self) -> np.ndarray:
        """Return expanded joint velocities in rad/s for all 16 DOF.

        Passive DIP velocities are estimated from the PIP velocity via the
        placeholder coupling in :mod:`midas_hand_api.kinematics`.
        """
        return self._expand_velocities(self._read_motor_vel())

    def read_tactile(self) -> TactileFrame:
        """Return one tactile frame from the configured tactile sensor."""

        if self.tactile_sensor is None:
            raise OSError("No tactile sensor is configured")
        return self.tactile_sensor.read_frame()

    def _read_motor_pos(self) -> np.ndarray:
        return self._raw_to_motor_pos(self._client().read_pos())

    def _read_motor_vel(self) -> np.ndarray:
        raw_velocities = self._client().read_vel()
        return raw_velocities * self.config.joint_signs_array

    def _read_motor_pos_vel(self) -> Tuple[np.ndarray, np.ndarray]:
        raw_pos, raw_vel = self._client().read_pos_vel()
        motor_pos = self._raw_to_motor_pos(raw_pos)
        motor_vel = raw_vel * self.config.joint_signs_array
        return motor_pos, motor_vel

    def _read_motor_pos_vel_cur(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        raw_pos, raw_vel, cur = self._client().read_pos_vel_cur()
        motor_pos = self._raw_to_motor_pos(raw_pos)
        motor_vel = raw_vel * self.config.joint_signs_array
        return motor_pos, motor_vel, cur

    def _raw_to_motor_pos(self, raw_pos: np.ndarray) -> np.ndarray:
        return (
            raw_pos - self.config.home_offsets_array
        ) * self.config.joint_signs_array

    def _expand_positions(self, motor_pos: np.ndarray) -> np.ndarray:
        """Expand 13-motor positions to 16-DOF joint space.

        Passive DIP joints are inserted at ``config.passive_joint_indices``
        and estimated from their coupled PIP motors via
        :func:`~midas_hand_api.kinematics.pip_to_dip_position`.
        """
        result = np.empty(self.config.n_dof, dtype=np.float64)
        passive_map = dict(zip(self.config.passive_joint_indices, self.config.pip_motor_indices))
        motor_idx = 0
        for dof_idx in range(self.config.n_dof):
            if dof_idx in passive_map:
                result[dof_idx] = kinematics.pip_to_dip_position(
                    motor_pos[passive_map[dof_idx]]
                )
            else:
                result[dof_idx] = motor_pos[motor_idx]
                motor_idx += 1
        return result

    def _expand_velocities(self, motor_vel: np.ndarray) -> np.ndarray:
        """Expand 13-motor velocities to 16-DOF joint space.

        Passive DIP velocities are estimated via
        :func:`~midas_hand_api.kinematics.pip_to_dip_velocity`.
        """
        result = np.empty(self.config.n_dof, dtype=np.float64)
        passive_map = dict(zip(self.config.passive_joint_indices, self.config.pip_motor_indices))
        motor_idx = 0
        for dof_idx in range(self.config.n_dof):
            if dof_idx in passive_map:
                result[dof_idx] = kinematics.pip_to_dip_velocity(
                    motor_vel[passive_map[dof_idx]]
                )
            else:
                result[dof_idx] = motor_vel[motor_idx]
                motor_idx += 1
        return result

    def clip_positions(self, positions_rad: Sequence[float]) -> np.ndarray:
        return np.clip(
            np.asarray(positions_rad, dtype=np.float64),
            self.config.joint_lower_limits_array,
            self.config.joint_upper_limits_array,
        )

    def enable_torque(self, motor_ids: Optional[Sequence[int]] = None) -> None:
        """Enable torque for the specified motors (defaults to all)."""
        ids = list(motor_ids) if motor_ids is not None else self.motor_ids
        self._client().set_torque_enabled(ids, True)

    def disable_torque(
        self,
        motor_ids: Optional[Sequence[int]] = None,
        retries: int = 3,
        verify: bool = True,
    ) -> None:
        """Disable torque for the specified motors, retrying and verifying by default."""
        ids = list(motor_ids) if motor_ids is not None else self.motor_ids
        self._client().set_torque_enabled(ids, False, retries=retries, verify=verify)

    def set_gains(
        self,
        motor_ids: Optional[Sequence[int]] = None,
        p: Optional[Union[int, Sequence[int]]] = None,
        i: Optional[Union[int, Sequence[int]]] = None,
        d: Optional[Union[int, Sequence[int]]] = None,
    ) -> None:
        """Set position PID gains for the specified motors (defaults to all).

        Pass a scalar to apply the same value to every targeted motor, or a
        sequence of the same length as motor_ids for per-motor values.
        Gains can be written while torque is enabled.

        Example — different P gain for thumb vs fingers::

            hand.set_gains([0, 1, 2, 3], p=600)        # thumb
            hand.set_gains([4, 5, 6, 7, 8, 9, 10, 11, 12], p=900)  # fingers
        """
        ids = list(motor_ids) if motor_ids is not None else self.motor_ids
        client = self._client()
        if p is not None:
            client.sync_write(
                ids,
                [p] * len(ids) if isinstance(p, int) else list(p),
                ct.ADDR_POSITION_P_GAIN,
                ct.LEN_GAIN,
            )
        if i is not None:
            client.sync_write(
                ids,
                [i] * len(ids) if isinstance(i, int) else list(i),
                ct.ADDR_POSITION_I_GAIN,
                ct.LEN_GAIN,
            )
        if d is not None:
            client.sync_write(
                ids,
                [d] * len(ids) if isinstance(d, int) else list(d),
                ct.ADDR_POSITION_D_GAIN,
                ct.LEN_GAIN,
            )

    def set_current_limit(
        self,
        current_ma: Union[int, Sequence[int]],
        motor_ids: Optional[Sequence[int]] = None,
    ) -> None:
        """Set the goal current limit (mA) for the specified motors (defaults to all).

        Pass a scalar to apply the same limit to every targeted motor, or a
        sequence of the same length as motor_ids for per-motor values.
        """
        ids = list(motor_ids) if motor_ids is not None else self.motor_ids
        vals = (
            [current_ma] * len(ids)
            if np.isscalar(current_ma)
            else list(current_ma)
        )
        self._client().sync_write(ids, vals, ct.ADDR_GOAL_CURRENT, ct.LEN_GOAL_CURRENT)

    def set_motion_profile(
        self,
        profile_velocity_rad_s: Optional[float] = None,
        profile_acceleration_raw: Optional[int] = None,
        motor_ids: Optional[Sequence[int]] = None,
    ) -> None:
        """Set Dynamixel position-profile velocity and acceleration.

        ``profile_velocity_rad_s=None`` writes ``0``, which lets the actuator use
        its unlimited velocity profile behavior. Acceleration is passed in raw
        Dynamixel units because the exact unit is firmware/model dependent.
        """

        ids = list(motor_ids) if motor_ids is not None else self.motor_ids
        client = self._client()
        if profile_acceleration_raw is not None:
            client.sync_write(
                ids,
                [int(profile_acceleration_raw)] * len(ids),
                ct.ADDR_PROFILE_ACCELERATION,
                ct.LEN_PROFILE_ACCELERATION,
            )
        if profile_velocity_rad_s is not None:
            raw_vel = max(1, int(profile_velocity_rad_s / client.vel_scale))
        else:
            raw_vel = 0
        client.sync_write(
            ids,
            [raw_vel] * len(ids),
            ct.ADDR_PROFILE_VELOCITY,
            ct.LEN_PROFILE_VELOCITY,
        )

    def close(self, disable_torque: bool = True) -> None:
        if self.dxl_client:
            self.dxl_client.disconnect(disable_torque=disable_torque)
        if self.tactile_sensor:
            self.tactile_sensor.close()

    def shutdown(
        self,
        attempts: int = 10,
        torque_retries: int = 10,
        retry_interval_s: float = 0.2,
    ) -> None:
        """Disable torque robustly and close the hand connection.

        This is intended for script shutdown paths. It catches repeated
        ``KeyboardInterrupt`` during torque-disable attempts so Ctrl-C spam is
        less likely to leave one motor torqued on.
        """

        torque_disabled = False
        for attempt in range(1, attempts + 1):
            try:
                self.disable_torque(retries=torque_retries, verify=True)
                torque_disabled = True
                break
            except KeyboardInterrupt:
                print("Interrupt received during shutdown; still disabling torque...")
            except OSError as exc:
                print(f"Torque disable attempt {attempt} failed: {exc}")
                time.sleep(retry_interval_s)

        try:
            self.close(disable_torque=not torque_disabled)
        except KeyboardInterrupt:
            print("Interrupt received while closing port; retrying close...")
            self.close(disable_torque=not torque_disabled)

    def _client(self) -> DynamixelClient:
        if not self.dxl_client:
            raise OSError("MidasHand is not connected")
        return self.dxl_client

    def __enter__(self) -> "MidasHand":
        if not self.is_connected:
            self.connect()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def scale(value, lower, upper):
    """Map ``[-1, 1]`` to ``[lower, upper]``."""

    return 0.5 * (np.asarray(value, dtype=np.float64) + 1.0) * (upper - lower) + lower


def _port_has_responders(port: str, baudrate: int, motor_ids: Sequence[int]) -> bool:
    import dynamixel_sdk

    port_handler = dynamixel_sdk.PortHandler(port)
    packet_handler = dynamixel_sdk.PacketHandler(ct.PROTOCOL_VERSION)
    if not port_handler.openPort():
        return False
    try:
        if not port_handler.setBaudRate(int(baudrate)):
            return False
        for motor_id in motor_ids:
            _, comm_result, _ = packet_handler.ping(port_handler, int(motor_id))
            if comm_result == dynamixel_sdk.COMM_SUCCESS:
                return True
        return False
    finally:
        port_handler.closePort()
