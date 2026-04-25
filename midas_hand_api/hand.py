"""High-level API for commanding a Dynamixel-based Midas hand."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np

from . import control_table as ct
from . import kinematics
from .config import HandConfig
from .dynamixel_client import DynamixelClient, discover_ports


class MidasHand:
    """Convenience wrapper around a group of Dynamixel hand motors."""

    def __init__(self, config: Optional[HandConfig] = None, autoconnect: bool = True):
        self.config = config or HandConfig()
        self.config.validate()
        self.motor_ids = list(self.config.motor_ids)
        self.port = self.config.port
        self.dxl_client: Optional[DynamixelClient] = None
        self.curr_pos = np.zeros(len(self.motor_ids), dtype=np.float64)
        self.prev_pos = self.curr_pos.copy()
        if autoconnect:
            self.connect()

    def connect(self) -> None:
        """Connect to the configured port, or try likely ports if none was given."""

        ports = [self.port] if self.port else discover_ports()
        if not ports:
            ports = ["/dev/ttyUSB0"]

        last_error: Optional[Exception] = None
        for port in ports:
            if port is None:
                continue
            try:
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
                return
            except Exception as exc:
                last_error = exc
        raise OSError(f"Could not connect to Midas hand. Last error: {last_error}")

    @property
    def is_connected(self) -> bool:
        return bool(self.dxl_client and self.dxl_client.is_connected)

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
        self.prev_pos = self.curr_pos
        self.curr_pos = commanded.copy()
        self._client().write_desired_pos(self.motor_ids, raw_space)

    def set_normalized(self, values: Sequence[float]) -> None:
        """Command joints from normalized ``[-1, 1]`` values."""

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
        """Return joint positions in radians for all 16 DOF.

        Active joints are read from hardware; passive DIP joints are estimated
        via the placeholder coupling in :mod:`midas_hand_api.kinematics`.
        """
        raw_positions = self._client().read_pos()
        motor_pos = (raw_positions - self.config.home_offsets_array) * self.config.joint_signs_array
        return self._expand_positions(motor_pos)

    def read_vel(self) -> np.ndarray:
        """Return joint velocities in rad/s for all 16 DOF.

        Passive DIP velocities are estimated from the PIP velocity via the
        placeholder coupling in :mod:`midas_hand_api.kinematics`.
        """
        raw_velocities = self._client().read_vel()
        motor_vel = raw_velocities * self.config.joint_signs_array
        return self._expand_velocities(motor_vel)

    def read_cur(self) -> np.ndarray:
        """Return motor currents in mA for the 13 active motors."""
        return self._client().read_cur()

    def read_pos_vel(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (positions, velocities) each expanded to 16 DOF."""
        raw_pos, raw_vel = self._client().read_pos_vel()
        motor_pos = (raw_pos - self.config.home_offsets_array) * self.config.joint_signs_array
        motor_vel = raw_vel * self.config.joint_signs_array
        return self._expand_positions(motor_pos), self._expand_velocities(motor_vel)

    def read_pos_vel_cur(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (positions 16-DOF, velocities 16-DOF, currents 13-motor)."""
        raw_pos, raw_vel, cur = self._client().read_pos_vel_cur()
        motor_pos = (raw_pos - self.config.home_offsets_array) * self.config.joint_signs_array
        motor_vel = raw_vel * self.config.joint_signs_array
        return self._expand_positions(motor_pos), self._expand_velocities(motor_vel), cur

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

    def disable_torque(self, motor_ids: Optional[Sequence[int]] = None) -> None:
        """Disable torque for the specified motors (defaults to all)."""
        ids = list(motor_ids) if motor_ids is not None else self.motor_ids
        self._client().set_torque_enabled(ids, False)

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

            hand.set_gains([1, 2, 3, 4], p=600)        # thumb
            hand.set_gains([5, 6, 7, 8, 9, 10, 11, 12, 13], p=900)  # fingers
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
        vals = [current_ma] * len(ids) if isinstance(current_ma, int) else list(current_ma)
        self._client().sync_write(ids, vals, ct.ADDR_GOAL_CURRENT, ct.LEN_GOAL_CURRENT)

    def close(self, disable_torque: bool = True) -> None:
        if self.dxl_client:
            self.dxl_client.disconnect(disable_torque=disable_torque)

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


def unscale(value, lower, upper):
    """Map ``[lower, upper]`` to ``[-1, 1]``."""

    return (2.0 * np.asarray(value, dtype=np.float64) - upper - lower) / (
        upper - lower
    )
