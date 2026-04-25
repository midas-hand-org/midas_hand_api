"""Configuration objects for the Midas hand Dynamixel API."""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from math import pi
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import yaml  # pyyaml

from . import control_table as ct


DEFAULT_MOTOR_IDS = tuple(range(0, 13))

# Default path for persisting hand calibration across sessions.
# Lives in the user's home directory so it survives pip upgrades and
# is never inside the git working tree.
DEFAULT_CONFIG_PATH: pathlib.Path = pathlib.Path.home() / ".midas_hand" / "config.yaml"

# Indices into the 16-DOF joint-space array where passive DIP joints sit.
# Layout: thumb(0-3), index(4,DIP@5,6,7), middle(8,DIP@9,10,11), ring(12,DIP@13,14,15).
DEFAULT_PASSIVE_JOINT_INDICES: Tuple[int, ...] = (5, 9, 13)

# 0-indexed positions in the 13-motor array for the PIP motors that drive each
# passive DIP (same order as DEFAULT_PASSIVE_JOINT_INDICES).
# Motor IDs 4, 7, 10 (index/middle/ring PIP) sit at array positions 4, 7, 10.
DEFAULT_PIP_MOTOR_INDICES: Tuple[int, ...] = (4, 7, 10)


def _repeat(value: float, n: int) -> Tuple[float, ...]:
    return tuple(value for _ in range(n))


@dataclass(frozen=True)
class HandConfig:
    """Runtime configuration for a Dynamixel-driven hand.

    The defaults are intentionally conservative placeholders for a 13-motor hand.
    Update ``motor_ids``, ``home_offsets``, ``joint_signs``, and limits after
    calibration against the real Midas hand.
    """

    motor_ids: Tuple[int, ...] = DEFAULT_MOTOR_IDS
    motor_model: str = "XM335-T323-T"
    expected_model_number: Optional[int] = ct.XM335_T323_MODEL_NUMBER
    port: Optional[str] = None
    baudrate: int = 1_000_000

    # Topology: passive joints and their driving PIP motors.
    # passive_joint_indices: positions in the full n_dof joint array where
    #   passive (linkage-driven) DIP joints are inserted.
    # pip_motor_indices: index into the motor array for the PIP joint that
    #   drives each corresponding passive DIP (same order).
    passive_joint_indices: Tuple[int, ...] = DEFAULT_PASSIVE_JOINT_INDICES
    pip_motor_indices: Tuple[int, ...] = DEFAULT_PIP_MOTOR_INDICES

    position_p_gain: int = 900
    position_i_gain: int = 0
    position_d_gain: int = 0
    goal_current_limit: int = 500
    max_goal_current_limit: int = ct.XM335_T323_MAX_CURRENT_LIMIT
    operating_mode: int = ct.OPERATING_MODE_CURRENT_BASED_POSITION

    counts_per_rev: int = 4096
    velocity_unit_rpm: float = 0.229
    current_unit_ma: float = ct.XM335_T323_CURRENT_UNIT_MA

    home_offsets: Sequence[float] = field(
        default_factory=lambda: _repeat(0.0, len(DEFAULT_MOTOR_IDS))
    )
    joint_signs: Sequence[float] = field(
        default_factory=lambda: _repeat(1.0, len(DEFAULT_MOTOR_IDS))
    )
    joint_lower_limits: Sequence[float] = field(
        default_factory=lambda: _repeat(-pi, len(DEFAULT_MOTOR_IDS))
    )
    joint_upper_limits: Sequence[float] = field(
        default_factory=lambda: _repeat(pi, len(DEFAULT_MOTOR_IDS))
    )

    @property
    def n_motors(self) -> int:
        """Number of active motors (= len(motor_ids))."""
        return len(self.motor_ids)

    @property
    def n_dof(self) -> int:
        """Total joint-space DOF, including passive linkage joints."""
        return self.n_motors + len(self.passive_joint_indices)

    @property
    def position_scale(self) -> float:
        """Radians per encoder count."""

        return 2.0 * pi / float(self.counts_per_rev)

    @property
    def velocity_scale(self) -> float:
        """Radians/sec per raw velocity unit."""

        return self.velocity_unit_rpm * 2.0 * pi / 60.0

    @property
    def home_offsets_array(self) -> np.ndarray:
        return np.asarray(self.home_offsets, dtype=np.float64)

    @property
    def joint_signs_array(self) -> np.ndarray:
        return np.asarray(self.joint_signs, dtype=np.float64)

    @property
    def joint_lower_limits_array(self) -> np.ndarray:
        return np.asarray(self.joint_lower_limits, dtype=np.float64)

    @property
    def joint_upper_limits_array(self) -> np.ndarray:
        return np.asarray(self.joint_upper_limits, dtype=np.float64)

    def validate(self) -> None:
        n = len(self.motor_ids)
        motor_fields = {
            "home_offsets": self.home_offsets,
            "joint_signs": self.joint_signs,
            "joint_lower_limits": self.joint_lower_limits,
            "joint_upper_limits": self.joint_upper_limits,
        }
        for name, values in motor_fields.items():
            if len(values) != n:
                raise ValueError(f"{name} must have {n} values (one per motor), got {len(values)}")
        if len(self.passive_joint_indices) != len(self.pip_motor_indices):
            raise ValueError(
                "passive_joint_indices and pip_motor_indices must have the same length"
            )
        for idx in self.pip_motor_indices:
            if idx >= n:
                raise ValueError(
                    f"pip_motor_indices contains {idx}, but only {n} motors are configured"
                )
        if self.goal_current_limit > self.max_goal_current_limit:
            raise ValueError(
                "goal_current_limit must be <= "
                f"{self.max_goal_current_limit} for {self.motor_model}"
            )

    @classmethod
    def xm335_t323(
        cls,
        motor_ids: Tuple[int, ...] = DEFAULT_MOTOR_IDS,
        port: Optional[str] = None,
        baudrate: int = 1_000_000,
        **overrides,
    ) -> "HandConfig":
        n = len(motor_ids)
        values = {
            "motor_ids": motor_ids,
            "motor_model": "XM335-T323-T",
            "expected_model_number": ct.XM335_T323_MODEL_NUMBER,
            "port": port,
            "baudrate": baudrate,
            "position_p_gain": 900,
            "position_i_gain": 0,
            "position_d_gain": 0,
            "goal_current_limit": 500,
            "max_goal_current_limit": ct.XM335_T323_MAX_CURRENT_LIMIT,
            "operating_mode": ct.OPERATING_MODE_CURRENT_BASED_POSITION,
            "counts_per_rev": 4096,
            "velocity_unit_rpm": 0.229,
            "current_unit_ma": ct.XM335_T323_CURRENT_UNIT_MA,
            "home_offsets": _repeat(0.0, n),
            "joint_signs": _repeat(1.0, n),
            "joint_lower_limits": _repeat(-pi, n),
            "joint_upper_limits": _repeat(pi, n),
            "passive_joint_indices": tuple(
                pj
                for pj, pm in zip(DEFAULT_PASSIVE_JOINT_INDICES, DEFAULT_PIP_MOTOR_INDICES)
                if pm < n
            ),
            "pip_motor_indices": tuple(pm for pm in DEFAULT_PIP_MOTOR_INDICES if pm < n),
        }
        values.update(overrides)
        return cls(**values)

    def save(self, path: Union[str, pathlib.Path] = DEFAULT_CONFIG_PATH) -> None:
        """Save this configuration to a YAML file (default: ``~/.midas_hand/config.yaml``).

        Includes all calibration data (home_offsets, joint_signs, limits) so a
        single save after the initial calibration run is enough for future sessions.
        The directory is created automatically if it does not exist.
        """
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "motor_ids": list(self.motor_ids),
            "motor_model": self.motor_model,
            "expected_model_number": self.expected_model_number,
            "port": self.port,
            "baudrate": self.baudrate,
            "passive_joint_indices": list(self.passive_joint_indices),
            "pip_motor_indices": list(self.pip_motor_indices),
            "position_p_gain": self.position_p_gain,
            "position_i_gain": self.position_i_gain,
            "position_d_gain": self.position_d_gain,
            "goal_current_limit": self.goal_current_limit,
            "max_goal_current_limit": self.max_goal_current_limit,
            "operating_mode": self.operating_mode,
            "counts_per_rev": self.counts_per_rev,
            "velocity_unit_rpm": float(self.velocity_unit_rpm),
            "current_unit_ma": float(self.current_unit_ma),
            "home_offsets": [float(v) for v in self.home_offsets],
            "joint_signs": [float(v) for v in self.joint_signs],
            "joint_lower_limits": [float(v) for v in self.joint_lower_limits],
            "joint_upper_limits": [float(v) for v in self.joint_upper_limits],
        }
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load(cls, path: Union[str, pathlib.Path] = DEFAULT_CONFIG_PATH) -> "HandConfig":
        """Load a configuration from a YAML file (default: ``~/.midas_hand/config.yaml``)."""
        with open(path) as f:
            data = yaml.safe_load(f)
        for key in (
            "motor_ids",
            "passive_joint_indices",
            "pip_motor_indices",
            "home_offsets",
            "joint_signs",
            "joint_lower_limits",
            "joint_upper_limits",
        ):
            if key in data and isinstance(data[key], list):
                data[key] = tuple(data[key])
        return cls(**data)
