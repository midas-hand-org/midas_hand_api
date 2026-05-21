"""Configuration objects for the Midas hand Dynamixel API."""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field, replace
from math import pi
from typing import Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import yaml  # pyyaml

from .actuators import control_table as ct


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
_MOTOR_VALUE_FIELDS = (
    "home_offsets",
    "joint_signs",
    "joint_lower_limits",
    "joint_upper_limits",
)


def _repeat(value: float, n: int) -> Tuple[float, ...]:
    return tuple(value for _ in range(n))


def _coerce_motor_values(
    values: Union[Sequence[float], Mapping[Union[int, str], float]],
    motor_ids: Sequence[int],
    name: str,
) -> Tuple[float, ...]:
    """Return motor-indexed values ordered to match ``motor_ids``.

    Saved configs may store calibration fields as ``{motor_id: value}`` for
    robustness. Runtime math uses tuples/arrays aligned with ``motor_ids``.
    """

    if isinstance(values, Mapping):
        ordered = []
        for motor_id in motor_ids:
            if motor_id in values:
                ordered.append(float(values[motor_id]))
            elif str(motor_id) in values:
                ordered.append(float(values[str(motor_id)]))
            else:
                raise ValueError(f"{name} is missing value for motor ID {motor_id}")
        return tuple(ordered)

    values_tuple = tuple(float(value) for value in values)
    if len(values_tuple) == len(motor_ids):
        return values_tuple
    if len(values_tuple) == len(DEFAULT_MOTOR_IDS):
        default_index_by_id = {
            motor_id: index for index, motor_id in enumerate(DEFAULT_MOTOR_IDS)
        }
        if all(motor_id in default_index_by_id for motor_id in motor_ids):
            return tuple(
                values_tuple[default_index_by_id[motor_id]]
                for motor_id in motor_ids
            )
        if len(set(values_tuple)) == 1:
            return tuple(values_tuple[0] for _ in motor_ids)
    return values_tuple


def _motor_value_map(
    motor_ids: Sequence[int], values: Sequence[float]
) -> dict[int, float]:
    return {int(motor_id): float(value) for motor_id, value in zip(motor_ids, values)}


def _lookup_motor_value(
    values: Mapping[Union[int, str], float],
    motor_id: int,
    name: str,
) -> float:
    if motor_id in values:
        return float(values[motor_id])
    key = str(motor_id)
    if key in values:
        return float(values[key])
    raise ValueError(f"{name} is missing value for motor ID {motor_id}")


def _filtered_passive_topology(
    source_n_motors: int,
    source_passive_joint_indices: Sequence[int],
    source_pip_motor_indices: Sequence[int],
    selected_motor_indices: Sequence[int],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    selected_old_to_new = {
        old_index: new_index
        for new_index, old_index in enumerate(selected_motor_indices)
    }
    passive_by_dof = dict(zip(source_passive_joint_indices, source_pip_motor_indices))

    passive_joint_indices = []
    pip_motor_indices = []
    motor_index = 0
    filtered_dof_index = 0
    for dof_index in range(source_n_motors + len(source_passive_joint_indices)):
        if dof_index in passive_by_dof:
            pip_motor_index = passive_by_dof[dof_index]
            if pip_motor_index in selected_old_to_new:
                passive_joint_indices.append(filtered_dof_index)
                pip_motor_indices.append(selected_old_to_new[pip_motor_index])
                filtered_dof_index += 1
        else:
            if motor_index in selected_old_to_new:
                filtered_dof_index += 1
            motor_index += 1

    return tuple(passive_joint_indices), tuple(pip_motor_indices)


@dataclass(frozen=True)
class HandConfig:
    """Runtime configuration for the Midas Dynamixel hand.

    Actuator model constants are fixed for the XM335-T323-T motors used by this
    hand. This object keeps runtime connection settings and hand calibration.
    """

    motor_ids: Tuple[int, ...] = DEFAULT_MOTOR_IDS
    port: Optional[str] = None
    baudrate: int = 1_000_000

    # Topology: passive joints and their driving PIP motors.
    # passive_joint_indices: positions in the full n_dof joint array where
    #   passive (linkage-driven) DIP joints are inserted.
    # pip_motor_indices: index into the motor array for the PIP joint that
    #   drives each corresponding passive DIP (same order).
    passive_joint_indices: Tuple[int, ...] = DEFAULT_PASSIVE_JOINT_INDICES
    pip_motor_indices: Tuple[int, ...] = DEFAULT_PIP_MOTOR_INDICES

    position_p_gain: int = 800
    position_i_gain: int = 0
    position_d_gain: int = 500
    goal_current_limit: int = 600
    operating_mode: int = ct.OPERATING_MODE_CURRENT_BASED_POSITION

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

    def __post_init__(self) -> None:
        motor_ids = tuple(int(motor_id) for motor_id in self.motor_ids)
        object.__setattr__(self, "motor_ids", motor_ids)
        if (
            tuple(self.passive_joint_indices) == DEFAULT_PASSIVE_JOINT_INDICES
            and tuple(self.pip_motor_indices) == DEFAULT_PIP_MOTOR_INDICES
        ):
            selected_default_indices = [
                DEFAULT_MOTOR_IDS.index(motor_id)
                for motor_id in motor_ids
                if motor_id in DEFAULT_MOTOR_IDS
            ]
            passive_joint_indices, pip_motor_indices = _filtered_passive_topology(
                len(DEFAULT_MOTOR_IDS),
                DEFAULT_PASSIVE_JOINT_INDICES,
                DEFAULT_PIP_MOTOR_INDICES,
                selected_default_indices,
            )
            object.__setattr__(self, "passive_joint_indices", passive_joint_indices)
            object.__setattr__(self, "pip_motor_indices", pip_motor_indices)
        for name in _MOTOR_VALUE_FIELDS:
            object.__setattr__(
                self,
                name,
                _coerce_motor_values(getattr(self, name), motor_ids, name),
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

        return ct.XM335_T323_POSITION_UNIT_RAD

    @property
    def velocity_scale(self) -> float:
        """Radians/sec per raw velocity unit."""

        return ct.XM335_T323_VELOCITY_UNIT_RAD_S

    @property
    def current_unit_ma(self) -> float:
        """mA per raw current unit."""

        return ct.XM335_T323_CURRENT_UNIT_MA

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
            name: getattr(self, name)
            for name in _MOTOR_VALUE_FIELDS
        }
        for name, values in motor_fields.items():
            if len(values) != n:
                raise ValueError(
                    f"{name} must have {n} values (one per motor), got {len(values)}"
                )
        if len(self.passive_joint_indices) != len(self.pip_motor_indices):
            raise ValueError(
                "passive_joint_indices and pip_motor_indices must have the same length"
            )
        for idx in self.pip_motor_indices:
            if idx >= n:
                raise ValueError(
                    "pip_motor_indices contains "
                    f"{idx}, but only {n} motors are configured"
                )
        if self.goal_current_limit > ct.XM335_T323_MAX_CURRENT_LIMIT:
            raise ValueError(
                "goal_current_limit must be <= "
                f"{ct.XM335_T323_MAX_CURRENT_LIMIT} for XM335-T323-T"
            )

    @classmethod
    def xm335_t323(
        cls,
        motor_ids: Tuple[int, ...] = DEFAULT_MOTOR_IDS,
        port: Optional[str] = None,
        baudrate: int = 1_000_000,
        **overrides,
    ) -> "HandConfig":
        """Backward-compatible constructor; ``HandConfig()`` is XM335-only."""

        for key in (
            "motor_model",
            "expected_model_number",
            "max_goal_current_limit",
            "counts_per_rev",
            "velocity_unit_rpm",
            "current_unit_ma",
        ):
            overrides.pop(key, None)
        values = {
            "motor_ids": motor_ids,
            "port": port,
            "baudrate": baudrate,
        }
        values.update(overrides)
        return cls(**values)

    def subset(self, motor_ids: Sequence[int]) -> "HandConfig":
        """Return this config narrowed to ``motor_ids`` while preserving calibration."""

        requested_ids = set(int(motor_id) for motor_id in motor_ids)
        selected_ids = tuple(
            motor_id for motor_id in self.motor_ids if motor_id in requested_ids
        )
        selected_indices = [
            index
            for index, motor_id in enumerate(self.motor_ids)
            if motor_id in requested_ids
        ]
        if len(selected_ids) != len(requested_ids):
            missing = sorted(requested_ids - set(selected_ids))
            raise ValueError(f"Cannot subset config for unknown motor IDs: {missing}")

        def select(values):
            return tuple(values[index] for index in selected_indices)

        passive_joint_indices, pip_motor_indices = _filtered_passive_topology(
            self.n_motors,
            self.passive_joint_indices,
            self.pip_motor_indices,
            selected_indices,
        )
        return replace(
            self,
            motor_ids=selected_ids,
            passive_joint_indices=passive_joint_indices,
            pip_motor_indices=pip_motor_indices,
            **{name: select(getattr(self, name)) for name in _MOTOR_VALUE_FIELDS},
        )

    def save(self, path: Union[str, pathlib.Path] = DEFAULT_CONFIG_PATH) -> None:
        """Save hand calibration to YAML (default: ``~/.midas_hand/config.yaml``).

        The YAML intentionally stores only hand-specific calibration values.
        Controller gains, current limits, model constants, and baudrate live in
        code defaults and can be overridden explicitly at runtime.
        """
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "motor_ids": list(self.motor_ids),
            "home_offsets": _motor_value_map(self.motor_ids, self.home_offsets),
            "joint_signs": _motor_value_map(self.motor_ids, self.joint_signs),
            "joint_lower_limits": _motor_value_map(
                self.motor_ids, self.joint_lower_limits
            ),
            "joint_upper_limits": _motor_value_map(
                self.motor_ids, self.joint_upper_limits
            ),
        }
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def save_merged_offsets(
        self,
        motor_indices: Sequence[int],
        path: Union[str, pathlib.Path] = DEFAULT_CONFIG_PATH,
    ) -> None:
        """Merge selected home offsets into the saved calibration YAML.

        Homing may be run on only part of the hand. This method preserves
        existing calibration entries for other motors, updates/replaces only
        the successfully homed motor IDs, and drops stale runtime tuning fields.
        """

        path = pathlib.Path(path)
        if path.exists():
            with open(path) as f:
                data = yaml.safe_load(f) or {}
        else:
            data = {}

        saved_motor_ids = [int(motor_id) for motor_id in data.get("motor_ids", [])]
        existing_ids = list(saved_motor_ids)
        for motor_id in self.motor_ids:
            if int(motor_id) not in existing_ids:
                existing_ids.append(int(motor_id))

        merged_data = {"motor_ids": existing_ids}

        for key in _MOTOR_VALUE_FIELDS:
            values = getattr(self, key)
            existing = data.get(key, {})
            if isinstance(existing, list):
                existing = {
                    int(motor_id): float(value)
                    for motor_id, value in zip(saved_motor_ids, existing)
                }
            else:
                existing = {
                    int(motor_id): float(value)
                    for motor_id, value in existing.items()
                }

            if key == "home_offsets":
                indices_to_write = motor_indices
                for index in indices_to_write:
                    existing[int(self.motor_ids[index])] = float(values[index])
            else:
                for index, motor_id in enumerate(self.motor_ids):
                    existing.setdefault(int(motor_id), float(values[index]))
            merged_data[key] = existing

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(merged_data, f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load(cls, path: Union[str, pathlib.Path] = DEFAULT_CONFIG_PATH) -> "HandConfig":
        """Load calibration from YAML over the code defaults.

        Runtime fields that may exist in older YAML files, such as gains,
        current limits, port, baudrate, and model constants, are ignored.
        """
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        motor_ids = tuple(
            int(motor_id) for motor_id in data.get("motor_ids", DEFAULT_MOTOR_IDS)
        )
        config = cls(motor_ids=motor_ids)
        updates = {}
        for key in _MOTOR_VALUE_FIELDS:
            if key in data:
                val = data[key]
                if isinstance(val, dict):
                    updates[key] = tuple(
                        _lookup_motor_value(val, int(mid), key) for mid in motor_ids
                    )
                elif isinstance(val, list):
                    updates[key] = tuple(float(value) for value in val)
        if updates:
            config = replace(config, **updates)
        return config
