"""Paxini PX-6AX GEN3 tactile sensor API surface.

This module defines the Midas-side tactile interface and data containers. The
wire protocol is intentionally not implemented yet because Paxini's public
pages describe product capabilities and communication accessories, but do not
publish a Python SDK or packet protocol.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

import numpy as np


DEFAULT_PAXINI_MODEL = "PX6AX-GEN3-DP-S2015-Elite"


@dataclass(frozen=True)
class PaxiniConfig:
    """Runtime configuration for Paxini PX-6AX GEN3 tactile sensors.

    ``finger_map`` maps logical Midas hand names such as ``"thumb"`` or
    ``"index"`` to sensor IDs, adapter channels, or vendor SDK handles. Keep it
    generic until the actual Paxini transport is known.
    """

    model: str = DEFAULT_PAXINI_MODEL
    port: Optional[str] = None
    baudrate: int = 921_600
    output_hz: float = 1_000.0
    finger_map: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class TactileFrame:
    """One timestamped tactile sample from one or more Paxini sensors."""

    timestamp_s: float
    sensor_ids: tuple[str, ...]
    force_xyz_n: Optional[np.ndarray] = None
    torque_xyz_nm: Optional[np.ndarray] = None
    taxel_forces_n: Optional[np.ndarray] = None
    raw: Optional[object] = None


class PaxiniSensor:
    """Placeholder Paxini driver matching the future Midas tactile interface."""

    def __init__(self, config: Optional[PaxiniConfig] = None) -> None:
        self.config = config or PaxiniConfig()
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        """Connect to the Paxini adapter or SDK.

        Replace this with the vendor SDK/protocol implementation once Paxini
        provides a Python API, serial frame spec, or C/C++ library binding.
        """

        raise NotImplementedError(
            "Paxini transport is not implemented yet. Need the vendor SDK or "
            "serial/SPI/high-speed-board protocol for PX-6AX GEN3."
        )

    def close(self) -> None:
        self._connected = False

    def read_frame(self) -> TactileFrame:
        """Read one tactile frame from the connected sensor stack."""

        raise NotImplementedError(
            "Paxini frame parsing is not implemented yet. Need the vendor data "
            "format for force, torque, taxel, and raw tactile channels."
        )

    def make_empty_frame(self, sensor_ids: Sequence[str] = ()) -> TactileFrame:
        """Return an empty frame useful for tests before hardware support lands."""

        return TactileFrame(timestamp_s=time.monotonic(), sensor_ids=tuple(sensor_ids))

    def __enter__(self) -> "PaxiniSensor":
        self.connect()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
