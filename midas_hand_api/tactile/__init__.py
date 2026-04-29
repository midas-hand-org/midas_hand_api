"""Tactile sensor backends and data types for the Midas hand."""

from .paxini import (
    DEFAULT_PAXINI_MODEL,
    PaxiniConfig,
    PaxiniSensor,
    TactileFrame,
)

__all__ = [
    "DEFAULT_PAXINI_MODEL",
    "PaxiniConfig",
    "PaxiniSensor",
    "TactileFrame",
]
