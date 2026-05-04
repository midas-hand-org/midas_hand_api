"""Kinematic coupling for Midas hand passive DIP joints.

The 3 non-thumb fingers each have a passive DIP joint mechanically coupled
to the PIP joint via a rigid linkage. The functions below interpolate the
four-bar lookup table generated from the MIDAS finger linkage geometry.

Full 16-DOF joint-space layout (joint index 0–15):

    Joint | Motor ID | Finger | Joint name      | Type
    ------|----------|--------|-----------------|-------
    0     | 0        | Thumb  | IP              | active
    1     | 1        | Thumb  | MCP             | active
    2     | 2        | Thumb  | CMC flex        | active
    3     | 3        | Thumb  | CMC abduct      | active
    4     | 4        | Index  | PIP             | active
    5     | —        | Index  | DIP             | passive (coupled to joint 4)
    6     | 5        | Index  | MCP flex        | active
    7     | 6        | Index  | MCP abduct      | active
    8     | 7        | Middle | PIP             | active
    9     | —        | Middle | DIP             | passive (coupled to joint 8)
    10    | 8        | Middle | MCP flex        | active
    11    | 9        | Middle | MCP abduct      | active
    12    | 10       | Ring   | PIP             | active
    13    | —        | Ring   | DIP             | passive (coupled to joint 12)
    14    | 11       | Ring   | MCP flex        | active
    15    | 12       | Ring   | MCP abduct      | active

Passive joints sit at indices (5, 9, 13). The driving PIP motors are at
0-indexed positions (4, 7, 10) in the 13-element motor array (motor IDs 4, 7, 10).
"""

from __future__ import annotations

from functools import lru_cache
from importlib import resources
from typing import Any

import numpy as np

from .fourbar_lookup import PipDipLookup

_LOOKUP_FILENAME = "pip_dip_linkage_lookup.csv"


@lru_cache(maxsize=1)
def _pip_dip_lookup() -> PipDipLookup:
    """Load the packaged PIP-DIP lookup table once."""

    resource = resources.files(__package__).joinpath("assets", _LOOKUP_FILENAME)
    with resources.as_file(resource) as path:
        return PipDipLookup.from_csv(path)


@lru_cache(maxsize=1)
def _pip_dip_jacobian_table() -> tuple[np.ndarray, np.ndarray]:
    lookup = _pip_dip_lookup()
    return lookup.q_pip, np.gradient(lookup.q_dip, lookup.q_pip)


def _is_scalar(value: Any) -> bool:
    return np.isscalar(value) or np.asarray(value).ndim == 0


def _maybe_scalar(reference: Any, value: np.ndarray | float) -> np.ndarray | float:
    return float(value) if _is_scalar(reference) else value


def pip_to_dip_position(
    pip_rad: float | np.ndarray,
    *,
    clamp: bool = True,
) -> float | np.ndarray:
    """Map PIP joint position to the coupled passive DIP position.

    Args:
        pip_rad: PIP simulator joint position in radians.
        clamp: Clip PIP positions outside the lookup range to the nearest
            endpoint before interpolation.
    """

    q_dip = _pip_dip_lookup().evaluate(pip_rad, clamp=clamp)["q_dip"]
    return _maybe_scalar(pip_rad, q_dip)


def pip_to_dip_jacobian(
    pip_rad: float | np.ndarray,
    *,
    clamp: bool = True,
) -> float | np.ndarray:
    """Return ``d(DIP position) / d(PIP position)`` from the lookup table."""

    q_pip, jacobian = _pip_dip_jacobian_table()
    q = np.asarray(pip_rad, dtype=float)
    q_eval = np.clip(q, q_pip[0], q_pip[-1]) if clamp else q
    result = np.interp(q_eval, q_pip, jacobian)
    return _maybe_scalar(pip_rad, result)


def pip_to_dip_velocity(
    pip_vel_rad_s: float | np.ndarray,
    pip_rad: float | np.ndarray = 0.0,
    *,
    clamp: bool = True,
) -> float | np.ndarray:
    """Map PIP velocity to coupled passive DIP velocity.

    The four-bar coupling is nonlinear, so velocity is
    ``pip_to_dip_jacobian(pip_rad) * pip_vel_rad_s``. ``pip_rad`` defaults to
    zero for backwards compatibility with the previous one-argument API.
    """

    result = np.asarray(pip_to_dip_jacobian(pip_rad, clamp=clamp)) * np.asarray(
        pip_vel_rad_s,
        dtype=float,
    )
    if _is_scalar(pip_vel_rad_s) and _is_scalar(pip_rad):
        return float(result)
    return result
