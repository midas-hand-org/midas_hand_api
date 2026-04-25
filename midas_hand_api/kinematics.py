"""Placeholder kinematic coupling for Midas hand passive joints.

The 3 non-thumb fingers each have a passive DIP joint mechanically coupled
to the PIP joint via a rigid linkage. The functions below are placeholder
linear approximations; replace them with a calibrated polynomial or lookup
table measured on the physical hand.

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

Passive joints sit at indices (5, 9, 13).  The driving PIP motors are at
0-indexed positions (4, 7, 10) in the 13-element motor array (motor IDs 4, 7, 10).
"""

from __future__ import annotations


def pip_to_dip_position(pip_rad: float) -> float:
    """Placeholder PIP → DIP position coupling.

    Replace with a calibrated polynomial fit or lookup table.
    The current linear ratio (2/3) is a rough anatomical approximation.
    """
    return pip_rad * (2.0 / 3.0)


def pip_to_dip_velocity(pip_vel_rad_s: float) -> float:
    """Placeholder PIP → DIP velocity coupling.

    For a linear coupling DIP = k·PIP the velocity scales by the same k.
    If the position coupling is non-linear, compute the Jacobian dDIP/dPIP
    and multiply by PIP velocity here.
    """
    return pip_vel_rad_s * (2.0 / 3.0)
