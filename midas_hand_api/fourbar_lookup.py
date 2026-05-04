#!/usr/bin/env python3
"""
fourbar_lookup.py

Simulator-independent PIP-DIP four-bar lookup-table generator.

Default geometry matches the uploaded MIDAS finger MJCF:

    PIP joint         : revolute_2_0
    DIP joint         : revolute_1_0
    linkage base joint: revolute_5_0

Geometry, in the local planar frame at the PIP joint:

    O = PIP pivot
    A = linkage base pivot = (-5 mm, 12 mm)
    B = DIP pivot, |O-B| = 33 mm
    C = distal linkage pin
    |A-C| = 33 mm
    |B-C| = 12 mm

The generated table maps simulator PIP qpos to simulator passive joint qpos:

    q_pip_sim -> q_dip_sim, q_linkage_base_sim

For this MJCF, PIP joint axis is approximately -Z, while DIP/linkage axes are +Z,
so the default uses pip_sign=-1.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def wrap_to_pi(angle: np.ndarray | float) -> np.ndarray | float:
    """Wrap angle(s) to [-pi, pi]."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def circle_intersection(
    c0: np.ndarray,
    r0: float,
    c1: np.ndarray,
    r1: float,
    branch: int = -1,
    eps: float = 1e-10,
) -> np.ndarray:
    """
    Return one intersection point of two circles.

    branch = +1 or -1 selects the assembly mode.
    If your linkage appears mirrored or bends the wrong way, flip branch.
    """
    c0 = np.asarray(c0, dtype=float)
    c1 = np.asarray(c1, dtype=float)

    d_vec = c1 - c0
    d = np.linalg.norm(d_vec)

    if d < eps:
        raise ValueError("Circle centers are identical or too close.")

    if d > r0 + r1 + eps:
        raise ValueError(
            f"No closure: circles are too far apart. d={d:.6g}, r0+r1={r0+r1:.6g}"
        )

    if d < abs(r0 - r1) - eps:
        raise ValueError(
            f"No closure: one circle is inside the other. d={d:.6g}, |r0-r1|={abs(r0-r1):.6g}"
        )

    a = (r0 * r0 - r1 * r1 + d * d) / (2.0 * d)
    h2 = r0 * r0 - a * a
    h = np.sqrt(max(h2, 0.0))

    e = d_vec / d
    n = np.array([-e[1], e[0]])

    p_mid = c0 + a * e
    return p_mid + branch * h * n


@dataclass
class FourBarPipDipMapping:
    """
    Pure kinematic mapping from PIP angle to passive DIP/linkage angles.

    All lengths are meters.
    All angles are radians.

    Coordinate convention:
        O: PIP joint center, fixed at [0, 0]
        A: linkage base pivot, fixed relative to O
        B: DIP joint center
        C: distal linkage mounting point

    The returned q_dip and q_linkage_base are relative to the initial CAD pose.
    """

    A: Tuple[float, float] = (-0.005, 0.012)
    pip_len: float = 0.033
    cross_len: float = 0.033
    distal_mount_len: float = 0.012
    pip_initial_angle: float = np.pi / 2.0

    # For the uploaded MJCF, branch=-1 usually matches the physical folding side.
    branch: int = -1

    # Simulator sign convention.
    # In the uploaded MJCF, PIP axis is -Z, DIP/linkage axes are +Z.
    pip_sign: float = -1.0
    dip_sign: float = 1.0
    linkage_sign: float = 1.0

    def __post_init__(self) -> None:
        self.O = np.array([0.0, 0.0], dtype=float)
        self.A_np = np.array(self.A, dtype=float)

        init = self.forward_points(0.0)
        B0 = init["B"]
        C0 = init["C"]

        self.pip_abs0 = self._angle(B0 - self.O)
        self.dip_abs0 = self._angle(C0 - B0)
        self.cross_abs0 = self._angle(C0 - self.A_np)

        self.dip_relative0 = wrap_to_pi(self.dip_abs0 - self.pip_abs0)
        self.cross_distal_relative0 = wrap_to_pi(self.cross_abs0 - self.dip_abs0)

    @staticmethod
    def _angle(v: np.ndarray) -> float:
        return float(np.arctan2(v[1], v[0]))

    def forward_points(self, q_pip_sim: float) -> Dict[str, np.ndarray]:
        """
        Compute planar pivot points O, A, B, C.

        q_pip_sim is the simulator joint qpos for the PIP joint.
        """
        q_pip_geom = self.pip_sign * q_pip_sim
        pip_abs = self.pip_initial_angle + q_pip_geom

        B = self.O + self.pip_len * np.array([np.cos(pip_abs), np.sin(pip_abs)])

        C = circle_intersection(
            self.A_np,
            self.cross_len,
            B,
            self.distal_mount_len,
            branch=self.branch,
        )

        return {
            "O": self.O.copy(),
            "A": self.A_np.copy(),
            "B": B,
            "C": C,
        }

    def solve(self, q_pip_sim: float) -> Dict[str, float | np.ndarray]:
        """
        Map simulator PIP qpos to simulator passive joint qpos.

        Returns:
            q_dip
            q_linkage_base
            q_linkage_distal
            O, A, B, C
        """
        pts = self.forward_points(float(q_pip_sim))
        O, A, B, C = pts["O"], pts["A"], pts["B"], pts["C"]

        pip_abs = self._angle(B - O)
        dip_abs = self._angle(C - B)
        cross_abs = self._angle(C - A)

        q_dip_geom = wrap_to_pi((dip_abs - pip_abs) - self.dip_relative0)
        q_linkage_base_geom = wrap_to_pi(cross_abs - self.cross_abs0)
        q_linkage_distal_geom = wrap_to_pi(
            (cross_abs - dip_abs) - self.cross_distal_relative0
        )

        return {
            "q_pip": float(q_pip_sim),
            "q_dip": float(self.dip_sign * q_dip_geom),
            "q_linkage_base": float(self.linkage_sign * q_linkage_base_geom),
            "q_linkage_distal": float(q_linkage_distal_geom),
            "O": O,
            "A": A,
            "B": B,
            "C": C,
        }

    def make_table(self, q_min: float, q_max: float, n: int = 801) -> np.ndarray:
        """
        Build lookup table with columns:
            q_pip, q_dip, q_linkage_base, q_linkage_distal
        """
        q_values = np.linspace(q_min, q_max, n)
        rows = []

        for q in q_values:
            try:
                out = self.solve(float(q))
                rows.append([
                    out["q_pip"],
                    out["q_dip"],
                    out["q_linkage_base"],
                    out["q_linkage_distal"],
                ])
            except ValueError:
                rows.append([q, np.nan, np.nan, np.nan])

        return np.asarray(rows, dtype=float)

    def save_csv(self, filename: str | Path, q_min: float, q_max: float, n: int = 801) -> None:
        table = self.make_table(q_min, q_max, n)
        header = "q_pip,q_dip,q_linkage_base,q_linkage_distal"
        np.savetxt(filename, table, delimiter=",", header=header, comments="")


@dataclass
class PipDipLookup:
    """Small runtime interpolator for the generated CSV lookup table."""

    q_pip: np.ndarray
    q_dip: np.ndarray
    q_linkage_base: np.ndarray
    q_linkage_distal: np.ndarray

    @classmethod
    def from_csv(cls, filename: str | Path) -> "PipDipLookup":
        data = np.genfromtxt(filename, delimiter=",", names=True)
        valid = (
            np.isfinite(data["q_pip"])
            & np.isfinite(data["q_dip"])
            & np.isfinite(data["q_linkage_base"])
        )

        q_pip = np.asarray(data["q_pip"][valid], dtype=float)
        order = np.argsort(q_pip)

        return cls(
            q_pip=q_pip[order],
            q_dip=np.asarray(data["q_dip"][valid], dtype=float)[order],
            q_linkage_base=np.asarray(data["q_linkage_base"][valid], dtype=float)[order],
            q_linkage_distal=np.asarray(data["q_linkage_distal"][valid], dtype=float)[order],
        )

    def evaluate(self, q_pip: float | np.ndarray, clamp: bool = True) -> Dict[str, np.ndarray | float]:
        """
        Interpolate passive joint positions from q_pip.

        If clamp=True, values outside the table range are clipped to the table range.
        """
        q = np.asarray(q_pip, dtype=float)

        if clamp:
            q_eval = np.clip(q, self.q_pip[0], self.q_pip[-1])
        else:
            q_eval = q

        q_dip = np.interp(q_eval, self.q_pip, self.q_dip)
        q_linkage = np.interp(q_eval, self.q_pip, self.q_linkage_base)
        q_linkage_distal = np.interp(q_eval, self.q_pip, self.q_linkage_distal)

        if np.isscalar(q_pip):
            return {
                "q_dip": float(q_dip),
                "q_linkage_base": float(q_linkage),
                "q_linkage_distal": float(q_linkage_distal),
            }

        return {
            "q_dip": q_dip,
            "q_linkage_base": q_linkage,
            "q_linkage_distal": q_linkage_distal,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MIDAS finger PIP-DIP linkage lookup table.")
    parser.add_argument("--out", default="pip_dip_linkage_lookup.csv", help="Output CSV filename.")
    parser.add_argument("--qmin-deg", type=float, default=0.0, help="Minimum PIP simulator angle in degrees.")
    parser.add_argument("--qmax-deg", type=float, default=80.0, help="Maximum PIP simulator angle in degrees.")
    parser.add_argument("--n", type=int, default=801, help="Number of lookup samples.")

    parser.add_argument("--ground-x", type=float, default=-0.005, help="A_x relative to PIP joint, meters.")
    parser.add_argument("--ground-y", type=float, default=0.012, help="A_y relative to PIP joint, meters.")
    parser.add_argument("--pip-len", type=float, default=0.033, help="PIP-to-DIP length, meters.")
    parser.add_argument("--cross-len", type=float, default=0.033, help="Linkage base-to-tip length, meters.")
    parser.add_argument("--distal-mount-len", type=float, default=0.012, help="DIP-to-distal-pin distance, meters.")
    parser.add_argument("--branch", type=int, default=-1, choices=[-1, 1], help="Four-bar assembly branch.")
    parser.add_argument("--pip-sign", type=float, default=-1.0, help="Simulator PIP sign to geometric PIP sign.")
    parser.add_argument("--dip-sign", type=float, default=1.0, help="Geometric DIP sign to simulator DIP sign.")
    parser.add_argument("--linkage-sign", type=float, default=1.0, help="Geometric linkage sign to simulator linkage sign.")

    args = parser.parse_args()

    mapping = FourBarPipDipMapping(
        A=(args.ground_x, args.ground_y),
        pip_len=args.pip_len,
        cross_len=args.cross_len,
        distal_mount_len=args.distal_mount_len,
        branch=args.branch,
        pip_sign=args.pip_sign,
        dip_sign=args.dip_sign,
        linkage_sign=args.linkage_sign,
    )

    q_min = np.deg2rad(args.qmin_deg)
    q_max = np.deg2rad(args.qmax_deg)

    mapping.save_csv(args.out, q_min, q_max, args.n)

    lookup = PipDipLookup.from_csv(args.out)
    print(f"Saved lookup table: {args.out}")
    print(f"PIP range: {args.qmin_deg:.2f} deg to {args.qmax_deg:.2f} deg")
    print(f"Rows: {len(lookup.q_pip)}")

    for deg in [args.qmin_deg, 0.5 * (args.qmin_deg + args.qmax_deg), args.qmax_deg]:
        out = lookup.evaluate(np.deg2rad(deg))
        print(
            f"q_pip={deg:7.2f} deg -> "
            f"q_dip={np.rad2deg(out['q_dip']):8.3f} deg, "
            f"q_linkage={np.rad2deg(out['q_linkage_base']):8.3f} deg"
        )


if __name__ == "__main__":
    main()
