from importlib import resources

import numpy as np

from midas_hand_api.fourbar_lookup import FourBarPipDipMapping, PipDipLookup
from midas_hand_api.hand import MidasHand
from midas_hand_api.kinematics import (
    pip_to_dip_jacobian,
    pip_to_dip_position,
    pip_to_dip_velocity,
)


def _packaged_lookup() -> PipDipLookup:
    resource = resources.files("midas_hand_api").joinpath(
        "assets",
        "pip_dip_linkage_lookup.csv",
    )
    with resources.as_file(resource) as path:
        return PipDipLookup.from_csv(path)


def test_packaged_lookup_interpolates_midpoint() -> None:
    lookup = _packaged_lookup()
    index = 123
    q_pip = 0.5 * (lookup.q_pip[index] + lookup.q_pip[index + 1])
    expected_q_dip = 0.5 * (lookup.q_dip[index] + lookup.q_dip[index + 1])

    assert np.isclose(pip_to_dip_position(q_pip), expected_q_dip)


def test_lookup_matches_default_fourbar_geometry() -> None:
    q_pip = np.deg2rad(40.0)
    expected = FourBarPipDipMapping().solve(q_pip)["q_dip"]

    assert np.isclose(pip_to_dip_position(q_pip), expected, atol=1e-10)


def test_position_clamps_to_lookup_range() -> None:
    lookup = _packaged_lookup()

    assert np.isclose(pip_to_dip_position(-0.1), lookup.q_dip[0])
    assert np.isclose(pip_to_dip_position(10.0), lookup.q_dip[-1])


def test_velocity_uses_jacobian_at_current_pip_position() -> None:
    q_pip = np.deg2rad(35.0)
    pip_vel = 0.42

    assert np.isclose(
        pip_to_dip_velocity(pip_vel, pip_rad=q_pip),
        pip_to_dip_jacobian(q_pip) * pip_vel,
    )


def test_velocity_keeps_one_argument_home_position_default() -> None:
    pip_vel = 0.42

    assert np.isclose(
        pip_to_dip_velocity(pip_vel),
        pip_to_dip_jacobian(0.0) * pip_vel,
    )


def test_hand_velocity_expansion_uses_pip_positions_for_passive_joints() -> None:
    hand = MidasHand(autoconnect=False)
    motor_pos = np.linspace(0.0, 0.6, hand.config.n_motors)
    motor_vel = np.linspace(0.1, 1.3, hand.config.n_motors)

    expanded = hand._expand_velocities(motor_vel, motor_pos)

    for passive_dof, pip_motor in zip(
        hand.config.passive_joint_indices,
        hand.config.pip_motor_indices,
    ):
        assert np.isclose(
            expanded[passive_dof],
            pip_to_dip_velocity(motor_vel[pip_motor], pip_rad=motor_pos[pip_motor]),
        )
