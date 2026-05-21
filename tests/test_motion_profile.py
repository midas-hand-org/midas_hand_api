import pytest

from midas_hand_api.actuators import control_table as ct
from midas_hand_api.config import HandConfig
from midas_hand_api.hand import MidasHand


class FakeDynamixelClient:
    def __init__(self) -> None:
        self.vel_scale = HandConfig().velocity_scale
        self.writes = []
        self.torque_commands = []

    def sync_write(self, motor_ids, values, address, size) -> None:
        self.writes.append((list(motor_ids), list(values), address, size))

    def set_torque_enabled(self, motor_ids, enabled, **kwargs) -> None:
        self.torque_commands.append((list(motor_ids), enabled, kwargs))


def _hand_with_fake_client() -> tuple[MidasHand, FakeDynamixelClient]:
    hand = MidasHand(autoconnect=False)
    client = FakeDynamixelClient()
    hand.dxl_client = client
    return hand, client


def test_motion_profile_converts_acceleration_rad_s2_to_xm335_raw() -> None:
    hand, client = _hand_with_fake_client()
    acceleration = ct.XM335_T323_PROFILE_ACCELERATION_UNIT_RAD_S2 * 300

    hand.set_motion_profile(
        profile_velocity_rad_s=1.5,
        profile_acceleration_rad_s2=acceleration,
        motor_ids=[0, 2],
    )

    assert client.writes[0] == (
        [0, 2],
        [300, 300],
        ct.ADDR_PROFILE_ACCELERATION,
        ct.LEN_PROFILE_ACCELERATION,
    )
    expected_velocity = int(1.5 / client.vel_scale)
    assert client.writes[1] == (
        [0, 2],
        [expected_velocity, expected_velocity],
        ct.ADDR_PROFILE_VELOCITY,
        ct.LEN_PROFILE_VELOCITY,
    )


def test_motion_profile_zero_acceleration_and_unlimited_velocity_write_zero() -> None:
    hand, client = _hand_with_fake_client()

    hand.set_motion_profile(
        profile_velocity_rad_s=0.0,
        profile_acceleration_rad_s2=None,
        motor_ids=[1],
    )

    assert client.writes == [
        ([1], [0], ct.ADDR_PROFILE_ACCELERATION, ct.LEN_PROFILE_ACCELERATION),
        ([1], [0], ct.ADDR_PROFILE_VELOCITY, ct.LEN_PROFILE_VELOCITY),
    ]


def test_motion_profile_rejects_negative_acceleration() -> None:
    hand, _ = _hand_with_fake_client()

    with pytest.raises(ValueError, match="profile_acceleration_rad_s2"):
        hand.set_motion_profile(profile_acceleration_rad_s2=-1.0)


def test_configure_sets_velocity_based_profile_units() -> None:
    hand, client = _hand_with_fake_client()

    hand.configure(enable_torque=False)

    assert client.torque_commands == [(hand.motor_ids, False, {})]
    assert client.writes[0] == (
        hand.motor_ids,
        [ct.DRIVE_MODE_VELOCITY_BASED_PROFILE] * len(hand.motor_ids),
        ct.ADDR_DRIVE_MODE,
        ct.LEN_DRIVE_MODE,
    )
