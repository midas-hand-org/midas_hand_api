"""Homing routines for the Midas hand."""

from __future__ import annotations

import dataclasses
import time
from typing import Optional

import numpy as np

from . import control_table as ct
from .config import DEFAULT_CONFIG_PATH, HandConfig
from .hand import MidasHand


# (motor_id, joint_name, cad_offset_rad, homing_direction)
# cad_offset_rad: added to hard-stop raw position to get software zero
# homing_direction: +1 = positive encoder direction toward hard stop, -1 = negative
THUMB_HOMING_TABLE = [
    (0, "Thumb IP", -1.59840798, +1),
    (1, "Thumb MCP", -1.60300992, +1),
    (2, "Thumb CMC flex", -0.88510692, +1),
    (3, "Thumb CMC abduct", 0.0, -1),
]


def home_motor(
    hand: MidasHand,
    motor_id: int,
    direction: int,
    cad_offset_rad: float,
    homing_velocity_rad_s: float = 1.0,
    profile_acceleration_raw: int = 20,
    current_threshold_ma: float = 100.0,
    baseline_duration_s: float = 0.5,
    poll_interval_s: float = 0.01,
    timeout_s: float = 15.0,
    profile_velocity_rad_s: Optional[float] = None,
) -> float:
    """Drive a single motor to its hard stop and return the computed home_offset.

    home_offset = hard_stop_raw_rad + cad_offset_rad

    After homing, (raw_rad - home_offset) == 0 at the software zero position.
    """
    client = hand._client()
    motor_idx = hand.motor_ids.index(motor_id)
    if profile_velocity_rad_s is not None:
        homing_velocity_rad_s = profile_velocity_rad_s

    _enter_velocity_homing_mode(
        hand,
        motor_id,
        profile_acceleration_raw=profile_acceleration_raw,
    )

    # Sample baseline current before motion so the hard-stop threshold is stable.
    baseline_samples = []
    hard_stop_raw: Optional[float] = None
    try:
        t0 = time.monotonic()
        while time.monotonic() - t0 < baseline_duration_s:
            baseline_samples.append(abs(float(client.read_cur()[motor_idx])))
            time.sleep(poll_interval_s)
        baseline_ma = float(np.mean(baseline_samples)) if baseline_samples else 0.0
        trigger_ma = baseline_ma + current_threshold_ma
        print(
            f"  Motor {motor_id}: baseline {baseline_ma:.1f} mA, "
            f"trigger {trigger_ma:.1f} mA"
        )

        raw_velocity = _velocity_rad_s_to_raw(
            client, direction * abs(homing_velocity_rad_s)
        )
        client.sync_write(
            [motor_id],
            [raw_velocity],
            ct.ADDR_GOAL_VELOCITY,
            ct.LEN_GOAL_VELOCITY,
        )

        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_s:
            if abs(float(client.read_cur()[motor_idx])) > trigger_ma:
                hard_stop_raw = float(client.read_pos()[motor_idx])
                break
            time.sleep(poll_interval_s)
    finally:
        client.sync_write([motor_id], [0], ct.ADDR_GOAL_VELOCITY, ct.LEN_GOAL_VELOCITY)
        time.sleep(0.05)
        _restore_position_mode(hand, motor_id)

    if hard_stop_raw is None:
        raise TimeoutError(f"Motor {motor_id} did not reach hard stop within {timeout_s}s")

    home_offset = hard_stop_raw + cad_offset_rad
    print(f"  Motor {motor_id}: hard stop {hard_stop_raw:.4f} rad -> home_offset {home_offset:.4f} rad")
    return home_offset


def _enter_velocity_homing_mode(
    hand: MidasHand,
    motor_id: int,
    profile_acceleration_raw: int,
) -> None:
    client = hand._client()
    client.set_torque_enabled([motor_id], False)
    client.sync_write(
        [motor_id],
        [ct.OPERATING_MODE_VELOCITY],
        ct.ADDR_OPERATING_MODE,
        ct.LEN_OPERATING_MODE,
    )
    client.sync_write([motor_id], [0], ct.ADDR_GOAL_VELOCITY, ct.LEN_GOAL_VELOCITY)
    client.sync_write(
        [motor_id],
        [profile_acceleration_raw],
        ct.ADDR_PROFILE_ACCELERATION,
        ct.LEN_PROFILE_ACCELERATION,
    )
    client.set_torque_enabled([motor_id], True)


def _restore_position_mode(hand: MidasHand, motor_id: int) -> None:
    client = hand._client()
    client.set_torque_enabled([motor_id], False)
    client.sync_write(
        [motor_id],
        [hand.config.operating_mode],
        ct.ADDR_OPERATING_MODE,
        ct.LEN_OPERATING_MODE,
    )
    client.sync_write(
        [motor_id],
        [hand.config.position_p_gain],
        ct.ADDR_POSITION_P_GAIN,
        ct.LEN_GAIN,
    )
    client.sync_write(
        [motor_id],
        [hand.config.position_i_gain],
        ct.ADDR_POSITION_I_GAIN,
        ct.LEN_GAIN,
    )
    client.sync_write(
        [motor_id],
        [hand.config.position_d_gain],
        ct.ADDR_POSITION_D_GAIN,
        ct.LEN_GAIN,
    )
    client.sync_write(
        [motor_id],
        [hand.config.goal_current_limit],
        ct.ADDR_GOAL_CURRENT,
        ct.LEN_GOAL_CURRENT,
    )
    client.sync_write(
        [motor_id], [0], ct.ADDR_PROFILE_ACCELERATION, ct.LEN_PROFILE_ACCELERATION
    )
    client.sync_write(
        [motor_id], [0], ct.ADDR_PROFILE_VELOCITY, ct.LEN_PROFILE_VELOCITY
    )
    client.set_torque_enabled([motor_id], True)


def _velocity_rad_s_to_raw(client, velocity_rad_s: float) -> int:
    raw = int(round(abs(velocity_rad_s) / client.vel_scale))
    raw = max(1, raw)
    return raw if velocity_rad_s >= 0 else -raw


def home_thumb(
    hand: MidasHand,
    save: bool = True,
    **kwargs,
) -> HandConfig:
    """Home all 4 thumb motors in sequence and return the updated HandConfig.

    Updates hand.config for the current session so read_pos/set_positions
    immediately reflect the new offsets. Saves to disk by default.

    Pass any home_motor keyword argument (e.g. current_threshold_ma=200) via **kwargs.
    """
    offsets = list(hand.config.home_offsets)
    homed_motor_indices = []

    for motor_id, joint_name, cad_offset, direction in THUMB_HOMING_TABLE:
        if motor_id not in hand.motor_ids:
            print(f"Skipping motor {motor_id} ({joint_name}): not in connected motors")
            continue
        print(f"Homing motor {motor_id} ({joint_name})...")
        motor_idx = hand.motor_ids.index(motor_id)
        offsets[motor_idx] = home_motor(hand, motor_id, direction, cad_offset, **kwargs)
        homed_motor_indices.append(motor_idx)

    new_config = dataclasses.replace(hand.config, home_offsets=tuple(offsets))
    hand.config = new_config

    if not homed_motor_indices:
        print("No thumb motors were homed.")
        return new_config

    time.sleep(0.1)
    print("Moving to software zero...")
    target = hand._read_motor_pos()
    target[homed_motor_indices] = 0.0
    hand.set_motion_profile(profile_velocity_rad_s=0.3, profile_acceleration_raw=20)
    hand.set_positions_blocking(
        target,
        tolerance_rad=0.035,
        velocity_threshold_rad_s=0.05,
        timeout_s=12.0,
        poll_interval_s=0.01,
    )
    hand.set_motion_profile(profile_velocity_rad_s=None, profile_acceleration_raw=0)

    if save:
        new_config.save()
        print(f"Config saved -> {DEFAULT_CONFIG_PATH}")
    
    time.sleep(2.0)

    return new_config
