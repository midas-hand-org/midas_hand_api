"""Homing routines for the Midas hand."""

from __future__ import annotations

import dataclasses
import time
from typing import Optional, Sequence, Tuple

import numpy as np

from .actuators import control_table as ct
from .config import DEFAULT_CONFIG_PATH, HandConfig
from .hand import MidasHand


# (motor_id, joint_name, cad_offset_rad, homing_direction)
# cad_offset_rad: added to hard-stop raw position to get software zero
# homing_direction: +1 = positive encoder direction toward hard stop, -1 = negative
THUMB_HOMING_TABLE = [
    (0, "Thumb IP", 1.61221381, -1),
    (1, "Thumb MCP", 1.4450099, -1),
    (2, "Thumb CMC flex", -0.88510692, +1),
    (3, "Thumb CMC abduct", 0.0, -1),
]

FINGER_HOMING_TABLE = [
    (4, "Index PIP", 0.0, +1),
    (5, "Index MCP flex", -0.05, +1),
    (6, "Index MCP abduct", 0.81, -1),
    (7, "Middle PIP", 0.0, +1),
    (8, "Middle MCP flex", -0.05, +1),
    (9, "Middle MCP abduct", 0.81, -1),
    (10, "Ring PIP", 0.0, +1),
    (11, "Ring MCP flex", -0.05, +1),
    (12, "Ring MCP abduct", -0.81, +1),
]

INDEX_FINGER_HOMING_TABLE = FINGER_HOMING_TABLE[0:3]
MIDDLE_FINGER_PRE_ABDUCT_HOMING_TABLE = FINGER_HOMING_TABLE[3:5]
MIDDLE_FINGER_ABDUCT_HOMING_TABLE = FINGER_HOMING_TABLE[5:6]
RING_FINGER_HOMING_TABLE = FINGER_HOMING_TABLE[6:9]
MIDDLE_MCP_FLEX_MOTOR_ID = 8
MIDDLE_MCP_ABDUCT_MOTOR_ID = 9
MIDDLE_MCP_FLEX_PRE_ABDUCT_POSITION_RAD = -0.5 * np.pi

HomingEntry = Tuple[int, str, float, int]


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
    return _home_table(
        hand,
        THUMB_HOMING_TABLE,
        "thumb",
        save=save,
        check_connected=True,
        **kwargs,
    )


def home_fingers(
    hand: MidasHand,
    save: bool = True,
    **kwargs,
) -> HandConfig:
    """Home index, middle, and ring finger motors."""
    return _home_finger_sequence(hand, "finger", save=save, **kwargs)


def home_hand(
    hand: MidasHand,
    save: bool = True,
    **kwargs,
) -> HandConfig:
    """Home thumb plus index, middle, and ring finger motors."""
    offsets = list(hand.config.home_offsets)
    offsets, homed_motor_indices = _home_entries(
        hand, THUMB_HOMING_TABLE, "thumb", offsets, **kwargs
    )
    return _home_finger_sequence(
        hand,
        "hand",
        save=save,
        initial_offsets=offsets,
        initial_homed_motor_indices=homed_motor_indices,
        **kwargs,
    )


def _home_finger_sequence(
    hand: MidasHand,
    label: str,
    save: bool = True,
    initial_offsets: Optional[Sequence[float]] = None,
    initial_homed_motor_indices: Optional[Sequence[int]] = None,
    **kwargs,
) -> HandConfig:
    offsets = list(initial_offsets or hand.config.home_offsets)
    homed_motor_indices = list(initial_homed_motor_indices or [])

    for table, table_label in (
        (INDEX_FINGER_HOMING_TABLE, "index finger"),
        (MIDDLE_FINGER_PRE_ABDUCT_HOMING_TABLE, "middle finger"),
    ):
        offsets, new_indices = _home_entries(hand, table, table_label, offsets, **kwargs)
        homed_motor_indices.extend(new_indices)

    _set_middle_mcp_flex_for_abduction(hand, offsets)

    offsets, new_indices = _home_entries(
        hand,
        MIDDLE_FINGER_ABDUCT_HOMING_TABLE,
        "middle finger abduction",
        offsets,
        **kwargs,
    )
    homed_motor_indices.extend(new_indices)

    offsets, new_indices = _home_entries(
        hand, RING_FINGER_HOMING_TABLE, "ring finger", offsets, **kwargs
    )
    homed_motor_indices.extend(new_indices)

    return _finish_homing(
        hand,
        offsets,
        homed_motor_indices,
        label,
        save,
        pre_zero_motor_id=MIDDLE_MCP_ABDUCT_MOTOR_ID,
    )


def _home_table(
    hand: MidasHand,
    table: Sequence[HomingEntry],
    label: str,
    save: bool = True,
    check_connected: bool = True,
    **kwargs,
) -> HandConfig:
    offsets = list(hand.config.home_offsets)
    offsets, homed_motor_indices = _home_entries(
        hand,
        table,
        label,
        offsets,
        check_connected=check_connected,
        **kwargs,
    )
    return _finish_homing(hand, offsets, homed_motor_indices, label, save)


def _home_entries(
    hand: MidasHand,
    table: Sequence[HomingEntry],
    label: str,
    offsets: list[float],
    check_connected: bool = True,
    **kwargs,
) -> Tuple[list[float], list[int]]:
    homed_motor_indices = []
    table_motor_ids = [entry[0] for entry in table]
    configured_table_ids = [motor_id for motor_id in table_motor_ids if motor_id in hand.motor_ids]
    connected_ids = set(table_motor_ids)

    if check_connected:
        ping_result = hand.ping()
        connected_ids = set(ping_result)
        expected_ids = set(configured_table_ids)
        missing_expected = sorted(expected_ids - connected_ids)
        if missing_expected:
            print(
                f"ERROR: Missing {label} motor IDs: {missing_expected}. "
                "They will be skipped."
            )

    for motor_id, joint_name, cad_offset, direction in table:
        if motor_id not in hand.motor_ids:
            print(f"Skipping motor {motor_id} ({joint_name}): not in connected motors")
            continue
        if motor_id not in connected_ids:
            print(f"Skipping motor {motor_id} ({joint_name}): no ping response")
            continue
        print(f"Homing motor {motor_id} ({joint_name})...")
        motor_idx = hand.motor_ids.index(motor_id)
        offsets[motor_idx] = home_motor(hand, motor_id, direction, cad_offset, **kwargs)
        homed_motor_indices.append(motor_idx)

    return offsets, homed_motor_indices


def _finish_homing(
    hand: MidasHand,
    offsets: Sequence[float],
    homed_motor_indices: Sequence[int],
    label: str,
    save: bool,
    pre_zero_motor_id: Optional[int] = None,
) -> HandConfig:
    new_config = dataclasses.replace(hand.config, home_offsets=tuple(offsets))
    hand.config = new_config

    if not homed_motor_indices:
        print(f"No {label} motors were homed.")
        return new_config

    if pre_zero_motor_id is not None:
        _move_homed_motor_to_zero_first(
            hand,
            pre_zero_motor_id,
            homed_motor_indices,
        )

    _move_homed_motors_to_zero(hand, homed_motor_indices)

    if save:
        new_config.save_merged_offsets(homed_motor_indices)
        print(f"Config saved -> {DEFAULT_CONFIG_PATH}")

    time.sleep(2.0)

    return new_config


def _move_homed_motor_to_zero_first(
    hand: MidasHand,
    motor_id: int,
    homed_motor_indices: Sequence[int],
) -> None:
    if motor_id not in hand.motor_ids:
        return

    motor_idx = hand.motor_ids.index(motor_id)
    if motor_idx not in homed_motor_indices:
        return

    print(f"Moving motor {motor_id} to software zero before group zero move...")
    _set_single_motor_position_and_wait(
        hand,
        motor_idx,
        0.0,
        home_offsets=hand.config.home_offsets,
        tolerance_rad=0.035,
        velocity_threshold_rad_s=0.05,
        timeout_s=8.0,
        poll_interval_s=0.01,
    )


def _set_middle_mcp_flex_for_abduction(
    hand: MidasHand,
    home_offsets: Sequence[float],
) -> None:
    if MIDDLE_MCP_FLEX_MOTOR_ID not in hand.motor_ids:
        print(
            "Skipping middle MCP flex pre-abduction move: "
            f"motor {MIDDLE_MCP_FLEX_MOTOR_ID} is not configured"
        )
        return

    motor_idx = hand.motor_ids.index(MIDDLE_MCP_FLEX_MOTOR_ID)
    print("Moving Middle MCP flex to -90 deg before Middle MCP abduct homing...")
    _set_single_motor_position_and_wait(
        hand,
        motor_idx,
        MIDDLE_MCP_FLEX_PRE_ABDUCT_POSITION_RAD,
        home_offsets=home_offsets,
        tolerance_rad=0.035,
        velocity_threshold_rad_s=0.05,
        timeout_s=8.0,
        poll_interval_s=0.01,
    )


def _set_single_motor_position_and_wait(
    hand: MidasHand,
    motor_idx: int,
    target_rad: float,
    home_offsets: Sequence[float],
    tolerance_rad: float,
    velocity_threshold_rad_s: float,
    timeout_s: float,
    poll_interval_s: float,
) -> None:
    motor_id = hand.motor_ids[motor_idx]
    previous_config = hand.config
    hand.config = dataclasses.replace(hand.config, home_offsets=tuple(home_offsets))

    try:
        target = hand._read_motor_pos()
        target[motor_idx] = target_rad
        hand.set_motion_profile(
            profile_velocity_rad_s=0.5,
            profile_acceleration_raw=200,
            motor_ids=[motor_id],
        )
        hand.set_positions(target, clip=False)

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            pos = hand._read_motor_pos()[motor_idx]
            vel = hand._read_motor_vel()[motor_idx]
            if (
                abs(pos - target_rad) <= tolerance_rad
                and abs(vel) <= velocity_threshold_rad_s
            ):
                return
            time.sleep(poll_interval_s)
        raise TimeoutError(
            f"Motor {motor_id} did not reach {target_rad:.3f} rad within {timeout_s:.2f}s"
        )
    finally:
        try:
            hand.set_motion_profile(
                profile_velocity_rad_s=None,
                profile_acceleration_raw=0,
                motor_ids=[motor_id],
            )
        finally:
            hand.config = previous_config


def _move_homed_motors_to_zero(
    hand: MidasHand,
    homed_motor_indices: Sequence[int],
) -> None:
    time.sleep(0.1)
    print("Moving homed motors to software zero...")
    target = hand._read_motor_pos()
    target[homed_motor_indices] = 0.0
    hand.set_motion_profile(profile_velocity_rad_s=0.5, profile_acceleration_raw=200)
    try:
        hand.set_positions(target, clip=False)
        reached = hand.wait_until_reached(
            target,
            tolerance_rad=0.05,
            velocity_threshold_rad_s=0.05,
            timeout_s=12.0,
            poll_interval_s=0.01,
        )
        if not reached:
            raise TimeoutError("Homed motors did not reach software zero within 12.00s")
    finally:
        hand.set_motion_profile(profile_velocity_rad_s=None, profile_acceleration_raw=0)
