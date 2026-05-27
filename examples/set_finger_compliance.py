"""Adjust hand motor PID/current settings for more compliant contact.

This script first moves all motors to calibrated software zero with the normal
configured gains. It then writes Dynamixel RAM registers for position P/I/D
gains and, unless disabled, goal current limit, leaving torque enabled so the
hand holds that zero pose with the softer settings. Lower P and D gains make
the hand less stiff around its commanded positions. A lower current limit can
further reduce impact force.

These settings are live hardware settings: they last until power cycle,
``MidasHand.configure()``, or another script writes different gains.

Usage::

    python examples/set_finger_compliance.py
    python examples/set_finger_compliance.py --preset soft
    python examples/set_finger_compliance.py --motors fingers
    python examples/set_finger_compliance.py --motors curl --p-gain 300 --d-gain 120
    python examples/set_finger_compliance.py --no-hold-zero
    python examples/set_finger_compliance.py --preset default
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace

from midas_hand_api import DEFAULT_CONFIG_PATH, HandConfig, MidasHand
from midas_hand_api.actuators import control_table as ct


THUMB_MOTOR_IDS = (0, 1, 2, 3)
INDEX_MOTOR_IDS = (4, 5, 6)
MIDDLE_MOTOR_IDS = (7, 8, 9)
RING_MOTOR_IDS = (10, 11, 12)
FINGER_MOTOR_IDS = INDEX_MOTOR_IDS + MIDDLE_MOTOR_IDS + RING_MOTOR_IDS
HAND_MOTOR_IDS = THUMB_MOTOR_IDS + FINGER_MOTOR_IDS
CURL_MOTOR_IDS = (4, 5, 7, 8, 10, 11)

NAMED_MOTOR_GROUPS = {
    "all": HAND_MOTOR_IDS,
    "hand": HAND_MOTOR_IDS,
    "thumb": THUMB_MOTOR_IDS,
    "fingers": FINGER_MOTOR_IDS,
    "curl": CURL_MOTOR_IDS,
    "index": INDEX_MOTOR_IDS,
    "middle": MIDDLE_MOTOR_IDS,
    "ring": RING_MOTOR_IDS,
}


@dataclass(frozen=True)
class CompliancePreset:
    p_gain: int
    i_gain: int
    d_gain: int
    current_limit_ma: int | None


PRESETS = {
    # Good first try for contact: noticeably softer than the API defaults.
    "compliant": CompliancePreset(
        p_gain=400,
        i_gain=0,
        d_gain=180,
        current_limit_ma=450,
    ),
    # Softer, useful for impact testing or fragile objects.
    "soft": CompliancePreset(
        p_gain=250,
        i_gain=0,
        d_gain=100,
        current_limit_ma=350,
    ),
    # Matches HandConfig defaults.
    "default": CompliancePreset(
        p_gain=800,
        i_gain=0,
        d_gain=500,
        current_limit_ma=600,
    ),
    # Firmer than default, included mostly for quick comparison.
    "stiff": CompliancePreset(
        p_gain=1000,
        i_gain=0,
        d_gain=700,
        current_limit_ma=650,
    ),
}

DEFAULT_ZERO_TIMEOUT_S = 5.0
DEFAULT_ZERO_TOLERANCE_RAD = 0.08


def load_config(hand_port: str | None) -> HandConfig:
    config = (
        HandConfig.load(DEFAULT_CONFIG_PATH)
        if DEFAULT_CONFIG_PATH.exists()
        else HandConfig()
    )
    if hand_port:
        config = replace(config, port=hand_port)
    return config


def parse_motor_ids(spec: str) -> tuple[int, ...]:
    ids: list[int] = []
    for raw_part in spec.split(","):
        part = raw_part.strip().lower()
        if not part:
            continue
        if part in NAMED_MOTOR_GROUPS:
            ids.extend(NAMED_MOTOR_GROUPS[part])
        elif "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise argparse.ArgumentTypeError(
                    f"invalid motor range {part!r}: end is before start"
                )
            ids.extend(range(start, end + 1))
        else:
            ids.append(int(part))

    if not ids:
        raise argparse.ArgumentTypeError("at least one motor ID is required")
    return tuple(dict.fromkeys(ids))


def validate_gain(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    if value > 16383:
        raise ValueError(f"{name} is unusually high; expected 0..16383")


def validate_current_limit(value: int | None) -> None:
    if value is None:
        return
    if value < 0:
        raise ValueError("--current-limit-ma must be non-negative")
    if value > ct.XM335_T323_MAX_CURRENT_LIMIT:
        raise ValueError(
            "--current-limit-ma must be <= "
            f"{ct.XM335_T323_MAX_CURRENT_LIMIT} for XM335-T323-T"
        )


def resolve_settings(args: argparse.Namespace) -> CompliancePreset:
    preset = PRESETS[args.preset]
    current_limit_ma = None if args.no_current_limit else preset.current_limit_ma
    if args.current_limit_ma is not None:
        current_limit_ma = args.current_limit_ma

    p_gain = preset.p_gain if args.p_gain is None else args.p_gain
    i_gain = preset.i_gain if args.i_gain is None else args.i_gain
    d_gain = preset.d_gain if args.d_gain is None else args.d_gain

    validate_gain("--p-gain", p_gain)
    validate_gain("--i-gain", i_gain)
    validate_gain("--d-gain", d_gain)
    validate_current_limit(current_limit_ma)

    return CompliancePreset(
        p_gain=p_gain,
        i_gain=i_gain,
        d_gain=d_gain,
        current_limit_ma=current_limit_ma,
    )


def configured_motor_ids(
    hand: MidasHand,
    requested_ids: tuple[int, ...],
) -> tuple[int, ...]:
    missing = sorted(set(requested_ids) - set(hand.motor_ids))
    if missing:
        raise RuntimeError(f"Selected motor IDs are not in this hand config: {missing}")
    return tuple(motor_id for motor_id in requested_ids if motor_id in hand.motor_ids)


def responding_motor_ids(
    hand: MidasHand,
    requested_ids: tuple[int, ...],
    *,
    allow_missing: bool,
) -> tuple[int, ...]:
    responding = set(hand.ping())
    missing = sorted(set(requested_ids) - responding)
    if missing and not allow_missing:
        raise RuntimeError(
            "Selected motor IDs did not respond: "
            f"{missing}. Use --allow-missing to skip them."
        )
    return tuple(motor_id for motor_id in requested_ids if motor_id in responding)


def require_all_motors_for_hold(hand: MidasHand) -> None:
    responding = set(hand.ping())
    missing = sorted(set(hand.motor_ids) - responding)
    if missing:
        raise RuntimeError(
            "Cannot hold all joints at zero because these configured motor IDs "
            f"did not respond: {missing}. Check wiring or use --no-hold-zero."
        )


def apply_compliance_settings(
    hand: MidasHand,
    motor_ids: tuple[int, ...],
    settings: CompliancePreset,
) -> None:
    hand.set_gains(
        motor_ids=motor_ids,
        p=settings.p_gain,
        i=settings.i_gain,
        d=settings.d_gain,
    )
    if settings.current_limit_ma is not None:
        hand.set_current_limit(settings.current_limit_ma, motor_ids=motor_ids)


def command_zero_and_hold(
    hand: MidasHand,
    *,
    timeout_s: float,
    tolerance_rad: float,
):
    target = hand.clip_positions((0.0,) * len(hand.motor_ids))
    current_position = hand.read_pos()
    print("Moving all motors to software zero and enabling torque.")
    hand.set_positions(current_position, clip=True)
    hand.enable_torque()
    hand.set_positions(target, clip=False)
    if timeout_s > 0.0:
        reached = hand.wait_until_reached(
            target,
            tolerance_rad=tolerance_rad,
            timeout_s=timeout_s,
            poll_interval_s=0.02,
        )
        if not reached:
            print(f"  Warning: zero pose not fully reached within {timeout_s:.1f}s.")
    return target


def command_current_pose_and_hold(hand: MidasHand):
    current_position = hand.read_pos()
    print("Holding current motor positions and enabling torque.")
    hand.set_positions(current_position, clip=True)
    hand.enable_torque()
    hand.set_positions(current_position, clip=True)
    return hand.clip_positions(current_position)


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.zero_timeout < 0.0:
        parser.error("--zero-timeout must be non-negative")
    if args.zero_tolerance < 0.0:
        parser.error("--zero-tolerance must be non-negative")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--hand-port",
        default=None,
        help="Serial port for Dynamixel hand",
    )
    parser.add_argument(
        "--motors",
        type=parse_motor_ids,
        default=HAND_MOTOR_IDS,
        help=(
            "Motor group or IDs to tune. Default is all, including thumb. "
            "Named groups: all, hand, thumb, fingers, curl, index, middle, "
            "ring. Ranges like 4-12 also work."
        ),
    )
    parser.add_argument(
        "--preset",
        choices=tuple(PRESETS),
        default="compliant",
        help="Gain/current preset to start from",
    )
    parser.add_argument(
        "--p-gain",
        type=int,
        default=None,
        help="Override position P gain",
    )
    parser.add_argument(
        "--i-gain",
        type=int,
        default=None,
        help="Override position I gain",
    )
    parser.add_argument(
        "--d-gain",
        type=int,
        default=None,
        help="Override position D gain",
    )
    parser.add_argument(
        "--current-limit-ma",
        type=int,
        default=None,
        help="Override goal current limit in mA",
    )
    parser.add_argument(
        "--no-current-limit",
        action="store_true",
        help="Only write PID gains; leave current limit unchanged",
    )
    parser.add_argument(
        "--configure-first",
        action="store_true",
        help=(
            "Apply the normal HandConfig mode/gain/current setup before tuning. "
            "This is automatic when holding zero or current position."
        ),
    )
    hold_group = parser.add_mutually_exclusive_group()
    hold_group.add_argument(
        "--no-hold-zero",
        action="store_true",
        help="Only write settings; do not command zero or enable torque",
    )
    hold_group.add_argument(
        "--hold-current-position",
        action="store_true",
        help="Set the current pose as the goal instead of moving to software zero",
    )
    parser.add_argument(
        "--zero-timeout",
        type=float,
        default=DEFAULT_ZERO_TIMEOUT_S,
        help="Seconds to wait for the zero pose; 0 skips waiting",
    )
    parser.add_argument(
        "--zero-tolerance",
        type=float,
        default=DEFAULT_ZERO_TOLERANCE_RAD,
        help="Position tolerance for the zero pose in radians",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Skip selected motors that do not respond",
    )
    parser.add_argument(
        "--disable-torque-on-exit",
        action="store_true",
        help="Disable torque when the script exits, releasing the held pose",
    )
    args = parser.parse_args()
    validate_args(args, parser)

    try:
        settings = resolve_settings(args)
    except ValueError as exc:
        parser.error(str(exc))

    hand: MidasHand | None = None
    try:
        hand = MidasHand(load_config(args.hand_port))
        print(f"Hand connected on {hand.port}")

        requested_ids = configured_motor_ids(hand, args.motors)
        motor_ids = responding_motor_ids(
            hand,
            requested_ids,
            allow_missing=args.allow_missing,
        )
        if not motor_ids:
            raise RuntimeError("No selected motors responded.")

        holding_pose = not args.no_hold_zero or args.hold_current_position
        if holding_pose:
            require_all_motors_for_hold(hand)

        if args.configure_first or holding_pose:
            hand.configure(enable_torque=False)

        hold_target = None
        if args.hold_current_position:
            hold_target = command_current_pose_and_hold(hand)
        elif not args.no_hold_zero:
            hold_target = command_zero_and_hold(
                hand,
                timeout_s=args.zero_timeout,
                tolerance_rad=args.zero_tolerance,
            )

        apply_compliance_settings(hand, motor_ids, settings)
        if hold_target is not None:
            hand.set_positions(hold_target, clip=False)

        current_text = (
            "unchanged"
            if settings.current_limit_ma is None
            else f"{settings.current_limit_ma} mA"
        )
        print(
            "Applied compliance settings to motor IDs "
            f"{motor_ids}: P={settings.p_gain}, I={settings.i_gain}, "
            f"D={settings.d_gain}, current_limit={current_text}"
        )
        print(
            "Note: later calls to MidasHand.configure() will overwrite these "
            "live hardware settings."
        )
        if not args.disable_torque_on_exit and holding_pose:
            print("Torque remains enabled so the hand keeps holding the pose.")
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        if hand is not None:
            hand.close(disable_torque=args.disable_torque_on_exit)


if __name__ == "__main__":
    main()
