"""Fast phased curl motion for the three non-thumb fingers.

Only the six curl motors receive time-varying targets:

    index:  motor IDs 4, 5   (PIP, MCP flex)
    middle: motor IDs 7, 8   (PIP, MCP flex)
    ring:   motor IDs 10, 11 (PIP, MCP flex)

Thumb motors and finger MCP ab/ad motors are held at their initial readback
positions for the whole motion.

Usage::

    python examples/phased_finger_curl.py
    python examples/phased_finger_curl.py --hand-port /dev/ttyUSB0
    python examples/phased_finger_curl.py --frequency 2.0 --duration 20
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import replace

import numpy as np

from midas_hand_api import DEFAULT_CONFIG_PATH, HandConfig, MidasHand


FINGER_CURL_MOTORS = {
    "index": (4, 5),
    "middle": (7, 8),
    "ring": (10, 11),
}
CURL_MOTOR_IDS = tuple(
    motor_id
    for motor_ids in FINGER_CURL_MOTORS.values()
    for motor_id in motor_ids
)


def load_config(hand_port: str | None) -> HandConfig:
    cfg = (
        HandConfig.load(DEFAULT_CONFIG_PATH)
        if DEFAULT_CONFIG_PATH.exists()
        else HandConfig()
    )
    if hand_port:
        cfg = replace(cfg, port=hand_port)
    return cfg


def parse_phase_degrees(value: str) -> tuple[float, float, float]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) != len(FINGER_CURL_MOTORS):
        raise argparse.ArgumentTypeError(
            "expected three comma-separated phase offsets, e.g. 0,120,240"
        )
    try:
        index_phase, middle_phase, ring_phase = (
            math.radians(float(part)) for part in parts
        )
        return index_phase, middle_phase, ring_phase
    except ValueError as exc:
        raise argparse.ArgumentTypeError("phase offsets must be numbers") from exc


def validate_args(args: argparse.Namespace) -> None:
    if args.frequency <= 0.0:
        raise ValueError("--frequency must be positive")
    if args.rate_hz <= 0.0:
        raise ValueError("--rate-hz must be positive")
    if args.ramp_s < 0.0:
        raise ValueError("--ramp-s must be non-negative")
    if args.duration < 0.0:
        raise ValueError("--duration must be non-negative; use 0 to run forever")
    if args.profile_velocity < 0.0:
        raise ValueError("--profile-velocity must be non-negative")
    if args.profile_acceleration < 0.0:
        raise ValueError("--profile-acceleration must be non-negative")


def require_curl_motors(hand: MidasHand) -> None:
    missing = sorted(set(CURL_MOTOR_IDS) - set(hand.motor_ids))
    if missing:
        raise RuntimeError(
            f"Configured hand is missing required curl motor IDs: {missing}"
        )


def set_motor_targets(
    hand: MidasHand,
    target: np.ndarray,
    motor_ids: tuple[int, ...],
    values: tuple[float, ...] | np.ndarray,
) -> None:
    for motor_id, value in zip(motor_ids, values):
        target[hand.motor_ids.index(motor_id)] = value


def make_open_target(
    hand: MidasHand,
    base_pose: np.ndarray,
    open_pip: float,
    open_mcp: float,
) -> np.ndarray:
    target = base_pose.copy()
    for motor_ids in FINGER_CURL_MOTORS.values():
        set_motor_targets(hand, target, motor_ids, (open_pip, open_mcp))
    return target


def command_and_wait(
    hand: MidasHand,
    target: np.ndarray,
    label: str,
    timeout_s: float = 4.0,
) -> None:
    print(label)
    clipped_target = hand.clip_positions(target)
    hand.set_positions(clipped_target, clip=False)
    if not hand.wait_until_reached(
        clipped_target,
        tolerance_rad=0.08,
        velocity_threshold_rad_s=0.15,
        timeout_s=timeout_s,
        poll_interval_s=0.01,
    ):
        print(f"  Warning: target was not fully reached within {timeout_s:.1f}s.")


def phased_target(
    hand: MidasHand,
    base_pose: np.ndarray,
    elapsed_s: float,
    phases_rad: tuple[float, float, float],
    frequency_hz: float,
    ramp_s: float,
    open_values: np.ndarray,
    closed_values: np.ndarray,
) -> np.ndarray:
    target = base_pose.copy()
    envelope = 1.0 if ramp_s == 0.0 else min(1.0, elapsed_s / ramp_s)
    omega = 2.0 * math.pi * frequency_hz

    for motor_ids, phase in zip(FINGER_CURL_MOTORS.values(), phases_rad):
        curl = 0.5 * (1.0 - math.cos(omega * elapsed_s + phase))
        values = open_values + envelope * curl * (closed_values - open_values)
        set_motor_targets(hand, target, motor_ids, values)

    return target


def run_phased_curl(
    hand: MidasHand,
    args: argparse.Namespace,
) -> None:
    require_curl_motors(hand)

    base_pose = hand.read_pos()
    open_values = np.array([args.open_pip, args.open_mcp], dtype=np.float64)
    closed_values = np.array([args.closed_pip, args.closed_mcp], dtype=np.float64)
    open_target = make_open_target(hand, base_pose, args.open_pip, args.open_mcp)

    command_and_wait(hand, open_target, "Moving curl motors to the open pose.")
    base_pose = open_target

    print(
        "Curling index/middle/ring with phases "
        f"{args.phase_deg}; active motor IDs={CURL_MOTOR_IDS}; "
        "Ctrl-C to stop."
    )

    dt = 1.0 / args.rate_hz
    start = time.monotonic()
    next_tick = start

    try:
        while True:
            now = time.monotonic()
            elapsed_s = now - start
            if args.duration > 0.0 and elapsed_s >= args.duration:
                break

            target = phased_target(
                hand,
                base_pose,
                elapsed_s,
                args.phases_rad,
                args.frequency,
                args.ramp_s,
                open_values,
                closed_values,
            )
            hand.set_positions(target, clip=True)

            next_tick += dt
            sleep_s = next_tick - time.monotonic()
            if sleep_s > 0.0:
                time.sleep(sleep_s)
            else:
                next_tick = time.monotonic()
    finally:
        if not args.no_return_open:
            command_and_wait(hand, open_target, "Returning curl motors to the open pose.")


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
        "--duration",
        type=float,
        default=15.0,
        help="Motion duration in seconds; 0 runs forever",
    )
    parser.add_argument(
        "--frequency",
        type=float,
        default=1.5,
        help="Curl cycles per second",
    )
    parser.add_argument(
        "--rate-hz",
        type=float,
        default=100.0,
        help="Command update rate",
    )
    parser.add_argument(
        "--phase-deg",
        type=parse_phase_degrees,
        default=parse_phase_degrees("0,60,120"),
        help="Comma-separated phase offsets for index,middle,ring",
    )
    parser.add_argument(
        "--ramp-s",
        type=float,
        default=1.0,
        help="Seconds to ramp into the phased motion",
    )
    parser.add_argument(
        "--open-pip",
        type=float,
        default=0.0,
        help="Open PIP target in radians",
    )
    parser.add_argument(
        "--open-mcp",
        type=float,
        default=0.0,
        help="Open MCP flex target in radians",
    )
    parser.add_argument(
        "--closed-pip",
        type=float,
        default=-0.55,
        help="Closed PIP target in radians",
    )
    parser.add_argument(
        "--closed-mcp",
        type=float,
        default=-0.95,
        help="Closed MCP flex target in radians",
    )
    parser.add_argument(
        "--profile-velocity",
        type=float,
        default=8.0,
        help="Dynamixel profile velocity for curl motors in rad/s; 0 disables the limit",
    )
    parser.add_argument(
        "--profile-acceleration",
        type=float,
        default=120.0,
        help="Dynamixel profile acceleration for curl motors in rad/s^2; 0 disables the limit",
    )
    parser.add_argument(
        "--no-return-open",
        action="store_true",
        help="Leave curl motors at the last commanded phase target",
    )
    args = parser.parse_args()
    args.phases_rad = args.phase_deg
    args.phase_deg = ",".join(f"{math.degrees(phase):g}" for phase in args.phases_rad)
    try:
        validate_args(args)
    except ValueError as exc:
        parser.error(str(exc))

    hand: MidasHand | None = None
    try:
        hand = MidasHand(load_config(args.hand_port))
        print(f"Hand connected on {hand.port}")
        hand.configure(enable_torque=False)
        hand.enable_torque()
        hand.set_motion_profile(
            profile_velocity_rad_s=args.profile_velocity,
            profile_acceleration_rad_s2=args.profile_acceleration,
            motor_ids=CURL_MOTOR_IDS,
        )
        run_phased_curl(hand, args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        print("Disabling torque. Please wait...")
        if hand is not None:
            hand.shutdown()


if __name__ == "__main__":
    main()
