"""Finger-touch sequence with live Paxini tactile feedback.

Moves each finger to the Kapandji target position while placing the thumb at
the corresponding recorded touch waypoint. A Paxini tactile sensor is required;
force data is read through ``MidasHand.read_tactile()`` and visualized live in a
browser tab by default.

Usage::

    python examples/finger_touch.py --paxini-port /dev/ttyACM0
    python examples/finger_touch.py --paxini-port /dev/ttyACM0 --hand-port /dev/ttyUSB0

Use ``--no-viz`` to skip the browser visualizer (e.g. no display available).
"""

import argparse
import time

import numpy as np

from midas_hand_api import DEFAULT_CONFIG_PATH, HandConfig, MidasHand
from midas_hand_api.tactile import PaxiniConfig, PaxiniHandSensor
from midas_hand_api.tactile.paxini_visualizer import PaxiniVisualizer


THUMB_IDS = (0, 1, 2, 3)
FINGER_IDS = {
    "pointer": (4, 5, 6),
    "middle": (7, 8, 9),
    "ring": (10, 11, 12),
}

FINGER_TARGET_RAD = np.array([-0.51388356, -0.95322344, 0.0])

# Recorded thumb positions for each finger touch.
THUMB_TOUCH = {
    "pointer": np.array([-0.2408, -0.948, 0.1902, 0.8314]),
    "middle":  np.array([0.1273, -0.9434, 0.5016, 1.5585]),
    "ring":    np.array([0.3068, -0.7102, 0.6688, 2.0264]),
}

SEQUENCE = ["pointer", "middle", "ring"]


def load_config(hand_port: str | None) -> HandConfig:
    cfg = HandConfig.load(DEFAULT_CONFIG_PATH) if DEFAULT_CONFIG_PATH.exists() else HandConfig.xm335_t323()
    if hand_port:
        from dataclasses import replace
        cfg = replace(cfg, port=hand_port)
    return cfg


def set_targets(
    target: np.ndarray,
    hand: MidasHand,
    motor_ids: tuple[int, ...],
    values: np.ndarray,
) -> None:
    for mid, val in zip(motor_ids, values):
        target[hand.motor_ids.index(mid)] = val


def tactile_summary(data: dict[str, np.ndarray]) -> str:
    """Return a compact max-force summary for terminal feedback."""
    parts = []
    for name, vectors in data.items():
        max_force = np.linalg.norm(vectors, axis=1).max(initial=0.0)
        max_fz = vectors[:, 2].max(initial=0.0)
        parts.append(f"{name}: |F|max={max_force:.2f} N, Fzmax={max_fz:.2f} N")
    return "  |  ".join(parts)


def print_tactile(hand: MidasHand, prefix: str) -> None:
    try:
        print(f"    {prefix}: {tactile_summary(hand.read_tactile())}")
    except RuntimeError as exc:
        print(f"    {prefix}: waiting for data ({exc})")


def move_and_wait(hand: MidasHand, target: np.ndarray, label: str, hold_s: float = 2.0) -> None:
    print(f"  -> {label}")
    hand.set_positions(target, clip=True)
    deadline = time.monotonic() + 6.0
    reached = False
    next_tactile_print = 0.0

    while time.monotonic() < deadline:
        pos, vel, _cur = hand.read_pos_vel_cur()
        position_reached = np.all(np.abs(target - pos) <= 0.1)
        velocity_settled = np.all(np.abs(vel) <= 0.1)
        if position_reached and velocity_settled:
            reached = True
            break

        if time.monotonic() >= next_tactile_print:
            print_tactile(hand, "tactile")
            next_tactile_print = time.monotonic() + 0.5

        time.sleep(0.02)

    if not reached:
        print("    Warning: target not reached within 6 s.")
    print_tactile(hand, "final tactile")
    if hold_s > 0:
        time.sleep(hold_s)


def run_sequence(hand: MidasHand) -> None:
    target = np.zeros(len(hand.motor_ids))
    prev_finger = None

    for finger_name in SEQUENCE:
        finger_ids = FINGER_IDS[finger_name]

        # Step 1: release previous finger and move thumb to touch position.
        if prev_finger is not None:
            set_targets(target, hand, FINGER_IDS[prev_finger], np.zeros(3))
        set_targets(target, hand, THUMB_IDS, THUMB_TOUCH[finger_name])
        move_and_wait(hand, target, f"thumb → {np.round(THUMB_TOUCH[finger_name], 4)}", hold_s=0.0)

        # Step 2: bring finger to target.
        set_targets(target, hand, finger_ids, FINGER_TARGET_RAD)
        move_and_wait(hand, target, f"{finger_name} → target")

        prev_finger = finger_name

    print("Returning to zero.")
    move_and_wait(hand, np.zeros(len(hand.motor_ids)), "all → zero")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--paxini-port", default="/dev/ttyACM0", help="Serial port for Paxini sensor, e.g. /dev/ttyUSB1")
    parser.add_argument("--hand-port", default=None, help="Serial port for Dynamixel hand (auto-detected if omitted)")
    parser.add_argument("--no-viz", action="store_true", help="Skip the browser visualizer")
    parser.add_argument("--viz-port", type=int, default=8050, help="Dash server port (default 8050)")
    args = parser.parse_args()

    sensor = PaxiniHandSensor(PaxiniConfig(port=args.paxini_port))
    sensor.connect()
    hand: MidasHand | None = None
    try:
        hand = MidasHand(load_config(args.hand_port), tactile_sensor=sensor)
        print(f"Hand connected on {hand.port}")
        print(f"Paxini connected on {args.paxini_port}")

        if not args.no_viz:
            PaxiniVisualizer(hand.read_tactile).start_background(port=args.viz_port)

        hand.configure(enable_torque=False)

        hand.enable_torque()

        hand.set_motion_profile(
            profile_velocity_rad_s=1.5,
            profile_acceleration_raw=300,
            motor_ids=hand.motor_ids,
        )

        run_sequence(hand)

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        print("Disabling torque. Please wait...")
        if hand is not None:
            hand.shutdown()
        else:
            sensor.disconnect()


if __name__ == "__main__":
    main()
