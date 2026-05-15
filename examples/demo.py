"""Full demo: finger-touch sequence followed by Kapandji sweep.

Usage::

    python examples/demo.py --paxini-port /dev/ttyACM0
    python examples/demo.py --paxini-port /dev/ttyACM0 --hand-port /dev/ttyUSB0
    python examples/demo.py --paxini-port /dev/ttyACM0 --no-viz
"""

import argparse
import time

import numpy as np

from midas_hand_api import DEFAULT_CONFIG_PATH, HandConfig, MidasHand
from midas_hand_api.tactile import PaxiniConfig, PaxiniHandSensor
from midas_hand_api.tactile.paxini_visualizer import PaxiniVisualizer


# ---------------------------------------------------------------------------
# Motor layout
# ---------------------------------------------------------------------------

THUMB_IDS = (0, 1, 2, 3)
FINGER_IDS = {
    "pointer": (4, 5, 6),
    "middle":  (7, 8, 9),
    "ring":    (10, 11, 12),
}

# ---------------------------------------------------------------------------
# Finger-touch data
# ---------------------------------------------------------------------------

FINGER_TARGET_RAD = np.array([-0.51388356, -0.95322344, 0.0])

THUMB_TOUCH = {
    "pointer": np.array([-0.2408, -0.948,   0.1902, 0.8314]),
    "middle":  np.array([ 0.1273, -0.9434,  0.5016, 1.5585]),
    "ring":    np.array([ 0.4679, -0.9787, 0.6504, 1.919 ]),
}

# ---------------------------------------------------------------------------
# Kapandji data  (finger name → index finger uses key "index")
# ---------------------------------------------------------------------------

KAPANDJI_FINGER_IDS = {
    "index":  (4, 5, 6),
    "middle": (7, 8, 9),
    "ring":   (10, 11, 12),
}

THUMB_TARGETS_RAD = {
    "index": np.array([
        [-1.18576715, -0.01227184, -0.32060198, 0.59211658],
        [-0.7823302,  -0.13805827,  0.06902914, 1.14588365],
        [-0.96333994, -0.40957287,  0.42491268, 1.10139821],
    ]),
    "middle": np.array([
        [-0.21629129, -0.69489329, -0.31293208, 1.11367005],
        [-0.18100973, -0.65807776,  0.11965051, 1.40972834],
        [-0.1043107,  -0.77772826,  0.48166997, 1.44040796],
    ]),
    "ring": np.array([
        [-0.06902914, -0.89124284,  0.19634955, 1.98957308],
        [-0.55223309, -0.41570879,  0.35434957, 1.93895172],
        [-0.34667966, -0.61052435,  0.70869913, 1.8208352 ],
    ]),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
    parts = []
    for name, vectors in data.items():
        max_force = np.linalg.norm(vectors, axis=1).max(initial=0.0)
        max_fz = vectors[:, 2].max(initial=0.0)
        parts.append(f"{name}: |F|max={max_force:.2f} N  Fzmax={max_fz:.2f} N")
    return "  |  ".join(parts)


def print_tactile(hand: MidasHand, prefix: str) -> None:
    try:
        print(f"    {prefix}: {tactile_summary(hand.read_tactile())}")
    except RuntimeError:
        pass


def move_and_wait(
    hand: MidasHand,
    target: np.ndarray,
    label: str,
    hold_s: float = 2.0,
    timeout_s: float = 6.0,
    tolerance: float = 0.1,
) -> None:
    print(f"  -> {label}")
    hand.set_positions(target, clip=True)
    deadline = time.monotonic() + timeout_s
    reached = False
    next_print = 0.0

    while time.monotonic() < deadline:
        pos, vel, _ = hand.read_pos_vel_cur()
        if np.all(np.abs(target - pos) <= tolerance) and np.all(np.abs(vel) <= tolerance):
            reached = True
            break
        if time.monotonic() >= next_print:
            print_tactile(hand, "tactile")
            next_print = time.monotonic() + 0.5
        time.sleep(0.02)

    if not reached:
        print(f"    Warning: target not reached within {timeout_s:.0f} s.")
    print_tactile(hand, "final tactile")
    if hold_s > 0:
        time.sleep(hold_s)


# ---------------------------------------------------------------------------
# Finger-touch sequence
# ---------------------------------------------------------------------------

def run_finger_touch(hand: MidasHand) -> None:
    print("\n=== Finger-touch sequence ===")
    target = np.zeros(len(hand.motor_ids))
    prev_finger = None

    for finger_name in ("pointer", "middle", "ring"):
        if prev_finger is not None:
            set_targets(target, hand, FINGER_IDS[prev_finger], np.zeros(3))
        set_targets(target, hand, THUMB_IDS, THUMB_TOUCH[finger_name])
        move_and_wait(hand, target, f"thumb → {np.round(THUMB_TOUCH[finger_name], 4)}", hold_s=0.0)

        set_targets(target, hand, FINGER_IDS[finger_name], FINGER_TARGET_RAD)
        move_and_wait(hand, target, f"{finger_name} → target")

        prev_finger = finger_name

    print("Returning to zero.")
    move_and_wait(hand, np.zeros(len(hand.motor_ids)), "all → zero", hold_s=1.0)


# ---------------------------------------------------------------------------
# Kapandji sequence
# ---------------------------------------------------------------------------

def run_kapandji(hand: MidasHand) -> None:
    print("\n=== Kapandji sequence ===")
    target = np.zeros(len(hand.motor_ids))
    prev_finger = None

    for finger_name in ("index", "middle", "ring"):
        finger_ids = KAPANDJI_FINGER_IDS[finger_name]

        if prev_finger is not None:
            set_targets(target, hand, KAPANDJI_FINGER_IDS[prev_finger], np.zeros(3))

        set_targets(target, hand, finger_ids, FINGER_TARGET_RAD)
        for wp_idx, thumb_target in enumerate(THUMB_TARGETS_RAD[finger_name]):
            set_targets(target, hand, THUMB_IDS, thumb_target)
            label = (
                f"{finger_name} to Kapandji + thumb waypoint 1"
                if wp_idx == 0
                else f"thumb waypoint {wp_idx + 1} for {finger_name}"
            )
            move_and_wait(hand, target, label, hold_s=0.1, tolerance=0.05)

        prev_finger = finger_name

    print("Returning to zero.")
    move_and_wait(hand, np.zeros(len(hand.motor_ids)), "all → zero", hold_s=0.0)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--paxini-port", default="/dev/ttyACM0", help="Paxini serial port")
    parser.add_argument("--hand-port", default=None, help="Dynamixel hand port (auto-detected if omitted)")
    parser.add_argument("--no-viz", action="store_true", help="Skip the browser visualizer")
    parser.add_argument("--viz-port", type=int, default=8050)
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

        time.sleep(1.0)

        run_finger_touch(hand)
        run_kapandji(hand)

        time.sleep(3.0)

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
