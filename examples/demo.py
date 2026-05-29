"""Full tactile demo: finger-touch sequence followed by Kapandji sweep.

Usage::

    python examples/demo.py
    python examples/demo.py --hand-port /dev/ttyUSB0
    python examples/demo.py --paxini-port /dev/ttyACM0

For local Qt tactile visualization, run
``python -m midas_hand_api.tactile.paxini_tactile_qt`` in a separate terminal.
"""

import argparse
import time

import numpy as np

import finger_touch
import kapandji_test as kapandji
from midas_hand_api import DEFAULT_CONFIG_PATH, HandConfig, MidasHand
from midas_hand_api.tactile import PaxiniConfig, PaxiniHandSensor


def load_config(hand_port: str | None) -> HandConfig:
    cfg = HandConfig.load(DEFAULT_CONFIG_PATH) if DEFAULT_CONFIG_PATH.exists() else HandConfig()
    if hand_port:
        from dataclasses import replace
        cfg = replace(cfg, port=hand_port)
    return cfg


def run_kapandji(hand: MidasHand) -> None:
    print("\n=== Kapandji sequence ===")
    target = np.zeros(len(hand.motor_ids))
    previous_finger_name = None

    for finger_name in ("index", "middle", "ring"):
        kapandji.run_finger_sequence(
            hand,
            finger_name,
            target,
            previous_finger_name,
        )
        previous_finger_name = finger_name

    print("Returning to zero.")
    kapandji.command_and_wait(hand, np.zeros(len(hand.motor_ids)))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--paxini-port", default=None, help="Paxini serial port (auto-detected if omitted)")
    parser.add_argument("--hand-port", default=None, help="Dynamixel hand port (auto-detected if omitted)")
    args = parser.parse_args()

    sensor = PaxiniHandSensor(PaxiniConfig(port=args.paxini_port))  # port=None → auto-detect
    sensor.connect()
    hand: MidasHand | None = None
    try:
        hand = MidasHand(load_config(args.hand_port), tactile_sensor=sensor)
        print(f"Hand connected on {hand.port}")
        print(f"Paxini connected on {sensor.port}")

        hand.configure(enable_torque=False)
        hand.enable_torque()
        hand.set_motion_profile(
            profile_velocity_rad_s=1.5,
            profile_acceleration_rad_s2=50.0,
            motor_ids=hand.motor_ids,
        )

        time.sleep(1.0)

        print("\n=== Finger-touch sequence ===")
        finger_touch.run_sequence(hand)

        hand.set_motion_profile(
            profile_velocity_rad_s=3.0,
            profile_acceleration_rad_s2=20.0,
            motor_ids=hand.motor_ids,
        )

        run_kapandji(hand)

        time.sleep(100.0)

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
