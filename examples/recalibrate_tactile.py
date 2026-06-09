"""Move all fingers to zero, then recalibrate the Paxini tactile sensors.

Run this as a warm-up before any tactile script: it opens the hand flat (the
no-load pose) and re-zeros every connected sensor's distributed-force baseline
so each session starts from a clean zero.

Usage::

    python examples/recalibrate_tactile.py
    python examples/recalibrate_tactile.py --hand-port /dev/ttyUSB0 --paxini-port /dev/ttyACM0

Keep the fingertips clear while it runs — whatever the sensors touch at
recalibration time becomes the new zero.
"""

import argparse

import numpy as np

from midas_hand_api import (
    DEFAULT_CONFIG_PATH,
    HandConfig,
    MidasHand,
    PaxiniConfig,
    PaxiniHandSensor,
)


def load_config(hand_port: str | None) -> HandConfig:
    cfg = HandConfig.load(DEFAULT_CONFIG_PATH) if DEFAULT_CONFIG_PATH.exists() else HandConfig()
    if hand_port:
        from dataclasses import replace
        cfg = replace(cfg, port=hand_port)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--hand-port", default=None, help="Dynamixel hand port (auto-detected if omitted)")
    parser.add_argument("--paxini-port", default=None, help="Paxini board serial port (auto-detected if omitted)")
    args = parser.parse_args()

    tactile = PaxiniHandSensor(PaxiniConfig(port=args.paxini_port))

    hand: MidasHand | None = None
    try:
        hand = MidasHand(load_config(args.hand_port), tactile_sensor=tactile)
        print(f"Hand connected on {hand.port}")
        print(f"Paxini connected on {tactile.port}")

        hand.configure(enable_torque=False)
        hand.enable_torque()

        print("Moving all fingers to zero...")
        hand.set_motion_profile(
            profile_velocity_rad_s=3.5,
            profile_acceleration_rad_s2=30.0,
            motor_ids=hand.motor_ids,
        )
        zero = np.zeros(len(hand.motor_ids))
        hand.set_positions(zero, clip=True)
        if not hand.wait_until_reached(
            zero,
            tolerance_rad=0.1,
            velocity_threshold_rad_s=0.05,
            timeout_s=6.0,
            poll_interval_s=0.01,
        ):
            raise TimeoutError("Hand did not reach zero within 6.00s")
        print(f"Motor positions rad: {hand.read_pos()}")

        print("Recalibrating tactile sensors — make sure nothing is touching them...")
        hand.recalibrate_tactile()
        print("Tactile sensors recalibrated.")

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        print("Disabling torque. Please wait...")
        if hand is not None:
            hand.shutdown()


if __name__ == "__main__":
    main()
