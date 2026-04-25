"""CLI entry point for the Midas hand API.

Installed as the ``midas-hand`` command via pyproject.toml.
Also runnable directly with ``python -m midas_hand_api``.
"""

import argparse
import logging
import time

from . import HandConfig, MidasHand


def main() -> None:
    parser = argparse.ArgumentParser(prog="midas-hand")
    parser.add_argument("--port", default=None, help="Serial port, e.g. /dev/ttyUSB0")
    parser.add_argument("--baudrate", type=int, default=1_000_000)
    parser.add_argument(
        "--motors",
        default="0,1,2,3,4,5,6,7,8,9,10,11,12",
        help="Comma-separated motor IDs",
    )
    parser.add_argument("--configure", action="store_true", help="Apply PID/current settings")
    parser.add_argument("--config", default=None, help="Path to a saved config YAML")
    args = parser.parse_args()

    if args.config:
        config = HandConfig.load(args.config)
    else:
        motor_ids = tuple(int(item) for item in args.motors.split(",") if item)
        config = HandConfig.xm335_t323(
            motor_ids=motor_ids, port=args.port, baudrate=args.baudrate
        )

    # Discovery: ping all configured motors, suppress SDK packet-level noise
    logging.disable(logging.ERROR)
    with MidasHand(config) as hand:
        ping_result = hand.ping()
    logging.disable(logging.NOTSET)

    all_ids = sorted(config.motor_ids)
    responding_ids = sorted(ping_result.keys())
    missing_ids = sorted(set(all_ids) - set(responding_ids))

    print(f"Motor scan ({len(all_ids)} configured):")
    print(f"  Responding : {responding_ids}")
    if missing_ids:
        print(f"  Not found  : {missing_ids}  <-- check wiring/IDs")

    if not responding_ids:
        print("No motors found. Check connections and motor IDs.")
        return

    if missing_ids:
        config = HandConfig.xm335_t323(
            motor_ids=tuple(responding_ids),
            port=config.port,
            baudrate=config.baudrate,
        )

    with MidasHand(config) as hand:
        print(f"Connected on {hand.port}")
        unexpected = hand.verify_models()
        if unexpected:
            print(f"Unexpected model numbers: {unexpected}")
        if args.configure:
            hand.configure(enable_torque=True)

        try:
            while True:
                print(f"Position: {hand.read_pos()}")
                time.sleep(0.03)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
