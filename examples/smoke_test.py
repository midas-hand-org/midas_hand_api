"""Hardware smoke test for one or more Dynamixel motors."""

import argparse

from midas_hand_api import HandConfig, MidasHand, discover_ports


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=None)
    parser.add_argument("--baudrate", type=int, default=1_000_000)
    parser.add_argument("--motors", default="0,1,2,3,4,5,6,7,8,9,10,11,12")
    args = parser.parse_args()

    print(f"Candidate ports: {discover_ports()}")
    motor_ids = tuple(int(item) for item in args.motors.split(",") if item)
    config = HandConfig.xm335_t323(
        motor_ids=motor_ids, port=args.port, baudrate=args.baudrate
    )

    with MidasHand(config) as hand:
        print(f"Connected on: {hand.port}")
        print(f"Ping result: {hand.ping()}")
        print(f"Unexpected model numbers: {hand.verify_models()}")
        print(f"Positions rad: {hand.read_pos()}")


if __name__ == "__main__":
    main()
