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
    config = HandConfig(motor_ids=motor_ids, port=args.port, baudrate=args.baudrate)

    with MidasHand(config) as hand:
        print(f"Connected on: {hand.port}")
        ping_result = hand.ping()
        print(f"Ping result: {ping_result}")

    if not ping_result:
        print("No motors responded.")
        return

    responding_ids = tuple(sorted(ping_result.keys()))
    if responding_ids != motor_ids:
        print(f"Re-running read checks with responding IDs only: {responding_ids}")
        config = config.subset(responding_ids)

    with MidasHand(config) as hand:
        print(f"Unexpected model numbers: {hand.verify_models()}")
        hardware_errors = {
            motor_id: status
            for motor_id, status in hand.read_hardware_error_status().items()
            if status
        }
        if hardware_errors:
            print(f"Hardware Error Status(70): {hardware_errors}")
        print(f"Positions rad: {hand.read_pos()}")


if __name__ == "__main__":
    main()
