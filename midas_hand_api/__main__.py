"""CLI entry point for the Midas hand API.

Installed as the ``midas-hand`` command via pyproject.toml.
Also runnable directly with ``python -m midas_hand_api``.
"""

import argparse
import dataclasses
import logging
import time

from . import HandConfig, MidasHand
from .actuators import control_table as ct, discover_ports
from .config import DEFAULT_CONFIG_PATH, DEFAULT_MOTOR_IDS
from .homing import home_fingers, home_hand, home_thumb


def _parse_id_spec(spec: str) -> list[int]:
    ids = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            ids.extend(range(int(start), int(end) + 1))
        else:
            ids.append(int(part))
    return ids


def _scan_raw(
    port: str,
    baudrate: int,
    motor_ids: list[int],
    protocol: float = ct.PROTOCOL_VERSION,
    verbose: bool = False,
) -> dict[int, int]:
    import dynamixel_sdk

    port_handler = dynamixel_sdk.PortHandler(port)
    packet_handler = dynamixel_sdk.PacketHandler(protocol)
    if not port_handler.openPort():
        raise OSError(f"Failed to open {port}")
    try:
        if not port_handler.setBaudRate(baudrate):
            raise OSError(f"Failed to set baudrate {baudrate} on {port}")

        found = {}
        for motor_id in motor_ids:
            model, comm_result, dxl_error = packet_handler.ping(
                port_handler, int(motor_id)
            )
            if comm_result == dynamixel_sdk.COMM_SUCCESS:
                found[int(motor_id)] = int(model)
                if verbose:
                    if dxl_error:
                        error = packet_handler.getRxPacketError(dxl_error)
                        print(f"    ID {motor_id}: model {model}, warning: {error}")
                    else:
                        print(f"    ID {motor_id}: model {model}")
            elif verbose:
                error = packet_handler.getTxRxResult(comm_result)
                print(f"    ID {motor_id}: {error}")
        return found
    finally:
        port_handler.closePort()


def _disable_unselected_motors(config: HandConfig, selected_ids: set[int]) -> None:
    scan_config = HandConfig.xm335_t323(
        motor_ids=DEFAULT_MOTOR_IDS,
        port=config.port,
        baudrate=config.baudrate,
    )
    logging.disable(logging.ERROR)
    try:
        with MidasHand(scan_config) as scan_hand:
            connected_ids = set(scan_hand.ping())
    finally:
        logging.disable(logging.NOTSET)

    ids_to_disable = sorted(connected_ids - selected_ids)
    if not ids_to_disable:
        return

    print(f"Disabling torque on unselected connected motors: {ids_to_disable}")
    disable_config = HandConfig.xm335_t323(
        motor_ids=tuple(ids_to_disable),
        port=config.port,
        baudrate=config.baudrate,
    )
    with MidasHand(disable_config) as disable_hand:
        disable_hand.disable_torque()


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
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Print unhomed motor positions in radians",
    )
    parser.add_argument("--home", action="store_true", help="Run full hand homing sequence and save config")
    parser.add_argument("--home-thumb", action="store_true", help="Run thumb homing sequence and save config")
    parser.add_argument("--home-fingers", action="store_true", help="Run finger homing sequence and save config")
    parser.add_argument("--config", default=None, help="Path to a saved config YAML")
    parser.add_argument(
        "--scan",
        action="store_true",
        help="Scan a range of motor IDs and exit",
    )
    parser.add_argument(
        "--scan-range",
        default="0-20",
        help="Motor ID range for --scan, e.g. 0-20 or 0,1,2,10",
    )
    parser.add_argument(
        "--scan-baudrates",
        default=None,
        help="Comma-separated baudrates for --scan. Defaults to --baudrate only.",
    )
    parser.add_argument(
        "--scan-verbose",
        action="store_true",
        help="Print per-ID scan errors",
    )
    args = parser.parse_args()

    if args.scan:
        scan_ids = _parse_id_spec(args.scan_range)
        ports = discover_ports()
        port = args.port or (ports[0] if ports else "/dev/ttyUSB0")
        baudrates = (
            [int(item) for item in args.scan_baudrates.split(",") if item]
            if args.scan_baudrates
            else [args.baudrate]
        )
        for baudrate in baudrates:
            ping_result = _scan_raw(
                port=port,
                baudrate=baudrate,
                motor_ids=scan_ids,
                verbose=args.scan_verbose,
            )
            print(f"Scan baudrate {baudrate} on {port}:")
            print(f"  Responding : {sorted(ping_result.keys())}")
            if ping_result:
                print(f"  Models     : {ping_result}")
        return

    if args.config:
        config = HandConfig.load(args.config)
    elif DEFAULT_CONFIG_PATH.exists():
        config = HandConfig.load(DEFAULT_CONFIG_PATH)
        if args.port:
            config = dataclasses.replace(config, port=args.port)
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
        config = config.subset(responding_ids)

    with MidasHand(config) as hand:
        print(f"Connected on {hand.port}")
        unexpected = hand.verify_models()
        if unexpected:
            print(f"Unexpected model numbers: {unexpected}")
        hardware_errors = {
            motor_id: status
            for motor_id, status in hand.read_hardware_error_status().items()
            if status
        }
        if hardware_errors:
            print(f"Hardware Error Status(70): {hardware_errors}")
        if args.configure:
            hand.configure(enable_torque=True)

        if args.home or args.home_thumb or args.home_fingers:
            _disable_unselected_motors(config, set(config.motor_ids))
            hand.configure(enable_torque=True)
            if args.home:
                home_hand(hand)
            elif args.home_thumb:
                home_thumb(hand)
            else:
                home_fingers(hand)
            return

        try:
            while True:
                if args.raw:
                    print(f"Position (raw): {hand._client().read_pos()}")
                else:
                    print(f"Position: {hand.read_pos()}")
                time.sleep(0.03)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
