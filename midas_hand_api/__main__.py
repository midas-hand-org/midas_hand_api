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
from .config import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_DYNAMIXEL_BAUDRATE,
    DEFAULT_MOTOR_IDS,
)
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
    scan_config = HandConfig(
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
    disable_config = HandConfig(
        motor_ids=tuple(ids_to_disable),
        port=config.port,
        baudrate=config.baudrate,
    )
    with MidasHand(disable_config) as disable_hand:
        disable_hand.disable_torque()


def _has_nonzero_home_offsets(config: HandConfig) -> bool:
    return any(abs(offset) > 1e-9 for offset in config.home_offsets)


def main() -> None:
    parser = argparse.ArgumentParser(prog="midas-hand")
    parser.add_argument("--port", default=None, help="Serial port, e.g. /dev/ttyUSB0")
    parser.add_argument("--baudrate", type=int, default=None)
    parser.add_argument(
        "--motors",
        default=None,
        help="Comma-separated motor IDs. Defaults to 0-12.",
    )
    parser.add_argument("--configure", action="store_true", help="Apply PID/current settings")
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Print unhomed motor positions in radians",
    )
    parser.add_argument(
        "--joints",
        action="store_true",
        help="Print expanded joint positions, including passive DIP joints",
    )
    parser.add_argument(
        "--recalibrate",
        action="store_true",
        help="Recalibrate (re-zero) the Paxini tactile sensors and exit. "
        "Does not connect the Dynamixel bus or move any actuators.",
    )
    parser.add_argument(
        "--paxini-port",
        default=None,
        help="Paxini tactile board serial port (auto-detected if omitted)",
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
    default_baudrate = (
        args.baudrate if args.baudrate is not None else DEFAULT_DYNAMIXEL_BAUDRATE
    )

    if args.scan:
        scan_ids = _parse_id_spec(args.scan_range)
        ports = [args.port] if args.port else discover_ports()
        if not ports:
            ports = ["/dev/ttyUSB0"]
        baudrates = (
            [int(item) for item in args.scan_baudrates.split(",") if item]
            if args.scan_baudrates
            else [default_baudrate]
        )
        for port in ports:
            for baudrate in baudrates:
                try:
                    ping_result = _scan_raw(
                        port=port,
                        baudrate=baudrate,
                        motor_ids=scan_ids,
                        verbose=args.scan_verbose,
                    )
                except OSError as exc:
                    print(f"Scan baudrate {baudrate} on {port}: {exc}")
                    continue
                print(f"Scan baudrate {baudrate} on {port}:")
                print(f"  Responding : {sorted(ping_result.keys())}")
                if ping_result:
                    print(f"  Models     : {ping_result}")
        return

    if args.recalibrate:
        from .tactile import PaxiniConfig, PaxiniHandSensor

        sensor = PaxiniHandSensor(PaxiniConfig(port=args.paxini_port))
        try:
            sensor.connect()
            print(f"Paxini connected on {sensor.port}")
            print(
                "Recalibrating tactile sensors — make sure nothing is touching them..."
            )
            sensor.recalibrate()
            print("Tactile sensors recalibrated.")
        finally:
            sensor.disconnect()
        return

    homing_requested = args.home or args.home_thumb or args.home_fingers
    requested_motor_ids = (
        tuple(_parse_id_spec(args.motors)) if args.motors else DEFAULT_MOTOR_IDS
    )
    config_source = None

    if args.config:
        config = HandConfig.load(args.config)
        config_source = args.config
        if args.motors:
            config = config.subset(requested_motor_ids)
        config_updates = {}
        if args.port:
            config_updates["port"] = args.port
        if args.baudrate is not None:
            config_updates["baudrate"] = args.baudrate
        if config_updates:
            config = dataclasses.replace(config, **config_updates)
    elif homing_requested:
        config = HandConfig(
            motor_ids=requested_motor_ids, port=args.port, baudrate=default_baudrate
        )
    elif DEFAULT_CONFIG_PATH.exists():
        config = HandConfig.load(DEFAULT_CONFIG_PATH)
        config_source = str(DEFAULT_CONFIG_PATH)
        if args.motors:
            config = config.subset(requested_motor_ids)
        # The saved config owns calibration, not the live USB enumeration. Unless
        # the user explicitly pins --port, re-probe and pick the responding bus.
        config_updates = {"port": args.port}
        if args.baudrate is not None:
            config_updates["baudrate"] = args.baudrate
        config = dataclasses.replace(config, **config_updates)
    else:
        config = HandConfig(
            motor_ids=requested_motor_ids, port=args.port, baudrate=default_baudrate
        )

    if config_source:
        suffix = " (raw reads bypass home_offsets)" if args.raw else ""
        print(f"Loaded config: {config_source}{suffix}")
    elif not homing_requested:
        print(
            f"No saved config at {DEFAULT_CONFIG_PATH}; "
            "using default zero home_offsets."
        )

    if not args.raw and not homing_requested and not _has_nonzero_home_offsets(config):
        print("home_offsets are all zero; calibrated positions will match --raw.")

    # Discovery: ping all configured motors, suppress SDK packet-level noise
    try:
        logging.disable(logging.ERROR)
        with MidasHand(config) as hand:
            ping_result = hand.ping()
            config = hand.config
    except OSError as exc:
        logging.disable(logging.NOTSET)
        print(f"Could not connect to the Dynamixel bus: {exc}")
        return
    finally:
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
                    print(f"Motor position (raw, 13): {hand._client().read_pos()}")
                elif args.joints:
                    print(f"Joint position (16): {hand.read_joint_pos()}")
                else:
                    print(f"Motor position (13): {hand.read_pos()}")
                time.sleep(0.03)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
