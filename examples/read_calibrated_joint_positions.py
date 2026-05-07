"""Print calibrated MIDAS hand joint positions after homing offsets.

This script intentionally uses ``MidasHand.read_pos()`` and
``MidasHand.read_joint_pos()`` instead of raw Dynamixel reads. That means the
reported values are software joint positions:

    calibrated = (raw_motor_position - home_offset) * joint_sign

By default it loads ``~/.midas_hand/config.yaml`` so the saved homing offsets
are applied.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

import numpy as np

# Allow running this file directly from a source checkout without installing.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from midas_hand_api import DEFAULT_CONFIG_PATH, HandConfig, MidasHand  # noqa: E402
from midas_hand_api.config import DEFAULT_MOTOR_IDS  # noqa: E402


# 13 active motor positions, ordered to match HandConfig.DEFAULT_MOTOR_IDS.
ACTIVE_MOTOR_JOINT_NAMES = (
    "thumb_dip_joint",
    "thumb_mcp_joint",
    "thumb_cmc_side_joint",
    "thumb_cmc_roll_joint",
    "index_pip_joint",
    "index_mcp_pitch_joint",
    "index_mcp_abad_joint",
    "middle_pip_joint",
    "middle_mcp_pitch_joint",
    "middle_mcp_abad_joint",
    "ring_pip_joint",
    "ring_mcp_pitch_joint",
    "ring_mcp_abad_joint",
)


def parse_id_spec(spec: str) -> tuple[int, ...]:
    ids: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            ids.extend(range(int(start), int(end) + 1))
        else:
            ids.append(int(part))
    return tuple(ids)


def load_config(args: argparse.Namespace) -> tuple[HandConfig, str]:
    if args.config:
        config = HandConfig.load(args.config)
        source = args.config
    elif DEFAULT_CONFIG_PATH.exists():
        config = HandConfig.load(DEFAULT_CONFIG_PATH)
        source = str(DEFAULT_CONFIG_PATH)
    else:
        config = HandConfig.xm335_t323()
        source = "code defaults; no saved homing config found"

    if args.motors:
        config = config.subset(parse_id_spec(args.motors))

    updates = {}
    if args.port:
        updates["port"] = args.port
    if args.baudrate is not None:
        updates["baudrate"] = args.baudrate
    if updates:
        config = dataclasses.replace(config, **updates)
    return config, source


def active_joint_name_for_motor(motor_id: int) -> str:
    if motor_id in DEFAULT_MOTOR_IDS:
        return ACTIVE_MOTOR_JOINT_NAMES[DEFAULT_MOTOR_IDS.index(motor_id)]
    return f"motor_{motor_id}_joint"


def active_joint_names(config: HandConfig) -> list[str]:
    return [active_joint_name_for_motor(motor_id) for motor_id in config.motor_ids]


def passive_joint_name(driver_name: str) -> str:
    if driver_name.endswith("_pip_joint"):
        return driver_name.removesuffix("_pip_joint") + "_dip_joint"
    return driver_name + "_passive"


def expanded_joint_names_and_sources(config: HandConfig) -> tuple[list[str], list[str]]:
    active_names = active_joint_names(config)
    passive_by_dof = dict(zip(config.passive_joint_indices, config.pip_motor_indices))

    names: list[str] = []
    sources: list[str] = []
    motor_index = 0
    for dof_index in range(config.n_dof):
        if dof_index in passive_by_dof:
            pip_motor_index = passive_by_dof[dof_index]
            driver_name = active_names[pip_motor_index]
            driver_id = config.motor_ids[pip_motor_index]
            names.append(passive_joint_name(driver_name))
            sources.append(f"passive<-motor {driver_id}")
        else:
            motor_id = config.motor_ids[motor_index]
            names.append(active_names[motor_index])
            sources.append(f"motor {motor_id}")
            motor_index += 1
    return names, sources


def print_table(
    title: str,
    names: list[str],
    values: np.ndarray,
    sources: list[str],
) -> None:
    print(title)
    print(f"{'idx':>3}  {'joint':<28}  {'source':<16}  {'rad':>10}  {'deg':>10}")
    for index, (name, value, source) in enumerate(zip(names, values, sources)):
        print(
            f"{index:>3}  {name:<28}  {source:<16}  "
            f"{float(value):>+10.4f}  {np.rad2deg(float(value)):>+10.2f}"
        )


def print_positions(hand: MidasHand, *, active_only: bool, include_raw: bool) -> None:
    config = hand.config
    active_values = hand.read_pos()
    active_names = active_joint_names(config)
    active_sources = [f"motor {motor_id}" for motor_id in config.motor_ids]

    if active_only:
        print_table(
            "Calibrated active motor joints after home_offsets:",
            active_names,
            active_values,
            active_sources,
        )
    else:
        expanded_values = hand.read_joint_pos()
        expanded_names, expanded_sources = expanded_joint_names_and_sources(config)
        print_table(
            "Calibrated expanded joints after home_offsets:",
            expanded_names,
            expanded_values,
            expanded_sources,
        )

    if include_raw:
        raw_values = hand._client().read_pos()
        print()
        print_table(
            "Raw motor positions before home_offsets:",
            active_names,
            raw_values,
            active_sources,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read calibrated MIDAS joint positions after homing offsets."
    )
    parser.add_argument("--config", default=None, help="Saved HandConfig YAML.")
    parser.add_argument("--port", default=None, help="Serial port, e.g. /dev/ttyUSB0.")
    parser.add_argument("--baudrate", type=int, default=None)
    parser.add_argument(
        "--motors",
        default=None,
        help="Optional motor IDs, e.g. 0-3 or 0,1,2,3. Defaults to saved config.",
    )
    parser.add_argument(
        "--active-only",
        action="store_true",
        help="Print only the 13 active motor joints, not passive DIP joints.",
    )
    parser.add_argument(
        "--include-raw",
        action="store_true",
        help="Also print raw motor positions for comparison.",
    )
    parser.add_argument("--watch", action="store_true", help="Keep printing.")
    parser.add_argument("--rate", type=float, default=5.0, help="Watch rate in Hz.")
    args = parser.parse_args()

    config, source = load_config(args)
    print(f"Loaded config: {source}")
    print("Reported values are calibrated software joint positions, not raw motor positions.")
    print()

    with MidasHand(config) as hand:
        print(f"Connected on: {hand.port}")
        if args.watch:
            period = 1.0 / max(args.rate, 1e-6)
            try:
                while True:
                    print_positions(
                        hand,
                        active_only=args.active_only,
                        include_raw=args.include_raw,
                    )
                    print()
                    time.sleep(period)
            except KeyboardInterrupt:
                pass
        else:
            print_positions(
                hand,
                active_only=args.active_only,
                include_raw=args.include_raw,
            )


if __name__ == "__main__":
    main()
