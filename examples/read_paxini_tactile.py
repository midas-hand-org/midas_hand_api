"""Standalone Paxini tactile reader with live visualization.

This example connects only to the Paxini tactile board; it does not require a
Dynamixel hand connection. It reads the high-speed AA56 stream through
``PaxiniHandSensor`` and, by default, opens the shared ``PaxiniVisualizer`` with
the dark tactile maps, regional force arrows, and split history plots.

Usage::

    python examples/read_paxini_tactile.py
    python examples/read_paxini_tactile.py --port /dev/ttyACM0
    python examples/read_paxini_tactile.py --fingers thumb,index --viz-port 8051
    python examples/read_paxini_tactile.py --no-viz --print-rate-hz 5
    python examples/read_paxini_tactile.py --update-ms 100 --history-update-ms 300
    python examples/read_paxini_tactile.py --recording-csv data/paxini_recording.csv

Install visualization dependencies first if needed::

    python -m pip install -e ".[viz]"
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from midas_hand_api.tactile import PaxiniConfig, PaxiniHandSensor
from midas_hand_api.tactile.paxini_visualizer import PaxiniVisualizer


DEFAULT_FINGERS = ("thumb", "index", "middle", "ring")
COMPONENT_CHOICES = ("Fz", "|F|", "Fx", "Fy")


def parse_fingers(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return list(DEFAULT_FINGERS)

    names = [part.strip().lower() for part in value.split(",") if part.strip()]
    if not names:
        raise argparse.ArgumentTypeError("Expected comma-separated finger names.")

    unknown = sorted(set(names) - set(DEFAULT_FINGERS))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown finger(s): {', '.join(unknown)}. "
            f"Expected one or more of: {', '.join(DEFAULT_FINGERS)}."
        )
    return names


def tactile_summary(data: dict[str, np.ndarray]) -> str:
    parts = []
    for name, vectors in data.items():
        magnitudes = np.linalg.norm(vectors, axis=1)
        max_index = int(np.argmax(magnitudes)) if magnitudes.size else -1
        max_force = float(magnitudes[max_index]) if max_index >= 0 else 0.0
        total_force = float(np.sum(magnitudes))
        resultant = np.sum(vectors, axis=0) if len(vectors) else np.zeros(3)
        parts.append(
            f"{name}: max|F|={max_force:.2f}N"
            f"@{max_index + 1 if max_index >= 0 else '-'} "
            f"sum|F|={total_force:.2f}N "
            f"sumF=({resultant[0]:.2f},{resultant[1]:.2f},{resultant[2]:.2f})N"
        )
    return "  |  ".join(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--port",
        "--paxini-port",
        dest="port",
        default=None,
        help="Paxini serial port. If omitted, the driver tries to auto-detect it.",
    )
    parser.add_argument("--baudrate", type=int, default=921600)
    parser.add_argument(
        "--fingers",
        type=parse_fingers,
        default=list(DEFAULT_FINGERS),
        help="Comma-separated fingers to read, or 'all'. Default: all.",
    )
    parser.add_argument(
        "--publish-rate-hz",
        type=float,
        default=60.0,
        help="Rate at which read_latest() publishes samples from the reader thread.",
    )
    parser.add_argument(
        "--median-window",
        type=int,
        default=3,
        help="Rolling median window over raw frames. Use 1 to disable.",
    )
    parser.add_argument(
        "--discard-startup-frames",
        type=int,
        default=5,
        help="AA56 frames to discard after enabling stream.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.1,
        help="Force scale in Newtons per LSB.",
    )
    parser.add_argument(
        "--signed-z",
        action="store_true",
        help="Parse Fz as signed int8. Default keeps Fz unsigned.",
    )
    parser.add_argument(
        "--dtr",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="USB serial DTR state after opening.",
    )
    parser.add_argument(
        "--rts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="USB serial RTS state after opening.",
    )
    parser.add_argument("--serial-settle", type=float, default=0.75)
    parser.add_argument("--response-timeout", type=float, default=1.0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--viz-port", type=int, default=8050)
    parser.add_argument("--update-ms", type=int, default=100)
    parser.add_argument(
        "--history-update-ms",
        type=int,
        default=300,
        help="Browser refresh interval for the two history plots.",
    )
    parser.add_argument(
        "--history-len",
        type=int,
        default=300,
        help="Number of summarized samples kept in the visual history.",
    )
    parser.add_argument(
        "--component",
        choices=COMPONENT_CHOICES,
        default="Fz",
        help="Initial tactile map intensity component.",
    )
    parser.add_argument(
        "--no-arrows",
        action="store_true",
        help="Disable regional force arrows for maximum rendering speed.",
    )
    parser.add_argument(
        "--arrow-region-size",
        type=int,
        default=3,
        help="Approximate tactile point region size for summed force arrows.",
    )
    parser.add_argument(
        "--arrow-min-force",
        type=float,
        default=0.25,
        help="Minimum regional summed force in Newtons before drawing an arrow.",
    )
    parser.add_argument(
        "--recording-csv",
        default="paxini_recording.csv",
        help="CSV file overwritten by the visualizer Stop button after recording.",
    )
    parser.add_argument("--no-viz", action="store_true", help="Do not start Dash visualizer.")
    parser.add_argument(
        "--no-print",
        action="store_true",
        help="Suppress terminal force summaries.",
    )
    parser.add_argument(
        "--print-rate-hz",
        type=float,
        default=1.0,
        help="Terminal summary print rate. Ignored with --no-print.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Optional run duration in seconds. Default: run until Ctrl-C.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PaxiniConfig(
        port=args.port,
        fingers=args.fingers,
        baudrate=args.baudrate,
        publish_rate_hz=args.publish_rate_hz,
        scale_n=args.scale,
        signed_z=args.signed_z,
        discard_startup_frames=args.discard_startup_frames,
        median_window=args.median_window,
        response_timeout_s=args.response_timeout,
        serial_settle_s=args.serial_settle,
        dtr=args.dtr,
        rts=args.rts,
    )

    sensor = PaxiniHandSensor(config)
    try:
        sensor.connect()
        print(f"Paxini connected on {sensor.port}")
        print(f"Reading fingers: {', '.join(args.fingers)}")

        if not args.no_viz:
            PaxiniVisualizer(
                sensor.read_latest,
                update_ms=args.update_ms,
                component=args.component,
                history_len=args.history_len,
                history_update_ms=args.history_update_ms,
                show_arrows=not args.no_arrows,
                arrow_region_size=args.arrow_region_size,
                arrow_min_force_n=args.arrow_min_force,
                recording_csv=args.recording_csv,
            ).start_background(host=args.host, port=args.viz_port)

        print_interval_s = 1.0 / max(args.print_rate_hz, 1e-6)
        next_print_s = 0.0
        stop_s = None if args.duration is None else time.monotonic() + args.duration

        while stop_s is None or time.monotonic() < stop_s:
            now = time.monotonic()
            if not args.no_print and now >= next_print_s:
                try:
                    print(tactile_summary(sensor.read_latest()))
                except RuntimeError as exc:
                    print(f"Waiting for tactile data: {exc}")
                next_print_s = now + print_interval_s
            time.sleep(0.02)

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        sensor.disconnect()
        print("Paxini disconnected.")


if __name__ == "__main__":
    main()
