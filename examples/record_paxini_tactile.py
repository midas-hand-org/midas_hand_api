"""Terminal-only Paxini tactile recorder.

This script connects to the Paxini high-speed board without opening any
visualizer. Press ``r`` to start recording at 30 Hz, press ``r`` again to stop
and overwrite the CSV file, and press ``q`` to quit.

Run from the ``midas_hand_api`` package directory, for example::

    python examples/record_paxini_tactile.py --port /dev/ttyACM0
    python examples/record_paxini_tactile.py --csv examples/paxini_recording.csv

The output CSV uses the same row format as the live visualizer recording:

    sample_index,time_s,finger,point_index,fx_n,fy_n,fz_n
"""

from __future__ import annotations

import argparse
import csv
import select
import sys
import termios
import time
import tty
from pathlib import Path

import numpy as np

from midas_hand_api.tactile import PaxiniConfig, PaxiniHandSensor


DEFAULT_FINGERS = ("thumb", "index", "middle", "ring")


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


class KeyReader:
    """Read single keys without blocking the tactile sampling loop."""

    def __init__(self) -> None:
        self._fd: int | None = None
        self._old_attrs: list | None = None
        self._is_tty = False

    def __enter__(self) -> "KeyReader":
        self._is_tty = sys.stdin.isatty()
        if self._is_tty:
            self._fd = sys.stdin.fileno()
            self._old_attrs = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        return self

    def __exit__(self, *_args: object) -> None:
        if self._is_tty and self._fd is not None and self._old_attrs is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_attrs)

    def read_key(self) -> str | None:
        if not self._is_tty:
            return None
        ready, _, _ = select.select([sys.stdin], [], [], 0.0)
        if not ready:
            return None
        return sys.stdin.read(1).lower()


def recording_sample(timestamp_s: float, data: dict[str, np.ndarray], fingers: list[str]) -> dict:
    copied = {}
    for name in fingers:
        vectors = data.get(name)
        if vectors is not None:
            copied[name] = np.asarray(vectors, dtype=np.float64).copy()
    return {"time": timestamp_s, "data": copied}


def write_recording_csv(samples: list[dict], path: Path) -> None:
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_index", "time_s", "finger", "point_index", "fx_n", "fy_n", "fz_n"])
        if not samples:
            return

        t0 = samples[0]["time"]
        for sample_index, sample in enumerate(samples):
            time_s = sample["time"] - t0
            for finger, vectors in sample["data"].items():
                for point_index, vector in enumerate(vectors):
                    writer.writerow(
                        [
                            sample_index,
                            f"{time_s:.9f}",
                            finger,
                            point_index,
                            f"{float(vector[0]):.9f}",
                            f"{float(vector[1]):.9f}",
                            f"{float(vector[2]):.9f}",
                        ]
                    )


def wait_for_first_sample(sensor: PaxiniHandSensor, timeout_s: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            sensor.read_latest()
            return True
        except RuntimeError:
            time.sleep(0.05)
    return False


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
        "--record-rate-hz",
        type=float,
        default=30.0,
        help="CSV recording sample rate. Default: 30 Hz.",
    )
    parser.add_argument(
        "--publish-rate-hz",
        type=float,
        default=60.0,
        help="Rate at which the Paxini driver publishes read_latest() samples.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("paxini_recording.csv"),
        help="CSV file to overwrite each time recording stops.",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    record_interval_s = 1.0 / max(args.record_rate_hz, 1e-6)
    config = PaxiniConfig(
        port=args.port,
        fingers=args.fingers,
        baudrate=args.baudrate,
        publish_rate_hz=max(args.publish_rate_hz, args.record_rate_hz),
        scale_n=args.scale,
        signed_z=args.signed_z,
        discard_startup_frames=args.discard_startup_frames,
        median_window=args.median_window,
        response_timeout_s=args.response_timeout,
        serial_settle_s=args.serial_settle,
        dtr=args.dtr,
        rts=args.rts,
    )

    samples: list[dict] = []
    recording = False
    next_sample_s = time.monotonic()
    last_status_s = 0.0
    recording_started_s = 0.0

    sensor = PaxiniHandSensor(config)
    try:
        sensor.connect()
        print(f"Paxini connected on {sensor.port}")
        print(f"Reading fingers: {', '.join(args.fingers)}")
        if not wait_for_first_sample(sensor):
            print("Warning: no tactile data yet. Recording will wait for samples.")

        print("Press r to start/stop recording. Press q to quit.")
        if not sys.stdin.isatty():
            print("Keyboard control requires an interactive terminal.")

        with KeyReader() as keys:
            while True:
                key = keys.read_key()
                if key == "q":
                    if recording:
                        write_recording_csv(samples, args.csv)
                        print(f"\nSaved {len(samples)} frames to {args.csv}")
                    break
                if key == "r":
                    if recording:
                        recording = False
                        write_recording_csv(samples, args.csv)
                        elapsed_s = time.monotonic() - recording_started_s
                        rate = len(samples) / elapsed_s if elapsed_s > 0 else 0.0
                        print(f"\nStopped. Saved {len(samples)} frames to {args.csv} ({rate:.1f} Hz).")
                    else:
                        samples = []
                        recording = True
                        recording_started_s = time.monotonic()
                        next_sample_s = recording_started_s
                        last_status_s = 0.0
                        print("\nRecording started.")

                now = time.monotonic()
                if recording and now >= next_sample_s:
                    try:
                        data = sensor.read_latest()
                        samples.append(recording_sample(now, data, args.fingers))
                    except RuntimeError as exc:
                        if now - last_status_s >= 1.0:
                            print(f"\nWaiting for tactile data: {exc}")
                            last_status_s = now

                    next_sample_s += record_interval_s
                    if next_sample_s < now - record_interval_s:
                        next_sample_s = now + record_interval_s

                if recording and now - last_status_s >= 1.0:
                    elapsed_s = now - recording_started_s
                    rate = len(samples) / elapsed_s if elapsed_s > 0 else 0.0
                    print(f"\rRecording {len(samples)} frames at {rate:.1f} Hz...", end="", flush=True)
                    last_status_s = now

                time.sleep(0.001)

    except KeyboardInterrupt:
        if recording:
            write_recording_csv(samples, args.csv)
            print(f"\nInterrupted. Saved {len(samples)} frames to {args.csv}")
        else:
            print("\nInterrupted.")
    finally:
        sensor.disconnect()
        print("Paxini disconnected.")


if __name__ == "__main__":
    main()
