# Paxini Tactile Notes

The `midas_hand_api.tactile` module drives the Paxini GEN3 high-speed board
(`PX6AX-GEN3-DP-S2015-Elite`) over USB serial using the `AA56` auto-push stream.
The current supported UI is the local Qt/PyQtGraph tool; the old web-based
visualizer has been removed, but the hardware layout and Python API remain the
same.

## Hardware Layout

| Finger | Force points | Board device address |
|--------|--------------|----------------------|
| thumb | 127 | 1 |
| index | 52 | 2 |
| middle | 52 | 3 |
| ring | 52 | 4 |

The Paxini board connects over USB serial at `921600` baud. On this setup it
usually appears as `/dev/ttyACM0`, but `PaxiniConfig(port=None)` auto-detects
boards whose USB description or manufacturer contains `paxini`. Make sure your
user is in the `dialout` group.

The reader consumes frames as fast as the board delivers them, up to roughly
`83.3 Hz`. `publish_rate_hz` controls how often `read_latest()` exposes a new
sample to user code.

## Python API

Integrated with `MidasHand`:

```python
from midas_hand_api import HandConfig, MidasHand, PaxiniConfig, PaxiniHandSensor

hand_config = HandConfig.load()

with PaxiniHandSensor(PaxiniConfig()) as tactile:
    with MidasHand(hand_config, tactile_sensor=tactile) as hand:
        hand.configure(enable_torque=True)
        data = hand.read_tactile()      # dict[str, ndarray shape (N, 3)]
        fz = hand.read_tactile_fz()     # dict[str, ndarray shape (N,)]
```

Standalone usage:

```python
from midas_hand_api import PaxiniConfig, PaxiniHandSensor

config = PaxiniConfig(port=None)

with PaxiniHandSensor(config) as sensor:
    data = sensor.read_latest()
    print(data["thumb"].shape)   # (127, 3)
    print(data["index"].shape)   # (52, 3)

    fz = sensor.read_tactile_fz()
    print(fz["index"])
```

`read_latest()` returns a dictionary keyed by finger name. Each value has shape
`(N, 3)` with columns `[Fx, Fy, Fz]` in Newtons. The single-axis helpers return
`dict[finger_name, ndarray shape (N,)]`.

If every finger listed in `PaxiniConfig.fingers` is not physically connected,
`connect()` raises immediately. To read a subset:

```python
config = PaxiniConfig(port=None, fingers=["index", "middle"])
```

## Key Config Options

| Option | Default | Description |
|--------|---------|-------------|
| `port` | `None` | Auto-detect Paxini board; pass a serial path such as `/dev/ttyACM0` to force one. |
| `fingers` | all four | Ordered list of fingers to read. |
| `baudrate` | `921600` | Paxini board serial baudrate. |
| `publish_rate_hz` | `60.0` | Rate at which `read_latest()` delivers a new value. |
| `median_window` | `3` | Rolling median window; set `1` to disable. |
| `scale_n` | `0.1` | Force scale in Newtons per LSB. |
| `signed_z` | `False` | Keep Fz unsigned, matching the Paxini protocol document. |

## Current Qt Tool

Paxini tactile code lives in the package folder:

```text
midas_hand_api/tactile/
```

The combined Qt app handles live streaming, CSV recording, and CSV replay:

```bash
python -m midas_hand_api.tactile.paxini_tactile_qt
```

After installing the package with the Qt extra, the same tool is available as:

```bash
midas-paxini
```

This app combines live streaming, CSV recording, and CSV replay. The previous
web-based visualizer and separate example-level live/record/replay scripts were
removed to keep the tactile stack simpler and faster.

For the full workflow and controls, see:

```text
midas_hand_api/tactile/README_paxini_qt_replay.md
```

Common commands:

```bash
python -m midas_hand_api.tactile.paxini_tactile_qt --port /dev/ttyACM0
python -m midas_hand_api.tactile.paxini_tactile_qt --csv paxini_recording.csv
python -m midas_hand_api.tactile.paxini_tactile_qt --replay-only --csv paxini_recording.csv
```
