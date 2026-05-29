## Paxini Qt Tactile Tool

The tactile tools now live inside `midas_hand_api/tactile/`. The active tool is
`paxini_tactile_qt.py`, which combines live streaming, CSV recording, and CSV
replay in one local PyQtGraph app.

### Hardware Reference

The driver supports the Paxini GEN3 high-speed board
`PX6AX-GEN3-DP-S2015-Elite` using the `AA56` auto-push stream.

| Finger | Force points | Board device address |
|--------|--------------|----------------------|
| thumb | 127 | 1 |
| index | 52 | 2 |
| middle | 52 | 3 |
| ring | 52 | 4 |

The board uses USB serial at `921600` baud. If no port is passed, the driver
auto-detects a board whose USB description or manufacturer contains `paxini`.
Explicit ports such as `/dev/ttyACM0` also work.

`read_latest()` returns `dict[str, ndarray]`; each array has shape `(N, 3)` with
columns `[Fx, Fy, Fz]` in Newtons. The board can stream at roughly `83.3 Hz`;
`publish_rate_hz` controls how often the API exposes the latest value.

Standalone API usage:

```python
from midas_hand_api import PaxiniConfig, PaxiniHandSensor

with PaxiniHandSensor(PaxiniConfig()) as sensor:
    data = sensor.read_latest()
    print(data["thumb"].shape)   # (127, 3)
    print(data["index"].shape)   # (52, 3)
```

### Install

From the `midas_hand_api` directory:

```bash
python -m pip install -e ".[qt]"
```

### Run Live Streaming

```bash
python -m midas_hand_api.tactile.paxini_tactile_qt
```

Useful options:

```bash
python -m midas_hand_api.tactile.paxini_tactile_qt --port /dev/ttyACM0
python -m midas_hand_api.tactile.paxini_tactile_qt --qt-update-hz 30
python -m midas_hand_api.tactile.paxini_tactile_qt --no-arrows
```

If the package is installed, the same app is available as:

```bash
midas-paxini
```

### Record

Use the same app in live mode:

```bash
python -m midas_hand_api.tactile.paxini_tactile_qt --csv paxini_recording.csv
```

Click `Record` to begin writing samples and `Stop` to save the CSV. Recording
runs at `30 Hz` by default:

```bash
python -m midas_hand_api.tactile.paxini_tactile_qt --record-rate-hz 30
```

### Replay

Replay can happen inside the same window after recording, or from a saved CSV:

```bash
python -m midas_hand_api.tactile.paxini_tactile_qt --replay-only --csv paxini_recording.csv
```

Replay uses interpolation between recorded frames, so a `30 Hz` recording can
display smoothly at the Qt update rate.

### Controls

- `Live`: return to live sensor display
- `Record`: start recording live samples
- `Stop`: stop recording and save the CSV, or pause replay
- `Replay`: replay the last recording or the loaded CSV
- `Load CSV`: load a saved tactile recording
- `Play` / `Pause`: pause or resume replay
- `Restart`: jump replay back to the start
- `Loop`: continuously replay the CSV
- `Speed`: slow down or speed up replay
- `Map intensity`: choose `|F|`, `|Fz|`, `|Fx|`, or `|Fy|`
- `History finger`: choose which finger appears in the Fx/Fy/Fz plot

### Notes

The old web-based tactile visualizer and separate example-level live, record,
and replay scripts were removed. Use this Qt app for current Paxini tactile
workflows.
