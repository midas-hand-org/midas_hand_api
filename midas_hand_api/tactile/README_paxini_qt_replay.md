## Paxini Qt Tactile Tool

The tactile tools now live inside `midas_hand_api/tactile/`. The active tool is
`paxini_tactile_qt.py`, which combines live streaming, CSV recording, and CSV
replay in one local PyQtGraph app.

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
