## Paxini Qt Tactile Tools

Use these tools for smooth local tactile visualization, recording, and replay.

### 1. Install Qt Dependencies

Install these once in the active Python environment:

```bash
python -m pip install pyqtgraph PyQt6
```

Or install the package extra from the `midas_hand_api` directory:

```bash
python -m pip install -e ".[qt]"
```

### 2. Live Visualize

From the `midas_hand_api` directory:

```bash
python examples/read_paxini_tactile.py
```

Useful options:

```bash
python examples/read_paxini_tactile.py --port /dev/ttyACM0
python examples/read_paxini_tactile.py --qt-update-hz 30
python examples/read_paxini_tactile.py --no-arrows
python examples/read_paxini_tactile.py --no-viz --print-rate-hz 5
```

### 3. Record A CSV

From the `midas_hand_api` directory:

```bash
python examples/record_paxini_tactile.py --csv paxini_recording.csv
```

Keyboard controls:

- `r`: start recording
- `r`: stop recording and save `paxini_recording.csv`
- `q`: quit

The recorder samples at 30 Hz by default. To change that:

```bash
python examples/record_paxini_tactile.py --csv paxini_recording.csv --record-rate-hz 30
```

### 4. Replay Locally At 30 Hz

From the `midas_hand_api` directory:

```bash
python examples/replay_paxini_recording_qt.py --csv paxini_recording.csv --replay-rate-hz 30
```

Useful options:

```bash
python examples/replay_paxini_recording_qt.py --csv paxini_recording.csv --replay-rate-hz 30
python examples/replay_paxini_recording_qt.py --csv paxini_recording.csv --component '|F|'
python examples/replay_paxini_recording_qt.py --csv paxini_recording.csv --no-arrows
```

Qt replay controls:

- `Pause` / `Play`: pause or resume replay
- `Restart`: jump back to the start
- `Loop`: keep replaying the CSV
- `Speed`: slow down or speed up playback
- `Map intensity`: choose `|F|`, `|Fz|`, `|Fx|`, or `|Fy|`
- `History finger`: choose which finger appears in the Fx/Fy/Fz plot
