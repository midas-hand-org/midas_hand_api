# Paxini Tactile Notes

Paxini tactile code now lives in the package folder:

```text
midas_hand_api/tactile/
```

The current supported UI is the local Qt/PyQtGraph tool:

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
