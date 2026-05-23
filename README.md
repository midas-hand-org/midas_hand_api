# Midas Hand API

Python API for the Midas 13-motor Dynamixel hand (`XM335-T323-T` actuators)
with Paxini GEN3 tactile sensing. Provides a low-level Dynamixel client, a
high-level hand object, motor homing/calibration, and a full tactile sensor
driver.

```text
midas_hand_api/
  actuators/   # Dynamixel client and control-table constants
  tactile/     # Paxini GEN3 high-speed board driver (AA56 auto-push stream)
  hand.py      # MidasHand — unified interface for motors and tactile
  homing.py    # motor homing and host-side calibration persistence
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Give your user serial-port access:

```bash
sudo usermod -aG dialout "$USER"
newgrp dialout # for it to work immediately, but it will not carry over to new terminals
```

Log out and back in after changing groups. You can also inspect stable adapter
paths with:

```bash
ls /dev/serial/by-id/*
```

## Smoke Test

```bash
source .venv/bin/activate
python examples/smoke_test.py --port /dev/ttyUSB0 --baudrate 1000000
```

By default, the smoke test uses motor IDs `0` through `12`. If you only want to
test a subset, pass `--motors`, for example `--motors 0,1,2,3`. Do not leave
Dynamixel Wizard open while running the API because it keeps the serial port
busy.

## First-Time Homing

After assembling the hand and confirming communication with the smoke test, run
homing before commanding normal poses. Homing drives each calibrated joint toward
its hard stop, computes the software zero from the CAD offset, and saves the
resulting calibration. After the homing sequence completes, the hand commands
all fingers to their software `0` position.

```bash
source .venv/bin/activate
python -m midas_hand_api --port /dev/ttyUSB0 --baudrate 1000000 --home
```

By default, this targets motor IDs `0` through `12`. Use `--motors` if you want
to home or debug a subset supported by the current homing routine.
Use `--home-thumb` or `--home-fingers` instead of `--home` for partial homing.

By default, the saved config is written to:

```text
~/.midas_hand/config.yaml
```

The calibration is saved to the **host computer**, not the hand itself. If you
move the hand to a different machine, homing must be run again on that machine.

Run homing on the first startup after assembly, after changing actuator horns or
linkages, whenever the physical relationship between motor raw position and joint
zero changes, or whenever you switch to a new host computer. You do not need to
home on every program start if the hand and host have not changed; load the saved
config instead.

Saved calibration files store only hand-specific calibration fields keyed by
motor ID:

```yaml
motor_ids:
- 0
- 1
- 2
- 3
# ...
- 12
home_offsets:
  0: 3.169204308748572
  1: 3.1461945992939455
  2: 1.888330344497239
  3: 1.842310926250655
  # ...
  12: 0.0
```

At runtime, these maps are converted into arrays ordered to match `motor_ids`.
`home_offsets` are raw actuator positions, in radians, where the software joint
position is defined as zero.

Controller gains, current limits, operating mode, encoder scale, baudrate, and
model constants are code defaults in `HandConfig.xm335_t323()`. They are not
persisted by homing, so tuning changes in code are not hidden by stale YAML
values.

## Example

```python
from midas_hand_api import HandConfig, MidasHand
import numpy as np

config = HandConfig.load()

with MidasHand(config) as hand:
    print(hand.ping())
    print(hand.verify_models())
    hand.configure(enable_torque=True)
    hand.set_positions(np.zeros(13))
    print(hand.read_pos())
    print(hand.read_joint_pos())
    print(hand.read_pos_vel_cur())
```

Before using real grasps, calibrate `joint_signs` and joint limits in
`HandConfig` for the actual Midas hand mechanics.

## Tactile Sensors

The `midas_hand_api.tactile` module drives the Paxini GEN3 high-speed board
(`PX6AX-GEN3-DP-S2015-Elite`) over USB serial using the AA56 auto-push stream.

### Hardware layout

| Finger | Force points | Board device address |
|--------|-------------|----------------------|
| thumb  | 127         | 1                    |
| index  | 52          | 2                    |
| middle | 52          | 3                    |
| ring   | 52          | 4                    |

The Paxini board connects via USB (typically `/dev/ttyUSB1`) at 921 600 baud.
Make sure your user is in the `dialout` group (see **Setup** above).

### Integrated with MidasHand

```python
from midas_hand_api import HandConfig, MidasHand, PaxiniConfig, PaxiniHandSensor

hand_config = HandConfig.load()
tactile = PaxiniHandSensor(PaxiniConfig(port="/dev/ttyUSB1"))

with MidasHand(hand_config, tactile_sensor=tactile) as hand:
    hand.configure(enable_torque=True)
    data = hand.read_tactile()       # dict[str, ndarray (N, 3)]
    fz = hand.read_tactile_fz()    # dict[str, ndarray (N,)]
```

### Standalone usage

```python
from midas_hand_api import PaxiniConfig, PaxiniHandSensor

config = PaxiniConfig(port="/dev/ttyUSB1")

with PaxiniHandSensor(config) as sensor:
    # read_latest() → dict[finger_name, ndarray shape (N, 3)]
    # columns are [Fx, Fy, Fz] in Newtons
    data = sensor.read_latest()
    print(data["thumb"].shape)   # (127, 3)
    print(data["index"].shape)   # (52, 3)

    # Single-axis convenience accessors → dict[finger_name, ndarray shape (N,)]
    fz = sensor.read_tactile_fz()
    print(fz["index"])
```

`connect()` raises `RuntimeError` immediately if any finger listed in
`PaxiniConfig.fingers` is not physically connected to the board. By default all
four fingers are expected. To use a subset:

```python
config = PaxiniConfig(port="/dev/ttyUSB1", fingers=["index", "middle"])
```

### Key config options

| Option | Default | Description |
|--------|---------|-------------|
| `port` | *(required)* | Serial port, e.g. `"/dev/ttyUSB1"` |
| `fingers` | all four | Ordered list of fingers to read |
| `publish_rate_hz` | `60.0` | Rate at which `read_latest()` delivers a new value |
| `median_window` | `3` | Rolling median window size; set `1` to disable |

The reader thread consumes frames as fast as the board delivers them
(hardware ceiling ~83.3 Hz). `publish_rate_hz` caps delivery independently.

## XM335-T323-T Defaults

- Protocol: 2.0
- Resolution: 4096 counts/rev
- Default baudrate: `1_000_000`
- Operating mode: current-based position control, value `5`
- Model number checked by `verify_models()`: `1710`
- Current unit: about `1 mA`
- Goal current limit default: `500`, max allowed by config: `910`
