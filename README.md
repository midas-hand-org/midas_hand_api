# Midas Hand API

Python API for a Dynamixel-based hand using XM335-T323-T actuators. The
structure follows the LEAP Hand Python API: a low-level Dynamixel client plus a
higher-level hand object.

The package is organized around the hand rather than individual vendors:

```text
midas_hand_api/
  actuators/   # Dynamixel client and control-table constants
  tactile/     # Paxini tactile API surface and frame/config types
  hand.py      # integrated high-level MidasHand facade
  homing.py    # motor homing and saved motor calibration
```

`homing.py` is the current motor calibration workflow. A separate calibration
package can be added later if tactile calibration grows beyond simple config
fields.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Give your user serial-port access:

```bash
sudo usermod -aG dialout "$USER"
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

Run homing on the first startup after assembly, after changing actuator horns or
linkages, or whenever the physical relationship between motor raw position and
joint zero changes. You do not need to home on every program start if the hand
has not been reassembled; load the saved config instead.

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

The package includes a provisional `midas_hand_api.tactile` namespace for the
Paxini `PX6AX-GEN3-DP-S2015-Elite` sensor:

```python
from midas_hand_api import PaxiniConfig, PaxiniSensor

paxini = PaxiniSensor(PaxiniConfig(port="/dev/ttyUSB1"))
```

The public Paxini pages describe product specs and communication accessories,
but not a Python SDK or packet protocol. `PaxiniSensor` is therefore a typed
placeholder until the vendor SDK, serial protocol, SPI adapter protocol, or
high-speed-board protocol is available. Once that transport is implemented,
`PaxiniSensor` can be passed into `MidasHand(tactile_sensor=paxini)` and read
through `hand.read_tactile()`. See `docs/paxini_px6ax_notes.md`.

## XM335-T323-T Defaults

- Protocol: 2.0
- Resolution: 4096 counts/rev
- Default baudrate: `1_000_000`
- Operating mode: current-based position control, value `5`
- Model number checked by `verify_models()`: `1710`
- Current unit: about `1 mA`
- Goal current limit default: `500`, max allowed by config: `910`
