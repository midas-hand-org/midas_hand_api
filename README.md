# Midas Hand API

Python API for the Midas 13-motor Dynamixel hand (`XM335-T323-T` actuators)
with Paxini tactile support. Provides a low-level Dynamixel client, a high-level
hand object, motor homing/calibration utilities, and local Qt tactile tools.

```text
midas_hand_api/
  actuators/   # Dynamixel client and control-table constants
  hand.py      # MidasHand high-level motor interface
  homing.py    # motor homing and host-side calibration persistence
  tactile/     # Paxini driver, Qt visualizer, recording, and replay
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install .          # or pip install -e . for a dev/editable install
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

### Dynamixel USB Latency

On Linux, FTDI-based Dynamixel adapters such as the U2D2 can default to a
`16 ms` USB serial latency timer. That can cap request/response control loops
near 62.5 Hz even when the Dynamixel baudrate is much higher. For clean high-rate
sync reads, install the persistent latency rule:

```bash
./setup_dynamixel_latency.sh
```

The script lets you select the Dynamixel adapter, writes a udev rule matching
that adapter's USB serial number, reloads udev, and tells you if the adapter
needs to be unplugged and replugged.

## Smoke Test

```bash
source .venv/bin/activate
python examples/smoke_test.py
python -m midas_hand_api --help  # see options for motor scan, live position read, homing, configure, tactile recalibration, etc.
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
python -m midas_hand_api --home
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

Controller gains, current limits, operating mode, baudrate, and XM335-T323-T
unit constants are code defaults in `HandConfig`. They are not persisted by
homing, so tuning changes in code are not hidden by stale YAML values.

## Example

```python
from midas_hand_api import HandConfig, MidasHand
import numpy as np

config = HandConfig.load()

with MidasHand(config) as hand:
    print(hand.ping())
    print(hand.verify_models())
    hand.configure(enable_torque=True)
    hand.set_motion_profile(
        # None or 0.0 disables velocity-profile limiting.
        profile_velocity_rad_s=2.5,
        # None or 0.0 disables acceleration profiling.
        profile_acceleration_rad_s2=30.0,
    )
    hand.set_positions(np.zeros(13))
    print(hand.read_pos())
    print(hand.read_joint_pos())
    print(hand.read_pos_vel_cur())
```

Joint directions follow the right-hand rule, with the thumb pointing along the
servo horn axis.

**NOTE:** For robot learning or teleoperated demonstrations,
disabling the motion profiling with `profile_velocity_rad_s=None` and
`profile_acceleration_rad_s2=None` will maximize reactivity. This removes extra actuator-side trajectory
shaping; motion is still subject to current limits, gains, update rate, and
contact.

## Tactile Sensors

The `midas_hand_api.tactile` module drives the Paxini GEN3 high-speed board
(`PX6AX-GEN3-DP-S2015-Elite`) over USB serial using the AA56 auto-push stream.

### Hardware layout

| Finger | Force points (N) | Board device address |
|--------|-------------|----------------------|
| thumb  | 127         | 1                    |
| index  | 52          | 2                    |
| middle | 52          | 3                    |
| ring   | 52          | 4                    |

The Paxini board connects via USB (typically `/dev/ttyACM0`) at 921 600 baud.
Make sure your user is in the `dialout` group (see **Setup** above). If only one
Paxini board is connected, `PaxiniConfig` will auto-detect the port; otherwise
specify it explicitly.

### Integrated with MidasHand

```python
from midas_hand_api import HandConfig, MidasHand, PaxiniConfig, PaxiniHandSensor

hand_config = HandConfig.load()
tactile = PaxiniHandSensor(PaxiniConfig())  # auto-detects port

with MidasHand(hand_config, tactile_sensor=tactile) as hand:
    hand.configure(enable_torque=True)
    hand.recalibrate_tactile()     # re-zero tactile sensors — make sure nothing is touching them
    data = hand.read_tactile()     # dict[str, ndarray (N, 3)]
    fz   = hand.read_tactile_fz() # dict[str, ndarray (N,)]
```

### Standalone usage

```python
from midas_hand_api import PaxiniConfig, PaxiniHandSensor

with PaxiniHandSensor(PaxiniConfig()) as sensor:
    sensor.recalibrate()              # re-zero tactile sensors — make sure nothing is touching them
    data = sensor.read_latest()       # dict[finger_name, ndarray (N, 3)] — columns: [Fx, Fy, Fz] N
    print(data["thumb"].shape)        # (127, 3)
    print(data["index"].shape)        # (52, 3)

    fz = sensor.read_tactile_fz()     # dict[finger_name, ndarray (N,)]
    print(fz["index"])
```

> **Habit:** call `recalibrate()` (or `hand.recalibrate_tactile()`) once right
> after connecting, with nothing touching the sensors. It re-zeros every
> connected sensor's distributed-force baseline so each run starts from a clean
> zero. Any contact present at call time becomes the new zero, so keep the
> fingertips clear. The Qt tool does this automatically on startup; pass
> `--no-recalibrate` to skip it.

`connect()` raises `RuntimeError` if any finger in `PaxiniConfig.fingers` is not
physically connected. To use a subset:

```python
sensor = PaxiniHandSensor(PaxiniConfig(port="/dev/ttyACM0", fingers=["index", "middle"]))
```

### Key config options

| Option | Default | Description |
|--------|---------|-------------|
| `port` | auto-detect | Serial port, e.g. `"/dev/ttyACM0"` |
| `fingers` | all four | Ordered list of fingers to read |
| `publish_rate_hz` | `60.0` | Rate at which `read_latest()` delivers a new value |
| `median_window` | `3` | Rolling median window for noise reduction; set `1` to disable |

The reader thread consumes frames as fast as the board delivers them
(hardware ceiling ~83.3 Hz). `publish_rate_hz` caps delivery independently.

## Paxini Tactile Qt Tool

Paxini tactile support lives under `midas_hand_api/tactile/`. The Qt tool
combines live streaming, CSV recording, and CSV replay in one local app.

Install the optional Qt dependencies:

```bash
pip install ".[qt]"    # or pip install -e ".[qt]" for a dev/editable install
```

Run the tactile app from the `midas_hand_api` directory:

```bash
python -m midas_hand_api.tactile.paxini_tactile_qt
```

After installing the package, you can also use:

```bash
midas-paxini
```

Useful options:

```bash
python -m midas_hand_api.tactile.paxini_tactile_qt --port /dev/ttyACM0
python -m midas_hand_api.tactile.paxini_tactile_qt --csv paxini_recording.csv
python -m midas_hand_api.tactile.paxini_tactile_qt --replay-only --csv paxini_recording.csv
```

See `midas_hand_api/tactile/README_paxini_qt_replay.md` for the full recording
and replay workflow.

## XM335-T323-T Defaults

- Protocol: 2.0
- Resolution: 4096 counts/rev
- Default baudrate: `4_000_000`
- Default operating mode: current-based position control, value `5`
- Model number checked by `verify_models()`: `1710`
- Current unit: about `1 mA`
- Goal current limit default: `600`, max allowed by config: `910`
- Motion profile: velocity-based profile mode; profile acceleration is accepted
  by the API in `rad/s^2` and converted to the XM335 unit of
  `214.577 rev/min^2` per raw count.

### Operating Modes

The Midas hand uses XM335-T323-T actuators. The supported Dynamixel operating
modes are:

| Value | Mode | Use case |
|-------|------|----------|
| `0` | Current control | Direct torque/current experiments, low-level force behaviors, and diagnostics where the controller provides its own position or velocity loop. Use carefully because there is no position holding loop. |
| `1` | Velocity control | Homing moves, continuous sweeps, spin-at-speed tests, and debugging motion direction or bus communication. This is not the normal grasping mode. |
| `3` | Position control | Standard single-turn joint position mode. Use this for repeatable pose playback, calibration checks, and precise scripted motions when you want the actuator to prioritize tracking the commanded position. |
| `4` | Extended position control | Multi-turn position commands for mechanisms that intentionally rotate beyond one revolution. The Midas hand joints normally should not need this. |
| `5` | Current-based position control | Default. Still commands position, but also uses `Goal Current(102)` as the current/torque limit. Use this for robot learning, teleoperation, contact-rich interaction, and grasping because the hand can yield under contact instead of forcing the exact target as rigidly when the current limit is reached. |
| `16` | PWM control | Raw voltage/PWM-style experiments and actuator characterization. This bypasses the normal current/position abstractions and should only be used for low-level testing. |

`MidasHand.configure()` applies the default current-based position mode (`5`).
If you change `HandConfig.operating_mode`, disable torque before switching modes.
Changing modes resets profile settings and mode-specific gains on the actuator.
