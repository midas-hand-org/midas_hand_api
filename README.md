# Midas Hand API

Python API for a Dynamixel-based hand using XM335-T323-T actuators. The
structure follows the LEAP Hand Python API: a low-level Dynamixel client plus a
higher-level hand object.

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
python examples/smoke_test.py --port /dev/ttyUSB0 --baudrate 1000000 --motors 0,1,2,3
```

If your motor IDs or baudrate differ, update the command. Do not leave
Dynamixel Wizard open while running the API because it keeps the serial port
busy.

## Example

```python
from midas_hand_api import HandConfig, MidasHand
import numpy as np

config = HandConfig.xm335_t323(
    motor_ids=tuple(range(0, 13)),
    port="/dev/ttyUSB0",
    baudrate=1_000_000,
)

with MidasHand(config) as hand:
    print(hand.ping())
    print(hand.verify_models())
    hand.configure(enable_torque=True)
    hand.set_positions(np.zeros(13))
    print(hand.read_pos_vel_cur())
```

Before using real grasps, calibrate `home_offsets`, `joint_signs`, and joint
limits in `HandConfig` for the actual Midas hand mechanics.

## XM335-T323-T Defaults

- Protocol: 2.0
- Resolution: 4096 counts/rev
- Default baudrate: `1_000_000`
- Operating mode: current-based position control, value `5`
- Model number checked by `verify_models()`: `1701`
- Current unit: about `1 mA`
- Goal current limit default: `350`, max allowed by config: `910`
