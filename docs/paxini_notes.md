# Paxini Tactile Sensor — Developer Notes

## Hardware

**Sensor:** Paxini PX-6AX GEN3 (`PX6AX-GEN3-DP-S2015-Elite`)

| Property | Value |
|---|---|
| Form factor | Fingertip, 20 × 15 × 10 mm |
| Normal force range | 0 – 25 N |
| Tangential force range | ±10 N |
| Min recognizable force | 0.1 N |
| Repeatability | < 0.5% FS |
| Internal sampling | 1,000,000 Hz (claimed) |
| Max output frequency | 1,000 Hz (spec); **~83.3 Hz in practice** over the high-speed board |
| Resolution | 1 LSB = 0.1 N |
| Protection | IP68 |

**Communication board:** Paxini high-speed integration board, USB CDC serial at 921,600 baud.

**Finger mapping on the Midas hand:**

| Finger | Module # | Device address | Force points | Sensor size |
|---|---|---|---|---|
| Thumb | 0 | 1 | 127 | 26 mm pad |
| Index | 1 | 2 | 52 | 15 mm pad |
| Middle | 2 | 3 | 52 | 15 mm pad |
| Ring | 3 | 4 | 52 | 15 mm pad |

Device address = module number + 1 (Paxini spec §3.1).

---

## Serial Port Setup

The Paxini board enumerates as a USB CDC device. On Linux it typically appears as `/dev/ttyACM0`. Serial settings:

```
Baudrate:      921600
Byte size:     8
Parity:        None
Stop bits:     1
DTR:           False   (must be off — Paxini default)
RTS:           True    (must be on — Paxini default)
```

After opening the port, wait ~0.75 s before writing (`serial_settle_s`). This gives the CDC driver time to stabilize. Getting DTR/RTS wrong is the most common cause of no frames being received.

---

## Wire Protocol

The board supports two frame types. This API uses **AA56 auto-push only**. The AA55 request-response path is used only for initialization writes.

### AA56 Auto-Push Frame (board → host)

```
AA 56  <reserved:1>  <valid_frame_len:2 LE>  <status:1>
  [per finger, in configured order:
    <resultant_fx:2 LE>  <resultant_fy:2 LE>  <resultant_fz:2 LE>   (6 bytes, skipped)
    <distributed_force: N×3 bytes>
  ]
<LRC:1>
```

- `valid_frame_len` — number of bytes from `status` to the last data byte (before LRC), little-endian.
- `status` — `0x00` means OK; non-zero indicates a board error.
- Each force point is 3 bytes: `[Fx, Fy, Fz]`, each a raw `uint8` interpreted as a **signed** value for Fx/Fy (range −128 to 127, centered at 0), and **unsigned** for Fz by default (0–255). Multiply by 0.1 to get Newtons.
- Fingers appear **in the order they were configured** on the board. The Midas hand uses thumb → index → middle → ring.
- **LRC** (Longitudinal Redundancy Check): `(-sum(all_preceding_bytes)) & 0xFF`.

### AA55 Write Frame (host → board, initialization only)

Used to set register `0x0016` (auto-push data type) to `0x03` (resultant + distributed). Sent **no-ack** style — the host does not wait for a response — because some CDC drivers produce delayed ACK responses that confuse the framing.

```
55 AA  00 10  <address:2 LE>  <data_len:2 LE>  <data>  <LRC:1>
```

### Initialization Sequence

```
1. Open serial port (DTR=False, RTS=True)
2. Sleep 0.75 s
3. Flush serial buffers
4. Write DISABLE_AUTO_PUSH  (55 AA 00 10 17 00 01 00 00 D9)
5. Sleep 0.2 s, drain any response
6. Write register 0x0016 = 0x03  (no-ack: resultant + distributed)
7. Sleep 0.05 s, drain
8. Flush input buffer
9. Write ENABLE_AUTO_PUSH   (55 AA 00 10 17 00 01 00 01 D8)
10. Discard N startup frames (default 5) — these contain stale data
11. Parse one validation frame to confirm all configured fingers respond
12. Start reader + publisher threads
```

---

## Python API

### `PaxiniConfig`

Frozen dataclass — create once, pass to `PaxiniHandSensor`.

```python
from midas_hand_api.tactile import PaxiniConfig

config = PaxiniConfig(
    port="/dev/ttyACM0",
    fingers=["thumb", "index", "middle", "ring"],   # default: all four
    baudrate=921_600,
    publish_rate_hz=60.0,
    scale_n=0.1,
    signed_z=False,
    discard_startup_frames=5,
    median_window=3,
    response_timeout_s=1.0,
    startup_attempts=3,
    serial_settle_s=0.75,
    dtr=False,
    rts=True,
)
```

| Parameter | Default | Notes |
|---|---|---|
| `port` | — | Required. `/dev/ttyACM0` typical on Linux. |
| `fingers` | all four | Subset e.g. `["thumb", "index"]` if others not connected. |
| `publish_rate_hz` | 60.0 | Rate at which `read_latest()` reflects new data. Hardware ceiling is ~83.3 Hz. |
| `scale_n` | 0.1 | Newtons per LSB. Matches Paxini GEN3 spec. |
| `signed_z` | False | Parse Fz as signed int8. Keep False unless Paxini changes their protocol. |
| `discard_startup_frames` | 5 | Frames thrown away right after enabling stream. Prevents stale data. |
| `median_window` | 3 | Rolling median over this many consecutive frames. Set to 1 to disable. |
| `response_timeout_s` | 1.0 | Max seconds to wait for one AA56 frame before raising `TimeoutError`. |
| `startup_attempts` | 3 | Retries the auto-push enable sequence if the first frames do not arrive. |
| `serial_settle_s` | 0.75 | Seconds to wait after opening the port before any writes. |
| `dtr` | False | Leave False. Setting DTR True on some boards disables output. |
| `rts` | True | Leave True. Required by the Paxini board. |

### `PaxiniHandSensor`

```python
from midas_hand_api.tactile import PaxiniConfig, PaxiniHandSensor

sensor = PaxiniHandSensor(PaxiniConfig(port="/dev/ttyACM0"))
sensor.connect()

data = sensor.read_latest()       # dict[str, ndarray (N, 3)]  — columns: Fx, Fy, Fz (N)
fz   = sensor.read_tactile_fz()  # dict[str, ndarray (N,)]    — Fz only
fx   = sensor.read_tactile_fx()  # dict[str, ndarray (N,)]
fy   = sensor.read_tactile_fy()  # dict[str, ndarray (N,)]

sensor.disconnect()

# Or as a context manager:
with PaxiniHandSensor(config) as sensor:
    data = sensor.read_latest()
```

#### `connect()`

Does everything: opens the serial port, runs the initialization sequence, validates all configured fingers, and starts two background daemon threads. **Raises immediately** if any configured finger is not physically connected:

```
RuntimeError: Finger validation failed: Auto-push payload too short for finger 'ring':
payload has 558 bytes but need at least 714.
Verify that 'ring' is physically connected to the board.
```

Other exceptions: `serial.SerialException` (port not found/in use), `ValueError` (unknown finger name).

#### `read_latest() → dict[str, ndarray (N, 3)]`

Returns the last sample promoted by the publish thread. The dict keys are the configured finger names. Each value is a `float64` array with shape `(N, 3)`:

```
column 0 — Fx (tangential, Newtons, signed)
column 1 — Fy (tangential, Newtons, signed)
column 2 — Fz (normal, Newtons, unsigned by default)
```

Raises `RuntimeError` if not connected or if no frame has arrived yet (brief window after `connect()` returns).

#### `read_tactile_fx/fy/fz()`

Convenience slices of `read_latest()`. Each returns `dict[str, ndarray (N,)]`.

#### `disconnect()` / `close()`

Stops both threads, sends `DISABLE_AUTO_PUSH` to the board, closes the serial port. `close()` is an alias (for compatibility with `MidasHand.close()`).

### Two-Thread Model

```
Hardware (~83.3 Hz max)
       │
  [reader thread]  ←── reads AA56 frames at hardware rate, applies median filter,
       │                writes to _raw_latest under lock
       ▼
  [publish thread] ←── at publish_rate_hz (default 60 Hz), promotes _raw_latest
       │                to _latest under lock
       ▼
  read_latest()    ←── always returns _latest (last published value)
```

This decouples acquisition rate from delivery rate. The reader never sleeps — it loops on `_read_frame()` which only blocks on `ser.in_waiting`. The publisher sleeps the remainder of each 1/60 s tick using `threading.Event.wait()` so `disconnect()` can interrupt it promptly.

### Median Filter

When `median_window > 1` (default 3), the reader keeps a rolling `deque` of the last N raw samples and applies `np.median` element-wise across the window before storing. This smooths transient spikes without introducing significant latency (at 83 Hz, a window of 3 = ~36 ms lag).

### Integration with MidasHand

```python
from midas_hand_api import MidasHand, HandConfig
from midas_hand_api.tactile import PaxiniConfig, PaxiniHandSensor

hand = MidasHand(HandConfig(...))
hand.tactile_sensor = PaxiniHandSensor(PaxiniConfig(port="/dev/ttyACM0"))
hand.tactile_sensor.connect()

data = hand.read_tactile()    # same as hand.tactile_sensor.read_latest()
fz   = hand.read_tactile_fz()
```

---

## `PaxiniQtVisualizer`

Local Qt visualization of Paxini force data using `pyqtgraph`. It accepts any
callable that returns `dict[str, ndarray (N, 3)]`, so it can display
`PaxiniHandSensor.read_latest` directly.

### Install

```bash
pip install -e ".[qt]"
```

### Usage

```python
from midas_hand_api.tactile import PaxiniConfig, PaxiniHandSensor, PaxiniQtVisualizer

sensor = PaxiniHandSensor(PaxiniConfig(port="/dev/ttyACM0"))
sensor.connect()

PaxiniQtVisualizer(sensor.read_latest, update_hz=30).run()
```

The simplest command-line entry point is:

```bash
python examples/read_paxini_tactile.py
python examples/read_paxini_tactile.py --qt-update-hz 30
python examples/read_paxini_tactile.py --no-arrows
```

### Constructor

```python
PaxiniQtVisualizer(
    get_data,                       # Callable[[], dict[str, ndarray (N, 3)]]
    *,
    coords=None,                    # dict[str, ndarray (N, 3)] in mm — auto-detected if None
    update_hz=30.0,                 # Qt redraw rate
    component="Fz",                 # initial force component shown
    history_len=600,                # rolling history depth
    history_window_s=10.0,          # seconds shown in history plots
    show_arrows=True,
)
```

`get_data` is called on every Qt timer tick. Passing `sensor.read_latest`
directly works because `read_latest` is a bound method — no lambda needed.

`coords` are the physical X/Y/Z positions of each force point on the fingertip pad in millimetres. If omitted, auto-detected by matching the array's row count against the bundled CSV files:

| Row count | CSV file | Pad |
|---|---|---|
| 127 | `paxini_fingertip_26mm_127pts.csv` | Thumb (26 mm) |
| 52 | `paxini_fingertip_15mm_52pts.csv` | Index/middle/ring (15 mm) |

### What The Qt UI Shows

**One tactile map per finger** — each force point rendered at its physical X/Y coordinate on the fingertip. Points are blue at zero force and trend red as the selected force component increases. The aspect ratio is 1:1 so the pad geometry is not distorted.

**Regional arrows** — summed force arrows are drawn over small regions of the tactile map, not every point, so force direction is visible without clutter.

**Split histories** — the left history plot shows ΣFx/ΣFy/ΣFz for one selected finger. The right history plot shows Σ|F| for all fingers.

**Controls** — choose the map intensity (`|F|`, `|Fz|`, `|Fx|`, `|Fy|`), choose the component-history finger, and freeze/unfreeze the display.

### Passing Custom Coordinates

If you have a sensor with a different pad geometry:

```python
import numpy as np

my_coords = {
    "thumb": np.loadtxt("my_thumb_coords.csv", delimiter=",", skiprows=1),  # (127, 3)
    "index": np.loadtxt("my_index_coords.csv", delimiter=",", skiprows=1),  # (52, 3)
}
viz = PaxiniQtVisualizer(sensor.read_latest, coords=my_coords)
```

Fingers with no entry in `coords` will leave the tactile map empty until
coordinates are provided.

---

## Common Issues

### No frames received / `TimeoutError`

- Check DTR is `False` and RTS is `True` (most common issue).
- Verify baud rate is 921600 — the board does not auto-negotiate.
- Try increasing `serial_settle_s` to 1.5 if the CDC driver is slow to enumerate.
- Try `response_timeout_s=3.0`, `startup_attempts=5`, or `discard_startup_frames=0` while debugging startup.
- Run `python -m serial.tools.miniterm /dev/ttyACM0 921600` and press enter — you should see binary garbage if the board is alive.

### `RuntimeError: Finger validation failed`

A configured finger is not physically plugged into the board. The error names the specific finger. Either connect the missing finger or remove it from `PaxiniConfig.fingers`.

### Fz values look wrong / always zero

Try setting `signed_z=True` in `PaxiniConfig`. Paxini's protocol document specifies Fz as unsigned, but some board firmware versions return it signed.

### High noise

Increase `median_window` in `PaxiniConfig` (e.g. 5–7). At 83 Hz a window of 5 introduces ~60 ms of lag.
