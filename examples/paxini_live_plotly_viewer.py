"""Live Plotly viewer for one Paxini PX-6AX GEN3 USB/serial sensor.

Run after installing the optional visualization dependencies:

    python -m pip install -e ".[viz]"
    python examples/paxini_live_plotly_viewer.py --port /dev/ttyUSB0 --sensor-id 2

The distributed-force points can be displayed on a model-specific approximate
physical layout or on an index grid. The Paxini manual shows the DP-S2015-Elite
force-signal locations on the fingertip surface, but it does not label each
point's numeric index, so use the flip/serpentine controls to empirically align
the display with touches on known locations.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import serial

EXAMPLES_DIR = Path(__file__).resolve().parent
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from Read_Single_Sensor_Usb import (  # noqa: E402
    BAUDRATE,
    DEFAULT_FORCE_POINTS,
    DEFAULT_SENSOR_ID,
    calibrate_sensor,
    device_address_from_sensor_id,
    read_distributed_force,
    read_resultant_force,
)


Dash = None
dcc = None
html = None
Input = None
Output = None
go = None

FORCE_COLOR_MIN_N = 0.0
FORCE_COLOR_MAX_N = 20.0
DEFAULT_LAYOUT_MODE = "grid"
DEFAULT_READ_INTERVAL_S = 0.05
DEFAULT_UPDATE_MS = 100


def load_dash() -> None:
    """Import optional Dash/Plotly dependencies when the app starts."""

    global Dash, dcc, html, Input, Output, go
    if Dash is not None:
        return
    try:
        from dash import Dash as Dash_
        from dash import dcc as dcc_
        from dash import html as html_
        from dash.dependencies import Input as Input_
        from dash.dependencies import Output as Output_
        import plotly.graph_objects as go_
    except ImportError as exc:  # pragma: no cover - depends on optional package.
        raise SystemExit(
            "This viewer needs optional dependencies. Install them with:\n"
            '  python -m pip install -e ".[viz]"\n'
            "or:\n"
            "  python -m pip install dash plotly"
        ) from exc

    Dash = Dash_
    dcc = dcc_
    html = html_
    Input = Input_
    Output = Output_
    go = go_


MODEL_FORCE_POINTS = {
    "DP-S1813-Elite": 31,
    "DP-S2015-Elite": 52,
    "IP-S1610-Elite": 25,
    "MC-M2020-Elite": 9,
    "DP-S1813-Core": 51,
    "DP-S2716-Core": 116,
    "DP-S3013-Core": 96,
    "IP-M2324-Core": 68,
    "CP-M3025-Core": 77,
    "DP-M2826-Omega": 127,
    "DP-L3530-Omega": 135,
    "CP-L5325-Omega": 239,
}

MODEL_LAYOUT_ROWS = {
    # Derived from the DP-S2015-Elite coordinate table. The explicit coordinate
    # map below is preferred; these counts are used only as a generated fallback.
    "DP-S2015-Elite": (7, 7, 7, 7, 7, 7, 7, 3),
}

MODEL_LAYOUT_ROW_GROUPS = {
    # Zero-based taxel indexes grouped by tail-to-fingertip rows, left to right.
    "DP-S2015-Elite": (
        (0, 5, 14, 22, 34, 41, 51),
        (1, 6, 15, 23, 30, 42, 48),
        (2, 7, 16, 24, 35, 43, 49),
        (3, 8, 17, 25, 36, 44, 50),
        (4, 10, 18, 26, 31, 45, 46),
        (9, 11, 19, 27, 32, 39, 47),
        (12, 13, 20, 28, 33, 38, 40),
        (21, 29, 37),
    ),
}

MODEL_LAYOUT_POINTS_XY = {
    # Coordinate table from examples/temp.txt. Index N in the vendor table maps
    # to zero-based force point N - 1 in the live reader output.
    "DP-S2015-Elite": (
        (-7.45866948, -0.80442889),
        (-7.24032101, 1.9370259),
        (-7.00988064, 5.22921165),
        (-6.82610763, 7.97333403),
        (-6.53146658, 11.26026387),
        (-6.00006622, -1.35478789),
        (-6.00372647, 1.50179184),
        (-5.94055243, 4.93336235),
        (-5.81660022, 7.79060929),
        (-6.07106495, 13.96973406),
        (-5.51333141, 11.20680746),
        (-5.0017979, 14.00765355),
        (-5.09707632, 16.57086028),
        (-4.40866763, 16.2194301),
        (-3.1037709, -1.55570508),
        (-3.15461212, 1.5038129),
        (-3.20348468, 5.19993),
        (-3.21601164, 8.3044369),
        (-3.1409429, 11.42030336),
        (-2.83527189, 15.06167934),
        (-2.42756984, 17.70908967),
        (-2.71496388, 18.81872693),
        (0.00000238, -1.55808447),
        (0.00000199, 1.60830268),
        (0.00000124, 5.43294958),
        (-0.00000002, 8.63424733),
        (-0.00000039, 12.48579856),
        (-0.00000042, 15.63993516),
        (0.00000008, 18.02972832),
        (0.00000158, 19.32616275),
        (3.15461212, 1.5038129),
        (3.1409429, 11.42030336),
        (2.83527189, 15.06167934),
        (2.42756984, 17.70908967),
        (3.1037709, -1.55570508),
        (3.20348468, 5.19993),
        (3.21601164, 8.3044369),
        (2.71496388, 18.81872693),
        (4.40866763, 16.2194301),
        (5.0017979, 14.00765355),
        (5.09707632, 16.57086028),
        (6.00006622, -1.35478789),
        (6.00372647, 1.50179184),
        (5.94055243, 4.93336235),
        (5.81660022, 7.79060929),
        (5.51333141, 11.20680746),
        (6.53146658, 11.26026387),
        (6.07106495, 13.96973406),
        (7.24032101, 1.9370259),
        (7.00988064, 5.22921165),
        (6.82610763, 7.97333403),
        (7.45866948, -0.80442889),
    ),
}


def configure_http_logging() -> None:
    """Suppress Dash/Werkzeug per-request access logs."""

    logging.getLogger("werkzeug").setLevel(logging.WARNING)


def force_color_marker(
    values: np.ndarray,
    colorbar_title: str,
    size: int,
    *,
    showscale: bool = True,
    line: Optional[dict[str, object]] = None,
) -> dict[str, object]:
    marker: dict[str, object] = {
        "size": size,
        "color": values,
        "colorscale": "Viridis",
        "cmin": FORCE_COLOR_MIN_N,
        "cmax": FORCE_COLOR_MAX_N,
        "colorbar": {"title": colorbar_title},
    }
    if showscale:
        marker["showscale"] = True
    if line is not None:
        marker["line"] = line
    return marker


@dataclass(frozen=True)
class TactileSample:
    timestamp_s: float
    resultant_n: tuple[float, float, float]
    distributed_n: np.ndarray


class PaxiniReader:
    """Background serial reader for the Dash app."""

    def __init__(
        self,
        port: str,
        baudrate: int,
        sensor_id: int,
        force_points: int,
        model: str,
        interval_s: float,
        calibrate: bool,
    ) -> None:
        self.port = port
        self.baudrate = baudrate
        self.sensor_id = sensor_id
        self.device_address = device_address_from_sensor_id(sensor_id)
        self.force_points = force_points
        self.model = model
        self.interval_s = interval_s
        self.calibrate = calibrate

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._serial: Optional[serial.Serial] = None
        self.latest: Optional[TactileSample] = None
        self.error: Optional[str] = None
        self.history = deque(maxlen=300)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._serial and self._serial.is_open:
            self._serial.close()

    def snapshot(self) -> tuple[Optional[TactileSample], Optional[str], list[TactileSample]]:
        with self._lock:
            return self.latest, self.error, list(self.history)

    def _run(self) -> None:
        try:
            self._serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=1,
                write_timeout=0.5,
                inter_byte_timeout=0.001,
                xonxoff=False,
                rtscts=False,
            )
            if self.calibrate:
                calibrate_sensor(self._serial, self.device_address)

            while not self._stop.is_set():
                resultant = read_resultant_force(self._serial, self.device_address)
                distributed = np.asarray(
                    read_distributed_force(
                        self._serial,
                        self.device_address,
                        self.force_points,
                    ),
                    dtype=np.float64,
                )
                sample = TactileSample(
                    timestamp_s=time.time(),
                    resultant_n=resultant,
                    distributed_n=distributed,
                )
                with self._lock:
                    self.latest = sample
                    self.error = None
                    self.history.append(sample)
                time.sleep(self.interval_s)
        except Exception as exc:
            with self._lock:
                self.error = str(exc)


def make_grid_shape(point_count: int) -> tuple[int, int]:
    rows = max(1, int(math.floor(math.sqrt(point_count))))
    cols = int(math.ceil(point_count / rows))
    while rows * cols < point_count:
        rows += 1
    return rows, cols


def make_grid_positions(point_count: int) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    rows, cols = make_grid_shape(point_count)
    x = np.arange(point_count) % cols
    y = np.arange(point_count) // cols
    return x.astype(float), y.astype(float), (rows, cols)


def make_physical_positions(
    point_count: int,
    model: str,
    serpentine: bool,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    point_positions = MODEL_LAYOUT_POINTS_XY.get(model)
    if point_positions is not None and len(point_positions) == point_count:
        positions = np.asarray(point_positions, dtype=float)
        return positions[:, 0], -positions[:, 1]

    row_counts = MODEL_LAYOUT_ROWS.get(model)
    if row_counts is None or sum(row_counts) != point_count:
        return None

    x_values = []
    y_values = []
    max_count = max(row_counts)
    for row, count in enumerate(row_counts):
        # Center each variable-width row. The sensor origin in the manual is at
        # the connector/tail, so lower row numbers are drawn near the tail.
        x_row = np.linspace(-(count - 1) / 2.0, (count - 1) / 2.0, count)
        x_row *= max_count / count
        if serpentine and row % 2 == 1:
            x_row = x_row[::-1]
        y_row = np.full(count, row, dtype=float)
        x_values.extend(x_row.tolist())
        y_values.extend(y_row.tolist())
    return np.asarray(x_values, dtype=float), np.asarray(y_values, dtype=float)


def make_taxel_positions(
    point_count: int,
    model: str,
    layout_mode: str,
    serpentine: bool,
    flip_x: bool,
    flip_y: bool,
) -> tuple[np.ndarray, np.ndarray, str]:
    positions = None
    if layout_mode == "physical":
        positions = make_physical_positions(point_count, model, serpentine)
    if positions is None:
        x, y, _ = make_grid_positions(point_count)
        layout_name = "index grid"
    else:
        x, y = positions
        layout_name = f"{model} approximate physical layout"

    if flip_x:
        x = -x
    if flip_y:
        y = -y
    return x, y, layout_name


def empty_figure(title: str, message: str = "Waiting for tactile data...") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 18},
    )
    fig.update_layout(title=title, template="plotly_white", margin=dict(l=30, r=20, t=50, b=30))
    return fig


def make_taxel_map(
    sample: Optional[TactileSample],
    component_name: str,
    model: str,
    layout_mode: str,
    serpentine: bool,
    flip_x: bool,
    flip_y: bool,
) -> go.Figure:
    if sample is None:
        return empty_figure("Distributed Force Map")

    points = sample.distributed_n
    component_index = {"Fx": 0, "Fy": 1, "Fz": 2}[component_name]
    x, y, layout_name = make_taxel_positions(
        len(points),
        model,
        layout_mode,
        serpentine,
        flip_x,
        flip_y,
    )
    values = points[:, component_index]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers+text",
            text=[str(i) for i in range(len(points))],
            textposition="middle center",
            textfont={"size": 9, "color": "white"},
            marker=force_color_marker(
                values,
                f"{component_name} (N)",
                24,
                line={"color": "#1F2937", "width": 1},
            ),
            customdata=np.column_stack(
                [
                    np.arange(len(points)),
                    points[:, 0],
                    points[:, 1],
                    points[:, 2],
                ]
            ),
            hovertemplate=(
                "point=%{customdata[0]}<br>"
                "Fx=%{customdata[1]:.2f} N<br>"
                "Fy=%{customdata[2]:.2f} N<br>"
                "Fz=%{customdata[3]:.2f} N<extra></extra>"
            ),
            name="force points",
        )
    )
    fig.update_layout(
        title=f"{component_name} Distributed Force Map ({layout_name})",
        template="plotly_white",
        xaxis_title="sensor lateral axis",
        yaxis_title="tail to fingertip axis",
        yaxis_autorange="reversed",
        margin=dict(l=45, r=20, t=50, b=45),
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def make_vector_field(
    sample: Optional[TactileSample],
    model: str,
    layout_mode: str,
    serpentine: bool,
    flip_x: bool,
    flip_y: bool,
) -> go.Figure:
    if sample is None:
        return empty_figure("Force Vectors")

    points = sample.distributed_n
    x, y, layout_name = make_taxel_positions(
        len(points),
        model,
        layout_mode,
        serpentine,
        flip_x,
        flip_y,
    )
    fx = points[:, 0]
    fy = points[:, 1]
    fz = points[:, 2]
    magnitude = np.linalg.norm(points, axis=1)

    max_xy = max(float(np.max(np.hypot(fx, fy))), 1.0)
    scale = 0.35 / max_xy
    line_x = []
    line_y = []
    for x0, y0, dx, dy in zip(x, y, fx * scale, fy * scale):
        line_x.extend([x0, x0 + dx, None])
        line_y.extend([y0, y0 - dy, None])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=force_color_marker(fz, "Fz (N)", 10, showscale=True),
            text=[f"|F|={value:.2f} N" for value in magnitude],
            hovertemplate="point=%{pointNumber}<br>x=%{x}<br>y=%{y}<br>%{text}<extra></extra>",
            name="taxels",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=line_x,
            y=line_y,
            mode="lines",
            line={"color": "#D62728", "width": 2},
            hoverinfo="skip",
            name="Fx/Fy direction",
        )
    )
    fig.update_layout(
        title="Tangential Force Vectors With Fz Color",
        template="plotly_white",
        xaxis_title="grid column",
        yaxis_title="grid row",
        yaxis_autorange="reversed",
        margin=dict(l=45, r=20, t=50, b=45),
        annotations=[
            {
                "text": layout_name,
                "xref": "paper",
                "yref": "paper",
                "x": 0,
                "y": 1.08,
                "showarrow": False,
                "font": {"size": 12},
            }
        ],
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def make_resultant_history(history: list[TactileSample]) -> go.Figure:
    if not history:
        return empty_figure("Resultant Force History")

    t0 = history[0].timestamp_s
    times = [sample.timestamp_s - t0 for sample in history]
    values = np.asarray([sample.resultant_n for sample in history], dtype=np.float64)

    fig = go.Figure()
    for index, name in enumerate(("Fx", "Fy", "Fz")):
        fig.add_trace(go.Scatter(x=times, y=values[:, index], mode="lines", name=name))
    fig.update_layout(
        title="Resultant Force History",
        template="plotly_white",
        xaxis_title="time (s)",
        yaxis_title="force (N)",
        margin=dict(l=45, r=20, t=50, b=45),
    )
    return fig


def make_app(reader: PaxiniReader, update_ms: int) -> Dash:
    app = Dash(__name__)
    app.layout = html.Div(
        [
            html.Div(
                [
                    html.H2("Paxini Live Tactile Viewer"),
                    html.Div(id="status"),
                ],
                style={"padding": "12px 16px"},
            ),
            html.Div(
                [
                    dcc.Dropdown(
                        id="component",
                        options=[{"label": name, "value": name} for name in ("Fz", "Fx", "Fy")],
                        value="Fz",
                        clearable=False,
                        style={"width": "180px"},
                    ),
                    dcc.Dropdown(
                        id="layout-mode",
                        options=[
                            {"label": "Physical layout", "value": "physical"},
                            {"label": "Index grid", "value": "grid"},
                        ],
                        value=DEFAULT_LAYOUT_MODE,
                        clearable=False,
                        style={"width": "190px"},
                    ),
                    dcc.Checklist(
                        id="layout-options",
                        options=[
                            {"label": "Serpentine", "value": "serpentine"},
                            {"label": "Flip X", "value": "flip_x"},
                            {"label": "Flip Y", "value": "flip_y"},
                        ],
                        value=[],
                        inline=True,
                        style={"display": "flex", "gap": "14px", "alignItems": "center"},
                    ),
                ],
                style={
                    "padding": "0 16px 8px 16px",
                    "display": "flex",
                    "gap": "12px",
                    "alignItems": "center",
                },
            ),
            html.Div(
                [
                    dcc.Graph(id="taxel-map", style={"height": "46vh"}),
                    dcc.Graph(id="vectors", style={"height": "46vh"}),
                ],
                style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "8px"},
            ),
            dcc.Graph(id="history", style={"height": "32vh"}),
            dcc.Interval(id="tick", interval=update_ms, n_intervals=0),
        ],
        style={"fontFamily": "Arial, sans-serif"},
    )

    @app.callback(
        Output("status", "children"),
        Output("taxel-map", "figure"),
        Output("vectors", "figure"),
        Output("history", "figure"),
        Input("tick", "n_intervals"),
        Input("component", "value"),
        Input("layout-mode", "value"),
        Input("layout-options", "value"),
    )
    def update(_: int, component: str, layout_mode: str, layout_options: list[str]):
        layout_options = layout_options or []
        sample, error, history = reader.snapshot()
        serpentine = "serpentine" in layout_options
        flip_x = "flip_x" in layout_options
        flip_y = "flip_y" in layout_options
        if error:
            status = f"Error: {error}"
        elif sample:
            fx, fy, fz = sample.resultant_n
            status = (
                f"port={reader.port} sensor_id={reader.sensor_id} "
                f"points={len(sample.distributed_n)} "
                f"resultant=({fx:.2f}, {fy:.2f}, {fz:.2f}) N"
            )
        else:
            status = "Waiting for sensor data..."
        return (
            status,
            make_taxel_map(
                sample,
                component,
                reader.model,
                layout_mode,
                serpentine,
                flip_x,
                flip_y,
            ),
            make_vector_field(
                sample,
                reader.model,
                layout_mode,
                serpentine,
                flip_x,
                flip_y,
            ),
            make_resultant_history(history),
        )

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, help="Serial port, e.g. /dev/ttyUSB0")
    parser.add_argument("--baudrate", type=int, default=BAUDRATE)
    parser.add_argument("--sensor-id", type=int, default=DEFAULT_SENSOR_ID)
    parser.add_argument("--force-points", type=int, default=None)
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_FORCE_POINTS),
        default="DP-S2015-Elite",
        help="Used only when --force-points is not provided.",
    )
    parser.add_argument("--skip-calibration", action="store_true")
    parser.add_argument("--read-interval", type=float, default=DEFAULT_READ_INTERVAL_S)
    parser.add_argument("--update-ms", type=int, default=DEFAULT_UPDATE_MS)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port-web", type=int, default=8050)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_http_logging()
    load_dash()
    force_points = args.force_points or MODEL_FORCE_POINTS.get(
        args.model,
        DEFAULT_FORCE_POINTS,
    )
    reader = PaxiniReader(
        port=args.port,
        baudrate=args.baudrate,
        sensor_id=args.sensor_id,
        force_points=force_points,
        model=args.model,
        interval_s=args.read_interval,
        calibrate=not args.skip_calibration,
    )
    reader.start()
    app = make_app(reader, args.update_ms)
    try:
        app.run(host=args.host, port=args.port_web, debug=False)
    finally:
        reader.stop()


if __name__ == "__main__":
    main()
