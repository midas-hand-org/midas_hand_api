"""Live Dash visualizer for Paxini force data.

Accepts any callable returning ``dict[str, ndarray (N, 3)]`` — typically
``PaxiniHandSensor.read_latest`` or ``MidasHand.read_tactile`` — so it stays
decoupled from the sensor.
Coordinates are auto-detected from array shape (127 pts → 26 mm thumb CSV,
52 pts → 15 mm finger CSV) or supplied explicitly via ``coords``.
The browser UI can record one tactile sequence to CSV and replay the last
recording.

Usage::

    with PaxiniHandSensor(PaxiniConfig(port="/dev/ttyACM0")) as sensor:
        PaxiniVisualizer(sensor.read_latest).run()   # blocks, open http://127.0.0.1:8050

    # Non-blocking inside a motion script:
    viz = PaxiniVisualizer(sensor.read_latest)
    viz.start_background()   # returns immediately

Requires optional deps::

    pip install dash plotly
"""

from __future__ import annotations

import csv
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np


_ASSET_DIR = Path(__file__).resolve().parents[1] / "assets"
_COORD_CSV: dict[int, Path] = {
    127: _ASSET_DIR / "paxini_fingertip_26mm_127pts.csv",
    52: _ASSET_DIR / "paxini_fingertip_15mm_52pts.csv",
}
_DARK_BG = "#080c14"
_PANEL_BG = "#101826"
_GRID_COLOR = "#273244"
_TEXT_COLOR = "#dbe7f3"
_MUTED_TEXT_COLOR = "#8fa3b8"
_FORCE_COLORSCALE = [
    [0.0, "#1e63ff"],
    [0.35, "#10b7ff"],
    [0.7, "#ffb000"],
    [1.0, "#ff2a2a"],
]
_HISTORY_COLORS = {
    "thumb": "#6dd3ff",
    "index": "#ffb84d",
    "middle": "#75e08f",
    "ring": "#ff6fae",
}
_COMPONENT_OPTIONS = (
    ("|F|", "|F|"),
    ("|Fz|", "Fz"),
    ("|Fx|", "Fx"),
    ("|Fy|", "Fy"),
)


def _load_coords(path: Path) -> np.ndarray:
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rows.append((float(row["X"]), float(row["Y"]), float(row["Z"])))
    if not rows:
        raise ValueError(f"Empty coordinate file: {path}")
    return np.asarray(rows, dtype=np.float64)


def _auto_coords(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    cache: dict[int, np.ndarray] = {}
    result: dict[str, np.ndarray] = {}
    for name, arr in data.items():
        n = arr.shape[0]
        if n not in cache:
            path = _COORD_CSV.get(n)
            if path and path.exists():
                cache[n] = _load_coords(path)
        if n in cache:
            result[name] = cache[n]
    return result


def _first_sample(
    get_data: Callable[[], dict[str, np.ndarray]],
    retries: int = 30,
    interval_s: float = 0.1,
) -> dict[str, np.ndarray]:
    """Poll until one valid sample is available."""
    for _ in range(retries):
        try:
            return get_data()
        except RuntimeError:
            time.sleep(interval_s)
    raise RuntimeError(
        f"No sensor data after {retries * interval_s:.1f}s. "
        "Ensure the sensor is connected before constructing PaxiniVisualizer."
    )


class PaxiniVisualizer:
    """Live Dash visualizer for Paxini distributed force data.

    Args:
        get_data: Callable returning ``dict[str, ndarray (N, 3)]``.
            Pass ``sensor.read_latest`` or ``hand.read_tactile`` directly.
        coords: Optional per-finger coordinate arrays ``(N, 3)`` in mm.
            If omitted, auto-detected from array shape.
        update_ms: Browser refresh interval in milliseconds.
        component: Default non-negative force intensity shown on startup.
            One of ``"Fz"``, ``"|F|"``, ``"Fx"``, ``"Fy"``. Tangential
            components use absolute value for color intensity.
        history_len: Number of samples kept in the rolling history plot.
        history_update_ms: Browser refresh interval for the history plots.
            Keep this slower than ``update_ms`` for smoother tactile maps.
        show_arrows: Draw regional summed force arrows on the tactile maps.
        arrow_region_size: Approximate force-point region size for arrows.
        arrow_min_force_n: Minimum regional summed force before drawing an arrow.
        recording_csv: CSV file overwritten when the browser Stop button ends
            a recording.
    """

    def __init__(
        self,
        get_data: Callable[[], dict[str, np.ndarray]],
        *,
        coords: Optional[dict[str, np.ndarray]] = None,
        update_ms: int = 100,
        component: str = "Fz",
        history_len: int = 300,
        history_update_ms: int = 300,
        show_arrows: bool = True,
        arrow_region_size: int = 3,
        arrow_min_force_n: float = 0.25,
        recording_csv: str | Path = "paxini_recording.csv",
    ) -> None:
        self._get_data = get_data
        self._coords = coords
        self._update_ms = update_ms
        self._default_component = component
        self._history_len = history_len
        self._history_update_ms = history_update_ms
        self._show_arrows = show_arrows
        self._arrow_region_size = arrow_region_size
        self._arrow_min_force_n = arrow_min_force_n
        self._recording_csv = Path(recording_csv).expanduser()

    def run(self, host: str = "127.0.0.1", port: int = 8050) -> None:
        """Start Dash and block until the process exits (Ctrl-C)."""
        print(f"Paxini visualizer → http://{host}:{port}")
        self._build_app().run(host=host, port=port, debug=False)

    def start_background(self, host: str = "127.0.0.1", port: int = 8050) -> None:
        """Start Dash in a daemon thread and return immediately."""
        app = self._build_app()
        threading.Thread(
            target=lambda: app.run(host=host, port=port, debug=False),
            name="paxini-viz",
            daemon=True,
        ).start()
        print(f"Paxini visualizer → http://{host}:{port}")

    # ------------------------------------------------------------------

    def _build_app(self):
        try:
            from dash import Dash, Input, Output, dcc, html, ctx
            import plotly.graph_objects as go
        except ImportError as exc:
            raise SystemExit("Visualization requires: pip install dash plotly") from exc

        get_data = self._get_data
        recording_csv = self._recording_csv

        # Fetch one real sample so we know finger names before building the layout.
        # Dash requires all callback output IDs to exist in the initial layout.
        initial = _first_sample(get_data)
        fingers = list(initial.keys())
        coords = self._coords if self._coords is not None else _auto_coords(initial)
        history: list[dict] = []
        history_lock = threading.Lock()
        recording_lock = threading.Lock()
        recording_state = {
            "mode": "live",
            "current": [],
            "last": [],
            "replay_samples": [],
            "replay_start_s": None,
            "message": f"Ready. Recording file: {recording_csv}",
        }
        graph_config = {"displayModeBar": False, "responsive": True}
        button_style = {
            "backgroundColor": "#1a2638",
            "border": "1px solid #33445d",
            "borderRadius": "6px",
            "color": _TEXT_COLOR,
            "cursor": "pointer",
            "fontSize": "13px",
            "height": "32px",
            "padding": "0 14px",
        }

        app = Dash(__name__)
        app.layout = html.Div(
            [
                html.Div(
                    [
                        html.H3(
                            "Paxini Live Force",
                            style={"margin": "0 0 2px 0", "color": _TEXT_COLOR},
                        ),
                        html.Div(
                            id="status",
                            style={"fontSize": "12px", "color": _MUTED_TEXT_COLOR},
                        ),
                    ],
                    style={"padding": "10px 16px 4px", "backgroundColor": _DARK_BG},
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Label(
                                    "Map intensity",
                                    style={
                                        "fontSize": "12px",
                                        "color": _MUTED_TEXT_COLOR,
                                        "display": "block",
                                        "marginBottom": "4px",
                                    },
                                ),
                                dcc.Dropdown(
                                    id="component",
                                    options=[
                                        {"label": label, "value": value}
                                        for label, value in _COMPONENT_OPTIONS
                                    ],
                                    value=self._default_component,
                                    clearable=False,
                                    style={"width": "130px", "fontSize": "13px"},
                                ),
                            ]
                        ),
                        html.Div(
                            [
                                html.Label(
                                    "Component history",
                                    style={
                                        "fontSize": "12px",
                                        "color": _MUTED_TEXT_COLOR,
                                        "display": "block",
                                        "marginBottom": "4px",
                                    },
                                ),
                                dcc.Dropdown(
                                    id="history-finger",
                                    options=[
                                        {"label": name.title(), "value": name}
                                        for name in fingers
                                    ],
                                    value=fingers[0] if fingers else None,
                                    clearable=False,
                                    style={"width": "160px", "fontSize": "13px"},
                                ),
                            ]
                        ),
                        html.Div(
                            [
                                html.Button(
                                    "Record",
                                    id="record-button",
                                    n_clicks=0,
                                    style={**button_style, "backgroundColor": "#173321"},
                                ),
                                html.Button(
                                    "Stop",
                                    id="stop-button",
                                    n_clicks=0,
                                    style={**button_style, "backgroundColor": "#3a2020"},
                                ),
                                html.Button(
                                    "Replay",
                                    id="replay-button",
                                    n_clicks=0,
                                    style=button_style,
                                ),
                            ],
                            style={
                                "display": "flex",
                                "gap": "8px",
                                "alignItems": "end",
                            },
                        ),
                        html.Div(
                            id="record-status",
                            style={
                                "color": _MUTED_TEXT_COLOR,
                                "fontSize": "12px",
                                "minWidth": "280px",
                            },
                        ),
                    ],
                    style={
                        "display": "flex",
                        "gap": "14px",
                        "alignItems": "end",
                        "padding": "0 16px 10px",
                        "backgroundColor": _DARK_BG,
                    },
                ),
                html.Div(
                    [
                        dcc.Graph(
                            id=f"digit-{name}",
                            style={"height": "40vh"},
                            config=graph_config,
                        )
                        for name in fingers
                    ],
                    style={
                        "display": "grid",
                        "gridTemplateColumns": f"repeat({max(1, min(len(fingers), 4))}, 1fr)",
                        "gap": "6px",
                        "padding": "0 8px",
                        "backgroundColor": _DARK_BG,
                    },
                ),
                html.Div(
                    [
                        dcc.Graph(
                            id="single-finger-history",
                            style={"height": "24vh", "minWidth": "0"},
                            config=graph_config,
                        ),
                        dcc.Graph(
                            id="total-history",
                            style={"height": "24vh", "minWidth": "0"},
                            config=graph_config,
                        ),
                    ],
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "1fr 1fr",
                        "gap": "8px",
                        "padding": "8px",
                        "backgroundColor": _DARK_BG,
                    },
                ),
                dcc.Interval(id="map-tick", interval=self._update_ms),
                dcc.Interval(id="history-tick", interval=self._history_update_ms),
            ],
            style={
                "fontFamily": "Arial, sans-serif",
                "backgroundColor": _DARK_BG,
                "minHeight": "100vh",
            },
        )

        outputs = [Output("status", "children")]
        outputs += [Output(f"digit-{name}", "figure") for name in fingers]

        @app.callback(
            outputs,
            Input("map-tick", "n_intervals"),
            Input("component", "value"),
        )
        def _update_maps(_tick, component):
            display_source = "live"
            with recording_lock:
                mode = recording_state["mode"]

            if mode == "replay":
                with recording_lock:
                    data, replay_done = _recording_replay_frame(recording_state)
                    if replay_done:
                        recording_state["mode"] = "live"
                        recording_state["message"] = "Replay finished."
                display_source = "replay"
            else:
                try:
                    data = get_data()
                except RuntimeError:
                    data = None
                if data is not None:
                    with recording_lock:
                        if recording_state["mode"] == "recording":
                            recording_state["current"].append(
                                _recording_sample(time.time(), data, fingers)
                            )

            if data is not None:
                with history_lock:
                    history.append(_history_sample(time.time(), data, fingers))
                    if len(history) > self._history_len:
                        del history[: len(history) - self._history_len]

            with recording_lock:
                record_text = _recording_status_text(recording_state, recording_csv)
            if data is not None:
                status = f"source={display_source} fingers: {', '.join(fingers)} | {record_text}"
            else:
                status = f"Waiting for sensor data... | {record_text}"
            figs = [
                _finger_fig(
                    name,
                    data.get(name) if data else None,
                    component,
                    coords.get(name),
                    go,
                    show_arrows=self._show_arrows,
                    arrow_region_size=self._arrow_region_size,
                    arrow_min_force_n=self._arrow_min_force_n,
                )
                for name in fingers
            ]
            return [status, *figs]

        @app.callback(
            Output("record-status", "children"),
            Input("map-tick", "n_intervals"),
            Input("record-button", "n_clicks"),
            Input("stop-button", "n_clicks"),
            Input("replay-button", "n_clicks"),
        )
        def _record_control(_tick, _record_clicks, _stop_clicks, _replay_clicks):
            triggered_id = ctx.triggered_id
            samples_to_save = None
            message = None
            clear_history = False

            with recording_lock:
                if triggered_id == "record-button":
                    recording_state["mode"] = "recording"
                    recording_state["current"] = []
                    recording_state["replay_samples"] = []
                    recording_state["replay_start_s"] = None
                    recording_state["message"] = "Recording..."
                    clear_history = True
                elif triggered_id == "stop-button":
                    if recording_state["mode"] == "recording":
                        samples_to_save = list(recording_state["current"])
                        recording_state["last"] = samples_to_save
                        recording_state["mode"] = "live"
                        recording_state["current"] = []
                    elif recording_state["mode"] == "replay":
                        recording_state["mode"] = "live"
                        recording_state["message"] = "Replay stopped."
                    else:
                        recording_state["message"] = "Not recording."
                elif triggered_id == "replay-button":
                    if recording_state["mode"] == "recording":
                        recording_state["message"] = "Stop recording before replay."
                    else:
                        samples = list(recording_state["last"])
                        if not samples:
                            samples = _load_recording_csv(recording_csv)
                        if samples:
                            recording_state["last"] = samples
                            recording_state["replay_samples"] = samples
                            recording_state["replay_start_s"] = time.monotonic()
                            recording_state["mode"] = "replay"
                            recording_state["message"] = (
                                f"Replaying {len(samples)} frames from {recording_csv}."
                            )
                            clear_history = True
                        else:
                            recording_state["message"] = f"No recording found at {recording_csv}."

                message = _recording_status_text(recording_state, recording_csv)

            if clear_history:
                with history_lock:
                    history.clear()

            if samples_to_save is not None:
                try:
                    _write_recording_csv(samples_to_save, recording_csv)
                    message = f"Saved {len(samples_to_save)} frames to {recording_csv}."
                except OSError as exc:
                    message = f"Could not save recording: {exc}"
                with recording_lock:
                    recording_state["message"] = message

            return message

        @app.callback(
            Output("single-finger-history", "figure"),
            Output("total-history", "figure"),
            Input("history-tick", "n_intervals"),
            Input("history-finger", "value"),
        )
        def _update_history(_tick, history_finger):
            with history_lock:
                history_snapshot = list(history)
            finger = history_finger or (fingers[0] if fingers else "")
            return (
                _single_finger_history_fig(history_snapshot, finger, go),
                _total_history_fig(history_snapshot, fingers, go),
            )

        return app


# ---------------------------------------------------------------------------
# Pure figure helpers
# ---------------------------------------------------------------------------


def _finger_fig(
    name: str,
    vectors: Optional[np.ndarray],
    component: str,
    coords_mm: Optional[np.ndarray],
    go,
    *,
    show_arrows: bool = True,
    arrow_region_size: int = 3,
    arrow_min_force_n: float = 0.25,
):
    base_layout = dict(
        title=f"{name.title()} — {_component_label(component)} intensity",
        template="plotly_dark",
        paper_bgcolor=_PANEL_BG,
        plot_bgcolor=_PANEL_BG,
        font=dict(color=_TEXT_COLOR),
        margin=dict(l=35, r=10, t=35, b=30),
        showlegend=False,
        uirevision=name,
        transition=dict(duration=0),
        xaxis=dict(gridcolor=_GRID_COLOR, zerolinecolor=_GRID_COLOR),
        yaxis=dict(gridcolor=_GRID_COLOR, zerolinecolor=_GRID_COLOR),
    )
    if vectors is None or coords_mm is None:
        fig = go.Figure()
        fig.add_annotation(
            text="Waiting..." if vectors is None else "No coordinates",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(color=_MUTED_TEXT_COLOR),
        )
        fig.update_layout(**base_layout)
        return fig

    values = _component_intensity(vectors, component)
    magnitudes = np.linalg.norm(vectors, axis=1)

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=coords_mm[:, 0],
            y=coords_mm[:, 1],
            mode="markers",
            marker=dict(
                size=9 if len(vectors) > 80 else 11,
                color=values,
                cmin=0.0,
                cmax=_force_cmax(values),
                colorscale=_FORCE_COLORSCALE,
                colorbar=dict(
                    title=dict(
                        text=f"{_component_label(component)} (N)",
                        font=dict(color=_TEXT_COLOR),
                    ),
                    thickness=10,
                    len=0.75,
                    tickfont=dict(color=_TEXT_COLOR),
                ),
                line=dict(color="#06111f", width=0.5),
            ),
            customdata=np.column_stack([vectors, magnitudes]),
            hovertemplate=(
                "Fx=%{customdata[0]:.2f} N  "
                "Fy=%{customdata[1]:.2f} N  "
                "Fz=%{customdata[2]:.2f} N  "
                "|F|=%{customdata[3]:.2f} N<extra></extra>"
            ),
        )
    )
    if show_arrows:
        arrow_trace = _force_arrow_trace(
            coords_mm,
            vectors,
            go,
            region_size=arrow_region_size,
            min_force_n=arrow_min_force_n,
        )
        if arrow_trace is not None:
            fig.add_trace(arrow_trace)
    fig.update_layout(xaxis_title="X (mm)", yaxis_title="Y (mm)", **base_layout)
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def _single_finger_history_fig(history: list, finger: str, go):
    base_layout = _history_layout(
        title=f"{finger.title()} Fx/Fy/Fz History",
        yaxis_title="ΣF component (N)",
    )
    if not history:
        return go.Figure(layout=go.Layout(**base_layout))

    t0 = history[0]["time"]
    times = [sample["time"] - t0 for sample in history]
    fig = go.Figure()
    component_specs = (
        ("Fx", 0, "#6dd3ff"),
        ("Fy", 1, "#ffb84d"),
        ("Fz", 2, "#75e08f"),
    )
    for label, index, color in component_specs:
        vals = [
            sample["components"].get(finger, (np.nan, np.nan, np.nan))[index]
            for sample in history
        ]
        fig.add_trace(
            go.Scattergl(
                x=times,
                y=vals,
                mode="lines",
                name=label,
                line=dict(color=color, width=2),
            )
        )
    fig.update_layout(**base_layout)
    return fig


def _total_history_fig(history: list, fingers: list[str], go):
    base_layout = dict(
        title="Total Distributed Force History",
        template="plotly_dark",
        paper_bgcolor=_PANEL_BG,
        plot_bgcolor=_PANEL_BG,
        font=dict(color=_TEXT_COLOR),
        xaxis_title="time (s)",
        yaxis_title="Σ|F| (N)",
        margin=dict(l=40, r=10, t=35, b=35),
        uirevision="history",
        transition=dict(duration=0),
        xaxis=dict(gridcolor=_GRID_COLOR, zerolinecolor=_GRID_COLOR),
        yaxis=dict(gridcolor=_GRID_COLOR, zerolinecolor=_GRID_COLOR),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
    )
    if not history:
        return go.Figure(layout=go.Layout(**base_layout))

    t0 = history[0]["time"]
    times = [sample["time"] - t0 for sample in history]
    fig = go.Figure()
    for name in fingers:
        vals = [sample["totals"].get(name, float("nan")) for sample in history]
        fig.add_trace(
            go.Scattergl(
                x=times,
                y=vals,
                mode="lines",
                name=name,
                line=dict(color=_HISTORY_COLORS.get(name), width=2),
            )
        )
    fig.update_layout(**base_layout)
    return fig


def _history_sample(
    timestamp_s: float,
    data: dict[str, np.ndarray],
    fingers: list[str],
) -> dict:
    components = {}
    totals = {}
    for name in fingers:
        vectors = data.get(name)
        if vectors is None:
            continue
        components[name] = tuple(float(v) for v in np.sum(vectors, axis=0))
        totals[name] = float(np.sum(np.linalg.norm(vectors, axis=1)))
    return {"time": timestamp_s, "components": components, "totals": totals}


def _recording_sample(
    timestamp_s: float,
    data: dict[str, np.ndarray],
    fingers: list[str],
) -> dict:
    copied = {}
    for name in fingers:
        vectors = data.get(name)
        if vectors is not None:
            copied[name] = np.asarray(vectors, dtype=np.float64).copy()
    return {"time": timestamp_s, "data": copied}


def _recording_status_text(recording_state: dict, recording_csv: Path) -> str:
    mode = recording_state["mode"]
    if mode == "recording":
        return f"Recording {len(recording_state['current'])} frames..."
    if mode == "replay":
        return f"Replaying {len(recording_state['replay_samples'])} frames..."
    return recording_state.get("message") or f"Ready. Recording file: {recording_csv}"


def _recording_replay_frame(recording_state: dict) -> tuple[Optional[dict[str, np.ndarray]], bool]:
    samples = recording_state["replay_samples"]
    if not samples:
        return None, True

    replay_start_s = recording_state.get("replay_start_s")
    if replay_start_s is None:
        replay_start_s = time.monotonic()
        recording_state["replay_start_s"] = replay_start_s

    elapsed_s = time.monotonic() - replay_start_s
    t0 = samples[0]["time"]
    relative_times = [sample["time"] - t0 for sample in samples]
    duration_s = relative_times[-1] if relative_times else 0.0
    if duration_s <= 0.0:
        return samples[-1]["data"], elapsed_s > 0.5

    index = int(np.searchsorted(relative_times, elapsed_s, side="right") - 1)
    index = int(np.clip(index, 0, len(samples) - 1))
    done = elapsed_s >= duration_s
    return samples[index]["data"], done


def _write_recording_csv(samples: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_index", "time_s", "finger", "point_index", "fx_n", "fy_n", "fz_n"])
        if not samples:
            return

        t0 = samples[0]["time"]
        for sample_index, sample in enumerate(samples):
            time_s = sample["time"] - t0
            for finger, vectors in sample["data"].items():
                for point_index, vector in enumerate(vectors):
                    writer.writerow(
                        [
                            sample_index,
                            f"{time_s:.9f}",
                            finger,
                            point_index,
                            f"{float(vector[0]):.9f}",
                            f"{float(vector[1]):.9f}",
                            f"{float(vector[2]):.9f}",
                        ]
                    )


def _load_recording_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []

    grouped: dict[int, dict] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            sample_index = int(row["sample_index"])
            point_index = int(row["point_index"])
            entry = grouped.setdefault(
                sample_index,
                {"time": float(row["time_s"]), "points": {}},
            )
            entry["points"].setdefault(row["finger"], []).append(
                (
                    point_index,
                    (
                        float(row["fx_n"]),
                        float(row["fy_n"]),
                        float(row["fz_n"]),
                    ),
                )
            )

    samples = []
    for sample_index in sorted(grouped):
        entry = grouped[sample_index]
        data = {}
        for finger, points in entry["points"].items():
            points.sort(key=lambda item: item[0])
            data[finger] = np.asarray([vector for _point, vector in points], dtype=np.float64)
        samples.append({"time": entry["time"], "data": data})
    return samples


def _history_layout(title: str, yaxis_title: str) -> dict:
    return dict(
        title=title,
        template="plotly_dark",
        paper_bgcolor=_PANEL_BG,
        plot_bgcolor=_PANEL_BG,
        font=dict(color=_TEXT_COLOR),
        xaxis_title="time (s)",
        yaxis_title=yaxis_title,
        margin=dict(l=45, r=10, t=35, b=35),
        uirevision=title,
        transition=dict(duration=0),
        xaxis=dict(gridcolor=_GRID_COLOR, zerolinecolor=_GRID_COLOR),
        yaxis=dict(gridcolor=_GRID_COLOR, zerolinecolor=_GRID_COLOR),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
    )


def _component_intensity(vectors: np.ndarray, component: str) -> np.ndarray:
    if component == "|F|":
        return np.linalg.norm(vectors, axis=1)
    index = {"Fx": 0, "Fy": 1, "Fz": 2}[component]
    return np.abs(vectors[:, index])


def _component_label(component: str) -> str:
    if component == "|F|":
        return "|F|"
    return f"|{component}|"


def _force_cmax(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 1.0
    return max(1.0, float(np.percentile(finite, 95)))


def _force_arrow_trace(
    coords_mm: np.ndarray,
    vectors: np.ndarray,
    go,
    *,
    region_size: int = 3,
    min_force_n: float = 0.25,
):
    regions = _regional_force_vectors(coords_mm, vectors, region_size=region_size)
    if not regions:
        return None

    x_span = float(np.ptp(coords_mm[:, 0]))
    y_span = float(np.ptp(coords_mm[:, 1]))
    span = max(x_span, y_span, 1.0)
    min_length = 0.025 * span
    max_length = 0.12 * span
    max_force = max(region["force_norm"] for region in regions)
    line_x = []
    line_y = []
    for region in regions:
        tangent_norm = region["tangent_norm"]
        force_norm = region["force_norm"]
        if force_norm < min_force_n:
            continue
        if tangent_norm >= 1e-6:
            direction = region["force_xy"] / tangent_norm
        elif abs(region["force_z"]) >= 1e-6:
            direction = np.array([0.0, np.sign(region["force_z"])], dtype=np.float64)
        else:
            continue
        normalized = np.sqrt(force_norm / max(max_force, 1e-6))
        length = min_length + (max_length - min_length) * float(np.clip(normalized, 0.0, 1.0))
        start = region["center"] - 0.5 * length * direction
        end = region["center"] + 0.5 * length * direction
        perp = np.array([-direction[1], direction[0]], dtype=np.float64)
        head_length = min(0.34 * length, 0.035 * span)
        head_width = 0.18 * length
        head_base = end - head_length * direction
        head_left = head_base + head_width * perp
        head_right = head_base - head_width * perp

        line_x.extend([start[0], end[0], None])
        line_y.extend([start[1], end[1], None])
        line_x.extend([end[0], head_left[0], None, end[0], head_right[0], None])
        line_y.extend([end[1], head_left[1], None, end[1], head_right[1], None])

    if not line_x:
        return None

    return go.Scattergl(
        x=line_x,
        y=line_y,
        mode="lines",
        line=dict(color="rgba(255, 220, 130, 0.92)", width=2.4),
        hoverinfo="skip",
        name="regional force",
    )


def _regional_force_vectors(
    coords_mm: np.ndarray,
    vectors: np.ndarray,
    *,
    region_size: int,
) -> list[dict]:
    n_points = len(coords_mm)
    if n_points == 0:
        return []

    x_bins, y_bins = _region_bins(coords_mm, region_size=region_size)
    x_indices = _bin_indices(coords_mm[:, 0], x_bins)
    y_indices = _bin_indices(coords_mm[:, 1], y_bins)
    regions = []
    for x_bin in range(x_bins):
        for y_bin in range(y_bins):
            mask = (x_indices == x_bin) & (y_indices == y_bin)
            if not np.any(mask):
                continue
            region_vectors = vectors[mask]
            force = np.sum(region_vectors, axis=0)
            force_xy = force[:2]
            regions.append(
                {
                    "center": np.mean(coords_mm[mask, :2], axis=0),
                    "force_xy": force_xy,
                    "force_z": float(force[2]),
                    "tangent_norm": float(np.linalg.norm(force_xy)),
                    "force_norm": float(np.sum(np.linalg.norm(region_vectors, axis=1))),
                }
            )
    return regions


def _region_bins(coords_mm: np.ndarray, *, region_size: int) -> tuple[int, int]:
    n_points = max(len(coords_mm), 1)
    approx_columns = max(1, int(round(np.sqrt(n_points))))
    approx_rows = max(1, int(np.ceil(n_points / approx_columns)))
    return (
        max(1, int(np.ceil(approx_columns / max(region_size, 1)))),
        max(1, int(np.ceil(approx_rows / max(region_size, 1)))),
    )


def _bin_indices(values: np.ndarray, n_bins: int) -> np.ndarray:
    if n_bins <= 1 or float(np.ptp(values)) < 1e-9:
        return np.zeros(len(values), dtype=np.int64)
    normalized = (values - float(np.min(values))) / float(np.ptp(values))
    return np.clip((normalized * n_bins).astype(np.int64), 0, n_bins - 1)
