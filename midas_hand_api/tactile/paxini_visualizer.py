"""Live Dash visualizer for Paxini force data.

Accepts any callable returning ``dict[str, ndarray (N, 3)]`` — typically
``PaxiniHandSensor.read_latest`` or ``MidasHand.read_tactile`` — so it stays
decoupled from the sensor.
Coordinates are auto-detected from array shape (127 pts → 26 mm thumb CSV,
52 pts → 15 mm finger CSV) or supplied explicitly via ``coords``.

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
        component: Default force component shown on startup.
            One of ``"Fz"``, ``"|F|"``, ``"Fx"``, ``"Fy"``.
        history_len: Number of samples kept in the rolling history plot.
    """

    def __init__(
        self,
        get_data: Callable[[], dict[str, np.ndarray]],
        *,
        coords: Optional[dict[str, np.ndarray]] = None,
        update_ms: int = 100,
        component: str = "Fz",
        history_len: int = 300,
    ) -> None:
        self._get_data = get_data
        self._coords = coords
        self._update_ms = update_ms
        self._default_component = component
        self._history_len = history_len

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
            from dash import Dash, Input, Output, dcc, html
            import plotly.graph_objects as go
        except ImportError as exc:
            raise SystemExit("Visualization requires: pip install dash plotly") from exc

        get_data = self._get_data

        # Fetch one real sample so we know finger names before building the layout.
        # Dash requires all callback output IDs to exist in the initial layout.
        initial = _first_sample(get_data)
        fingers = list(initial.keys())
        coords = self._coords if self._coords is not None else _auto_coords(initial)
        history: list[tuple[float, dict[str, np.ndarray]]] = []

        app = Dash(__name__)
        app.layout = html.Div(
            [
                html.Div(
                    [
                        html.H3("Paxini Live Force", style={"margin": "0 0 2px 0"}),
                        html.Div(id="status", style={"fontSize": "12px", "color": "#666"}),
                    ],
                    style={"padding": "10px 16px 4px"},
                ),
                html.Div(
                    dcc.Dropdown(
                        id="component",
                        options=[{"label": c, "value": c} for c in ("Fz", "|F|", "Fx", "Fy")],
                        value=self._default_component,
                        clearable=False,
                        style={"width": "120px", "fontSize": "13px"},
                    ),
                    style={"padding": "0 16px 6px"},
                ),
                html.Div(
                    [
                        dcc.Graph(id=f"digit-{name}", style={"height": "40vh"})
                        for name in fingers
                    ],
                    style={
                        "display": "grid",
                        "gridTemplateColumns": f"repeat({max(1, min(len(fingers), 4))}, 1fr)",
                        "gap": "6px",
                        "padding": "0 8px",
                    },
                ),
                dcc.Graph(id="history", style={"height": "22vh", "padding": "0 8px 8px"}),
                dcc.Interval(id="tick", interval=self._update_ms),
            ],
            style={"fontFamily": "Arial, sans-serif", "backgroundColor": "#fafafa"},
        )

        outputs = [Output("status", "children")]
        outputs += [Output(f"digit-{name}", "figure") for name in fingers]
        outputs += [Output("history", "figure")]

        @app.callback(outputs, Input("tick", "n_intervals"), Input("component", "value"))
        def _update(_tick, component):
            try:
                data = get_data()
            except RuntimeError:
                data = None

            if data is not None:
                history.append((time.time(), data))
                if len(history) > self._history_len:
                    del history[: len(history) - self._history_len]

            status = (
                f"fingers: {', '.join(fingers)}" if data is not None else "Waiting for sensor data..."
            )
            figs = [
                _finger_fig(name, data.get(name) if data else None, component, coords.get(name), go)
                for name in fingers
            ]
            return [status, *figs, _history_fig(history, fingers, go)]

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
):
    base_layout = dict(
        title=f"{name.title()} — {component}",
        template="plotly_white",
        margin=dict(l=35, r=10, t=35, b=30),
        showlegend=False,
        uirevision=name,
        transition=dict(duration=0),
    )
    if vectors is None or coords_mm is None:
        fig = go.Figure()
        fig.add_annotation(
            text="Waiting..." if vectors is None else "No coordinates",
            x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
        )
        fig.update_layout(**base_layout)
        return fig

    idx = {"Fx": 0, "Fy": 1, "Fz": 2, "|F|": None}[component]
    values = np.linalg.norm(vectors, axis=1) if idx is None else vectors[:, idx]

    fig = go.Figure(
        go.Scattergl(
            x=coords_mm[:, 0],
            y=coords_mm[:, 1],
            mode="markers",
            marker=dict(
                size=9 if len(vectors) > 80 else 11,
                color=values,
                colorscale="Viridis",
                colorbar=dict(title=f"{component} (N)", thickness=10, len=0.75),
                line=dict(color="#333", width=0.5),
            ),
            customdata=np.column_stack([vectors, np.linalg.norm(vectors, axis=1)]),
            hovertemplate=(
                "Fx=%{customdata[0]:.2f} N  "
                "Fy=%{customdata[1]:.2f} N  "
                "Fz=%{customdata[2]:.2f} N  "
                "|F|=%{customdata[3]:.2f} N<extra></extra>"
            ),
        )
    )
    fig.update_layout(xaxis_title="X (mm)", yaxis_title="Y (mm)", **base_layout)
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def _history_fig(history: list, fingers: list[str], go):
    base_layout = dict(
        title="Total Force History",
        template="plotly_white",
        xaxis_title="time (s)",
        yaxis_title="Σ|F| (N)",
        margin=dict(l=40, r=10, t=35, b=35),
        uirevision="history",
        transition=dict(duration=0),
    )
    if not history:
        return go.Figure(layout=go.Layout(**base_layout))

    t0 = history[0][0]
    fig = go.Figure()
    for name in fingers:
        vals = [
            float(np.sum(np.linalg.norm(d[name], axis=1))) if name in d else float("nan")
            for _, d in history
        ]
        fig.add_trace(go.Scattergl(x=[t - t0 for t, _ in history], y=vals, mode="lines", name=name))
    fig.update_layout(**base_layout)
    return fig
