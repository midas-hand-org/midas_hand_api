"""Local pyqtgraph visualizer for live Paxini force data.

It reads the latest sensor sample from a callable and repaints locally at a
fixed Qt timer rate for smooth live tactile checks.
"""

from __future__ import annotations

import csv
import sys
import time
from collections import deque
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
_TEXT_COLOR = "#dbe7f3"
_MUTED_TEXT_COLOR = "#8fa3b8"
_DOT_OUTLINE = "#06111f"
_ARROW_COLOR = "#ffdc82"
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
_COMPONENT_CHOICES = tuple(value for _label, value in _COMPONENT_OPTIONS)
_FORCE_COLOR_STOPS = (
    (0.0, (30, 99, 255)),
    (0.35, (16, 183, 255)),
    (0.7, (255, 176, 0)),
    (1.0, (255, 42, 42)),
)

pg = None
QtCore = None
QtWidgets = None


def _load_qt() -> None:
    global pg, QtCore, QtWidgets
    if pg is not None:
        return
    try:
        import pyqtgraph as pg_module
        from pyqtgraph.Qt import QtCore as qt_core
        from pyqtgraph.Qt import QtWidgets as qt_widgets
    except ImportError as exc:
        raise SystemExit(
            "Qt live visualizer requires pyqtgraph and a Qt binding.\n"
            "Install with:\n"
            "  python -m pip install -e \".[qt]\""
        ) from exc

    pg = pg_module
    QtCore = qt_core
    QtWidgets = qt_widgets


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
        n_points = arr.shape[0]
        if n_points not in cache:
            path = _COORD_CSV.get(n_points)
            if path and path.exists():
                cache[n_points] = _load_coords(path)
        if n_points in cache:
            result[name] = cache[n_points]
    return result


def _first_sample(
    get_data: Callable[[], dict[str, np.ndarray]],
    retries: int = 30,
    interval_s: float = 0.1,
) -> dict[str, np.ndarray]:
    for _ in range(retries):
        try:
            return get_data()
        except RuntimeError:
            time.sleep(interval_s)
    raise RuntimeError(
        f"No sensor data after {retries * interval_s:.1f}s. "
        "Ensure the sensor is connected before constructing PaxiniQtVisualizer."
    )


def _style_plot(plot_item) -> None:
    plot_item.setMenuEnabled(False)
    plot_item.showGrid(x=True, y=True, alpha=0.25)
    for axis_name in ("left", "bottom"):
        axis = plot_item.getAxis(axis_name)
        axis.setPen(pg.mkPen(_TEXT_COLOR))
        axis.setTextPen(pg.mkPen(_TEXT_COLOR))


def _component_intensity(vectors: np.ndarray, component: str) -> np.ndarray:
    if component == "|F|":
        return np.linalg.norm(vectors, axis=1)
    index = {"Fx": 0, "Fy": 1, "Fz": 2}[component]
    return np.abs(vectors[:, index])


def _component_label(component: str) -> str:
    if component == "|F|":
        return "|F|"
    return f"|{component}|"


def _interpolate_color(value: float) -> tuple[int, int, int]:
    if value <= _FORCE_COLOR_STOPS[0][0]:
        return _FORCE_COLOR_STOPS[0][1]
    for (left_x, left_rgb), (right_x, right_rgb) in zip(
        _FORCE_COLOR_STOPS,
        _FORCE_COLOR_STOPS[1:],
    ):
        if value <= right_x:
            alpha = (value - left_x) / max(right_x - left_x, 1e-9)
            return tuple(
                int(round((1.0 - alpha) * left_rgb[channel] + alpha * right_rgb[channel]))
                for channel in range(3)
            )
    return _FORCE_COLOR_STOPS[-1][1]


def _force_cmax(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 1.0
    return max(1.0, float(np.percentile(finite, 95)))


def _force_brushes(values: np.ndarray):
    cmax = _force_cmax(values)
    normalized = np.clip(values / max(cmax, 1e-9), 0.0, 1.0)
    return [pg.mkBrush(*_interpolate_color(float(value))) for value in normalized]


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


def _regional_force_vectors(
    coords_mm: np.ndarray,
    vectors: np.ndarray,
    *,
    region_size: int,
) -> list[dict]:
    if len(coords_mm) == 0:
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


def _force_arrow_lines(
    coords_mm: np.ndarray,
    vectors: np.ndarray,
    *,
    region_size: int,
    min_force_n: float,
) -> tuple[list[float], list[float]]:
    regions = _regional_force_vectors(coords_mm, vectors, region_size=region_size)
    if not regions:
        return [], []

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

        line_x.extend([start[0], end[0], np.nan])
        line_y.extend([start[1], end[1], np.nan])
        line_x.extend([end[0], head_left[0], np.nan, end[0], head_right[0], np.nan])
        line_y.extend([end[1], head_left[1], np.nan, end[1], head_right[1], np.nan])

    return line_x, line_y


class PaxiniQtVisualizer:
    """Live local visualizer for ``dict[str, ndarray (N, 3)]`` Paxini samples."""

    def __init__(
        self,
        get_data: Callable[[], dict[str, np.ndarray]],
        *,
        coords: Optional[dict[str, np.ndarray]] = None,
        update_hz: float = 30.0,
        component: str = "Fz",
        history_len: int = 600,
        history_window_s: float = 10.0,
        show_arrows: bool = True,
        arrow_region_size: int = 3,
        arrow_min_force_n: float = 0.25,
        duration_s: Optional[float] = None,
    ) -> None:
        if component not in _COMPONENT_CHOICES:
            raise ValueError(f"Unknown component {component!r}; expected one of {_COMPONENT_CHOICES}.")
        self.get_data = get_data
        self.coords_arg = coords
        self.update_hz = update_hz
        self.component = component
        self.history_len = history_len
        self.history_window_s = history_window_s
        self.show_arrows = show_arrows
        self.arrow_region_size = arrow_region_size
        self.arrow_min_force_n = arrow_min_force_n
        self.duration_s = duration_s

        self.start_s = time.perf_counter()
        self.frame_count = 0
        self.last_error = ""
        self.fingers: list[str] = []
        self.coords: dict[str, np.ndarray] = {}
        self.map_items = {}
        self.history_times: deque[float] = deque(maxlen=max(2, history_len))
        self.history_components: dict[str, deque[np.ndarray]] = {}
        self.history_totals: dict[str, deque[float]] = {}

    def run(self) -> None:
        _load_qt()
        pg.setConfigOptions(antialias=True, background=_PANEL_BG, foreground=_TEXT_COLOR)

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
        app.setStyleSheet(self._stylesheet())

        initial = _first_sample(self.get_data)
        self.fingers = list(initial.keys())
        self.coords = self.coords_arg if self.coords_arg is not None else _auto_coords(initial)
        self.history_components = {
            finger: deque(maxlen=max(2, self.history_len)) for finger in self.fingers
        }
        self.history_totals = {
            finger: deque(maxlen=max(2, self.history_len)) for finger in self.fingers
        }

        self.window = QtWidgets.QMainWindow()
        self.window.setWindowTitle("Paxini Live Force")
        self.window.resize(1800, 820)
        self.root = QtWidgets.QWidget()
        self.root.setObjectName("root")
        self.window.setCentralWidget(self.root)
        self.root_layout = QtWidgets.QVBoxLayout(self.root)
        self.root_layout.setContentsMargins(10, 8, 10, 8)
        self.root_layout.setSpacing(8)

        self._build_controls()
        self._build_maps()
        self._build_history()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._on_tick)
        self.timer.start(max(1, int(round(1000.0 / max(self.update_hz, 1e-6)))))

        if self.duration_s is not None:
            QtCore.QTimer.singleShot(max(1, int(round(self.duration_s * 1000.0))), self.window.close)

        self._append_history(initial)
        self._update_maps(initial)
        self._update_history()
        self.window.show()
        exec_method = getattr(app, "exec", None) or app.exec_
        raise SystemExit(exec_method())

    def _build_controls(self) -> None:
        controls = QtWidgets.QHBoxLayout()
        controls.setSpacing(10)

        title = QtWidgets.QLabel("Paxini Live Force")
        title.setObjectName("title")
        controls.addWidget(title)

        self.status = QtWidgets.QLabel("")
        self.status.setObjectName("status")
        controls.addWidget(self.status, stretch=1)

        controls.addWidget(QtWidgets.QLabel("Map intensity"))
        self.component_combo = QtWidgets.QComboBox()
        for label, value in _COMPONENT_OPTIONS:
            self.component_combo.addItem(label, value)
        self.component_combo.setCurrentIndex(max(0, _COMPONENT_CHOICES.index(self.component)))
        self.component_combo.currentIndexChanged.connect(self._on_component_changed)
        controls.addWidget(self.component_combo)

        controls.addWidget(QtWidgets.QLabel("Component history"))
        self.history_finger_combo = QtWidgets.QComboBox()
        for finger in self.fingers:
            self.history_finger_combo.addItem(finger.title(), finger)
        self.history_finger_combo.currentIndexChanged.connect(self._update_history)
        controls.addWidget(self.history_finger_combo)

        self.freeze_button = QtWidgets.QPushButton("Freeze")
        self.freeze_button.setCheckable(True)
        controls.addWidget(self.freeze_button)

        self.root_layout.addLayout(controls)

    def _build_maps(self) -> None:
        grid = QtWidgets.QGridLayout()
        grid.setSpacing(6)
        columns = max(1, min(len(self.fingers), 4))
        for index, finger in enumerate(self.fingers):
            plot_widget = pg.PlotWidget(background=_PANEL_BG)
            plot_item = plot_widget.getPlotItem()
            _style_plot(plot_item)
            plot_item.setAspectLocked(True)
            plot_item.setLabel("bottom", "X", units="mm", color=_TEXT_COLOR)
            plot_item.setLabel("left", "Y", units="mm", color=_TEXT_COLOR)

            scatter = pg.ScatterPlotItem(size=9 if finger == "thumb" else 11)
            plot_item.addItem(scatter)
            arrow_line = pg.PlotDataItem(pen=pg.mkPen(_ARROW_COLOR, width=2.2))
            plot_item.addItem(arrow_line)

            coords = self.coords.get(finger)
            if coords is not None:
                pad = 2.0
                plot_item.setXRange(float(np.min(coords[:, 0])) - pad, float(np.max(coords[:, 0])) + pad)
                plot_item.setYRange(float(np.min(coords[:, 1])) - pad, float(np.max(coords[:, 1])) + pad)

            row = index // columns
            column = index % columns
            grid.addWidget(plot_widget, row, column)
            self.map_items[finger] = {
                "plot": plot_item,
                "scatter": scatter,
                "arrows": arrow_line,
            }
        self.root_layout.addLayout(grid, stretch=3)

    def _build_history(self) -> None:
        layout = QtWidgets.QHBoxLayout()
        layout.setSpacing(8)

        self.single_history = pg.PlotWidget(background=_PANEL_BG)
        single_item = self.single_history.getPlotItem()
        _style_plot(single_item)
        single_item.setTitle("Fx/Fy/Fz History", color=_TEXT_COLOR)
        single_item.setLabel("bottom", "time", units="s", color=_TEXT_COLOR)
        single_item.setLabel("left", "sum force", units="N", color=_TEXT_COLOR)
        single_item.addLegend(offset=(-10, 10))
        self.fx_curve = single_item.plot(pen=pg.mkPen("#6dd3ff", width=2), name="Fx")
        self.fy_curve = single_item.plot(pen=pg.mkPen("#ffb84d", width=2), name="Fy")
        self.fz_curve = single_item.plot(pen=pg.mkPen("#75e08f", width=2), name="Fz")
        layout.addWidget(self.single_history, stretch=1)

        self.total_history = pg.PlotWidget(background=_PANEL_BG)
        total_item = self.total_history.getPlotItem()
        _style_plot(total_item)
        total_item.setTitle("Total Distributed Force History", color=_TEXT_COLOR)
        total_item.setLabel("bottom", "time", units="s", color=_TEXT_COLOR)
        total_item.setLabel("left", "sum |F|", units="N", color=_TEXT_COLOR)
        total_item.addLegend(offset=(-10, 10))
        self.total_curves = {}
        for finger in self.fingers:
            color = _HISTORY_COLORS.get(finger, _TEXT_COLOR)
            self.total_curves[finger] = total_item.plot(pen=pg.mkPen(color, width=2), name=finger)
        layout.addWidget(self.total_history, stretch=1)

        self.root_layout.addLayout(layout, stretch=2)

    def _on_component_changed(self, _index: int = 0) -> None:
        self.component = self.component_combo.currentData()
        try:
            self._update_maps(self.get_data())
        except RuntimeError:
            pass

    def _on_tick(self) -> None:
        if self.freeze_button.isChecked():
            self._set_status("frozen")
            return

        try:
            data = self.get_data()
            self.last_error = ""
        except RuntimeError as exc:
            self.last_error = str(exc)
            self._set_status("waiting")
            return

        self.frame_count += 1
        self._append_history(data)
        self._update_maps(data)
        self._update_history()
        self._set_status("live")

    def _append_history(self, data: dict[str, np.ndarray]) -> None:
        now_s = time.perf_counter() - self.start_s
        self.history_times.append(now_s)
        for finger in self.fingers:
            vectors = data.get(finger)
            if vectors is None:
                components = np.full(3, np.nan, dtype=np.float64)
                total = np.nan
            else:
                components = np.sum(vectors, axis=0)
                total = float(np.sum(np.linalg.norm(vectors, axis=1)))
            self.history_components[finger].append(components)
            self.history_totals[finger].append(total)

    def _update_maps(self, data: dict[str, np.ndarray]) -> None:
        label = _component_label(self.component)
        for finger in self.fingers:
            vectors = data.get(finger)
            coords = self.coords.get(finger)
            items = self.map_items[finger]
            items["plot"].setTitle(f"{finger.title()} - {label} intensity", color=_TEXT_COLOR)
            if vectors is None or coords is None:
                items["scatter"].setData(spots=[])
                items["arrows"].setData([], [])
                continue

            values = _component_intensity(vectors, self.component)
            items["scatter"].setData(
                x=coords[:, 0],
                y=coords[:, 1],
                brush=_force_brushes(values),
                pen=pg.mkPen(_DOT_OUTLINE, width=0.5),
                size=9 if finger == "thumb" else 11,
            )

            if self.show_arrows:
                line_x, line_y = _force_arrow_lines(
                    coords,
                    vectors,
                    region_size=self.arrow_region_size,
                    min_force_n=self.arrow_min_force_n,
                )
                items["arrows"].setData(line_x, line_y)
            else:
                items["arrows"].setData([], [])

    def _update_history(self, _index: int = 0) -> None:
        if not self.history_times:
            return

        all_times = np.asarray(self.history_times, dtype=np.float64)
        start_time = max(float(all_times[-1]) - self.history_window_s, float(all_times[0]))
        start = int(np.searchsorted(all_times, start_time, side="left"))
        plot_times = all_times[start:]

        finger = self.history_finger_combo.currentData() or self.fingers[0]
        components = np.asarray(self.history_components[finger], dtype=np.float64)[start:]
        self.fx_curve.setData(plot_times, components[:, 0])
        self.fy_curve.setData(plot_times, components[:, 1])
        self.fz_curve.setData(plot_times, components[:, 2])
        self.single_history.getPlotItem().setTitle(
            f"{finger.title()} Fx/Fy/Fz History",
            color=_TEXT_COLOR,
        )

        for finger_name, curve in self.total_curves.items():
            totals = np.asarray(self.history_totals[finger_name], dtype=np.float64)[start:]
            curve.setData(plot_times, totals)

    def _set_status(self, mode: str) -> None:
        elapsed_s = max(time.perf_counter() - self.start_s, 1e-6)
        fps = self.frame_count / elapsed_s
        error = f" | {self.last_error}" if self.last_error else ""
        self.status.setText(
            f"source={mode} fingers={', '.join(self.fingers)} | "
            f"ui={self.update_hz:.1f} Hz | frames={self.frame_count} | avg={fps:.1f} Hz{error}"
        )

    @staticmethod
    def _stylesheet() -> str:
        return f"""
        QWidget#root {{ background-color: {_DARK_BG}; color: {_TEXT_COLOR}; }}
        QLabel {{ color: {_TEXT_COLOR}; }}
        QLabel#title {{ font-size: 18px; font-weight: 700; }}
        QLabel#status {{ color: {_MUTED_TEXT_COLOR}; }}
        QComboBox, QPushButton {{
            background-color: #172235;
            border: 1px solid #33445d;
            border-radius: 4px;
            color: {_TEXT_COLOR};
            min-height: 24px;
            padding: 2px 8px;
        }}
        QPushButton:checked {{ background-color: #3a2020; }}
        """
