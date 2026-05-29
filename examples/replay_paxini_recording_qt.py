"""Replay Paxini tactile CSV recordings with a local pyqtgraph UI.

This visualizer does not connect to live hardware. It is intended for smooth
local replay of CSV recordings, including 30 Hz playback.

Run from the ``midas_hand_api`` package directory, for example::

    python examples/replay_paxini_recording_qt.py --csv paxini_recording.csv
    python examples/replay_paxini_recording_qt.py --csv examples/paxini_recording.csv

If the Qt dependencies are missing, install them with::

    python -m pip install pyqtgraph PyQt6
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np


ASSET_DIR = Path(__file__).resolve().parents[1] / "midas_hand_api" / "assets"
COORD_CSV = {
    127: ASSET_DIR / "paxini_fingertip_26mm_127pts.csv",
    52: ASSET_DIR / "paxini_fingertip_15mm_52pts.csv",
}

DARK_BG = "#080c14"
PANEL_BG = "#101826"
GRID_COLOR = "#273244"
TEXT_COLOR = "#dbe7f3"
MUTED_TEXT_COLOR = "#8fa3b8"
DOT_OUTLINE = "#06111f"
ARROW_COLOR = "#ffdc82"
HISTORY_COLORS = {
    "thumb": "#6dd3ff",
    "index": "#ffb84d",
    "middle": "#75e08f",
    "ring": "#ff6fae",
}
COMPONENT_OPTIONS = (
    ("|F|", "|F|"),
    ("|Fz|", "Fz"),
    ("|Fx|", "Fx"),
    ("|Fy|", "Fy"),
)
COMPONENT_CHOICES = tuple(value for _label, value in COMPONENT_OPTIONS)
FORCE_COLOR_STOPS = (
    (0.0, (30, 99, 255)),
    (0.35, (16, 183, 255)),
    (0.7, (255, 176, 0)),
    (1.0, (255, 42, 42)),
)

pg = None
QtCore = None
QtWidgets = None


def load_qt() -> None:
    global pg, QtCore, QtWidgets
    if pg is not None:
        return
    try:
        import pyqtgraph as pg_module
        from pyqtgraph.Qt import QtCore as qt_core
        from pyqtgraph.Qt import QtWidgets as qt_widgets
    except ImportError as exc:
        raise SystemExit(
            "Qt replay requires pyqtgraph and a Qt binding.\n"
            "Install with:\n"
            "  python -m pip install pyqtgraph PyQt6"
        ) from exc

    pg = pg_module
    QtCore = qt_core
    QtWidgets = qt_widgets


def load_coords(path: Path) -> np.ndarray:
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rows.append((float(row["X"]), float(row["Y"]), float(row["Z"])))
    if not rows:
        raise ValueError(f"Empty coordinate file: {path}")
    return np.asarray(rows, dtype=np.float64)


def auto_coords(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    cache: dict[int, np.ndarray] = {}
    result: dict[str, np.ndarray] = {}
    for name, arr in data.items():
        n_points = arr.shape[0]
        if n_points not in cache:
            path = COORD_CSV.get(n_points)
            if path and path.exists():
                cache[n_points] = load_coords(path)
        if n_points in cache:
            result[name] = cache[n_points]
    return result


def load_recording_csv(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)

    grouped: dict[int, dict] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"sample_index", "time_s", "finger", "point_index", "fx_n", "fy_n", "fz_n"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"{path} is not a Paxini recording CSV. Expected columns: "
                f"{', '.join(sorted(required))}"
            )
        for row in reader:
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


def recording_relative_times(samples: list[dict]) -> np.ndarray:
    if not samples:
        return np.asarray([], dtype=np.float64)
    t0 = float(samples[0]["time"])
    return np.asarray([float(sample["time"]) - t0 for sample in samples], dtype=np.float64)


def finger_order(samples: list[dict]) -> list[str]:
    names: list[str] = []
    for sample in samples:
        for name in sample["data"]:
            if name not in names:
                names.append(name)
    return names


def first_data_by_finger(samples: list[dict], fingers: list[str]) -> dict[str, np.ndarray]:
    result = {}
    for finger in fingers:
        for sample in samples:
            vectors = sample["data"].get(finger)
            if vectors is not None:
                result[finger] = vectors
                break
    return result


def interpolate_recording_data(
    samples: list[dict],
    relative_times: np.ndarray,
    replay_time_s: float,
) -> tuple[dict[str, np.ndarray], int]:
    if not samples:
        return {}, 0
    if len(samples) == 1 or relative_times.size <= 1:
        return copy_data(samples[0]["data"]), 0

    if replay_time_s <= relative_times[0]:
        return copy_data(samples[0]["data"]), 0
    if replay_time_s >= relative_times[-1]:
        return copy_data(samples[-1]["data"]), len(samples) - 1

    right = int(np.searchsorted(relative_times, replay_time_s, side="right"))
    left = max(0, right - 1)
    t0 = float(relative_times[left])
    t1 = float(relative_times[right])
    alpha = 0.0 if t1 <= t0 else (replay_time_s - t0) / (t1 - t0)

    left_data = samples[left]["data"]
    right_data = samples[right]["data"]
    data = {}
    for finger in set(left_data) | set(right_data):
        a = left_data.get(finger)
        b = right_data.get(finger)
        if a is not None and b is not None and a.shape == b.shape:
            data[finger] = (1.0 - alpha) * a + alpha * b
        elif b is not None:
            data[finger] = np.asarray(b, dtype=np.float64).copy()
        elif a is not None:
            data[finger] = np.asarray(a, dtype=np.float64).copy()
    return data, left


def copy_data(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        finger: np.asarray(vectors, dtype=np.float64).copy()
        for finger, vectors in data.items()
    }


def precompute_history(samples: list[dict], fingers: list[str]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    components = {finger: np.full((len(samples), 3), np.nan, dtype=np.float64) for finger in fingers}
    totals = {finger: np.full(len(samples), np.nan, dtype=np.float64) for finger in fingers}
    for sample_index, sample in enumerate(samples):
        for finger in fingers:
            vectors = sample["data"].get(finger)
            if vectors is None:
                continue
            components[finger][sample_index, :] = np.sum(vectors, axis=0)
            totals[finger][sample_index] = np.sum(np.linalg.norm(vectors, axis=1))
    return components, totals


def component_intensity(vectors: np.ndarray, component: str) -> np.ndarray:
    if component == "|F|":
        return np.linalg.norm(vectors, axis=1)
    index = {"Fx": 0, "Fy": 1, "Fz": 2}[component]
    return np.abs(vectors[:, index])


def component_label(component: str) -> str:
    if component == "|F|":
        return "|F|"
    return f"|{component}|"


def force_cmax(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 1.0
    return max(1.0, float(np.percentile(finite, 95)))


def force_brushes(values: np.ndarray):
    cmax = force_cmax(values)
    normalized = np.clip(values / max(cmax, 1e-9), 0.0, 1.0)
    return [pg.mkBrush(*interpolate_color(float(value))) for value in normalized]


def interpolate_color(value: float) -> tuple[int, int, int]:
    if value <= FORCE_COLOR_STOPS[0][0]:
        return FORCE_COLOR_STOPS[0][1]
    for (left_x, left_rgb), (right_x, right_rgb) in zip(FORCE_COLOR_STOPS, FORCE_COLOR_STOPS[1:]):
        if value <= right_x:
            alpha = (value - left_x) / max(right_x - left_x, 1e-9)
            return tuple(
                int(round((1.0 - alpha) * left_rgb[channel] + alpha * right_rgb[channel]))
                for channel in range(3)
            )
    return FORCE_COLOR_STOPS[-1][1]


def regional_force_vectors(
    coords_mm: np.ndarray,
    vectors: np.ndarray,
    *,
    region_size: int,
) -> list[dict]:
    n_points = len(coords_mm)
    if n_points == 0:
        return []

    x_bins, y_bins = region_bins(coords_mm, region_size=region_size)
    x_indices = bin_indices(coords_mm[:, 0], x_bins)
    y_indices = bin_indices(coords_mm[:, 1], y_bins)
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


def region_bins(coords_mm: np.ndarray, *, region_size: int) -> tuple[int, int]:
    n_points = max(len(coords_mm), 1)
    approx_columns = max(1, int(round(np.sqrt(n_points))))
    approx_rows = max(1, int(np.ceil(n_points / approx_columns)))
    return (
        max(1, int(np.ceil(approx_columns / max(region_size, 1)))),
        max(1, int(np.ceil(approx_rows / max(region_size, 1)))),
    )


def bin_indices(values: np.ndarray, n_bins: int) -> np.ndarray:
    if n_bins <= 1 or float(np.ptp(values)) < 1e-9:
        return np.zeros(len(values), dtype=np.int64)
    normalized = (values - float(np.min(values))) / float(np.ptp(values))
    return np.clip((normalized * n_bins).astype(np.int64), 0, n_bins - 1)


def force_arrow_lines(
    coords_mm: np.ndarray,
    vectors: np.ndarray,
    *,
    region_size: int,
    min_force_n: float,
) -> tuple[list[float], list[float]]:
    regions = regional_force_vectors(coords_mm, vectors, region_size=region_size)
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


def style_plot(plot_item) -> None:
    plot_item.setMenuEnabled(False)
    plot_item.showGrid(x=True, y=True, alpha=0.25)
    for axis_name in ("left", "bottom"):
        axis = plot_item.getAxis(axis_name)
        axis.setPen(pg.mkPen(TEXT_COLOR))
        axis.setTextPen(pg.mkPen(TEXT_COLOR))


class PaxiniQtReplay:
    def __init__(self, args: argparse.Namespace, samples: list[dict], csv_path: Path) -> None:
        self.args = args
        self.samples = samples
        self.csv_path = csv_path
        self.times = recording_relative_times(samples)
        self.duration_s = float(self.times[-1]) if self.times.size else 0.0
        self.fingers = finger_order(samples)
        self.coords = auto_coords(first_data_by_finger(samples, self.fingers))
        self.history_components, self.history_totals = precompute_history(samples, self.fingers)

        self.playing = True
        self.playback_time_s = 0.0
        self.last_wall_s = time.perf_counter()
        self.map_items = {}
        self.component = args.component

        self.window = QtWidgets.QMainWindow()
        self.window.setWindowTitle("Paxini Recording Replay")
        self.window.resize(1800, 820)
        self.root = QtWidgets.QWidget()
        self.root.setObjectName("root")
        self.window.setCentralWidget(self.root)
        self.root_layout = QtWidgets.QVBoxLayout(self.root)
        self.root_layout.setContentsMargins(10, 8, 10, 8)
        self.root_layout.setSpacing(8)

        self.build_controls()
        self.build_maps()
        self.build_history()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.on_tick)
        self.timer.start(max(1, int(round(1000.0 / max(args.replay_rate_hz, 1e-6)))))
        self.update_frame()

    def show(self) -> None:
        self.window.show()

    def build_controls(self) -> None:
        controls = QtWidgets.QHBoxLayout()
        controls.setSpacing(10)

        title = QtWidgets.QLabel("Paxini Recording Replay")
        title.setObjectName("title")
        controls.addWidget(title)

        self.status = QtWidgets.QLabel("")
        self.status.setObjectName("status")
        controls.addWidget(self.status, stretch=1)

        controls.addWidget(QtWidgets.QLabel("Map intensity"))
        self.component_combo = QtWidgets.QComboBox()
        for label, value in COMPONENT_OPTIONS:
            self.component_combo.addItem(label, value)
        self.component_combo.setCurrentIndex(max(0, COMPONENT_CHOICES.index(self.args.component)))
        self.component_combo.currentIndexChanged.connect(self.on_component_changed)
        controls.addWidget(self.component_combo)

        controls.addWidget(QtWidgets.QLabel("History finger"))
        self.history_finger_combo = QtWidgets.QComboBox()
        for finger in self.fingers:
            self.history_finger_combo.addItem(finger.title(), finger)
        self.history_finger_combo.currentIndexChanged.connect(self.update_history)
        controls.addWidget(self.history_finger_combo)

        self.play_button = QtWidgets.QPushButton("Pause")
        self.play_button.clicked.connect(self.toggle_play)
        controls.addWidget(self.play_button)

        restart_button = QtWidgets.QPushButton("Restart")
        restart_button.clicked.connect(self.restart)
        controls.addWidget(restart_button)

        self.loop_check = QtWidgets.QCheckBox("Loop")
        self.loop_check.setChecked(not self.args.no_loop)
        controls.addWidget(self.loop_check)

        controls.addWidget(QtWidgets.QLabel("Speed"))
        self.speed_spin = QtWidgets.QDoubleSpinBox()
        self.speed_spin.setRange(0.1, 5.0)
        self.speed_spin.setSingleStep(0.1)
        self.speed_spin.setValue(self.args.speed)
        controls.addWidget(self.speed_spin)

        self.root_layout.addLayout(controls)

    def build_maps(self) -> None:
        grid = QtWidgets.QGridLayout()
        grid.setSpacing(6)
        columns = max(1, min(len(self.fingers), 4))
        for index, finger in enumerate(self.fingers):
            plot_widget = pg.PlotWidget(background=PANEL_BG)
            plot_item = plot_widget.getPlotItem()
            style_plot(plot_item)
            plot_item.setAspectLocked(True)
            plot_item.setLabel("bottom", "X", units="mm", color=TEXT_COLOR)
            plot_item.setLabel("left", "Y", units="mm", color=TEXT_COLOR)

            scatter = pg.ScatterPlotItem(size=9 if finger == "thumb" else 11)
            plot_item.addItem(scatter)
            arrow_line = pg.PlotDataItem(pen=pg.mkPen(ARROW_COLOR, width=2.2))
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

    def build_history(self) -> None:
        layout = QtWidgets.QHBoxLayout()
        layout.setSpacing(8)

        self.single_history = pg.PlotWidget(background=PANEL_BG)
        single_item = self.single_history.getPlotItem()
        style_plot(single_item)
        single_item.setTitle("Fx/Fy/Fz History", color=TEXT_COLOR)
        single_item.setLabel("bottom", "time", units="s", color=TEXT_COLOR)
        single_item.setLabel("left", "sum force", units="N", color=TEXT_COLOR)
        single_item.addLegend(offset=(-10, 10))
        self.fx_curve = single_item.plot(pen=pg.mkPen("#6dd3ff", width=2), name="Fx")
        self.fy_curve = single_item.plot(pen=pg.mkPen("#ffb84d", width=2), name="Fy")
        self.fz_curve = single_item.plot(pen=pg.mkPen("#75e08f", width=2), name="Fz")
        layout.addWidget(self.single_history, stretch=1)

        self.total_history = pg.PlotWidget(background=PANEL_BG)
        total_item = self.total_history.getPlotItem()
        style_plot(total_item)
        total_item.setTitle("Total Distributed Force History", color=TEXT_COLOR)
        total_item.setLabel("bottom", "time", units="s", color=TEXT_COLOR)
        total_item.setLabel("left", "sum |F|", units="N", color=TEXT_COLOR)
        total_item.addLegend(offset=(-10, 10))
        self.total_curves = {}
        for finger in self.fingers:
            color = HISTORY_COLORS.get(finger, "#dbe7f3")
            self.total_curves[finger] = total_item.plot(pen=pg.mkPen(color, width=2), name=finger)
        layout.addWidget(self.total_history, stretch=1)

        self.root_layout.addLayout(layout, stretch=2)

    def on_component_changed(self) -> None:
        self.component = self.component_combo.currentData()
        self.update_frame()

    def toggle_play(self) -> None:
        self.playing = not self.playing
        self.last_wall_s = time.perf_counter()
        self.play_button.setText("Pause" if self.playing else "Play")

    def restart(self) -> None:
        self.playback_time_s = 0.0
        self.last_wall_s = time.perf_counter()
        self.playing = True
        self.play_button.setText("Pause")
        self.update_frame()

    def on_tick(self) -> None:
        now_s = time.perf_counter()
        if self.playing:
            delta_s = now_s - self.last_wall_s
            self.playback_time_s += delta_s * float(self.speed_spin.value())
        self.last_wall_s = now_s

        if self.duration_s > 0.0:
            if self.loop_check.isChecked():
                self.playback_time_s %= self.duration_s
            elif self.playback_time_s >= self.duration_s:
                self.playback_time_s = self.duration_s
                self.playing = False
                self.play_button.setText("Play")
        self.update_frame()

    def update_frame(self) -> None:
        data, frame_index = interpolate_recording_data(
            self.samples,
            self.times,
            self.playback_time_s,
        )
        self.update_maps(data)
        self.update_history()
        self.status.setText(
            f"file={self.csv_path} | frames={len(self.samples)} | "
            f"time={self.playback_time_s:.2f}/{self.duration_s:.2f}s | "
            f"frame~{frame_index + 1}/{len(self.samples)} | "
            f"rate={self.args.replay_rate_hz:.1f} Hz"
        )

    def update_maps(self, data: dict[str, np.ndarray]) -> None:
        for finger in self.fingers:
            vectors = data.get(finger)
            coords = self.coords.get(finger)
            items = self.map_items[finger]
            label = component_label(self.component)
            items["plot"].setTitle(f"{finger.title()} - {label} intensity", color=TEXT_COLOR)
            if vectors is None or coords is None:
                items["scatter"].setData(spots=[])
                items["arrows"].setData([], [])
                continue

            values = component_intensity(vectors, self.component)
            brushes = force_brushes(values)
            pen = pg.mkPen(DOT_OUTLINE, width=0.5)
            spots = [
                {"pos": (float(coords[i, 0]), float(coords[i, 1])), "brush": brushes[i], "pen": pen}
                for i in range(len(coords))
            ]
            items["scatter"].setData(spots=spots)

            if self.args.no_arrows:
                items["arrows"].setData([], [])
            else:
                line_x, line_y = force_arrow_lines(
                    coords,
                    vectors,
                    region_size=self.args.arrow_region_size,
                    min_force_n=self.args.arrow_min_force,
                )
                items["arrows"].setData(line_x, line_y)

    def update_history(self) -> None:
        if self.times.size == 0:
            return

        end = int(np.searchsorted(self.times, self.playback_time_s, side="right"))
        start_time = max(0.0, self.playback_time_s - self.args.history_window_s)
        start = int(np.searchsorted(self.times, start_time, side="left"))
        if end <= start:
            end = min(len(self.times), start + 1)

        plot_times = self.times[start:end]
        history_finger = self.history_finger_combo.currentData() or self.fingers[0]
        components = self.history_components[history_finger][start:end]
        self.fx_curve.setData(plot_times, components[:, 0])
        self.fy_curve.setData(plot_times, components[:, 1])
        self.fz_curve.setData(plot_times, components[:, 2])
        self.single_history.getPlotItem().setTitle(
            f"{history_finger.title()} Fx/Fy/Fz History",
            color=TEXT_COLOR,
        )

        for finger, curve in self.total_curves.items():
            curve.setData(plot_times, self.history_totals[finger][start:end])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("paxini_recording.csv"),
        help="Paxini recording CSV to replay.",
    )
    parser.add_argument(
        "--replay-rate-hz",
        type=float,
        default=30.0,
        help="Qt timer update rate for replay. Default: 30 Hz.",
    )
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier.")
    parser.add_argument("--no-loop", action="store_true", help="Stop on the final frame instead of looping.")
    parser.add_argument(
        "--component",
        choices=COMPONENT_CHOICES,
        default="Fz",
        help="Initial tactile map intensity component.",
    )
    parser.add_argument(
        "--history-window-s",
        type=float,
        default=10.0,
        help="Seconds of history visible in the bottom plots.",
    )
    parser.add_argument("--no-arrows", action="store_true", help="Disable regional force arrows.")
    parser.add_argument(
        "--arrow-region-size",
        type=int,
        default=3,
        help="Approximate tactile point region size for summed force arrows.",
    )
    parser.add_argument(
        "--arrow-min-force",
        type=float,
        default=0.25,
        help="Minimum regional summed force in Newtons before drawing an arrow.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = args.csv.expanduser()
    samples = load_recording_csv(csv_path)
    if not samples:
        raise SystemExit(f"No samples found in {csv_path}")

    load_qt()
    pg.setConfigOptions(antialias=True, background=PANEL_BG, foreground=TEXT_COLOR)
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(
        f"""
        QWidget#root {{ background-color: {DARK_BG}; color: {TEXT_COLOR}; }}
        QLabel {{ color: {TEXT_COLOR}; }}
        QLabel#title {{ font-size: 18px; font-weight: 700; }}
        QLabel#status {{ color: {MUTED_TEXT_COLOR}; }}
        QComboBox, QPushButton, QDoubleSpinBox {{
            background-color: #172235;
            border: 1px solid #33445d;
            border-radius: 4px;
            color: {TEXT_COLOR};
            min-height: 24px;
            padding: 2px 8px;
        }}
        QCheckBox {{ color: {TEXT_COLOR}; }}
        """
    )

    duration_s = float(recording_relative_times(samples)[-1]) if len(samples) > 1 else 0.0
    print(f"Loaded {len(samples)} frames from {csv_path} ({duration_s:.2f}s).")
    print(f"Starting Qt replay at {args.replay_rate_hz:.1f} Hz.")
    viewer = PaxiniQtReplay(args, samples, csv_path)
    viewer.show()
    exec_method = getattr(app, "exec", None) or app.exec_
    raise SystemExit(exec_method())


if __name__ == "__main__":
    main()
