"""Combined Qt app for live Paxini tactile streaming, recording, and replay.

This is the main interactive tactile tool:

    python -m midas_hand_api.tactile.paxini_tactile_qt
    python -m midas_hand_api.tactile.paxini_tactile_qt --port /dev/ttyACM0
    python -m midas_hand_api.tactile.paxini_tactile_qt --csv paxini_recording.csv
    python -m midas_hand_api.tactile.paxini_tactile_qt --replay-only --csv paxini_recording.csv

If the package is installed, the same app is also available as:

    midas-paxini

Live mode reads the Paxini board, Record writes the live stream to CSV, and
Replay plays the last recording or a loaded CSV with interpolation.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import deque
from pathlib import Path
from typing import Callable

import numpy as np

from midas_hand_api.tactile import PaxiniConfig, PaxiniHandSensor
import midas_hand_api.tactile.paxini_qt_visualizer as qtviz


DEFAULT_FINGERS = ("thumb", "index", "middle", "ring")
COMPONENT_CHOICES = ("Fz", "|F|", "Fx", "Fy")


def parse_fingers(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return list(DEFAULT_FINGERS)

    names = [part.strip().lower() for part in value.split(",") if part.strip()]
    if not names:
        raise argparse.ArgumentTypeError("Expected comma-separated finger names.")

    unknown = sorted(set(names) - set(DEFAULT_FINGERS))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown finger(s): {', '.join(unknown)}. "
            f"Expected one or more of: {', '.join(DEFAULT_FINGERS)}."
        )
    return names


def copy_data(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        finger: np.asarray(vectors, dtype=np.float64).copy()
        for finger, vectors in data.items()
    }


def recording_sample(timestamp_s: float, data: dict[str, np.ndarray], fingers: list[str]) -> dict:
    copied = {}
    for name in fingers:
        vectors = data.get(name)
        if vectors is not None:
            copied[name] = np.asarray(vectors, dtype=np.float64).copy()
    return {"time": timestamp_s, "data": copied}


def write_recording_csv(samples: list[dict], path: Path) -> None:
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_index", "time_s", "finger", "point_index", "fx_n", "fy_n", "fz_n"])
        if not samples:
            return

        t0 = float(samples[0]["time"])
        for sample_index, sample in enumerate(samples):
            time_s = float(sample["time"]) - t0
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
        samples.append({"time": float(entry["time"]), "data": data})
    return samples


def recording_relative_times(samples: list[dict]) -> np.ndarray:
    if not samples:
        return np.asarray([], dtype=np.float64)
    t0 = float(samples[0]["time"])
    return np.asarray([float(sample["time"]) - t0 for sample in samples], dtype=np.float64)


def finger_order_from_samples(samples: list[dict]) -> list[str]:
    fingers: list[str] = []
    for sample in samples:
        for finger in sample["data"]:
            if finger not in fingers:
                fingers.append(finger)
    return fingers


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


def precompute_history(
    samples: list[dict],
    fingers: list[str],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
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


def wait_for_first_sample(
    get_data: Callable[[], dict[str, np.ndarray]],
    timeout_s: float = 3.0,
) -> dict[str, np.ndarray]:
    deadline = time.monotonic() + timeout_s
    last_error = None
    while time.monotonic() < deadline:
        try:
            return get_data()
        except RuntimeError as exc:
            last_error = exc
            time.sleep(0.05)
    raise RuntimeError(f"No tactile data available yet: {last_error}")


class PaxiniTactileQtApp:
    def __init__(
        self,
        *,
        get_live_data: Callable[[], dict[str, np.ndarray]] | None,
        initial_data: dict[str, np.ndarray],
        csv_path: Path,
        update_hz: float,
        record_rate_hz: float,
        component: str,
        history_len: int,
        history_window_s: float,
        show_arrows: bool,
        arrow_region_size: int,
        arrow_min_force_n: float,
        replay_samples: list[dict] | None = None,
        replay_only: bool = False,
    ) -> None:
        self.get_live_data = get_live_data
        self.csv_path = csv_path.expanduser()
        self.update_hz = update_hz
        self.record_rate_hz = record_rate_hz
        self.component = component
        self.history_len = max(2, history_len)
        self.history_window_s = history_window_s
        self.show_arrows = show_arrows
        self.arrow_region_size = arrow_region_size
        self.arrow_min_force_n = arrow_min_force_n
        self.replay_only = replay_only

        self.mode = "replay" if replay_only else "live"
        self.live_start_s = time.perf_counter()
        self.display_frame_count = 0
        self.last_error = ""
        self.current_live_data = copy_data(initial_data)
        self.current_display_data = copy_data(initial_data)
        self.fingers = list(initial_data.keys())
        self.coords = qtviz._auto_coords(initial_data)

        self.live_times: deque[float] = deque(maxlen=self.history_len)
        self.live_components: dict[str, deque[np.ndarray]] = {
            finger: deque(maxlen=self.history_len) for finger in self.fingers
        }
        self.live_totals: dict[str, deque[float]] = {
            finger: deque(maxlen=self.history_len) for finger in self.fingers
        }

        self.recording = False
        self.recording_samples: list[dict] = []
        self.recording_started_s = 0.0
        self.next_record_s = 0.0

        self.replay_samples: list[dict] = replay_samples or []
        self.replay_times = recording_relative_times(self.replay_samples)
        self.replay_components: dict[str, np.ndarray] = {}
        self.replay_totals: dict[str, np.ndarray] = {}
        self.replay_time_s = 0.0
        self.replay_last_wall_s = time.perf_counter()
        self.replay_playing = bool(self.replay_samples)
        if self.replay_samples:
            replay_fingers = finger_order_from_samples(self.replay_samples)
            self.replay_components, self.replay_totals = precompute_history(
                self.replay_samples,
                replay_fingers,
            )

        self.map_items = {}

        self.window = qtviz.QtWidgets.QMainWindow()
        self.window.setWindowTitle("Paxini Tactile Qt")
        self.window.resize(1800, 840)
        self.root = qtviz.QtWidgets.QWidget()
        self.root.setObjectName("root")
        self.window.setCentralWidget(self.root)
        self.root_layout = qtviz.QtWidgets.QVBoxLayout(self.root)
        self.root_layout.setContentsMargins(10, 8, 10, 8)
        self.root_layout.setSpacing(8)

        self._build_controls()
        self._build_maps()
        self._build_history()
        self._append_live_history(self.current_live_data)
        self._update_mode_buttons()
        self._update_frame()

        self.timer = qtviz.QtCore.QTimer()
        self.timer.timeout.connect(self._on_tick)
        self.timer.start(max(1, int(round(1000.0 / max(self.update_hz, 1e-6)))))

    def show(self) -> None:
        self.window.show()

    def _build_controls(self) -> None:
        controls = qtviz.QtWidgets.QHBoxLayout()
        controls.setSpacing(8)

        title = qtviz.QtWidgets.QLabel("Paxini Tactile Qt")
        title.setObjectName("title")
        controls.addWidget(title)

        self.status = qtviz.QtWidgets.QLabel("")
        self.status.setObjectName("status")
        controls.addWidget(self.status, stretch=1)

        controls.addWidget(qtviz.QtWidgets.QLabel("Map intensity"))
        self.component_combo = qtviz.QtWidgets.QComboBox()
        for label, value in qtviz._COMPONENT_OPTIONS:
            self.component_combo.addItem(label, value)
        self.component_combo.setCurrentIndex(max(0, qtviz._COMPONENT_CHOICES.index(self.component)))
        self.component_combo.currentIndexChanged.connect(self._on_component_changed)
        controls.addWidget(self.component_combo)

        controls.addWidget(qtviz.QtWidgets.QLabel("History finger"))
        self.history_finger_combo = qtviz.QtWidgets.QComboBox()
        self._populate_history_fingers(self.fingers)
        self.history_finger_combo.currentIndexChanged.connect(self._update_history)
        controls.addWidget(self.history_finger_combo)

        self.live_button = qtviz.QtWidgets.QPushButton("Live")
        self.live_button.clicked.connect(self._switch_live)
        controls.addWidget(self.live_button)

        self.record_button = qtviz.QtWidgets.QPushButton("Record")
        self.record_button.clicked.connect(self._start_recording)
        controls.addWidget(self.record_button)

        self.stop_button = qtviz.QtWidgets.QPushButton("Stop")
        self.stop_button.clicked.connect(self._stop_action)
        controls.addWidget(self.stop_button)

        self.replay_button = qtviz.QtWidgets.QPushButton("Replay")
        self.replay_button.clicked.connect(self._switch_replay)
        controls.addWidget(self.replay_button)

        self.load_button = qtviz.QtWidgets.QPushButton("Load CSV")
        self.load_button.clicked.connect(self._load_csv_dialog)
        controls.addWidget(self.load_button)

        self.play_button = qtviz.QtWidgets.QPushButton("Pause")
        self.play_button.clicked.connect(self._toggle_replay)
        controls.addWidget(self.play_button)

        self.restart_button = qtviz.QtWidgets.QPushButton("Restart")
        self.restart_button.clicked.connect(self._restart_replay)
        controls.addWidget(self.restart_button)

        self.loop_check = qtviz.QtWidgets.QCheckBox("Loop")
        self.loop_check.setChecked(True)
        controls.addWidget(self.loop_check)

        controls.addWidget(qtviz.QtWidgets.QLabel("Speed"))
        self.speed_spin = qtviz.QtWidgets.QDoubleSpinBox()
        self.speed_spin.setRange(0.1, 5.0)
        self.speed_spin.setSingleStep(0.1)
        self.speed_spin.setValue(1.0)
        controls.addWidget(self.speed_spin)

        self.root_layout.addLayout(controls)

    def _build_maps(self) -> None:
        self.map_grid = qtviz.QtWidgets.QGridLayout()
        self.map_grid.setSpacing(6)
        columns = max(1, min(len(self.fingers), 4))
        for index, finger in enumerate(self.fingers):
            plot_widget = qtviz.pg.PlotWidget(background=qtviz._PANEL_BG)
            plot_item = plot_widget.getPlotItem()
            qtviz._style_plot(plot_item)
            plot_item.setAspectLocked(True)
            plot_item.setLabel("bottom", "X", units="mm", color=qtviz._TEXT_COLOR)
            plot_item.setLabel("left", "Y", units="mm", color=qtviz._TEXT_COLOR)

            scatter = qtviz.pg.ScatterPlotItem(size=9 if finger == "thumb" else 11)
            plot_item.addItem(scatter)
            arrow_line = qtviz.pg.PlotDataItem(pen=qtviz.pg.mkPen(qtviz._ARROW_COLOR, width=2.2))
            plot_item.addItem(arrow_line)

            coords = self.coords.get(finger)
            if coords is not None:
                pad = 2.0
                plot_item.setXRange(float(np.min(coords[:, 0])) - pad, float(np.max(coords[:, 0])) + pad)
                plot_item.setYRange(float(np.min(coords[:, 1])) - pad, float(np.max(coords[:, 1])) + pad)

            self.map_grid.addWidget(plot_widget, index // columns, index % columns)
            self.map_items[finger] = {
                "plot": plot_item,
                "scatter": scatter,
                "arrows": arrow_line,
            }
        self.root_layout.addLayout(self.map_grid, stretch=3)

    def _build_history(self) -> None:
        layout = qtviz.QtWidgets.QHBoxLayout()
        layout.setSpacing(8)

        self.single_history = qtviz.pg.PlotWidget(background=qtviz._PANEL_BG)
        single_item = self.single_history.getPlotItem()
        qtviz._style_plot(single_item)
        single_item.setTitle("Fx/Fy/Fz History", color=qtviz._TEXT_COLOR)
        single_item.setLabel("bottom", "time", units="s", color=qtviz._TEXT_COLOR)
        single_item.setLabel("left", "sum force", units="N", color=qtviz._TEXT_COLOR)
        single_item.addLegend(offset=(-10, 10))
        self.fx_curve = single_item.plot(pen=qtviz.pg.mkPen("#6dd3ff", width=2), name="Fx")
        self.fy_curve = single_item.plot(pen=qtviz.pg.mkPen("#ffb84d", width=2), name="Fy")
        self.fz_curve = single_item.plot(pen=qtviz.pg.mkPen("#75e08f", width=2), name="Fz")
        layout.addWidget(self.single_history, stretch=1)

        self.total_history = qtviz.pg.PlotWidget(background=qtviz._PANEL_BG)
        total_item = self.total_history.getPlotItem()
        qtviz._style_plot(total_item)
        total_item.setTitle("Total Distributed Force History", color=qtviz._TEXT_COLOR)
        total_item.setLabel("bottom", "time", units="s", color=qtviz._TEXT_COLOR)
        total_item.setLabel("left", "sum |F|", units="N", color=qtviz._TEXT_COLOR)
        total_item.addLegend(offset=(-10, 10))
        self.total_curves = {}
        for finger in self.fingers:
            color = qtviz._HISTORY_COLORS.get(finger, qtviz._TEXT_COLOR)
            self.total_curves[finger] = total_item.plot(pen=qtviz.pg.mkPen(color, width=2), name=finger)
        layout.addWidget(self.total_history, stretch=1)

        self.root_layout.addLayout(layout, stretch=2)

    def _on_tick(self) -> None:
        if self.mode == "live":
            self._tick_live()
        else:
            self._tick_replay()
        self._update_frame()

    def _tick_live(self) -> None:
        if self.get_live_data is None:
            self.last_error = "No live sensor connection."
            return
        try:
            self.current_live_data = copy_data(self.get_live_data())
            self.current_display_data = copy_data(self.current_live_data)
            self.last_error = ""
        except RuntimeError as exc:
            self.last_error = str(exc)
            return

        self.display_frame_count += 1
        self._append_live_history(self.current_live_data)
        self._record_if_due(self.current_live_data)

    def _tick_replay(self) -> None:
        now_s = time.perf_counter()
        if self.replay_playing:
            delta_s = now_s - self.replay_last_wall_s
            self.replay_time_s += delta_s * float(self.speed_spin.value())
        self.replay_last_wall_s = now_s

        duration_s = self._replay_duration_s()
        if duration_s > 0.0:
            if self.loop_check.isChecked():
                self.replay_time_s %= duration_s
            elif self.replay_time_s >= duration_s:
                self.replay_time_s = duration_s
                self.replay_playing = False
        self.current_display_data, _frame_index = interpolate_recording_data(
            self.replay_samples,
            self.replay_times,
            self.replay_time_s,
        )
        self.display_frame_count += 1

    def _update_frame(self) -> None:
        self._update_maps(self.current_display_data)
        self._update_history()
        self._update_status()
        self._update_mode_buttons()

    def _update_maps(self, data: dict[str, np.ndarray]) -> None:
        label = qtviz._component_label(self.component)
        for finger in self.fingers:
            vectors = data.get(finger)
            coords = self.coords.get(finger)
            items = self.map_items[finger]
            items["plot"].setTitle(f"{finger.title()} - {label} intensity", color=qtviz._TEXT_COLOR)
            if vectors is None or coords is None:
                items["scatter"].setData(spots=[])
                items["arrows"].setData([], [])
                continue

            values = qtviz._component_intensity(vectors, self.component)
            items["scatter"].setData(
                x=coords[:, 0],
                y=coords[:, 1],
                brush=qtviz._force_brushes(values),
                pen=qtviz.pg.mkPen(qtviz._DOT_OUTLINE, width=0.5),
                size=9 if finger == "thumb" else 11,
            )

            if self.show_arrows:
                line_x, line_y = qtviz._force_arrow_lines(
                    coords,
                    vectors,
                    region_size=self.arrow_region_size,
                    min_force_n=self.arrow_min_force_n,
                )
                items["arrows"].setData(line_x, line_y)
            else:
                items["arrows"].setData([], [])

    def _append_live_history(self, data: dict[str, np.ndarray]) -> None:
        now_s = time.perf_counter() - self.live_start_s
        self.live_times.append(now_s)
        for finger in self.fingers:
            vectors = data.get(finger)
            if vectors is None:
                components = np.full(3, np.nan, dtype=np.float64)
                total = np.nan
            else:
                components = np.sum(vectors, axis=0)
                total = float(np.sum(np.linalg.norm(vectors, axis=1)))
            self.live_components[finger].append(components)
            self.live_totals[finger].append(total)

    def _update_history(self, _index: int = 0) -> None:
        if self.mode == "replay" and self.replay_samples:
            self._update_replay_history()
        else:
            self._update_live_history()

    def _update_live_history(self) -> None:
        if not self.live_times:
            return
        all_times = np.asarray(self.live_times, dtype=np.float64)
        start_time = max(float(all_times[-1]) - self.history_window_s, float(all_times[0]))
        start = int(np.searchsorted(all_times, start_time, side="left"))
        plot_times = all_times[start:]

        finger = self.history_finger_combo.currentData() or self.fingers[0]
        components = np.asarray(self.live_components[finger], dtype=np.float64)[start:]
        self.fx_curve.setData(plot_times, components[:, 0])
        self.fy_curve.setData(plot_times, components[:, 1])
        self.fz_curve.setData(plot_times, components[:, 2])
        self.single_history.getPlotItem().setTitle(f"{finger.title()} Fx/Fy/Fz History", color=qtviz._TEXT_COLOR)

        for finger_name, curve in self.total_curves.items():
            totals = np.asarray(self.live_totals[finger_name], dtype=np.float64)[start:]
            curve.setData(plot_times, totals)

    def _update_replay_history(self) -> None:
        if self.replay_times.size == 0:
            return
        end = int(np.searchsorted(self.replay_times, self.replay_time_s, side="right"))
        start_time = max(0.0, self.replay_time_s - self.history_window_s)
        start = int(np.searchsorted(self.replay_times, start_time, side="left"))
        if end <= start:
            end = min(len(self.replay_times), start + 1)

        plot_times = self.replay_times[start:end]
        finger = self.history_finger_combo.currentData() or self.fingers[0]
        if finger not in self.replay_components:
            return
        components = self.replay_components[finger][start:end]
        self.fx_curve.setData(plot_times, components[:, 0])
        self.fy_curve.setData(plot_times, components[:, 1])
        self.fz_curve.setData(plot_times, components[:, 2])
        self.single_history.getPlotItem().setTitle(f"{finger.title()} Fx/Fy/Fz History", color=qtviz._TEXT_COLOR)

        for finger_name, curve in self.total_curves.items():
            totals = self.replay_totals.get(finger_name)
            if totals is None:
                curve.setData([], [])
            else:
                curve.setData(plot_times, totals[start:end])

    def _record_if_due(self, data: dict[str, np.ndarray]) -> None:
        if not self.recording:
            return
        now_s = time.perf_counter()
        interval_s = 1.0 / max(self.record_rate_hz, 1e-6)
        if now_s < self.next_record_s:
            return
        self.recording_samples.append(recording_sample(now_s, data, self.fingers))
        self.next_record_s += interval_s
        if self.next_record_s < now_s - interval_s:
            self.next_record_s = now_s + interval_s

    def _start_recording(self) -> None:
        if self.mode != "live" or self.get_live_data is None:
            self.last_error = "Recording is only available in live mode."
            return
        self.recording_samples = []
        self.recording = True
        self.recording_started_s = time.perf_counter()
        self.next_record_s = self.recording_started_s
        self.last_error = ""

    def _stop_recording(self) -> None:
        if not self.recording:
            return
        self.recording = False
        write_recording_csv(self.recording_samples, self.csv_path)
        self._load_replay_samples(self.recording_samples, self.csv_path)
        self.last_error = f"Saved {len(self.recording_samples)} frames to {self.csv_path}"

    def _stop_action(self) -> None:
        if self.recording:
            self._stop_recording()
            return
        if self.mode == "replay":
            self.replay_playing = False
            return
        self.last_error = "Live display is already running."

    def _load_replay_samples(self, samples: list[dict], csv_path: Path) -> None:
        self.replay_samples = [
            {"time": float(sample["time"]), "data": copy_data(sample["data"])}
            for sample in samples
        ]
        self.csv_path = csv_path.expanduser()
        self.replay_times = recording_relative_times(self.replay_samples)
        replay_fingers = finger_order_from_samples(self.replay_samples)
        self.replay_components, self.replay_totals = precompute_history(self.replay_samples, replay_fingers)
        if replay_fingers and replay_fingers != self.fingers:
            self.last_error = "Loaded CSV fingers differ from live view; showing common fingers only."

    def _load_csv_dialog(self) -> None:
        path, _selected = qtviz.QtWidgets.QFileDialog.getOpenFileName(
            self.window,
            "Load Paxini CSV",
            str(self.csv_path.parent if self.csv_path.parent else Path.cwd()),
            "CSV files (*.csv);;All files (*)",
        )
        if not path:
            return
        try:
            samples = load_recording_csv(Path(path))
        except Exception as exc:
            self.last_error = f"Could not load CSV: {exc}"
            return
        if not samples:
            self.last_error = f"No samples found in {path}"
            return
        self._load_replay_samples(samples, Path(path))
        self._switch_replay()

    def _switch_live(self) -> None:
        if self.get_live_data is None:
            self.last_error = "No live sensor connection."
            return
        self.mode = "live"
        self.replay_playing = False
        self.live_start_s = time.perf_counter()
        self.last_error = ""

    def _switch_replay(self) -> None:
        if self.recording:
            self._stop_recording()
        if not self.replay_samples:
            try:
                self._load_replay_samples(load_recording_csv(self.csv_path), self.csv_path)
            except Exception as exc:
                self.last_error = f"No replay data loaded: {exc}"
                return
        self.mode = "replay"
        self.replay_time_s = 0.0
        self.replay_last_wall_s = time.perf_counter()
        self.replay_playing = True
        self.last_error = ""
        self.current_display_data, _frame_index = interpolate_recording_data(
            self.replay_samples,
            self.replay_times,
            self.replay_time_s,
        )

    def _toggle_replay(self) -> None:
        if self.mode != "replay":
            self._switch_replay()
            return
        self.replay_playing = not self.replay_playing
        self.replay_last_wall_s = time.perf_counter()

    def _restart_replay(self) -> None:
        if self.mode != "replay":
            self._switch_replay()
        self.replay_time_s = 0.0
        self.replay_last_wall_s = time.perf_counter()
        self.replay_playing = True

    def _on_component_changed(self, _index: int = 0) -> None:
        self.component = self.component_combo.currentData()
        self._update_maps(self.current_display_data)

    def _populate_history_fingers(self, fingers: list[str]) -> None:
        self.history_finger_combo.blockSignals(True)
        self.history_finger_combo.clear()
        for finger in fingers:
            self.history_finger_combo.addItem(finger.title(), finger)
        self.history_finger_combo.blockSignals(False)

    def _replay_duration_s(self) -> float:
        return float(self.replay_times[-1]) if self.replay_times.size else 0.0

    def _update_mode_buttons(self) -> None:
        has_live = self.get_live_data is not None
        has_replay = bool(self.replay_samples)
        self.live_button.setEnabled(has_live)
        self.record_button.setEnabled(has_live and self.mode == "live" and not self.recording)
        self.stop_button.setEnabled(self.recording or self.mode == "replay")
        self.replay_button.setEnabled(has_replay or self.csv_path.exists())
        self.play_button.setEnabled(has_replay or self.csv_path.exists())
        self.restart_button.setEnabled(has_replay or self.csv_path.exists())
        self.play_button.setText("Pause" if self.mode == "replay" and self.replay_playing else "Play")
        self.record_button.setText("Recording..." if self.recording else "Record")

    def _update_status(self) -> None:
        status_parts = [f"mode={self.mode}"]
        status_parts.append(f"fingers={', '.join(self.fingers)}")
        if self.recording:
            elapsed_s = max(time.perf_counter() - self.recording_started_s, 1e-6)
            rate = len(self.recording_samples) / elapsed_s
            status_parts.append(f"recording={len(self.recording_samples)} frames/{rate:.1f} Hz")
        elif self.mode == "replay":
            duration_s = self._replay_duration_s()
            status_parts.append(f"replay={self.replay_time_s:.2f}/{duration_s:.2f}s")
            status_parts.append(f"frames={len(self.replay_samples)}")
        else:
            elapsed_s = max(time.perf_counter() - self.live_start_s, 1e-6)
            status_parts.append(f"ui={self.display_frame_count / elapsed_s:.1f} Hz")
        status_parts.append(f"csv={self.csv_path}")
        if self.last_error:
            status_parts.append(self.last_error)
        self.status.setText(" | ".join(status_parts))


def build_config(args: argparse.Namespace) -> PaxiniConfig:
    return PaxiniConfig(
        port=args.port,
        fingers=args.fingers,
        baudrate=args.baudrate,
        publish_rate_hz=max(args.publish_rate_hz, args.record_rate_hz, args.qt_update_hz),
        scale_n=args.scale,
        signed_z=args.signed_z,
        discard_startup_frames=args.discard_startup_frames,
        median_window=args.median_window,
        response_timeout_s=args.response_timeout,
        startup_attempts=args.startup_attempts,
        serial_settle_s=args.serial_settle,
        dtr=args.dtr,
        rts=args.rts,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--port", "--paxini-port", dest="port", default=None)
    parser.add_argument("--baudrate", type=int, default=921600)
    parser.add_argument("--fingers", type=parse_fingers, default=list(DEFAULT_FINGERS))
    parser.add_argument("--publish-rate-hz", type=float, default=60.0)
    parser.add_argument("--record-rate-hz", type=float, default=30.0)
    parser.add_argument("--qt-update-hz", type=float, default=30.0)
    parser.add_argument("--median-window", type=int, default=3)
    parser.add_argument("--discard-startup-frames", type=int, default=5)
    parser.add_argument("--scale", type=float, default=0.1)
    parser.add_argument("--signed-z", action="store_true")
    parser.add_argument("--dtr", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--rts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--serial-settle", type=float, default=0.75)
    parser.add_argument("--response-timeout", type=float, default=1.0)
    parser.add_argument("--startup-attempts", type=int, default=3)
    parser.add_argument(
        "--recalibrate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Recalibrate (re-zero) the sensors on startup with nothing touching "
        "them. Use --no-recalibrate to skip.",
    )
    parser.add_argument("--csv", type=Path, default=Path("paxini_recording.csv"))
    parser.add_argument("--replay-only", action="store_true", help="Skip hardware and open CSV replay only.")
    parser.add_argument("--component", choices=COMPONENT_CHOICES, default="Fz")
    parser.add_argument("--history-len", type=int, default=600)
    parser.add_argument("--history-window-s", type=float, default=10.0)
    parser.add_argument("--no-arrows", action="store_true")
    parser.add_argument("--arrow-region-size", type=int, default=3)
    parser.add_argument("--arrow-min-force", type=float, default=0.25)
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Optional seconds to keep the Qt window open. Default: run until closed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    qtviz._load_qt()
    qtviz.pg.setConfigOptions(antialias=True, background=qtviz._PANEL_BG, foreground=qtviz._TEXT_COLOR)
    app = qtviz.QtWidgets.QApplication.instance() or qtviz.QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(PaxiniTactileQtAppStyles.stylesheet())

    sensor: PaxiniHandSensor | None = None
    replay_samples: list[dict] = []
    get_live_data = None

    if args.replay_only:
        replay_samples = load_recording_csv(args.csv.expanduser())
        if not replay_samples:
            raise SystemExit(f"No samples found in {args.csv}")
        fingers = finger_order_from_samples(replay_samples)
        initial_data = first_data_by_finger(replay_samples, fingers)
    else:
        sensor = PaxiniHandSensor(build_config(args))
        sensor.connect()
        print(f"Paxini connected on {sensor.port}")
        print(f"Reading fingers: {', '.join(args.fingers)}")
        if args.recalibrate:
            print("Recalibrating sensors — make sure nothing is touching them...")
            sensor.recalibrate()
        initial_data = wait_for_first_sample(sensor.read_latest)
        get_live_data = sensor.read_latest
        if args.csv.expanduser().exists():
            try:
                replay_samples = load_recording_csv(args.csv.expanduser())
            except Exception as exc:
                print(f"Could not preload replay CSV {args.csv}: {exc}")

    viewer = PaxiniTactileQtApp(
        get_live_data=get_live_data,
        initial_data=initial_data,
        csv_path=args.csv,
        update_hz=args.qt_update_hz,
        record_rate_hz=args.record_rate_hz,
        component=args.component,
        history_len=args.history_len,
        history_window_s=args.history_window_s,
        show_arrows=not args.no_arrows,
        arrow_region_size=args.arrow_region_size,
        arrow_min_force_n=args.arrow_min_force,
        replay_samples=replay_samples,
        replay_only=args.replay_only,
    )
    viewer.show()
    if args.duration is not None:
        qtviz.QtCore.QTimer.singleShot(
            max(1, int(round(args.duration * 1000.0))),
            viewer.window.close,
        )
    try:
        exec_method = getattr(app, "exec", None) or app.exec_
        raise SystemExit(exec_method())
    finally:
        if sensor is not None:
            sensor.disconnect()
            print("Paxini disconnected.")


class PaxiniTactileQtAppStyles:
    @staticmethod
    def stylesheet() -> str:
        return f"""
        QWidget#root {{ background-color: {qtviz._DARK_BG}; color: {qtviz._TEXT_COLOR}; }}
        QLabel {{ color: {qtviz._TEXT_COLOR}; }}
        QLabel#title {{ font-size: 18px; font-weight: 700; }}
        QLabel#status {{ color: {qtviz._MUTED_TEXT_COLOR}; }}
        QComboBox, QPushButton, QDoubleSpinBox {{
            background-color: #172235;
            border: 1px solid #33445d;
            border-radius: 4px;
            color: {qtviz._TEXT_COLOR};
            min-height: 24px;
            padding: 2px 8px;
        }}
        QPushButton:disabled {{
            color: #5f7084;
            background-color: #101827;
        }}
        QCheckBox {{ color: {qtviz._TEXT_COLOR}; }}
        """


if __name__ == "__main__":
    main()
