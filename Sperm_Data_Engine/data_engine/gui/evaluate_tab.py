"""
evaluate_tab.py -- the hub's "Evaluate" tab: three sub-tabs.

- Metrics: run evaluate.py against a checkpoint, show per-clip + density-
  tier-summary tables (evaluate() writes files to disk; this just reads
  them back rather than evaluate() returning the tables directly).
- Run History: scans every data_engine/models/*/run_manifest.json +
  its eval_*/_density_tier_summary.csv, plots HOTA/IDF1/MOTA (n_clips-
  weighted mean across tiers) vs. run timestamp via matplotlib. Older
  training_weights/*/ runs predate run_manifest.json and are skipped, not
  errored on.
- Crossover Impact: a thin GUI over analysis/crossover_impact.py -- ground
  truth + repeatable (name, prediction CSV) rows, same comparison table the
  CLI prints, plus a button to render crossover_detector.py's video overlay.

All stage calls run on a BackgroundTask (gui/widgets.py) -- evaluation and
crossover analysis both finish in seconds-to-low-minutes, so (unlike
Train) a background thread is enough; no subprocess/detachment needed.
"""
import json
from pathlib import Path

from PyQt6.QtCore import QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QMessageBox, QPushButton, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

from data_engine.gui.widgets import (
    BackgroundTask, find_checkpoints, labeled_combo, labeled_file_picker, labeled_spinbox,
    set_combo_items, set_list_items,
)
from data_engine.registry.registry import REPO_ROOT, Registry
from data_engine.stages.validate.evaluate import evaluate

_find_checkpoints = find_checkpoints  # local alias -- moved to widgets.py so label_tab.py can share it


METRIC_VIEWS = [("Both (unfiltered + filtered)", "both"), ("Unfiltered", "unfiltered"), ("Filtered (stationary sperm removed)", "filtered")]
_VIEW_FILES = {"both": ("_per_clip_both.csv", "_density_tier_summary_both.csv"),
               "unfiltered": (None, "_density_tier_summary.csv"),
               "filtered": (None, "_density_tier_summary_filtered.csv")}


def _read_eval_results(out_dir: Path, view: str = "both"):
    """(per-clip df, density-tier summary df) for the chosen view, read back
    from evaluate()'s on-disk output (it doesn't return these directly).
    'both' = each metric followed by its '(filt)' twin."""
    import pandas as pd
    per_clip_file, summary_file = _VIEW_FILES[view]
    if per_clip_file:
        p = out_dir / per_clip_file
        per_clip = pd.read_csv(p) if p.exists() else None
    else:
        col = "Unfiltered" if view == "unfiltered" else "Filtered"
        rows = {}
        for csv_path in sorted(out_dir.glob("*.csv")):
            if csv_path.name.startswith("_"):
                continue
            df = pd.read_csv(csv_path)
            rows[csv_path.stem] = dict(zip(df["Metric"], df[col]))
        per_clip = pd.DataFrame.from_dict(rows, orient="index").rename_axis("clip_id").reset_index() if rows else None
    summary_path = out_dir / summary_file
    summary = pd.read_csv(summary_path) if summary_path.exists() else None
    return per_clip, summary


def _fill_table_from_dict_of_dicts(table: QTableWidget, data: dict, row_header: str):
    if not data:
        table.setRowCount(0)
        table.setColumnCount(0)
        return
    columns = sorted({k for row in data.values() for k in row})
    table.setColumnCount(len(columns) + 1)
    table.setHorizontalHeaderLabels([row_header] + columns)
    table.setRowCount(len(data))
    for r, (row_id, row) in enumerate(sorted(data.items())):
        table.setItem(r, 0, QTableWidgetItem(row_id))
        for c, col in enumerate(columns, start=1):
            value = row.get(col)
            table.setItem(r, c, QTableWidgetItem(f"{value:.4f}" if isinstance(value, float) else str(value)))
    table.resizeColumnsToContents()


def _fill_table_from_dataframe(table: QTableWidget, df) -> None:
    if df is None or df.empty:
        table.setRowCount(0)
        table.setColumnCount(0)
        return
    table.setColumnCount(len(df.columns))
    table.setHorizontalHeaderLabels([str(c) for c in df.columns])
    table.setRowCount(len(df))
    for r in range(len(df)):
        for c, col in enumerate(df.columns):
            value = df.iloc[r][col]
            text = f"{value:.4f}" if isinstance(value, float) else str(value)
            table.setItem(r, c, QTableWidgetItem(text))
    table.resizeColumnsToContents()


def _combine_for_comparison(trackpy_summary, model_summary, model_label="YOLO+ByteTrack"):
    """Stack a trackpy density-tier summary and a model's into one table,
    trackpy directly above its matching model row for each tier (and for the
    OVERALL_LABEL row), the shape the paper's results tables use. Either
    side may be None (only one has been computed yet) -- returns just the
    other, tagged, rather than nothing."""
    import pandas as pd

    parts = []
    if trackpy_summary is not None:
        t = trackpy_summary.copy()
        t.insert(1, "Tracker", "trackpy")
        parts.append(t)
    if model_summary is not None:
        m = model_summary.copy()
        m.insert(1, "Tracker", model_label)
        parts.append(m)
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]

    t, m = parts
    tiers = list(t["density_tier"])  # trackpy's own row order (build_density_report sorts tiers, OVERALL last)
    interleaved = []
    for tier in tiers:
        interleaved.append(t[t["density_tier"] == tier])
        match = m[m["density_tier"] == tier]
        if not match.empty:
            interleaved.append(match)
    extra = m[~m["density_tier"].isin(tiers)]
    if not extra.empty:
        interleaved.append(extra)
    return pd.concat(interleaved, ignore_index=True)


class MetricsSubTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._task = None
        self._trackpy_task = None
        self._out_dir_done = None
        self._trackpy_out_dir = None
        self._build_ui()

    def refresh_choices(self):
        set_combo_items(self._model_combo, _find_checkpoints())
        set_list_items(self._tier_list, sorted({c.density_tier for c in Registry.load().clips}))

    def _build_ui(self):
        layout = QVBoxLayout(self)
        form = QFormLayout()

        self._model_combo = QComboBox(self)
        self._model_combo.setEditable(True)
        self._model_combo.addItems(_find_checkpoints())
        form.addRow("Checkpoint:", self._model_combo)

        self._tier_list = QListWidget(self)
        self._tier_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        set_list_items(self._tier_list, sorted({c.density_tier for c in Registry.load().clips}))
        form.addRow("Tiers (none = all):", self._tier_list)

        self._tracker_combo = labeled_combo(["bytetrack", "botsort"], default="bytetrack", parent=self)
        form.addRow("Tracker:", self._tracker_combo)

        out_row, self._out_dir = labeled_file_picker(self, mode="dir")
        form.addRow("Output dir (optional):", out_row)
        layout.addLayout(form)

        run_row = QHBoxLayout()
        run_btn = QPushButton("Run Evaluation", self)
        run_btn.clicked.connect(self._on_run_clicked)
        run_row.addWidget(run_btn)
        trackpy_btn = QPushButton("Compute Trackpy Baseline (same tiers, for comparison)", self)
        trackpy_btn.clicked.connect(self._on_trackpy_clicked)
        run_row.addWidget(trackpy_btn)
        layout.addLayout(run_row)

        self._status_label = QLabel("")
        layout.addWidget(self._status_label)

        view_row = QHBoxLayout()
        view_row.addWidget(QLabel("Show:"))
        self._view_combo = labeled_combo(METRIC_VIEWS, default="both", parent=self)
        self._view_combo.currentIndexChanged.connect(self._refresh_tables)
        view_row.addWidget(self._view_combo)
        view_row.addStretch(1)
        layout.addLayout(view_row)

        layout.addWidget(QLabel("Density-tier summary:"))
        self._summary_table = QTableWidget(self)
        layout.addWidget(self._summary_table)

        layout.addWidget(QLabel("Per-clip metrics:"))
        self._per_clip_table = QTableWidget(self)
        layout.addWidget(self._per_clip_table)

        layout.addWidget(QLabel(
            "Trackpy vs. model, density-tier summary (trackpy row above its matching model row per "
            "tier -- the layout used in the paper's results tables):"))
        self._comparison_table = QTableWidget(self)
        layout.addWidget(self._comparison_table)

    def _selected_tiers(self):
        return [item.text() for item in self._tier_list.selectedItems()] or None

    def _on_run_clicked(self):
        model_path = self._model_combo.currentText().strip()
        if not model_path:
            QMessageBox.warning(self, "Missing checkpoint", "Select or enter a model checkpoint path.")
            return
        self.setEnabled(False)
        self._status_label.setText("Running evaluation...")
        self._task = BackgroundTask(
            evaluate, model_path, tiers=self._selected_tiers(), tracker=self._tracker_combo.currentData(),
            out_dir=self._out_dir.text().strip() or None,
        )
        self._task.completed.connect(self._on_model_completed)
        self._task.error_occurred.connect(self._on_error)
        self._task.start()

    def _on_model_completed(self, out_dir):
        self.setEnabled(True)
        self._out_dir_done = Path(out_dir)
        self._status_label.setText(f"Done. Results in {out_dir}" + self._failed_clips_suffix(self._out_dir_done))
        self._refresh_tables()

    def _on_trackpy_clicked(self):
        self.setEnabled(False)
        self._status_label.setText("Running Trackpy baseline...")
        self._trackpy_task = BackgroundTask(evaluate, tiers=self._selected_tiers(), tracker="trackpy")
        self._trackpy_task.completed.connect(self._on_trackpy_completed)
        self._trackpy_task.error_occurred.connect(self._on_error)
        self._trackpy_task.start()

    def _on_trackpy_completed(self, out_dir):
        self.setEnabled(True)
        self._trackpy_out_dir = Path(out_dir)
        self._status_label.setText(
            f"Trackpy baseline done. Results in {out_dir}" + self._failed_clips_suffix(self._trackpy_out_dir))
        self._refresh_tables()

    @staticmethod
    def _failed_clips_suffix(out_dir: Path) -> str:
        """A one-line warning appended to the status label when evaluate()
        had to skip clip(s) with bad ground-truth data (e.g. a duplicate
        sperm ID from an unsplit crossover) rather than lose the whole run."""
        failed_path = out_dir / "_failed_clips.txt"
        if not failed_path.exists():
            return ""
        n = len([l for l in failed_path.read_text().splitlines() if l.strip()])
        return f"  WARNING: {n} clip(s) failed and were excluded -- see {failed_path}"

    def _refresh_tables(self):
        view = self._view_combo.currentData()
        model_summary = None
        if self._out_dir_done is not None:
            per_clip, model_summary = _read_eval_results(self._out_dir_done, view)
            _fill_table_from_dataframe(self._per_clip_table, per_clip)
            _fill_table_from_dataframe(self._summary_table, model_summary)
        trackpy_summary = None
        if self._trackpy_out_dir is not None:
            _, trackpy_summary = _read_eval_results(self._trackpy_out_dir, view)
        _fill_table_from_dataframe(self._comparison_table, _combine_for_comparison(trackpy_summary, model_summary))

    def _on_error(self, message):
        self.setEnabled(True)
        self._status_label.setText("Failed -- see details.")
        QMessageBox.critical(self, "Error", message)


def _weighted_mean_row(summary_df):
    """One n_clips-weighted mean row across every density tier in a run's
    _density_tier_summary.csv, for a single point per run on the history
    chart (per-tier lines would be sparse/cluttered given most runs to
    date only cover one or two tiers). build_density_report() already
    computes exactly this as its OVERALL_LABEL row -- reused here instead
    of re-deriving it, falling back to a manual weighted mean only for
    older summary CSVs saved before that row existed."""
    from data_engine.metrics.density_report import OVERALL_LABEL

    overall = summary_df[summary_df["density_tier"] == OVERALL_LABEL]
    tier_rows = summary_df[summary_df["density_tier"] != OVERALL_LABEL]
    if not overall.empty:
        source = overall.iloc[0]
        return {m: float(source[m]) for m in ("MOTA", "IDF1", "HOTA") if m in source}

    weights = tier_rows["n_clips"].to_numpy(dtype=float)
    total = weights.sum()
    row = {}
    for metric in ("MOTA", "IDF1", "HOTA"):
        if metric in tier_rows.columns and total > 0:
            row[metric] = float((tier_rows[metric].to_numpy(dtype=float) * weights).sum() / total)
    return row


def _scan_run_history():
    """[(timestamp, {MOTA, IDF1, HOTA}), ...] sorted by timestamp, one point
    per training run that has both a run_manifest.json and at least one
    eval_*/_density_tier_summary.csv under its weights/ dir. Older
    training_weights/*/ runs predate run_manifest.json and are silently
    skipped, not errored on."""
    import pandas as pd

    points = []
    for manifest_path in (REPO_ROOT / "data_engine" / "models").glob("*/run_manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        run_dir = manifest_path.parent
        eval_summaries = sorted((run_dir / "weights").glob("eval_*/_density_tier_summary.csv"))
        if not eval_summaries:
            continue
        summary_df = pd.read_csv(eval_summaries[-1])  # most recent eval for this run
        row = _weighted_mean_row(summary_df)
        if row:
            points.append((manifest.get("timestamp", manifest_path.stat().st_mtime_ns), row))
    points.sort(key=lambda p: p[0])
    return points


class RunHistorySubTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()
        self.refresh()

    def _build_ui(self):
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        layout = QVBoxLayout(self)
        refresh_btn = QPushButton("Refresh", self)
        refresh_btn.clicked.connect(self.refresh)
        layout.addWidget(refresh_btn)

        self._figure = Figure(figsize=(6, 4))
        self._canvas = FigureCanvasQTAgg(self._figure)
        layout.addWidget(self._canvas)

    def refresh(self):
        points = _scan_run_history()
        self._figure.clear()
        ax = self._figure.add_subplot(111)
        if not points:
            ax.text(0.5, 0.5, "No runs with both a run_manifest.json and an evaluation yet.",
                    ha="center", va="center", transform=ax.transAxes)
        else:
            x = list(range(len(points)))
            labels = [str(ts)[:16] for ts, _ in points]
            for metric in ("MOTA", "IDF1", "HOTA"):
                ys = [row.get(metric) for _, row in points]
                if any(y is not None for y in ys):
                    ax.plot(x, ys, marker="o", label=metric)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
            ax.set_ylabel("score (n_clips-weighted mean across tiers)")
            ax.set_title("Training run history")
            ax.legend()
            self._figure.tight_layout()
        self._canvas.draw()


TRACKER_OUTPUTS_DIR = REPO_ROOT / "data_engine" / "data" / "tracker_outputs"
TRACK_SOURCES = [("Compute with trackpy (baseline)", "trackpy"),
                 ("Compute with a model", "model"),
                 ("Existing CSV", "csv")]


def _compute_tracks_csv(video, source, model=None, tracker="bytetrack", reuse=True) -> str:
    """Runs a tracker over `video` and saves canonical frame,sperm,x,y tracks
    to data_engine/data/tracker_outputs/<video>__<tracker>.csv, returning
    that path. ONE known folder for every computed tracker output (Visualize
    and Crossover Impact both use this), instead of leaving people to hunt
    for where a bootstrap/tracker run happened to write its CSV. With
    reuse=True an existing output for the same video+tracker is returned
    as-is (trackpy takes seconds, a model a minute or more)."""
    from data_engine.common.trajectory_io import from_bytetrack_output

    video = Path(video)
    if source == "trackpy":
        label = "trackpy"
    else:
        if not model:
            raise ValueError("A model checkpoint is required to compute model tracks.")
        m = Path(model)
        label = f"{m.parent.parent.name}_{m.stem}_{tracker}"
    out = TRACKER_OUTPUTS_DIR / f"{video.stem}__{label}.csv"
    if reuse and out.exists():
        return str(out)

    if source == "trackpy":
        from data_engine.stages.label.trackpy_tracker import run_trackpy_tracking
        tracks = run_trackpy_tracking(video)
    else:
        from data_engine.stages.validate.track import run_tracking
        tracks = from_bytetrack_output(run_tracking(model, video, tracker=tracker))
    out.parent.mkdir(parents=True, exist_ok=True)
    tracks[["frame", "sperm", "x", "y"]].to_csv(out, index=False)
    return str(out)


class TrackerRow(QWidget):
    """One tracker to compare: a name, where its tracks come from (compute
    with trackpy / compute with a model / an existing CSV), and the matching
    input -- only the one relevant to the chosen source is enabled."""
    def __init__(self, name="", source="csv", model="", parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.name_edit = QLineEdit(name, self)
        self.name_edit.setPlaceholderText("name, e.g. trackpy")
        self.source_combo = labeled_combo(TRACK_SOURCES, default=source, parent=self)
        self.model_combo = QComboBox(self)
        self.model_combo.setEditable(True)
        self.model_combo.addItems(_find_checkpoints())
        if model:
            self.model_combo.setCurrentText(model)
        self._csv_picker, self.csv_edit = labeled_file_picker(self, filter_str="CSV files (*.csv)")
        layout.addWidget(self.name_edit, 2)
        layout.addWidget(self.source_combo, 3)
        layout.addWidget(self.model_combo, 4)
        layout.addWidget(self._csv_picker, 4)
        self.source_combo.currentIndexChanged.connect(self._sync)
        self._sync()

    def refresh_choices(self):
        set_combo_items(self.model_combo, _find_checkpoints())

    def _sync(self, _index=None):
        source = self.source_combo.currentData()
        self.model_combo.setEnabled(source == "model")
        self._csv_picker.setEnabled(source == "csv")

    def spec(self):
        return (self.name_edit.text().strip(), self.source_combo.currentData(),
                self.model_combo.currentText().strip() or None, self.csv_edit.text().strip() or None)


class CrossoverImpactSubTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._task = None
        self._rows = []
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        form = QFormLayout()

        self._clip_combo = QComboBox(self)
        self._populate_clip_combo()
        self._clip_combo.currentIndexChanged.connect(self._on_clip_selected)
        form.addRow("From a registered clip (fills video + ground truth):", self._clip_combo)

        video_row, self._video = labeled_file_picker(
            self, filter_str="Video files (*.mp4 *.avi *.mov *.mkv);;All files (*)")
        form.addRow("Video:", video_row)
        gt_row, self._gt_path = labeled_file_picker(self, filter_str="CSV files (*.csv)")
        form.addRow("Ground truth CSV:", gt_row)

        self._tracker_combo = labeled_combo(["bytetrack", "botsort"], default="bytetrack", parent=self)
        form.addRow("Tracker (for model rows):", self._tracker_combo)
        layout.addLayout(form)

        layout.addWidget(QLabel(
            "Trackers to compare -- name | where the tracks come from | model checkpoint | CSV.\n"
            "'Compute' runs the tracker on the video above right now; no need to find its output first."))
        self._rows_container = QVBoxLayout()
        layout.addLayout(self._rows_container)
        checkpoints = _find_checkpoints()
        self._add_row("trackpy", "trackpy")
        self._add_row("yolo_bytetrack", "model", checkpoints[0] if checkpoints else "")

        add_btn = QPushButton("+ Add tracker", self)
        add_btn.clicked.connect(lambda: self._add_row("", "csv"))
        layout.addWidget(add_btn)

        self._reuse = QCheckBox("Reuse previously computed outputs for the same video + tracker", self)
        self._reuse.setChecked(True)
        layout.addWidget(self._reuse)

        run_btn = QPushButton("Run Comparison", self)
        run_btn.clicked.connect(self._on_run_clicked)
        layout.addWidget(run_btn)

        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

        self._results_table = QTableWidget(self)
        layout.addWidget(self._results_table)

        outputs_btn = QPushButton("Open the folder where computed tracker outputs are saved", self)
        outputs_btn.clicked.connect(self._on_open_outputs)
        layout.addWidget(outputs_btn)

        overlay_btn = QPushButton("Render crossover-marker overlay video (uses the video + ground truth above)", self)
        overlay_btn.clicked.connect(self._on_render_overlay)
        layout.addWidget(overlay_btn)

    def _populate_clip_combo(self):
        combo = self._clip_combo
        current = combo.currentText()
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("(pick files manually below)", None)
        for c in Registry.load().clips:
            if c.source_video and c.ground_truth_csv:
                combo.addItem(f"{c.clip_id}  ({c.density_tier}, {c.split})",
                              (str(REPO_ROOT / c.source_video), str(REPO_ROOT / c.ground_truth_csv)))
        idx = combo.findText(current)
        combo.setCurrentIndex(max(idx, 0))
        combo.blockSignals(False)

    def refresh_choices(self):
        self._populate_clip_combo()
        for row in self._rows:
            row.refresh_choices()

    def _on_clip_selected(self, _index):
        data = self._clip_combo.currentData()
        if data:
            self._video.setText(data[0])
            self._gt_path.setText(data[1])

    def _on_open_outputs(self):
        TRACKER_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(TRACKER_OUTPUTS_DIR)))

    def _add_row(self, name, source, model=""):
        row = TrackerRow(name, source, model, self)
        self._rows.append(row)
        self._rows_container.addWidget(row)

    def _on_run_clicked(self):
        gt_path = self._gt_path.text().strip()
        video = self._video.text().strip()
        if not gt_path:
            QMessageBox.warning(self, "Missing input", "Select a ground-truth CSV.")
            return
        specs = [r.spec() for r in self._rows if r.spec()[0]]
        if not specs:
            QMessageBox.warning(self, "Missing input", "Add at least one named tracker.")
            return
        for name, source, model, csv in specs:
            if source in ("trackpy", "model") and (not video or not Path(video).exists()):
                QMessageBox.warning(self, "Missing video", f"'{name}' is computed from the video -- select one.")
                return
            if source == "model" and not model:
                QMessageBox.warning(self, "Missing checkpoint", f"'{name}' needs a model checkpoint.")
                return
            if source == "csv" and not csv:
                QMessageBox.warning(self, "Missing CSV", f"'{name}' needs a CSV file.")
                return

        tracker, reuse = self._tracker_combo.currentData(), self._reuse.isChecked()

        def _run_all():
            from data_engine.analysis.crossover_impact import crossover_impact
            import pandas as pd
            summaries, paths = [], []
            for name, source, model, csv in specs:
                path = csv if source == "csv" else _compute_tracks_csv(video, source, model, tracker, reuse)
                paths.append(f"{name}: {path}")
                _events, summary = crossover_impact(gt_path, path, tracker_name=name)
                summaries.append(summary)
            return pd.DataFrame(summaries), paths

        self.setEnabled(False)
        self._status_label.setText("Computing trackers (if needed) and comparing... this can take a few minutes.")
        self._task = BackgroundTask(_run_all)
        self._task.completed.connect(self._on_completed)
        self._task.error_occurred.connect(self._on_error)
        self._task.start()

    def _on_completed(self, result):
        comparison_df, paths = result
        self.setEnabled(True)
        self._status_label.setText("Done. Tracks used:\n" + "\n".join(paths))
        _fill_table_from_dataframe(self._results_table, comparison_df)

    def _on_error(self, message):
        self.setEnabled(True)
        self._status_label.setText("Failed -- see details.")
        QMessageBox.critical(self, "Error", message)

    def _on_render_overlay(self):
        gt_path = self._gt_path.text().strip()
        video_path = self._video.text().strip()
        if not gt_path or not video_path:
            QMessageBox.warning(self, "Missing input", "A ground-truth CSV and a source video are both required.")
            return

        def _render():
            from data_engine.stages.label.crossover_detector import detect_crossovers, load_tracks, visualize_crossovers
            df = load_tracks(gt_path)
            events = detect_crossovers(df)
            return visualize_crossovers(video_path, events)

        self.setEnabled(False)
        self._status_label.setText("Rendering overlay...")
        self._task = BackgroundTask(_render)
        self._task.completed.connect(
            lambda path: (self.setEnabled(True), self._status_label.setText(f"Wrote: {path}")))
        self._task.error_occurred.connect(self._on_error)
        self._task.start()


VISUALIZATIONS_DIR = REPO_ROOT / "data_engine" / "data" / "visualizations"


def _render_visualization(video, source, csv, model, tracker, mode, trail_frames):
    """Get tracks (compute with trackpy / a model, or load a CSV), then
    render the overlay mp4. Runs on a BackgroundTask -- tracking + rendering
    a real clip takes a while. Computed tracks are saved to the shared
    tracker_outputs folder (see _compute_tracks_csv) and reused next time."""
    from data_engine.analysis.flow_viewer import render_overlay
    from data_engine.common.trajectory_io import load_tracks

    tracks_csv = csv if source == "csv" else _compute_tracks_csv(video, source, model, tracker)
    # Computed outputs are already named <video>__<tracker>; an arbitrary CSV isn't.
    base = Path(tracks_csv).stem if source != "csv" else f"{Path(video).stem}__{Path(tracks_csv).stem}"
    out = VISUALIZATIONS_DIR / f"{base}__{mode}.mp4"
    return str(render_overlay(video, load_tracks(tracks_csv), out, mode=mode, trail_frames=trail_frames))


class VisualizeSubTab(QWidget):
    """Watch a model's (or any CSV's) tracks drawn over a video -- motion
    trails or ID labels -- without leaving the hub.
    Builds on the old UnifiedFramework/visualizer.py's flow view; see
    analysis/flow_viewer.py."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._task = None
        self._build_ui()

    def _build_ui(self):
        from data_engine.analysis.flow_viewer import MODES
        from data_engine.gui.video_player import VideoPlayer

        layout = QVBoxLayout(self)
        form = QFormLayout()

        video_row, self._video = labeled_file_picker(
            self, filter_str="Video files (*.mp4 *.avi *.mov *.mkv);;All files (*)")
        form.addRow("Video:", video_row)

        self._source = labeled_combo(TRACK_SOURCES, default="model", parent=self)
        self._source.currentIndexChanged.connect(self._on_source_changed)
        form.addRow("Tracks from:", self._source)

        self._model_combo = QComboBox(self)
        self._model_combo.setEditable(True)
        self._model_combo.addItems(_find_checkpoints())
        form.addRow("Checkpoint:", self._model_combo)

        self._tracker_combo = labeled_combo(["bytetrack", "botsort"], default="bytetrack", parent=self)
        form.addRow("Tracker:", self._tracker_combo)

        csv_row, self._csv = labeled_file_picker(self, filter_str="CSV files (*.csv)")
        form.addRow("Tracks CSV (ground truth, trackpy output, ...):", csv_row)

        self._mode = labeled_combo(
            [("Motion trails (flow)", "flow"), ("Track ID labels", "ids")], default="flow", parent=self)
        form.addRow("Style:", self._mode)

        self._trail = labeled_spinbox(0, 500, 30, parent=self)
        self._trail.setToolTip("How many frames a trail lingers before fading. 0 = never fade.")
        form.addRow("Trail length (frames, 0 = persistent):", self._trail)
        layout.addLayout(form)

        self._render_btn = QPushButton("Render + Play", self)
        self._render_btn.clicked.connect(self._on_render_clicked)
        layout.addWidget(self._render_btn)
        self._status = QLabel("", self)
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        self._player = VideoPlayer(self)
        layout.addWidget(self._player, 1)
        self._on_source_changed()

    def refresh_choices(self):
        set_combo_items(self._model_combo, _find_checkpoints())

    def _on_source_changed(self, _index=None):
        source = self._source.currentData()
        self._model_combo.setEnabled(source == "model")
        self._tracker_combo.setEnabled(source == "model")
        self._csv.setEnabled(source == "csv")

    def _on_render_clicked(self):
        video = self._video.text().strip()
        if not video or not Path(video).exists():
            QMessageBox.warning(self, "Missing video", "Select a video file that exists.")
            return
        source = self._source.currentData()
        model = self._model_combo.currentText().strip() if source == "model" else None
        csv = self._csv.text().strip() if source == "csv" else None
        if source == "model" and not model:
            QMessageBox.warning(self, "Missing checkpoint", "Select or enter a model checkpoint.")
            return
        if source == "csv" and not csv:
            QMessageBox.warning(self, "Missing CSV", "Select a tracks CSV.")
            return
        self._render_btn.setEnabled(False)
        self._status.setText("Tracking + rendering... this can take a minute or two.")
        self._task = BackgroundTask(
            _render_visualization, video, source, csv, model, self._tracker_combo.currentData(),
            self._mode.currentData(), self._trail.value())
        self._task.completed.connect(self._on_done)
        self._task.error_occurred.connect(self._on_error)
        self._task.start()

    def _on_done(self, path):
        self._render_btn.setEnabled(True)
        self._status.setText(f"Saved: {path}")
        self._player.load(path)

    def _on_error(self, message):
        self._render_btn.setEnabled(True)
        self._status.setText("Failed -- see details.")
        QMessageBox.critical(self, "Error", message)


class EvaluateTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        sub_tabs = QTabWidget(self)
        layout.addWidget(sub_tabs)
        sub_tabs.addTab(MetricsSubTab(sub_tabs), "Metrics")
        sub_tabs.addTab(VisualizeSubTab(sub_tabs), "Visualize")
        sub_tabs.addTab(RunHistorySubTab(sub_tabs), "Run History")
        sub_tabs.addTab(CrossoverImpactSubTab(sub_tabs), "Crossover Impact")
        self._sub_tabs = sub_tabs

    def refresh_choices(self):
        """Re-scan checkpoints/registry for every sub-tab (a run finished or
        a clip was registered since the hub started)."""
        for i in range(self._sub_tabs.count()):
            sub = self._sub_tabs.widget(i)
            hook = getattr(sub, "refresh_choices", None) or getattr(sub, "refresh", None)
            if hook:
                hook()
