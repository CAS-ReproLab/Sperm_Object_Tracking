import numpy as np
import pandas as pd

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QTabWidget, QSizePolicy,
    QMessageBox, QFileDialog,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

import stats as stats_module
import utils


class StatsWorker(QThread):
    completed    = pyqtSignal(object)   # emits the augmented DataFrame
    error_occurred = pyqtSignal(str)

    def __init__(self, tracks, fps, pixel_size, parent=None):
        super().__init__(parent)
        self.tracks     = tracks.copy()
        self.fps        = fps
        self.pixel_size = pixel_size

    def run(self):
        try:
            result = stats_module.computeAllStats(
                self.tracks, fps=self.fps, pixel_size=self.pixel_size
            )
            self.completed.emit(result)
        except Exception as e:
            import traceback
            self.error_occurred.emit(traceback.format_exc())


class ResultsPanel(QWidget):
    """
    Tabbed panel showing:
      • Summary  — track count, frame count, motility estimate
      • Stats     — per-track table of VCL / VAP / VSL / ALH / BCF
      • Histograms — embedded matplotlib plots
    """

    tracks_updated = pyqtSignal(object)   # propagated upward if main window needs it

    def __init__(self, parent=None):
        super().__init__(parent)
        self._tracks      = None
        self._stats_data  = None
        self._fps         = 9.0
        self._pixel_size  = 1.0476
        self._worker      = None
        self._build_ui()

    # ── public API ────────────────────────────────────────────────────────────

    def set_tracks(self, tracks):
        self._tracks     = tracks
        self._stats_data = None
        self._update_summary()
        self._clear_stats_table()
        self._clear_histograms()

    def set_global_params(self, fps, pixel_size):
        self._fps        = fps
        self._pixel_size = pixel_size

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Top bar: summary + action buttons
        top_row = QHBoxLayout()
        self._lbl_summary = QLabel("No tracks loaded.")
        self._lbl_summary.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        top_row.addWidget(self._lbl_summary)
        top_row.addStretch()

        self._btn_compute = QPushButton("Compute Stats")
        self._btn_compute.setEnabled(False)
        self._btn_compute.clicked.connect(self._run_stats)
        top_row.addWidget(self._btn_compute)

        self._btn_export = QPushButton("Export CSV…")
        self._btn_export.setEnabled(False)
        self._btn_export.clicked.connect(self._export_csv)
        top_row.addWidget(self._btn_export)

        layout.addLayout(top_row)

        # Tab widget
        self._tabs = QTabWidget()
        layout.addWidget(self._tabs)

        # ── Stats table tab ───────────────────────────────────────────────────
        self._table = QTableWidget()
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._tabs.addTab(self._table, "Per-Track Stats")

        # ── Histograms tab ────────────────────────────────────────────────────
        hist_widget = QWidget()
        hist_layout = QVBoxLayout(hist_widget)
        hist_layout.setContentsMargins(0, 0, 0, 0)

        self._fig    = Figure(figsize=(10, 3), tight_layout=True)
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        hist_layout.addWidget(self._canvas)

        self._tabs.addTab(hist_widget, "Histograms")

    # ── summary ───────────────────────────────────────────────────────────────

    def _update_summary(self):
        if self._tracks is None:
            self._lbl_summary.setText("No tracks loaded.")
            self._btn_compute.setEnabled(False)
            self._btn_export.setEnabled(False)
            return

        t = self._tracks
        n_tracks = t["sperm"].nunique()
        n_frames = t["frame"].nunique()
        self._lbl_summary.setText(
            f"{n_tracks} tracks  |  {n_frames} frames"
        )
        self._btn_compute.setEnabled(True)

    # ── stats computation ─────────────────────────────────────────────────────

    def _run_stats(self):
        if self._tracks is None:
            return
        self._btn_compute.setEnabled(False)
        self._btn_compute.setText("Computing…")

        self._worker = StatsWorker(self._tracks, self._fps, self._pixel_size, parent=self)
        self._worker.completed.connect(self._on_stats_done)
        self._worker.error_occurred.connect(self._on_stats_error)
        self._worker.start()

    def _on_stats_done(self, result):
        self._stats_data = result
        self._btn_compute.setEnabled(True)
        self._btn_compute.setText("Compute Stats")
        self._btn_export.setEnabled(True)

        n_tracks = result["sperm"].nunique()
        n_frames = result["frame"].nunique()

        # Motility estimate from VCL if available
        motility_str = ""
        if "VCL" in result.columns:
            per_sperm = result.groupby("sperm")["VCL"].first()
            motile = (per_sperm >= 25).sum()
            motility_str = f"  |  {motile}/{n_tracks} motile (VCL ≥ 25 μm/s)"

        self._lbl_summary.setText(
            f"{n_tracks} tracks  |  {n_frames} frames{motility_str}"
        )
        self._populate_stats_table(result)
        self._draw_histograms(result)
        self.tracks_updated.emit(result)

    def _on_stats_error(self, msg):
        self._btn_compute.setEnabled(True)
        self._btn_compute.setText("Compute Stats")
        QMessageBox.critical(self, "Stats error", msg)

    # ── stats table ───────────────────────────────────────────────────────────

    _STAT_COLS = ["sperm", "VCL", "VAP", "VSL", "ALH_mean", "ALH_max", "BCF"]

    def _clear_stats_table(self):
        self._table.setRowCount(0)
        self._table.setColumnCount(0)

    def _populate_stats_table(self, data):
        cols = [c for c in self._STAT_COLS if c in data.columns]
        per_sperm = data.groupby("sperm").first().reset_index()[cols]

        self._table.setRowCount(len(per_sperm))
        self._table.setColumnCount(len(cols))
        self._table.setHorizontalHeaderLabels(cols)

        for row_i, (_, row) in enumerate(per_sperm.iterrows()):
            for col_i, col in enumerate(cols):
                val = row[col]
                if isinstance(val, float):
                    text = f"{val:.2f}"
                else:
                    text = str(val)
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self._table.setItem(row_i, col_i, item)

        self._table.resizeColumnsToContents()

    # ── histograms ────────────────────────────────────────────────────────────

    def _clear_histograms(self):
        self._fig.clear()
        self._canvas.draw()

    def _draw_histograms(self, data):
        self._fig.clear()
        plot_cols = [c for c in ["VCL", "VAP", "VSL"] if c in data.columns]
        if not plot_cols:
            self._canvas.draw()
            return

        axes = self._fig.subplots(1, len(plot_cols))
        if len(plot_cols) == 1:
            axes = [axes]

        per_sperm = data.groupby("sperm").first()
        colors    = ["#7c4dff", "#448aff", "#00c853"]

        for ax, col, color in zip(axes, plot_cols, colors):
            values = per_sperm[col].dropna()
            ax.hist(values, bins=20, color=color, edgecolor="white", linewidth=0.4)
            ax.set_xlabel(f"{col} (μm/s)", fontsize=9)
            ax.set_ylabel("Count", fontsize=9)
            ax.tick_params(labelsize=8)
            ax.spines[["top", "right"]].set_visible(False)

        self._canvas.draw()

    # ── export ────────────────────────────────────────────────────────────────

    def _export_csv(self):
        if self._stats_data is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export CSV", "", "CSV files (*.csv);;All files (*)"
        )
        if path:
            utils.saveDataFrame(self._stats_data, path)
