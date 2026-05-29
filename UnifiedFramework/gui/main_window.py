import json

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QSplitter, QVBoxLayout,
    QStatusBar, QMessageBox, QFileDialog,
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction

from pipeline import Pipeline
from gui.video_panel   import VideoPanel
from gui.control_panel import ControlPanel
from gui.results_panel import ResultsPanel

import utils


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sperm Tracker Workbench")
        self.resize(1280, 800)

        self._pipeline = Pipeline(parent=self)
        self._build_ui()
        self._connect_signals()
        self._load_default_config()

    # ── UI layout ─────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Central: horizontal splitter — (video + results) | control panel
        outer_split = QSplitter(Qt.Orientation.Horizontal)

        # Left side: video on top, results on bottom
        left_split = QSplitter(Qt.Orientation.Vertical)

        self._video_panel   = VideoPanel()
        self._results_panel = ResultsPanel()
        self._results_panel.setMaximumHeight(280)

        left_split.addWidget(self._video_panel)
        left_split.addWidget(self._results_panel)
        left_split.setStretchFactor(0, 3)
        left_split.setStretchFactor(1, 1)

        self._control_panel = ControlPanel()

        outer_split.addWidget(left_split)
        outer_split.addWidget(self._control_panel)
        outer_split.setStretchFactor(0, 1)
        outer_split.setStretchFactor(1, 0)

        self.setCentralWidget(outer_split)

        # Status bar
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage("Ready.")

        self._build_menu()

    def _build_menu(self):
        bar = self.menuBar()

        # File menu
        file_menu = bar.addMenu("File")

        act_open_video = QAction("Open Video…", self)
        act_open_video.setShortcut("Ctrl+O")
        act_open_video.triggered.connect(self._control_panel._pick_video)
        file_menu.addAction(act_open_video)

        act_open_config = QAction("Open Config…", self)
        act_open_config.triggered.connect(self._control_panel._pick_config)
        file_menu.addAction(act_open_config)

        act_save_config = QAction("Save Config…", self)
        act_save_config.triggered.connect(self._control_panel._save_config)
        file_menu.addAction(act_save_config)

        file_menu.addSeparator()

        act_save_csv = QAction("Save Tracks CSV…", self)
        act_save_csv.triggered.connect(self._save_tracks_csv)
        file_menu.addAction(act_save_csv)

        file_menu.addSeparator()

        act_quit = QAction("Quit", self)
        act_quit.setShortcut("Ctrl+Q")
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        # View menu — mirrors video panel overlay checkboxes
        view_menu = bar.addMenu("View")

        for chk, label in [
            (self._video_panel._chk_detections,    "Show Detections"),
            (self._video_panel._chk_tracks,         "Show Tracks"),
            (self._video_panel._chk_segmentations,  "Show Segmentations"),
            (self._video_panel._chk_bboxes,         "Show Bounding Boxes"),
        ]:
            act = QAction(label, self, checkable=True)
            act.setChecked(chk.isChecked())
            # Keep menu item and checkbox in sync (both directions)
            act.toggled.connect(chk.setChecked)
            chk.toggled.connect(act.setChecked)
            view_menu.addAction(act)

        # Help menu
        help_menu = bar.addMenu("Help")
        act_about = QAction("About", self)
        act_about.triggered.connect(self._show_about)
        help_menu.addAction(act_about)

    # ── signal wiring ─────────────────────────────────────────────────────────

    def _connect_signals(self):
        # Control panel → pipeline / video
        self._control_panel.video_requested.connect(self._on_video_requested)
        self._control_panel.run_requested.connect(self._on_run_requested)
        self._control_panel.preview_preprocess_requested.connect(self._on_preview_preprocess)
        self._control_panel.preview_detect_requested.connect(self._on_preview_detect)
        self._control_panel.config_loaded.connect(self._on_config_loaded)

        # Pipeline → UI
        self._pipeline.stage_started.connect(self._on_stage_started)
        self._pipeline.stage_completed.connect(self._on_stage_completed)
        self._pipeline.error_occurred.connect(self._on_pipeline_error)
        self._pipeline.finished.connect(self._on_pipeline_finished)

        # Results panel: when stats are computed, refresh tracks (now augmented with stats)
        self._results_panel.tracks_updated.connect(self._on_stats_tracks_updated)

    # ── event handlers ────────────────────────────────────────────────────────

    def _on_video_requested(self, path):
        self._status.showMessage(f"Loading video: {path}")
        try:
            frames = self._pipeline.load_video(path)
            self._video_panel.set_frames(frames)
            self._control_panel.set_video_loaded(True)
            n = len(frames)
            h, w = frames[0].shape[:2]
            self._status.showMessage(f"Loaded {n} frames  ({w}×{h})")
        except Exception as e:
            QMessageBox.critical(self, "Video load error", str(e))
            self._status.showMessage("Error loading video.")

    def _on_preview_preprocess(self, config):
        self._start_run(config, stop_after="preprocessing",
                        status_msg="Running preprocessing preview…")

    def _on_preview_detect(self, config):
        self._start_run(config, stop_after="perception",
                        status_msg="Running detection preview…")

    def _on_run_requested(self, config):
        # Push fps/pixel_size to results panel before running
        self._results_panel.set_global_params(
            float(config.get("fps", 9)),
            float(config.get("pixel_size", 1.0476)),
        )
        self._video_panel.set_playback_fps(int(config.get("fps", 9)))
        self._start_run(config, stop_after=None, status_msg="Full pipeline running…")

    def _start_run(self, config, stop_after, status_msg):
        self._control_panel.set_running(True)
        self._status.showMessage(status_msg)
        try:
            self._pipeline.run(config, stop_after=stop_after)
        except RuntimeError as e:
            QMessageBox.critical(self, "Pipeline error", str(e))
            self._control_panel.set_running(False)
            self._status.showMessage("Pipeline error.")

    def _on_config_loaded(self, config):
        self._status.showMessage("Config loaded.")

    def _on_stage_started(self, stage):
        labels = {
            "preprocessing":  "Preprocessing…",
            "perception":     "Running perception (detect/segment)…",
            "tracking":       "Tracking…",
        }
        if stage.startswith("postprocessing:"):
            name = stage.split(":", 1)[1]
            msg  = f"Postprocessing: {name}…"
        else:
            msg = labels.get(stage, f"{stage}…")
        self._status.showMessage(msg)

    def _on_stage_completed(self, stage, result):
        if stage == "preprocessing":
            self._video_panel.set_preprocessed_frames(result)

        elif stage == "perception":
            n = len(result) if result is not None else 0
            self._status.showMessage(f"Perception done — {n} detections.")
            # Show raw detections (no tracking yet) — used by preview detect mode
            self._video_panel.set_detections_only(result)

        elif stage == "tracking":
            tracks = result
            if tracks is not None:
                n_sperm  = tracks["sperm"].nunique()
                n_frames = tracks["frame"].nunique()
                self._status.showMessage(
                    f"Tracking done — {n_sperm} tracks over {n_frames} frames."
                )
                self._video_panel.set_tracks(tracks)
                self._results_panel.set_tracks(tracks)

        elif stage.startswith("postprocessing:"):
            tracks = result
            if tracks is not None:
                self._video_panel.set_tracks(tracks)
                self._results_panel.set_tracks(tracks)
            name = stage.split(":", 1)[1]
            self._status.showMessage(f"Postprocessor '{name}' done.")

    def _on_pipeline_finished(self):
        self._control_panel.set_running(False)
        self._status.showMessage("Done.")

    def _on_pipeline_error(self, msg):
        self._control_panel.set_running(False)
        self._status.showMessage("Pipeline error — see dialog.")
        QMessageBox.critical(self, "Pipeline Error", msg)

    def _on_stats_tracks_updated(self, tracks_with_stats):
        # Re-push augmented tracks to video panel so stat-based overlays
        # (e.g. speed colouring) could be added in future.
        self._video_panel.set_tracks(tracks_with_stats)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _load_default_config(self):
        try:
            with open("configs/default.json") as f:
                config = json.load(f)
            self._control_panel.apply_config(config)
        except FileNotFoundError:
            pass

    def _save_tracks_csv(self):
        tracks = self._pipeline.tracks
        if tracks is None:
            QMessageBox.information(self, "No tracks", "Run the pipeline first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Tracks CSV", "", "CSV files (*.csv);;All files (*)"
        )
        if path:
            utils.saveDataFrame(tracks, path)
            self._status.showMessage(f"Saved to {path}")

    def _show_about(self):
        QMessageBox.about(
            self, "Sperm Tracker Workbench",
            "Modular sperm tracking GUI.\n\n"
            "Pipeline: Preprocessor → Perception → Tracker → Postprocessors\n\n"
            "Add new algorithms by registering functions in plugins/."
        )
