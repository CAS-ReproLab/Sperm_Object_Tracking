"""
label_tab.py -- the hub's "Label" tab: get a brand-new video labeled.

Deliberately does NOT reuse correction_flow.launch_labeler_and_wait() --
that function blocks on a second, nested QApplication.exec() call, fine
for a one-shot CLI script that owns the whole process, wrong once the hub
already has its own event loop and other tabs/windows running (a second
nested exec() only returns once ALL top-level windows close, not just the
labeler's). Instead this opens LabelerWindow directly under the hub's
existing QApplication and detects closure via a QEvent.Type.Close event
filter (see _CloseWatcher -- NOT the `destroyed` signal, which only fires
on actual object destruction, not close()/hide alone; a real bug this
session caught by testing against an actual window rather than assuming).

Scope, per the professor's decision: registering an ALREADY-labeled video
lives in the Prepare Data tab (ingest.py). This tab only covers getting a
brand-new video labeled, either from scratch (an empty starting CSV,
mirroring label.py) or pre-tracked first (trackpy by default, or YOLO+
ByteTrack if a model is given -- mirroring bootstrap_label.py's backend
resolution, reused directly via _resolve_backend rather than
reimplemented) so a human corrects predictions instead of starting from
nothing.

Note: opened LabelerWindow instances are kept in self._open_windows and
never explicitly deleted for the tab's lifetime (not just until closed) --
deliberately, per this session's research: filter_panel.py's FilterWorker
(a QThread) isn't joined/stopped on window close, so force-deleting a
window that still has one running could surface a real crash. Letting a
closed window's Python object linger un-destroyed is a safer default for a
handful of labeling sessions per hub run than aggressively cleaning up.
"""
from pathlib import Path

import pandas as pd
from PyQt6.QtCore import QEvent, QObject
from PyQt6.QtWidgets import (
    QComboBox, QFormLayout, QGroupBox, QLabel, QMessageBox, QPushButton, QVBoxLayout, QWidget,
)

from data_engine.gui.widgets import BackgroundTask, checkpoint_picker, labeled_combo, labeled_file_picker
from data_engine.registry.registry import REPO_ROOT
from data_engine.stages.label.correction_flow import corrected_csv_path

PENDING_LABEL_DIR = REPO_ROOT / "data" / "pending_label"


def _find_pending_videos():
    if not PENDING_LABEL_DIR.is_dir():
        return []
    return sorted(PENDING_LABEL_DIR.glob("*/*.mp4"))


def _pretrack(video_path, backend, model_path, tracker) -> str:
    """Runs an initial tracker over video_path and writes a
    <stem>_bootstrap.csv, ready for the labeler to pre-load for correction.
    Mirrors bootstrap_label()'s first half exactly (same backend
    resolution, same tracker calls) but stops before launching the labeler
    itself -- the tab does that separately, since it needs to keep the
    window reference/close-watcher alive."""
    from data_engine.common.trajectory_io import from_bytetrack_output, save_tracks
    from data_engine.stages.label.bootstrap_label import _resolve_backend
    from data_engine.stages.label.trackpy_tracker import run_trackpy_tracking
    from data_engine.stages.validate.track import run_tracking

    video_path = Path(video_path)
    resolved = _resolve_backend(backend, model_path)
    if resolved == "yolo" and not model_path:
        raise ValueError("A model checkpoint is required for the yolo backend.")

    initial_csv = video_path.parent / f"{video_path.stem}_bootstrap.csv"
    if resolved == "yolo":
        raw = run_tracking(model_path, video_path, tracker=tracker)
        canonical = from_bytetrack_output(raw)
    else:
        canonical = run_trackpy_tracking(video_path)

    save_tracks(canonical, initial_csv)
    return str(initial_csv)


class _CloseWatcher(QObject):
    """Detects a window's close event without needing WA_DeleteOnClose (and
    the FilterWorker-QThread-destruction risk that carries -- see the
    module docstring). `destroyed` fires on actual object destruction, NOT
    on close()/hide() alone -- confirmed empirically, a real bug caught by
    testing this against an actual LabelerWindow rather than assuming the
    signal meant what its name suggests. An event filter on QEvent.Type.Close
    is the correct hook instead."""
    def __init__(self, callback, parent=None):
        super().__init__(parent)
        self._callback = callback

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.Close:
            self._callback()
        return False  # never consume it -- LabelerWindow's own closeEvent must still run


class LabelTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._open_windows = []
        self._watchers = []  # keep _CloseWatcher instances alive alongside their window
        self._task = None
        self._build_ui()

    def refresh_choices(self):
        """Re-scan for newly finished training runs (the hub calls this on
        every tab switch -- see hub.py's _on_tab_changed)."""
        self._model_path.refresh()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        box = QGroupBox("Label a new video")
        form = QFormLayout(box)

        self._pending_combo = QComboBox(box)
        self._pending_combo.addItem("(pick manually below)", None)
        for path in _find_pending_videos():
            self._pending_combo.addItem(str(path.relative_to(REPO_ROOT)), str(path))
        self._pending_combo.currentIndexChanged.connect(self._on_pending_selected)
        try:
            rel = PENDING_LABEL_DIR.relative_to(REPO_ROOT)
        except ValueError:
            rel = PENDING_LABEL_DIR
        form.addRow(f"From {rel}/:", self._pending_combo)

        video_row, self._video_path = labeled_file_picker(
            box, filter_str="Video files (*.mp4 *.avi *.mov *.mkv);;All files (*)")
        form.addRow("Video:", video_row)

        self._start_mode = QComboBox(box)
        self._start_mode.addItem("Empty (label from scratch)")
        self._start_mode.addItem("Pre-track first (trackpy or a model), then correct")
        self._start_mode.currentIndexChanged.connect(self._on_mode_changed)
        form.addRow("Start from:", self._start_mode)

        csv_row, self._csv_path = labeled_file_picker(box, filter_str="CSV files (*.csv)")
        form.addRow("Existing tracks CSV (optional, empty mode only):", csv_row)

        self._backend_combo = labeled_combo(
            [("auto (trackpy unless a model is given)", "auto"), ("trackpy", "trackpy"), ("yolo", "yolo")],
            default="auto", parent=box)
        form.addRow("Pre-track backend:", self._backend_combo)

        model_row, self._model_path = checkpoint_picker(box)
        form.addRow("Model checkpoint (for yolo/auto):", model_row)

        self._tracker_combo = labeled_combo(["bytetrack", "botsort"], default="bytetrack", parent=box)
        form.addRow("Tracker (for yolo):", self._tracker_combo)

        self._launch_btn = QPushButton("Open Labeler", box)
        self._launch_btn.clicked.connect(self._on_launch_clicked)
        form.addRow(self._launch_btn)

        layout.addWidget(box)

        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)
        layout.addStretch(1)

        self._on_mode_changed(0)

    def _on_mode_changed(self, index):
        pretrack = index == 1
        self._backend_combo.setEnabled(pretrack)
        self._model_path.setEnabled(pretrack)
        self._tracker_combo.setEnabled(pretrack)
        self._csv_path.setEnabled(not pretrack)

    def _on_pending_selected(self, _index):
        path = self._pending_combo.currentData()
        if path:
            self._video_path.setText(path)
            self._csv_path.setText("")

    def _on_launch_clicked(self):
        video = self._video_path.text().strip()
        if not video or not Path(video).exists():
            QMessageBox.warning(self, "Missing video", "Select a video file that exists.")
            return

        if self._start_mode.currentIndex() == 1:
            backend = self._backend_combo.currentData()
            model_path = self._model_path.currentText().strip() or None
            if backend == "yolo" and not model_path:
                QMessageBox.warning(self, "Missing model", "The yolo backend needs a model checkpoint.")
                return
            tracker = self._tracker_combo.currentData()
            self._launch_btn.setEnabled(False)
            self._status_label.setText("Pre-tracking... this can take a little while.")
            self._task = BackgroundTask(_pretrack, video, backend, model_path, tracker)
            self._task.completed.connect(lambda csv: self._on_pretrack_done(video, csv))
            self._task.error_occurred.connect(self._on_pretrack_error)
            self._task.start()
            return

        csv = self._csv_path.text().strip()
        if not csv:
            video_path = Path(video)
            csv_path = video_path.parent / f"{video_path.stem}_empty.csv"
            if not csv_path.exists():
                pd.DataFrame(columns=["frame", "sperm", "x", "y", "label"]).to_csv(csv_path, index=False)
            csv = str(csv_path)
        self._open_labeler(video, csv)

    def _on_pretrack_done(self, video, csv):
        self._launch_btn.setEnabled(True)
        self._open_labeler(video, csv)

    def _on_pretrack_error(self, message):
        self._launch_btn.setEnabled(True)
        self._status_label.setText("Pre-tracking failed -- see details.")
        QMessageBox.critical(self, "Error", message)

    def _open_labeler(self, video, csv):
        # Lazy import, matching label.py/bootstrap_label.py/correction_flow.py's
        # own convention -- relies on the hub already having put
        # data_engine/stages/label/ on sys.path at startup (hub.py).
        try:
            from labeler_gui_pkg.labeler_window import LabelerWindow
        except ImportError as e:
            QMessageBox.critical(self, "Error", f"Could not import the labeler: {e}")
            return

        window = LabelerWindow(video, csv)
        watcher = _CloseWatcher(lambda: self._on_labeler_closed(csv), parent=window)
        window.installEventFilter(watcher)
        window.show()
        self._open_windows.append(window)
        self._watchers.append(watcher)
        self._status_label.setText(f"Labeler opened for {Path(video).name}.")

    def _on_labeler_closed(self, csv_path):
        corrected = corrected_csv_path(csv_path)
        if corrected.exists():
            self._status_label.setText(
                f"Saved corrections: {corrected}\nSwitch to the Prepare Data tab to ingest it.")
        else:
            self._status_label.setText("Labeler closed without saving.")
