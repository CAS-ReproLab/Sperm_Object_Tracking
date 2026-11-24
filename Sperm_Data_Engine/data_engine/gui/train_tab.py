"""
train_tab.py -- the hub's "Train" tab.

Unlike Prepare Data/Evaluate (call their stage function directly on a
BackgroundTask), this launches train.py as a genuinely detached OS
subprocess (gui/jobs.py) -- training can run for hours, and closing the
hub, a crash, or a remote-desktop disconnect must not kill it. State lives
on disk (jobs.json + per-job progress/log files), so re-opening the hub
reattaches to whatever's still running automatically -- see jobs.py.

Form fields are built by reflecting train.py's own argparse.ArgumentParser
(train.build_parser()) rather than hand-duplicated -- the one tab where
that's the simpler option, not an added layer, since this tab has to
produce a subprocess argv list regardless of how the form is built.
"""
from pathlib import Path

from PyQt6.QtCore import QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QMessageBox, QProgressBar, QPushButton, QTextEdit, QVBoxLayout, QWidget,
)

from data_engine.gui import jobs
from data_engine.gui.widgets import labeled_file_picker, set_list_items
from data_engine.registry.registry import Registry
from data_engine.stages.train.train import build_parser

# Handled specially, not by the generic per-action loop below: the
# mutually-exclusive weights group and --tiers both need domain-specific
# widgets (a resume/cold-start toggle; a registry-aware multi-select), and
# --progress-file is set internally by jobs.launch(), never user-facing.
_SPECIAL_DESTS = {"base_weights", "resume_from", "tiers", "progress_file", "help"}


class GenericField:
    """One argparse action -> one widget, generic enough for the plain
    scalar/flag flags (--epochs, --lr0, --plots, ...). Values round-trip as
    strings either way -- train.py's own argparse in the subprocess does
    the real int/float coercion, so the GUI never needs to duplicate it."""
    def __init__(self, action, parent=None):
        self.flag = action.option_strings[0]
        self.is_flag = (action.const is True and action.nargs == 0)  # store_true
        if self.is_flag:
            self.widget = QCheckBox(parent)
            self.widget.setChecked(bool(action.default))
        else:
            self.widget = QLineEdit(parent)
            if action.default is not None:
                self.widget.setText(str(action.default))

    def to_argv(self) -> list:
        if self.is_flag:
            return [self.flag] if self.widget.isChecked() else []
        value = self.widget.text().strip()
        return [self.flag, value] if value else []


class JobPanel(QGroupBox):
    """One job's live (or historical) status: progress bar, latest loss,
    tailed log. Polls jobs.read_progress()/tail_log() on a QTimer while
    the job is still running; a completed/failed job just renders its
    last-known state once.

    Emits `removed(job_id)` when dismissed -- TrainTab (which owns this
    panel's place in the layout) is the one that actually deletes the
    widget; this class only asks jobs.py to drop the underlying record."""
    removed = pyqtSignal(str)

    def __init__(self, job: dict, parent=None):
        super().__init__(job["job_id"], parent)
        self.job = job
        self._build_ui()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.refresh)
        self.refresh()
        if job["status"] == "running":
            self._timer.start(3000)

    def _build_ui(self):
        layout = QVBoxLayout(self)
        self._status_label = QLabel(self)
        layout.addWidget(self._status_label)

        self._progress_bar = QProgressBar(self)
        layout.addWidget(self._progress_bar)

        self._loss_label = QLabel(self)
        layout.addWidget(self._loss_label)

        self._log_view = QTextEdit(self)
        self._log_view.setReadOnly(True)
        self._log_view.setMaximumHeight(150)
        layout.addWidget(self._log_view)

        btn_row = QHBoxLayout()
        self._stop_btn = QPushButton("Stop", self)
        self._stop_btn.clicked.connect(self._on_stop_clicked)
        btn_row.addWidget(self._stop_btn)

        self._remove_btn = QPushButton("✕ Remove", self)
        self._remove_btn.setToolTip("Dismiss this job from the list (doesn't touch its trained weights)")
        self._remove_btn.clicked.connect(self._on_remove_clicked)
        btn_row.addWidget(self._remove_btn)
        layout.addLayout(btn_row)

    def refresh(self):
        if self.job["status"] == "running":
            fresh = {j["job_id"]: j for j in jobs.list_jobs()}.get(self.job["job_id"])
            if fresh is not None:
                self.job = fresh

        status = self.job["status"]
        self._status_label.setText(f"Status: {status}  (PID {self.job.get('pid')})")
        self._stop_btn.setEnabled(status == "running")
        self._remove_btn.setEnabled(status != "running")

        progress = jobs.read_progress(self.job)
        if progress:
            epoch, epochs = progress.get("epoch", 0), progress.get("epochs", 1)
            self._progress_bar.setRange(0, epochs)
            self._progress_bar.setValue(epoch)
            self._progress_bar.setFormat(f"Epoch {epoch}/{epochs}")
            loss = progress.get("train_loss") or {}
            self._loss_label.setText("  ".join(f"{k}={v:.4f}" for k, v in loss.items()))
        else:
            self._progress_bar.setRange(0, 1)
            self._progress_bar.setValue(0)
            self._progress_bar.setFormat("Waiting for first epoch...")

        self._log_view.setPlainText(jobs.tail_log(self.job, n_lines=100))
        self._log_view.verticalScrollBar().setValue(self._log_view.verticalScrollBar().maximum())

        if status != "running":
            self._timer.stop()

    def _on_stop_clicked(self):
        reply = QMessageBox.question(
            self, "Stop training?", f"Stop {self.job['job_id']}? This cannot be undone.")
        if reply == QMessageBox.StandardButton.Yes:
            jobs.stop(self.job)
            self.refresh()

    def _on_remove_clicked(self):
        jobs.remove(self.job["job_id"])
        self.removed.emit(self.job["job_id"])


class TrainTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._fields = {}
        self._job_panels = {}
        self._build_ui()
        self._update_clip_preview()
        self._reattach_existing_jobs()

    def refresh_choices(self):
        """Re-read the registry (new clips/tiers appear without a restart)."""
        set_list_items(self._tier_list, sorted({c.density_tier for c in Registry.load().clips}))
        self._update_clip_preview()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        form_box = QGroupBox("Start a training run")
        form = QFormLayout(form_box)

        self._weights_mode = QComboBox(form_box)
        self._weights_mode.addItems(["Cold-start (base weights)", "Resume from checkpoint"])
        form.addRow("Mode:", self._weights_mode)

        base_row, self._base_weights = labeled_file_picker(form_box, filter_str="Weights (*.pt)")
        form.addRow("Base weights (optional, default yolo11x.pt):", base_row)

        resume_row, self._resume_from = labeled_file_picker(form_box, filter_str="Weights (*.pt)")
        form.addRow("Resume from checkpoint:", resume_row)

        self._tier_list = QListWidget(form_box)
        self._tier_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        set_list_items(self._tier_list, sorted({c.density_tier for c in Registry.load().clips}))
        self._tier_list.itemSelectionChanged.connect(self._update_clip_preview)
        form.addRow("Tiers (none = all):", self._tier_list)

        self._clip_preview = QLabel(form_box)
        self._clip_preview.setWordWrap(True)
        self._clip_preview.setStyleSheet("color: palette(dark);")
        form.addRow("Clips this will train on:", self._clip_preview)

        parser = build_parser()
        for action in parser._actions:
            if action.dest in _SPECIAL_DESTS:
                continue
            field = GenericField(action, form_box)
            self._fields[action.dest] = field
            form.addRow(f"{action.option_strings[0]}:", field.widget)

        layout.addWidget(form_box)

        start_btn = QPushButton("Start Training", self)
        start_btn.clicked.connect(self._on_start_clicked)
        layout.addWidget(start_btn)

        self._jobs_box = QGroupBox("Training jobs (persists across hub restarts)", self)
        self._jobs_layout = QVBoxLayout(self._jobs_box)
        layout.addWidget(self._jobs_box)
        layout.addStretch(1)

    def _reattach_existing_jobs(self):
        for job in jobs.list_jobs():
            self._add_job_panel(job)

    def _add_job_panel(self, job):
        panel = JobPanel(job, self._jobs_box)
        panel.removed.connect(self._on_job_panel_removed)
        self._jobs_layout.addWidget(panel)
        self._job_panels[job["job_id"]] = panel

    def _on_job_panel_removed(self, job_id):
        panel = self._job_panels.pop(job_id, None)
        if panel is not None:
            self._jobs_layout.removeWidget(panel)
            panel.setParent(None)
            panel.deleteLater()

    def _update_clip_preview(self):
        """Shows exactly which registered clips the current tier selection
        would train/validate on -- the same filtering train()/build_yaml()
        apply (empty selection = all tiers, matching train()'s own
        tiers=None), just surfaced before the run starts instead of only
        visible in the log afterward."""
        tiers = [item.text() for item in self._tier_list.selectedItems()] or None
        clips = [c for c in Registry.load().clips if tiers is None or c.density_tier in tiers]
        train_ids = sorted(c.clip_id for c in clips if c.split == "train")
        val_ids = sorted(c.clip_id for c in clips if c.split == "val")
        self._clip_preview.setText(
            f"Train ({len(train_ids)}): {', '.join(train_ids) or '(none)'}\n"
            f"Val ({len(val_ids)}): {', '.join(val_ids) or '(none)'}"
        )

    def _build_argv(self) -> list:
        argv = []
        if self._weights_mode.currentIndex() == 1:
            resume = self._resume_from.text().strip()
            if not resume:
                raise ValueError("Select a checkpoint to resume from.")
            argv += ["--resume-from", resume]
        else:
            base = self._base_weights.text().strip()
            if base:
                argv += ["--base-weights", base]

        tiers = [item.text() for item in self._tier_list.selectedItems()]
        for tier in tiers:
            argv += ["--tiers", tier]

        for field in self._fields.values():
            argv += field.to_argv()
        return argv

    def _on_start_clicked(self):
        try:
            argv = self._build_argv()
        except ValueError as e:
            QMessageBox.warning(self, "Invalid input", str(e))
            return
        job = jobs.launch(argv)
        self._add_job_panel(job)
