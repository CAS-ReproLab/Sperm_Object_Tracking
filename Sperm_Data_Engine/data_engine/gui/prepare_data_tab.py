"""
prepare_data_tab.py -- the hub's "Prepare Data" tab: a registry coverage
dashboard, plus three forms wrapping ingest.py/synthesize.py/augment.py.

Deliberately calls those stage functions directly (not via CLI/subprocess)
on a BackgroundTask (gui/widgets.py) -- they're fast (seconds, at most a
couple minutes for a large synthetic clip), so a detached-process
architecture like the Train tab's isn't warranted here; a background QThread
is enough to keep the window responsive.

"Register" scope, per the professor's decision: this tab only ingests
ALREADY-LABELED video+CSV pairs. Getting a brand-new video labeled (tracker
-> human correction) lives in the Label tab instead, since that flow ends
by opening the labeler either way.
"""
import shutil
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QMessageBox, QPushButton, QSpinBox, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from data_engine.gui.widgets import BackgroundTask, labeled_combo, labeled_file_picker, labeled_spinbox
from data_engine.registry.registry import REPO_ROOT, Registry
from data_engine.stages.generate.augment import augment
from data_engine.stages.generate.ingest import ingest
from data_engine.stages.generate.synthesize import synthesize

SOURCE_TYPES = ["real", "synthetic", "augmented"]
SPLITS = ["train", "val"]
REAL_TIERS = ["low", "medium", "high", "very_high", "unknown"]


def _synthesize_then_ingest(tier, frames, frame_rate, pixel_size, param_specs, out_dir, split, clip_id):
    video_path, csv_path = synthesize(
        tier, frames=frames, frame_rate=frame_rate, pixel_size=pixel_size,
        out_dir=out_dir or None, param_specs=param_specs or None,
    )
    return ingest(video_path, csv_path, tier=tier, source_type="synthetic", split=split,
                  clip_id=clip_id or None)


def _augment_then_ingest(clip_ids, rotate, mirror, out_dir, split, clip_id):
    video_path, csv_path = augment(clip_ids, rotate=rotate, mirror=mirror, out_dir=out_dir or None)
    return ingest(video_path, csv_path, source_type="augmented", split=split, clip_id=clip_id or None)


def _delete_clips(clip_ids):
    """Removes each clip's registry entry AND its own directory on disk --
    real cleanup for "created incorrectly" junk, not just hiding it from
    the registry. A clip whose directory fails to delete keeps its
    registry entry (reported as failed) rather than silently losing track
    of it."""
    registry = Registry.load()
    deleted, failed = [], []
    for cid in clip_ids:
        clip = registry.get(cid)
        if clip is None:
            continue
        clip_dir = REPO_ROOT / clip.path
        try:
            if clip_dir.exists():
                shutil.rmtree(clip_dir)
        except OSError as e:
            failed.append(f"{cid}: {e}")
            continue
        registry.clips.remove(clip)
        deleted.append(cid)
    registry.save()
    return deleted, failed


class PrepareDataTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._task = None  # keep a reference so a running BackgroundTask isn't GC'd
        self._build_ui()
        self.refresh_dashboard()

    # ---- layout ----

    def _build_ui(self):
        layout = QVBoxLayout(self)

        layout.addWidget(self._build_dashboard())
        layout.addWidget(self._build_ingest_form())
        layout.addWidget(self._build_synthesize_form())
        layout.addWidget(self._build_augment_form())
        layout.addWidget(self._build_manage_form())
        layout.addWidget(self._build_delete_form())

        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)
        layout.addStretch(1)

    def _build_dashboard(self) -> QGroupBox:
        box = QGroupBox("Registry coverage (clips by source type x density tier)")
        v = QVBoxLayout(box)
        self._dashboard_table = QTableWidget(box)
        # Room for 7 data rows (+ header) before the table scrolls internally.
        t = self._dashboard_table
        t.setMinimumHeight(7 * t.verticalHeader().defaultSectionSize()
                           + t.horizontalHeader().sizeHint().height() + 2 * t.frameWidth())
        v.addWidget(self._dashboard_table)
        refresh_btn = QPushButton("Refresh", box)
        refresh_btn.clicked.connect(self.refresh_dashboard)
        v.addWidget(refresh_btn)
        return box

    def _build_ingest_form(self) -> QGroupBox:
        box = QGroupBox("Ingest an already-labeled video + CSV")
        form = QFormLayout(box)

        video_row, self._ingest_video = labeled_file_picker(
            box, filter_str="Video files (*.mp4 *.avi *.mov *.mkv);;All files (*)")
        form.addRow("Video:", video_row)

        csv_row, self._ingest_csv = labeled_file_picker(box, filter_str="CSV files (*.csv)")
        form.addRow("Ground-truth CSV:", csv_row)

        self._ingest_source = labeled_combo(SOURCE_TYPES, default="real", parent=box)
        form.addRow("Source type:", self._ingest_source)

        self._ingest_tier = QLineEdit(box)
        self._ingest_tier.setPlaceholderText("blank = auto-computed for real/augmented; "
                                              "required (num_agents) for synthetic")
        form.addRow("Tier:", self._ingest_tier)

        self._ingest_split = labeled_combo(SPLITS, default="train", parent=box)
        form.addRow("Split:", self._ingest_split)

        self._ingest_clip_id = QLineEdit(box)
        self._ingest_clip_id.setPlaceholderText("blank = auto-generated")
        form.addRow("Clip ID (optional):", self._ingest_clip_id)

        run_btn = QPushButton("Ingest", box)
        run_btn.clicked.connect(self._on_ingest_clicked)
        form.addRow(run_btn)
        return box

    def _build_synthesize_form(self) -> QGroupBox:
        box = QGroupBox("Generate a synthetic clip")
        form = QFormLayout(box)

        self._synth_tier = labeled_spinbox(1, 100000, 40, parent=box)
        form.addRow("Tier (num_agents):", self._synth_tier)

        self._synth_frames = labeled_spinbox(1, 100000, 270, parent=box)
        form.addRow("Frames:", self._synth_frames)

        self._synth_frame_rate = QLineEdit("9", box)
        form.addRow("Frame rate (fps):", self._synth_frame_rate)

        self._synth_pixel_size = QLineEdit("1.8", box)
        form.addRow("Pixel size (um/px):", self._synth_pixel_size)

        specs_row, self._synth_param_specs = labeled_file_picker(box, filter_str="JSON files (*.json)")
        form.addRow("Param specs (optional):", specs_row)

        out_row, self._synth_out_dir = labeled_file_picker(box, mode="dir")
        form.addRow("Output dir (optional):", out_row)

        self._synth_split = labeled_combo(SPLITS, default="train", parent=box)
        form.addRow("Split:", self._synth_split)

        self._synth_clip_id = QLineEdit(box)
        self._synth_clip_id.setPlaceholderText("blank = auto-generated")
        form.addRow("Clip ID (optional):", self._synth_clip_id)

        run_btn = QPushButton("Generate + Ingest", box)
        run_btn.clicked.connect(self._on_synthesize_clicked)
        form.addRow(run_btn)
        return box

    def _build_augment_form(self) -> QGroupBox:
        box = QGroupBox("Combine registered clips into a denser augmented clip")
        v = QVBoxLayout(box)

        v.addWidget(QLabel("Select 2+ clips to combine (real or augmented only):"))
        self._augment_list = QListWidget(box)
        self._augment_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        v.addWidget(self._augment_list)

        form = QFormLayout()
        self._augment_rotate = QCheckBox("Add a rotated (90°) copy of each clip", box)
        self._augment_rotate.setChecked(True)
        form.addRow("Rotate:", self._augment_rotate)

        self._augment_mirror = QCheckBox("Add a mirrored copy of each clip", box)
        self._augment_mirror.setChecked(True)
        form.addRow("Mirror:", self._augment_mirror)

        out_row, self._augment_out_dir = labeled_file_picker(box, mode="dir")
        form.addRow("Output dir (optional):", out_row)

        self._augment_split = labeled_combo(SPLITS, default="train", parent=box)
        form.addRow("Split:", self._augment_split)

        self._augment_clip_id = QLineEdit(box)
        self._augment_clip_id.setPlaceholderText("blank = auto-generated")
        form.addRow("Clip ID (optional):", self._augment_clip_id)
        v.addLayout(form)

        run_btn = QPushButton("Combine + Ingest", box)
        run_btn.clicked.connect(self._on_augment_clicked)
        v.addWidget(run_btn)
        return box

    def _build_manage_form(self) -> QGroupBox:
        box = QGroupBox("Manage registered clips (train/val split, or take off the registry)")
        v = QVBoxLayout(box)
        v.addWidget(QLabel("Changing a split saves immediately. 'Remove from registry' keeps every "
                           "file on disk, so a clip can be re-ingested later."))
        self._manage_table = QTableWidget(box)
        self._manage_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._manage_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        t = self._manage_table
        t.setMinimumHeight(7 * t.verticalHeader().defaultSectionSize()
                           + t.horizontalHeader().sizeHint().height() + 2 * t.frameWidth())
        v.addWidget(t)
        remove_btn = QPushButton("Remove selected from registry (keep files)", box)
        remove_btn.clicked.connect(self._on_unregister_clicked)
        v.addWidget(remove_btn)
        return box

    def _refresh_manage_table(self, registry):
        t = self._manage_table
        t.setColumnCount(4)
        t.setHorizontalHeaderLabels(["clip_id", "source", "tier", "split"])
        t.setRowCount(len(registry.clips))
        for row, c in enumerate(registry.clips):
            for col, text in enumerate([c.clip_id, c.source_type, c.density_tier]):
                t.setItem(row, col, QTableWidgetItem(text))
            combo = QComboBox(t)
            combo.addItems(SPLITS)
            combo.setCurrentText(c.split)
            combo.currentTextChanged.connect(lambda split, cid=c.clip_id: self._on_split_changed(cid, split))
            t.setCellWidget(row, 3, combo)
        t.resizeColumnsToContents()

    def _on_split_changed(self, clip_id, split):
        registry = Registry.load()
        registry.set_split(clip_id, split)
        registry.save()
        self._status_label.setText(f"{clip_id} moved to {split}.")
        self.refresh_dashboard()

    def _on_unregister_clicked(self):
        rows = sorted({i.row() for i in self._manage_table.selectedIndexes()})
        clip_ids = [self._manage_table.item(r, 0).text() for r in rows]
        if not clip_ids:
            QMessageBox.information(self, "Nothing selected", "Select one or more rows first.")
            return
        if QMessageBox.question(self, "Remove from registry?",
                                "Take these off the registry (files stay on disk)?\n\n" + "\n".join(clip_ids)
                                ) != QMessageBox.StandardButton.Yes:
            return
        registry = Registry.load()
        for cid in clip_ids:
            registry.unregister(cid)
        registry.save()
        self._status_label.setText(f"Removed from registry (files kept): {', '.join(clip_ids)}")
        self.refresh_dashboard()

    def _build_delete_form(self) -> QGroupBox:
        box = QGroupBox("Delete a synthetic or augmented clip (created incorrectly, etc.)")
        v = QVBoxLayout(box)
        v.addWidget(QLabel("Real clips are not deletable here -- only synthetic/augmented ones:"))
        self._delete_list = QListWidget(box)
        self._delete_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        v.addWidget(self._delete_list)
        delete_btn = QPushButton("Delete Selected (permanent)", box)
        delete_btn.clicked.connect(self._on_delete_clicked)
        v.addWidget(delete_btn)
        return box

    # ---- dashboard ----

    def refresh_dashboard(self):
        registry = Registry.load()
        tiers = sorted({c.density_tier for c in registry.clips}) or ["(none)"]
        table = self._dashboard_table
        columns = [(source_type, split) for source_type in SOURCE_TYPES for split in SPLITS]
        table.setColumnCount(len(columns) + 1)
        table.setHorizontalHeaderLabels(["Density tier"] + [f"{s} ({sp})" for s, sp in columns])
        table.setRowCount(len(tiers))
        for row, tier in enumerate(tiers):
            table.setItem(row, 0, QTableWidgetItem(tier))
            for col, (source_type, split) in enumerate(columns, start=1):
                n = sum(1 for c in registry.clips
                        if c.density_tier == tier and c.source_type == source_type and c.split == split)
                item = QTableWidgetItem(str(n) if n else "-")
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                table.setItem(row, col, item)
        table.resizeColumnsToContents()

        self._refresh_manage_table(registry)
        self._augment_list.clear()
        self._delete_list.clear()
        for c in registry.clips:
            if c.source_type in ("real", "augmented"):
                item = QListWidgetItem(f"{c.clip_id}  (tier={c.density_tier}, split={c.split})")
                item.setData(Qt.ItemDataRole.UserRole, c.clip_id)
                self._augment_list.addItem(item)
            if c.source_type in ("synthetic", "augmented"):
                item = QListWidgetItem(
                    f"{c.clip_id}  ({c.source_type}, tier={c.density_tier}, split={c.split})")
                item.setData(Qt.ItemDataRole.UserRole, c.clip_id)
                self._delete_list.addItem(item)

    # ---- run handlers ----

    def _run(self, fn, *args, on_success_prefix="Done", **kwargs):
        self._set_forms_enabled(False)
        self._status_label.setText("Running...")
        self._task = BackgroundTask(fn, *args, **kwargs)
        self._task.completed.connect(lambda result: self._on_task_completed(on_success_prefix, result))
        self._task.error_occurred.connect(self._on_task_error)
        self._task.start()

    def _on_task_completed(self, prefix, result):
        self._set_forms_enabled(True)
        clip_id = getattr(result, "clip_id", result)
        self._status_label.setText(f"{prefix}: {clip_id}")
        self.refresh_dashboard()

    def _on_task_error(self, message):
        self._set_forms_enabled(True)
        self._status_label.setText("Failed -- see details.")
        QMessageBox.critical(self, "Error", message)

    def _set_forms_enabled(self, enabled: bool):
        for box in self.findChildren(QGroupBox):
            box.setEnabled(enabled)

    def _on_ingest_clicked(self):
        video = self._ingest_video.text().strip()
        csv = self._ingest_csv.text().strip()
        if not video or not csv:
            QMessageBox.warning(self, "Missing input", "Both a video and a CSV are required.")
            return
        source_type = self._ingest_source.currentData()
        tier = self._ingest_tier.text().strip() or None
        if source_type == "synthetic" and tier is None:
            QMessageBox.warning(self, "Missing tier", "Tier (num_agents) is required for synthetic clips.")
            return
        self._run(
            ingest, video, csv, tier=tier, source_type=source_type,
            split=self._ingest_split.currentData(), clip_id=self._ingest_clip_id.text().strip() or None,
            on_success_prefix="Ingested",
        )

    def _on_synthesize_clicked(self):
        try:
            frame_rate = float(self._synth_frame_rate.text())
            pixel_size = float(self._synth_pixel_size.text())
        except ValueError:
            QMessageBox.warning(self, "Invalid input", "Frame rate and pixel size must be numbers.")
            return
        self._run(
            _synthesize_then_ingest,
            str(self._synth_tier.value()), self._synth_frames.value(), frame_rate, pixel_size,
            self._synth_param_specs.text().strip(), self._synth_out_dir.text().strip(),
            self._synth_split.currentData(), self._synth_clip_id.text().strip(),
            on_success_prefix="Generated + ingested",
        )

    def _on_augment_clicked(self):
        selected = self._augment_list.selectedItems()
        if len(selected) < 2:
            QMessageBox.warning(self, "Select clips", "Select at least 2 clips to combine.")
            return
        clip_ids = [item.data(Qt.ItemDataRole.UserRole) for item in selected]
        self._run(
            _augment_then_ingest, clip_ids, self._augment_rotate.isChecked(), self._augment_mirror.isChecked(),
            self._augment_out_dir.text().strip(), self._augment_split.currentData(),
            self._augment_clip_id.text().strip(),
            on_success_prefix="Combined + ingested",
        )

    def _on_delete_clicked(self):
        selected = self._delete_list.selectedItems()
        if not selected:
            QMessageBox.warning(self, "Select clips", "Select at least one clip to delete.")
            return
        clip_ids = [item.data(Qt.ItemDataRole.UserRole) for item in selected]
        reply = QMessageBox.question(
            self, "Delete clips?",
            "Permanently delete these clips (registry entry + files on disk)? "
            "This cannot be undone.\n\n" + "\n".join(clip_ids),
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self._set_forms_enabled(False)
        self._status_label.setText("Deleting...")
        self._task = BackgroundTask(_delete_clips, clip_ids)
        self._task.completed.connect(self._on_delete_completed)
        self._task.error_occurred.connect(self._on_task_error)
        self._task.start()

    def _on_delete_completed(self, result):
        deleted, failed = result
        self._set_forms_enabled(True)
        msg = f"Deleted {len(deleted)} clip(s)."
        if failed:
            msg += f"  Failed: {'; '.join(failed)}"
        self._status_label.setText(msg)
        self.refresh_dashboard()
