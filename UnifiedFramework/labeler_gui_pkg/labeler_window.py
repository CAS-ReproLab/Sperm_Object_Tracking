import pandas as pd

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QSplitter, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QStatusBar, QMessageBox, QDialog, QDialogButtonBox,
    QListWidget, QListWidgetItem, QLineEdit, QCheckBox, QFileDialog, QTabWidget,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QKeySequence, QShortcut

import utils
from labeler_gui_pkg.labeler_video_panel import LabelerVideoPanel
from labeler_gui_pkg.sperm_panel import SpermPanel
from labeler_gui_pkg.filter_panel import FilterPanel
from labeler_gui_pkg.labels_config import load_labels, save_labels, DEFAULT_LABELS_PATH

_UNDO_LIMIT = 25


class MergeDialog(QDialog):
    """Confirms which sperm ID survives a merge. Defaults to the lower ID; swappable."""

    def __init__(self, id_a, id_b, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Confirm Merge")
        self._keep_id   = min(id_a, id_b)
        self._remove_id = max(id_a, id_b)

        layout = QVBoxLayout(self)
        self._label = QLabel()
        self._update_label()
        layout.addWidget(self._label)

        btn_swap = QPushButton("Swap (keep the other ID instead)")
        btn_swap.clicked.connect(self._swap)
        layout.addWidget(btn_swap)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _update_label(self):
        self._label.setText(
            f"Merge sperm {self._remove_id} into sperm {self._keep_id}.\n"
            f"ID {self._keep_id} will survive; ID {self._remove_id} will be removed."
        )

    def _swap(self):
        self._keep_id, self._remove_id = self._remove_id, self._keep_id
        self._update_label()

    def result_ids(self):
        return self._keep_id, self._remove_id


class ExportByLabelDialog(QDialog):
    """Lets the user pick which label(s) to include in a filtered export CSV."""

    def __init__(self, labels, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export by Label")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Include tracks with these labels:"))

        self._checks = []
        for label in labels:
            display = label if label else "(blank)"
            chk = QCheckBox(display)
            chk.setProperty("label_value", label)
            self._checks.append(chk)
            layout.addWidget(chk)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def selected_labels(self):
        return [c.property("label_value") for c in self._checks if c.isChecked()]


class EditLabelsDialog(QDialog):
    """Simple editor for the label config list (configs/labels.json)."""

    def __init__(self, labels, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Labels")
        self.setMinimumWidth(300)
        layout = QVBoxLayout(self)

        self._list = QListWidget()
        for label in labels:
            if label == "":
                continue   # blank is implicit, not user-editable
            self._list.addItem(QListWidgetItem(label))
        layout.addWidget(self._list)

        add_row = QHBoxLayout()
        self._new_label = QLineEdit()
        self._new_label.setPlaceholderText("New label…")
        btn_add = QPushButton("Add")
        btn_add.clicked.connect(self._add_label)
        add_row.addWidget(self._new_label)
        add_row.addWidget(btn_add)
        layout.addLayout(add_row)

        btn_remove = QPushButton("Remove Selected")
        btn_remove.clicked.connect(self._remove_selected)
        layout.addWidget(btn_remove)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _add_label(self):
        text = self._new_label.text().strip()
        if text:
            self._list.addItem(QListWidgetItem(text))
            self._new_label.clear()

    def _remove_selected(self):
        for item in self._list.selectedItems():
            self._list.takeItem(self._list.row(item))

    def result_labels(self):
        labels = [""]
        for i in range(self._list.count()):
            labels.append(self._list.item(i).text())
        return labels


class LabelerWindow(QMainWindow):

    def __init__(self, videofile, csvfile):
        super().__init__()
        self.setWindowTitle("Sperm Track Labeler")
        self.resize(1400, 850)

        self._videofile = videofile
        self._csvfile   = csvfile
        self._tracks    = None
        self._undo_stack = []
        self._dirty      = False

        self._labels = load_labels()

        self._build_ui()
        self._connect_signals()
        self._update_action_buttons([])
        self._load_data()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        splitter = QSplitter(Qt.Orientation.Horizontal)

        self._video_panel = LabelerVideoPanel()
        splitter.addWidget(self._video_panel)

        right_tabs = QTabWidget()

        # ── "Tracks" tab ──────────────────────────────────────────────────────
        tracks_tab = QWidget()
        right_layout = QVBoxLayout(tracks_tab)
        right_layout.setContentsMargins(4, 4, 4, 4)

        self._sperm_panel = SpermPanel(self._labels)
        right_layout.addWidget(self._sperm_panel)

        self._selection_status = QLabel()
        self._selection_status.setWordWrap(True)
        self._selection_status.setStyleSheet("color: #888; font-style: italic;")
        right_layout.addWidget(self._selection_status)

        self._chk_isolate_selected = QCheckBox("Isolate Selected Tracks")
        self._chk_isolate_selected.setToolTip(
            "Only show the currently selected sperm in the video. "
            "Click cells or table rows (Ctrl+Click to add more) to change the selection."
        )
        self._chk_isolate_selected.toggled.connect(self._on_isolate_selected_toggled)
        right_layout.addWidget(self._chk_isolate_selected)

        action_row = QHBoxLayout()
        self._btn_merge  = QPushButton("Merge Selected")
        self._btn_split  = QPushButton("Split at Current Frame")
        self._btn_swap   = QPushButton("Swap Trajectories at Frame")
        self._btn_delete = QPushButton("Delete Selected")
        for b in (self._btn_merge, self._btn_split, self._btn_swap, self._btn_delete):
            b.setEnabled(False)

        self._btn_merge.setToolTip(
            "Requires exactly 2 selected sperm.\n"
            "Select two: click one cell in the video, then Ctrl+Click another "
            "(or Ctrl+Click a second row below). Combines them into one track."
        )
        self._btn_split.setToolTip(
            "Requires exactly 1 selected sperm.\n"
            "Splits it into two tracks at the current playhead frame."
        )
        self._btn_swap.setToolTip(
            "Requires exactly 2 selected sperm.\n"
            "Fixes a crossover: swaps the two tracks' trajectories after the "
            "current frame, so each keeps its own ID but the paths that were "
            "mistakenly tracked through the crossing point are corrected."
        )
        self._btn_delete.setToolTip("Deletes all selected sperm tracks entirely.")

        self._btn_merge.clicked.connect(self._on_merge)
        self._btn_split.clicked.connect(self._on_split)
        self._btn_swap.clicked.connect(self._on_swap)
        self._btn_delete.clicked.connect(self._on_delete)
        action_row.addWidget(self._btn_merge)
        action_row.addWidget(self._btn_split)
        action_row.addWidget(self._btn_swap)
        action_row.addWidget(self._btn_delete)
        right_layout.addLayout(action_row)

        right_tabs.addTab(tracks_tab, "Tracks")

        # ── "Filters" tab ─────────────────────────────────────────────────────
        self._filter_panel = FilterPanel()
        self._filter_panel.isolate_requested.connect(self._on_filter_isolate_requested)
        self._filter_panel.event_selected.connect(self._on_filter_event_selected)
        right_tabs.addTab(self._filter_panel, "Filters")

        splitter.addWidget(right_tabs)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        self.setCentralWidget(splitter)

        self._status = QStatusBar()
        self.setStatusBar(self._status)

        self._build_toolbar()

    def _build_toolbar(self):
        bar = self.addToolBar("Actions")
        bar.setMovable(False)

        self._act_new_track = QAction("New Track", self)
        self._act_new_track.triggered.connect(self._on_new_track_start)
        bar.addAction(self._act_new_track)

        self._act_finish_track = QAction("Finish New Track", self)
        self._act_finish_track.triggered.connect(self._on_new_track_finish)
        self._act_finish_track.setEnabled(False)
        bar.addAction(self._act_finish_track)

        self._act_cancel_track = QAction("Cancel", self)
        self._act_cancel_track.triggered.connect(self._on_new_track_cancel)
        self._act_cancel_track.setEnabled(False)
        bar.addAction(self._act_cancel_track)

        bar.addSeparator()

        self._act_undo = QAction("Undo", self)
        self._act_undo.setShortcut(QKeySequence.StandardKey.Undo)
        self._act_undo.triggered.connect(self._on_undo)
        self._act_undo.setEnabled(False)
        bar.addAction(self._act_undo)

        self._act_save = QAction("Save", self)
        self._act_save.setShortcut(QKeySequence.StandardKey.Save)
        self._act_save.triggered.connect(self._on_save)
        bar.addAction(self._act_save)

        bar.addSeparator()

        act_randomize = QAction("Randomize Colors", self)
        act_randomize.triggered.connect(lambda: self._video_panel.randomize_colors())
        bar.addAction(act_randomize)

        act_interpolate = QAction("Interpolate Missing Frames", self)
        act_interpolate.triggered.connect(self._on_interpolate)
        bar.addAction(act_interpolate)

        bar.addSeparator()

        act_export = QAction("Export by Label…", self)
        act_export.triggered.connect(self._on_export_by_label)
        bar.addAction(act_export)

        act_edit_labels = QAction("Edit Labels…", self)
        act_edit_labels.triggered.connect(self._on_edit_labels)
        bar.addAction(act_edit_labels)

    def _connect_signals(self):
        self._video_panel.sperm_clicked.connect(self._on_video_sperm_clicked)
        self._sperm_panel.selection_changed.connect(self._on_table_selection_changed)
        self._sperm_panel.label_changed.connect(self._on_label_changed)

    # ── data loading ──────────────────────────────────────────────────────────

    def _load_data(self):
        self._status.showMessage(f"Loading video: {self._videofile}")
        frames = utils.loadVideo(self._videofile)
        self._video_panel.set_video(frames)

        self._status.showMessage(f"Loading tracks: {self._csvfile}")
        tracks = utils.loadDataFrame(self._csvfile, convert_segmentation=False)

        if "label" not in tracks.columns:
            if "keep" in tracks.columns:
                # Migrate legacy binary keep/unusable to the new label scheme.
                fallback_label = self._labels[1] if len(self._labels) > 1 else ""
                tracks["label"] = tracks["keep"].apply(lambda k: fallback_label if k == 1 else "")
            else:
                tracks["label"] = ""
        tracks["label"] = tracks["label"].fillna("")

        self._set_tracks(tracks, push_undo=False)
        self._status.showMessage("Ready.")

    # ── tracks state management ──────────────────────────────────────────────

    def _set_tracks(self, tracks, push_undo=True):
        if push_undo and self._tracks is not None:
            self._push_undo()
        self._tracks = tracks
        self._video_panel.set_tracks(self._tracks)
        self._sperm_panel.set_tracks(self._tracks)
        self._filter_panel.set_tracks(self._tracks)
        if push_undo:
            self._mark_dirty()

    def _push_undo(self):
        self._undo_stack.append(self._tracks.copy(deep=True))
        if len(self._undo_stack) > _UNDO_LIMIT:
            self._undo_stack.pop(0)
        self._act_undo.setEnabled(True)

    def _mark_dirty(self):
        self._dirty = True
        self.setWindowTitle("Sperm Track Labeler *")

    def _mark_clean(self):
        self._dirty = False
        self.setWindowTitle("Sperm Track Labeler")

    # ── selection sync ────────────────────────────────────────────────────────

    def _on_video_sperm_clicked(self, sperm_id, ctrl_held):
        current = self._sperm_panel.selected_ids()
        if ctrl_held:
            if sperm_id in current:
                current = [s for s in current if s != sperm_id]
            else:
                current = current + [sperm_id]
        else:
            current = [sperm_id]
        self._sperm_panel.select_ids(current)
        # select_ids() suppresses selection_changed, so update video + buttons directly
        self._video_panel.set_selected_ids(current)
        self._update_action_buttons(current)

    def _on_table_selection_changed(self, ids):
        self._video_panel.set_selected_ids(ids)
        self._update_action_buttons(ids)

    def _update_action_buttons(self, ids):
        self._btn_merge.setEnabled(len(ids) == 2)
        self._btn_split.setEnabled(len(ids) == 1)
        self._btn_swap.setEnabled(len(ids) == 2)
        self._btn_delete.setEnabled(len(ids) >= 1)
        self._update_selection_status(ids)
        if self._chk_isolate_selected.isChecked():
            self._video_panel.set_visible_ids(ids)

    def _update_selection_status(self, ids):
        if len(ids) == 0:
            text = ("No selection. Click a cell in the video, or select a row below. "
                    "Ctrl+Click adds a second sperm to the selection.")
        elif len(ids) == 1:
            text = (f"1 selected: sperm {ids[0]}. "
                    f"Ctrl+Click another cell (or Ctrl+Click a second row below) "
                    f"to enable Merge / Swap.")
        elif len(ids) == 2:
            text = f"2 selected: sperm {ids[0]} and {ids[1]}. Ready for Merge or Swap."
        else:
            text = f"{len(ids)} selected. Merge and Swap need exactly 2; Split needs exactly 1."
        self._selection_status.setText(text)

    # ── track filtering ───────────────────────────────────────────────────────

    def _on_isolate_selected_toggled(self, checked):
        if checked:
            # Mutually exclusive with the Filters tab's isolate — most recent wins.
            self._filter_panel.set_isolate_checked(False)
        ids = self._sperm_panel.selected_ids()
        self._video_panel.set_visible_ids(ids if checked else None)
        self._status.showMessage(
            f"Isolating {len(ids)} selected track(s)." if checked else "Showing all tracks."
        )

    def _on_filter_isolate_requested(self, ids):
        if ids is not None:
            self._chk_isolate_selected.blockSignals(True)
            self._chk_isolate_selected.setChecked(False)
            self._chk_isolate_selected.blockSignals(False)
            self._status.showMessage(f"Isolating {len(ids)} filtered track(s).")
        else:
            self._status.showMessage("Showing all tracks.")
        self._video_panel.set_visible_ids(ids)

    def _on_filter_event_selected(self, frame, sperm_ids):
        self._video_panel.goto_frame(frame)
        self._sperm_panel.select_ids(sperm_ids)
        self._video_panel.set_selected_ids(sperm_ids)
        self._update_action_buttons(sperm_ids)
        self._status.showMessage(
            f"Jumped to frame {frame}, selected sperm {', '.join(str(i) for i in sperm_ids)}."
        )

    # ── label editing ─────────────────────────────────────────────────────────

    def _on_label_changed(self, sperm_id, new_label):
        self._push_undo()
        self._tracks.loc[self._tracks["sperm"] == sperm_id, "label"] = new_label
        self._mark_dirty()
        self._status.showMessage(f"Sperm {sperm_id} labeled '{new_label or '(blank)'}'.")

    # ── merge / split / delete ────────────────────────────────────────────────

    def _on_merge(self):
        ids = self._sperm_panel.selected_ids()
        if len(ids) != 2:
            return
        dialog = MergeDialog(ids[0], ids[1], parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        keep_id, remove_id = dialog.result_ids()

        self._push_undo()
        original_label = self._tracks.loc[self._tracks["sperm"] == keep_id, "label"]
        keep_label = original_label.iloc[0] if len(original_label) > 0 else ""

        self._tracks.loc[self._tracks["sperm"] == remove_id, "sperm"] = keep_id
        self._tracks.loc[self._tracks["sperm"] == keep_id, "label"] = keep_label

        self._refresh_after_edit()
        self._mark_dirty()
        self._status.showMessage(f"Merged {remove_id} into {keep_id}.")

    def _on_split(self):
        ids = self._sperm_panel.selected_ids()
        if len(ids) != 1:
            return
        sperm_id  = ids[0]
        frame_num = self._video_panel.current_frame

        reply = QMessageBox.question(
            self, "Confirm Split",
            f"Split sperm {sperm_id} at frame {frame_num}?\n"
            f"Frames after {frame_num} will become a new track.",
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self._push_undo()
        new_id = int(self._tracks["sperm"].max()) + 1
        mask = (self._tracks["sperm"] == sperm_id) & (self._tracks["frame"] > frame_num)
        self._tracks.loc[mask, "sperm"] = new_id

        self._refresh_after_edit()
        self._mark_dirty()
        self._status.showMessage(f"Split sperm {sperm_id} at frame {frame_num} → new ID {new_id}.")

    def _on_swap(self):
        """
        Fix a crossover: swap the trajectories of two sperm after the current frame.
        Equivalent to splitting both at the frame and merging each new fragment
        into the other original ID, but done as one direct reassignment so no
        intermediate IDs are created.
        """
        ids = self._sperm_panel.selected_ids()
        if len(ids) != 2:
            return
        id_a, id_b = ids
        frame_num  = self._video_panel.current_frame

        reply = QMessageBox.question(
            self, "Confirm Swap",
            f"Swap trajectories of sperm {id_a} and {id_b} after frame {frame_num}?\n\n"
            f"Frames after {frame_num} currently belonging to {id_a} will become "
            f"part of {id_b}, and vice versa. Both IDs are kept; only the "
            f"post-crossing paths are exchanged.",
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self._push_undo()

        label_a = self._tracks.loc[self._tracks["sperm"] == id_a, "label"].iloc[0]
        label_b = self._tracks.loc[self._tracks["sperm"] == id_b, "label"].iloc[0]

        mask_a = (self._tracks["sperm"] == id_a) & (self._tracks["frame"] > frame_num)
        mask_b = (self._tracks["sperm"] == id_b) & (self._tracks["frame"] > frame_num)
        self._tracks.loc[mask_a, "sperm"] = id_b
        self._tracks.loc[mask_b, "sperm"] = id_a

        # Each ID keeps its own pre-existing label — the swap corrects a tracking
        # error, it does not merge two distinct organisms' identities.
        self._tracks.loc[self._tracks["sperm"] == id_a, "label"] = label_a
        self._tracks.loc[self._tracks["sperm"] == id_b, "label"] = label_b

        self._refresh_after_edit()
        self._mark_dirty()
        self._status.showMessage(
            f"Swapped trajectories of {id_a} and {id_b} after frame {frame_num}."
        )

    def _on_delete(self):
        ids = self._sperm_panel.selected_ids()
        if not ids:
            return
        reply = QMessageBox.question(
            self, "Confirm Delete",
            f"Delete {len(ids)} track(s): {', '.join(str(i) for i in ids)}?",
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self._push_undo()
        self._tracks = self._tracks[~self._tracks["sperm"].isin(ids)]

        self._refresh_after_edit()
        self._mark_dirty()
        self._status.showMessage(f"Deleted {len(ids)} track(s).")

    def _refresh_after_edit(self):
        self._video_panel.set_tracks(self._tracks)
        self._sperm_panel.set_tracks(self._tracks)
        self._filter_panel.set_tracks(self._tracks)
        self._video_panel.set_selected_ids([])

    # ── new track placement ──────────────────────────────────────────────────

    def _on_new_track_start(self):
        self._video_panel.start_placement()
        self._act_new_track.setEnabled(False)
        self._act_finish_track.setEnabled(True)
        self._act_cancel_track.setEnabled(True)
        self._status.showMessage("Placing new track — click each frame's cell location.")

    def _on_new_track_finish(self):
        points = self._video_panel.finish_placement()
        self._act_new_track.setEnabled(True)
        self._act_finish_track.setEnabled(False)
        self._act_cancel_track.setEnabled(False)

        if not points:
            self._status.showMessage("No points placed — new track discarded.")
            return

        self._push_undo()
        new_id = int(self._tracks["sperm"].max()) + 1
        new_rows = pd.DataFrame([
            {"frame": f, "sperm": new_id, "x": x, "y": y, "label": ""}
            for f, x, y in points
        ])
        self._tracks = pd.concat([self._tracks, new_rows], ignore_index=True)
        self._tracks = self._tracks.fillna(0)

        self._refresh_after_edit()
        self._mark_dirty()
        self._status.showMessage(f"Created new track {new_id} with {len(points)} points.")

    def _on_new_track_cancel(self):
        self._video_panel.cancel_placement()
        self._act_new_track.setEnabled(True)
        self._act_finish_track.setEnabled(False)
        self._act_cancel_track.setEnabled(False)
        self._status.showMessage("New track cancelled.")

    # ── undo ──────────────────────────────────────────────────────────────────

    def _on_undo(self):
        if not self._undo_stack:
            return
        self._tracks = self._undo_stack.pop()
        self._video_panel.set_tracks(self._tracks)
        self._sperm_panel.set_tracks(self._tracks)
        self._filter_panel.set_tracks(self._tracks)
        self._video_panel.set_selected_ids([])
        self._act_undo.setEnabled(bool(self._undo_stack))
        self._mark_dirty()
        self._status.showMessage("Undo applied.")

    # ── interpolation ─────────────────────────────────────────────────────────

    def _on_interpolate(self):
        self._push_undo()
        self._status.showMessage("Interpolating missing frames…")
        cleaned = utils.dropDuplicates(self._tracks)
        interpolated = utils.interpolateTracks(cleaned)
        interpolated["label"] = interpolated["label"].fillna("") if "label" in interpolated.columns else ""
        interpolated = interpolated.fillna(0)
        self._tracks = interpolated

        self._refresh_after_edit()
        self._mark_dirty()
        self._status.showMessage("Interpolation complete.")

    # ── save / export ─────────────────────────────────────────────────────────

    def _on_save(self):
        savefile = self._csvfile.replace(".csv", "_corrected.csv")
        utils.saveDataFrame(self._tracks, savefile)
        self._mark_clean()
        self._status.showMessage(f"Saved to {savefile}")

    def _on_export_by_label(self):
        dialog = ExportByLabelDialog(self._labels, parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        selected_labels = dialog.selected_labels()
        if not selected_labels:
            QMessageBox.information(self, "No labels selected", "Select at least one label to export.")
            return

        subset = self._tracks[self._tracks["label"].isin(selected_labels)]
        default_path = self._csvfile.replace(".csv", "_filtered.csv")
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Filtered CSV", default_path, "CSV files (*.csv)"
        )
        if path:
            utils.saveDataFrame(subset, path)
            self._status.showMessage(f"Exported {len(subset)} rows to {path}")

    def _on_edit_labels(self):
        dialog = EditLabelsDialog(self._labels, parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self._labels = dialog.result_labels()
        save_labels(self._labels, DEFAULT_LABELS_PATH)
        self._sperm_panel.set_labels(self._labels)
        self._status.showMessage("Labels updated.")

    # ── close handling ────────────────────────────────────────────────────────

    def closeEvent(self, event):
        if self._dirty:
            reply = QMessageBox.question(
                self, "Unsaved changes",
                "You have unsaved changes. Save before closing?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No |
                QMessageBox.StandardButton.Cancel,
            )
            if reply == QMessageBox.StandardButton.Cancel:
                event.ignore()
                return
            if reply == QMessageBox.StandardButton.Yes:
                self._on_save()
        event.accept()
