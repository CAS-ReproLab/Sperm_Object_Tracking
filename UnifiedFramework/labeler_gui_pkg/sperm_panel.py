from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QTableWidget, QTableWidgetItem, QComboBox, QAbstractItemView,
    QHeaderView,
)
from PyQt6.QtCore import Qt, pyqtSignal


class SpermPanel(QWidget):
    """
    Searchable table of every sperm ID: frame range, point count, and a
    per-row label dropdown (populated from the labels config). Selection is
    synced bidirectionally with the video panel's click-to-select.
    """

    selection_changed = pyqtSignal(list)         # ordered list of selected sperm IDs
    label_changed      = pyqtSignal(int, str)    # sperm_id, new label

    _ID_COL, _FRAMES_COL, _COUNT_COL, _LABEL_COL = range(4)

    def __init__(self, labels, parent=None):
        super().__init__(parent)
        self._labels        = labels
        self._tracks        = None
        self._row_by_id      = {}     # sperm_id -> row index
        self._suppress_signal = False  # guards against feedback loops while rebuilding

        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        search_row = QHBoxLayout()
        search_row.addWidget(QLabel("Search:"))
        self._search = QLineEdit()
        self._search.setPlaceholderText("Filter by ID or label…")
        self._search.textChanged.connect(self._apply_filter)
        search_row.addWidget(self._search)
        layout.addLayout(search_row)

        self._table = QTableWidget()
        self._table.setColumnCount(4)
        self._table.setHorizontalHeaderLabels(["ID", "Frames", "Points", "Label"])
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.horizontalHeader().setSectionResizeMode(
            self._LABEL_COL, QHeaderView.ResizeMode.Stretch
        )
        self._table.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self._table)

        self._summary = QLabel("0 tracks")
        layout.addWidget(self._summary)

    # ── public API ────────────────────────────────────────────────────────────

    def set_labels(self, labels):
        self._labels = labels
        if self._tracks is not None:
            self.set_tracks(self._tracks)   # rebuild dropdowns with new options

    def set_tracks(self, tracks):
        """Rebuild the full table from the tracks DataFrame. Preserves selection by ID."""
        previous_selection = set(self.selected_ids())
        self._tracks = tracks
        self._suppress_signal = True

        grouped = tracks.groupby("sperm")
        sperm_ids = sorted(grouped.groups.keys())

        self._table.setRowCount(len(sperm_ids))
        self._row_by_id = {}

        for row, sperm_id in enumerate(sperm_ids):
            sid   = int(sperm_id)
            group = grouped.get_group(sperm_id)
            frame_min = int(group["frame"].min())
            frame_max = int(group["frame"].max())
            count     = len(group)
            label     = group["label"].iloc[0] if "label" in group.columns else ""

            self._row_by_id[sid] = row

            id_item = QTableWidgetItem(str(sid))
            id_item.setData(Qt.ItemDataRole.UserRole, sid)
            self._table.setItem(row, self._ID_COL, id_item)
            self._table.setItem(row, self._FRAMES_COL,
                                QTableWidgetItem(f"{frame_min}–{frame_max}"))
            self._table.setItem(row, self._COUNT_COL, QTableWidgetItem(str(count)))

            combo = QComboBox()
            combo.addItems(self._labels)
            idx = combo.findText(label)
            combo.setCurrentIndex(idx if idx >= 0 else 0)
            combo.currentTextChanged.connect(
                lambda text, sid=sid: self.label_changed.emit(sid, text)
            )
            self._table.setCellWidget(row, self._LABEL_COL, combo)

        self._table.resizeColumnsToContents()
        self._summary.setText(f"{len(sperm_ids)} tracks")

        # Restore selection for IDs that still exist
        still_present = [sid for sid in previous_selection if sid in self._row_by_id]
        self._select_ids_internal(still_present)

        self._suppress_signal = False
        self._apply_filter(self._search.text())

    def selected_ids(self):
        """Return selected sperm IDs in the order the rows appear in the table (ascending row index)."""
        rows = sorted(set(idx.row() for idx in self._table.selectedIndexes()))
        ids  = []
        for row in rows:
            item = self._table.item(row, self._ID_COL)
            if item is not None:
                ids.append(item.data(Qt.ItemDataRole.UserRole))
        return ids

    def select_ids(self, ids):
        """Programmatically set the table selection (e.g. from a video click)."""
        self._select_ids_internal(ids)

    def set_label_for_row(self, sperm_id, label):
        """Update a row's combo box without re-triggering label_changed (external sync)."""
        row = self._row_by_id.get(sperm_id)
        if row is None:
            return
        combo = self._table.cellWidget(row, self._LABEL_COL)
        if combo is not None:
            combo.blockSignals(True)
            idx = combo.findText(label)
            if idx >= 0:
                combo.setCurrentIndex(idx)
            combo.blockSignals(False)

    # ── internal ──────────────────────────────────────────────────────────────

    def _select_ids_internal(self, ids):
        self._suppress_signal = True
        self._table.clearSelection()
        for sid in ids:
            row = self._row_by_id.get(sid)
            if row is not None:
                self._table.selectRow(row)
        self._suppress_signal = False

    def _on_selection_changed(self):
        if self._suppress_signal:
            return
        self.selection_changed.emit(self.selected_ids())

    def _apply_filter(self, text):
        text = text.strip().lower()
        for row in range(self._table.rowCount()):
            if not text:
                self._table.setRowHidden(row, False)
                continue
            id_text    = self._table.item(row, self._ID_COL).text().lower()
            combo      = self._table.cellWidget(row, self._LABEL_COL)
            label_text = combo.currentText().lower() if combo else ""
            match = text in id_text or text in label_text
            self._table.setRowHidden(row, not match)
