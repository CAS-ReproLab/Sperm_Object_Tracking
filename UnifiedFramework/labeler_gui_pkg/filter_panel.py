import traceback

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QLabel, QComboBox, QSpinBox, QDoubleSpinBox, QPushButton,
    QTableWidget, QTableWidgetItem, QAbstractItemView, QMessageBox,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from labeler_gui_pkg.filters import FILTERS, FILTER_PARAMS


class FilterWorker(QThread):
    completed      = pyqtSignal(object)   # FilterResult
    error_occurred = pyqtSignal(str)

    def __init__(self, fn, tracks, config, parent=None):
        super().__init__(parent)
        self._fn     = fn
        self._tracks = tracks
        self._config = config

    def run(self):
        try:
            result = self._fn(self._tracks, self._config)
            self.completed.emit(result)
        except Exception:
            self.error_occurred.emit(traceback.format_exc())


class FilterPanel(QWidget):
    """
    Runs a registered track filter (see labeler_gui_pkg/filters/) against the
    current tracks and shows a results table of flagged events. Double-clicking
    a row jumps the video to that event's frame and selects the sperm involved
    (ready for Merge/Swap). "Isolate Results" restricts the video view to only
    the sperm the filter flagged.
    """

    isolate_requested = pyqtSignal(object)   # set of sperm IDs, or None to show all
    event_selected     = pyqtSignal(int, list)  # frame, sperm_ids

    def __init__(self, parent=None):
        super().__init__(parent)
        self._tracks       = None
        self._last_result   = None
        self._param_widgets = {}
        self._worker        = None
        self._build_ui()
        self._rebuild_params()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        form_group = QGroupBox("Filter")
        form = QFormLayout(form_group)

        self._combo_filter = QComboBox()
        self._combo_filter.addItems(list(FILTERS.keys()))
        self._combo_filter.currentIndexChanged.connect(self._rebuild_params)
        form.addRow("Type:", self._combo_filter)

        self._params_box = QGroupBox()
        self._params_box.setFlat(True)
        self._params_layout = QFormLayout(self._params_box)
        form.addRow(self._params_box)

        layout.addWidget(form_group)

        btn_row = QHBoxLayout()
        self._btn_run = QPushButton("Run Filter")
        self._btn_run.clicked.connect(self._on_run)
        btn_row.addWidget(self._btn_run)

        self._btn_isolate = QPushButton("Isolate Results")
        self._btn_isolate.setCheckable(True)
        self._btn_isolate.setEnabled(False)
        self._btn_isolate.toggled.connect(self._on_isolate_toggled)
        btn_row.addWidget(self._btn_isolate)

        self._btn_clear = QPushButton("Show All")
        self._btn_clear.clicked.connect(self._on_clear)
        btn_row.addWidget(self._btn_clear)
        layout.addLayout(btn_row)

        self._summary = QLabel("No results yet. Choose a filter and click Run.")
        self._summary.setWordWrap(True)
        layout.addWidget(self._summary)

        self._table = QTableWidget()
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setToolTip("Double-click a row to jump to that event's frame "
                               "and select the sperm involved.")
        self._table.cellDoubleClicked.connect(self._on_row_double_clicked)
        layout.addWidget(self._table)

    # ── public API ────────────────────────────────────────────────────────────

    def set_tracks(self, tracks):
        self._tracks = tracks

    def set_isolate_checked(self, checked):
        """External sync point (e.g. main window enforcing mutual exclusion)."""
        self._btn_isolate.blockSignals(True)
        self._btn_isolate.setChecked(checked)
        self._btn_isolate.blockSignals(False)

    # ── param widgets ─────────────────────────────────────────────────────────

    def _rebuild_params(self, *_):
        name   = self._combo_filter.currentText()
        params = FILTER_PARAMS.get(name, {})

        while self._params_layout.rowCount():
            self._params_layout.removeRow(0)
        self._param_widgets = {}

        for key, schema in params.items():
            widget = self._make_widget(schema)
            self._params_layout.addRow(schema.get("label", key) + ":", widget)
            self._param_widgets[key] = widget

    def _make_widget(self, schema):
        kind = schema.get("type", "float")
        if kind == "int":
            w = QSpinBox()
            w.setRange(schema.get("min", 0), schema.get("max", 9999))
            w.setSingleStep(schema.get("step", 1))
            w.setValue(schema.get("default", 0))
        else:
            w = QDoubleSpinBox()
            w.setRange(schema.get("min", 0.0), schema.get("max", 9999.0))
            w.setDecimals(3)
            w.setSingleStep(schema.get("step", 0.1))
            w.setValue(schema.get("default", 0.0))
        return w

    def _get_config(self):
        return {key: widget.value() for key, widget in self._param_widgets.items()}

    # ── run filter ────────────────────────────────────────────────────────────

    def _on_run(self):
        if self._tracks is None:
            QMessageBox.information(self, "No data", "Load a video and tracks first.")
            return

        name   = self._combo_filter.currentText()
        fn     = FILTERS[name]
        config = self._get_config()

        self._btn_run.setEnabled(False)
        self._btn_run.setText("Running…")
        self._summary.setText("Running filter…")

        self._worker = FilterWorker(fn, self._tracks, config, parent=self)
        self._worker.completed.connect(self._on_completed)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.start()

    def _on_completed(self, result):
        self._btn_run.setEnabled(True)
        self._btn_run.setText("Run Filter")
        self._last_result = result

        n_ids    = len(result.sperm_ids)
        n_events = 0 if result.events is None else len(result.events)
        self._summary.setText(f"{n_events} event(s) found, involving {n_ids} sperm.")

        self._populate_table(result.events)

        self._btn_isolate.setEnabled(n_ids > 0)
        if n_ids == 0:
            self._btn_isolate.setChecked(False)
        elif self._btn_isolate.isChecked():
            # Refresh an already-active isolation with the new result
            self.isolate_requested.emit(result.sperm_ids)

    def _on_error(self, message):
        self._btn_run.setEnabled(True)
        self._btn_run.setText("Run Filter")
        self._summary.setText("Filter failed — see error dialog.")
        QMessageBox.critical(self, "Filter error", message)

    # ── results table ─────────────────────────────────────────────────────────

    def _populate_table(self, events):
        self._table.clear()
        if events is None or events.empty:
            self._table.setRowCount(0)
            self._table.setColumnCount(0)
            return

        columns = list(events.columns)
        self._table.setColumnCount(len(columns))
        self._table.setHorizontalHeaderLabels(columns)
        self._table.setRowCount(len(events))

        for row_i, (_, row) in enumerate(events.iterrows()):
            for col_i, col in enumerate(columns):
                val = row[col]
                if col == "sperm_ids":
                    text = ", ".join(str(int(v)) for v in val)
                elif isinstance(val, float):
                    text = f"{val:.3f}"
                else:
                    text = str(val)
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self._table.setItem(row_i, col_i, item)

        self._table.resizeColumnsToContents()

    def _on_row_double_clicked(self, row, _col):
        if self._last_result is None or self._last_result.events is None:
            return
        events = self._last_result.events
        if row >= len(events):
            return
        record    = events.iloc[row]
        frame     = int(record["frame"])
        sperm_ids = [int(s) for s in record["sperm_ids"]]
        self.event_selected.emit(frame, sperm_ids)

    # ── isolate / clear ───────────────────────────────────────────────────────

    def _on_isolate_toggled(self, checked):
        if checked and self._last_result is not None:
            self.isolate_requested.emit(self._last_result.sperm_ids)
        else:
            self.isolate_requested.emit(None)

    def _on_clear(self):
        self._btn_isolate.setChecked(False)
        self.isolate_requested.emit(None)
