"""
widgets.py -- small reusable PyQt6 helpers shared across the hub's tabs, so
hand-written forms stay visually consistent without a generic reflection-
based form builder (Prepare Data/Evaluate call their underlying Python
functions directly -- there's no argparse layer to reflect there in the
first place; see gui/train_tab.py for the one place reflection genuinely is
the simpler option, because that tab has to produce a subprocess argv list
regardless).

BackgroundTask generalizes the existing QThread pattern already used in
data_engine/stages/label/labeler_gui_pkg/filter_panel.py's FilterWorker
(subclass QThread, override run(), emit a result/error signal rather than
raising across threads) -- reused here instead of reinvented, and every
hub tab that calls a stage function directly (not via subprocess) should
run it through this instead of blocking the GUI thread.
"""
import traceback

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import QComboBox, QFileDialog, QHBoxLayout, QLineEdit, QPushButton, QSpinBox, QWidget

from data_engine.registry.registry import REPO_ROOT

MODEL_SEARCH_DIRS = ["training_weights", "data_engine/models"]


def find_checkpoints():
    """Every trained checkpoint under the two conventional output roots
    (legacy training_weights/<run>/weights/*.pt and data_engine/models/
    <run>/weights/*.pt), repo-relative -- shared by every tab that lets you
    pick a previous training run's weights (Evaluate's Metrics/Visualize/
    Crossover sub-tabs, Label's pre-track model picker)."""
    paths = []
    for rel in MODEL_SEARCH_DIRS:
        root = REPO_ROOT / rel
        if root.is_dir():
            paths.extend(sorted(root.glob("*/weights/*.pt")))
    return [str(p.relative_to(REPO_ROOT)) for p in paths]


def set_combo_items(combo, items):
    """Repopulate a combo from a fresh list without losing what the user has
    selected/typed (signals blocked so a refresh never looks like a choice)."""
    items = list(items)
    current = combo.currentText()
    combo.blockSignals(True)
    combo.clear()
    combo.addItems(items)
    if combo.isEditable() or current in items:
        combo.setCurrentText(current)
    combo.blockSignals(False)


def set_list_items(list_widget, items):
    """Repopulate a QListWidget, re-selecting whatever was selected and still exists."""
    selected = {i.text() for i in list_widget.selectedItems()}
    list_widget.blockSignals(True)
    list_widget.clear()
    list_widget.addItems(list(items))
    for row in range(list_widget.count()):
        if list_widget.item(row).text() in selected:
            list_widget.item(row).setSelected(True)
    list_widget.blockSignals(False)


class BackgroundTask(QThread):
    """Runs `fn(*args, **kwargs)` off the GUI thread. Connect `completed`
    for the return value, `error_occurred` for a formatted traceback string
    (exceptions must never cross a thread boundary directly)."""
    completed = pyqtSignal(object)
    error_occurred = pyqtSignal(str)

    def __init__(self, fn, *args, parent=None, **kwargs):
        super().__init__(parent)
        self._fn = fn
        self._args = args
        self._kwargs = kwargs

    def run(self):
        try:
            result = self._fn(*self._args, **self._kwargs)
        except Exception:
            self.error_occurred.emit(traceback.format_exc())
        else:
            self.completed.emit(result)


def labeled_file_picker(parent=None, filter_str="All files (*)", mode="open", initial=""):
    """A QLineEdit + Browse button side by side, as one composite widget --
    add it to a QFormLayout with addRow("Label:", widget), and read/set the
    path via the returned QLineEdit. `mode`: "open", "save", or "dir"."""
    container = QWidget(parent)
    layout = QHBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    line_edit = QLineEdit(initial, container)
    browse_btn = QPushButton("Browse...", container)
    layout.addWidget(line_edit)
    layout.addWidget(browse_btn)

    def _browse():
        if mode == "dir":
            path = QFileDialog.getExistingDirectory(container, "Select directory")
        elif mode == "save":
            path, _ = QFileDialog.getSaveFileName(container, "Select file", "", filter_str)
        else:
            path, _ = QFileDialog.getOpenFileName(container, "Select file", "", filter_str)
        if path:
            line_edit.setText(path)

    browse_btn.clicked.connect(_browse)
    return container, line_edit


def checkpoint_picker(parent=None):
    """An editable combo box pre-filled with find_checkpoints() (pick a
    previous training run's weights with one click) plus a Browse... button
    for anything outside those two folders -- read/set the path via the
    returned QComboBox's currentText()/setCurrentText(). Call refresh() on
    the returned combo (attached as an attribute) to re-scan after a new
    run finishes."""
    container = QWidget(parent)
    layout = QHBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    combo = QComboBox(container)
    combo.setEditable(True)
    combo.addItems(find_checkpoints())
    browse_btn = QPushButton("Browse...", container)
    layout.addWidget(combo, 1)
    layout.addWidget(browse_btn)

    def _browse():
        path, _ = QFileDialog.getOpenFileName(container, "Select checkpoint", "", "Weights (*.pt)")
        if path:
            combo.setCurrentText(path)

    def _refresh():
        set_combo_items(combo, find_checkpoints())

    browse_btn.clicked.connect(_browse)
    combo.refresh = _refresh
    return container, combo


def labeled_combo(choices, default=None, parent=None) -> QComboBox:
    """choices: list of strings, or list of (display_text, value) pairs."""
    combo = QComboBox(parent)
    for choice in choices:
        if isinstance(choice, tuple):
            text, value = choice
            combo.addItem(text, value)
        else:
            combo.addItem(str(choice), choice)
    if default is not None:
        idx = combo.findData(default)
        if idx < 0:
            idx = combo.findText(str(default))
        if idx >= 0:
            combo.setCurrentIndex(idx)
    return combo


def labeled_spinbox(minimum, maximum, default, parent=None) -> QSpinBox:
    spin = QSpinBox(parent)
    spin.setRange(minimum, maximum)
    spin.setValue(default)
    return spin
