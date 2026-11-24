"""
hub.py -- the data-engine GUI hub: one app, one tab per loop stage.

Run as a module from the repo root (matches every other stage's invocation
convention in this codebase, which matters here specifically because
labeler_gui_pkg/labels_config.py's DEFAULT_LABELS_PATH is repo-root-relative
-- see label_tab.py's docstring):

    python -m data_engine.gui.hub
"""
import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication, QMainWindow, QScrollArea, QTabWidget

REPO_ROOT = Path(__file__).resolve().parents[2]

# labeler_gui_pkg uses bare imports (`import utils`, etc.) assuming
# data_engine/stages/label/ itself (not labeler_gui_pkg/) is on sys.path --
# same shim as labeler_gui.py/correction_flow.py, done once here so any tab
# that imports labeler_gui_pkg works regardless of import order.
_LABEL_STAGE_DIR = REPO_ROOT / "data_engine" / "stages" / "label"
if str(_LABEL_STAGE_DIR) not in sys.path:
    sys.path.insert(0, str(_LABEL_STAGE_DIR))


def _scrollable(widget) -> QScrollArea:
    """Wraps a tab's content in a vertically-scrolling area instead of
    letting it get squished into the tab page's fixed height -- every tab
    here is a stack of QGroupBoxes that only grows as features are added
    (Prepare Data's forms, Train's accumulating job panels), so this is
    applied to all of them rather than patched in per-tab as each one
    happens to overflow."""
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setWidget(widget)
    return scroll


class HubWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sperm Data Engine")
        self.resize(1100, 800)

        tabs = QTabWidget(self)
        self.setCentralWidget(tabs)

        from data_engine.gui.evaluate_tab import EvaluateTab
        from data_engine.gui.label_tab import LabelTab
        from data_engine.gui.prepare_data_tab import PrepareDataTab
        from data_engine.gui.train_tab import TrainTab
        tabs.currentChanged.connect(self._on_tab_changed)
        tabs.addTab(_scrollable(LabelTab()), "Label")
        tabs.addTab(_scrollable(PrepareDataTab()), "Prepare Data")
        tabs.addTab(_scrollable(TrainTab()), "Train")
        tabs.addTab(_scrollable(EvaluateTab()), "Evaluate")
        self._tabs = tabs

    def _on_tab_changed(self, index):
        """Re-scan the registry / checkpoint folders whenever a tab is opened,
        so new clips, tiers, and trained checkpoints show up without a restart."""
        tabs = getattr(self, "_tabs", None)
        if tabs is None:
            return
        hook = getattr(tabs.widget(index).widget(), "refresh_choices", None)
        if hook:
            hook()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Sperm Data Engine")
    app.setStyle("Fusion")

    window = HubWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
