"""
correction_flow.py -- shared "launch the labeler, wait, find what got saved"
logic used by both label.py (from-scratch labeling) and bootstrap_label.py
(model-assisted labeling). The only difference between those two callers is
how the starting CSV gets produced; everything after that is identical.

labeler_window.py's Save action never overwrites its input CSV -- it writes
<input>.csv -> <input>_corrected.csv. On close, if there are unsaved edits,
it prompts Yes/No/Cancel; if the human picks No, or made no edits at all, no
_corrected.csv ever gets written. launch_labeler_and_wait() reflects that:
it returns None rather than assuming a corrected file exists.
"""
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))


def corrected_csv_path(initial_csv) -> Path:
    """Where labeler_window.py's Save action would write, for a given input CSV."""
    return Path(str(initial_csv).replace(".csv", "_corrected.csv"))


def launch_labeler_and_wait(video_path, initial_csv):
    """
    Blocks until the labeler GUI window closes.

    Returns the path to the human-corrected CSV if one was saved, or None if
    the human closed without saving (declined the save prompt, or made no
    edits at all -- both are legitimate outcomes, not errors).
    """
    from PyQt6.QtWidgets import QApplication

    from labeler_gui_pkg.labeler_window import LabelerWindow

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("Sperm Track Labeler")
    app.setStyle("Fusion")

    window = LabelerWindow(str(video_path), str(initial_csv))
    window.show()
    app.exec()

    out_path = corrected_csv_path(initial_csv)
    return out_path if out_path.exists() else None
