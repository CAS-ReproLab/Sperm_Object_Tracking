#!/usr/bin/env python3
"""
labeler_gui.py -- copied from UnifiedFramework/labeler_gui.py verbatim except
for the sys.path shim below. labeler_gui_pkg and its dependencies (utils.py,
crossover_detector.py) use bare top-level imports (`import utils`, `from
crossover_detector import ...`) assuming a flat namespace -- same situation
as ctc_metrics in data_engine/stages/validate/tracking_metrics.py, same fix:
put this directory on sys.path once, in one documented place, rather than
requiring an external PYTHONPATH.

Run as a module from the repo root, e.g.:

    python -m data_engine.stages.label.labeler_gui --videofile path/to/video.mp4 --csvfile path/to/tracks.csv
"""
import argparse
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from PyQt6.QtWidgets import QApplication, QFileDialog


def main():
    parser = argparse.ArgumentParser(description="Manual sperm track labeler (PyQt6 GUI)")
    parser.add_argument("--videofile", type=str, default=None, help="Path to the video file")
    parser.add_argument("--csvfile",   type=str, default=None, help="Path to the tracks CSV")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setApplicationName("Sperm Track Labeler")
    app.setStyle("Fusion")

    videofile = args.videofile
    csvfile   = args.csvfile

    if videofile is None:
        videofile, _ = QFileDialog.getOpenFileName(
            None, "Select the video file", "",
            "Video files (*.mp4 *.avi *.mov *.mkv);;All files (*)",
        )
        if not videofile:
            print("No video file selected. Exiting.")
            sys.exit(0)

    if csvfile is None:
        csvfile, _ = QFileDialog.getOpenFileName(
            None, "Select the tracks CSV file", "",
            "CSV files (*.csv);;All files (*)",
        )
        if not csvfile:
            print("No CSV file selected. Exiting.")
            sys.exit(0)

    from labeler_gui_pkg.labeler_window import LabelerWindow
    window = LabelerWindow(videofile, csvfile)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
