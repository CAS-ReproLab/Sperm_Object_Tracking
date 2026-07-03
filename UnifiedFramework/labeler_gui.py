#!/usr/bin/env python3
import argparse
import sys

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
