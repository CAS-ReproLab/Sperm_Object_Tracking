"""
video_player.py -- a minimal in-app video player (play/pause + seek slider)
for previewing rendered overlay videos inside the hub, so nothing needs to
open in an external window (cv.imshow can't run inside a Qt app anyway).
Reads frames straight from cv2.VideoCapture on a QTimer; no extra dependency
(no QtMultimedia).
"""
import cv2
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSizePolicy, QSlider, QVBoxLayout, QWidget


class VideoPlayer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._cap = None
        self._n_frames = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance)

        layout = QVBoxLayout(self)
        self._view = QLabel("No video loaded.", self)
        self._view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._view.setMinimumSize(480, 480)
        # Ignored, not Expanding: a QLabel reports its pixmap as its preferred
        # size, so with Expanding every frame (scaled to the label's current
        # size) nudged the layout to make the label a little bigger, which the
        # next frame scaled up to in turn -- the slow "zoom" feedback loop.
        # Ignored makes the label size itself from the layout only.
        self._view.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self._view.setStyleSheet("background: black; color: gray;")
        layout.addWidget(self._view, 1)

        controls = QHBoxLayout()
        self._play_btn = QPushButton("Play", self)
        self._play_btn.setEnabled(False)
        self._play_btn.clicked.connect(self.toggle_play)
        controls.addWidget(self._play_btn)
        self._slider = QSlider(Qt.Orientation.Horizontal, self)
        self._slider.setEnabled(False)
        self._slider.sliderMoved.connect(self._seek)
        controls.addWidget(self._slider, 1)
        self._pos_label = QLabel("", self)
        controls.addWidget(self._pos_label)
        layout.addLayout(controls)

    def load(self, path, autoplay=True):
        self._timer.stop()
        if self._cap is not None:
            self._cap.release()
        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            self._cap = None
            self._view.setText(f"Could not open {path}")
            return
        self._n_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = self._cap.get(cv2.CAP_PROP_FPS) or 9.0
        self._interval_ms = max(1, int(1000 / fps))
        self._slider.setRange(0, max(0, self._n_frames - 1))
        self._slider.setEnabled(True)
        self._play_btn.setEnabled(True)
        self._show_frame_at(0)
        if autoplay:
            self.toggle_play()

    def toggle_play(self):
        if self._cap is None:
            return
        if self._timer.isActive():
            self._timer.stop()
            self._play_btn.setText("Play")
        else:
            self._timer.start(self._interval_ms)
            self._play_btn.setText("Pause")

    def _advance(self):
        ok, frame = self._cap.read()
        if not ok:  # end of video: loop back to the start, stay paused
            self._timer.stop()
            self._play_btn.setText("Play")
            self._show_frame_at(0)
            return
        self._render(frame, int(self._cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1)

    def _seek(self, index):
        self._show_frame_at(index)

    def _show_frame_at(self, index):
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self._cap.read()
        if ok:
            self._render(frame, index)

    def _render(self, frame_bgr, index):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        image = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        self._view.setPixmap(QPixmap.fromImage(image).scaled(
            self._view.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        self._slider.blockSignals(True)
        self._slider.setValue(index)
        self._slider.blockSignals(False)
        self._pos_label.setText(f"{index + 1}/{self._n_frames}")

    def closeEvent(self, event):
        self._timer.stop()
        if self._cap is not None:
            self._cap.release()
        super().closeEvent(event)
