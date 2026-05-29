import numpy as np
import cv2 as cv

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider,
    QPushButton, QCheckBox, QSpinBox, QSizePolicy, QGroupBox,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap

import utils


class VideoPanel(QWidget):
    """
    Displays video frames with optional overlays for detections, tracks,
    segmentations, and bounding boxes. Provides a scrubber and play/pause.
    """

    frame_changed = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.frames              = None   # (N, H, W) uint8 grayscale — raw
        self._preprocessed_frames = None  # (N, H, W) uint8 — after preprocessing
        self._detections         = None   # raw detections DataFrame (no 'sperm' col)
        self.tracks              = None   # full tracks DataFrame (has 'sperm' col)
        self.colors              = None   # (max_sperm_id+1, 3) uint8 BGR

        self._current_frame = 0
        self._playing       = False
        self._show_preprocessed = False   # toggle: raw vs preprocessed frames

        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._advance_frame)

        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Frame display
        self._display = QLabel("Load a video to begin.")
        self._display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._display.setMinimumSize(640, 480)
        self._display.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._display.setStyleSheet("background: #111;")
        layout.addWidget(self._display)

        # Scrubber
        scrubber_row = QHBoxLayout()
        self._frame_label = QLabel("Frame: 0 / 0")
        self._frame_label.setFixedWidth(110)
        scrubber_row.addWidget(self._frame_label)

        self._scrubber = QSlider(Qt.Orientation.Horizontal)
        self._scrubber.setMinimum(0)
        self._scrubber.setMaximum(0)
        self._scrubber.setValue(0)
        self._scrubber.valueChanged.connect(self._on_scrubber_moved)
        scrubber_row.addWidget(self._scrubber)
        layout.addLayout(scrubber_row)

        # Playback controls + overlay toggles
        controls_row = QHBoxLayout()

        self._btn_prev = QPushButton("◀")
        self._btn_prev.setFixedWidth(36)
        self._btn_prev.clicked.connect(self._step_back)

        self._btn_play = QPushButton("▶")
        self._btn_play.setFixedWidth(36)
        self._btn_play.clicked.connect(self._toggle_play)

        self._btn_next = QPushButton("▶|")
        self._btn_next.setFixedWidth(36)
        self._btn_next.clicked.connect(self._step_forward)

        self._fps_box = QSpinBox()
        self._fps_box.setRange(1, 120)
        self._fps_box.setValue(9)
        self._fps_box.setPrefix("FPS: ")
        self._fps_box.setFixedWidth(80)
        self._fps_box.valueChanged.connect(self._update_timer_interval)

        # Toggle between raw and preprocessed frames
        self._btn_frame_src = QPushButton("Showing: Raw")
        self._btn_frame_src.setCheckable(True)
        self._btn_frame_src.setEnabled(False)
        self._btn_frame_src.clicked.connect(self._toggle_frame_source)

        controls_row.addWidget(self._btn_prev)
        controls_row.addWidget(self._btn_play)
        controls_row.addWidget(self._btn_next)
        controls_row.addWidget(self._fps_box)
        controls_row.addWidget(self._btn_frame_src)
        controls_row.addStretch()

        # Overlay group
        overlay_group = QGroupBox("Overlays")
        overlay_layout = QHBoxLayout(overlay_group)
        overlay_layout.setContentsMargins(6, 2, 6, 2)

        self._chk_detections    = QCheckBox("Detections")
        self._chk_tracks        = QCheckBox("Tracks")
        self._chk_segmentations = QCheckBox("Segmentations")
        self._chk_bboxes        = QCheckBox("Bounding Boxes")

        self._chk_detections.setChecked(True)
        self._chk_tracks.setChecked(True)
        self._chk_segmentations.setChecked(False)
        self._chk_bboxes.setChecked(False)

        self._trail_box = QSpinBox()
        self._trail_box.setRange(1, 200)
        self._trail_box.setValue(15)
        self._trail_box.setPrefix("Trail: ")
        self._trail_box.setFixedWidth(80)

        for w in (self._chk_detections, self._chk_tracks,
                  self._chk_segmentations, self._chk_bboxes, self._trail_box):
            w.toggled.connect(self._refresh_display) if hasattr(w, 'toggled') else \
            w.valueChanged.connect(self._refresh_display)
            overlay_layout.addWidget(w)

        controls_row.addWidget(overlay_group)
        layout.addLayout(controls_row)

    # ── public API ────────────────────────────────────────────────────────────

    def set_frames(self, frames):
        """Load a numpy array of grayscale frames (N, H, W)."""
        self.frames = frames
        self.tracks = None
        self.colors = None
        self._current_frame = 0
        self._scrubber.setMaximum(len(frames) - 1)
        self._scrubber.setValue(0)
        self._display_frame(0)

    def set_preprocessed_frames(self, frames):
        """Store preprocessed frames and enable the raw/preprocessed toggle."""
        self._preprocessed_frames = frames
        self._btn_frame_src.setEnabled(True)
        self._refresh_display()

    def set_detections_only(self, detections):
        """
        Show raw detections (no tracking yet) as dots on the current frames.
        Clears any existing track overlay.
        """
        self._detections = detections
        self.tracks      = None
        self.colors      = None
        self._refresh_display()

    def set_tracks(self, tracks):
        """Provide a finished tracks DataFrame. Regenerates colors."""
        self._detections = None   # detections are now superseded by tracks
        self.tracks = tracks
        max_id = int(tracks["sperm"].max())
        self.colors = utils.generateRandomColors(max_id + 1)
        self._refresh_display()

    def set_playback_fps(self, fps):
        self._fps_box.setValue(fps)

    @property
    def current_frame(self):
        return self._current_frame

    # ── overlay compositing ───────────────────────────────────────────────────

    def _compose(self, frame_idx):
        """Return a BGR (H, W, 3) uint8 array for the given frame with overlays."""
        src = (self._preprocessed_frames
               if self._show_preprocessed and self._preprocessed_frames is not None
               else self.frames)
        frame = src[frame_idx].copy()
        if len(frame.shape) == 2:
            frame = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)

        # ── detection-only mode (no tracking yet) ─────────────────────────────
        if self._detections is not None and self.tracks is None:
            current = self._detections[self._detections["frame"] == frame_idx]
            for _, row in current.iterrows():
                x, y = int(row["x"]), int(row["y"])
                cv.circle(frame, (x, y), 3, (0, 255, 255), -1)   # cyan dots
            return frame

        if self.tracks is None or self.colors is None:
            return frame

        t      = self.tracks
        colors = self.colors

        # ── segmentation fill (draw first so other overlays sit on top) ──────
        if self._chk_segmentations.isChecked() and "segmentation" in t.columns:
            current = t[t["frame"] == frame_idx]
            for _, row in current.iterrows():
                seg = row["segmentation"]
                if seg is None or (hasattr(seg, "__len__") and len(seg) == 0):
                    continue
                seg = np.array(seg, dtype=int)
                if seg.ndim != 2 or seg.shape[1] != 2:
                    continue
                sperm_id = int(row["sperm"])
                color = colors[sperm_id].tolist()
                frame[seg[:, 0], seg[:, 1]] = color

        # ── bounding boxes ────────────────────────────────────────────────────
        if self._chk_bboxes.isChecked() and "bbox_x" in t.columns:
            current = t[t["frame"] == frame_idx]
            for _, row in current.iterrows():
                bx = int(row.get("bbox_x", -1))
                if bx < 0:
                    continue
                by, bw, bh = int(row["bbox_y"]), int(row["bbox_w"]), int(row["bbox_h"])
                sperm_id = int(row["sperm"])
                color = colors[sperm_id].tolist()
                cv.rectangle(frame, (bx, by), (bx + bw, by + bh), color, 1)

        # ── track trails ──────────────────────────────────────────────────────
        if self._chk_tracks.isChecked():
            trail = self._trail_box.value()
            sperm_ids = t["sperm"].unique()
            for sperm_id in sperm_ids:
                seg_data = t[
                    (t["sperm"] == sperm_id) &
                    (t["frame"] >= frame_idx - trail) &
                    (t["frame"] <= frame_idx)
                ].sort_values("frame")
                if len(seg_data) < 2:
                    continue
                pts   = seg_data[["x", "y"]].values.astype(int)
                color = colors[int(sperm_id)].tolist()
                n     = len(pts)
                for i in range(1, n):
                    alpha = i / n                         # fade older segments
                    c     = [int(ch * alpha) for ch in color]
                    cv.line(frame, tuple(pts[i - 1]), tuple(pts[i]), c, 1)

        # ── centroid dots ─────────────────────────────────────────────────────
        if self._chk_detections.isChecked():
            current = t[t["frame"] == frame_idx]
            for _, row in current.iterrows():
                x, y     = int(row["x"]), int(row["y"])
                sperm_id = int(row["sperm"])
                color    = colors[sperm_id].tolist()
                cv.circle(frame, (x, y), 3, color, -1)

        return frame

    # ── display helpers ───────────────────────────────────────────────────────

    def _display_frame(self, frame_idx):
        if self.frames is None:
            return
        frame_idx = max(0, min(frame_idx, len(self.frames) - 1))
        self._current_frame = frame_idx

        bgr = self._compose(frame_idx)
        rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        qimg    = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        pixmap  = QPixmap.fromImage(qimg)
        scaled  = pixmap.scaled(
            self._display.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._display.setPixmap(scaled)
        total = len(self.frames) if self.frames is not None else 0
        self._frame_label.setText(f"Frame: {frame_idx} / {max(0, total - 1)}")
        self.frame_changed.emit(frame_idx)

    def _refresh_display(self, *_):
        self._display_frame(self._current_frame)

    # ── scrubber / playback ───────────────────────────────────────────────────

    def _toggle_frame_source(self, checked):
        self._show_preprocessed = checked
        self._btn_frame_src.setText(
            "Showing: Preprocessed" if checked else "Showing: Raw"
        )
        self._refresh_display()

    def _on_scrubber_moved(self, value):
        self._display_frame(value)

    def _toggle_play(self):
        if self._playing:
            self._play_timer.stop()
            self._btn_play.setText("▶")
            self._playing = False
        else:
            self._update_timer_interval()
            self._play_timer.start()
            self._btn_play.setText("⏸")
            self._playing = True

    def _advance_frame(self):
        if self.frames is None:
            return
        nxt = self._current_frame + 1
        if nxt >= len(self.frames):
            nxt = 0
        self._scrubber.setValue(nxt)   # triggers _on_scrubber_moved

    def _step_forward(self):
        if self.frames is None:
            return
        self._scrubber.setValue(min(self._current_frame + 1, len(self.frames) - 1))

    def _step_back(self):
        self._scrubber.setValue(max(self._current_frame - 1, 0))

    def _update_timer_interval(self, *_):
        self._play_timer.setInterval(int(1000 / self._fps_box.value()))

    # ── resize event — redraw at new size ─────────────────────────────────────
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_display()
