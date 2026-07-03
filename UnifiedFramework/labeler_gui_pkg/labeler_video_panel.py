import numpy as np
import cv2 as cv

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider,
    QPushButton, QSpinBox, QSizePolicy,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QKeyEvent

import utils


class LabelerVideoPanel(QWidget):
    """
    Dedicated video display for the labeler GUI. Shows track overlays with
    per-sperm trails, supports click-to-select (with row-sync to the sperm
    table), a "new track" placement mode, and both keyboard (j/k/l) and
    button-based playback.
    """

    sperm_clicked      = pyqtSignal(int, bool)    # sperm_id, ctrl_held
    point_placed        = pyqtSignal(int, int, int)  # frame, x, y  (placement mode)
    frame_changed        = pyqtSignal(int)

    _CLICK_RADIUS = 8   # px — matches original labeler.py's 5px half-box (10px box)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.video_original = None   # (N, H, W, 3) BGR uint8
        self.tracks         = None   # DataFrame: frame, sperm, x, y, label, ...
        self.colors         = None   # (max_id+1, 3) BGR uint8

        self._current_frame  = 0
        self._playing         = False
        self._show_original    = False
        self._trail_length     = 60

        self._selected_ids      = []   # sperm IDs currently highlighted (synced with table)
        self._visible_ids       = None  # None = show all tracks; else only these sperm IDs
        self._placement_mode    = False
        self._placement_pts     = []   # list of (frame, x, y) collected during placement

        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._advance_frame)

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self._display = QLabel(
            "Load a video + tracks CSV to begin.\n\n"
            "Keyboard (while video has focus):\n"
            "  k = play/pause   l = next frame   j = previous frame"
        )
        self._display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._display.setMinimumSize(640, 480)
        self._display.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._display.setStyleSheet("background: #111; color: #aaa;")
        self._display.mousePressEvent = self._on_mouse_press
        self._display.setToolTip(
            "Click a cell to select it.\n"
            "Ctrl+Click a second cell to add it to the selection (needed for Merge/Swap)."
        )
        layout.addWidget(self._display)

        scrubber_row = QHBoxLayout()
        self._frame_label = QLabel("Frame: 0 / 0")
        self._frame_label.setFixedWidth(110)
        scrubber_row.addWidget(self._frame_label)

        self._scrubber = QSlider(Qt.Orientation.Horizontal)
        self._scrubber.setMinimum(0)
        self._scrubber.setMaximum(0)
        self._scrubber.valueChanged.connect(self._on_scrubber_moved)
        scrubber_row.addWidget(self._scrubber)
        layout.addLayout(scrubber_row)

        controls_row = QHBoxLayout()

        self._btn_prev = QPushButton("◀")
        self._btn_prev.setFixedWidth(36)
        self._btn_prev.clicked.connect(self.step_back)

        self._btn_play = QPushButton("▶")
        self._btn_play.setFixedWidth(36)
        self._btn_play.clicked.connect(self.toggle_play)

        self._btn_next = QPushButton("▶|")
        self._btn_next.setFixedWidth(36)
        self._btn_next.clicked.connect(self.step_forward)

        self._fps_box = QSpinBox()
        self._fps_box.setRange(1, 120)
        self._fps_box.setValue(9)
        self._fps_box.setPrefix("FPS: ")
        self._fps_box.setFixedWidth(80)
        self._fps_box.valueChanged.connect(self._update_timer_interval)

        self._trail_box = QSpinBox()
        self._trail_box.setRange(1, 300)
        self._trail_box.setValue(self._trail_length)
        self._trail_box.setPrefix("Trail: ")
        self._trail_box.setFixedWidth(90)
        self._trail_box.valueChanged.connect(self._on_trail_changed)

        self._btn_original = QPushButton("Show: Overlay")
        self._btn_original.setCheckable(True)
        self._btn_original.clicked.connect(self._toggle_original)

        controls_row.addWidget(self._btn_prev)
        controls_row.addWidget(self._btn_play)
        controls_row.addWidget(self._btn_next)
        controls_row.addWidget(self._fps_box)
        controls_row.addWidget(self._trail_box)
        controls_row.addWidget(self._btn_original)
        controls_row.addStretch()
        layout.addLayout(controls_row)

        self._placement_label = QLabel("")
        self._placement_label.setStyleSheet("color: #ffcc00; font-weight: bold;")
        layout.addWidget(self._placement_label)

    # ── public API — data ─────────────────────────────────────────────────────

    def set_video(self, frames):
        """frames: (N, H, W, 3) BGR uint8, as returned by utils.loadVideo()."""
        self.video_original = frames
        self._current_frame = 0
        self._scrubber.setMaximum(len(frames) - 1)
        self._scrubber.setValue(0)
        self._render()

    def set_tracks(self, tracks, colors=None):
        self.tracks = tracks
        if colors is not None:
            self.colors = colors
        elif self.colors is None or len(self.colors) <= int(tracks["sperm"].max()):
            self.randomize_colors()
        self._render()

    def randomize_colors(self):
        if self.tracks is None:
            return
        max_id = int(self.tracks["sperm"].max())
        self.colors = utils.generateRandomColors(2 * max_id + 2)
        self._render()

    def set_selected_ids(self, ids):
        """Called externally (e.g. by the sperm table) to sync highlight state."""
        self._selected_ids = list(ids)
        self._render()

    def set_visible_ids(self, ids):
        """
        Restrict which tracks are drawn. Pass None to show all tracks again,
        or an iterable of sperm IDs to show only those (all others hidden).
        """
        self._visible_ids = set(int(i) for i in ids) if ids is not None else None
        self._render()

    @property
    def visible_ids(self):
        return self._visible_ids

    @property
    def current_frame(self):
        return self._current_frame

    @property
    def num_frames(self):
        return len(self.video_original) if self.video_original is not None else 0

    # ── playback controls ─────────────────────────────────────────────────────

    def toggle_play(self):
        if self._playing:
            self._play_timer.stop()
            self._btn_play.setText("▶")
            self._playing = False
        else:
            self._update_timer_interval()
            self._play_timer.start()
            self._btn_play.setText("⏸")
            self._playing = True

    def step_forward(self):
        if self.video_original is None:
            return
        self._scrubber.setValue(min(self._current_frame + 1, self.num_frames - 1))

    def step_back(self):
        self._scrubber.setValue(max(self._current_frame - 1, 0))

    def goto_frame(self, frame_idx):
        self._scrubber.setValue(max(0, min(frame_idx, self.num_frames - 1)))

    def _advance_frame(self):
        nxt = self._current_frame + 1
        if nxt >= self.num_frames:
            nxt = 0
        self._scrubber.setValue(nxt)

    def _update_timer_interval(self, *_):
        self._play_timer.setInterval(int(1000 / self._fps_box.value()))

    def _on_trail_changed(self, val):
        self._trail_length = val
        self._render()

    def _toggle_original(self, checked):
        self._show_original = checked
        self._btn_original.setText("Show: Original" if checked else "Show: Overlay")
        self._render()

    def _on_scrubber_moved(self, value):
        self._current_frame = value
        self._render()
        self._frame_label.setText(f"Frame: {value} / {max(0, self.num_frames - 1)}")
        self.frame_changed.emit(value)

    # ── new-track placement mode ──────────────────────────────────────────────

    def start_placement(self):
        self._placement_mode = True
        self._placement_pts  = []
        self._playing = False
        self._play_timer.stop()
        self._btn_play.setText("▶")
        self._placement_label.setText(
            "Placement mode: click the cell's location each frame. "
            "Frame auto-advances after each click. Click 'Finish New Track' when done."
        )
        self._render()

    def cancel_placement(self):
        self._placement_mode = False
        self._placement_pts  = []
        self._placement_label.setText("")
        self._render()

    def finish_placement(self):
        pts = list(self._placement_pts)
        self._placement_mode = False
        self._placement_pts  = []
        self._placement_label.setText("")
        self._render()
        return pts

    @property
    def in_placement_mode(self):
        return self._placement_mode

    # ── mouse handling ────────────────────────────────────────────────────────

    def _on_mouse_press(self, event):
        if self.video_original is None:
            return

        pixmap = self._display.pixmap()
        if pixmap is None or pixmap.isNull():
            return

        # Map widget-space click to frame-space coordinates (pixmap is letterboxed)
        label_w, label_h = self._display.width(), self._display.height()
        pix_w,   pix_h   = pixmap.width(), pixmap.height()
        offset_x = (label_w - pix_w) / 2
        offset_y = (label_h - pix_h) / 2

        click_x = event.position().x() - offset_x
        click_y = event.position().y() - offset_y
        if click_x < 0 or click_y < 0 or click_x >= pix_w or click_y >= pix_h:
            return

        frame_h, frame_w = self.video_original.shape[1:3]
        x = int(click_x * frame_w / pix_w)
        y = int(click_y * frame_h / pix_h)

        if self._placement_mode:
            self._placement_pts.append((self._current_frame, x, y))
            self.point_placed.emit(self._current_frame, x, y)
            self._render()
            self.step_forward()
            return

        # Selection mode: find nearest sperm within click radius on current frame
        if self.tracks is None:
            return
        current = self.tracks[self.tracks["frame"] == self._current_frame]
        best_id, best_dist = None, self._CLICK_RADIUS
        for _, row in current.iterrows():
            dist = ((row["x"] - x) ** 2 + (row["y"] - y) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_id   = int(row["sperm"])

        if best_id is not None:
            ctrl_held = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
            self.sperm_clicked.emit(best_id, ctrl_held)

    # ── keyboard handling ─────────────────────────────────────────────────────

    def keyPressEvent(self, event: QKeyEvent):
        key = event.text().lower()
        if key == "k" or key == " ":
            self.toggle_play()
        elif key == "l":
            self.step_forward()
        elif key == "j":
            self.step_back()
        else:
            super().keyPressEvent(event)

    # ── rendering ─────────────────────────────────────────────────────────────

    def _compose(self):
        frame = self.video_original[self._current_frame].copy()

        if self._show_original or self.tracks is None or self.colors is None:
            return frame

        t = self.tracks
        colors = self.colors

        # Track trails, most recent `trail_length` frames
        sperm_ids = t["sperm"].unique()
        if self._visible_ids is not None:
            sperm_ids = [s for s in sperm_ids if int(s) in self._visible_ids]
        for sperm_id in sperm_ids:
            sid = int(sperm_id)
            seg = t[
                (t["sperm"] == sperm_id) &
                (t["frame"] >= self._current_frame - self._trail_length) &
                (t["frame"] <= self._current_frame)
            ].sort_values("frame")
            if len(seg) < 1:
                continue
            pts   = seg[["x", "y"]].values.astype(int)
            color = colors[sid % len(colors)].tolist()

            for i in range(1, len(pts)):
                alpha = i / len(pts)
                c     = [int(ch * alpha) for ch in color]
                cv.line(frame, tuple(pts[i - 1]), tuple(pts[i]), c, 2)

            # Centroid dot at current frame
            last_frame_pts = seg[seg["frame"] == self._current_frame]
            if len(last_frame_pts) > 0:
                x, y = int(last_frame_pts.iloc[0]["x"]), int(last_frame_pts.iloc[0]["y"])
                cv.circle(frame, (x, y), 3, color, -1)

                if sid in self._selected_ids:
                    order = self._selected_ids.index(sid)
                    ring_color = (0, 255, 255) if order == 0 else (255, 0, 255)  # yellow / magenta (BGR)
                    cv.circle(frame, (x, y), 9, ring_color, 2)
                    cv.putText(frame, str(sid), (x + 12, y - 8),
                              cv.FONT_HERSHEY_SIMPLEX, 0.5, ring_color, 1, cv.LINE_AA)

        # Placement-mode live trail
        if self._placement_mode and self._placement_pts:
            for i, (f, x, y) in enumerate(self._placement_pts):
                cv.circle(frame, (x, y), 3, (0, 200, 255), -1)
                if i > 0:
                    _, px, py = self._placement_pts[i - 1]
                    cv.line(frame, (px, py), (x, y), (0, 200, 255), 1)

        return frame

    def _render(self):
        if self.video_original is None:
            return
        bgr = self._compose()
        rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        qimg   = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(
            self._display.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._display.setPixmap(scaled)
        self._frame_label.setText(f"Frame: {self._current_frame} / {max(0, self.num_frames - 1)}")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._render()
