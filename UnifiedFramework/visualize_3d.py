#!/usr/bin/env python3
"""
3D point cloud visualization of video frames in (x, y, frame) space.

Every bright pixel becomes a 3D point at (x, y, frame * z_scale). Dark pixels
are discarded, causing the volume to appear as a skeleton-like structure where
tracks leave visible trails through time.

Run with no arguments for an interactive launcher GUI.

Usage examples:
    python visualize_3d.py                                          # GUI launcher
    python visualize_3d.py video.mp4
    python visualize_3d.py video.mp4 --preprocessor median_filter --fill_holes
    python visualize_3d.py video.mp4 --csv tracks.csv
    python visualize_3d.py video.mp4 --config configs/10X.json --z_scale 3.0

Navigation in the 3D viewer:
    Left drag    Orbit
    Right drag   Pan
    Scroll       Zoom
    R            Reset view
    Q / Esc      Quit
"""

import argparse
import json
import sys
import numpy as np
import open3d as o3d
import matplotlib.cm as cm
from pathlib import Path
from scipy.spatial.transform import Rotation

import utils
from plugins.preprocessors import PREPROCESSORS, apply_fill_holes


# ── Processing helpers ────────────────────────────────────────────────────────

def load_config(args):
    config = {}
    if args.config:
        with open(args.config) as f:
            config = json.load(f)
    config.update({
        "preprocessor":          args.preprocessor,
        "fill_holes":            args.fill_holes or config.get("fill_holes", False),
        "positive_phase_cutoff": args.positive_phase_cutoff,
        "morph_kernel_size":     [args.morph_kernel_size, args.morph_kernel_size],
        "threshold_method":      args.threshold_method,
        "global_thresh":         args.global_thresh,
    })
    return config


def preprocess(frames, config):
    preprocessor = config.get("preprocessor", "none")
    print(f"  Preprocessor : {preprocessor}"
          + (" + fill_holes" if config.get("fill_holes") else ""))
    out = PREPROCESSORS[preprocessor](frames, config)
    if config.get("fill_holes"):
        out = apply_fill_holes(out, config)
    return out


def guess_z_scale(H, W, N):
    """Scale frame axis so total extent ≈ shorter spatial dimension (cubic bounding box)."""
    return min(H, W) / max(N, 1)


def frames_to_points(frames, threshold, max_points):
    fi, ys, xs = np.where(frames > threshold)
    intensities = frames[fi, ys, xs].astype(np.float32) / 255.0
    coords = np.column_stack([
        xs.astype(np.float64),
        ys.astype(np.float64),
        fi.astype(np.float64),
    ])
    if len(coords) > max_points:
        print(f"  Subsampling {len(coords):,} → {max_points:,} points")
        idx    = np.random.default_rng(0).choice(len(coords), max_points, replace=False)
        coords = coords[idx]
        intensities = intensities[idx]
    return coords, intensities


def color_by_frame(frame_z, n_frames, cmap_name):
    cmap = cm.get_cmap(cmap_name)
    return cmap(frame_z / max(n_frames - 1, 1))[:, :3].astype(np.float64)


def color_by_intensity(intensities, cmap_name):
    cmap = cm.get_cmap(cmap_name)
    return cmap(intensities)[:, :3].astype(np.float64)


# ── Track geometry ────────────────────────────────────────────────────────────

def _rotation_z_to_vec(target):
    """Rotation matrix mapping +Z to unit vector `target`."""
    target = target / np.linalg.norm(target)
    z      = np.array([0.0, 0.0, 1.0])
    axis   = np.cross(z, target)
    norm   = np.linalg.norm(axis)
    if norm < 1e-8:
        return np.eye(3) if np.dot(z, target) > 0 else np.diag([1.0, -1.0, -1.0])
    angle = np.arccos(np.clip(np.dot(z, target), -1.0, 1.0))
    return Rotation.from_rotvec((angle / norm) * axis).as_matrix()


def build_track_tubes(tracks_df, z_scale, radius=2.0, resolution=6):
    """Cylinder-mesh tubes — proper 3D geometry, thickness independent of line-width cap."""
    max_id  = int(tracks_df["sperm"].max())
    palette = utils.generateRandomColors(max_id + 1).astype(np.float64) / 255.0
    combined = o3d.geometry.TriangleMesh()

    for sperm_id, group in tracks_df.groupby("sperm"):
        group = group.sort_values("frame")
        if len(group) < 2:
            continue
        pts   = np.column_stack([
            group["x"].values.astype(np.float64),
            group["y"].values.astype(np.float64),
            group["frame"].values.astype(np.float64) * z_scale,
        ])
        color = palette[int(sperm_id) % len(palette)]
        for i in range(len(pts) - 1):
            p0, p1    = pts[i], pts[i + 1]
            direction = p1 - p0
            length    = np.linalg.norm(direction)
            if length < 1e-6:
                continue
            cyl = o3d.geometry.TriangleMesh.create_cylinder(
                radius=radius, height=length, resolution=resolution, split=1,
            )
            R      = _rotation_z_to_vec(direction / length)
            center = (p0 + p1) / 2.0
            T      = np.eye(4)
            T[:3, :3] = R
            T[:3,  3] = center
            cyl.transform(T)
            cyl.paint_uniform_color(color.tolist())
            combined += cyl

    if len(combined.vertices) > 0:
        combined.compute_vertex_normals()
    return combined


def build_track_lines(tracks_df, z_scale):
    """Thin LineSet fallback (subject to 1 px OpenGL cap on most platforms)."""
    max_id  = int(tracks_df["sperm"].max())
    palette = utils.generateRandomColors(max_id + 1).astype(np.float64) / 255.0
    all_pts, all_lines, all_colors = [], [], []
    offset = 0
    for sperm_id, group in tracks_df.groupby("sperm"):
        group = group.sort_values("frame")
        if len(group) < 2:
            continue
        pts   = np.column_stack([
            group["x"].values.astype(np.float64),
            group["y"].values.astype(np.float64),
            group["frame"].values.astype(np.float64) * z_scale,
        ])
        n     = len(pts)
        lines = [[offset + i, offset + i + 1] for i in range(n - 1)]
        color = palette[int(sperm_id) % len(palette)].tolist()
        all_pts.extend(pts.tolist())
        all_lines.extend(lines)
        all_colors.extend([color] * (n - 1))
        offset += n
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.array(all_pts))
    ls.lines  = o3d.utility.Vector2iVector(np.array(all_lines, dtype=np.int32))
    ls.colors = o3d.utility.Vector3dVector(np.array(all_colors))
    return ls


def build_axes(H, W, N, z_scale):
    pts    = np.array([[0,0,0],[W,0,0],[0,0,0],[0,H,0],[0,0,0],[0,0,N*z_scale]], dtype=np.float64)
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines  = o3d.utility.Vector2iVector([[0,1],[2,3],[4,5]])
    ls.colors = o3d.utility.Vector3dVector([[1,0,0],[0,1,0],[0,0.5,1]])
    return ls


# ── GUI launcher ──────────────────────────────────────────────────────────────

def _make_launcher_dialog():
    """
    Build and return a QDialog that collects all visualisation parameters.
    Imported lazily so PyQt6 is only required when the GUI path is taken.
    """
    from PyQt6.QtWidgets import (
        QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
        QLabel, QLineEdit, QPushButton, QComboBox, QCheckBox,
        QSpinBox, QDoubleSpinBox, QDialogButtonBox, QFileDialog,
        QSizePolicy,
    )
    from PyQt6.QtCore import Qt

    _COLORMAPS = ["plasma", "viridis", "turbo", "inferno", "magma",
                  "hot", "cool", "gray", "jet"]

    class LauncherDialog(QDialog):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("3D Track Volume — Launch")
            self.setMinimumWidth(480)
            self._build_ui()

        def _build_ui(self):
            root = QVBoxLayout(self)
            root.setSpacing(10)

            # ── Files ─────────────────────────────────────────────────────────
            file_group = QGroupBox("Files")
            file_form  = QFormLayout(file_group)

            self._video_edit = QLineEdit()
            self._video_edit.setPlaceholderText("Required")
            vid_row = QHBoxLayout()
            vid_row.addWidget(self._video_edit)
            btn_vid = QPushButton("Browse…")
            btn_vid.clicked.connect(self._browse_video)
            vid_row.addWidget(btn_vid)
            file_form.addRow("Video file:", vid_row)

            self._csv_edit = QLineEdit()
            self._csv_edit.setPlaceholderText("Optional — enables track overlay")
            csv_row = QHBoxLayout()
            csv_row.addWidget(self._csv_edit)
            btn_csv = QPushButton("Browse…")
            btn_csv.clicked.connect(self._browse_csv)
            csv_row.addWidget(btn_csv)
            file_form.addRow("Tracks CSV:", csv_row)

            root.addWidget(file_group)

            # ── Preprocessing ──────────────────────────────────────────────────
            pre_group = QGroupBox("Preprocessing")
            pre_form  = QFormLayout(pre_group)

            self._combo_pre = QComboBox()
            self._combo_pre.addItems(list(PREPROCESSORS.keys()))
            pre_form.addRow("Method:", self._combo_pre)

            self._chk_fill = QCheckBox("Fill holes  (closes phase-contrast donuts)")
            pre_form.addRow(self._chk_fill)

            root.addWidget(pre_group)

            # ── Visualisation ──────────────────────────────────────────────────
            vis_group = QGroupBox("Visualisation")
            vis_form  = QFormLayout(vis_group)

            self._spin_thresh = QSpinBox()
            self._spin_thresh.setRange(0, 254)
            self._spin_thresh.setValue(70)
            self._spin_thresh.setToolTip("Pixels at or below this value are hidden")
            vis_form.addRow("Threshold:", self._spin_thresh)

            # Z scale row: Auto checkbox + spinbox
            z_row = QHBoxLayout()
            self._chk_auto_z = QCheckBox("Auto")
            self._chk_auto_z.setChecked(True)
            self._spin_z = QDoubleSpinBox()
            self._spin_z.setRange(0.01, 9999.0)
            self._spin_z.setValue(1.0)
            self._spin_z.setDecimals(3)
            self._spin_z.setEnabled(False)
            self._spin_z.setToolTip("Pixels per frame on the z axis")
            self._chk_auto_z.toggled.connect(lambda checked: self._spin_z.setEnabled(not checked))
            z_row.addWidget(self._chk_auto_z)
            z_row.addWidget(self._spin_z)
            vis_form.addRow("Z scale:", z_row)

            self._combo_colorby = QComboBox()
            self._combo_colorby.addItems(["frame", "intensity"])
            vis_form.addRow("Color by:", self._combo_colorby)

            self._combo_cmap = QComboBox()
            self._combo_cmap.addItems(_COLORMAPS)
            vis_form.addRow("Colormap:", self._combo_cmap)

            self._spin_tube = QDoubleSpinBox()
            self._spin_tube.setRange(0.0, 50.0)
            self._spin_tube.setValue(3.0)
            self._spin_tube.setDecimals(1)
            self._spin_tube.setToolTip("Track tube radius in pixel units. Set to 0 for thin lines.")
            vis_form.addRow("Tube radius:", self._spin_tube)

            self._spin_pt = QDoubleSpinBox()
            self._spin_pt.setRange(0.5, 20.0)
            self._spin_pt.setValue(1.0)
            self._spin_pt.setDecimals(1)
            vis_form.addRow("Point size:", self._spin_pt)

            self._chk_axes = QCheckBox("Show reference axes")
            self._chk_axes.setChecked(True)
            vis_form.addRow(self._chk_axes)

            root.addWidget(vis_group)

            # ── Buttons ────────────────────────────────────────────────────────
            buttons = QDialogButtonBox(
                QDialogButtonBox.StandardButton.Ok |
                QDialogButtonBox.StandardButton.Cancel
            )
            buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Launch")
            buttons.accepted.connect(self._on_accept)
            buttons.rejected.connect(self.reject)
            root.addWidget(buttons)

        # ── file pickers ───────────────────────────────────────────────────────

        def _browse_video(self):
            path, _ = QFileDialog.getOpenFileName(
                self, "Select video file", "",
                "Video files (*.mp4 *.avi *.mov *.mkv);;All files (*)",
            )
            if path:
                self._video_edit.setText(path)

        def _browse_csv(self):
            path, _ = QFileDialog.getOpenFileName(
                self, "Select tracks CSV", "",
                "CSV files (*.csv);;All files (*)",
            )
            if path:
                self._csv_edit.setText(path)

        # ── validation ────────────────────────────────────────────────────────

        def _on_accept(self):
            if not self._video_edit.text().strip():
                self._video_edit.setPlaceholderText("⚠ Please select a video file")
                self._video_edit.setStyleSheet("border: 1px solid red;")
                return
            self.accept()

        # ── result ────────────────────────────────────────────────────────────

        def get_args(self):
            """Return an argparse.Namespace matching what the CLI parser produces."""
            import argparse as _ap
            csv_path = self._csv_edit.text().strip() or None
            return _ap.Namespace(
                videofile             = self._video_edit.text().strip(),
                csv                   = csv_path,
                config                = None,
                preprocessor          = self._combo_pre.currentText(),
                fill_holes            = self._chk_fill.isChecked(),
                positive_phase_cutoff = 5.0,
                morph_kernel_size     = 3,
                threshold_method      = "hybrid",
                global_thresh         = 50,
                threshold             = self._spin_thresh.value(),
                z_scale               = None if self._chk_auto_z.isChecked()
                                             else self._spin_z.value(),
                colorby               = self._combo_colorby.currentText(),
                colormap              = self._combo_cmap.currentText(),
                show_tracks           = csv_path is not None,
                tube_radius           = self._spin_tube.value(),
                no_axes               = not self._chk_axes.isChecked(),
                point_size            = self._spin_pt.value(),
                max_points            = 2_000_000,
            )

    return LauncherDialog()


# ── Core visualisation logic ──────────────────────────────────────────────────

def run(args):
    # ── Load video ────────────────────────────────────────────────────────────
    print(f"\nLoading video : {args.videofile}")
    frames = utils.loadVideo(args.videofile, as_gray=True)
    N, H, W = frames.shape
    print(f"  Dimensions  : {W} × {H} px, {N} frames")

    # ── Preprocessing ─────────────────────────────────────────────────────────
    config = load_config(args)
    if config.get("preprocessor", "none") != "none" or config.get("fill_holes"):
        print("Preprocessing…")
        frames = preprocess(frames, config)
    else:
        print("Preprocessing : none")

    # ── Z scale ───────────────────────────────────────────────────────────────
    if args.z_scale is not None:
        z_scale = args.z_scale
        print(f"Z scale       : {z_scale:.4f}  (user)")
    else:
        z_scale = guess_z_scale(H, W, N)
        print(f"Z scale       : {z_scale:.4f}  (auto — "
              f"frame extent {N * z_scale:.0f} px ≈ shorter dim {min(H, W)} px)")
        print( "                Override with --z_scale")

    # ── Point cloud ───────────────────────────────────────────────────────────
    print(f"\nBuilding point cloud (threshold = {args.threshold})…")
    coords, intensities = frames_to_points(frames, args.threshold, args.max_points)

    if len(coords) == 0:
        print("No points above threshold — try lowering --threshold.")
        sys.exit(1)

    print(f"  Points      : {len(coords):,}")
    coords[:, 2] *= z_scale

    if args.colorby == "frame":
        colors = color_by_frame(coords[:, 2] / z_scale, N, args.colormap)
    else:
        colors = color_by_intensity(intensities, args.colormap)

    # ── Assemble geometries ───────────────────────────────────────────────────
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    geometries = [pcd]

    if args.show_tracks:
        print(f"Loading tracks: {args.csv}")
        tracks_df = utils.loadDataFrame(args.csv)
        n_tracks  = tracks_df["sperm"].nunique()
        print(f"  Tracks      : {n_tracks}")
        if args.tube_radius > 0:
            print(f"  Track style : tubes  (radius = {args.tube_radius} px)")
            geometries.append(build_track_tubes(tracks_df, z_scale, radius=args.tube_radius))
        else:
            print(f"  Track style : lines  (thin — OpenGL 1 px cap applies)")
            geometries.append(build_track_lines(tracks_df, z_scale))

    if not args.no_axes:
        geometries.append(build_axes(H, W, N, z_scale))

    # ── Open3D viewer ─────────────────────────────────────────────────────────
    stem = Path(args.videofile).stem
    print(f"\nOpening viewer…  (Q or Esc to quit, R to reset view)\n")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"3D Track Volume — {stem}", width=1280, height=800)
    for g in geometries:
        vis.add_geometry(g)
    opt = vis.get_render_option()
    opt.background_color = np.array([0.05, 0.05, 0.05])
    opt.point_size       = args.point_size
    vis.run()
    vis.destroy_window()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) == 1:
        # No arguments — show the GUI launcher
        from PyQt6.QtWidgets import QApplication
        app = QApplication(sys.argv)
        app.setStyle("Fusion")
        dialog = _make_launcher_dialog()
        if dialog.exec() != dialog.DialogCode.Accepted:
            sys.exit(0)
        args = dialog.get_args()
        # Let Qt clean up before Open3D creates its own window
        app.quit()
        del app, dialog
        run(args)
        return

    # ── CLI path ──────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("videofile", help="Path to video file")
    parser.add_argument("--csv",    default=None, help="Tracks CSV")
    parser.add_argument("--config", default=None, help="JSON config file")

    pre = parser.add_argument_group("preprocessing")
    pre.add_argument("--preprocessor", default="none", choices=list(PREPROCESSORS.keys()))
    pre.add_argument("--fill_holes",   action="store_true")
    pre.add_argument("--positive_phase_cutoff", type=float, default=5.0,   metavar="N")
    pre.add_argument("--morph_kernel_size",      type=int,   default=3,     metavar="N")
    pre.add_argument("--threshold_method", default="hybrid",
                     choices=["global","median","otsu","adaptive","hybrid"])
    pre.add_argument("--global_thresh", type=int, default=50, metavar="N")

    parser.add_argument("--threshold",  type=int,   default=70,       metavar="N")
    parser.add_argument("--z_scale",    type=float, default=None)

    vis = parser.add_argument_group("visualisation")
    vis.add_argument("--colorby",     default="frame",   choices=["frame","intensity"])
    vis.add_argument("--colormap",    default="plasma")
    vis.add_argument("--show_tracks", action="store_true")
    vis.add_argument("--tube_radius", type=float, default=3.0)
    vis.add_argument("--no_axes",     action="store_true")
    vis.add_argument("--point_size",  type=float, default=1.0)
    vis.add_argument("--max_points",  type=int,   default=2_000_000)

    args = parser.parse_args()

    if args.csv is not None:
        args.show_tracks = True
    if args.show_tracks and args.csv is None:
        parser.error("--show_tracks requires --csv")

    run(args)


if __name__ == "__main__":
    main()
