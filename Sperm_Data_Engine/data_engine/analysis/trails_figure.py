"""
trails_figure.py -- a 3-panel "original video / partial trails / full trails"
comparison figure, for the paper: frame 0 alone (density at a glance), the
accumulated sperm trails up to a chosen mid-point frame overlaid on the
video frame at that point, and the full-video accumulated trails overlaid on
the final frame -- side by side, so both the video's density and the trails'
growing complexity are visible in one figure.

Trails are drawn over the ACTUAL video frame at each panel's end-point (not
over frame 0), matching analysis/flow_viewer.py's "flow" overlay convention
(current frame + accumulated motion), for visual consistency with the hub's
Visualize sub-tab.

Run as a module from the repo root, e.g.:

    python -m data_engine.analysis.trails_figure \\
        --video "data/pending_label/high/10X Ph- 9Fps Wash P017me R2.mp4" \\
        --csv "data/pending_label/high/10X Ph- 9Fps Wash P017me R2_bootstrap_corrected.csv" \\
        --mid-frame 17 --mid-label "After 2 Seconds" \\
        --out data_engine/data/visualizations/P017me_trails_figure.png
"""
import argparse
import colorsys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt

from data_engine.common.trajectory_io import load_tracks

REPO_ROOT = Path(__file__).resolve().parents[2]


def _grayscale_frame_at(video_path, index):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"Could not read frame {index} from {video_path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _color_for_id(sperm_id):
    """Deterministic, well-separated RGB color per track id -- same hue
    spacing as flow_viewer.py's _color_for_id, so a sperm's color is
    consistent between this figure and the hub's Visualize overlays."""
    hue = (int(sperm_id) * 47) % 180 / 180.0
    return colorsys.hsv_to_rgb(hue, 0.9, 1.0)


def _panel(ax, frame, tracks, up_to_frame, label, linewidth, alpha):
    ax.imshow(frame, cmap="gray", vmin=0, vmax=255)
    if up_to_frame is not None and tracks is not None:
        sub = tracks[tracks["frame"] <= up_to_frame]
        for sperm_id, g in sub.groupby("sperm"):
            g = g.sort_values("frame")
            if len(g) < 2:
                continue
            ax.plot(g["x"], g["y"], color=_color_for_id(sperm_id), linewidth=linewidth, alpha=alpha)
    ax.set_xlim(0, frame.shape[1])
    ax.set_ylim(frame.shape[0], 0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel(label, fontsize=13)


def render_trails_figure(
    video_path, csv_path, mid_frame, out_path,
    original_label="Original Video", mid_label="After 2 Seconds", full_label="Full Video",
    full_frame=None, linewidth=0.5, alpha=0.6, dpi=200,
):
    tracks = load_tracks(csv_path).dropna(subset=["x", "y"])
    last_frame = full_frame if full_frame is not None else int(tracks["frame"].max())

    frame0 = _grayscale_frame_at(video_path, 0)
    frame_mid = _grayscale_frame_at(video_path, mid_frame)
    frame_last = _grayscale_frame_at(video_path, last_frame)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    _panel(axes[0], frame0, None, None, original_label, linewidth, alpha)
    _panel(axes[1], frame_mid, tracks, mid_frame, mid_label, linewidth, alpha)
    _panel(axes[2], frame_last, tracks, last_frame, full_label, linewidth, alpha)

    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.08, wspace=0.02)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def _cli():
    parser = argparse.ArgumentParser(description="Render a 3-panel original/partial-trails/full-trails figure")
    parser.add_argument("--video", required=True)
    parser.add_argument("--csv", required=True, help="Canonical frame,sperm,x,y tracks CSV")
    parser.add_argument("--mid-frame", type=int, required=True)
    parser.add_argument("--full-frame", type=int, default=None, help="Default: the tracks' last frame")
    parser.add_argument("--original-label", default="Original Video")
    parser.add_argument("--mid-label", default="After 2 Seconds")
    parser.add_argument("--full-label", default="Full Video")
    parser.add_argument("--linewidth", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out = render_trails_figure(
        args.video, args.csv, args.mid_frame, args.out,
        original_label=args.original_label, mid_label=args.mid_label, full_label=args.full_label,
        full_frame=args.full_frame, linewidth=args.linewidth, alpha=args.alpha, dpi=args.dpi,
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    _cli()
