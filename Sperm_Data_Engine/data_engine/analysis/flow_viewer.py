"""
flow_viewer.py -- render tracked sperm over a video as an annotated mp4, in
the spirit of UnifiedFramework/visualizer.py's "flow" visualization (colored
per-sperm motion trails), reworked for the hub:

  - Headless: writes an mp4 instead of blocking on cv.imshow/waitKey, which
    can't run inside a Qt app's worker thread. The hub plays the result
    back in its own player (gui/video_player.py).
  - Works off the canonical frame,sperm,x,y trajectory format directly, so
    it can visualize a model's tracker output, a trackpy run, or a ground-
    truth CSV interchangeably. The old version's segmentation/VAP-column
    modes are dropped -- they depended on columns canonical CSVs don't have.
  - Trails fade (trail_frames) instead of accumulating forever (the old mask
    only ever grew, which turns to a solid smear on a long clip); pass
    trail_frames=0 for the old persistent behavior.
  - Everything stays BGR end to end (the old code mixed RGB/BGR conversions).

Modes:
  flow    colored motion trails + a dot at each current position (old "flow")
  ids     a dot and its track id at each current position (good for
          eyeballing ID switches)

Run as a module from the repo root:

    python -m data_engine.analysis.flow_viewer --video v.mp4 --csv tracks.csv --mode flow
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

MODES = ("flow", "ids")


def _color_for_id(sperm_id: int):
    """Deterministic, well-separated BGR color per track id."""
    hue = int((sperm_id * 47) % 180)
    bgr = cv2.cvtColor(np.uint8([[[hue, 230, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def render_overlay(video_path, tracks: pd.DataFrame, out_path, mode: str = "flow",
                   trail_frames: int = 30, progress_cb=None) -> Path:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    missing = {"frame", "sperm", "x", "y"} - set(tracks.columns)
    if missing:
        raise ValueError(f"tracks is missing columns {missing}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 9.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    by_frame = {int(f): g for f, g in tracks.groupby("frame")}
    decay = 1.0 - 1.0 / trail_frames if trail_frames > 0 else 1.0  # 0 -> persistent
    mask = np.zeros((height, width, 3), dtype=np.uint8)
    prev_pos = {}

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if decay < 1.0:
            mask = (mask * decay).astype(np.uint8)

        current = by_frame.get(frame_idx)
        cur_pos = {}
        if current is not None:
            for sid, x, y in zip(current["sperm"].to_numpy(), current["x"].to_numpy(), current["y"].to_numpy()):
                sid, pt = int(sid), (int(round(x)), int(round(y)))
                cur_pos[sid] = pt
                color = _color_for_id(sid)
                if mode == "flow" and sid in prev_pos:
                    cv2.line(mask, prev_pos[sid], pt, color, 2)
                if mode == "ids":
                    cv2.circle(frame, pt, 3, color, -1)
                    cv2.putText(frame, str(sid), (pt[0] + 5, pt[1] - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                0.4, color, 1, cv2.LINE_AA)
                else:
                    cv2.circle(mask, pt, 2, color, -1)
        prev_pos = cur_pos

        img = cv2.add(frame, mask) if mode != "ids" else frame
        cv2.putText(img, f"frame {frame_idx}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(img)

        frame_idx += 1
        if progress_cb and n_frames:
            progress_cb(frame_idx, n_frames)

    cap.release()
    writer.release()
    return out_path


def _cli():
    from data_engine.common.trajectory_io import load_tracks
    parser = argparse.ArgumentParser(description="Render tracked sperm over a video as an annotated mp4")
    parser.add_argument("--video", required=True)
    parser.add_argument("--csv", required=True, help="Canonical frame,sperm,x,y CSV")
    parser.add_argument("--mode", choices=MODES, default="flow")
    parser.add_argument("--trail-frames", type=int, default=30, help="Trail fade length; 0 = persistent")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    out = args.out or str(Path(args.video).with_name(f"{Path(args.video).stem}_{args.mode}.mp4"))
    print("Wrote", render_overlay(args.video, load_tracks(args.csv), out, args.mode, args.trail_frames))


if __name__ == "__main__":
    _cli()
