"""
label.py -- label a video from scratch, then register it.

Thin wrapper around the labeler GUI for a video with no model prediction to
start from: writes an empty frame,sperm,x,y,label stub (the labeler requires
its input CSV to already exist), launches the labeler, and ingests the
result once something was actually saved. For a video a trained model can
already help with, use bootstrap_label.py instead -- it pre-populates the
starting CSV with predictions rather than nothing.

Run as a module from the repo root, e.g.:

    python -m data_engine.stages.label.label --video path/to/new_video.mp4 --split train
"""
import argparse
from pathlib import Path

import pandas as pd

from data_engine.stages.generate.ingest import ingest
from data_engine.stages.label.correction_flow import launch_labeler_and_wait


def label(video_path, split, tier=None, clip_id=None):
    video_path = Path(video_path)
    initial_csv = video_path.parent / f"{video_path.stem}_empty.csv"
    pd.DataFrame(columns=["frame", "sperm", "x", "y", "label"]).to_csv(initial_csv, index=False)

    print("Opening labeler. Save (File > Save, or the toolbar action) before closing to keep your work.")
    corrected_csv = launch_labeler_and_wait(video_path, initial_csv)

    if corrected_csv is None:
        print(f"No saved file found -- nothing was kept, so nothing is being registered.\n"
              f"To pick up where you left off, re-run this command, or open the labeler directly:\n"
              f"  python -m data_engine.stages.label.labeler_gui "
              f"--videofile {video_path} --csvfile {initial_csv}")
        return None

    clip = ingest(video_path, corrected_csv, tier, "real", split, clip_id=clip_id, label_status="human")
    print(f"Registered clip: {clip.clip_id}")
    return clip


def _cli():
    parser = argparse.ArgumentParser(description="Label a video from scratch and register it")
    parser.add_argument("--video", required=True)
    parser.add_argument("--split", required=True, choices=["train", "val"])
    parser.add_argument("--tier", default=None,
                         help="low|medium|high|very_high -- optional, auto-computed from the "
                              "corrected labels' measured density if omitted (recommended)")
    parser.add_argument("--clip-id", default=None)
    args = parser.parse_args()

    label(args.video, args.split, tier=args.tier, clip_id=args.clip_id)


if __name__ == "__main__":
    _cli()
