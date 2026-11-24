"""
augment.py -- combine multiple registry clips into one denser clip.

Rebuilds the algorithm recovered from UnifiedFramework/dense_from_sparse.ipynb
(previously undocumented -- located and copied in by the user during Phase 3
planning; see data_engine/README.md). For N clips, truncated to the shortest
clip's frame count, all sharing the same square frame dimensions:

  1. Load each as grayscale frames; combine via elementwise max across all N
     (bright-sperm-on-dark-background compositing) -> one denser video.
  2. Load + clean each clip's ground-truth CSV; offset sperm IDs to stay
     unique across clips; concat -> one combined trajectory CSV.
  3. If --rotate/--mirror (both default on, the notebook's final recipe):
     also build rot90 and/or mirror copies of EACH clip's frames and
     matching coordinate-transformed trajectories (rotate90_df/mirror_h_df),
     then combine ALL variants -- not just the N originals -- the same way,
     multiplying effective density further rather than just merging.
     --rotate alone adds a rot90 variant; --mirror alone adds a mirror-only
     (not rotated) variant; both together add both of those PLUS the
     combined mirror+rot90 variant (the original notebook only ever built
     that combined form, never a plain un-rotated mirror -- independent
     flags are a data-engine addition for finer control).

The notebook hardcoded exactly 3 source videos and a 1024px canvas center
(511.5 = (1024-1)/2); this generalizes to any N and to each clip's actual
width/height.

Run as a module from the repo root, e.g.:

    python -m data_engine.stages.generate.augment --clips 10_1 10_2 --no-mirror
"""
import argparse
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from data_engine.common.trajectory_io import load_tracks
from data_engine.registry.registry import REPO_ROOT, Registry

DEFAULT_OUT_ROOT = Path(__file__).resolve().parents[2] / "data"


def _load_gray_frames(video_path) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.release()
    return np.array(frames)


def _save_gray_frames(frames: np.ndarray, path, fps: float) -> None:
    """frames: (T, H, W) grayscale -> 3-channel mp4 (matches UnifiedFramework/utils.py's saveVideo)."""
    frames3 = np.repeat(frames[..., np.newaxis], 3, axis=-1)
    height, width = frames.shape[1], frames.shape[2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    for f in frames3:
        out.write(f)
    out.release()


def rotate90_df(df: pd.DataFrame, center_x: float, center_y: float) -> pd.DataFrame:
    df = df.copy()
    df["x"], df["y"] = -(df["y"] - center_y) + center_x, (df["x"] - center_x) + center_y
    return df


def mirror_h_df(df: pd.DataFrame, center_x: float) -> pd.DataFrame:
    df = df.copy()
    df["x"] = 2 * center_x - df["x"]
    return df


def combine_sperm_dfs(dfs) -> pd.DataFrame:
    """Offset each dataframe's sperm IDs past the running max so they stay
    unique, then concat. Reproduces dense_from_sparse.ipynb's combineSpermDF
    exactly: offsets are computed from each df's PRE-offset max, in order."""
    dfs_c = [df.copy() for df in dfs if not df.empty]
    max_sperm_id = [int(df["sperm"].max()) for df in dfs_c]
    for i, df in enumerate(dfs_c):
        df["sperm"] = df["sperm"] + sum(max_sperm_id[:i]) + i + 1
    return pd.concat(dfs_c, ignore_index=True)


def augment(clip_ids, rotate: bool = True, mirror: bool = True, out_dir=None, fps: float = 9.0):
    if len(clip_ids) < 2:
        raise ValueError("Need at least 2 clip_ids to combine")

    registry = Registry.load()
    clips = []
    for cid in clip_ids:
        clip = registry.get(cid)
        if clip is None:
            raise ValueError(f"clip_id {cid!r} not found in registry")
        if not clip.source_video or not clip.ground_truth_csv:
            raise ValueError(f"clip_id {cid!r} has no source_video/ground_truth_csv registered")
        clips.append(clip)

    frames_list = [_load_gray_frames(REPO_ROOT / c.source_video) for c in clips]
    dfs = [load_tracks(REPO_ROOT / c.ground_truth_csv) for c in clips]

    shapes = {f.shape[1:] for f in frames_list}
    if len(shapes) > 1:
        detail = ", ".join(f"{cid}={f.shape[1:]}" for cid, f in zip(clip_ids, frames_list))
        raise ValueError(f"Clips must share the same frame dimensions, got: {detail}")
    height, width = shapes.pop()
    if width != height:
        raise ValueError(f"augment.py requires square frames (rot90 changes non-square "
                          f"dimensions); got {width}x{height}")

    min_frames = min(f.shape[0] for f in frames_list)
    frames_list = [f[:min_frames] for f in frames_list]
    dfs = [df[df["frame"] < min_frames] for df in dfs]

    center_x = (width - 1) / 2
    center_y = (height - 1) / 2

    frame_variants = []
    df_variants = []
    for frames, df in zip(frames_list, dfs):
        frame_variants.append(frames)
        df_variants.append(df)
        if rotate:
            rot = np.rot90(frames, axes=(2, 1))
            frame_variants.append(rot)
            df_variants.append(rotate90_df(df, center_x, center_y))
        if mirror:
            mirror_frames = np.flip(frames, axis=2)
            frame_variants.append(mirror_frames)
            df_variants.append(mirror_h_df(df, center_x))
        if rotate and mirror:
            mirror_rot = np.rot90(np.flip(frames, axis=2), axes=(2, 1))
            frame_variants.append(mirror_rot)
            df_variants.append(rotate90_df(mirror_h_df(df, center_x), center_x, center_y))

    combined_frames = np.stack(frame_variants, axis=0).max(axis=0)
    combined_df = combine_sperm_dfs(df_variants)

    if out_dir is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_dir = DEFAULT_OUT_ROOT / f"aug_{'_'.join(clip_ids)}_{stamp}"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    video_path = out_dir / "video.mp4"
    csv_path = out_dir / "trajectories_trackpy_format.csv"
    _save_gray_frames(combined_frames, video_path, fps=fps)
    combined_df.to_csv(csv_path, index=False)

    applied = ", ".join(name for name, on in [("rotate", rotate), ("mirror", mirror)] if on) or "no transforms"
    print(f"Combined {len(clip_ids)} clips ({applied}): "
          f"{min_frames} frames, {combined_df['sperm'].nunique()} unique sperm IDs")
    print(f"Video: {video_path}")
    print(f"Canonical CSV: {csv_path}")

    return video_path, csv_path


def _cli():
    parser = argparse.ArgumentParser(description="Combine multiple registry clips into one denser clip")
    parser.add_argument("--clips", nargs="+", required=True, help="2+ registry clip_ids to combine")
    parser.add_argument("--rotate", dest="rotate", action="store_true", default=True,
                         help="Also add a rot90 copy of each clip (default: on)")
    parser.add_argument("--no-rotate", dest="rotate", action="store_false")
    parser.add_argument("--mirror", dest="mirror", action="store_true", default=True,
                         help="Also add a mirrored copy of each clip (default: on)")
    parser.add_argument("--no-mirror", dest="mirror", action="store_false")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--fps", type=float, default=9.0)
    args = parser.parse_args()

    augment(args.clips, rotate=args.rotate, mirror=args.mirror, out_dir=args.out_dir, fps=args.fps)


if __name__ == "__main__":
    _cli()
