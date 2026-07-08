"""
ultrack_tracker.py

Tracks sperm cells using ultrack's Integer Linear Programming (ILP) tracker
as a drop-in comparison to the TrackPy / forecaster pipeline in tracker.py.

Produces a CSV in the same format (y, x, frame, sperm) so it feeds directly
into metrics.py without any conversion step.

Pipeline:
    Video frames (grayscale)
        ├── threshold('hybrid')          →  binary foreground (T, H, W)
        │       └── morphological gradient  →  cell-edge image  (T, H, W)
        └── ultrack Tracker(config).track(foreground, edges)
                └── to_tracks_layer()  →  rename columns  →  CSV

Usage:
    python ultrack_tracker.py --videofile path/to/video.mp4
    python ultrack_tracker.py --videofile path/to/video.mp4 --configfile configs/10X.json
    python ultrack_tracker.py --videofile path/to/video.mp4 --output my_tracks.csv

    # Then compare against ground truth with the normal metrics pipeline:
    python metrics.py --groundtruth gt.csv --prediction my_tracks.csv --crossover

IMPORTANT: ultrack uses multiprocessing internally. Always run this script
via the  `if __name__ == '__main__':` guard (already in place below).

Installation (in your conda env):
    pip install ultrack
    pip install highspy        # optional but strongly recommended — much faster solver
"""

import argparse
import json
import sys
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd
from tqdm import trange

import utils


# ---------------------------------------------------------------------------
# Thresholding — copied from tracker.py so this file stays self-contained
# and doesn't pull the trackpy dependency chain.
# Keep in sync with tracker.threshold() if that function changes.
# ---------------------------------------------------------------------------
def threshold(frame, method="hybrid", global_thresh=50):
    """Binary threshold a grayscale frame. See tracker.py for full docs."""
    if len(frame.shape) == 3:
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    if method == "global":
        _, bw = cv.threshold(frame, global_thresh, 255, cv.THRESH_BINARY)
    elif method == "median":
        thresh_val = np.median(frame) + 20
        _, bw = cv.threshold(frame, thresh_val, 255, cv.THRESH_BINARY)
    elif method == "otsu":
        _, bw = cv.threshold(frame, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    elif method == "adaptive":
        bw = cv.adaptiveThreshold(
            frame, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, -2
        )
    elif method == "hybrid":
        _, bw1 = cv.threshold(frame, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        bw2 = cv.adaptiveThreshold(
            frame, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, -2
        )
        bw = cv.bitwise_or(bw1, bw2)
    else:
        raise ValueError(f"Invalid thresholding method: {method}")

    return bw


# ---------------------------------------------------------------------------
# Default config — mirrors the key ultrack knobs for sperm tracking.
# Any of these can be overridden by values in the --configfile JSON.
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = {
    # Thresholding / preprocessing
    "threshold_method": "hybrid",   # same methods as tracker.py threshold()
    "morph_open_size": 3,           # kernel size for opening noise before edge detection

    # Segmentation candidates — tune to your sperm head pixel size.
    # min_area is applied both as a pre-filter (removes tiny noise blobs before
    # ultrack sees them) and as ultrack's own segmentation_config.min_area.
    # 7 is the practical minimum: ultrack's hierarchy builder fails on regions
    # smaller than ~6 px, and sperm blobs after morphological opening land
    # mostly in the 7-55 px range at 10X magnification.
    "min_area": 7,                  # minimum blob area in pixels
    "max_area": 500,                # maximum blob area in pixels

    # Linking — ultrack uses IoU (blob overlap) as the base link weight.
    # For moving sperm the blobs barely overlap between frames, so IoU ~ 0
    # for every candidate link.  Without distance weighting all candidates
    # look equally good and the ILP picks randomly — producing jumps.
    #
    # search_range: max inter-frame displacement considered.  Set higher than
    # TrackPy's 21 px to cover fast-moving sperm (override per-video if needed
    # via ultrack_search_range in the JSON config).
    "search_range": 35,
    # distance_weight: subtracted from IoU before the link score is stored.
    # Effective link score = (IoU - distance_weight * dist).  With power=1 this
    # keeps the sign: closer matches score less negative, so the ILP prefers them.
    "distance_weight": 0.5,

    # Tracking weights (all negative; larger magnitude = stronger penalty)
    "division_weight":  -1000.0,    # effectively disables division (sperm don't divide)
    # appear/disappear calibration:
    #   With distance_weight=0.5 and power=1 the worst valid link (at 35 px)
    #   scores ~-17.5.  Setting drop penalty to -50 each (-100 total) means the
    #   ILP will always prefer linking — even a marginal 35 px match — over
    #   letting the cell disappear and reappear.
    "appear_weight":    -50.0,
    "disappear_weight": -50.0,
    # power=1: identity transform — preserves the negative sign produced by
    # distance_weight so that CLOSER links (less negative) truly beat FAR links.
    # power=2 or 4 would SQUARE the negative value, making far links look better
    # than close ones (the opposite of what we want).
    "power": 1,
    # image_border_size: cells whose centroid lies within this many pixels of the
    # frame boundary are exempt from appear/disappear penalties, since sperm
    # legitimately swim in/out through the edge.  [Y_pixels, X_pixels].
    "image_border_size": [50, 50],
}


# ---------------------------------------------------------------------------
# Step 1: foreground + edge images from raw frames
# ---------------------------------------------------------------------------

def _remove_small_blobs(bw, min_area):
    """
    Zero out any connected foreground component with pixel area < min_area.

    ultrack's watershed hierarchy builder raises a RuntimeError on regions
    that are too small, and it runs *before* its own min_area filter applies,
    so we strip tiny blobs here in preprocessing instead.
    """
    n_labels, labels, stats, _ = cv.connectedComponentsWithStats(bw, 4, cv.CV_32S)
    for label_idx in range(1, n_labels):           # 0 is background — always keep
        if stats[label_idx, cv.CC_STAT_AREA] < min_area:
            bw[labels == label_idx] = 0
    return bw


def build_foreground_and_edges(frames, thresh_method="hybrid", morph_open_size=3,
                               min_area=20):
    """
    Convert raw grayscale video frames into the two arrays ultrack expects.

    Foreground
        Binary mask: 1 where sperm are present, 0 elsewhere.
        Negative-phase-contrast video → sperm are bright → threshold directly,
        no inversion needed.

    Edges
        Morphological gradient of the foreground (dilation − erosion).
        Highlights the boundary of each binary blob — this is what ultrack
        uses to delineate individual cell candidates.

    Parameters
    ----------
    frames : np.ndarray of shape (T, H, W), uint8
    thresh_method : str
        Passed straight to tracker.threshold() — 'hybrid', 'otsu', etc.
    morph_open_size : int
        Structuring-element size for morphological opening applied to the
        binary foreground before edge extraction. Set to 1 to skip.
    min_area : int
        Blobs with fewer pixels than this are removed before passing to
        ultrack. Should match (or be slightly above) the ultrack
        segmentation_config.min_area to prevent hierarchy-build failures
        on tiny noise regions.

    Returns
    -------
    foreground : float32 ndarray (T, H, W) in [0, 1]
    edges      : float32 ndarray (T, H, W) in [0, 1]
    """
    T = len(frames)
    H, W = frames[0].shape[:2]

    foreground = np.zeros((T, H, W), dtype=np.float32)
    edges      = np.zeros((T, H, W), dtype=np.float32)

    kernel_open = cv.getStructuringElement(
        cv.MORPH_ELLIPSE, (morph_open_size, morph_open_size)
    )
    kernel_grad = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))

    for i in trange(T, desc="Thresholding frames"):
        bw = threshold(frames[i], method=thresh_method)   # uint8, 0/255

        if morph_open_size > 1:
            bw = cv.morphologyEx(bw, cv.MORPH_OPEN, kernel_open)

        # Strip blobs too small for ultrack's hierarchy builder
        bw = _remove_small_blobs(bw, min_area)

        # Morphological gradient = cell edges
        grad = cv.morphologyEx(bw, cv.MORPH_GRADIENT, kernel_grad)

        foreground[i] = bw.astype(np.float32)   / 255.0
        edges[i]      = grad.astype(np.float32) / 255.0

    return foreground, edges


# ---------------------------------------------------------------------------
# Step 2: run ultrack
# ---------------------------------------------------------------------------

def _make_ultrack_config(cfg_dict, workdir):
    """Build a ultrack MainConfig from our flat config dict."""
    try:
        from ultrack import MainConfig
    except ImportError:
        print(
            "\nERROR: ultrack is not installed.\n"
            "Install it with:  pip install ultrack\n"
            "For a faster solver also run:  pip install highspy\n",
            file=sys.stderr,
        )
        raise

    cfg = MainConfig()

    # Working directory — ultrack writes an SQLite database here
    cfg.data_config.working_dir = str(workdir)

    # Each parameter is read with an ultrack_ prefix first, then falls back to
    # the unprefixed name, then to the hard-coded default. This lets a single
    # JSON config serve both the TrackPy pipeline and ultrack without key
    # collisions (e.g. "search_range" is shared; "ultrack_min_area" is not).
    def _get(key, default):
        return cfg_dict.get(f"ultrack_{key}", cfg_dict.get(key, default))

    # Segmentation candidates
    cfg.segmentation_config.min_area = int(_get("min_area", 7))
    cfg.segmentation_config.max_area = int(_get("max_area", 500))

    # Linking — reuses search_range from the shared TrackPy key by default;
    # use ultrack_search_range in the JSON to override for ultrack only.
    cfg.linking_config.max_distance  = float(_get("search_range",    35))
    # max_neighbors: how many candidate blobs per node are offered to the ILP.
    # Default of 5 is far too low for fast-moving sperm in a dense field —
    # the correct match can easily be ranked beyond the 5 nearest neighbours.
    cfg.linking_config.max_neighbors = int(_get("max_neighbors",     25))
    # distance_weight: penalises distance when computing link scores during the
    # linking step.  Effective edge weight = IoU - distance_weight * distance.
    # For sperm, IoU ~ 0 for all links (cells move between frames), so this
    # term is what differentiates close from far candidates.  With power=1 the
    # sign is preserved and the ILP consistently picks the nearest match.
    cfg.linking_config.distance_weight = float(_get("distance_weight", 0.5))

    # Tracking — suppress division (sperm don't divide)
    cfg.tracking_config.division_weight  = float(_get("division_weight",  -1000.0))
    cfg.tracking_config.appear_weight    = float(_get("appear_weight",      -50.0))
    cfg.tracking_config.disappear_weight = float(_get("disappear_weight",   -50.0))
    # power=1 (identity): preserves the sign of negative link scores so closer
    # links (less negative) beat farther ones in the ILP objective.
    # WARNING: power=2 or 4 would SQUARE the negative value produced by
    # distance_weight, accidentally rewarding far links over close ones.
    cfg.tracking_config.power = float(_get("power", 1))

    # image_border_size: nodes within this pixel margin of the frame edge are
    # exempt from appear/disappear penalties (legitimate in/out through boundary).
    # Accepts a list/tuple [Y_px, X_px] from the config, or a single int.
    border = _get("image_border_size", None)
    if border is not None:
        if isinstance(border, (list, tuple)):
            cfg.tracking_config.image_border_size = tuple(int(b) for b in border)
        else:
            b = int(border)
            cfg.tracking_config.image_border_size = (b, b)

    return cfg


def run_ultrack(foreground, edges, cfg_dict, workdir):
    """
    Run the ultrack tracker on foreground/edge arrays.

    Parameters
    ----------
    foreground, edges : float32 ndarray (T, H, W) in [0, 1]
    cfg_dict : dict — flat config (from DEFAULT_CONFIG + user overrides)
    workdir : Path — where ultrack stores its SQLite working data

    Returns
    -------
    pd.DataFrame with columns from ultrack's tracks-layer format
    (track_id, t, y, x — and z if 3-D data, which we drop)
    """
    from ultrack import Tracker

    cfg = _make_ultrack_config(cfg_dict, workdir)

    print("Running ultrack segmentation + linking + ILP solve...")
    tracker = Tracker(cfg)
    tracker.track(foreground=foreground, edges=edges, overwrite=True)

    tracks_df, _graph = tracker.to_tracks_layer()
    return tracks_df


# ---------------------------------------------------------------------------
# Step 3: convert to pipeline CSV format
# ---------------------------------------------------------------------------

def convert_to_pipeline_format(tracks_df):
    """
    Convert ultrack output to the standard (y, x, frame, sperm) format
    used by tracker.py / metrics.py.

    ultrack returns:  track_id | t | [z] | y | x
    We rename:        sperm    | frame   | y | x
    """
    df = tracks_df.copy()

    if "z" in df.columns:
        df = df.drop(columns=["z"])

    df = df.rename(columns={"track_id": "sperm", "t": "frame"})

    df["frame"] = df["frame"].astype(int)
    df["sperm"] = df["sperm"].astype(int)
    df["x"]     = df["x"].astype(float)
    df["y"]     = df["y"].astype(float)

    df = df[["y", "x", "frame", "sperm"]].sort_values(
        ["sperm", "frame"]
    ).reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------

def track_video(videofile, cfg_dict, outputfile=None):
    """
    Full pipeline: video file → ultrack CSV.

    Parameters
    ----------
    videofile : str
    cfg_dict  : dict — merged DEFAULT_CONFIG + user config
    outputfile: str or None — defaults to <videofile stem>_ultrack.csv

    Returns
    -------
    pd.DataFrame in pipeline format
    """
    videofile = str(videofile)

    # Default output path
    if outputfile is None:
        outputfile = ".".join(videofile.split(".")[:-1]) + "_ultrack.csv"

    # Working directory — unique per video so parallel runs don't collide
    video_stem = Path(videofile).stem
    workdir = Path(cfg_dict.get("ultrack_workdir", f"{video_stem}_ultrack_workdir"))

    print(f"Loading video: {videofile}")
    frames = utils.loadVideo(videofile, as_gray=True)
    print(f"  {len(frames)} frames, {frames[0].shape[1]}x{frames[0].shape[0]} px")

    def _get(key, default):
        return cfg_dict.get(f"ultrack_{key}", cfg_dict.get(key, default))

    thresh_method   = _get("threshold_method", "hybrid")
    morph_open_size = int(_get("morph_open_size", 3))
    min_area        = int(_get("min_area", 7))

    print(f"Building foreground + edges (threshold='{thresh_method}', "
          f"morph_open={morph_open_size}px, min_area={min_area}px)...")
    foreground, edges = build_foreground_and_edges(
        frames, thresh_method, morph_open_size, min_area
    )

    raw_tracks = run_ultrack(foreground, edges, cfg_dict, workdir)

    result = convert_to_pipeline_format(raw_tracks)
    print(
        f"  Done: {result['sperm'].nunique()} tracks across "
        f"{result['frame'].nunique()} frames"
    )

    utils.saveDataFrame(result, outputfile)
    print(f"Saved to: {outputfile}")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tkinter as tk
    from tkinter import filedialog

    parser = argparse.ArgumentParser(
        description=(
            "Track sperm using ultrack ILP tracker. "
            "Output CSV is in the same format as tracker.py."
        )
    )
    parser.add_argument(
        "--videofile", type=str, default=None,
        help="Path to the video file (opens file dialog if omitted)"
    )
    parser.add_argument(
        "--configfile", type=str, default=None,
        help="Path to JSON config file. Ultrack-specific keys are read alongside "
             "the existing pipeline keys (e.g. search_range). "
             "See DEFAULT_CONFIG in this file for all supported keys."
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output CSV path (default: <videofile>_ultrack.csv)"
    )
    args = parser.parse_args()

    videofile = args.videofile
    if videofile is None:
        root = tk.Tk()
        root.withdraw()
        videofile = filedialog.askopenfilename(title="Select the video file")
        if not videofile:
            raise ValueError("No video file selected.")
        print("Selected file:", videofile)

    cfg = DEFAULT_CONFIG.copy()
    if args.configfile:
        with open(args.configfile) as f:
            cfg.update(json.load(f))

    track_video(videofile, cfg, args.output)
