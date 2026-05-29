"""
Postprocessor plugins — each function takes (tracks_df, frames, config) and
returns a modified tracks_df.

'frames' is the full numpy array of video frames (grayscale). Postprocessors
that only operate on the DataFrame (interpolation, stat computation) can ignore
it. Postprocessors that need pixel data (segmentation) use it.
"""

import numpy as np
import cv2 as cv
import pandas as pd
from tqdm import trange


# ── helpers ───────────────────────────────────────────────────────────────────

def _threshold(frame, method="hybrid", global_thresh=50):
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
        raise ValueError(f"Unknown threshold method: {method}")
    return bw


def _label_im_to_segmentations(label_im, num_labels):
    segs = [[] for _ in range(num_labels)]
    rows, cols = label_im.shape
    for i in range(rows):
        for j in range(cols):
            lbl = label_im[i, j]
            if lbl != -1:
                segs[lbl].append([i, j])
    return segs


# ── postprocessor functions ───────────────────────────────────────────────────

def _none(tracks, frames, config):
    return tracks


def _interpolate(tracks, frames, config):
    """Linear interpolation of missing frames within each track."""
    max_gap = config.get("interpolate_max_gap", 15)

    tracks["sperm"] = tracks["sperm"].astype(int)
    tracks["x"]     = tracks["x"].astype(float)
    tracks["y"]     = tracks["y"].astype(float)

    new_rows = []
    for sperm_id in tracks["sperm"].unique():
        sperm = tracks[tracks["sperm"] == sperm_id].sort_values("frame")
        frames_arr = sperm["frame"].values
        x_arr      = sperm["x"].values
        y_arr      = sperm["y"].values

        for i in range(1, len(frames_arr)):
            gap = frames_arr[i] - frames_arr[i - 1]
            if gap <= 1:
                continue
            if gap > max_gap:
                continue
            for f in range(1, gap):
                alpha   = f / gap
                new_row = sperm.iloc[i - 1].copy()
                new_row["frame"] = frames_arr[i - 1] + f
                new_row["x"]     = x_arr[i - 1] + alpha * (x_arr[i] - x_arr[i - 1])
                new_row["y"]     = y_arr[i - 1] + alpha * (y_arr[i] - y_arr[i - 1])
                new_rows.append(new_row)

    if new_rows:
        combined = pd.concat(
            [tracks, pd.DataFrame(new_rows, columns=tracks.columns)],
            ignore_index=True,
        )
    else:
        combined = tracks.copy()

    return combined.sort_values(["sperm", "frame"]).reset_index(drop=True)


def _segment(tracks, frames, config):
    """
    Associate each centroid with a segmentation mask using connected components.
    Adds columns: area, bbox_x, bbox_y, bbox_w, bbox_h, segmentation.
    Use this when the Perception method did not produce segmentations.
    """
    method        = config.get("seg_threshold_method", "hybrid")
    global_thresh = config.get("global_thresh", 50)

    final = tracks.copy(deep=True)
    for col in ["area", "bbox_x", "bbox_y", "bbox_w", "bbox_h"]:
        final[col] = 0
    final["segmentation"] = None

    all_label_ims    = np.zeros(
        (len(frames), frames[0].shape[0], frames[0].shape[1]), dtype=np.int32
    )
    all_bboxs        = []
    all_areas        = []
    all_segmentations = []

    for n in trange(len(frames), desc="Segmenting"):
        bw = _threshold(frames[n], method=method, global_thresh=global_thresh)
        _, label_im, stats, _ = cv.connectedComponentsWithStats(bw, 4, cv.CV_32S)

        areas  = stats[1:, 4]
        bboxs  = stats[1:, 0:4]
        label_im = label_im - 1    # background → -1

        segs = _label_im_to_segmentations(label_im, len(stats))

        all_label_ims[n] = label_im
        all_bboxs.append(bboxs)
        all_areas.append(areas)
        all_segmentations.append(segs)

    out_of_bounds = 0
    for idx, row in final.iterrows():
        n = int(row["frame"])
        r = int(row["y"])
        c = int(row["x"])
        lbl_im = all_label_ims[n]

        if r < 0 or c < 0 or r >= lbl_im.shape[0] or c >= lbl_im.shape[1]:
            out_of_bounds += 1
            final.at[idx, "area"]        = -1
            final.at[idx, "bbox_x"]      = -1
            final.at[idx, "bbox_y"]      = -1
            final.at[idx, "bbox_w"]      = -1
            final.at[idx, "bbox_h"]      = -1
            final.at[idx, "segmentation"] = []
            continue

        r2 = min(r + 1, lbl_im.shape[0] - 1)
        c2 = min(c + 1, lbl_im.shape[1] - 1)
        candidates = [lbl_im[r, c], lbl_im[r, c2], lbl_im[r2, c], lbl_im[r2, c2]]
        label = next((v for v in candidates if v >= 0), -1)

        if label == -1:
            out_of_bounds += 1
            final.at[idx, "area"]        = -1
            final.at[idx, "bbox_x"]      = -1
            final.at[idx, "bbox_y"]      = -1
            final.at[idx, "bbox_w"]      = -1
            final.at[idx, "bbox_h"]      = -1
            final.at[idx, "segmentation"] = []
            continue

        bbox = all_bboxs[n][label]
        final.at[idx, "area"]        = all_areas[n][label]
        final.at[idx, "bbox_x"]      = bbox[0]
        final.at[idx, "bbox_y"]      = bbox[1]
        final.at[idx, "bbox_w"]      = bbox[2]
        final.at[idx, "bbox_h"]      = bbox[3]
        final.at[idx, "segmentation"] = all_segmentations[n][label + 1]

    print(f"Segmentation: {out_of_bounds} centroids fell in background across {len(frames)} frames.")
    return final


# ── registry ─────────────────────────────────────────────────────────────────

POSTPROCESSORS = {
    "none":        _none,
    "interpolate": _interpolate,
    "segment":     _segment,
}

POSTPROCESSOR_PARAMS = {
    "none": {},
    "interpolate": {
        "interpolate_max_gap": {
            "type": "int", "default": 15, "min": 1, "max": 50, "step": 1,
            "label": "Max gap to interpolate (frames)",
        },
    },
    "segment": {
        "seg_threshold_method": {
            "type": "choice",
            "default": "hybrid",
            "choices": ["global", "median", "otsu", "adaptive", "hybrid"],
            "label": "Threshold method",
        },
        "global_thresh": {
            "type": "int", "default": 50, "min": 0, "max": 255, "step": 1,
            "label": "Global threshold",
        },
    },
}
