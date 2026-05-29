"""
Perception plugins — each function takes (frames, config) and returns a DataFrame
with columns [frame, x, y] at minimum, and optionally [bbox_x, bbox_y, bbox_w, bbox_h,
area, segmentation] when the method also produces segmentations.

This unified stage replaces the separate detect + segment steps so that pipelines
where segmentation precedes tracking (e.g. IoU-based) can be expressed naturally.
"""

import numpy as np
import cv2 as cv
import pandas as pd
import trackpy as tp
from tqdm import trange


# ── helpers ──────────────────────────────────────────────────────────────────

def _threshold(frame, method="otsu", global_thresh=50):
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


def _centroids_to_df(centroids_list, frame_idx):
    rows = []
    for cx, cy in centroids_list:
        rows.append({"y": cy, "x": cx, "frame": frame_idx,
                     "bbox_x": None, "bbox_y": None, "bbox_w": None, "bbox_h": None,
                     "area": None, "segmentation": None})
    return rows


# ── perception methods ────────────────────────────────────────────────────────

def _trackpy(frames, config):
    """Trackpy blob detector — detects bright spots using bandpass + Gaussian."""
    diameter = config.get("diameter", 11)
    minmass  = config.get("minmass", 500)
    maxsize  = config.get("maxsize", None)

    kwargs = {"diameter": diameter, "minmass": minmass}
    if maxsize is not None:
        kwargs["maxsize"] = maxsize

    f = tp.batch(frames, **kwargs)

    # tp.batch returns x,y already; add empty segmentation columns
    f["bbox_x"]      = None
    f["bbox_y"]      = None
    f["bbox_w"]      = None
    f["bbox_h"]      = None
    f["area"]        = None
    f["segmentation"] = None

    return f


def _threshold_simple(frames, config):
    """Threshold + connected components. Fast, no mass filter."""
    method        = config.get("threshold_method", "hybrid")
    global_thresh = config.get("global_thresh", 50)

    rows = []
    for i in trange(len(frames), desc="Detecting"):
        bw = _threshold(frames[i], method=method, global_thresh=global_thresh)
        _, _, _, centroids = cv.connectedComponentsWithStats(bw, 4, cv.CV_32S)
        for cx, cy in centroids[1:]:   # skip background
            rows.append({"y": cy, "x": cx, "frame": i,
                         "bbox_x": None, "bbox_y": None,
                         "bbox_w": None, "bbox_h": None,
                         "area": None,  "segmentation": None})

    return pd.DataFrame(rows, columns=["y", "x", "frame",
                                        "bbox_x", "bbox_y", "bbox_w", "bbox_h",
                                        "area", "segmentation"])


def _morphology(frames, config):
    """Otsu threshold + morphological opening before centroid extraction."""
    kernel_size = tuple(config.get("morph_kernel_size", [3, 3]))

    rows = []
    for i in trange(len(frames), desc="Detecting"):
        bw     = _threshold(frames[i], method="otsu")
        kernel = np.ones(kernel_size, np.uint8)
        bw     = cv.morphologyEx(bw, cv.MORPH_OPEN, kernel)
        _, _, _, centroids = cv.connectedComponentsWithStats(bw, 4, cv.CV_32S)
        for cx, cy in centroids[1:]:
            rows.append({"y": cy, "x": cx, "frame": i,
                         "bbox_x": None, "bbox_y": None,
                         "bbox_w": None, "bbox_h": None,
                         "area": None,  "segmentation": None})

    return pd.DataFrame(rows, columns=["y", "x", "frame",
                                        "bbox_x", "bbox_y", "bbox_w", "bbox_h",
                                        "area", "segmentation"])


def _threshold_with_segmentation(frames, config):
    """
    Hybrid threshold + connected components, returning full segmentation masks
    alongside centroids. Use this when a downstream tracker needs mask data
    (e.g. IoU-based linking), or to skip the separate segmentation postprocessor.
    """
    method        = config.get("threshold_method", "hybrid")
    global_thresh = config.get("global_thresh", 50)

    rows = []
    for i in trange(len(frames), desc="Detecting + segmenting"):
        bw = _threshold(frames[i], method=method, global_thresh=global_thresh)
        _, label_im, stats, centroids = cv.connectedComponentsWithStats(bw, 4, cv.CV_32S)

        areas = stats[1:, 4]
        bboxs = stats[1:, 0:4]
        label_im = label_im - 1        # background → -1

        segs = _label_im_to_segmentations(label_im, len(stats))

        for idx, (cx, cy) in enumerate(centroids[1:]):
            bbox = bboxs[idx]
            rows.append({
                "y":           cy,
                "x":           cx,
                "frame":       i,
                "bbox_x":      bbox[0],
                "bbox_y":      bbox[1],
                "bbox_w":      bbox[2],
                "bbox_h":      bbox[3],
                "area":        areas[idx],
                "segmentation": segs[idx + 1],
            })

    return pd.DataFrame(rows, columns=["y", "x", "frame",
                                        "bbox_x", "bbox_y", "bbox_w", "bbox_h",
                                        "area", "segmentation"])


# ── registry ─────────────────────────────────────────────────────────────────

PERCEPTION_METHODS = {
    "trackpy":                    _trackpy,
    "threshold_simple":           _threshold_simple,
    "morphology":                 _morphology,
    "threshold_with_segmentation":_threshold_with_segmentation,
}

# Which methods already populate segmentation columns (so the GUI can skip
# showing the segmentation postprocessor as a separate option).
PRODUCES_SEGMENTATION = {
    "trackpy":                     False,
    "threshold_simple":            False,
    "morphology":                  False,
    "threshold_with_segmentation": True,
}

PERCEPTION_PARAMS = {
    "trackpy": {
        "diameter": {
            "type": "int", "default": 11, "min": 3, "max": 51, "step": 2,
            "label": "Diameter (px, odd)",
        },
        "minmass": {
            "type": "int", "default": 500, "min": 0, "max": 10000, "step": 50,
            "label": "Min mass",
        },
        "maxsize": {
            "type": "float", "default": None, "min": 0.0, "max": 20.0, "step": 0.5,
            "label": "Max size (optional)",
        },
    },
    "threshold_simple": {
        "threshold_method": {
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
    "morphology": {
        "morph_kernel_size": {
            "type": "int", "default": 3, "min": 1, "max": 15, "step": 2,
            "label": "Morph kernel size",
        },
    },
    "threshold_with_segmentation": {
        "threshold_method": {
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
