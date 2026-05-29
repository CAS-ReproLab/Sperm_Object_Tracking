import numpy as np
import cv2 as cv


def _median_filter(frames, config):
    """Subtract temporal median from each pixel. Good for removing static background."""
    frames = frames.astype(np.float32)
    med = np.median(frames, axis=0)
    frames = np.abs(frames - med)
    return frames.astype(np.uint8)


def _median_filter_per_frame(frames, config):
    """Subtract per-frame spatial median. Normalizes uneven illumination frame-by-frame."""
    frames = frames.astype(np.float32)
    for i in range(len(frames)):
        med = np.median(frames[i], axis=0)
        frames[i] = np.abs(frames[i] - med)
    return frames.astype(np.uint8)


def _positive_phase_filter(frames, config):
    """
    Subtract temporal median then brighten dark residuals past a cutoff.
    Useful when cells appear as dark blobs on a bright background.
    """
    cutoff = config.get("positive_phase_cutoff", 5)
    frames = frames.astype(np.float32)
    med = np.median(frames, axis=0)
    frames = frames - med
    tails = np.where(frames < -cutoff)
    frames[tails] = 255 - frames[tails]
    frames = np.clip(frames, 0, 255)
    return frames.astype(np.uint8)


def _none(frames, config):
    return frames


# Registry — keys are what appear in config["preprocessor"] and GUI dropdowns.
PREPROCESSORS = {
    "none":                   _none,
    "median_filter":          _median_filter,
    "median_filter_per_frame":_median_filter_per_frame,
    "positive_phase_filter":  _positive_phase_filter,
}

# Parameter schema used by the GUI to build widgets dynamically.
# Each entry: {type, default, min, max, step, label}
PREPROCESSOR_PARAMS = {
    "none": {},
    "median_filter": {},
    "median_filter_per_frame": {},
    "positive_phase_filter": {
        "positive_phase_cutoff": {
            "type": "float", "default": 5.0, "min": 0.0, "max": 50.0, "step": 0.5,
            "label": "Phase cutoff",
        },
    },
}
