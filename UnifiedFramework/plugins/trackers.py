"""
Tracker plugins — each function takes (detections_df, config) and returns a
tracks DataFrame with a 'sperm' column added (renamed from trackpy's 'particle').

Detections come from a Perception plugin and always have at minimum:
  [frame, x, y]
and optionally:
  [bbox_x, bbox_y, bbox_w, bbox_h, area, segmentation]

Trackers that require segmentation masks (e.g. IoU-based) should document that
requirement in their TRACKER_PARAMS entry and the GUI will warn the user if the
chosen Perception method does not produce segmentations.
"""

import trackpy as tp


def _trackpy(detections, config):
    """Standard nearest-neighbour linking via trackpy with adaptive search range."""
    search_range  = config.get("search_range", 21)
    memory        = config.get("memory", 3)
    adaptive_stop = config.get("adaptive_stop", 0.2)
    adaptive_step = config.get("adaptive_step", 0.95)
    min_track_len = config.get("min_track_len", 15)

    t = tp.link(
        detections,
        search_range=search_range,
        memory=memory,
        adaptive_stop=adaptive_stop,
        adaptive_step=adaptive_step,
    )
    t = tp.filter_stubs(t, min_track_len)
    t = t.rename(columns={"particle": "sperm"})
    t = t.reset_index(drop=True)
    return t


# ── registry ─────────────────────────────────────────────────────────────────

TRACKERS = {
    "trackpy": _trackpy,
}

# Set to True for trackers that require segmentation masks from Perception.
REQUIRES_SEGMENTATION = {
    "trackpy": False,
}

TRACKER_PARAMS = {
    "trackpy": {
        "search_range": {
            "type": "int", "default": 21, "min": 1, "max": 100, "step": 1,
            "label": "Search range (px)",
        },
        "memory": {
            "type": "int", "default": 3, "min": 0, "max": 20, "step": 1,
            "label": "Memory (frames)",
        },
        "adaptive_stop": {
            "type": "float", "default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05,
            "label": "Adaptive stop",
        },
        "adaptive_step": {
            "type": "float", "default": 0.95, "min": 0.5, "max": 1.0, "step": 0.01,
            "label": "Adaptive step",
        },
        "min_track_len": {
            "type": "int", "default": 15, "min": 1, "max": 100, "step": 1,
            "label": "Min track length (frames)",
        },
    },
}
