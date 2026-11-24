import pandas as pd

from crossover_detector import detect_crossovers
from labeler_gui_pkg.filters.base import register, FilterResult

_EVENT_COLUMNS = [
    "frame", "sperm_ids", "likelihood", "min_distance_px",
    "segments_cross", "position_swap",
]


@register("crossover", params={
    "threshold": {
        "type": "float", "default": 25.0, "min": 1.0, "max": 300.0, "step": 1.0,
        "label": "Proximity threshold (px)",
    },
    "min_likelihood": {
        "type": "float", "default": 0.4, "min": 0.0, "max": 1.0, "step": 0.05,
        "label": "Min likelihood",
    },
})
def crossover_filter(tracks_df, config):
    """
    Flag sperm pairs likely to have crossed (and therefore swapped IDs during
    tracking). Wraps crossover_detector.detect_crossovers().
    """
    df = tracks_df[["frame", "sperm", "x", "y"]].copy()
    df["frame"] = df["frame"].astype(int)
    df["sperm"] = df["sperm"].astype(int)

    threshold      = config.get("threshold", 25.0)
    min_likelihood = config.get("min_likelihood", 0.4)

    events = detect_crossovers(df, threshold=threshold)
    if not events.empty:
        events = events[events["likelihood"] >= min_likelihood].reset_index(drop=True)

    if events.empty:
        return FilterResult(sperm_ids=set(), events=pd.DataFrame(columns=_EVENT_COLUMNS))

    normalized = pd.DataFrame({
        "frame":           events["closest_frame"].astype(int),
        "sperm_ids":       list(zip(events["sperm_a"].astype(int), events["sperm_b"].astype(int))),
        "likelihood":      events["likelihood"],
        "min_distance_px": events["min_distance_px"],
        "segments_cross":  events["segments_cross"],
        "position_swap":   events["position_swap"],
    }).sort_values("likelihood", ascending=False).reset_index(drop=True)

    all_ids = set(events["sperm_a"]).union(set(events["sperm_b"]))
    return FilterResult(sperm_ids=all_ids, events=normalized)
