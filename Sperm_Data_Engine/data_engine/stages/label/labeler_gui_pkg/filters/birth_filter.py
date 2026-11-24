import pandas as pd

from labeler_gui_pkg.filters.base import register, FilterResult

_EVENT_COLUMNS = ["frame", "sperm_ids", "x", "y", "dist_from_edge_px"]


@register("birth", params={
    "min_frame": {
        "type": "int", "default": 10, "min": 0, "max": 100000,
        "label": "Flag births after frame",
    },
    "edge_margin_px": {
        "type": "float", "default": 50.0, "min": 0.0, "max": 2000.0, "step": 5.0,
        "label": "Edge margin (px)",
    },
})
def birth_filter(tracks_df, config):
    """
    Flag sperm whose first-appearance ("birth") frame is suspicious: it
    happens after the first few frames of the video AND it's far from every
    frame edge. New IDs are expected early (tracking start) or near an edge
    (a sperm drifting into view); a birth that is both late and central is
    usually a false detection or a track that broke and restarted under a
    new ID, not a real new sperm entering the frame.
    """
    df = tracks_df[["frame", "sperm", "x", "y"]].copy()
    df["frame"] = df["frame"].astype(int)
    df["sperm"] = df["sperm"].astype(int)

    min_frame      = config.get("min_frame", 10)
    edge_margin_px = config.get("edge_margin_px", 50.0)
    width  = config.get("_frame_width")
    height = config.get("_frame_height")

    births = df.loc[df.groupby("sperm")["frame"].idxmin()].reset_index(drop=True)
    births = births[births["frame"] > min_frame]

    if births.empty:
        return FilterResult(sperm_ids=set(), events=pd.DataFrame(columns=_EVENT_COLUMNS))

    if width and height:
        edge_dists = [births["x"], width - births["x"], births["y"], height - births["y"]]
    else:
        # Frame size wasn't available (shouldn't normally happen -- the
        # labeler sets it from the loaded video) -- fall back to distance
        # from the observed track extent rather than skipping the edge
        # check entirely.
        x_min, x_max = df["x"].min(), df["x"].max()
        y_min, y_max = df["y"].min(), df["y"].max()
        edge_dists = [births["x"] - x_min, x_max - births["x"],
                      births["y"] - y_min, y_max - births["y"]]
    births = births.assign(dist_from_edge_px=pd.concat(edge_dists, axis=1).min(axis=1))

    flagged = births[births["dist_from_edge_px"] > edge_margin_px].sort_values(
        "dist_from_edge_px", ascending=False).reset_index(drop=True)

    if flagged.empty:
        return FilterResult(sperm_ids=set(), events=pd.DataFrame(columns=_EVENT_COLUMNS))

    normalized = pd.DataFrame({
        "frame":             flagged["frame"],
        "sperm_ids":         [[s] for s in flagged["sperm"]],
        "x":                 flagged["x"],
        "y":                 flagged["y"],
        "dist_from_edge_px": flagged["dist_from_edge_px"],
    })
    return FilterResult(sperm_ids=set(flagged["sperm"]), events=normalized)
