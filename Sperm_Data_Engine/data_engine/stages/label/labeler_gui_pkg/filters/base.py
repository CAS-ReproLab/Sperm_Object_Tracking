"""
Track filter plugins for the labeler GUI.

Each filter is a function (tracks_df, config) -> FilterResult. `tracks_df` is
the full tracks DataFrame (frame, sperm, x, y, label, ...). `config` is a dict
of the filter's own parameters (see FILTER_PARAMS for the schema).

FilterResult.sperm_ids is the set of sperm IDs the filter flags as relevant —
used to isolate the video view to just those tracks.

FilterResult.events (optional) is a normalized DataFrame for navigation, one
row per "event of interest". If present it must include:
  frame       int   — frame to jump to for this event
  sperm_ids   list  — sperm IDs involved (any length, e.g. a pair for crossovers)
Any additional columns are shown as extra info in the filter results table.
"""

FILTERS        = {}   # name -> function(tracks_df, config) -> FilterResult
FILTER_PARAMS  = {}   # name -> {param_key: {type, default, min, max, step, label}}


class FilterResult:
    def __init__(self, sperm_ids, events=None):
        self.sperm_ids = set(int(s) for s in sperm_ids)
        self.events = events


def register(name, params=None):
    """Decorator: register a filter function under `name` with a param schema."""
    def decorator(fn):
        FILTERS[name] = fn
        FILTER_PARAMS[name] = params or {}
        return fn
    return decorator
