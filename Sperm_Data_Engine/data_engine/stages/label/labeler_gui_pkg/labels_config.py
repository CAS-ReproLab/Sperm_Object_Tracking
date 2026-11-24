import json
from pathlib import Path

# DATA-ENGINE FIX: was the bare relative string "configs/labels.json",
# resolved against the process's current working directory rather than
# this module's location -- worked only by coincidence, because every
# entry point so far (labeler_gui.py, correction_flow.py) happened to
# always be launched from the repo root. The GUI hub is a longer-lived
# process where that assumption is easier to violate by accident, so this
# is now anchored to data_engine/stages/label/configs/labels.json instead
# (this labeler_gui_pkg/ dir's parent), independent of CWD.
DEFAULT_LABELS_PATH = str(Path(__file__).resolve().parent.parent / "configs" / "labels.json")
DEFAULT_LABELS       = ["", "sperm", "cluster", "debris", "immotile"]


def load_labels(path=DEFAULT_LABELS_PATH):
    """Return the list of label strings. Falls back to defaults if the file is missing."""
    try:
        with open(path) as f:
            data = json.load(f)
        labels = data.get("labels", DEFAULT_LABELS)
        if "" not in labels:
            labels = [""] + labels
        return labels
    except FileNotFoundError:
        return list(DEFAULT_LABELS)


def save_labels(labels, path=DEFAULT_LABELS_PATH):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"labels": labels}, f, indent=4)
