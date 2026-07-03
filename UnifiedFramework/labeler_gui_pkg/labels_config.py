import json

DEFAULT_LABELS_PATH = "configs/labels.json"
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
    with open(path, "w") as f:
        json.dump({"labels": labels}, f, indent=4)
