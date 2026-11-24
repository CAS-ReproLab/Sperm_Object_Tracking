"""
trajectory_io.py -- read/write the canonical sperm-trajectory CSV format used
across the whole data engine: columns `frame, sperm, x, y` (+ optional `label`
for human-in-the-loop correction). Adapted from
UnifiedFramework/utils.py's loadDataFrame()/saveDataFrame() -- same on-disk
format, made a bit more defensive since it's now a cross-stage contract
instead of a single tool's internal helper.
"""
import pandas as pd

REQUIRED_COLUMNS = {"frame", "sperm", "x", "y"}


def load_tracks(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing required columns {missing} "
                          f"(found: {list(df.columns)})")
    return df


def save_tracks(df: pd.DataFrame, path) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Cannot save {path}: dataframe missing required columns {missing}")
    df.to_csv(path, index=False)


def from_bytetrack_output(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a track_video_bytetrack.py-style dataframe
    (frame, track_id, class_id, x, y, w, h, conf) into the canonical
    frame,sperm,x,y,label format the labeler GUI already reads/writes, ready
    for human correction. This is the adapter the Phase-5 bootstrap-labeling
    stage will use.
    """
    missing = {"frame", "track_id", "x", "y"} - set(df.columns)
    if missing:
        raise ValueError(f"Not a ByteTrack-shaped dataframe: missing {missing}")

    out = df.rename(columns={"track_id": "sperm"})[["frame", "sperm", "x", "y"]].copy()
    out["sperm"] = out["sperm"].astype(int)
    out["label"] = ""
    return out
