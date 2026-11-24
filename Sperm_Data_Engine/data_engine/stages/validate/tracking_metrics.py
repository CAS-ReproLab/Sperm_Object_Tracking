"""
tracking_metrics.py -- CTC-style tracking metrics (DET/TRA/LNK/TF/MOTA/IDF1/HOTA).

Core logic copied from the reference metrics_v2.py, stripped of its
argparse/tkinter __main__ block down to plain importable functions, plus one
new convenience wrapper (score_tracks) that evaluate.py calls directly.

Depends on the vendored ctc_metrics package at data_engine/common/ctc_metrics
(copied from UnifiedFramework/ctc_metrics, already patched for NumPy 2.x --
see that package's utils/representations.py). ctc_metrics' own modules use
absolute intra-package imports (`from ctc_metrics.metrics import ...`), so
its parent directory needs to be on sys.path once -- done here, in one
documented place, rather than via an external PYTHONPATH the way the
reference metrics_v2.py required.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

_CTC_METRICS_PARENT = Path(__file__).resolve().parents[2] / "common"
if str(_CTC_METRICS_PARENT) not in sys.path:
    sys.path.insert(0, str(_CTC_METRICS_PARENT))

from ctc_metrics.utils.representations import merge_tracks, count_acyclic_graph_correction_operations
from ctc_metrics.metrics import (
    det, tra, ct, tf, mota, hota, idf1, lnk,
)

DESIRED_METRICS = ["DET", "TRA", "LNK", "TF", "MOTA", "IDF1", "HOTA"]


def interpolate_tracks(df: pd.DataFrame) -> pd.DataFrame:
    """Fill in missing frames for each sperm track via linear interpolation."""
    if df.empty:
        return df
    interpolated = []
    for sperm_id, group in df.groupby("sperm"):
        group = group.sort_values("frame").set_index("frame")
        dup_frames = group.index[group.index.duplicated()].unique().tolist()
        if dup_frames:
            # A real data problem, not a code bug: this sperm ID appears more
            # than once in the same frame -- two distinct tracks that were
            # never split apart after a crossover (or an ID reused by
            # mistake). pandas' own error here ("cannot reindex on an axis
            # with duplicate labels") gives no hint which sperm/frames are at
            # fault, so raise something a human can actually act on: reopen
            # the source video in the labeler and split this ID apart.
            raise ValueError(
                f"sperm id {sperm_id!r} has {len(dup_frames)} duplicate frame(s) "
                f"(e.g. frame {dup_frames[0]}) -- this ID was likely never split "
                f"into two after a crossover. Fix it in the labeler (Split) and "
                f"re-export before evaluating this clip."
            )
        full_range = np.arange(group.index.min(), group.index.max() + 1)
        group = group.reindex(full_range).interpolate(method="linear")
        group["sperm"] = sperm_id
        group = group.reset_index().rename(columns={"index": "frame"})
        interpolated.append(group)
    return pd.concat(interpolated).reset_index(drop=True)


def filter_stationary(df: pd.DataFrame, epsilon: float = 5.0) -> pd.DataFrame:
    """Drop sperm whose furthest excursion from their mean position is < epsilon px."""
    filter_list = []
    for sperm in df["sperm"].unique():
        locs = df[df["sperm"] == sperm]
        if len(locs) < 2:
            continue
        mean_loc = locs[["x", "y"]].mean()
        dists = locs[["x", "y"]].sub(mean_loc).pow(2).sum(axis=1).pow(0.5)
        if dists.max() < epsilon:
            filter_list.append(sperm)
    return df[~df["sperm"].isin(filter_list)]


def make_track_data(df: pd.DataFrame) -> np.ndarray:
    """[sperm_id, birth_frame, death_frame, parent] rows, CTC-metrics track-table shape."""
    tracks = []
    for s in df["sperm"].unique():
        s = int(s)
        frames = df[df["sperm"] == s]["frame"].values
        if len(frames):
            tracks.append([s, int(np.amin(frames)), int(np.amax(frames)), 0])
    return np.array(tracks, dtype=int)


def make_trajectory_data(pred: pd.DataFrame, gt: pd.DataFrame, cutoff: float = 10) -> dict:
    """Hungarian-match predicted to ground-truth sperm IDs in every frame."""
    labels_ref, labels_comp, mapped_ref, mapped_comp = [], [], [], []
    max_frame = max(pred["frame"].max(), gt["frame"].max())

    for f in range(0, int(max_frame) + 1):
        ref_data = gt[gt["frame"] == f][["sperm", "x", "y"]].values
        comp_data = pred[pred["frame"] == f][["sperm", "x", "y"]].values

        if len(ref_data) == 0 or len(comp_data) == 0:
            labels_ref.append([]); labels_comp.append([])
            mapped_ref.append([]); mapped_comp.append([])
            continue

        labels_ref_frame = ref_data[:, 0].astype(int)
        labels_comp_frame = comp_data[:, 0].astype(int)
        dist_matrix = cdist(ref_data[:, 1:], comp_data[:, 1:])
        row_ind, col_ind = linear_sum_assignment(dist_matrix)

        valid_row, valid_col = [], []
        for r, c in zip(row_ind, col_ind):
            if dist_matrix[r, c] <= cutoff:
                valid_row.append(r); valid_col.append(c)

        mapped_ref_frame = [labels_ref_frame[r] for r in valid_row]
        mapped_comp_frame = [labels_comp_frame[c] for c in valid_col]

        labels_ref.append(labels_ref_frame); labels_comp.append(labels_comp_frame)
        mapped_ref.append(mapped_ref_frame); mapped_comp.append(mapped_comp_frame)

    return {
        "labels_ref": labels_ref, "labels_comp": labels_comp,
        "mapped_ref": mapped_ref, "mapped_comp": mapped_comp,
    }


def append_merged_trajectory(ref_tracks, comp_tracks, traj: dict) -> dict:
    new_tracks, new_labels, new_mapped = merge_tracks(ref_tracks, traj["labels_ref"], traj["mapped_ref"])
    traj["ref_tracks_merged"] = new_tracks
    traj["labels_ref_merged"] = new_labels
    traj["mapped_ref_merged"] = new_mapped

    new_tracks, new_labels, new_mapped = merge_tracks(comp_tracks, traj["labels_comp"], traj["mapped_comp"])
    traj["comp_tracks_merged"] = new_tracks
    traj["labels_comp_merged"] = new_labels
    traj["mapped_comp_merged"] = new_mapped
    return traj


def compute_metrics(ref_tracks, comp_tracks, traj: dict) -> dict:
    graph_ops = count_acyclic_graph_correction_operations(
        ref_tracks, comp_tracks, traj["labels_ref"], traj["labels_comp"],
        traj["mapped_ref"], traj["mapped_comp"],
    )

    results = {}
    results["DET"] = det(**graph_ops)
    _tra, _aogm, _aogm0 = tra(**graph_ops)
    results["TRA"] = _tra
    results["LNK"] = lnk(**graph_ops)
    results["CT"] = ct(comp_tracks, ref_tracks, traj["labels_ref"], traj["mapped_ref"], traj["mapped_comp"])
    results["TF"] = tf(ref_tracks, traj["labels_ref"], traj["mapped_ref"], traj["mapped_comp"])

    results.update(mota(traj["labels_ref_merged"], traj["labels_comp_merged"],
                         traj["mapped_ref_merged"], traj["mapped_comp_merged"]))
    results.update(hota(traj["labels_ref_merged"], traj["labels_comp_merged"],
                         traj["mapped_ref_merged"], traj["mapped_comp_merged"]))
    results.update(idf1(traj["labels_ref_merged"], traj["labels_comp_merged"],
                         traj["mapped_ref_merged"], traj["mapped_comp_merged"]))
    return results


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()
    df = interpolate_tracks(df)
    df = df.dropna(subset=["sperm"])
    df["sperm"] = df["sperm"].astype(int)
    return df


def _score_or_zero(pred_df: pd.DataFrame, gt_df: pd.DataFrame) -> dict:
    """
    Score one pred/gt pair, or return {} (-> reported as 0 by score_tracks)
    if either side has no rows at all. Without this guard, make_trajectory_data
    computes max(pred["frame"].max(), gt["frame"].max()) which is NaN when a
    side is empty, crashing int(NaN) -- hit in practice when a model performs
    very poorly on an out-of-distribution clip (e.g. evaluated on a density
    tier it wasn't trained on) and produces zero usable detections, or when
    filter_stationary() removes every sperm on one side as near-motionless.
    """
    if pred_df.empty or gt_df.empty:
        print(f"  (empty after filtering -- pred={len(pred_df)} rows, gt={len(gt_df)} rows; "
              f"reporting 0 for this variant)")
        return {}
    tracks_pred, tracks_gt = make_track_data(pred_df), make_track_data(gt_df)
    traj = append_merged_trajectory(tracks_gt, tracks_pred, make_trajectory_data(pred_df, gt_df))
    return compute_metrics(tracks_gt, tracks_pred, traj)


def score_tracks(pred: pd.DataFrame, gt: pd.DataFrame) -> pd.DataFrame:
    """
    Score one clip's predicted tracks against ground truth.

    pred: dataframe with either a `sperm` or `track_id` column (track_id is
    renamed automatically, matching track.py's output schema), plus x, y, frame.
    gt: dataframe with sperm, x, y, frame (the canonical trajectory_io format).

    Returns a dataframe shaped like the reference metrics_v2.py's output:
    columns Metric/Unfiltered/Filtered, rows DET/TRA/LNK/TF/MOTA/IDF1/HOTA --
    this is exactly what data_engine/metrics/density_report.py expects, one
    file per clip.
    """
    if "track_id" in pred.columns and "sperm" not in pred.columns:
        pred = pred.rename(columns={"track_id": "sperm"})

    pred = _prepare(pred)
    gt = _prepare(gt)

    pred_filtered = filter_stationary(pred)
    gt_filtered = filter_stationary(gt)

    results = _score_or_zero(pred, gt)
    results_f = _score_or_zero(pred_filtered, gt_filtered)

    rows = [
        {"Metric": m, "Unfiltered": results.get(m, 0), "Filtered": results_f.get(m, 0)}
        for m in DESIRED_METRICS
    ]
    return pd.DataFrame(rows)
