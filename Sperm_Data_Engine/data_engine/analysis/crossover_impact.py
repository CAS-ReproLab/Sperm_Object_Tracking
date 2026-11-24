"""
crossover_impact.py -- does tracker failure (ID switches, dropped
detections) actually concentrate around geometric crossover events, and how
do trackers compare there?

Standalone diagnostic -- deliberately NOT part of the data engine loop
(stages/): it doesn't feed training or the registry, it's a side analysis
for understanding *why* a tracker's aggregate metrics look the way they do.
Point it at one ground-truth CSV and one or more PRE-CORRECTION tracker
prediction CSVs. bootstrap_label.py's <video>_bootstrap.csv is exactly the
right artifact for this -- the tracker's raw output before a human
corrected it, already in canonical format. Run bootstrap_label.py once per
backend (--backend trackpy, --backend yolo --model ...) on the same video,
keep both *_bootstrap.csv files before they'd otherwise be overwritten, and
point this script at both to compare.

Reuses rather than reimplements:
  - stages/label/crossover_detector.detect_crossovers() -- the labeler's
    own crossover-likelihood estimator, run here against ground truth to
    find WHERE two sperm passed close enough to risk confusing a tracker.
  - stages/validate/tracking_metrics.py's _prepare()/filter_stationary()
    for gt/pred preprocessing, matching evaluate.py's own convention.

Pred<->gt matching here is NOT tracking_metrics.make_trajectory_data() --
that function reruns an independent Hungarian match from scratch every
frame, which was tried first and confirmed empirically to disagree badly
with the standard metrics: on RL_1, it flagged YOLO+ByteTrack as having a
4x higher baseline failure rate than trackpy, while MOTA/IDF1/HOTA (scored
on the exact same two prediction files) all show YOLO+ByteTrack clearly
ahead. The cause: a from-scratch-every-frame match has no notion of "this
gt/pred pair was already matched" -- in a crowded frame, one-to-one
Hungarian competition for the globally optimal assignment can bump a
perfectly fine existing correspondence to a farther point (or leave it
unmatched) purely because a different gt point happened to be even closer
to that pred point that frame, with nothing wrong with the original match.
YOLO+ByteTrack's denser output hit this far harder (13 spurious per-frame
"switches" collapsed to 1 real one once fixed). build_gt_to_pred_map()
instead does a continuity-preserving ("sticky") match, matching how
standard MOT ID-switch counting (CLEAR-MOT) actually works: keep a gt
track's previous match if that predicted id is still nearby (a looser
gating distance than a fresh match needs), and only Hungarian-match
whatever's left unmatched. A switch is then a genuine re-assignment to a
different id, not one frame's transient competitive noise.

Core idea: for each of a sperm's ground-truth frames, decide whether that
frame falls within a (buffered) crossover window with any partner. Then
compare the ID-switch/drop rate INSIDE crossover windows against the SAME
video's rate OUTSIDE them (not against some absolute threshold) --
answering "is this tracker's failure actually elevated near crossovers, or
is it just bad everywhere" as a single relative-risk number, comparable
across trackers.

Caveat: a track the tracker never picks up at all (systematic miss, not
crossover-specific) inflates both the crossover and baseline failure counts
roughly in proportion to how much of its time is spent near a crossover --
relative_risk isn't perfectly isolated from that, but stays a reasonable
signal for genuinely crossover-driven failure. Treat this as a diagnostic,
not a certified metric.

Run as a module from the repo root:

    python -m data_engine.analysis.crossover_impact \\
        --gt data/RL_1/video_P001_corrected_trackpy_format.csv \\
        --pred trackpy=data_engine/data/bootstrap_trackpy/video_P001_bootstrap.csv \\
        --pred yolo_bytetrack=data_engine/data/bootstrap_yolo/video_P001_bootstrap.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from data_engine.common.trajectory_io import load_tracks
from data_engine.stages.label.crossover_detector import detect_crossovers
from data_engine.stages.validate.tracking_metrics import _prepare, filter_stationary

# Matches crossover_detector.py's own CLI default.
DEFAULT_CROSSOVER_THRESHOLD = 25.0
# A fresh (non-continuing) match must land within this distance in px.
DEFAULT_MATCH_CUTOFF = 10.0
# A CONTINUING match (same predicted id as the previous frame) is allowed
# this much slack instead -- a little looser than a fresh match, so minor
# jitter doesn't manufacture a spurious drop+reacquire right next to a
# perfectly good ongoing track.
DEFAULT_GATING_MULTIPLIER = 1.5
# Frames of slack added to each side of a crossover window: an occlusion's
# tracking fallout (the switch/drop itself) can land a couple frames before
# the geometric proximity window is detected, or linger a couple after.
DEFAULT_WINDOW_BUFFER = 5


def build_gt_to_pred_map(pred_df: pd.DataFrame, gt_df: pd.DataFrame, cutoff: float = DEFAULT_MATCH_CUTOFF,
                          gating_multiplier: float = DEFAULT_GATING_MULTIPLIER):
    """{gt_sperm_id: {frame: matched_pred_sperm_id}} via a continuity-
    preserving ("sticky") per-frame match: a gt track's previous match is
    kept as long as that predicted id is still within `cutoff *
    gating_multiplier`; only gt/pred points without a valid continuation
    are Hungarian-matched among themselves. A gt id/frame with no entry
    means unmatched (a drop). See the module docstring for why this isn't
    a fresh independent Hungarian match every frame (tracking_metrics.
    make_trajectory_data()'s approach) -- confirmed empirically to
    misattribute one-to-one Hungarian competition in crowded frames as
    spurious switches/drops, disproportionately for denser predictions."""
    gating = cutoff * gating_multiplier
    frames = sorted(set(gt_df["frame"]) | set(pred_df["frame"]))
    current_match: dict = {}   # gt_id -> pred_id (or None) as of the last frame processed
    gt_to_pred: dict = {}

    for f in frames:
        g = gt_df[gt_df["frame"] == f]
        p = pred_df[pred_df["frame"] == f]
        g_ids = g["sperm"].astype(int).to_numpy()
        g_pos = g[["x", "y"]].to_numpy()
        p_ids = p["sperm"].astype(int).to_numpy()
        p_pos = p[["x", "y"]].to_numpy()
        p_pos_by_id = dict(zip(p_ids, p_pos))

        matched_gt, used_pred = set(), set()

        # 1) continue existing matches wherever still valid
        for i, gid in enumerate(g_ids):
            pid = current_match.get(gid)
            if pid is not None and pid in p_pos_by_id:
                if np.linalg.norm(g_pos[i] - p_pos_by_id[pid]) <= gating:
                    gt_to_pred.setdefault(gid, {})[f] = pid
                    matched_gt.add(gid)
                    used_pred.add(pid)

        # 2) Hungarian-match whatever's left
        rem_g = [i for i, gid in enumerate(g_ids) if gid not in matched_gt]
        rem_p = [i for i, pid in enumerate(p_ids) if pid not in used_pred]
        if rem_g and rem_p:
            dmat = cdist(g_pos[rem_g], p_pos[rem_p])
            for r, c in zip(*linear_sum_assignment(dmat)):
                if dmat[r, c] <= cutoff:
                    gid, pid = g_ids[rem_g[r]], p_ids[rem_p[c]]
                    gt_to_pred.setdefault(gid, {})[f] = pid
                    matched_gt.add(gid)

        for gid in g_ids:
            current_match[gid] = gt_to_pred.get(gid, {}).get(f)  # None if unmatched -> fresh match next time

    return gt_to_pred


def find_switch_frames(match_by_frame: dict) -> set:
    """Frames where the matched predicted id changed from the previous
    MATCHED frame for this gt track (a per-track ID switch)."""
    frames = sorted(match_by_frame)
    return {f for prev_f, f in zip(frames, frames[1:]) if match_by_frame[prev_f] != match_by_frame[f]}


def find_drop_frames(match_by_frame: dict, gt_frames: set) -> set:
    """This gt track's own frames with no matched prediction at all."""
    return {f for f in gt_frames if f not in match_by_frame}


def mark_crossover_frames(events_df: pd.DataFrame, gt_frames_by_id: dict, window_buffer: int) -> dict:
    """{gt_sperm_id: set of that sperm's own gt frames falling within a
    (buffered) crossover window against any partner}."""
    in_crossover = {sid: set() for sid in gt_frames_by_id}
    for _, ev in events_df.iterrows():
        a, b = int(ev["sperm_a"]), int(ev["sperm_b"])
        window = set(range(int(ev["window_start_frame"]) - window_buffer,
                            int(ev["window_end_frame"]) + window_buffer + 1))
        if a in in_crossover:
            in_crossover[a] |= (window & gt_frames_by_id[a])
        if b in in_crossover:
            in_crossover[b] |= (window & gt_frames_by_id[b])
    return in_crossover


def frame_level_summary(gt_frames_by_id: dict, failure_frames_by_id: dict, in_crossover_by_id: dict) -> dict:
    """Crossover-window vs. baseline (elsewhere in the same video)
    switch/drop rate, per gt-frame -- and their ratio (relative_risk), the
    headline number this script exists to produce."""
    cross_total = cross_fail = base_total = base_fail = 0
    for sid, frames in gt_frames_by_id.items():
        cross_frames = in_crossover_by_id.get(sid, set())
        fail_frames = failure_frames_by_id.get(sid, set())
        for f in frames:
            if f in cross_frames:
                cross_total += 1
                cross_fail += f in fail_frames
            else:
                base_total += 1
                base_fail += f in fail_frames

    cross_rate = (cross_fail / cross_total) if cross_total else float("nan")
    base_rate = (base_fail / base_total) if base_total else float("nan")
    relative_risk = (cross_rate / base_rate) if base_rate and base_rate == base_rate else float("nan")

    return {
        "crossover_frames": cross_total, "crossover_failure_frames": cross_fail,
        "crossover_failure_rate": cross_rate,
        "baseline_frames": base_total, "baseline_failure_frames": base_fail,
        "baseline_failure_rate": base_rate,
        "relative_risk": relative_risk,
    }


def crossover_impact(
    gt_path, pred_path, tracker_name: str = "tracker",
    crossover_threshold: float = DEFAULT_CROSSOVER_THRESHOLD,
    match_cutoff: float = DEFAULT_MATCH_CUTOFF,
    window_buffer: int = DEFAULT_WINDOW_BUFFER,
    min_likelihood: float = 0.0,
    apply_stationary_filter: bool = True,
):
    """
    Returns (events_df, summary_dict).

    events_df: one row per ground-truth crossover event, with per-event
    pass/fail flags for each sperm involved -- useful for drilling into
    specific bad crossovers (sort by likelihood).

    summary_dict: the frame_level_summary() aggregate for this tracker,
    plus counts -- comparable across trackers via relative_risk.
    """
    gt = load_tracks(gt_path)
    pred = load_tracks(pred_path)
    if "track_id" in pred.columns and "sperm" not in pred.columns:
        pred = pred.rename(columns={"track_id": "sperm"})

    # Same prep as score_tracks() (evaluate.py's own scoring path) --
    # interpolates small gaps in both, so "tracker failure" here uses the
    # exact same track-continuity convention DET/TRA/MOTA/IDF1/HOTA do.
    gt = _prepare(gt)
    pred = _prepare(pred)
    if apply_stationary_filter:
        gt = filter_stationary(gt)
        pred = filter_stationary(pred)

    if gt.empty or pred.empty:
        raise ValueError(f"{tracker_name}: empty after prep/filtering (gt={len(gt)} rows, "
                          f"pred={len(pred)} rows) -- nothing to analyze")

    events = detect_crossovers(gt, threshold=crossover_threshold)
    if min_likelihood > 0:
        events = events[events["likelihood"] >= min_likelihood].reset_index(drop=True)

    gt_to_pred = build_gt_to_pred_map(pred, gt, cutoff=match_cutoff)
    gt_frames_by_id = {int(sid): set(g["frame"].astype(int)) for sid, g in gt.groupby("sperm")}

    switch_frames_by_id = {sid: find_switch_frames(gt_to_pred.get(sid, {})) for sid in gt_frames_by_id}
    drop_frames_by_id = {
        sid: find_drop_frames(gt_to_pred.get(sid, {}), gt_frames_by_id[sid]) for sid in gt_frames_by_id
    }
    failure_frames_by_id = {
        sid: switch_frames_by_id[sid] | drop_frames_by_id[sid] for sid in gt_frames_by_id
    }
    in_crossover_by_id = mark_crossover_frames(events, gt_frames_by_id, window_buffer)

    rows = []
    for _, ev in events.iterrows():
        a, b = int(ev["sperm_a"]), int(ev["sperm_b"])
        start = int(ev["window_start_frame"]) - window_buffer
        end = int(ev["window_end_frame"]) + window_buffer
        a_failed = any(start <= f <= end for f in failure_frames_by_id.get(a, ()))
        b_failed = any(start <= f <= end for f in failure_frames_by_id.get(b, ()))
        rows.append({
            "tracker": tracker_name, "sperm_a": a, "sperm_b": b,
            "window_start_frame": ev["window_start_frame"], "window_end_frame": ev["window_end_frame"],
            "likelihood": ev["likelihood"], "min_distance_px": ev["min_distance_px"],
            "segments_cross": ev["segments_cross"], "position_swap": ev["position_swap"],
            "sperm_a_failed": a_failed, "sperm_b_failed": b_failed,
            "either_failed": a_failed or b_failed,
        })
    events_out = pd.DataFrame(rows)

    summary = {"tracker": tracker_name, "n_crossover_events": len(events_out)}
    summary["n_events_with_failure"] = int(events_out["either_failed"].sum()) if len(events_out) else 0
    summary["event_failure_rate"] = (
        float(events_out["either_failed"].mean()) if len(events_out) else float("nan")
    )
    summary.update(frame_level_summary(gt_frames_by_id, failure_frames_by_id, in_crossover_by_id))
    return events_out, summary


def _parse_pred_arg(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"--pred must be NAME=PATH, got {value!r}")
    name, path = value.split("=", 1)
    return name, path


def _cli():
    parser = argparse.ArgumentParser(
        description="Does tracker failure (ID switches/drops) concentrate at crossover events?"
    )
    parser.add_argument("--gt", required=True, help="Ground-truth canonical CSV")
    parser.add_argument("--pred", action="append", required=True, type=_parse_pred_arg, metavar="NAME=PATH",
                         help="Repeatable, e.g. --pred trackpy=... --pred yolo_bytetrack=... "
                              "(a bootstrap_label.py *_bootstrap.csv is the right kind of file)")
    parser.add_argument("--crossover-threshold", type=float, default=DEFAULT_CROSSOVER_THRESHOLD,
                         help="Proximity threshold in px for crossover_detector.py (default: 25)")
    parser.add_argument("--match-cutoff", type=float, default=DEFAULT_MATCH_CUTOFF,
                         help="Pred<->gt Hungarian-match distance cutoff in px (default: 10, "
                              "matches tracking_metrics.py)")
    parser.add_argument("--window-buffer", type=int, default=DEFAULT_WINDOW_BUFFER,
                         help="Frames of slack around each crossover window (default: 5)")
    parser.add_argument("--min-likelihood", type=float, default=0.0,
                         help="Only analyze crossover events at or above this likelihood")
    parser.add_argument("--no-stationary-filter", action="store_false", dest="apply_stationary_filter",
                         default=True, help="Include near-motionless sperm (default: filtered out, "
                                             "matching evaluate.py's convention)")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.gt).parent / "crossover_impact"
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for name, path in args.pred:
        print(f"\n=== {name} ({path}) ===")
        events, summary = crossover_impact(
            args.gt, path, tracker_name=name,
            crossover_threshold=args.crossover_threshold, match_cutoff=args.match_cutoff,
            window_buffer=args.window_buffer, min_likelihood=args.min_likelihood,
            apply_stationary_filter=args.apply_stationary_filter,
        )
        events_path = out_dir / f"{name}_crossover_events.csv"
        events.to_csv(events_path, index=False)
        for k, v in summary.items():
            print(f"  {k}: {v}")
        print(f"  Wrote per-event table: {events_path}")
        summaries.append(summary)

    comparison = pd.DataFrame(summaries)
    comparison_path = out_dir / "comparison_summary.csv"
    comparison.to_csv(comparison_path, index=False)
    print(f"\n=== Comparison across trackers (relative_risk = crossover-window failure rate "
          f"/ baseline failure rate elsewhere in the same video; higher = worse at crossovers) ===")
    print(comparison[["tracker", "n_crossover_events", "event_failure_rate",
                       "crossover_failure_rate", "baseline_failure_rate", "relative_risk"]]
          .to_string(index=False))
    print(f"\nWrote comparison: {comparison_path}")


if __name__ == "__main__":
    _cli()
