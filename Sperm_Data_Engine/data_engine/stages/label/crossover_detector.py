"""
crossover_detector.py

Detects potential crossover events between sperm cells in ground truth tracking data.

A crossover occurs when two sperm swimming past each other — their spatial paths cross
or their relative ordering reverses — which can confuse tracking algorithms and inflate
error metrics that don't account for this case.

Three geometric signals are combined into a likelihood score per event:
  - Segment intersection: trajectory segments of A and B literally cross between frames
  - Position swap: B changes sides relative to A's direction of travel across the window
  - Proximity: how close (in pixels) the pair came at minimum

Usage:
    python crossover_detector.py path/to/ground_truth.csv [options]

    python crossover_detector.py SchmidtDataset/video_P001_corrected.csv
    python crossover_detector.py SchmidtDataset/video_P001_corrected.csv --threshold 20 --keep-only
    python crossover_detector.py SchmidtDataset/video_P001_corrected.csv --out events.csv --summary
    python crossover_detector.py SchmidtDataset/video_P001_corrected.csv --video myvideo.mp4 --summary
    python crossover_detector.py SchmidtDataset/video_P001_corrected.csv --video myvideo.mp4 --min-likelihood 0.4 --video-out annotated.mp4
"""

import argparse
import sys
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _cross2d(o, a, b):
    """Signed 2-D cross product of vectors (a - o) and (b - o)."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def segments_intersect(p1, p2, q1, q2):
    """Return True if open line-segment p1→p2 properly intersects q1→q2."""
    d1 = _cross2d(q1, q2, p1)
    d2 = _cross2d(q1, q2, p2)
    d3 = _cross2d(p1, p2, q1)
    d4 = _cross2d(p1, p2, q2)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True

    # Collinear / endpoint cases — treat as no intersection (avoids false positives
    # when sperm are momentarily stationary or at identical positions)
    return False


def approach_angle_deg(v_a, v_b):
    """Angle in degrees between two velocity vectors. Returns 0 if either is zero."""
    na, nb = np.linalg.norm(v_a), np.linalg.norm(v_b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    cos_theta = np.dot(v_a, v_b) / (na * nb)
    return float(np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0))))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_tracks(filepath, keep_only=False):
    """
    Load a ground-truth tracking CSV.

    Expected columns: frame, x, y, sperm  (case-insensitive).
    Optional: keep  — if present and keep_only=True, rows with keep==0 are dropped.
    """
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip().str.lower()

    required = {'frame', 'x', 'y', 'sperm'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    if keep_only and 'keep' in df.columns:
        df = df[df['keep'] != 0].copy()

    df = df.sort_values(['sperm', 'frame']).reset_index(drop=True)
    df['frame'] = df['frame'].astype(int)
    df['sperm'] = df['sperm'].astype(int)
    return df


# ---------------------------------------------------------------------------
# Per-pair analysis
# ---------------------------------------------------------------------------

def _find_proximity_windows(frames, distances, threshold):
    """
    Return a list of (start_idx, end_idx) index pairs (inclusive) into `frames`
    where distance < threshold for a contiguous run of at least 1 frame.
    """
    windows = []
    in_window = False
    start = 0
    for i, d in enumerate(distances):
        if d < threshold:
            if not in_window:
                start = i
                in_window = True
        else:
            if in_window:
                windows.append((start, i - 1))
                in_window = False
    if in_window:
        windows.append((start, len(distances) - 1))
    return windows


def analyze_pair(id_a, id_b, traj_a, traj_b, threshold):
    """
    Compute all crossover events for a single sperm pair.

    Parameters
    ----------
    id_a, id_b : int
        Sperm IDs.
    traj_a, traj_b : pd.DataFrame
        Sub-dataframes (already filtered to one sperm each), indexed by frame.
    threshold : float
        Proximity threshold in pixels.

    Returns
    -------
    list of dict — one dict per proximity window that qualifies.
    """
    common_frames = sorted(set(traj_a.index) & set(traj_b.index))
    if len(common_frames) < 2:
        return []

    pa = traj_a.loc[common_frames, ['x', 'y']].values  # shape (N, 2)
    pb = traj_b.loc[common_frames, ['x', 'y']].values

    distances = np.linalg.norm(pa - pb, axis=1)

    windows = _find_proximity_windows(common_frames, distances, threshold)
    if not windows:
        return []

    events = []
    dist_arr = distances  # same order as common_frames

    for (wi_start, wi_end) in windows:
        window_frames = common_frames[wi_start: wi_end + 1]
        pa_w = pa[wi_start: wi_end + 1]
        pb_w = pb[wi_start: wi_end + 1]
        dist_w = dist_arr[wi_start: wi_end + 1]

        min_dist_idx = int(np.argmin(dist_w))
        min_dist = float(dist_w[min_dist_idx])
        closest_frame = window_frames[min_dist_idx]

        # --- Signal 1: any segment intersection within the window? ---
        crossing = False
        crossing_frame = None
        for ti in range(len(window_frames) - 1):
            if segments_intersect(pa_w[ti], pa_w[ti + 1], pb_w[ti], pb_w[ti + 1]):
                crossing = True
                crossing_frame = window_frames[ti]
                break

        # --- Signal 2: position swap ---
        # Does sperm B change sides relative to sperm A's direction of travel
        # between the entry and exit of the proximity window?
        #
        # side(t) = sign of cross( v_a(t), pos_b(t) - pos_a(t) )
        # where v_a is A's instantaneous velocity at that end of the window.
        # A sign change from entry to exit means B passed through A's trajectory.
        swap = False
        if len(window_frames) >= 2:
            # Entry: use first two points of A for velocity estimate
            v_entry = pa_w[1] - pa_w[0] if len(pa_w) > 1 else np.array([1.0, 0.0])
            rel_entry = pb_w[0] - pa_w[0]
            side_entry = np.cross(v_entry, rel_entry)

            # Exit: use last two points
            v_exit = pa_w[-1] - pa_w[-2] if len(pa_w) > 1 else v_entry
            rel_exit = pb_w[-1] - pa_w[-1]
            side_exit = np.cross(v_exit, rel_exit)

            if side_entry != 0 and side_exit != 0:
                swap = np.sign(side_entry) != np.sign(side_exit)

        # --- Signal 3: approach angle at closest point ---
        angle = 0.0
        if min_dist_idx + 1 < len(window_frames):
            v_a = pa_w[min_dist_idx + 1] - pa_w[min_dist_idx]
            v_b = pb_w[min_dist_idx + 1] - pb_w[min_dist_idx]
        elif min_dist_idx > 0:
            v_a = pa_w[min_dist_idx] - pa_w[min_dist_idx - 1]
            v_b = pb_w[min_dist_idx] - pb_w[min_dist_idx - 1]
        else:
            v_a = v_b = np.array([0.0, 0.0])
        angle = approach_angle_deg(v_a, v_b)

        # --- Likelihood score ---
        # Proximity contribution alone → 0–0.3
        # + position swap → adds up to 0.4 more
        # + segment crossing → adds up to 0.4 more (overlapping with swap to reach 1.0)
        proximity_score = max(0.0, 1.0 - min_dist / threshold)

        if crossing:
            likelihood = 0.6 + 0.4 * proximity_score
        elif swap:
            likelihood = 0.35 + 0.35 * proximity_score
        else:
            likelihood = 0.3 * proximity_score

        midpoint_x = float((pa_w[min_dist_idx, 0] + pb_w[min_dist_idx, 0]) / 2)
        midpoint_y = float((pa_w[min_dist_idx, 1] + pb_w[min_dist_idx, 1]) / 2)

        events.append({
            'sperm_a': id_a,
            'sperm_b': id_b,
            'window_start_frame': window_frames[0],
            'window_end_frame': window_frames[-1],
            'closest_frame': closest_frame,
            'min_distance_px': round(min_dist, 2),
            'segments_cross': crossing,
            'crossing_frame': crossing_frame,
            'position_swap': swap,
            'approach_angle_deg': round(angle, 1),
            'midpoint_x': round(midpoint_x, 2),
            'midpoint_y': round(midpoint_y, 2),
            'likelihood': round(likelihood, 3),
        })

    return events


# ---------------------------------------------------------------------------
# Main detection loop
# ---------------------------------------------------------------------------

def detect_crossovers(df, threshold=25.0):
    """
    Detect crossover events across all sperm pairs in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Track data with columns: frame, x, y, sperm.
    threshold : float
        Proximity threshold in pixels below which a pair is considered at-risk.

    Returns
    -------
    pd.DataFrame of crossover events, sorted by likelihood descending.
    """
    sperm_ids = sorted(df['sperm'].unique())

    # Build per-sperm trajectory lookups indexed by frame for fast access.
    # Average duplicate detections in the same frame (rare but possible in raw GT).
    trajectories = {
        sid: (
            df[df['sperm'] == sid]
            .groupby('frame', as_index=False)[['x', 'y']]
            .mean()
            .set_index('frame')
        )
        for sid in sperm_ids
    }

    all_events = []
    pairs = list(combinations(sperm_ids, 2))
    total = len(pairs)

    for i, (id_a, id_b) in enumerate(pairs):
        if i % 500 == 0 and total > 1000:
            print(f"  Analyzing pair {i}/{total}...", end='\r', flush=True)
        events = analyze_pair(id_a, id_b, trajectories[id_a], trajectories[id_b], threshold)
        all_events.extend(events)

    if total > 1000:
        print()  # newline after progress

    if not all_events:
        return pd.DataFrame(columns=[
            'sperm_a', 'sperm_b', 'window_start_frame', 'window_end_frame',
            'closest_frame', 'min_distance_px', 'segments_cross', 'crossing_frame',
            'position_swap', 'approach_angle_deg', 'midpoint_x', 'midpoint_y', 'likelihood'
        ])

    events_df = pd.DataFrame(all_events)
    return events_df.sort_values('likelihood', ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def pair_summary(events_df):
    """
    Collapse per-window events into one row per sperm pair, keeping the
    maximum likelihood event and aggregating counts.
    """
    if events_df.empty:
        return events_df

    summary = (
        events_df
        .groupby(['sperm_a', 'sperm_b'])
        .agg(
            n_proximity_windows=('likelihood', 'count'),
            max_likelihood=('likelihood', 'max'),
            any_crossing=('segments_cross', 'any'),
            any_swap=('position_swap', 'any'),
            min_distance_px=('min_distance_px', 'min'),
        )
        .reset_index()
        .sort_values('max_likelihood', ascending=False)
        .reset_index(drop=True)
    )
    return summary


def print_summary(events_df, top_n=20):
    """Print a human-readable summary to stdout."""
    total_events = len(events_df)
    high = (events_df['likelihood'] >= 0.7).sum()
    medium = ((events_df['likelihood'] >= 0.4) & (events_df['likelihood'] < 0.7)).sum()
    low = (events_df['likelihood'] < 0.4).sum()

    print(f"\n=== Crossover Detection Summary ===")
    print(f"Total proximity events : {total_events}")
    print(f"  High likelihood (>=0.7)    : {high}")
    print(f"  Medium likelihood (0.4-0.7): {medium}")
    print(f"  Low likelihood (<0.4)      : {low}")
    print(f"  Segment crossings detected: {events_df['segments_cross'].sum()}")
    print(f"  Position swaps detected   : {events_df['position_swap'].sum()}")

    if total_events == 0:
        return

    print(f"\nTop {min(top_n, total_events)} events by likelihood:")
    cols = ['sperm_a', 'sperm_b', 'closest_frame', 'min_distance_px',
            'segments_cross', 'position_swap', 'approach_angle_deg', 'likelihood']
    print(events_df[cols].head(top_n).to_string(index=False))


# ---------------------------------------------------------------------------
# Video visualization
# ---------------------------------------------------------------------------

def _event_color_bgr(likelihood):
    """Return BGR color for a crossover marker based on likelihood."""
    if likelihood >= 0.7:
        return (0, 0, 255)      # red   — high confidence
    elif likelihood >= 0.4:
        return (0, 128, 255)    # orange — medium confidence
    else:
        return (0, 220, 220)    # yellow — low confidence


def _draw_x(img, cx, cy, size=14, color=(0, 0, 255), thickness=2):
    """Draw an X marker centered at (cx, cy)."""
    import cv2 as cv
    s = size // 2
    cv.line(img, (cx - s, cy - s), (cx + s, cy + s), color, thickness)
    cv.line(img, (cx + s, cy - s), (cx - s, cy + s), color, thickness)


def _draw_label(img, text, cx, cy, color, font_scale=0.45, thickness=1):
    import cv2 as cv
    font = cv.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv.getTextSize(text, font, font_scale, thickness)
    tx = max(0, cx - tw // 2)
    ty = max(th + baseline, cy - 16)
    # Dark background for readability
    cv.rectangle(img, (tx - 1, ty - th - 1), (tx + tw + 1, ty + baseline), (0, 0, 0), -1)
    cv.putText(img, text, (tx, ty), font, font_scale, color, thickness, cv.LINE_AA)


def _build_frame_event_map(events_df):
    """
    Build a dict mapping frame_number -> list of event dicts to draw on that frame.
    Each event is shown for every frame in [window_start_frame, window_end_frame].
    """
    frame_map = defaultdict(list)
    for _, row in events_df.iterrows():
        for f in range(int(row['window_start_frame']), int(row['window_end_frame']) + 1):
            frame_map[f].append(row)
    return frame_map


def visualize_crossovers(video_path, events_df, output_path=None, show_labels=True):
    """
    Write an annotated video with crossover markers overlaid.

    Parameters
    ----------
    video_path : str
        Path to the source video file.
    events_df : pd.DataFrame
        Crossover events from detect_crossovers(), already filtered to desired
        min_likelihood.
    output_path : str or None
        Where to save the annotated video. Defaults to
        <video_path stem>_crossovers.mp4.
    show_labels : bool
        Whether to draw sperm-ID and likelihood text next to each marker.

    Returns
    -------
    str — path to the saved video.
    """
    try:
        import cv2 as cv
    except ImportError:
        raise ImportError("opencv-python is required for visualization: pip install opencv-python")

    if output_path is None:
        base = video_path.rsplit('.', 1)[0]
        output_path = base + '_crossovers.mp4'

    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    width  = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv.CAP_PROP_FPS) or 9.0
    total  = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    writer = cv.VideoWriter(
        output_path,
        cv.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height),
    )

    frame_map = _build_frame_event_map(events_df)

    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        events_here = frame_map.get(frame_num, [])
        for ev in events_here:
            cx = int(round(ev['midpoint_x']))
            cy = int(round(ev['midpoint_y']))
            color = _event_color_bgr(ev['likelihood'])

            # Larger X on the frame where segments actually cross
            is_crossing_frame = (
                ev['segments_cross'] and
                not pd.isna(ev['crossing_frame']) and
                frame_num == int(ev['crossing_frame'])
            )
            x_size = 20 if is_crossing_frame else 14
            _draw_x(frame, cx, cy, size=x_size, color=color, thickness=2)

            if show_labels:
                label = f"A{int(ev['sperm_a'])}/B{int(ev['sperm_b'])} L={ev['likelihood']:.2f}"
                _draw_label(frame, label, cx, cy, color)

        writer.write(frame)
        frame_num += 1

    cap.release()
    writer.release()

    print(f"Annotated video saved to: {output_path}  ({frame_num} frames at {fps:.1f} fps)")
    return output_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect crossover events between sperm in ground-truth tracking data."
    )
    parser.add_argument('input', help="Path to ground-truth CSV file")
    parser.add_argument(
        '--threshold', type=float, default=25.0,
        help="Proximity threshold in pixels (default: 25). Pairs closer than this "
             "at any frame are analyzed for crossover."
    )
    parser.add_argument(
        '--out', default=None,
        help="Path to save per-event results CSV (default: <input>_crossovers.csv)"
    )
    parser.add_argument(
        '--summary-out', default=None,
        help="Path to save per-pair summary CSV (optional)"
    )
    parser.add_argument(
        '--keep-only', action='store_true',
        help="If the CSV has a 'keep' column, filter to rows where keep != 0"
    )
    parser.add_argument(
        '--summary', action='store_true',
        help="Print summary statistics to stdout"
    )
    parser.add_argument(
        '--min-likelihood', type=float, default=0.0,
        help="Only include events at or above this likelihood in the output CSV "
             "and (if --video is given) in the video overlay"
    )
    parser.add_argument(
        '--video', default=None,
        help="Path to the source video file. If provided, an annotated video is written "
             "with red X markers at crossover locations."
    )
    parser.add_argument(
        '--video-out', default=None,
        help="Path for the annotated output video (default: <video>_crossovers.mp4)"
    )
    parser.add_argument(
        '--no-labels', action='store_true',
        help="Suppress sperm-ID and likelihood text labels in the video overlay"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading tracks from: {args.input}")
    df = load_tracks(args.input, keep_only=args.keep_only)
    n_sperm = df['sperm'].nunique()
    n_frames = df['frame'].nunique()
    print(f"  {n_sperm} sperm tracks across {n_frames} frames")
    print(f"  Proximity threshold: {args.threshold} px")

    print("Analyzing pairs for crossovers...")
    events = detect_crossovers(df, threshold=args.threshold)

    if args.min_likelihood > 0:
        events = events[events['likelihood'] >= args.min_likelihood].reset_index(drop=True)

    if args.summary:
        print_summary(events)

    # Save per-event CSV
    if args.out is None:
        base = args.input.rsplit('.', 1)[0]
        args.out = base + '_crossovers.csv'

    events.to_csv(args.out, index=False)
    print(f"\nPer-event results saved to: {args.out}")

    # Save per-pair summary CSV
    if args.summary_out:
        pair_summary(events).to_csv(args.summary_out, index=False)
        print(f"Per-pair summary saved to: {args.summary_out}")

    # Annotated video
    if args.video:
        print(f"\nRendering annotated video from: {args.video}")
        visualize_crossovers(
            video_path=args.video,
            events_df=events,
            output_path=args.video_out,
            show_labels=not args.no_labels,
        )

    return events


if __name__ == '__main__':
    main()
