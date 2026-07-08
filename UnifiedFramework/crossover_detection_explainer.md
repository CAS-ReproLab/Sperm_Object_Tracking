# Crossover Detection in Multi-Object Tracking

## Why it matters

In densely-packed multi-object tracking scenarios, two objects can swim/move so close to each
other that their trajectories physically cross. This creates two problems:

1. **Tracking failure** — a tracker that uses greedy nearest-neighbour matching (e.g. TrackPy)
   may swap the IDs of the two objects at the crossing point, even though both objects continue
   in their original direction. The result is two tracks that correctly cover the combined
   trajectory but are attributed to the wrong identities after the crossing.

2. **Metric inflation** — standard multi-object tracking metrics (MOTA, IDF1, HOTA, etc.)
   penalise the identity swap as two independent errors and give no credit to the fact that
   the detector correctly found both objects at every frame. This makes metrics look worse than
   they really are *for the task that matters* (finding and following the objects).

The crossover detector described here lets you:

- Identify which object pairs are at risk of causing this kind of error.
- Compute metrics *restricted to* those high-risk trajectories so the numbers tell you
  specifically how well the tracker handles the hard cases.

---

## Input format

The detector works on any CSV tracking file with four required columns:

| Column  | Type  | Meaning                                      |
|---------|-------|----------------------------------------------|
| `frame` | int   | Frame index (0-based)                        |
| `x`     | float | Horizontal position of the object centroid   |
| `y`     | float | Vertical position of the object centroid     |
| `sperm` | int   | Object (track) ID                            |

The naming convention is domain-specific ("sperm") but the algorithm is completely general.

---

## Three geometric signals

A crossover is detected by combining three independent geometric tests on every pair of
objects, evaluated over every frame in which both objects appear. Each test produces a
boolean or continuous value that contributes to a final likelihood score.

### Signal 1 — Segment intersection

Between consecutive frames `t` and `t+1`, each object sweeps out a line segment:

```
A: (x_a[t], y_a[t]) → (x_a[t+1], y_a[t+1])
B: (x_b[t], y_b[t]) → (x_b[t+1], y_b[t+1])
```

If those two segments **properly cross** (i.e. one endpoint of each segment lies on opposite
sides of the other segment), the objects must have passed through each other's trajectory
between `t` and `t+1`. This is the strongest evidence of a crossover.

The test uses the 2-D signed cross product (no floating-point issues from explicit line
parameterisation):

```python
def _cross2d(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

def segments_intersect(p1, p2, q1, q2):
    d1 = _cross2d(q1, q2, p1)
    d2 = _cross2d(q1, q2, p2)
    d3 = _cross2d(p1, p2, q1)
    d4 = _cross2d(p1, p2, q2)
    return (
        ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and
        ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0))
    )
```

Collinear and endpoint-touching cases are deliberately classified as **no intersection** to
avoid false positives when objects are momentarily stationary or nearly parallel.

### Signal 2 — Position swap (cross-product sign change)

Even when the trajectory segments themselves don't cleanly intersect (e.g. the frame rate is
too low to catch the exact crossing moment), the *relative lateral position* of B with respect
to A's direction of travel reverses during the close-approach window.

For each end of the proximity window (see below), compute:

```
side(t) = sign( cross( v_A(t), pos_B(t) − pos_A(t) ) )
```

where `v_A(t)` is A's instantaneous velocity vector and `cross` is the 2-D cross product.
A sign change (`side_entry ≠ side_exit`) means B moved from one side of A's path to the other
— a reliable indicator of a crossover even when the exact intersection frame was missed.

### Signal 3 — Proximity

How close the two objects came (in pixels) at the point of minimum separation. This is not
binary; it feeds into the likelihood score as a continuous 0→1 value:

```python
proximity_score = max(0.0, 1.0 - min_distance / threshold)
```

`threshold` is a user-supplied pixel radius (default 25 px). An object pair that never comes
within `threshold` pixels is ignored entirely.

---

## Proximity windows

Rather than evaluating every frame independently, pairs are first filtered to **proximity
windows** — contiguous runs of frames in which the inter-object distance is below `threshold`.
This groups related frames into a single event:

```python
def _find_proximity_windows(frames, distances, threshold):
    windows = []
    in_window, start = False, 0
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
```

Each window becomes one *event* (one row in the output). A pair can produce multiple events
(e.g. they approach twice in different parts of the video).

---

## Likelihood scoring

Each event is assigned a likelihood in [0, 1] by combining the three signals:

| Condition                          | Likelihood formula                    | Range   |
|------------------------------------|---------------------------------------|---------|
| Segment crossing detected          | `0.6 + 0.4 × proximity_score`        | 0.6–1.0 |
| Position swap (no crossing)        | `0.35 + 0.35 × proximity_score`      | 0.35–0.70 |
| Only proximity (no crossing, no swap) | `0.3 × proximity_score`           | 0.0–0.30 |

```python
if crossing:
    likelihood = 0.6 + 0.4 * proximity_score
elif swap:
    likelihood = 0.35 + 0.35 * proximity_score
else:
    likelihood = 0.3 * proximity_score
```

The tier structure encodes our confidence hierarchy:
- A literal segment intersection is definitive evidence → baseline 0.6 guaranteed.
- A position swap without an observable crossing is strong but not certain → baseline 0.35.
- Pure proximity could just be two nearby objects — low confidence unless they were very close.

---

## Output per event

Each event record contains:

| Field                  | Meaning                                                       |
|------------------------|---------------------------------------------------------------|
| `sperm_a`, `sperm_b`   | IDs of the two objects involved                               |
| `window_start_frame`   | First frame of the proximity window                           |
| `window_end_frame`     | Last frame of the proximity window                            |
| `closest_frame`        | Frame at minimum separation                                   |
| `min_distance_px`      | Minimum pixel distance between the two objects                |
| `segments_cross`       | Boolean — did the trajectory segments intersect?              |
| `crossing_frame`       | Frame where the intersection was detected (or `NaN`)          |
| `position_swap`        | Boolean — did a side-reversal occur across the window?        |
| `approach_angle_deg`   | Angle between the two velocity vectors at closest approach    |
| `midpoint_x/y`         | Spatial midpoint between the two centroids at closest frame   |
| `likelihood`           | Crossover likelihood score in [0, 1]                          |

---

## Handling duplicate detections

Ground-truth data sometimes has duplicate frame entries for the same object ID. Averaging them
before analysis prevents shape mismatches when aligning the trajectories of two objects:

```python
trajectories = {
    sid: (
        df[df['sperm'] == sid]
        .groupby('frame', as_index=False)[['x', 'y']]
        .mean()
        .set_index('frame')
    )
    for sid in sperm_ids
}
```

---

## Complexity

The algorithm runs over **all pairs** of objects, so the worst-case complexity is
O(N² × T) where N is the number of tracks and T is the number of frames. For typical
datasets (80 objects, 200 frames) this is fast enough to run interactively. For very large
datasets (N > 500), a spatial index (KD-tree per frame) could be used to pre-filter pairs
that never come within `threshold` pixels, reducing the effective pair count significantly.

---

## Video visualisation

The detector can overlay coloured **X markers** on the source video at the midpoint location
of each crossover event for every frame in the proximity window:

- **Red** (likelihood ≥ 0.7) — high confidence crossover
- **Orange** (0.4 ≤ likelihood < 0.7) — medium confidence
- **Yellow** (likelihood < 0.4) — low confidence / proximity only

The X is drawn larger on the exact frame where the segment intersection was detected.
Optional text labels show the two object IDs and the likelihood score.

```python
# Core loop structure
cap = cv.VideoCapture(video_path)
writer = cv.VideoWriter(output_path, ...)
frame_map = _build_frame_event_map(events_df)  # frame -> [events]

while True:
    ret, frame = cap.read()
    for ev in frame_map.get(frame_num, []):
        cx, cy = int(ev['midpoint_x']), int(ev['midpoint_y'])
        color = _event_color_bgr(ev['likelihood'])
        _draw_x(frame, cx, cy, color=color)
    writer.write(frame)
```

---

## Integrating crossover detection with tracking metrics

This is the conceptually trickiest part, because **ground-truth (GT) and predicted (pred)
track IDs are completely independent label spaces**. A GT sperm with ID 7 has nothing to do
with the predicted sperm with ID 7 — they are just coincidental integers. You cannot simply
filter by "sperm 7" in both DataFrames.

### The bridging problem

1. Detect crossovers in the **GT** data → get a set of GT sperm IDs involved.
2. Determine *which predicted IDs* correspond to those GT sperm across the full video.
3. Filter **both** DataFrames to those ID sets and re-run metrics on the subsets.

### Step 2 in detail: spatial matching with the Hungarian algorithm

For each video frame, run a **Hungarian assignment** between GT centroids and predicted
centroids using Euclidean distance as the cost. This creates a per-frame list of
`(gt_id → pred_id)` pairs. Collect every `pred_id` that was ever matched to a GT sperm in
the crossover set:

```python
# Build full-data trajectory correspondence
traj_full = makeTrajectoryData(pred, gt)

pred_crossover_ids = set()
for frame_gt_ids, frame_pred_ids in zip(traj_full['mapped_ref'],
                                         traj_full['mapped_comp']):
    for gt_id, pred_id in zip(frame_gt_ids, frame_pred_ids):
        if gt_id in gt_crossover_ids:
            pred_crossover_ids.add(pred_id)
```

The key design choice is to run the Hungarian matching on the **full, unfiltered** dataset
first. This gives the globally optimal GT↔pred ID mapping. If you subset the data first and
then match, the assignment changes (nearby non-crossover sperm fill in the gaps), leading to
incorrect ID translation.

### Step 3: filtered metrics

Once you have `gt_crossover_ids` and `pred_crossover_ids`:

```python
gt_cross  = gt[gt['sperm'].isin(gt_crossover_ids)]
pred_cross = pred[pred['sperm'].isin(pred_crossover_ids)]

# Re-run the full metrics pipeline on the filtered subsets
gt_tracks   = makeTrackData(gt_cross)
pred_tracks = makeTrackData(pred_cross)
traj_cross  = makeTrajectoryData(pred_cross, gt_cross)
traj_cross  = appendMergedTrajectory(gt_tracks, pred_tracks, traj_cross)
results     = computeMetricsFromTracks(gt_tracks, pred_tracks, traj_cross)
```

The resulting metrics answer: *"How well does the tracker handle the specific trajectories
that are at risk of crossover-induced identity swaps?"*

### Displaying side-by-side

The final output table shows three columns — unfiltered, stationary-filtered, and
crossover-only — so you can directly compare performance on the overall population versus
the hard cases:

```
Metric     Unfiltered  Filtered  Crossover
GT Sperm           80        66         12
DET          0.8210    0.8503     0.7614
LNK          0.7934    0.8201     0.6822
TRA          0.7501    0.7890     0.6103
MOTA         0.6812    0.7234     0.5901
IDF1         0.7102    0.7489     0.5534
HOTA         0.6934    0.7301     0.5212
```

---

## Command-line usage

```bash
# Detect crossovers and print a summary
python crossover_detector.py ground_truth.csv --summary

# Also produce an annotated video
python crossover_detector.py ground_truth.csv \
    --video input.mp4 --min-likelihood 0.4 --video-out annotated.mp4

# Evaluate tracker metrics including a crossover-specific column
python metrics.py \
    --groundtruth ground_truth.csv \
    --prediction tracker_output.csv \
    --crossover \
    --crossover-likelihood 0.4 \
    --crossover-threshold 25
```

---

## Key configurable parameters

| Parameter            | Default | Effect                                                              |
|----------------------|---------|---------------------------------------------------------------------|
| `threshold`          | 25 px   | Proximity radius. Increase for sparse/slow data, decrease for dense/fast. |
| `min_likelihood`     | 0.4     | Minimum score for an event to be reported or used in metrics.       |
| `--crossover-threshold` | 25 px | Same as above, passed from the metrics CLI.                       |

---

## Files

| File                     | Role                                                                      |
|--------------------------|---------------------------------------------------------------------------|
| `crossover_detector.py`  | Self-contained detector: geometry helpers, detection loop, visualisation. |
| `metrics.py`             | Tracking metrics pipeline; `computeCrossoverMetrics()` bridges the two.   |
