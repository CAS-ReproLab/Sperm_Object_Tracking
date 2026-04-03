
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# SECTION: Detection-level merge flagging
#   Ported from Cameron's detect.py → detect_crossovers()
#
#   Operates on the segmented DataFrame (one row per detection per frame).
#   For each detection it gathers a spatiotemporal neighborhood of past
#   detections within `radius` pixels and `window` frames, then computes a
#   robust z-score and area ratio to flag anomalously large blobs that likely
#   represent two overlapping sperm.
# ─────────────────────────────────────────────────────────────────────────────

_EPS = 1e-9  # small epsilon used throughout merge detection


def flagMerges_detection(df,
                         window=3,
                         radius=15.0,
                         z_thresh=3.0,
                         ratio_thresh=1.6,
                         z_cap=25.0):
    """
    Detection-level merge/crossover flagging.

    Adds four columns to `df` in-place:
        area_roll_med   – rolling neighborhood median area
        area_mad        – median absolute deviation of that neighborhood
        area_z          – robust z-score  (A - med) / (1.4826 * MAD)
        area_ratio      – A / med
        merge_flag_det  – 1 if z >= z_thresh AND ratio >= ratio_thresh

    Parameters
    ----------
    df : DataFrame
        Output of segmentCells(); must have columns x, y, frame, area.
    window : int
        Number of past frames to include in the spatiotemporal neighborhood.
    radius : float
        Spatial radius (pixels) for neighborhood membership.
    z_thresh : float
        Robust z-score threshold for merge flagging.
    ratio_thresh : float
        Area-ratio threshold for merge flagging.
    z_cap : float
        Upper bound on |z| to prevent blowup when MAD ≈ 0.
    """
    df = df.copy()
    df['area_roll_med'] = np.nan
    df['area_mad'] = np.nan
    df['area_z'] = 0.0
    df['area_ratio'] = 1.0
    df['merge_flag_det'] = 0

    # Build a lightweight history pool as we walk through frames in order.
    # Each entry: {'x', 'y', 'area', 'frame'}
    history = []

    for frame_idx in sorted(df['frame'].unique()):
        frame_rows = df[df['frame'] == frame_idx]

        xs = np.array([h['x'] for h in history], dtype=float)
        ys = np.array([h['y'] for h in history], dtype=float)
        areas = np.array([h['area'] for h in history], dtype=float)
        frames_hist = np.array([h['frame'] for h in history], dtype=int)

        valid_time = frames_hist >= (frame_idx - window)

        for idx, row in frame_rows.iterrows():
            cx, cy = float(row['x']), float(row['y'])
            A = float(row['area'])
            if A <= 0:
                continue

            if len(history) > 0 and valid_time.any():
                dx = xs - cx
                dy = ys - cy
                dist = np.hypot(dx, dy)
                m = (dist <= radius) & valid_time
                neigh = areas[m]
            else:
                neigh = np.array([], dtype=float)

            if neigh.size >= 3:
                med = float(np.median(neigh))
                mad = float(np.median(np.abs(neigh - med)))
                robust_scale = 1.4826 * max(mad, _EPS)
                z = float((A - med) / robust_scale)
                z = float(np.sign(z) * min(abs(z), z_cap))
                ratio = A / max(med, _EPS)
            else:
                med, mad, z, ratio = float(A), 0.0, 0.0, 1.0

            df.at[idx, 'area_roll_med'] = med
            df.at[idx, 'area_mad'] = mad
            df.at[idx, 'area_z'] = z
            df.at[idx, 'area_ratio'] = ratio
            df.at[idx, 'merge_flag_det'] = int(
                (z >= z_thresh) and (ratio >= ratio_thresh))

            history.append({'x': cx, 'y': cy,
                            'area': A, 'frame': frame_idx})

    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION: Track-level merge flagging
#   Ported from Cameron's track.py → _merge_detect() and the annotation loop
#   inside offline_tracking().
#
#   After tracking is complete this function walks each sperm's time-series
#   and computes a per-row area z-score against that track's own history.
#   Final merge flag requires BOTH detection-level AND track-level agreement.
# ─────────────────────────────────────────────────────────────────────────────

def _track_merge_score(A_now, hist,
                       alpha=0.10, beta=10.0, z_cap=10.0):
    """
    Compute robust z-score and area ratio for one detection against its
    track's area history.

    Returns (z, ratio).  Returns (0.0, 1.0) when history is too short.
    """
    if hist and len(hist) >= 3:
        arr = np.asarray(hist, dtype=float)
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med)))
        scale = max(1.4826 * mad, alpha * max(med, 1e-9) + beta)
        z = float((A_now - med) / scale)
        z = float(np.sign(z) * min(abs(z), float(z_cap)))
        ratio = A_now / max(med, 1e-9)
        return z, ratio
    return 0.0, 1.0


def flagMerges_track(df,
                     z_thresh=3.5,
                     ratio_thresh=1.6,
                     hist_len=9,
                     alpha=0.10,
                     beta=10.0,
                     z_cap=10.0):
    """
    Track-level merge/crossover flagging.

    Walks each sperm's detections in frame order, maintaining a rolling area
    history.  Adds columns:
        track_area_z        – per-track robust z-score
        track_area_ratio    – per-track area ratio
        xo_track_flag       – 1 if ratio >= ratio_thresh OR z >= z_thresh
        merge_flag_final    – 1 if BOTH merge_flag_det AND xo_track_flag are 1

    Also annotates xo_parent_id and xo_child_id (nearest-track estimates)
    consumed later by the repair stage.

    Parameters
    ----------
    df : DataFrame
        Output of flagMerges_detection(); must have columns sperm, frame,
        x, y, area, bbox_x, bbox_y, bbox_w, bbox_h, merge_flag_det.
    """
    df = df.copy()
    df['track_area_z'] = 0.0
    df['track_area_ratio'] = 1.0
    df['xo_track_flag'] = 0
    df['merge_flag_final'] = 0
    df['xo_parent_id'] = -1
    df['xo_child_id'] = -1

    for sperm_id, grp in df.groupby('sperm'):
        grp = grp.sort_values('frame')
        area_hist = []

        for idx, row in grp.iterrows():
            bw = float(row['bbox_w'])
            bh = float(row['bbox_h'])
            A_bbox = bw * bh  # bbox area as the area metric for track-level

            z, ratio = _track_merge_score(A_bbox, area_hist,
                                          alpha=alpha, beta=beta, z_cap=z_cap)
            xo_flag = int((ratio >= ratio_thresh) or (z >= z_thresh))

            df.at[idx, 'track_area_z'] = z
            df.at[idx, 'track_area_ratio'] = ratio
            df.at[idx, 'xo_track_flag'] = xo_flag

            # Final merge requires both stages to agree
            det_flag = int(row.get('merge_flag_det', 0))
            df.at[idx, 'merge_flag_final'] = int(det_flag and xo_flag)

            # Rolling history (capped)
            area_hist.append(A_bbox)
            if len(area_hist) > hist_len:
                area_hist = area_hist[-hist_len:]

    # ── Parent / child annotation ───────────────────────────────────────────
    # For each frame with a merge event, tag the nearest other track as parent.
    # For child (continuation after merge), use median velocity extrapolation.
    merge_frames = df[df['merge_flag_final'] == 1]['frame'].unique()

    for frame_idx in merge_frames:
        frame_rows = df[df['frame'] == frame_idx]
        merge_rows = frame_rows[frame_rows['merge_flag_final'] == 1]
        non_merge_rows = frame_rows[frame_rows['merge_flag_final'] == 0]

        for idx, mrow in merge_rows.iterrows():
            mx, my = float(mrow['x']), float(mrow['y'])
            sperm_id = mrow['sperm']

            # ── Parent: nearest other track in the previous frame ──────────
            prev_frame = df[df['frame'] == frame_idx - 1]
            other_prev = prev_frame[prev_frame['sperm'] != sperm_id]
            if not other_prev.empty:
                dists = np.hypot(
                    other_prev['x'].to_numpy(float) - mx,
                    other_prev['y'].to_numpy(float) - my
                )
                best = other_prev.iloc[np.argmin(dists)]
                df.at[idx, 'xo_parent_id'] = int(best['sperm'])

            # ── Child: velocity-extrapolated nearest track in next frame ───
            hist_rows = (df[(df['sperm'] == sperm_id) &
                            (df['frame'] < frame_idx)]
                         .sort_values('frame')
                         .tail(5))
            if len(hist_rows) >= 2:
                dx_med = float(np.median(np.diff(hist_rows['x'].to_numpy(float))))
                dy_med = float(np.median(np.diff(hist_rows['y'].to_numpy(float))))
            else:
                dx_med, dy_med = 0.0, 0.0

            pred_x = mx + dx_med
            pred_y = my + dy_med

            next_frame = df[df['frame'] == frame_idx + 1]
            other_next = next_frame[next_frame['sperm'] != sperm_id]
            if not other_next.empty:
                dists = np.hypot(
                    other_next['x'].to_numpy(float) - pred_x,
                    other_next['y'].to_numpy(float) - pred_y
                )
                best_child = other_next.iloc[np.argmin(dists)]
                df.at[idx, 'xo_child_id'] = int(best_child['sperm'])

    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION: Track repair
#   Ported from Cameron's track_repair.py.
#
#   Two sequential stages:
#     1. Continuity repair – fills gaps in a parent track during a merge
#        segment and links "child" tracks that continue the parent afterward.
#     2. Swap repair – detects and corrects identity swaps caused by crossovers.
#
#   The repair functions expect column names that differ slightly from
#   tracker.py's conventions.  A thin adapter (_to_repair_df / _from_repair_df)
#   handles the rename so the rest of the pipeline is untouched.
# ─────────────────────────────────────────────────────────────────────────────

# ── Column name adapter ──────────────────────────────────────────────────────

def _to_repair_df(df):
    """
    Rename tracker.py columns → repair-module conventions.
        sperm     → track_id
        x, y      → cx, cy   (centroid aliases; bbox columns kept separately)
        bbox_x/y/w/h → x, y, w, h
    """
    d = df.copy()
    d = d.rename(columns={
        'sperm':  'track_id',
        'x':      'cx',
        'y':      'cy',
        'bbox_x': 'x',
        'bbox_y': 'y',
        'bbox_w': 'w',
        'bbox_h': 'h',
    })
    # add repair columns if missing
    if 'repair_flag' not in d.columns:
        d['repair_flag'] = 0
    if 'repair_type' not in d.columns:
        d['repair_type'] = ''
    return d


def _from_repair_df(df):
    """Reverse the rename applied by _to_repair_df."""
    d = df.copy()
    d = d.rename(columns={
        'track_id': 'sperm',
        'cx':       'x',
        'cy':       'y',
        'x':        'bbox_x',
        'y':        'bbox_y',
        'w':        'bbox_w',
        'h':        'bbox_h',
    })
    return d


# ── Internal repair helpers (from track_repair.py, lightly trimmed) ──────────

def _canonical_find(tid, parent_map):
    root = tid
    while parent_map.get(root, root) != root:
        root = parent_map[root]
    cur = tid
    while parent_map.get(cur, cur) != root:
        nxt = parent_map.get(cur, cur)
        parent_map[cur] = root
        cur = nxt
    return root


def _canonical_union(a, b, parent_map):
    ra = _canonical_find(a, parent_map)
    rb = _canonical_find(b, parent_map)
    if ra == rb:
        return
    root = min(ra, rb)
    other = max(ra, rb)
    parent_map[ra] = root
    parent_map[rb] = root
    parent_map[other] = root


def _enumerate_merge_segments(df_tracks):
    """Return list of (track_id, frame_start, frame_end) for every contiguous
    run of merge_flag_final == 1 on any track."""
    segments = []
    if 'merge_flag_final' not in df_tracks.columns:
        return segments

    for tid, grp in df_tracks.groupby('track_id'):
        grp = grp.sort_values('frame')
        merge_mask = grp['merge_flag_final'].fillna(0).astype(bool).to_numpy()
        if not merge_mask.any():
            continue
        frames = grp['frame'].to_numpy()
        idx_merge = np.where(merge_mask)[0]
        start = idx_merge[0]
        prev = idx_merge[0]
        for k in idx_merge[1:]:
            if frames[k] == frames[prev] + 1:
                prev = k
            else:
                segments.append((int(tid), int(frames[start]), int(frames[prev])))
                start = k
                prev = k
        segments.append((int(tid), int(frames[start]), int(frames[prev])))

    segments.sort(key=lambda x: x[1])
    return segments


def _estimate_parent_motion(df_tracks, parent_id, t_start, hist_len):
    hist = df_tracks[(df_tracks['track_id'] == parent_id) &
                     (df_tracks['frame'] < t_start)]
    if hist.shape[0] == 0:
        return None
    hist = hist.sort_values('frame').tail(max(int(hist_len), 2))
    t = hist['frame'].to_numpy(dtype=float)
    cx = hist['cx'].to_numpy(dtype=float)
    cy = hist['cy'].to_numpy(dtype=float)
    if hist.shape[0] < 2:
        return float(cx[-1]), float(cy[-1]), 0.0, 0.0, float(t[-1])
    dt = np.diff(t)
    dt[dt == 0] = 1.0
    vx = np.median(np.diff(cx) / dt)
    vy = np.median(np.diff(cy) / dt)
    return float(cx[-1]), float(cy[-1]), float(vx), float(vy), float(t[-1])


def _link_child_track(df_tracks, parent_id, t_end, params,
                      track_first_frame, child_claimed):
    if not params['enable_child_link']:
        return []
    max_gap = params['child_max_gap']
    max_dist = params['child_max_distance']
    motion = _estimate_parent_motion(df_tracks, parent_id, t_end + 1,
                                     params['motion_hist_len'])
    if motion is None:
        return []
    cx_prev, cy_prev, vx, vy, t_prev = motion
    best_child, best_dist = None, None

    for cid, first_frame in track_first_frame.items():
        if cid == parent_id or cid in child_claimed:
            continue
        if not (t_end < first_frame <= t_end + max_gap):
            continue
        cgrp = df_tracks[df_tracks['track_id'] == cid].sort_values('frame')
        row0 = cgrp.iloc[0]
        t_child = float(row0['frame'])
        cx_c = float(row0['cx'])
        cy_c = float(row0['cy'])
        dt = t_child - t_prev
        xhat = cx_prev + vx * dt
        yhat = cy_prev + vy * dt
        dist = float(np.hypot(cx_c - xhat, cy_c - yhat))
        if dist <= max_dist and (best_dist is None or dist < best_dist):
            best_dist = dist
            best_child = cid

    return [best_child] if best_child is not None else []


def _gap_fill_for_track(df_tracks, tid, t_start, t_end,
                        visible_tid, params, repair_rows, repair_type_suffix):
    mode = params['gap_fill_mode']
    if mode == 'none':
        return
    if (t_end - t_start + 1) > params['gap_fill_max_frames']:
        return

    frames_merge = np.arange(t_start, t_end + 1, dtype=int)
    grp_self = df_tracks[df_tracks['track_id'] == tid]
    grp_vis = df_tracks[df_tracks['track_id'] == visible_tid]

    pre_self = grp_self[grp_self['frame'] < t_start].sort_values('frame')
    post_self = grp_self[grp_self['frame'] > t_end].sort_values('frame')
    has_pre = pre_self.shape[0] > 0
    has_post = post_self.shape[0] > 0

    if mode == 'interp_self' and not (has_pre and has_post):
        mode = 'copy_visible'

    for t in frames_merge:
        if (grp_self['frame'] == t).any():
            continue
        vis_row = grp_vis[grp_vis['frame'] == t].head(1)
        if vis_row.empty and mode == 'copy_visible':
            continue

        if mode == 'copy_visible' and not vis_row.empty:
            r = vis_row.iloc[0]
            cx_new = float(r['cx'])
            cy_new = float(r['cy'])
            x_new = float(r.get('x', cx_new))
            y_new = float(r.get('y', cy_new))
            w_new = float(r.get('w', 0))
            h_new = float(r.get('h', 0))
            area_new = float(r.get('area', 0))
            merge_flag_final = int(r.get('merge_flag_final', 1))
        else:  # interp_self
            r_pre = pre_self.iloc[-1]
            r_post = post_self.iloc[0]
            t_pre = float(r_pre['frame'])
            t_post_val = float(r_post['frame'])
            if t_post_val <= t_pre:
                continue
            alpha_lerp = (t - t_pre) / (t_post_val - t_pre)

            def _lerp(a, b):
                return float((1.0 - alpha_lerp) * a + alpha_lerp * b)

            cx_new = _lerp(float(r_pre['cx']), float(r_post['cx']))
            cy_new = _lerp(float(r_pre['cy']), float(r_post['cy']))
            x_new = _lerp(float(r_pre.get('x', cx_new)), float(r_post.get('x', cx_new)))
            y_new = _lerp(float(r_pre.get('y', cy_new)), float(r_post.get('y', cy_new)))
            w_new = _lerp(float(r_pre.get('w', 0)), float(r_post.get('w', 0)))
            h_new = _lerp(float(r_pre.get('h', 0)), float(r_post.get('h', 0)))
            area_new = _lerp(float(r_pre.get('area', 0)), float(r_post.get('area', 0)))
            merge_flag_final = 1

        repair_rows.append({
            'track_id': tid, 'frame': int(t),
            'cx': cx_new, 'cy': cy_new,
            'x': x_new, 'y': y_new,
            'w': w_new, 'h': h_new,
            'area': area_new,
            'merge_flag_det': 0,
            'area_z': 0.0, 'area_ratio': 1.0,
            'xo_track_flag': 0,
            'track_area_z': 0.0, 'track_area_ratio': 1.0,
            'merge_flag_final': merge_flag_final,
            'xo_parent_id': -1, 'xo_child_id': -1,
            'repair_flag': 1,
            'repair_type': f'gap_fill_{repair_type_suffix}',
        })


def _apply_continuity_repair(df, params):
    track_first_frame = df.groupby('track_id')['frame'].min().to_dict()
    track_last_frame = df.groupby('track_id')['frame'].max().to_dict()
    canonical_parent = {}
    child_claimed = set()
    repair_rows = []
    segments = _enumerate_merge_segments(df)

    if params['verbose']:
        print(f'[track_repair] continuity: found {len(segments)} merge segments')

    for tid_vis, t_start, t_end in segments:
        mask_seg = ((df['track_id'] == tid_vis) &
                    (df['frame'].between(t_start, t_end)))
        seg_rows = df[mask_seg]
        if seg_rows.empty:
            continue

        parent_ids = seg_rows.get('xo_parent_id', pd.Series([-1])).unique()
        parent_id_raw = None
        for pid in parent_ids:
            if pd.isna(pid):
                continue
            pid_int = int(pid)
            if pid_int >= 0:
                parent_id_raw = pid_int
                break

        if parent_id_raw is None:
            if params['gap_fill_visible']:
                _gap_fill_for_track(df, tid_vis, t_start, t_end, tid_vis,
                                    params, repair_rows, 'visible_no_parent')
            continue

        if parent_id_raw not in track_first_frame:
            continue

        _gap_fill_for_track(df, parent_id_raw, t_start, t_end, tid_vis,
                            params, repair_rows, 'parent')
        if params['gap_fill_visible']:
            _gap_fill_for_track(df, tid_vis, t_start, t_end, tid_vis,
                                params, repair_rows, 'visible')

        entering_parents = [int(tid_vis)]
        if int(parent_id_raw) != int(tid_vis):
            entering_parents.append(int(parent_id_raw))

        for pid in entering_parents:
            last_parent_frame = track_last_frame.get(pid, -1)
            if last_parent_frame <= t_end:
                child_ids = _link_child_track(df, pid, t_end, params,
                                              track_first_frame, child_claimed)
                for cid in child_ids:
                    if params['verbose']:
                        print(f'[track_repair] continuity: linking child {cid} '
                              f'→ parent {pid} around frames {t_start}-{t_end}')
                    _canonical_union(pid, cid, canonical_parent)
                    child_claimed.add(cid)
                    mask_child = (df['track_id'] == cid)
                    df.loc[mask_child, 'repair_flag'] = 1
                    df.loc[mask_child, 'repair_type'] = 'child_link'

    if repair_rows:
        df_rep = pd.concat([df, pd.DataFrame(repair_rows)],
                           ignore_index=True, sort=False)
    else:
        df_rep = df.copy()

    unique_ids = df_rep['track_id'].unique().tolist()
    id_map = {tid: _canonical_find(tid, canonical_parent) for tid in unique_ids}
    df_rep['track_id'] = df_rep['track_id'].map(id_map).astype(int)
    df_rep = df_rep.sort_values(['track_id', 'frame']).reset_index(drop=True)
    return df_rep


def _apply_swap_repair(df, params):
    segments = _enumerate_merge_segments(df)
    if params['verbose']:
        print(f'[track_repair] swap: inspecting {len(segments)} merge segments')
    if not segments:
        return df

    df_rep = df.copy()
    swap_min_improvement = params['swap_min_improvement']
    track_last_frame = df_rep.groupby('track_id')['frame'].max().to_dict()

    for tid_vis, t_start, t_end in segments:
        mask_seg = ((df_rep['track_id'] == tid_vis) &
                    (df_rep['frame'].between(t_start, t_end)))
        seg_rows = df_rep[mask_seg]
        if seg_rows.empty:
            continue

        parent_ids = seg_rows.get('xo_parent_id', pd.Series([-1])).unique()
        parent_id_raw = None
        for pid in parent_ids:
            if pd.isna(pid):
                continue
            pid_int = int(pid)
            if pid_int >= 0:
                parent_id_raw = pid_int
                break

        if parent_id_raw is None or parent_id_raw == tid_vis:
            continue
        if track_last_frame.get(tid_vis, -1) <= t_end:
            continue
        if track_last_frame.get(parent_id_raw, -1) <= t_end:
            continue

        t_post = t_end + 1
        vis_post = df_rep[(df_rep['track_id'] == tid_vis) &
                          (df_rep['frame'] == t_post)]
        par_post = df_rep[(df_rep['track_id'] == parent_id_raw) &
                          (df_rep['frame'] == t_post)]
        if vis_post.empty or par_post.empty:
            continue

        row_vis_post = vis_post.iloc[0]
        row_par_post = par_post.iloc[0]

        mot_vis = _estimate_parent_motion(df_rep, tid_vis, t_start,
                                          params['motion_hist_len'])
        mot_par = _estimate_parent_motion(df_rep, parent_id_raw, t_start,
                                          params['motion_hist_len'])
        if mot_vis is None or mot_par is None:
            continue

        def _predict(motion, t_eval):
            cx_prev, cy_prev, vx, vy, t_prev = motion
            dt = float(t_eval - t_prev)
            return (cx_prev + vx * dt, cy_prev + vy * dt)

        xhat_vis, yhat_vis = _predict(mot_vis, t_post)
        xhat_par, yhat_par = _predict(mot_par, t_post)

        cx_vis, cy_vis = float(row_vis_post['cx']), float(row_vis_post['cy'])
        cx_par, cy_par = float(row_par_post['cx']), float(row_par_post['cy'])

        cost_keep = ((cx_vis - xhat_vis)**2 + (cy_vis - yhat_vis)**2 +
                     (cx_par - xhat_par)**2 + (cy_par - yhat_par)**2)
        cost_swap = ((cx_par - xhat_vis)**2 + (cy_par - yhat_vis)**2 +
                     (cx_vis - xhat_par)**2 + (cy_vis - yhat_par)**2)

        if params['verbose']:
            print(f'[track_repair] swap: seg {tid_vis} ↔ {parent_id_raw} '
                  f'frames {t_start}-{t_end}, '
                  f'cost_keep={cost_keep:.2f}, cost_swap={cost_swap:.2f}')

        if cost_swap + swap_min_improvement >= cost_keep:
            continue

        if params['verbose']:
            print(f'[track_repair] swap: applying swap '
                  f'{tid_vis} ↔ {parent_id_raw} from frame {t_post}')

        tmp_id = int(df_rep['track_id'].max()) + 1
        mask_vis_tail = ((df_rep['track_id'] == tid_vis) &
                         (df_rep['frame'] >= t_post))
        mask_par_tail = ((df_rep['track_id'] == parent_id_raw) &
                         (df_rep['frame'] >= t_post))
        mask_swap_region = mask_vis_tail | mask_par_tail
        df_rep.loc[mask_swap_region, 'repair_flag'] = 1
        df_rep.loc[mask_swap_region &
                   (df_rep['repair_type'] == ''), 'repair_type'] = 'swap_repair'
        df_rep.loc[mask_vis_tail, 'track_id'] = tmp_id
        df_rep.loc[mask_par_tail, 'track_id'] = tid_vis
        df_rep.loc[df_rep['track_id'] == tmp_id, 'track_id'] = parent_id_raw
        track_last_frame = df_rep.groupby('track_id')['frame'].max().to_dict()

    return df_rep.sort_values(['track_id', 'frame']).reset_index(drop=True)


def repairTracks(df,
                 enable_continuity_repair=True,
                 enable_swap_repair=True,
                 child_max_gap=5,
                 child_max_distance=30.0,
                 enable_child_link=True,
                 motion_hist_len=5,
                 gap_fill_mode='copy_visible',
                 gap_fill_max_frames=10,
                 gap_fill_visible=False,
                 swap_min_improvement=0.0,
                 verbose=False):
    """
    Run the two-stage track repair pipeline on the segmented + flagged DataFrame.

    Stage 1 – Continuity repair:
        Fills gaps in a parent track during a merge segment and optionally
        links a "child" continuation track that appears after the merge.

    Stage 2 – Swap repair:
        Detects and corrects identity swaps caused by crossovers by comparing
        the cost of keeping vs swapping track labels.

    Parameters
    ----------
    df : DataFrame
        Output of flagMerges_track(); must contain sperm, frame, x, y,
        bbox_x, bbox_y, bbox_w, bbox_h, area, merge_flag_final,
        xo_parent_id, xo_child_id.

    Returns
    -------
    DataFrame with the same columns plus repair_flag and repair_type.
    The 'sperm' column reflects any ID remappings made during repair.
    """
    params = {
        'enable_continuity_repair': enable_continuity_repair,
        'enable_swap_repair':       enable_swap_repair,
        'child_max_gap':            child_max_gap,
        'child_max_distance':       child_max_distance,
        'enable_child_link':        enable_child_link,
        'motion_hist_len':          motion_hist_len,
        'gap_fill_mode':            gap_fill_mode.lower(),
        'gap_fill_max_frames':      gap_fill_max_frames,
        'gap_fill_visible':         gap_fill_visible,
        'swap_min_improvement':     swap_min_improvement,
        'verbose':                  verbose,
    }

    # rename to repair-module conventions
    df_r = _to_repair_df(df)

    df_stage = df_r
    if enable_continuity_repair:
        df_stage = _apply_continuity_repair(df_stage, params)

    if enable_swap_repair:
        df_stage = _apply_swap_repair(df_stage, params)

    # rename back to tracker.py conventions
    df_out = _from_repair_df(df_stage)
    return df_out