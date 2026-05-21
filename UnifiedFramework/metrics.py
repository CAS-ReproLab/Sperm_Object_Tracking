import numpy as np
import matplotlib.pyplot as plt
import cv2

import argparse

import tkinter as tk
from tkinter import filedialog

import visualizer
import utils
import pandas as pd
from  scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from ctc_metrics.utils.representations import merge_tracks, count_acyclic_graph_correction_operations
from ctc_metrics.utils.representations import count_acyclic_graph_correction_operations
from ctc_metrics.metrics import (
    valid, det, seg, tra, ct, tf, bc, cca, mota, hota, idf1, chota, mtml, faf,
    op_ctb, op_csb, bio, op_clb, lnk
)

def filterSperm(df, epsilon=5.0):

    filter_list = []

    # Filter out sperm that are not moving
    for sperm in df['sperm'].unique():
        all_locs = df[df['sperm'] == sperm]
        # Determine the mean sperm location
        mean_loc = all_locs[['x', 'y']].mean()

        # If the furthest away is less than e, remove the sperm
        all_dists = all_locs[['x', 'y']].sub(mean_loc)
        all_dists = all_dists.pow(2)
        all_dists = all_dists.sum(axis=1)
        all_dists = all_dists.pow(0.5)
        if all_dists.max() < epsilon:
            filter_list.append(sperm)

    # Filter out sperm that are not moving
    df = df[~df['sperm'].isin(filter_list)]

    return df


def makeTrackData(df):
    tracks = []

    for s in range(0, df['sperm'].max() + 1):
        cur_frames = df[df['sperm'] == s]['frame'].values
        if len(cur_frames) > 0:
            cur_birth = np.amin(cur_frames)
            cur_death = np.amax(cur_frames)
            cur_parent = 0
            cur_track = [s, cur_birth, cur_death, cur_parent]
            tracks.append(cur_track)

    return np.array(tracks)

def makeTrajectoryData(pred,gt,cutoff=10):

    # Use Hungarian Algorithm to find best track matches between pred and gt in each frame
    labels_ref = []
    labels_comp = []
    mapped_ref = []
    mapped_comp = []

    # For each frame
    for f in range(0, pred['frame'].max() + 1):

        mapped_ref_frame = []
        mapped_comp_frame = []

        # Get the labels in the frame
        ref_data = gt[gt['frame'] == f][['sperm','x','y']].values
        comp_data = pred[pred['frame'] == f][['sperm','x','y']].values

        labels_ref_frame = ref_data[:,0].astype(int)
        labels_comp_frame = comp_data[:,0].astype(int)

        ref_centroids = ref_data[:,1:]
        comp_centroids = comp_data[:,1:]

        #labels_ref_frame = gt[gt['frame'] == f]['sperm'].values
        #labels_comp_frame = pred[pred['frame'] == f]['sperm'].values
        #print(labels_ref_frame.dtype, labels_ref_frame2.dtype)

        # Get the centroids in the frame
        #ref_centroids = gt[gt['frame'] == f][['x', 'y']].values
        #comp_centroids = pred[pred['frame'] == f][['x', 'y']].values

        # Compute the distance matrix
        dist_matrix = cdist(ref_centroids, comp_centroids)

        #print(dist_matrix.shape)

        #import matplotlib.pyplot as plt
        #plt.imshow(dist_matrix);plt.show()
        #print(np.amin(dist_matrix))

        # Use Hungarian Algorithm to find best matches
        row_ind, col_ind = linear_sum_assignment(dist_matrix)

        # Check for matches that are too far apart
        for r, c in zip(row_ind, col_ind):
            #print(dist_matrix[r, c])
            if dist_matrix[r, c] > cutoff:
                row_ind = np.delete(row_ind, np.where(row_ind == r))
                col_ind = np.delete(col_ind, np.where(col_ind == c))

        if (len(row_ind) != len(np.unique(row_ind))):
            print("row issue!")
            print(row_ind.shape,np.unique(row_ind).shape)

        if (len(col_ind) != len(np.unique(col_ind))):
            print("col issue!")
            print(col_ind.shape,np.unique(col_ind).shape)

        if len(labels_ref_frame) != len(np.unique(labels_ref_frame)):
            print("labels_ref issue!")
        
        if len(labels_comp_frame) != len(np.unique(labels_comp_frame)):
            print("labels_comp issue!")

        #for r, c in zip(row_ind, col_ind):
        #    print(r,c)

        # Save the matches
        for r, c in zip(row_ind, col_ind):
            mapped_ref_frame.append(labels_ref_frame[r])
            mapped_comp_frame.append(labels_comp_frame[c])

            #labels_ref_frame.append(ref_labels[r])
            #labels_comp_frame.append(comp_labels[c])
            #mapped_ref_frame.append(gt[(gt['frame'] == f) & (gt['sperm'] == ref_labels[r])]['sperm'].values[0])
            #mapped_comp_frame.append(pred[(pred['frame'] == f) & (pred['sperm'] == comp_labels[c])]['sperm'].values[0])

        labels_ref.append(labels_ref_frame)
        labels_comp.append(labels_comp_frame)
        mapped_ref.append(mapped_ref_frame)
        mapped_comp.append(mapped_comp_frame)

    traj = {}
    traj['labels_ref'] = labels_ref
    traj['labels_comp'] = labels_comp
    traj['mapped_ref'] = mapped_ref
    traj['mapped_comp'] = mapped_comp

    #print(labels_ref)
    #print(labels_comp)
    #print(mapped_ref)
    #print(mapped_comp)

    return traj

def appendMergedTrajectory(ref_tracks, comp_tracks, traj):

    new_tracks, new_labels, new_mapped = merge_tracks(
        ref_tracks, traj["labels_ref"], traj["mapped_ref"])
    traj["ref_tracks_merged"] = new_tracks
    traj["labels_ref_merged"] = new_labels
    traj["mapped_ref_merged"] = new_mapped
    new_tracks, new_labels, new_mapped = merge_tracks(
        comp_tracks, traj["labels_comp"], traj["mapped_comp"])
    traj["comp_tracks_merged"] = new_tracks
    traj["labels_comp_merged"] = new_labels
    traj["mapped_comp_merged"] = new_mapped

    return traj

def computeMetricsFromTracks(ref_tracks, comp_tracks, traj):

    graph_operations = \
                count_acyclic_graph_correction_operations(
                    ref_tracks, comp_tracks,
                    traj["labels_ref"], traj["labels_comp"],
                    traj["mapped_ref"], traj["mapped_comp"]
                )

    #print(graph_operations)

    results = {}
    results["DET"] = det(**graph_operations)
    _tra, _aogm, _aogm0 = tra(**graph_operations)
    results["TRA"] = _tra
    results["AOGM"] = _aogm
    results["AOGM_0"] = _aogm0
    for key in ("NS", "FN", "FP", "ED", "EA", "EC"):
        results[f"AOGM_{key}"] = graph_operations[key]

    results["LNK"] = lnk(**graph_operations)

    results["CT"] = ct(
                comp_tracks, ref_tracks,
                traj["labels_ref"], traj["mapped_ref"], traj["mapped_comp"])

    results["TF"] = tf(
        ref_tracks,
        traj["labels_ref"], traj["mapped_ref"], traj["mapped_comp"])

    results.update(mota(
        traj["labels_ref_merged"], traj["labels_comp_merged"],
        traj["mapped_ref_merged"], traj["mapped_comp_merged"]))


    # Remove empty tracks before computing HOTA and IDF1
    remove_inds = []
    for i in range(len(traj["mapped_ref_merged"])):
        if traj["mapped_ref_merged"][i] == []:
            remove_inds.append(i)
        elif traj["mapped_comp_merged"][i] == []:
            remove_inds.append(i)
    if len(remove_inds) > 0:
        print("Warning removing unmatched frames for HOTA and IDF1 score.")
        traj["mapped_ref_merged"] = [x for i, x in enumerate(traj["mapped_ref_merged"]) if i not in remove_inds]
        traj["mapped_comp_merged"] = [x for i, x in enumerate(traj["mapped_comp_merged"]) if i not in remove_inds]
        traj["labels_ref_merged"] = [x for i, x in enumerate(traj["labels_ref_merged"]) if i not in remove_inds]
        traj["labels_comp_merged"] = [x for i, x in enumerate(traj["labels_comp_merged"]) if i not in remove_inds]

    results.update(hota(
        traj["labels_ref_merged"], traj["labels_comp_merged"],
        traj["mapped_ref_merged"], traj["mapped_comp_merged"]))

    results.update(idf1(
        traj["labels_ref_merged"], traj["labels_comp_merged"],
        traj["mapped_ref_merged"], traj["mapped_comp_merged"]))
    
    return results

def computeCrossoverMetrics(gt_df, pred_df, min_likelihood=0.4, threshold=25.0):
    """
    Run tracking metrics restricted to trajectories involved in GT crossover events.

    Because GT and predicted sperm IDs are independent label spaces, we first run
    a full Hungarian spatial match on both complete datasets to discover which
    predicted IDs ever corresponded to a GT sperm that was part of a crossover.
    We then re-run metrics on just those filtered subsets.

    Parameters
    ----------
    gt_df, pred_df : pd.DataFrame
        Raw ground-truth and prediction DataFrames (columns: frame, x, y, sperm).
    min_likelihood : float
        Only GT crossover events at or above this likelihood are used (default 0.4).
    threshold : float
        Proximity threshold in pixels passed to detect_crossovers (default 25.0).

    Returns
    -------
    dict with keys:
        'results'           : metrics dict from computeMetricsFromTracks
        'gt_crossover_ids'  : set of GT sperm IDs involved in crossovers
        'pred_crossover_ids': set of pred sperm IDs matched to those GT sperm
        'events'            : crossover events DataFrame
    Returns None if no crossover events are found above min_likelihood.
    """
    from crossover_detector import detect_crossovers

    # --- 1. Preprocess (mirrors computeMetrics) ---
    gt_sorted = gt_df.sort_values(['sperm', 'frame']).reset_index(drop=True)
    pred_sorted = pred_df.sort_values(['sperm', 'frame']).reset_index(drop=True)

    gt_u = utils.dropDuplicates(gt_sorted)
    pred_u = utils.dropDuplicates(pred_sorted)

    gt = utils.interpolateTracks(gt_u)
    pred = utils.interpolateTracks(pred_u)

    # --- 2. Detect crossovers in GT ---
    print(f"Detecting crossovers in GT (threshold={threshold}px, min_likelihood={min_likelihood})...")
    events = detect_crossovers(gt, threshold=threshold)
    events = events[events['likelihood'] >= min_likelihood].reset_index(drop=True)

    if events.empty:
        print(f"No crossover events found at likelihood >= {min_likelihood}. "
              "Try lowering --crossover-likelihood or raising --crossover-threshold.")
        return None

    gt_crossover_ids = set(events['sperm_a'].tolist() + events['sperm_b'].tolist())
    print(f"  {len(events)} crossover events involving {len(gt_crossover_ids)} GT sperm: "
          f"{sorted(gt_crossover_ids)}")

    # --- 3. Map GT crossover IDs -> pred IDs via full-data spatial matching ---
    # We use the full dataset here so the Hungarian assignment is globally optimal,
    # giving the most accurate GT<->pred ID correspondence before we subset.
    print("Mapping GT crossover sperm to predicted sperm via spatial matching...")
    traj_full = makeTrajectoryData(pred, gt)

    pred_crossover_ids = set()
    for frame_gt_ids, frame_pred_ids in zip(traj_full['mapped_ref'], traj_full['mapped_comp']):
        for gt_id, pred_id in zip(frame_gt_ids, frame_pred_ids):
            if gt_id in gt_crossover_ids:
                pred_crossover_ids.add(pred_id)

    print(f"  {len(pred_crossover_ids)} predicted sperm matched to crossover GT sperm: "
          f"{sorted(pred_crossover_ids)}")

    # --- 4. Filter both DataFrames to crossover-involved sperm ---
    gt_cross = gt[gt['sperm'].isin(gt_crossover_ids)].copy()
    pred_cross = pred[pred['sperm'].isin(pred_crossover_ids)].copy()

    if gt_cross.empty or pred_cross.empty:
        print("Warning: filtered DataFrames are empty — no metrics can be computed.")
        return None

    print(f"  GT subset:   {gt_cross['sperm'].nunique()} sperm, "
          f"{gt_cross['frame'].nunique()} frames")
    print(f"  Pred subset: {pred_cross['sperm'].nunique()} sperm, "
          f"{pred_cross['frame'].nunique()} frames")

    # --- 5. Compute metrics on the filtered subset ---
    gt_tracks = makeTrackData(gt_cross)
    pred_tracks = makeTrackData(pred_cross)

    traj_cross = makeTrajectoryData(pred_cross, gt_cross)
    traj_cross = appendMergedTrajectory(gt_tracks, pred_tracks, traj_cross)

    results = computeMetricsFromTracks(gt_tracks, pred_tracks, traj_cross)

    return {
        'results': results,
        'gt_crossover_ids': gt_crossover_ids,
        'pred_crossover_ids': pred_crossover_ids,
        'events': events,
    }


def computeMetrics(gt_df,pred_df,return_filtered=True):

    # Ensure the dataframes are sorted by 'sperm' then 'frame' to gaurantee correct calculations
    gt_sorted = gt_df.sort_values(by=['sperm', 'frame']).reset_index(drop=True)
    pred_sorted = pred_df.sort_values(by=['sperm', 'frame']).reset_index(drop=True)

    gt_u = utils.dropDuplicates(gt_sorted)
    pred_u = utils.dropDuplicates(pred_sorted)

    gt = utils.interpolateTracks(gt_u)
    pred = utils.interpolateTracks(pred_u)

    print("GT sperm count:", gt['sperm'].nunique())
    print("Pred sperm count:", pred['sperm'].nunique())

    pred_tracks = makeTrackData(pred)
    gt_tracks = makeTrackData(gt)

    traj = makeTrajectoryData(pred,gt)

    traj = appendMergedTrajectory(gt_tracks, pred_tracks, traj)

    results = computeMetricsFromTracks(gt_tracks, pred_tracks, traj)

    if return_filtered:
        pred_filter = filterSperm(pred)
        gt_filter = filterSperm(gt)

        print("GT filtered sperm count:", gt_filter['sperm'].nunique())
        print("Pred filtered sperm count:", pred_filter['sperm'].nunique())

        pred_filter_tracks = makeTrackData(pred_filter)
        gt_filter_tracks = makeTrackData(gt_filter)

        traj_filter = makeTrajectoryData(pred_filter,gt_filter)

        traj_filter = appendMergedTrajectory(gt_filter_tracks, pred_filter_tracks, traj_filter)

        results_filter = computeMetricsFromTracks(gt_filter_tracks, pred_filter_tracks, traj_filter)

        counts = {
            'unfiltered': gt['sperm'].nunique(),
            'filtered': gt_filter['sperm'].nunique(),
        }
        return results, results_filter, counts

    else:
        return results

if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser(description='Track cells in a video')
    parser.add_argument('--prediction', type=str, default=None, help='Path to the prediction csv file')
    parser.add_argument('--groundtruth', type=str, default=None, help='Path to the ground truth csv file')
    parser.add_argument('--all', action='store_true', help='Compute all metrics')
    parser.add_argument('--crossover', action='store_true',
                        help='Compute metrics restricted to crossover-involved trajectories')
    parser.add_argument('--crossover-likelihood', type=float, default=0.4,
                        help='Min crossover likelihood to include (default: 0.4)')
    parser.add_argument('--crossover-threshold', type=float, default=25.0,
                        help='Proximity threshold in pixels for crossover detection (default: 25.0)')

    args = parser.parse_args()
    predictionfile = args.prediction
    groundtruthfile = args.groundtruth
    report_all = args.all

    if predictionfile is None:
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        predictionfile = filedialog.askopenfilename(title="Select the prediction csv file")

        if predictionfile:
            print("Selected file:", predictionfile)
        else:
            raise ValueError("No prediction file selected.")
        
    if groundtruthfile is None:
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        groundtruthfile = filedialog.askopenfilename(title="Select the groundtruth csv file")

        if groundtruthfile:
            print("Selected file:", groundtruthfile)
        else:
            raise ValueError("No groundtruth file selected.")


    # Load as Pandas DataFrame
    pred_src = utils.loadDataFrame(predictionfile)
    gt_src = utils.loadDataFrame(groundtruthfile)

    # --- Always run standard full metrics ---
    results, results_filter, counts = computeMetrics(gt_src, pred_src)

    key_metrics = ["DET", "LNK", "TRA", "TF", "MOTA", "IDF1", "HOTA"]
    if not report_all:
        results        = {k: results[k]        for k in key_metrics if k in results}
        results_filter = {k: results_filter[k] for k in key_metrics if k in results_filter}

    results_df = pd.DataFrame(
        [[k, results[k], results_filter[k]] for k in results],
        columns=["Metric", "Unfiltered", "Filtered"]
    )
    # Prepend GT sperm count row
    count_row = pd.DataFrame(
        [["GT Sperm", counts['unfiltered'], counts['filtered']]],
        columns=["Metric", "Unfiltered", "Filtered"]
    )
    results_df = pd.concat([count_row, results_df], ignore_index=True)

    if args.crossover:
        # --- Also run crossover-filtered metrics and add as a third column ---
        crossover_result = computeCrossoverMetrics(
            gt_src, pred_src,
            min_likelihood=args.crossover_likelihood,
            threshold=args.crossover_threshold,
        )
        if crossover_result is None:
            print("Could not compute crossover metrics.")
        else:
            print(f"\n  GT crossover sperm:   {sorted(int(x) for x in crossover_result['gt_crossover_ids'])}")
            print(f"  Pred matched sperm:   {sorted(int(x) for x in crossover_result['pred_crossover_ids'])}")
            print(f"  Crossover events:     {len(crossover_result['events'])}")

            res_cross = crossover_result['results']
            if not report_all:
                res_cross = {k: res_cross[k] for k in key_metrics if k in res_cross}

            cross_count = len(crossover_result['gt_crossover_ids'])
            results_df["Crossover"] = results_df["Metric"].map(
                {"GT Sperm": cross_count, **res_cross}
            )

        savefile = "results_crossover.csv"
    else:
        savefile = "results.csv"

    print()
    # Print with counts as integers and metric values as floats
    numeric_cols = [c for c in results_df.columns if c != "Metric"]
    display_df = results_df.copy()
    for col in numeric_cols:
        display_df[col] = display_df[col].apply(
            lambda x: f"{int(x)}" if pd.notna(x) and float(x) == int(float(x)) and abs(float(x)) < 1e6
            else (f"{x:.6f}" if pd.notna(x) else "")
        )
    print(display_df.to_string(index=False))
    utils.saveDataFrame(results_df, savefile)
    print(f"\nResults saved to {savefile}")
