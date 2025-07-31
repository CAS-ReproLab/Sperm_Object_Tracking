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

    results.update(hota(
        traj["labels_ref_merged"], traj["labels_comp_merged"],
        traj["mapped_ref_merged"], traj["mapped_comp_merged"]))

    results.update(idf1(
        traj["labels_ref_merged"], traj["labels_comp_merged"],
        traj["mapped_ref_merged"], traj["mapped_comp_merged"]))
    
    return results

def computeMetrics(gt_df,pred_df):

    # Ensure the dataframes are sorted by 'sperm' then 'frame' to gaurantee correct calculations
    gt_sorted = gt_df.sort_values(by=['sperm', 'frame']).reset_index(drop=True)
    pred_sorted = pred_df.sort_values(by=['sperm', 'frame']).reset_index(drop=True)

    gt_u = utils.dropDuplicates(gt_sorted)
    pred_u = utils.dropDuplicates(pred_sorted)

    gt = utils.interpolateTracks(gt_u)
    pred = utils.interpolateTracks(pred_u)

    pred_filter = filterSperm(pred)
    gt_filter = filterSperm(gt)

    pred_tracks = makeTrackData(pred)
    gt_tracks = makeTrackData(gt)
    pred_filter_tracks = makeTrackData(pred_filter)
    gt_filter_tracks = makeTrackData(gt_filter)

    traj = makeTrajectoryData(pred,gt)
    traj_filter = makeTrajectoryData(pred_filter,gt_filter)

    traj = appendMergedTrajectory(gt_tracks, pred_tracks, traj)
    traj_filter = appendMergedTrajectory(gt_filter_tracks, pred_filter_tracks, traj_filter)

    results = computeMetricsFromTracks(gt_tracks, pred_tracks, traj)
    results_filter = computeMetricsFromTracks(gt_filter_tracks, pred_filter_tracks, traj_filter)

    return results, results_filter

if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser(description='Track cells in a video')
    parser.add_argument('--prediction', type=str, default=None, help='Path to the prediction csv file')
    parser.add_argument('--groundtruth', type=str, default=None, help='Path to the ground truth csv file')
    parser.add_argument('--all', action='store_true', help='Compute all metrics')

    predictionfile = parser.parse_args().prediction
    groundtruthfile = parser.parse_args().groundtruth
    report_all = parser.parse_args().all

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
        
    results, results_filter = computeMetrics(gt_src, pred_src)
    
    if not report_all:
        # Filter the results to only include the metrics we want to report
        results = {key: results[key] for key in ["DET","TRA", "LNK", "TF", "MOTA", "IDF1", "HOTA"]}
        results_filter = {key: results_filter[key] for key in ["DET","TRA", "LNK", "TF", "MOTA", "IDF1", "HOTA"]}

    # Concatenate results into dataframe
    results_df = pd.DataFrame(columns=["Metric", "Unfiltered", "Filtered"])

    for key,val in results.items():        
        results_df = pd.concat([results_df, pd.DataFrame([[key, val, results_filter[key]]], columns=["Metric", "Unfiltered", "Filtered"])], ignore_index=True)

    #results_df.reset_index(drop=True, inplace=True)

    print(results_df)

    utils.saveDataFrame(results_df, "results.csv")
    print("Results saved to results.csv")
