import numpy as np
import cv2 as cv
from scipy.optimize import linear_sum_assignment
import pickle
import json

import argparse
from tqdm import tqdm, trange

import trackpy as tp
import utils

import pandas as pd

import tkinter as tk
from tkinter import filedialog

from crossover_repair import flagMerges_detection, flagMerges_track, repairTracks

#import pims
#@pims.pipeline
#def as_grey(frame):
#    return cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

#def load_video(videofile):
#    print(videofile)
#    return as_grey(pims.open(videofile))

def threshold(frame, method='otsu',global_thresh=50):
    
    # Check if the frame is grayscale
    if len(frame.shape) == 3:
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    #mid_val = np.median(gray)
    #gray = np.abs(gray - mid_val).astype(np.uint8)

    if method == 'global':
        _, bw = cv.threshold(frame,global_thresh,255,cv.THRESH_BINARY)
    elif method == 'median':
        thresh_val = np.median(frame) + 20
        #print(thresh_val)
        _, bw = cv.threshold(frame,thresh_val,255,cv.THRESH_BINARY)
    elif method == 'otsu':
        _, bw = cv.threshold(frame,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    elif method == 'adaptive':
        bw = cv.adaptiveThreshold(frame,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,11,-2)
    elif method == 'hybrid':
        _, bw1 = cv.threshold(frame,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

        bw2 = cv.adaptiveThreshold(frame,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,11,-2)
        bw = cv.bitwise_or(bw1,bw2)
    else:
        raise ValueError('Invalid thresholding method')

    return bw

def determineCentroids_simple(frames,method="hybrid",global_thresh=50):

    # Make dataframe to store centroids
    f = pd.DataFrame(columns=['y', 'x', 'frame'])

    # Find centroids by focusing on heads
    for i in trange(len(frames)):
        frame = frames[i]

        # Find centroids by focusing by taking every blob
        bw = threshold(frame, method=method, global_thresh=global_thresh)
        _, _, _, centroids = cv.connectedComponentsWithStats(bw, 4, cv.CV_32S) 

        # Filter out the background (always index 0)
        centroids = centroids[1:]

        # Add centroids to dataframe
        for centroid in centroids:
            f.loc[len(f.index)] = [centroid[1], centroid[0], i]

    return f

def determineCentroids_morphology(frames, kernel_size=(3,3)):

    # Make dataframe to store centroids
    f = pd.DataFrame(columns=['y', 'x', 'frame'])

    # Find centroids by focusing on heads
    for i in trange(len(frames)):
        frame = frames[i]

        # Find centroids by focusing on heads
        bw = threshold(frame, method='otsu')
        kernel = np.ones(kernel_size,np.uint8)
        bw = cv.morphologyEx(bw, cv.MORPH_OPEN, kernel)
        _, _, _, centroids = cv.connectedComponentsWithStats(bw, 4, cv.CV_32S) 

        # Filter out the background (always index 0)
        centroids = centroids[1:]

        # Add centroids to dataframe
        for centroid in centroids:
            f.loc[len(f.index)] = [centroid[1], centroid[0], i]
       
    return f

def determineCentroids(frames, diameter=7, minmass=100, maxsize=5):
    f = tp.batch(frames, diameter=diameter, minmass=minmass, maxsize=maxsize)
    
    return f

def trackCentroids(f, search_range=21, memory=3, adaptive_stop=0.2, adaptive_step=0.95):
    t = tp.link(f, search_range=search_range, memory=memory, adaptive_stop=adaptive_stop, adaptive_step=adaptive_step)
    t = tp.filter_stubs(t, 15)

    # Change the column name of particle to sperm
    t = t.rename(columns={'particle': 'sperm'})

    t = t.reset_index(drop=True)
    
    return t

def trackCentroids_forecaster(f, window_size=5, target_size=5, model_fn="forecast_tracker_model_5_5.pkl", dist_cutoff=20):

    # Load the trained model and set up the dataframe for tracking
    import joblib
    from tqdm import trange
    model = joblib.load(model_fn)

    df = f.copy()
    df['sperm'] = -1
    next_sperm_id = 0

    # Assign IDs to first frame
    first_frame_detections = df[df['frame'] == df['frame'].min()]   
    for index, det in first_frame_detections.iterrows():
        df.at[index, 'sperm'] = next_sperm_id
        next_sperm_id += 1

    for frame_num in trange(df['frame'].min(), df['frame'].max()):
        frame_detections = df[df['frame'] == frame_num]
        next_frame_num = frame_num + 1

        if next_frame_num > df['frame'].max():
            break

        # --- PASS 1: Try to link all existing tracks into the next frame ---
        for index, row in frame_detections.iterrows():
            sperm_id = row["sperm"]

            if sperm_id == -1:
                continue  # Skip unmatched for now, handle after

            # If this sperm already exists in the next frame, skip
            if sperm_id in df[df['frame'] == next_frame_num]['sperm'].values:
                continue

            prev_traj = df[df["sperm"] == sperm_id].sort_values(by='frame')

            if len(prev_traj) < window_size:
                # --- Radius matching ---
                next_frame_detections = df[df['frame'] == next_frame_num]
                if next_frame_detections.empty:
                    continue

                min_dist = float('inf')
                best_index = -1
                for det_index, det_row in next_frame_detections.iterrows():
                    dist = np.sqrt((det_row['x'] - row['x']) ** 2 + (det_row['y'] - row['y']) ** 2)
                    if dist < min_dist and dist < dist_cutoff and det_row['sperm'] == -1:
                        min_dist = dist
                        best_index = det_index

                if best_index != -1:
                    df.at[best_index, 'sperm'] = sperm_id

            else:
                # --- Model prediction matching ---
                input_window = []
                for w in range(window_size):
                    prev_row = prev_traj[prev_traj['frame'] == frame_num - w - 1]
                    cur_row = prev_traj[prev_traj['frame'] == frame_num - w]
                    if prev_row.empty or cur_row.empty:
                        input_window.extend([0, 0])
                    else:
                        input_window.extend([cur_row.iloc[0]['x'] - prev_row.iloc[0]['x'],
                                             cur_row.iloc[0]['y'] - prev_row.iloc[0]['y']])
                input_window = np.array(input_window).reshape(1, -1)

                prediction = model.predict(input_window)[0]

                predicted_positions = []
                last_x = row['x']
                last_y = row['y']
                for t in range(target_size):
                    last_x += prediction[2 * t]
                    last_y += prediction[2 * t + 1]
                    predicted_positions.append((last_x, last_y))

                for t in range(target_size):
                    target_frame = frame_num + 1 + t

                    if target_frame > df['frame'].max():
                        break

                    target_detections = df[df['frame'] == target_frame]
                    if target_detections.empty:
                        continue

                    # Skip if already linked into this frame
                    if sperm_id in target_detections['sperm'].values:
                        continue

                    pred_x, pred_y = predicted_positions[t]

                    min_dist = float('inf')
                    best_index = -1
                    for det_index, det_row in target_detections.iterrows():
                        dist = np.sqrt((det_row['x'] - pred_x) ** 2 + (det_row['y'] - pred_y) ** 2)
                        if dist < min_dist and dist < dist_cutoff and det_row['sperm'] == -1:
                            min_dist = dist
                            best_index = det_index

                    if best_index != -1:
                        df.at[best_index, 'sperm'] = sperm_id

        # --- PASS 2: Assign new IDs to anything still unmatched in the next frame ---
        next_frame_detections = df[df['frame'] == next_frame_num]
        for index, det_row in next_frame_detections.iterrows():
            if det_row['sperm'] == -1:
                df.at[index, 'sperm'] = next_sperm_id
                next_sperm_id += 1

    print("Here")

    return df

def segmentCells(frames, t):
    """
    Segment full sperm cells in each frame using the centroids and adaptive thresholding
    """

    final = t.copy(deep=True)

    # Add new columns for segmentations, areas, and bounding boxes
    final['area'] = 0
    final['bbox_x'] = 0
    final['bbox_y'] = 0
    final['bbox_w'] = 0
    final['bbox_h'] = 0
    final['segmentation'] = None

    # Initialize the labels_ims (whether frames is list or numpy array)
    all_label_ims = np.zeros((len(frames), frames[0].shape[0], frames[0].shape[1]), dtype=np.int32)

    # Generate all lists of segmentations, areas, and bounding boxes
    all_bboxs = []
    all_areas = []
    all_segmentations = []
    for n in trange(len(frames)):
        frame = frames[n]

        # Run connected components again with a lower threshold to get the segmentation
        bw2 = threshold(frame, method='hybrid')
        _, label_im, stats, _ = cv.connectedComponentsWithStats(bw2, 4, cv.CV_32S)

        # Seperate bbox from area
        areas = stats[:,4]
        bboxs = stats[:,0:4]

        # Filter out the background (always index 0)
        areas = areas[1:]
        bboxs = bboxs[1:]
        label_im -= 1

        # Turn label_im into list of segmentations
        segmentations = labelIm2Array(label_im, len(stats))

        all_label_ims[n] = label_im
        all_bboxs.append(bboxs)
        all_areas.append(areas)
        all_segmentations.append(segmentations)


    # For each row of the dataframe, associate the correct segmentation, area, and bounding box
    out_indices = 0
    for idx, row in final.iterrows():

        n = row['frame']
        x = row['x']
        y = row['y']

        r,c = int(y),int(x)
        if r < 0 or c < 0 or r >= label_im.shape[0] or c >= label_im.shape[1]:
            print("Warning: Centroid found out of bounds")
            continue

        # Check the label of the four surrounding pixels    
        r2 = r+1 if r+1 < label_im.shape[0] else r
        c2 = c+1 if c+1 < label_im.shape[1] else c
        label_tl = all_label_ims[n,r,c]
        label_tr = all_label_ims[n,r,c2]
        label_bl = all_label_ims[n,r2,c]
        label_br = all_label_ims[n,r2,c2]
        
        if label_tl >= 0:
            label = label_tl
        elif label_tr >= 0:
            label = label_tr
        elif label_bl >= 0:
            label = label_bl
        else:
            label = label_br
            # TODO: Check mode of the four labels if they are greater than 1

        if label == -1:
            #print("\n Warning: Centroid found in background")
            out_indices += 1
            final.at[idx,'area'] = -1
            final.at[idx,'bbox_x'] = -1
            final.at[idx,'bbox_y'] = -1
            final.at[idx,'bbox_w'] = -1
            final.at[idx,'bbox_h'] = -1
            final.at[idx,'segmentation'] = []
            #del_indices.append(i)
            continue

        bbox = all_bboxs[n][label]

        final.at[idx,'area'] = all_areas[n][label]
        final.at[idx,'bbox_x'] = bbox[0]
        final.at[idx,'bbox_y'] = bbox[1]
        final.at[idx,'bbox_w'] = bbox[2]
        final.at[idx,'bbox_h'] = bbox[3]
        final.at[idx,'segmentation'] = all_segmentations[n][label]
        
    print("Warning:", out_indices, "centroids found in background in", len(frames), "frames")
    
    return final


def processVideo(videofile, compute_segs=True, enable_repair=False):

    # Open the video file
    frames = utils.loadVideo(videofile,as_gray=True)
    
    #first_25_frames = frames[:25]

    # Determine the centroids info
    #f = determineCentroids_morphology(frames)
    f = determineCentroids(frames, 5, 50, 5)

    # Track the centroids
    t = trackCentroids(f)
    #t = trackCentroids_forecaster(f)

    # Segment the cells
    if compute_segs:
        t = segmentCells(frames, t)

    if enable_repair:
        t = flagMerges_detection(t)
            # window=det_window,
            # radius=det_radius,
            # z_thresh=det_z_thresh,
            # ratio_thresh=det_ratio_thresh,
            # )


        t = flagMerges_track(t)
                # z_thresh=track_z_thresh,
                # ratio_thresh=track_ratio_thresh,
                # hist_len=track_hist_len,
                # )


        t = repairTracks(t)
            # enable_continuity_repair=enable_continuity_repair,
            # enable_swap_repair=enable_swap_repair,
            # gap_fill_mode=gap_fill_mode,
            # child_max_gap=child_max_gap,
            # child_max_distance=child_max_distance,
            # verbose=repair_verbose,
            # )

    return t


def labelIm2Array(label_im, num_labels):
    segmentations = []
    for i in range(0, num_labels):
        segmentations.append([])

    rows, cols = label_im.shape
    for i in range(rows):
        for j in range(cols):
            if label_im[i,j] != -1:
                segmentations[label_im[i,j]].append([i,j])
    
    return segmentations


### Main Code ###
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Track cells in a video')
    parser.add_argument('--videofile', type=str, default=None, help='Path to the video file')
    parser.add_argument('--enable_repair', action='store_true', help='Enable reparametrization of tracks')
    parser.add_argument('--output', type=str, default=None, help='Path to the output file')
    parser.add_argument('--no_segmentation', action='store_false', help='Do not segment the cells')

    videofile = parser.parse_args().videofile
    outputfile = parser.parse_args().output
    compute_segs = parser.parse_args().no_segmentation
    enable_repair = parser.parse_args().enable_repair

    if videofile is None:
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        videofile = filedialog.askopenfilename(title="Select the video file")

        if videofile:
            print("Selected file:", videofile)
        else:
            raise ValueError("No video file selected.")

    final = processVideo(videofile,compute_segs,enable_repair)

    if outputfile is None:
        outputfile = ".".join(videofile.split('.')[:-1]) + '.csv'

    utils.saveDataFrame(final, outputfile)

    print(outputfile, 'file saved')