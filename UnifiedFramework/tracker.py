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

def determineCentroids(frames, diameter=11, minmass=500):
    f = tp.batch(frames, diameter=diameter, minmass=minmass)
    return f

def trackCentroids(f, search_range=7, memory=3):
    t = tp.link(f, search_range, memory=memory)
    t = tp.filter_stubs(t, 15)

    # Change the column name of particle to sperm
    t = t.rename(columns={'particle': 'sperm'})

    t = t.reset_index(drop=True)

    return t

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


def processVideo(videofile, compute_segs=True):

    # Open the video file
    frames = utils.loadVideo(videofile,as_gray=True)

    # Determine the centroids info
    #f = determineCentroids_morphology(frames)
    f = determineCentroids(frames)

    # Track the centroids
    t = trackCentroids(f)

    # Segment the cells
    if compute_segs:
        t = segmentCells(frames, t)

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
    parser.add_argument('--output', type=str, default=None, help='Path to the output file')
    parser.add_argument('--no_segmentation', action='store_false', help='Do not segment the cells')

    videofile = parser.parse_args().videofile
    outputfile = parser.parse_args().output
    compute_segs = parser.parse_args().no_segmentation

    if videofile is None:
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        videofile = filedialog.askopenfilename()

        if videofile:
            print("Selected file:", videofile)

    final = processVideo(videofile,compute_segs)

    if outputfile is None:
        outputfile = ".".join(videofile.split('.')[:-1]) + '.csv'

    utils.saveDataFrame(final, outputfile)

    print(outputfile, 'file saved')