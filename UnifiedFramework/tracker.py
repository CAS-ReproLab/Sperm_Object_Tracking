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
    f = pd.DataFrame(columns=['x', 'y', 'frame'])

    # Find centroids by focusing on heads
    for i in range(len(frames)):
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
            f.loc[len(f.index)] = [centroid[0], centroid[1], i]

    return f

def determineCentroids(frames, diameter=11, minmass=500):
    f = tp.batch(frames, diameter=diameter, minmass=minmass)
    return f

def trackCentroids(f, search_range=7, memory=3):
    t = tp.link(f, search_range, memory=memory)
    t = tp.filter_stubs(t, 15)

    # Change the column name of particle to sperm
    t = t.rename(columns={'particle': 'sperm'})

    return t

def segmentCells(frames, t):
    """
    Segment full sperm cells in each frame using the centroids and adaptive thresholding
    """

    final = t.copy(deep=True)

    # Add new columns for segmentations, areas, and bounding boxes
    final['segmentation'] = None
    final['area'] = 0
    final['bbox'] = None

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
    for row in final.iterrows():
        n = row[1]['frame']
        x = row[1]['x']
        y = row[1]['y']

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
            #del_indices.append(i)
            continue 

        final.at[row[0],'area'] = all_areas[n][label]
        final.at[row[0],'bbox'] = all_bboxs[n][label].tolist()
        final.at[row[0],'segmentation'] = all_segmentations[n][label]
        
    print("Warning:", out_indices, "centroids found in background in", len(frames), "frames")

    return final


def processVideo(videofile):

    # Open the video file
    frames = utils.loadVideo(videofile)

    # Determine the centroids info
    f = determineCentroids(frames)
    
    # Track the centroids
    t = trackCentroids(f)

    # Segment the cells
    final = segmentCells(frames, t)

    return final

def saveDataFrame(df, filename):
    df.to_csv(filename, index=False)

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

def makeSperm():
    sperm = {}
    sperm['centroid'] = {}
    sperm['bbox'] = {} 
    sperm['area'] = {}
    sperm['segmentation'] = {}
    sperm['visible'] = []

    return sperm


### Main Code ###
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Track cells in a video')
    parser.add_argument('videofile', type=str, help='Path to the video file')

    videofile = parser.parse_args().videofile

    final = processVideo(videofile)

    saveDataFrame(final, videofile.split('.')[0] + '_tracked.csv')

    # Save sperm data to pickle file
    #outputfile = videofile.split('.')[0] + '_tracked.pkl'
    #with open(outputfile, 'wb') as f:
    #    pickle.dump(all_sperm, f)

    # Save sperm data to json file
    #outputfile = videofile.split('.')[0] + '_tracked.json'
    #with open(outputfile, 'w') as f:
    #    json.dump(all_sperm, f)

    print(outputfile,' file saved')