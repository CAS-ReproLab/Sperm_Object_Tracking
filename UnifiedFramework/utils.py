import numpy as np
import cv2 as cv
from scipy.optimize import linear_sum_assignment

import argparse
from tqdm import tqdm, trange

import pandas as pd

import ast

def loadVideo(videofile, as_gray=False):
    cap = cv.VideoCapture(videofile)
    frames = []
    while True:
        ret, frame = cap.read()
        if frame is None:
            break
        if as_gray:
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frames.append(frame)
    cap.release()
    frames = np.array(frames)
    return frames

def saveVideo(frames, filename, fps=30):
    
    # Add color channel if in grayscale
    if len(frames.shape) == 3:
        frames = np.expand_dims(frames, axis=-1)
        frames = np.repeat(frames, 3, axis=-1)
    
    _, height, width, _ = frames.shape
    fourcc = cv.VideoWriter_fourcc(*'avc1')
    out = cv.VideoWriter(filename, fourcc, fps, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()

def loadDataFrame(filename, convert_segmentation=False):
    data = pd.read_csv(filename)

    # Convert frame and sperm columns to integer
    data['frame'] = data['frame'].astype(int)
    data['sperm'] = data['sperm'].astype(int)

    # Convert the string representation of segmentation to a list
    if convert_segmentation:
        print("Converting segmentations to list...")
        data['segmentation'] = data['segmentation'].apply(ast.literal_eval)
        print("Done.")

    return data

def saveDataFrame(df, filename):
    df.to_csv(filename, index=False)

def generateRandomColors(n):

    H = np.random.randint(0, 255, (n, 1)).astype(np.uint8)
    S = np.random.randint(50, 255, (n, 1)).astype(np.uint8)
    V = np.random.randint(150, 255, (n, 1)).astype(np.uint8)

    hsv = np.concatenate([H, S, V], axis=1)
    hsv = np.expand_dims(hsv, axis=0)
    colors = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    colors = colors[0]

    return colors

def medianFilter(frames, perFrame=False):
    frames = frames.astype(np.float32)

    if perFrame:
        for i in range(len(frames)):
            med = np.median(frames[i], axis=0)
            frames[i] = np.abs(frames[i] - med)
    else:
        med = np.median(frames, axis=0)
        frames = np.abs(frames - med)

    frames = frames.astype(np.uint8)
    return frames

def positivePhaseFilter(frames, cutoff=5):
    frames = frames.astype(np.float32)

    med = np.median(frames, axis=0)
    frames = frames - med

    # Brighten the dark tails
    tails = np.where(frames < -cutoff)
    frames[tails] = 255 - frames[tails]

    frames = np.clip(frames, 0, 255)

    frames = frames.astype(np.uint8)
    return frames


def dropDuplicates(df):
      
    # Detect duplicate rows based on 'col1' and 'col2'
    #duplicate_rows = df[df.duplicated(subset=['frame', 'sperm'], keep='first')]
    result = df.drop_duplicates(subset=['frame', 'sperm'], keep='first')

    return result

def interpolateTracks(df):

    result = df.copy()

    for sperm in range(0, df['sperm'].max() + 1):
        sperm_frames = df[df['sperm'] == sperm]['frame'].values
        if len(sperm_frames) > 1:
            birth = np.amin(sperm_frames)
            death = np.amax(sperm_frames)


            # Find if the sperm exists for all frames
            if len(sperm_frames) != death - birth + 1:
                #print("Missing frames for sperm: ", sperm)
                #print("Birth: ", birth, ", Death: ", death)
                #print("Frames: ", sperm_frames)

                for j in range(birth, death + 1):
                    if j not in sperm_frames:
                 
                        # Find closest frame after the missing frame
                        before = np.amax(sperm_frames[np.where(sperm_frames < j)])
                        after = np.amin(sperm_frames[np.where(sperm_frames > j)])

                        # interpolate x and y
                        before_x = df[(df['sperm'] == sperm) & (df['frame'] == before)]['x'].values[0]
                        before_y = df[(df['sperm'] == sperm) & (df['frame'] == before)]['y'].values[0]
                        after_x = df[(df['sperm'] == sperm) & (df['frame'] == after)]['x'].values[0]
                        after_y = df[(df['sperm'] == sperm) & (df['frame'] == after)]['y'].values[0]

                        x = before_x + (after_x - before_x) * (j - before) / (after - before)
                        y = before_y + (after_y - before_y) * (j - before) / (after - before)

                        #print("Adding frame: ", j)
                        #gt = gt.append({'frame': j, 'sperm': sperm, 'x': x, 'y': y}, ignore_index=True)
                        result = pd.concat([result, pd.DataFrame([[j, sperm, x, y]], columns=['frame', 'sperm', 'x', 'y'])], ignore_index=True)

    # Fill in missing fields
    result = result.fillna(0)

    return result