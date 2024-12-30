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