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
