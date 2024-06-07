import numpy as np
import cv2 as cv
from scipy.optimize import linear_sum_assignment
import pickle
import json

import argparse
from tqdm import tqdm, trange

import pandas as pd

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

def loadDataFrame(filename):
    return pd.read_csv(filename)

def saveDataFrame(df, filename):
    df.to_csv(filename, index=False)

