import numpy as np
import cv2 as cv
from scipy.optimize import linear_sum_assignment
import pickle
import json

import argparse
from tqdm import tqdm, trange

import pandas as pd

def loadVideo(videofile, as_grey=True):
    cap = cv.VideoCapture(videofile)
    frames = []
    while True:
        ret, frame = cap.read()
        if frame is None:
            break
        if as_grey:
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frames.append(frame)
    cap.release()
    frames = np.array(frames)
    return frames

def loadDataFrame(filename):
    return pd.read_csv(filename)

def saveDataFrame(df, filename):
    df.to_csv(filename, index=False)

def loadPKL(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
    
def savePKL(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def makeSperm():
    sperm = {}
    sperm['centroid'] = {}
    sperm['bbox'] = {} 
    sperm['area'] = {}
    sperm['segmentation'] = {}
    sperm['visible'] = []

    return sperm

def dataFrameToDict(df):
    pass

def dictToDataFrame(data):
    pass
