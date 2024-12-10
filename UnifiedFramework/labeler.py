import argparse
import os
import json
import pickle
import numpy as np
import cv2 as cv
import time

import utils
import visualizer

import tkinter as tk
from tkinter import filedialog

current_sperm = None

def mergeSperm(data,sperm1,sperm2):
    pass

def splitSperm(data,sperm,frame_num):
    pass

def deleteSperm(data,sperm):
    pass

def onMouse(event, x, y, flags, param):

    global current_sperm

    data = param[0]
    frame_num = param[1]

    if event == cv.EVENT_LBUTTONDOWN:

        # Get only data for the current frame
        current = data[data['frame'] == frame_num]

        for idx, sperm in current.iterrows():

            i = int(sperm['sperm'])
            xc = sperm["x"]
            yc = sperm["y"]

            x1 = xc - 5
            x2 = xc + 5
            y1 = yc - 5
            y2 = yc + 5

            if x >= x1 and x <= x2 and y >= y1 and y <= y2:
                print(f'Sperm {i} clicked')
                current_sperm = i
                break

def runLabeler(video, data):

    global current_sperm

    # Open the video file
    cap = cv.VideoCapture(videofile)

    # Capture the first frame
    width = cap.get(cv.CAP_PROP_FRAME_WIDTH )
    height = cap.get(cv.CAP_PROP_FRAME_HEIGHT )
    fps =  cap.get(cv.CAP_PROP_FPS)
    num_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    video_original = utils.loadVideo(videofile)

    wait_time = 1/fps

    # Create some random colors
    max_index = data['sperm'].max()
    colors = np.random.randint(0, 255, (2*max_index, 3)) # Allows for splitting and creating new sperm

    video = visualizer.createVisualization(video_original,data,visualization="flow", colors=colors)

    frame_num = 0

    cv.namedWindow('Labeler', cv.WINDOW_NORMAL)
    cv.resizeWindow('Labeler', int(width), int(height))

    playvid = True

    # Play through video until space is pressed
    while True:

        cv.setMouseCallback('Labeler', onMouse, [data,frame_num])

        if playvid:
            time.sleep(wait_time)

            frame_num += 1
            frame_num = frame_num % num_frames

            frame = video[frame_num]

            cv.imshow('Labeler', frame)

        key = cv.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if key == ord(' ') or key == ord('k'):
            playvid = not playvid

        if key == ord('l'):
            frame_num += 1
            frame_num = frame_num % num_frames
            frame = video[frame_num]
            cv.imshow('Labeler', frame)

        if key == ord('j'):
            frame_num -= 1
            frame_num = frame_num % num_frames
            frame = video[frame_num]
            cv.imshow('Labeler', frame)

    cv.destroyAllWindows()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Show the tracked cells in a video')
    parser.add_argument('--videofile', type=str, default=None, help='Path to the video file')
    parser.add_argument('--csvfile', type=str, default=None, help='Path to the csvfile')

    videofile = parser.parse_args().videofile
    csvfile = parser.parse_args().csvfile

    if videofile is None:
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        videofile = filedialog.askopenfilename()

        if videofile:
            print("Selected file:", videofile)

    if csvfile is None:
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        csvfile = filedialog.askopenfilename()

        if csvfile:
            print("Selected file:", csvfile)

    # Load dataframe
    dataframe = utils.loadDataFrame(csvfile,convert_segmentation=False)

    runLabeler(videofile,dataframe)
    
