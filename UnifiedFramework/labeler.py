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
    # Change all sperm2 to sperm1 in Pandas file
    data.loc[data['sperm'] == sperm2, 'sperm'] = sperm1
    return data

def splitSperm(data,sperm1,frame_num):
    # Make a new sperm with a new ID at frame_num with sperm1's data
    data.loc[(data['sperm'] == sperm1) & (data['frame']>frame_num), 'sperm'] = data['sperm'].max() + 1
    return data

def deleteSperm(data,sperm1):
    # Delete all rows with sperm1
    data = data[data['sperm'] != sperm1]
    return data

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
    print(
    '''
    This program runs the manual labeler for the sperm data. Here are the controls:
    k: Play/Pause
    l: Next frame
    j: Previous frame
    t: Mark track as "Keep" for current sperm
    u: Mark track as "Unusable" for current sperm
    s: Split the current sperm
    d: Delete the current sperm
    m: Merge two sperm
    r: Randomize colors
    o: Toggle original video
    q: Quit
    Mouse Click: Identify sperm number (make sure to click on the last location of the sperm in the frame)
    '''
    )

    global current_sperm

    # Add "keep" column to dataframe, assume all unusable
    data['keep'] = 0

    savefile = csvfile.replace(".csv","_corrected.csv")
    keepfile = csvfile.replace(".csv","_keep.csv")

    # Open the video file
    cap = cv.VideoCapture(videofile)

    # Capture the first frame
    width = cap.get(cv.CAP_PROP_FRAME_WIDTH )
    height = cap.get(cv.CAP_PROP_FRAME_HEIGHT )
    fps =  cap.get(cv.CAP_PROP_FPS)
    num_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    cap.release()

    video_original = utils.loadVideo(videofile)

    wait_time = 1/fps

    # Create some random colors
    max_index = data['sperm'].max()
    colors = utils.generateRandomColors(2*max_index) # Allows for splitting and creating new sperm
    #colors = np.random.randint(100, 255, (2*max_index, 3)) # Allows for splitting and creating new sperm

    video = visualizer.createVisualization(video_original,data,visualization="flow", colors=colors)

    frame_num = 0

    cv.namedWindow('Labeler', cv.WINDOW_NORMAL)
    cv.resizeWindow('Labeler', int(width), int(height))

    playvid = True
    show_original = False

    # Play through video until space is pressed
    while True:

        cv.setMouseCallback('Labeler', onMouse, [data,frame_num])

        if playvid:
            time.sleep(wait_time)

            frame_num += 1
            frame_num = frame_num % num_frames

            if show_original:
                frame = video_original[frame_num]
                cv.imshow('Labeler', frame)
            else:
                frame = video[frame_num]
                cv.imshow('Labeler', frame)

            cv.imshow('Labeler', frame)

        key = cv.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if key == ord(' ') or key == ord('k'):
            playvid = not playvid

        if key == ord('l'):
            frame_num += 1
            frame_num = frame_num % num_frames
            if show_original:
                frame = video_original[frame_num]
                cv.imshow('Labeler', frame)
            else:
                frame = video[frame_num]
                cv.imshow('Labeler', frame)
            cv.imshow('Labeler', frame)

        if key == ord('j'):
            frame_num -= 1
            frame_num = frame_num % num_frames
            if show_original:
                frame = video_original[frame_num]
                cv.imshow('Labeler', frame)
            else:
                frame = video[frame_num]
                cv.imshow('Labeler', frame)
            cv.imshow('Labeler', frame)

        if key == ord('s'):
            test = input(f"Split {current_sperm} at frame #{frame_num}? (y/n): ")
            if test == 'y':
                print("Processing...")
                data = splitSperm(data,current_sperm,frame_num)
                utils.saveDataFrame(data,savefile)
                video = visualizer.createVisualization(video_original,data,visualization="flow", colors=colors)
                show_original = False
                frame = video[frame_num]
                cv.imshow('Labeler', frame)
                print("Done!")

        if key == ord('d'):
            test = input(f"Delete {current_sperm}? (y/n): ")
            if test == 'y':
                print("Processing...")
                data = deleteSperm(data,current_sperm)
                utils.saveDataFrame(data,savefile)
                video = visualizer.createVisualization(video_original,data,visualization="flow", colors=colors)
                show_original = False
                frame = video[frame_num]
                cv.imshow('Labeler', frame)
                print("Done!")

        if key == ord('m'):
            sperm1 = int(input("Enter the first sperm to merge: "))
            sperm2 = int(input("Enter the second sperm to merge: "))
            test = input(f"Merge {sperm1} and {sperm2}? (y/n): ")
            if test == 'y':
                print("Processing...")
                data = mergeSperm(data,sperm1,sperm2)
                utils.saveDataFrame(data,savefile)
                video = visualizer.createVisualization(video_original,data,visualization="flow", colors=colors)
                show_original = False
                frame = video[frame_num]
                cv.imshow('Labeler', frame)
                print("Done!")

        if key == ord('t'):
            test = input("Keep track for sperm " + str(current_sperm) + "? (y/n): ")
            if test == 'y':
                data.loc[data['sperm'] == current_sperm, 'keep'] = 1
                utils.saveDataFrame(data,savefile)
                print("Track marked for sperm " + str(current_sperm) + " in savefile.")
        
        if key == ord('u'):
            test = input("Mark track unusable for sperm " + str(current_sperm) + "? (y/n): ")
            if test == 'y':
                data.loc[data['sperm'] == current_sperm, 'keep'] = 0
                utils.saveDataFrame(data,savefile)
                print("Track marked unusable for sperm " + str(current_sperm) + " in savefile.")

        if key == ord('r'):
            print("Randomizing colors...")
            colors = utils.generateRandomColors(2*max_index)
            video = visualizer.createVisualization(video_original,data,visualization="flow", colors=colors)
            show_original = False
            frame = video[frame_num]
            cv.imshow('Labeler', frame)
            print("Done!")

        if key == ord("o"):
            show_original = not show_original
            if show_original:
                frame = video_original[frame_num]
                cv.imshow('Labeler', frame)
            else:
                frame = video[frame_num]
                cv.imshow('Labeler', frame)
            

    # Save output dataframe with only "keep" values
    data = data[data['keep'] == 1]
    utils.saveDataFrame(data,keepfile)

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
    
