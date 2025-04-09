import argparse
import os
import json
import pickle
import numpy as np
import cv2 as cv
import time

import utils

import tkinter as tk
from tkinter import filedialog

current_sperm = None

def onMouse(event, x, y, flags, param):

    global current_sperm

    data = param[0]
    frame_num = param[1]

    if event == cv.EVENT_LBUTTONDOWN:

        # Get only data for the current frame
        current = data[data['frame'] == frame_num]

        for idx, sperm in current.iterrows():

            i = sperm['sperm']
            
            xc = int(sperm['x'])
            yc = int(sperm['y'])

            if (x >= xc - 5 and x <= xc + 5) and (y >= yc - 5 and y <= yc + 5):
                print(f'Sperm {i} clicked')
                current_sperm = i
                break

            #x1 = sperm['bbox_x']
            #y1 = sperm['bbox_y']
            #x2 = sperm['bbox_x'] + sperm['bbox_w']
            #y2 = sperm['bbox_y'] + sperm['bbox_h']

            #if x >= x1 and x <= x2 and y >= y1 and y <= y2:
            #    print(f'Sperm {i} clicked')
            #    current_sperm = i
            #    break

def runInteractive(videofile, data):

    global current_sperm
    global average_speed

    # Open the video file
    cap = cv.VideoCapture(videofile)

    # Capture the first frame
    width = cap.get(cv.CAP_PROP_FRAME_WIDTH )
    height = cap.get(cv.CAP_PROP_FRAME_HEIGHT )
    fps =  cap.get(cv.CAP_PROP_FPS)

    wait_time = 1/30

    # Create some random colors
    num_sperm = data['sperm'].nunique()
    max_index = data['sperm'].max()

    frame_num = 0

    cv.namedWindow('Interactive', cv.WINDOW_NORMAL)
    cv.resizeWindow('Interactive', int(width), int(height))

    # Loop through the video
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            cap.set(cv.CAP_PROP_POS_FRAMES, 0)
            frame_num = 0
            continue

        img = frame

        cv.setMouseCallback('Interactive', onMouse, [data,frame_num])

        # Get only data for the current frame
        current = data[data['frame'] == frame_num]

        sperm = current[current['sperm'] == current_sperm]
        if len(sperm) > 0:
            sperm = sperm.iloc[0] # Fail safe for duplicate sperm ids
            xc = int(sperm['x'])
            yc = int(sperm['y'])
            img = cv.circle(img, (xc, yc), 5, (0, 255, 0), -1)

            #x = int(sperm['bbox_x'])
            #y = int(sperm['bbox_y'])
            #w = int(sperm['bbox_w'])
            #h = int(sperm['bbox_h'])
            #img = cv.rectangle(img, (x, y), (x + w, y + h), (0, 128, 0), 3)

            #average_speed = sperm['average_speed']
            vap = sperm['VAP']
            vcl = sperm['VCL']
            vsl = sperm['VSL']
            alh = sperm['ALH_mean']
            bcf = sperm['BCF']

        if current_sperm is not None:

            # Output text properties
            font = cv.FONT_HERSHEY_SIMPLEX
            org = (50, 50)
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2
            text1 = f'Sperm {current_sperm}'
            text2 = f'VAP: {vap:.2f} microns/s'
            text3 = f'VCL: {vcl:.2f} microns/s'
            text4 = f'VSL: {vsl:.2f} microns/s'
            img = cv.putText(img, text1, org, font, fontScale, color, thickness, cv.LINE_AA)
            img = cv.putText(img, text2, (org[0], org[1]+50), font, fontScale, color, thickness, cv.LINE_AA)
            img = cv.putText(img, text3, (org[0], org[1]+100), font, fontScale, color, thickness, cv.LINE_AA)
            img = cv.putText(img, text4, (org[0], org[1]+150), font, fontScale, color, thickness, cv.LINE_AA)


        cv.imshow('Interactive', img)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        frame_num += 1

        time.sleep(wait_time)

    cv.destroyAllWindows()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Show the tracked cells in a video')
    parser.add_argument('--videofile', type=str, help='Path to the video file')
    parser.add_argument('--csvfile', type=str, help='Path to the csvfile')

    videofile = parser.parse_args().videofile
    csvfile = parser.parse_args().csvfile

    if videofile is None:
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        videofile = filedialog.askopenfilename(title="Select the video file")

        if videofile:
            print("Selected file:", videofile)

    if csvfile is None:
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        csvfile = filedialog.askopenfilename(title="Select the CSV file")

        if csvfile:
            print("Selected file:", csvfile)

    # Load dataframe
    dataframe = utils.loadDataFrame(csvfile,convert_segmentation=False)
    
    runInteractive(videofile,dataframe)
    
