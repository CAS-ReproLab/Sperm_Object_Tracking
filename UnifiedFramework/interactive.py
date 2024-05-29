import argparse
import os
import json
import pickle
import numpy as np
import cv2 as cv
import time

current_sperm = None

def onMouse(event, x, y, flags, param):

    global current_sperm

    trackdata = param[0]
    frame_num = param[1]

    if event == cv.EVENT_LBUTTONDOWN:
        for i,sperm in enumerate(trackdata):
            if sperm["visible"][frame_num] == 1:
                curr = sperm['bbox'][frame_num]
                x1 = curr[0]
                y1 = curr[1]
                x2 = curr[0] + curr[2]
                y2 = curr[1] + curr[3]
                if x >= x1 and x <= x2 and y >= y1 and y <= y2:
                    print(f'Sperm {i} clicked')
                    current_sperm = i
                    break

                #curr = sperm['centroid'][frame_num]
                #dist = np.sqrt((curr[0]-x)**2 + (curr[1]-y)**2)
                #if dist < 5:
                #    print(f'Sperm {i} clicked')
                #    current_sperm = i

def runInteractive(videofile,trackdata,statsdata=None):

    global current_sperm

    # Open the video file
    cap = cv.VideoCapture(videofile)

    # Capture the first frame
    width = cap.get(cv.CAP_PROP_FRAME_WIDTH )
    height = cap.get(cv.CAP_PROP_FRAME_HEIGHT )
    fps =  cap.get(cv.CAP_PROP_FPS)

    wait_time = 1/30

    # Create some random colors
    num_sperm = len(trackdata)

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

        cv.setMouseCallback('Interactive', onMouse, [trackdata,frame_num])

        if current_sperm is not None:
            sperm = trackdata[current_sperm]
            if sperm["visible"][frame_num] == 1:
                bbox = sperm['bbox'][frame_num]
                x = int(bbox[0])
                y = int(bbox[1])
                w = int(bbox[2])
                h = int(bbox[3])
                img = cv.rectangle(img, (x, y), (x + w, y + h), (0, 128, 0), 3)

        cv.imshow('Interactive', img)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        frame_num += 1

        time.sleep(wait_time)

    cv.destroyAllWindows()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Show the tracked cells in a video')
    parser.add_argument('videofile', type=str, help='Path to the video file')

    videofile = parser.parse_args().videofile

    trackfile = videofile.split('.')[0] + '_tracked.pkl'
    statsfile = videofile.split('.')[0] + '_stats.pkl'

    # Load the pkl files
    if os.path.exists(trackfile):
        with open(trackfile, 'rb') as f:
            trackdata = pickle.load(f)
    else:
        trackdata = None

    if os.path.exists(statsfile):
        with open(statsfile, 'rb') as f:
            statsdata = pickle.load(f)
    else:
        statsdata = None
    
    runInteractive(videofile,trackdata,statsdata)
    
