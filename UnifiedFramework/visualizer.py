import argparse
import os
import json
import pickle
import numpy as np
import cv2 as cv

import utils

def opticalFlow(frame,data,frame_num,mask,colors):

     # Get data for the current frame and previous frame
    current = data[data['frame'] == frame_num]
    prev = data[data['frame'] == frame_num-1]

    # Iterate over the sperm
    for row_idx, sperm in current.iterrows():
        i = sperm['sperm']
        x = int(sperm['x'])
        y = int(sperm['y'])
        prev_sperm = prev[prev['sperm'] == i]
        if len(prev_sperm) > 0:
            prev_sperm = prev_sperm.iloc[0] # Fail safe for duplicate sperm ids
            prev_x = int(prev_sperm['x'])
            prev_y = int(prev_sperm['y'])

            # Verify everything is in bounds
            #x = min(max(x, 0), frame.shape[1])
            #y = min(max(y, 0), frame.shape[0])
            #prev_x = min(max(prev_x, 0), frame.shape[1])
            #prev_y = min(max(prev_y, 0), frame.shape[0])

            # Draw
            mask = cv.line(mask, (prev_x, prev_y), (x, y), colors[int(i)].tolist(), 2)
            mask = cv.circle(mask, (x, y), 2, colors[int(i)].tolist(), -1)

    img = cv.add(frame, mask)

    return img

def boundingBoxes(frame,data,frame_num):
    
    mask = np.zeros_like(frame)

    # Get only data for the current frame
    current = data[data['frame'] == frame_num]

    # Iterate over the sperm
    for row_idx, sperm in current.iterrows():
        x = int(sperm['bbox_x'])
        y = int(sperm['bbox_y'])
        w = int(sperm['bbox_w'])
        h = int(sperm['bbox_h'])
        mask = cv.rectangle(mask, (x, y), (x + w, y + h), (0, 128, 0), 3)

    img = cv.add(frame, mask)

    return img

def coloring(frame,data,frame_num,colors):

    locs = np.zeros((frame.shape[0],frame.shape[1]))
    mask = np.zeros_like(frame)

    # Get only data for the current frame
    current = data[data['frame'] == frame_num]

    if "segmentation" not in current.columns:
        raise ValueError("Segmentation column not be found in dataframe. Check that it was computed and loaded.")

    for row_idx, sperm in current.iterrows():
        i = sperm['sperm']
        segm = sperm['segmentation']

        if segm is not None:
            segm = np.array(segm)
            color = colors[i]
            if len(segm) > 0:
                locs[segm[:,0],segm[:,1]] = 1
                mask[segm[:,0],segm[:,1]] = color

    #img = cv.add(frame, mask)
    img = np.copy(frame)
    img[np.where(locs == 1)] = mask[np.where(locs == 1)]

    return img

def runVisualization(videofile, data, visualization="flow",savefile=None):

    # Open the video file
    cap = cv.VideoCapture(videofile)

    # Capture the first frame
    width = cap.get(cv.CAP_PROP_FRAME_WIDTH )
    height = cap.get(cv.CAP_PROP_FRAME_HEIGHT )
    fps =  cap.get(cv.CAP_PROP_FPS)

    # Create a video writer
    if savefile is not None:
        result_vid = cv.VideoWriter(savefile,cv.VideoWriter_fourcc(*'mp4v'),fps,(int(width),int(height)))

    # Create some random colors
    num_sperm = data['sperm'].nunique()
    max_index = data['sperm'].max()
    colors = np.random.randint(0, 255, (max_index+1, 3))
    print(colors.shape)

    if visualization == "flow":
        ret, frame = cap.read()
        # Create a mask image for drawing purposes
        mask = np.zeros_like(frame)
        frame_num = 1
    else:
        frame_num = 0

    while(1):
        ret, frame = cap.read()
        if not ret:
            print("Video Finished.")
            break

        if visualization == "flow":
            img = opticalFlow(frame,data,frame_num,mask,colors)

        elif visualization == "bbox":
            img = boundingBoxes(frame,data,frame_num)

        elif visualization == "segments" or visualization == "coloring":
            img = coloring(frame,data,frame_num,colors)

        elif visualization == "original":
            img = frame
        else:
            raise ValueError("Unknown visualization type")

        if savefile is not None:
            result_vid.write(img)

        cv.imshow('frame', img)
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break

        frame_num += 1

    if savefile is not None:
        result_vid.release()

    cv.destroyAllWindows()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Show the tracked cells in a video')
    parser.add_argument('visualization', type=str, help='Type of visualization to create')
    parser.add_argument('videofile', type=str, help='Path to the video file')
    parser.add_argument('--csv', type=str, default = None, help='Path to the csvfile')
    parser.add_argument('--output', type=str, default = None, help='Path to the output file')

    visualization = parser.parse_args().visualization
    videofile = parser.parse_args().videofile
    csvfile = parser.parse_args().csv
    savefile = parser.parse_args().output

    if csvfile is None:
        csvfile = ".".join(videofile.split('.')[:-1]) + '.csv'

    if visualization == "segments" or visualization == "coloring":
        convert_segs = True
    else:
        convert_segs = False

    # Load dataframe
    dataframe = utils.loadDataFrame(csvfile,convert_segmentation=convert_segs)

    if savefile is None:
        savefile = "output_" + visualization + ".mp4"

    runVisualization(videofile,dataframe,visualization,savefile)
    
