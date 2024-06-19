import argparse
import os
import json
import pickle
import numpy as np
import cv2 as cv
import ast
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
            mask = cv.line(mask, (prev_x, prev_y), (x, y), colors[i].tolist(), 2)
            mask = cv.circle(mask, (x, y), 2, colors[i].tolist(), -1)

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

    mask = np.zeros_like(frame)

    # Get only data for the current frame
    current = data[data['frame'] == frame_num]

    for row_idx, sperm in current.iterrows():
        i = sperm['sperm']
        segm = sperm['segmentation']

        if segm is not None:
            segm = np.array(segm)
            color = colors[i]
            if len(segm) > 0:
                mask[segm[:,0],segm[:,1]] = color

    img = cv.add(frame, mask)

    return img

def colorSpeed(frame, data, frame_num):
    """
    Color the frame based on the average path velocity (VAP) of sperm cells.

    Parameters:
    frame (np.array): The image frame to be colored.
    data (pd.DataFrame): DataFrame containing sperm tracking data with columns 'sperm', 'frame', 'segmentation', and 'VAP'.
    frame_num (int): The frame number to process.

    Returns a np.array
    """
    # Create a mask the same size as the frame
    mask = np.zeros_like(frame)

    # Define specific colors for each speed category
    color_static = np.array([255, 0, 0], dtype=np.uint8)  # Red for static
    color_slow = np.array([128, 0, 128], dtype=np.uint8)  # Purple for slow
    color_medium = np.array([0, 255, 0], dtype=np.uint8)  # Green for medium
    color_fast = np.array([0, 0, 255], dtype=np.uint8)  # Blue for fast

    # Get only data for the current frame
    current = data[data['frame'] == frame_num]

    # Determine the VAP thresholds for coloring
    if not current.empty:
        vap_values = current['VAP']
        lower_threshold = vap_values.quantile(0.33)
        upper_threshold = vap_values.quantile(0.66)
    else:
        lower_threshold = upper_threshold = 0

    for row_idx, sperm in current.iterrows():
        segm = sperm['segmentation']

        if segm is not None:
            # Convert the segmentation string to a list of lists
            segm = ast.literal_eval(segm)
            # Convert the list to a numpy array
            segm = np.array(segm, dtype=int)

            if segm.ndim != 2 or segm.shape[1] != 2:
                continue

            vap = sperm['VAP']

            # Determine the color based on the VAP value
            if vap == 0.5:
                color = color_static  # Red for static sperm
            elif vap < lower_threshold:
                color = color_slow  # Purple for lower 33%
            elif vap < upper_threshold:
                color = color_medium  # Green for middle 33%
            else:
                color = color_fast  # Blue for upper 33%

            if len(segm) > 0:
                try:
                    mask[segm[:, 0], segm[:, 1]] = color
                except IndexError as e:
                    print(f"IndexError: {e} for segmentation: {segm} and color: {color}")

    img = cv.add(frame, mask)

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

    # Calculate global VAP thresholds
    #vap_values = data['VAP']
    #lower_threshold = vap_values.quantile(0.33)
    #upper_threshold = vap_values.quantile(0.66)


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

        elif visualization == "speed":
            img = colorSpeed(frame,data,frame_num)

        elif visualization == "original":
            img = frame
        else:
            raise ValueError("Unknown visualization type")

        bgr_img = cv.cvtColor(img, cv.COLOR_RGB2BGR)

        if savefile is not None:
            result_vid.write(bgr_img)

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
    parser.add_argument('csvfile', type=str, help='Path to the csvfile')
    parser.add_argument('--output', type=str, help='Path to the output file', default=None)

    visualization = parser.parse_args().visualization
    videofile = parser.parse_args().videofile
    csvfile = parser.parse_args().csvfile

    if visualization == "segments" or visualization == "coloring":
        convert_segs = True
    else:
        convert_segs = False

    # Load dataframe
    dataframe = utils.loadDataFrame(csvfile,convert_segmentation=convert_segs)

    savefile = parser.parse_args().output
    if savefile is None:
        savefile = "output_" + visualization + ".mp4"

    runVisualization(videofile,dataframe,visualization,savefile)

