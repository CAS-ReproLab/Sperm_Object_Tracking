import argparse
import os
import json
import pickle
import numpy as np
import cv2 as cv

def opticalFlow(frame,trackdata,frame_num,mask,colors):

    for i,sperm in enumerate(trackdata):
        if sperm["visible"][frame_num] == 1 and sperm["visible"][frame_num-1] == 1:
            curr = sperm['centroid'][frame_num]
            prev = sperm['centroid'][frame_num-1]
            #curr = sperm['centroid'][str(frame_num)] # .json
            #prev = sperm['centroid'][str(frame_num-1)] #.json

            mask = cv.line(mask, (int(prev[0]), int(prev[1])), (int(curr[0]), int(curr[1])), colors[i].tolist(), 2)
            frame = cv.circle(frame, (int(curr[0]), int(curr[1])), 5, colors[i].tolist(), -1)
    img = cv.add(frame, mask)

    return img

def boundingBoxes(frame,trackdata,frame_num):
    
    mask = np.zeros_like(frame)

    for sperm in trackdata:
        if sperm["visible"][frame_num] == 1:
            bbox = sperm['bbox'][frame_num] # .pkl
            # bbox = sperm['bbox'][str(frame_num)]  # .json
            x = int(bbox[0])
            y = int(bbox[1])
            w = int(bbox[2])
            h = int(bbox[3])
            mask = cv.rectangle(mask, (x, y), (x + w, y + h), (0, 128, 0), 3)
    img = cv.add(frame, mask)

    return img

def coloring(frame,trackdata,frame_num,colors):

    mask = np.zeros_like(frame)

    for i,sperm in enumerate(trackdata):
        if sperm["visible"][frame_num] == 1:
            segm = np.array(sperm['segmentation'][frame_num]) # .pkl
            #segm = np.array(sperm['segmentation'][str(frame_num)]) # .json
            color = colors[i]
            mask[segm[:,0],segm[:,1]] = color
    img = cv.add(frame, mask)

    return img


def colorSpeed(frame, trackdata, statsdata, frame_num):
    mask = np.zeros_like(frame)

    # Define specific colors for each speed category
    color_static = np.array([255, 0, 0], dtype=np.uint8)  # Red for static
    color_slow = np.array([128, 0, 128], dtype=np.uint8)  # Purple for slow
    color_medium = np.array([0, 255, 0], dtype=np.uint8)  # Green for medium
    color_fast = np.array([0, 0, 255], dtype=np.uint8)  # Blue for fast

    # Extract average speeds from statsdata
    speeds = [statsdata[i]['average_speed'] for i in range(len(statsdata))]

    # Calculate speed thresholds for categorization
    slow_threshold = 2
    medium_threshold = 4
    #fast_threshold = np.percentile(speeds, 75)

    #slow_threshold = np.percentile(speeds, 25)
    #medium_threshold = np.percentile(speeds, 50)
    #fast_threshold = np.percentile(speeds, 75)


    # Determine the color for each sperm based on
    # its average speed
    sperm_colors = []
    for speed in speeds:
        if speed == 0:
            color = color_static  # Red for static
            category = "static"
        elif 0 < speed <= slow_threshold:
            color = color_slow  # Purple for slow
            category = "slow"
        elif slow_threshold < speed <= medium_threshold:
            color = color_medium  # Green for medium
            category = "medium"
        else:
            color = color_fast  # Blue for fast
            category = "fast"
        sperm_colors.append(color)
        print(f"Sperm speed: {speed}, Category: {category}, Color: {color}")

    # Assign colors based on speed and apply to mask where visible and segmented
    for i, sperm in enumerate(trackdata):
        if sperm["visible"][frame_num] == 1:
            segm = np.array(sperm['segmentation'][frame_num])

            if segm.ndim != 2 or segm.shape[1] != 2:
                print(f"Invalid segmentation data format at frame {frame_num} for sperm index {i}")
                continue  # Skip this entry

            # Use precomputed color for this sperm
            color = sperm_colors[i]
            mask[segm[:, 0], segm[:, 1]] = color

    # Combine original frame with mask
    img = cv.add(frame, mask)

    return img


parser = argparse.ArgumentParser(description='Show the tracked cells in a video')
parser.add_argument('visualization', type=str, help='Type of visualization to create')
parser.add_argument('videofile', type=str, help='Path to the video file')

visualization = parser.parse_args().visualization
videofile = parser.parse_args().videofile

trackfile = videofile.split('.')[0] + '_tracked.pkl'
statsfile = videofile.split('.')[0] + '_stats.pkl'

#trackfile = videofile.split('.')[0] + '_tracked.json'
#statsfile = videofile.split('.')[0] + '_stats.json'

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

# Load the json files
#if os.path.exists(trackfile):
#    with open(trackfile, 'r') as f:
#        trackdata = json.load(f)
#else:
#    trackdata = None

#if os.path.exists(statsfile):
#    with open(statsfile, 'r') as f:
#        statsdata = json.load(f)
#else:
#    statsdata = None

# Open the video file
cap = cv.VideoCapture(videofile)

# Capture the first frame
width = cap.get(cv.CAP_PROP_FRAME_WIDTH )
height = cap.get(cv.CAP_PROP_FRAME_HEIGHT )
fps =  cap.get(cv.CAP_PROP_FPS)

# Create a video writer
result_vid = cv.VideoWriter("output_" + visualization + ".mp4",cv.VideoWriter_fourcc(*'mp4v'),fps,(int(width),int(height)))

# Create some random colors
num_sperm = len(trackdata)
colors = np.random.randint(0, 255, (num_sperm, 3))

if visualization == "flow":
    ret, frame = cap.read()
    if not ret:
        raise Exception("Failed to read the first frame.")
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
        img = opticalFlow(frame,trackdata,frame_num,mask,colors)

    elif visualization == "bbox":
        img = boundingBoxes(frame,trackdata,frame_num)

    elif visualization == "segments":
        img = coloring(frame,trackdata,frame_num,colors)

    elif visualization == "speed":
        img = colorSpeed(frame, trackdata, statsdata, frame_num)

    elif visualization == "original":
        img = frame

    bgr_img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    result_vid.write(img)

    cv.imshow('frame', img)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

    frame_num += 1

result_vid.release()

cv.destroyAllWindows()
