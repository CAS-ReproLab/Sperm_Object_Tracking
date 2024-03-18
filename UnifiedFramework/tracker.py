import numpy as np
import cv2 as cv
from scipy.optimize import linear_sum_assignment
#import pickle
import json

import argparse
from tqdm import tqdm, trange

def detect(frame, low_thresh=50, high_thresh=255, kernel_size=(3,3)):
    """
    Detects cells in a frame
    """
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    _, bw = cv.threshold(gray,low_thresh,high_thresh,cv.THRESH_BINARY)
    kernel = np.ones(kernel_size,np.uint8)
    bw = cv.morphologyEx(bw, cv.MORPH_OPEN, kernel)
    num_labels, label_im, stats, centroids = cv.connectedComponentsWithStats(bw, 4, cv.CV_32S) 

    # Seperate bbox from area
    areas = stats[:,4]
    bboxs = stats[:,0:4]

    # Filter out the background (always index 0)
    areas = areas[1:]
    bboxs = bboxs[1:]
    centroids = centroids[1:]
    label_im -= 1

    return centroids, label_im, bboxs, areas

def track(prev_centroids, centroids, thresh=10):

    # Get the number of centroids
    num_centroids = centroids.shape[0]
    num_prev_centroids = prev_centroids.shape[0]

    # Create a cost matrix
    cost_matrix = np.zeros((num_prev_centroids,num_centroids))

    # Fill in the cost matrix
    for i in range(num_prev_centroids):
        for j in range(num_centroids):
            cost_matrix[i,j] = np.linalg.norm(centroids[j] - prev_centroids[i])

    # Solve the assignment problem wiht Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Make mapping from previous centroids to current centroids (or -1 if no match)
    mapping = np.zeros(num_centroids) - 1

    for r,c in zip(row_ind,col_ind):
        if cost_matrix[r,c] < thresh:
            mapping[c] = r

    return mapping

def makeSperm():
    sperm = {}
    sperm['centroid'] = {}
    sperm['bbox'] = {} 
    sperm['area'] = {}
    sperm['segmentation'] = {}
    sperm['visible'] = []

    return sperm

def myStack(tuple):
    rows, cols = tuple
    return np.hstack((np.expand_dims(rows, axis=1), np.expand_dims(cols, axis=1)))


### Main Code ###

parser = argparse.ArgumentParser(description='Track cells in a video')
parser.add_argument('videofile', type=str, help='Path to the video file')

videofile = parser.parse_args().videofile
outputfile = videofile.split('.')[0] + '_tracked.json'

cap = cv.VideoCapture(videofile)

# Read the first frame
ret, first_frame = cap.read()

# Detect the cells in the first frame
centroids, label_im, bboxs, areas = detect(first_frame)

# Create a lists for the whole video
centroids_list = [centroids]
label_im_list = [label_im]
bboxs_list = [bboxs]
areas_list = [areas]
mappings = []

# Loop through the video to generate mappings
total_frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
pbar = tqdm(total=total_frame_count)

while True:
    # Read the next frame
    ret, frame = cap.read()

    # If the frame is None, then we have reached the end of the video
    if frame is None:
        break

    # Detect the cells in the frame
    centroids, label_im, bboxs, areas = detect(frame)

    # Track the cells
    mapping = track(centroids_list[-1], centroids)

    # Add the new centroids and label_im to the lists
    centroids_list.append(centroids)
    label_im_list.append(label_im)
    bboxs_list.append(bboxs)
    areas_list.append(areas)

    mappings.append(mapping)

    pbar.update(1)

# Close the video file
cap.release()
pbar.close()

all_sperm = []

# Process each individual sperm cell
for i in trange(len(centroids_list)):
    
    num_sperm = len(centroids_list[i])

    for j in range(num_sperm):
        # Go through every sperm in the first frame
        if i == 0:
            sperm = makeSperm()
        # Go through every newly discovered sperm in following frames
        elif mappings[i-1][j] == -1:
            sperm = makeSperm()
            sperm['visible'] = [0] * i
        # Don't double count previously discovered sperm
        else:
            continue

        # Add current frame properties
        sperm['centroid'][i] = centroids_list[i][j].tolist()
        sperm['bbox'][i] = bboxs_list[i][j].tolist()
        sperm['area'][i] = areas_list[i][j].tolist()
        sperm['segmentation'][i] = myStack(np.where(label_im_list[i] == j)).tolist()
        sperm['visible'].append(1)

        # Determine the sperm's properties in all subsequent frames
        cur_index = j
        for k in range(i+1, len(centroids_list)):
            new_index = np.where(mappings[k-1] == cur_index)[0]
            if new_index.size != 0:
                cur_index = new_index[0]
                sperm['visible'].append(1)
                sperm['centroid'][k] = centroids_list[k][cur_index].tolist()
                sperm['bbox'][k] = bboxs_list[k][cur_index].tolist()
                sperm['area'][k] = areas_list[k][cur_index].tolist()
                sperm['segmentation'][k] = myStack(np.where(label_im_list[k] == cur_index)).tolist()
            else:
                # The sperm is no longer visible and is no longer tracked
                for _ in range(k, len(centroids_list)):
                    sperm['visible'].append(0)
                break

        all_sperm.append(sperm)
                


# Save sperm data to pickle file
#with open(outputfile, 'wb') as f:
#    pickle.dump(all_sperm, f)

# Save sperm data to json file
with open(outputfile, 'w') as f:
    json.dump(all_sperm, f)

print(outputfile,' file saved')