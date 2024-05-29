import numpy as np
import cv2 as cv
from scipy.optimize import linear_sum_assignment
import pickle
import json

import argparse
from tqdm import tqdm, trange


def threshold(frame, method='otsu',global_thresh=50):
    
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    #mid_val = np.median(gray)
    #gray = np.abs(gray - mid_val).astype(np.uint8)

    if method == 'global':
        _, bw = cv.threshold(gray,global_thresh,255,cv.THRESH_BINARY)
    elif method == 'otsu':
        _, bw = cv.threshold(gray,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    elif method == 'adaptive':
        bw = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,11,-2)
    elif method == 'hybrid':
        _, bw1 = cv.threshold(gray,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

        bw2 = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,11,-2)
        bw = cv.bitwise_or(bw1,bw2)
    else:
        raise ValueError('Invalid thresholding method')
    
    return bw


def detect(frame, kernel_size=(3,3)):
    """
    Detects cells in a frame
    """
    
    # Find centroids by focusing on heads
    bw = threshold(frame, method='global', global_thresh=50)
    kernel = np.ones(kernel_size,np.uint8)
    bw = cv.morphologyEx(bw, cv.MORPH_OPEN, kernel)
    _, _, _, centroids = cv.connectedComponentsWithStats(bw, 4, cv.CV_32S) 

    # Run connected components again with a lower threshold to get the segmentation
    bw2 = threshold(frame, method='hybrid')
    _, label_im, stats, _ = cv.connectedComponentsWithStats(bw2, 4, cv.CV_32S)

    # Seperate bbox from area
    areas = stats[:,4]
    bboxs = stats[:,0:4]

    # Filter out the background (always index 0)
    areas = areas[1:]
    bboxs = bboxs[1:]
    centroids = centroids[1:]
    label_im -= 1

    # Turn label_im into list of segmentations
    segmentations = labelIm2Array(label_im, len(stats))

    # Associate centroids with correct segmentations
    new_segmentations = []
    new_areas = np.zeros(len(centroids), dtype=np.int32)
    new_bboxs = np.zeros((len(centroids),4), dtype=np.int32)
    for i, centroid in enumerate(centroids):
        r,c = int(centroid[1]),int(centroid[0])
        if r < 0 or c < 0 or r >= label_im.shape[0] or c >= label_im.shape[1]:
            print("Warning: Centroid found out of bounds")
            continue

        # Check the label of the four surrounding pixels    
        r2 = r+1 if r+1 < label_im.shape[0] else r
        c2 = c+1 if c+1 < label_im.shape[1] else c
        label_tl = label_im[r,c]
        label_tr = label_im[r,c2]
        label_bl = label_im[r2,c]
        label_br = label_im[r2,c2]
        
        if label_tl >= 0:
            label = label_tl
        elif label_tr >= 0:
            label = label_tr
        elif label_bl >= 0:
            label = label_bl
        else:
            label = label_br
            # TODO: Check mode of the four labels if they are greater than 1
        
        if label == -1:
            print("\n Warning: Centroid found in background")
            new_segmentations.append([-1,-1])

        new_segmentations.append(segmentations[label])
        new_areas[i] = areas[label]
        new_bboxs[i] = bboxs[label]

    # TODO Filter out crossing labels

    return centroids, new_segmentations, new_bboxs, new_areas

def track(prev_centroids, centroids, thresh=10):

    # Get the number of centroids
    num_centroids = centroids.shape[0]
    num_prev_centroids = prev_centroids.shape[0]

    # Fill in the cost matrix
    jv, iv = np.meshgrid(np.arange(num_centroids), np.arange(num_prev_centroids))
    cost_matrix = np.linalg.norm(centroids[jv] - prev_centroids[iv], axis=2)

    # Conceptually, the above code is equivalent to the following:
    #cost_matrix = np.zeros((num_prev_centroids,num_centroids))
    #for i in range(num_prev_centroids):
    #    for j in range(num_centroids):
    #        cost_matrix[i,j] = np.linalg.norm(centroids[j] - prev_centroids[i])

    # Solve the assignment problem wiht Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Make mapping from previous centroids to current centroids (or -1 if no match)
    mapping = np.zeros(num_centroids).astype(np.int64) - 1

    for r,c in zip(row_ind,col_ind):
        if cost_matrix[r,c] < thresh:
            mapping[c] = r

    return mapping

def processVideo(videofile):

    cap = cv.VideoCapture(videofile)

    # Read the first frame
    ret, first_frame = cap.read()

    if not ret:
        raise ValueError('Error: Could not read the video file or video file could not be found')


    # Detect the cells in the first frame
    centroids, segmentations, bboxs, areas = detect(first_frame)

    # Create a lists for the whole video
    centroids_list = [centroids]
    segmentations_list = [segmentations]
    bboxs_list = [bboxs]
    areas_list = [areas]

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
        centroids, segmentations, bboxs, areas = detect(frame)

        # Add the new centroids and properties to the lists
        centroids_list.append(centroids)
        segmentations_list.append(segmentations)
        bboxs_list.append(bboxs)
        areas_list.append(areas)

        pbar.update(1)

    # Close the video file
    cap.release()
    pbar.close()

    return centroids_list, segmentations_list, bboxs_list, areas_list

def trackCentroids(centroids_list):

    # Create a list to store the mappings
    mappings = []

    # Loop through the video to generate mappings
    for i in trange(1, len(centroids_list)):
        mapping = track(centroids_list[i-1], centroids_list[i])
        mappings.append(mapping)

    return mappings

def generateTrackingData(mappings, centroids_list, segmentations_list, bboxs_list, areas_list):

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
            sperm['segmentation'][i] = segmentations_list[i][j]
            sperm['visible'].append(1)

            # Determine the sperm's properties in all subsequent frames
            cur_index = j
            for k in range(i+1, len(centroids_list)):
                new_index = np.where(mappings[k-1] == cur_index)[0] # k-1 because their is one less mapping than centroids
                if new_index.size != 0:
                    cur_index = new_index[0]
                    sperm['visible'].append(1)
                    sperm['centroid'][k] = centroids_list[k][cur_index].tolist()
                    sperm['bbox'][k] = bboxs_list[k][cur_index].tolist()
                    sperm['area'][k] = areas_list[k][cur_index].tolist()
                    sperm['segmentation'][k] = segmentations_list[k][cur_index]
                else:
                    # The sperm is no longer visible and is no longer tracked
                    for _ in range(k, len(centroids_list)):
                        sperm['visible'].append(0)
                    break

            all_sperm.append(sperm)
    
    return all_sperm



def labelIm2Array(label_im, num_labels):
    segmentations = []
    for i in range(0, num_labels):
        segmentations.append([])

    rows, cols = label_im.shape
    for i in range(rows):
        for j in range(cols):
            if label_im[i,j] != -1:
                segmentations[label_im[i,j]].append([i,j])

    return segmentations

def makeSperm():
    sperm = {}
    sperm['centroid'] = {}
    sperm['bbox'] = {} 
    sperm['area'] = {}
    sperm['segmentation'] = {}
    sperm['visible'] = []

    return sperm


### Main Code ###
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Track cells in a video')
    parser.add_argument('videofile', type=str, help='Path to the video file')

    videofile = parser.parse_args().videofile

    centroids_list, segmentations_list, bboxs_list, areas_list = processVideo(videofile)
    mappings = trackCentroids(centroids_list)
    all_sperm = generateTrackingData(mappings, centroids_list, segmentations_list, bboxs_list, areas_list)

    # Save sperm data to pickle file
    outputfile = videofile.split('.')[0] + '_tracked.pkl'
    with open(outputfile, 'wb') as f:
        pickle.dump(all_sperm, f)

    # Save sperm data to json file
    #outputfile = videofile.split('.')[0] + '_tracked.json'
    #with open(outputfile, 'w') as f:
    #    json.dump(all_sperm, f)

    print(outputfile,' file saved')