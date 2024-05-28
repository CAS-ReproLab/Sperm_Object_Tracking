import numpy as np
import cv2 as cv
#import json
import pickle
import argparse
from math import sqrt

def calcAverageSpeed(centroids, visible, fps=30):
    """
    Calculate the average speed of a sperm cell
    """
    count = sum(visible) -1
    total = 0

    if count == 0:
        return 0

    for i in range(len(visible)):
        if visible[i] == 1 and visible[i-1] == 1 and i-1 >= 0:
            start = centroids[i-1]
            end = centroids[i]
            total += sqrt((end[1]-start[1])**2 + (end[0]-start[0])**2)

    # Calculate the average speed
    average_speed = (total/count)/fps

    return average_speed


def averagePathVelocity(centroids, visible, pix_size, win_size, fps=5):
    """
    Calculate the average path velocity (VAP)
    FPS frames per a second
    pixel size is determined by pix_size
    Window size is determined by win_size and is a user input for the units of frame to include in the averaging
    argument for VCl and VAP calculations
    """
    # Initialize total distance and count of valid frames
    total_distance = 0
    valid_frame_count = len(visible) - 1

    # Iterate over visible frames to calculate distances
    for i in range(1, len(visible)):
        if visible[i] == 1 and visible[i - 1] == 1:
            start = centroids[i - 1]
            end = centroids[i]
            distance = sqrt((end[1] - start[1]) ** 2 + (end[0] - start[0]) ** 2)
            total_distance += distance
            print(f"Distance from {start} to {end}: {distance}")


    print(f"Total distance: {total_distance}")

    # Calculate total number of frames minus the window size
    frame_window_rate = valid_frame_count - win_size

    if frame_window_rate > 0:
        # Calculate the VAP for microns per second
        velocity = total_distance * (fps/frame_window_rate) * pix_size
    else:
        velocity = 0

    print(f"Velocity: {velocity}")
    return velocity


def curvilinearVelocity(centroids, visible, pix_size, fps=5):
    """
       Calculate the average path velocity (VAP)
       FPS frames per a second
       pixel size is determined by pix_size
       Window size is determined by win_size and is a user input for the units of frame to include in the averaging
       argument for VCl and VAP calculations
       """
    # Initialize total distance and count of valid frames
    total_distance = 0

    # Start at -1 so it is total frames minus 1 in the end
    valid_frame_count = -1

    # Iterate over visible frames to calculate distances
    for i in range(1, len(visible)):
        if visible[i] == 1 and visible[i - 1] == 1:
            start = centroids[i - 1]
            end = centroids[i]
            distance = sqrt((end[1] - start[1]) ** 2 + (end[0] - start[0]) ** 2)
            total_distance += distance
            valid_frame_count += 1
            print(f"Distance from {start} to {end}: {distance}")

    print(f"Total distance: {total_distance}")


    if valid_frame_count > 0:
        # Calculate the VCl for microns per second
        velocity = (total_distance / valid_frame_count) * fps * pix_size
    else:
        velocity = 0

    print(f"Velocity: {velocity}")
    return velocity



parser = argparse.ArgumentParser(description='Compute statistics about sperm cells')
parser.add_argument('pklfile', type=str, help='Path to the tracker pkl file')

pklfile = parser.parse_args().pklfile

# Load the json file
with open(pklfile, 'rb') as f:
    data = pickle.load(f)

stats = {}

# Calculate each statstic for each sperm
for i in range(len(data)):
    # Get the centroids
    centroids = data[i]['centroid']
    visible = data[i]['visible']

    average_speed = averagePathVelocity(centroids, visible, 0.26, 2)

    stats[i] = {}
    stats[i]["VAP"] = average_speed

# Find the maximum average speed
#max_speed = max(stats[i]["average_speed"] for i in stats)

# Add the maximum average speed to the stats dictionary
#stats["max_average_speed"] = max_speed

# Save the stats to a json file
outputfile = pklfile.split('.')[0][:-8] + '_stats.pkl'

with open(outputfile, 'wb') as f:
    pickle.dump(stats, f)

print(f'Saved statistics to {outputfile}')