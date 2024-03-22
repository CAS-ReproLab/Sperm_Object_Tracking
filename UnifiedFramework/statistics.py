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

    average_speed = calcAverageSpeed(centroids,visible)

    stats[i] = {}
    stats[i]["average_speed"] = average_speed

# Save the stats to a json file
outputfile = pklfile.split('.')[0][:-8] + '_stats.pkl'

with open(outputfile, 'wb') as f:
    pickle.dump(stats, f)

print(f'Saved statistics to {outputfile}')