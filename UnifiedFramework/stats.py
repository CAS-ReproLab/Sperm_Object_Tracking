import numpy as np
import cv2 as cv
#import json
import pickle
import argparse
from math import sqrt

import utils

def calcAverageSpeed(data, fps=30):
    """
    Calculate the average speed of a sperm cell
    """
    # Add a column to the dataframe to store the average speed
    data['average_speed'] = 0.0

    # Determine the number of sperm
    sperm_count = data['sperm'].nunique()

    # Determine the number of frames
    frames = data['frame'].nunique()

    for i in range(sperm_count):
        # Filter the dataframe
        sperm = data[data['sperm'] == i]

        # Sort the dataframe by frame
        sperm = sperm.sort_values(by='frame')

        # Calculate the average speed for each sperm
        count = len(sperm) - 1
        total = 0
        for j in range(1, len(sperm)):
            start = (sperm['x'].iloc[j-1], sperm['y'].iloc[j-1])
            end = (sperm['x'].iloc[j], sperm['y'].iloc[j])
            total += sqrt((end[1]-start[1])**2 + (end[0]-start[0])**2)
        
        # Calculate the average speed
        average_speed = (total/count)/fps

        # Add the average speed to the dataframe
        data.loc[data['sperm'] == i, 'average_speed'] = average_speed

    return data

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Compute statistics about sperm cells')
    parser.add_argument('csvfile', type=str, help='Path to the tracker csv file')

    csvfile = parser.parse_args().csvfile

    data = utils.loadDataFrame(csvfile)

    # Run calcAverageSpeed
    average_speed = calcAverageSpeed(data)

    # Save the new data file with the statistics
    utils.saveDataFrame(average_speed, csvfile.split('.')[0] + '_withstats.csv')

