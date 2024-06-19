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

def averagePathVelocity(data, fps=30, pixel_size = 0.26, win_size= 1):
    '''Calculate the average path velocity (VAP) over all frames

    Parameters:
    data (pd.DataFrame): DataFrame containing sperm tracking data with columns 'sperm', 'frame', 'x', and 'y'.
    fps (int): Frames per second, default is 30.
    pixel_size (float): Size of one pixel in micrometers (or any other unit), default is 0.26.
    win_size (int): Size of the window in micrometers (or any other unit), default is 1 aka calculates
    curvilinear velocity (VCL). Ajust window_size accordingly for VAP.
    Returns the pd.DataFrame: DataFrame with an additional column 'VAP' containing the average speed of each sperm cell.
    '''
    # Add a column to the dataframe to store the average speed
    data['VAP'] = 0.0

    # Determine the number of sperm
    sperm_count = data['sperm'].nunique()


    # Iterate over each sperm
    for i in range(sperm_count):
        # Filter the dataframe for the current sperm
        sperm = data[data['sperm'] == i]

        # Sort the dataframe by frame
        sperm = sperm.sort_values(by='frame')

        # Set distance iteration based on window size
        distance_iteration = (len(sperm)) - win_size

        # Calculate the total distance traveled by the sperm
        total_distance = 0
        for j in range(1, distance_iteration):
            start = (sperm['x'].iloc[j - 1], sperm['y'].iloc[j - 1])
            end = (sperm['x'].iloc[j], sperm['y'].iloc[j])
            # Calculate the distance in pixels
            distance_pixels = sqrt((end[1] - start[1]) ** 2 + (end[0] - start[0]) ** 2)

            # Add total distance
            total_distance += distance_pixels

        # Calculate the number of frames
        win_frames = len(sperm) - win_size
        # Calculate the average speed in real-world units per second
        if win_frames > 0:
            average_speed = total_distance * (fps/win_frames) * pixel_size
        else:
            average_speed = 0

        # Add the average speed to the dataframe
        data.loc[data['sperm'] == i, 'VAP'] = average_speed

    return data




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Compute statistics about sperm cells')
    parser.add_argument('csvfile', type=str, help='Path to the tracker csv file')

    csvfile = parser.parse_args().csvfile

    data = utils.loadDataFrame(csvfile)

    # Run calcAverageSpeed
    average_speed = averagePathVelocity(data, fps= 30, pixel_size= 0.26, win_size= 5)

    # Save the new data file with the statistics
    outputfile = csvfile.split('.')[0] + '_withstats.csv'

    utils.saveDataFrame(average_speed, outputfile)

    print("Statistics computed and saved to", outputfile)

