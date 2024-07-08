import numpy as np
import cv2 as cv
import pickle
import argparse
from math import sqrt
import pandas as pd


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

def averagePathVelocity(data, fps=30, pixel_size=0.26, win_size=5):
    '''Calculate the average path velocity (VAP) over all frames

    Parameters:
    data (pd.DataFrame): DataFrame containing sperm tracking data with columns 'sperm', 'frame', 'x', and 'y'.
    fps (int): Frames per second, default is 30.
    pixel_size (float): Size of one pixel in micrometers (or any other unit), default is 0.26.
    win_size (int): Size of the window in micrometers (or any other unit), default is 1 aka calculates
    curvilinear velocity (VCL). Adjust window_size accordingly for VAP.
    Returns the pd.DataFrame: DataFrame with an additional column 'VAP' containing the average speed of each sperm cell.
    '''

    # Get unique sperm IDs
    sperm_ids = data['sperm'].unique()

    # Iterate over each sperm ID
    for sperm_id in sperm_ids:
        # Filter the dataframe for the current sperm
        sperm = data[data['sperm'] == sperm_id]

        # Sort the dataframe by frame
        sperm = sperm.sort_values(by='frame')

        # Set distance iteration based on window size
        distance_iteration = len(sperm) - win_size

        # Calculate the total distance traveled by the sperm
        total_distance = 0
        for j in range(1, distance_iteration):
            start = (sperm['x'].iloc[j - 1], sperm['y'].iloc[j - 1])
            end = (sperm['x'].iloc[j], sperm['y'].iloc[j])
            # Calculate the distance in pixels
            distance_pixels = sqrt((end[1] - start[1]) ** 2 + (end[0] - start[0]) ** 2)

            # Add total distance
            total_distance += distance_pixels

        # Calculate the average speed in real-world units per second
        if distance_iteration > 0:
            average_speed = total_distance * (fps / distance_iteration) * pixel_size
        else:
            average_speed = 0

        # Add the average speed to the dataframe
        data.loc[data['sperm'] == sperm_id, 'VAP'] = average_speed

    return data



def curvilinearVelocity(data, fps=30, pixel_size = 0.26):
    '''Calculate the average path velocity (VAP) over all frames

    Parameters:
    data (pd.DataFrame): DataFrame containing sperm tracking data with columns 'sperm', 'frame', 'x', and 'y'.
    fps (int): Frames per second, default is 30.
    pixel_size (float): Size of one pixel in micrometers (or any other unit), default is 0.26.
    win_size (int): Size of the window in micrometers (or any other unit), default is 1 aka calculates
    curvilinear velocity (VCL). Ajust window_size accordingly for VAP.
    Returns the pd.DataFrame: DataFrame with an additional column 'VAP' containing the average speed of each sperm cell.
    '''

    # Get unique sperm IDs
    sperm_ids = data['sperm'].unique()

    # Iterate over each sperm ID
    for sperm_id in sperm_ids:
        # Filter the dataframe for the current sperm
        sperm = data[data['sperm'] == sperm_id]

        # Sort the dataframe by frame
        sperm = sperm.sort_values(by='frame')

        # Get distance iteration
        distance_iteration = len(sperm) - 1

        # Calculate the total distance traveled by the sperm
        total_distance = 0
        for j in range(1, distance_iteration):
            start = (sperm['x'].iloc[j - 1], sperm['y'].iloc[j - 1])
            end = (sperm['x'].iloc[j], sperm['y'].iloc[j])
            # Calculate the distance in pixels
            distance_pixels = sqrt((end[1] - start[1]) ** 2 + (end[0] - start[0]) ** 2)

            # Add total distance
            total_distance += distance_pixels

        # Calculate the average speed in real-world units per second
        if distance_iteration > 0:
            average_speed = total_distance * (fps / distance_iteration) * pixel_size
        else:
            average_speed = 0

        # Add the average speed to the dataframe
        data.loc[data['sperm'] == sperm_id, 'VCL'] = average_speed

    return data


'''def straightLineVelocity(data, fps=30, pixel_size=0.26):
    Calculate the straight line velocity (VSL) for each sperm cell

    Parameters:
    data (pd.DataFrame): DataFrame containing sperm tracking data with columns 'sperm', 'frame', 'x', and 'y'.
    fps (int): Frames per second, default is 30.
    pixel_size (float): Size of one pixel in micrometers (or any other unit), default is 0.26.

    Returns:
    pd.DataFrame: DataFrame with an additional column 'VSL' containing the straight line velocity of each sperm cell.
    
    # Add a column to the dataframe to store the straight line velocity
    data['VSL'] = 0.0

    # Determine the number of sperm
    sperm_count = data['sperm'].nunique()

    # Iterate over each sperm
    for i in range(sperm_count):
        # Filter the dataframe for the current sperm
        sperm = data[data['sperm'] == i]

        # Sort the dataframe by frame
        sperm = sperm.sort_values(by='frame')

        # Get the start and end positions of the sperm
        start = (sperm['x'].iloc[0], sperm['y'].iloc[0])
        end = (sperm['x'].iloc[-1], sperm['y'].iloc[-1])

        # Calculate the straight line distance in pixels
        distance_pixels = sqrt((end[1] - start[1]) ** 2 + (end[0] - start[0]) ** 2)

        # Calculate the number of frames
        frame_count = len(sperm)

        if frame_count > 1:
            # Calculate the VSL in microns per second
            velocity = distance_pixels * (fps / (frame_count - 1)) * pixel_size
        else:
            velocity = 0

        # Add the straight line velocity to the dataframe
        data.loc[data['sperm'] == i, 'VSL'] = velocity

    return data
'''

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Compute statistics about sperm cells')
    parser.add_argument('csvfile', type=str, help='Path to the tracker csv file')

    csvfile = parser.parse_args().csvfile

    data = pd.read_csv(csvfile)

    # Run calcAverageSpeed
    vap = averagePathVelocity(data, fps= 30, pixel_size= 0.26, win_size= 5)
    vcl = curvilinearVelocity(data, fps= 30, pixel_size= 0.26)
    #vcl = straightLineVelocity(data, fps=30, pixel_size=0.26)

    # Save the new data file with the statistics
    outputfile = csvfile.split('.')[0] + '_withstats.csv'

    utils.saveDataFrame(vap, outputfile)
    utils.saveDataFrame(vcl, outputfile)

    print("Statistics computed and saved to", outputfile)

