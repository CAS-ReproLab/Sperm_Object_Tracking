import numpy as np
import cv2 as cv
import pickle
import argparse
from math import sqrt
import pandas as pd
import utils

def interpolate_missing_frames(data, fps=9, pixel_size=0.26):
    """Interpolate missing frames for each sperm.

       Parameters:
       data (pd.DataFrame): DataFrame containing sperm tracking data with columns 'sperm', 'frame', 'x', and 'y'.

       Returns:
       pd.DataFrame: DataFrame with interpolated frames.
       """
    # Ensure all sperm IDs are integers
    data['sperm'] = data['sperm'].astype(int)

    sperm_ids = data['sperm'].unique()
    interpolated_data = []

    for sperm_id in sperm_ids:
        sperm_data = data[data['sperm'] == sperm_id].sort_values(by='frame')
        frames = sperm_data['frame'].values
        x_coords = sperm_data['x'].values
        y_coords = sperm_data['y'].values

        for i in range(1, len(frames)):
            frame_diff = frames[i] - frames[i - 1]
            if frame_diff > 1:
                x_start, x_end = x_coords[i - 1], x_coords[i]
                y_start, y_end = y_coords[i - 1], y_coords[i]
                if frame_diff > 15:
                    print("Warning: The difference between frames is more than 15")
                    continue
                for f in range(1, frame_diff):
                    new_frame = frames[i - 1] + f
                    new_x = x_start + f * (x_end - x_start) / frame_diff
                    new_y = y_start + f * (y_end - y_start) / frame_diff

                    new_row = sperm_data.iloc[i - 1].copy()
                    new_row['frame'] = new_frame
                    new_row['x'] = new_x
                    new_row['y'] = new_y

                    interpolated_data.append(new_row)

    # Combine original and interpolated data
    if interpolated_data:  # Check if interpolated_data is not empty
        interpolated_df = pd.DataFrame(interpolated_data, columns=data.columns)
        combined_df = pd.concat([data, interpolated_df], ignore_index=True)
    else:
        combined_df = data.copy()

    combined_df = combined_df.sort_values(by=['sperm', 'frame']).reset_index(drop=True)

    return combined_df


def calcAverageSpeed(data, fps=9):
    """
    Calculate the average speed of a sperm cell
    """
    if "average_speed" in data.columns:
        print("Warning: The average_speed column already exists in the dataframe. Overwriting it.")

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



def averagePathVelocity(data, fps=9, pixel_size=0.26, win_size=5):
    """
    Compute the average path velocity (VAP) of each sperm cell.
    data : pd.DataFrame
    pixel_size : float, optional
        Size of one pixel in micrometers, by default 0.26.
    win_size : int, optional
        Window size for computing the average path, by default 5.

    Returns
    pd.DataFrame.
    """

    if "VAP" in data.columns:
        print("Warning: The VAP column already exists. Overwriting it.")

    # Process each sperm track separately
    sperm_ids = data['sperm'].unique()
    
    for sperm_id in sperm_ids:
        # Extract rows for just this sperm, sorted by frame
        sperm_df = data[data['sperm'] == sperm_id].sort_values(by='frame')
        x_vals = sperm_df['x'].to_numpy()
        y_vals = sperm_df['y'].to_numpy()
        n_points = len(sperm_df)
        
        # If there aren't enough points to form a single complete window, VAP = 0
        if n_points < win_size:
            data.loc[data['sperm'] == sperm_id, 'VAP'] = 0
            continue

        # Collect the average coordinates for each valid window
        avg_coords = []
        for i in range(n_points - win_size + 1):
            window_x = x_vals[i : i + win_size]
            window_y = y_vals[i : i + win_size]
            mean_x = window_x.mean()
            mean_y = window_y.mean()
            avg_coords.append((mean_x, mean_y))
        
        # Convert to NumPy array for vectorized distance calculation
        avg_coords = np.array(avg_coords)

        # Calculate the total distance along the "average path"
        if len(avg_coords) > 1:
            # Distance between consecutive points in avg_coords
            dist = np.sqrt(
                np.diff(avg_coords[:, 0])**2 + 
                np.diff(avg_coords[:, 1])**2
            )
            total_avg_path = dist.sum()
        else:
            total_avg_path = 0

        # Frames used is #windows (len(avg_coords)). 
        # Another approach is to consider total frames as well, 
        # but typically we scale by how many average coords we used:
        frames_used = len(avg_coords)
        
        # Convert distance to velocity
        # total_avg_path is in "pixels" → multiply by pixel_size to get micrometers (μm)
        # (fps / frames_used) scales distance to velocity
        vap = total_avg_path * (fps / frames_used) * pixel_size

        # Update DataFrame
        data.loc[data['sperm'] == sperm_id, 'VAP'] = vap

    return data




def curvilinearVelocity(data, fps=9, pixel_size = 0.26):
    '''Calculate the average path velocity (VAP) over all frames

    Parameters:
    data (pd.DataFrame): DataFrame containing sperm tracking data with columns 'sperm', 'frame', 'x', and 'y'.
    fps (int): Frames per second, default is 30.
    pixel_size (float): Size of one pixel in micrometers (or any other unit), default is 0.26.
    win_size (int): Size of the window in micrometers (or any other unit), default is 1 aka calculates
    curvilinear velocity (VCL). Ajust window_size accordingly for VAP.
    Returns the pd.DataFrame: DataFrame with an additional column 'VAP' containing the average speed of each sperm cell.
    '''

    if "VCL" in data.columns:
        print("Warning: The curvilinear velocity column already exists in the dataframe. Overwriting it.")

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
            distance_pixels = sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)

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

def straightLineVelocity(data, fps=9, pixel_size=0.26):
    '''Calculate the straight-line velocity (VSL) over all frames

     Parameters:
     data (pd.DataFrame): DataFrame containing sperm tracking data with columns 'sperm', 'frame', 'x', and 'y'.
     fps (int): Frames per second, default is 30.
     pixel_size (float): Size of one pixel in micrometers (or any other unit), default is 0.26.

     Returns:
     pd.DataFrame: DataFrame with an additional column 'VSL' containing the straight-line velocity of each sperm cell.
     '''
    if "VSL" in data.columns:
        print("Warning: The stright line velocity column already exists in the dataframe. Overwriting it.")

    # Get unique sperm IDs
    sperm_ids = data['sperm'].unique()

    # Iterate over each sperm ID
    for sperm_id in sperm_ids:
        # Filter the dataframe for the current sperm
        sperm = data[data['sperm'] == sperm_id]

        # Sort the dataframe by frame
        sperm = sperm.sort_values(by='frame')

        distance_iteration = len(sperm) - 1
        # Check if there are at least two points to calculate velocity
        if len(sperm) > 1:
            # Get the first and last points
            start = (sperm['x'].iloc[0], sperm['y'].iloc[0])
            end = (sperm['x'].iloc[-1], sperm['y'].iloc[-1])

            # Calculate the straight-line distance in pixels
            distance_pixels = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)

            time_step = fps/distance_iteration

         # Calculate the straight-line velocity with sufficient distance
            if time_step > 0:
                average_velocity = distance_pixels * (fps/ distance_iteration) * pixel_size
            else:
                average_velocity = 0

            # Add the average velocity to the dataframe
            data.loc[data['sperm'] == sperm_id, 'VSL'] = average_velocity

        else:
            # If there's only one point, velocity cannot be calculated
            data.loc[data['sperm'] == sperm_id, 'VSL'] = np.nan
    return data


def add_motility_column(data, vcl_column='VCL', threshold=25):

    """
    Adds a column 'motility' to the DataFrame that categorizes sperm as
    'motile' if VCL >= threshold and 'immotile' if VCL < threshold.
    """
    data['Motility'] = data['VCL'].apply(lambda vcl: 'motile' if vcl >= threshold else 'immotile')
    return data



def alh(data, fps=9, pixel_size=0.26, win_size=5):
    '''Calculate the amplitude of lateral head displacement (ALH) for each sperm cell
     It returns ALH_mean and ALH_max for each sperm cell.
     '''

    if "ALH_mean" in data.columns or "ALH_max" in data.columns:
        print("Warning: The amplitude lateral head displacment columns already exists in the dataframe. Overwriting it.")

    if "New_VAP" in data.columns:
        print("Warning: The NEW average path velocity column already exists in the dataframe. Overwriting it.")

    # Get unique sperm IDs
    sperm_ids = data['sperm'].unique()

    # Create empty numpy array
    #data['ALH_mean'] = np.nan
    #data['ALH_max'] = np.nan

    for sperm_id in sperm_ids:
        # Filter the dataframe for the current sperm and sort by frame
        sperm = data[data['sperm'] == sperm_id].sort_values(by='frame')
        distance = []
        avg_path_coords = []
        # Loop through sperm coordinates
        for i in range(1, len(sperm) - win_size):

            # Initialize arrays for window coordinates and average path
            window_coords = []
            arr1 = []

            # Add first point to average path
            first_point = (sperm['x'].iloc[0], sperm['y'].iloc[0])
            arr1.append(first_point)

            # Loop within sliding window to get average path
            for j in range(win_size):
                x, y = sperm['x'].iloc[i + j], sperm['y'].iloc[i + j]
                window_coords.append([x, y])

            # Convert window_coords to a NumPy array
            window_coords = np.array(window_coords)

            # Create average path coordinates
            aver_x = np.sum(window_coords[:, 0]) / win_size
            aver_y = np.sum(window_coords[:, 1]) / win_size
            arr1.append((aver_x, aver_y))
            avg_path_coords.append((aver_x, aver_y))

            # Add last point to average path
            last_point = (sperm['x'].iloc[-1], sperm['y'].iloc[-1])
            arr1.append(last_point)

            # Convert arr1 to a NumPy array
            arr1 = np.array(arr1)

            # Get midpoints of average path
            # Calculate midpoints of the average path
            mid_x = (arr1[0, 0] + arr1[2, 0]) / 2
            mid_y = (arr1[0, 1] + arr1[2, 1]) / 2

            # Get the first and last coordinates within the window
            first = window_coords[0]
            last = window_coords[-1]

            # Perpendicular distance formula for first point
            numerator = abs(
                (last[0] - first[0]) * (first[1] - mid_y) - (first[0] - mid_x) * (last[1] - first[1]))
            denominator = np.sqrt((last[0] - first[0]) ** 2 + (last[1] - first[1]) ** 2)

            # Check for a zero or near-zero denominator
            if denominator == 0:
                dist_first = 0  # or use np.nan to indicate invalid distance
            else:
                dist_first = numerator / denominator

            # Store distance for first point
            distance.append(dist_first)

       # Turn distance to numpy array
        distance = np.array(distance)
        # Lists to store distances between consecutive segments
        displacements = []
        # Loop through and fine the most displaced distance
        count = 0
        for l in range(1, len(distance)-1):
            if distance[l] > distance[l-1] and distance[l] > distance[l+1]:
                count += 1
                displacements.append(distance[l])



        # Calculate mean and max ALH, converting to micrometers
        if displacements:

            alh_mean = (np.sum(displacements) / count)  * pixel_size * 2
            alh_max = np.max(displacements) * pixel_size * 2
        else:
            alh_mean = np.nan
            alh_max = np.nan
        # Assign ALH values back to the dataframe
        data.loc[data['sperm'] == sperm_id, 'ALH_mean'] = alh_mean
        data.loc[data['sperm'] == sperm_id, 'ALH_max'] = alh_max


    return data


def intersect_segments(p1, p2, q1, q2):
    '''
    # This returns boolean if segment 1 (p1 and 2) are opposite sides of mid_segment (q1 ans q2)
    # and if mid_segment (q1 ans q2) are opposite sides of segment 1 (p1 and 2)
    '''
    return ( ((q1[1] - p1[1]) * (p2[0] - p1[0]) > (p2[1] - p1[1]) * (q1[0] - p1[0])) !=
        ((q2[1] - p1[1]) * (p2[0] - p1[0]) > (p2[1] - p1[1]) * (q2[0] - p1[0])) and
        ((q1[1] - p1[1]) * (q2[0] - q1[0]) > (q2[1] - q1[1]) * (q1[0] - p1[0])) !=
        ((q1[1] - p2[1]) * (q2[0] - q1[0]) > (q2[1] - q1[1]) * (q1[0] - p2[0]))
             )

def bcf (data, fps=9, pixel_size=0.26, win_size=5):
    '''Calculate the Beat-Cross Frequency (BCF) for each sperm cell.
       Returns the updated dataframe with BCF values for each sperm cell.
    '''

    sperm_ids = data['sperm'].unique()

    data['BCF'] = np.nan

    for sperm_id in sperm_ids:
        sperm = data[data['sperm'] == sperm_id].sort_values(by='frame')
        avg_path_coords = []
        cross_count = 0


        for i in range(len(sperm) - win_size):
            window_coords = sperm[['x', 'y']].iloc[i:i + win_size].to_numpy()
            aver_x = np.mean(window_coords[:, 0])
            aver_y = np.mean(window_coords[:, 1])
            avg_path_coords.append((aver_x, aver_y))

        avg_path_coords = np.array(avg_path_coords)

        # Check intersections between actual and average path segments
        for i in range(len(avg_path_coords) - 1):
            #if i >= len(sperm) - 1:
                #break

            # Actual path segment with the start of segment and end of segment
            actual_start = (sperm['x'].iloc[i], sperm['y'].iloc[i])
            actual_end = (sperm['x'].iloc[i + 1], sperm['y'].iloc[i + 1])

            # Average path segment
            avg_start = avg_path_coords[i]
            avg_end = avg_path_coords[i + 1]

            # Check segments intersect
            if intersect_segments(actual_start, actual_end, avg_start, avg_end):
                cross_count += 1

        # Calculate BCF
        total_time = (len(sperm) - win_size)
        if total_time > 0:
            bcf = cross_count * (fps /total_time)
        else:
            bcf = np.nan

        # Store results in the dataframe
        data.loc[data['sperm'] == sperm_id, 'BCF'] = bcf

    return data



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Compute statistics about sperm cells')
    parser.add_argument('csvfile', type=str, help='Path to the tracker csv file')
    parser.add_argument('--output', type=str, default=None, help='Path to the output file')


    csvfile = parser.parse_args().csvfile
    outputfile = parser.parse_args().output

    if outputfile is None:
        outputfile = csvfile

    data = pd.read_csv(csvfile)

    # Interpolate missing frames
    data = interpolate_missing_frames(data)

    '''Pixel Sizes
    5X 0.5512 pixels per micron
    10X 1.0476 pixels per micron
    20X 2.0619 pixels per micron
    '''

    # Run calcAverageSpeed
    vap = averagePathVelocity(data, fps= 9, pixel_size= 1.0476, win_size= 5)
    vcl = curvilinearVelocity(data, fps= 9, pixel_size= 1.0476)
    vsl = straightLineVelocity(data, fps=9, pixel_size= 1.0476)
    alh = alh(data, fps=9, pixel_size= 1.0476,win_size= 5)
    bcf = bcf(data, fps=9, pixel_size=1.0476, win_size=5)

    # Add motility status column based on VCL values
    motility = add_motility_column(data, vcl_column='VCL')

    # Save the new data file with the statistics
    utils.saveDataFrame(vap, outputfile)
    utils.saveDataFrame(vcl, outputfile)
    utils.saveDataFrame(motility, outputfile)
    utils.saveDataFrame(vsl, outputfile)
    utils.saveDataFrame(alh, outputfile)
    utils.saveDataFrame(bcf, outputfile)

    print("Statistics computed and saved to", outputfile)

