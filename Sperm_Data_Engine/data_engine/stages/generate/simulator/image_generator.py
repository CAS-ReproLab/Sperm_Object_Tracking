# -*- coding: utf-8 -*-
"""
Created on Thursday Nov 21 2024

@author: David Hart
"""

'''Generates realistics shapes based on agent positions'''

import cv2
import numpy as np
import pandas as pd

def load_positions_from_csv(filename):
    '''access the positions from the saved trajectories.csv, rather than directly from the simulation arrays'''
    
    # Read in trajectories.csv as a dataframe
    df = pd.read_csv(filename) 
    times = df['Time'].unique()
    agent_ids = df['Agent_ID'].unique()
    num_steps = len(times)
    num_agents = len(agent_ids)
    
    # for faster indexing of large dataframes, create dictionaries for time and agent_id indices
    time_to_index = {time: idx for idx, time in enumerate(sorted(times))}
    agent_id_to_index = {agent_id: idx for idx, agent_id in enumerate(sorted(agent_ids))}
    
    # Initialize a 2D array of NaNs to store the positions 
    positions = np.full((num_steps, num_agents, 2), np.nan)
    
    # Fill the positions array with coords from the dataframe and indices from the dictionaries
    # Set up this way b/c should be faster/more efficient than handling the dataframe directly
    for index, row in df.iterrows():
        t_idx = time_to_index[row['Time']]
        agent_idx = agent_id_to_index[row['Agent_ID']]
        positions[t_idx, agent_idx, 0] = row['x_coord']
        positions[t_idx, agent_idx, 1] = row['y_coord']
    return positions, num_steps, num_agents

class Generator:
    def __init__(self, positions, agent_size, width, height, num_steps, frame_rate= 9.0):
        self.positions = positions
        self.agent_size = agent_size
        self.width = int(width)
        self.height = int(height)
        self.num_steps = num_steps
        self.frame_rate= frame_rate
        self.images = self._compute_images()

    def _compute_images(self):
        # Initialize variables
        images = []
        num_agents = self.positions.shape[1]
        intensity = np.random.randint(200, 255, num_agents)
        sizes = np.random.randint(100, 255, num_agents)
        
        
        for n in range(self.num_steps):
            
            # Create a blank 2D image
            image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            x_coords = self.positions[n, :, 0]
            y_coords = self.positions[n, :, 1]
            
            # Filter for valid (non-NaN) indices
            valid_indices = ~np.isnan(x_coords) & ~np.isnan(y_coords)
            x_coords = x_coords[valid_indices]
            y_coords = y_coords[valid_indices]
            intensities = intensity[valid_indices]
            
            # Calculate agent orientation angles from current and previous positions
            if n > 0:
                prev_x_coords_all = self.positions[n - 1, :, 0]
                prev_y_coords_all = self.positions[n - 1, :, 1]
                prev_x_coords = prev_x_coords_all[valid_indices]
                prev_y_coords = prev_y_coords_all[valid_indices]
                angles = np.rad2deg(np.arctan2(y_coords - prev_y_coords, x_coords - prev_x_coords))
            else:
                angles = np.zeros_like(x_coords)
            
            # Draw the agents on the image    
            for x, y, angle, it in zip(x_coords, y_coords, angles, intensities):
                x_int = int(x)
                y_int = int(y)
                angle_float = float(angle)
                it_int = int(it)
                if 0 <= x_int < self.width and 0 <= y_int < self.height:
                    cv2.ellipse(
                        img=image,
                        center=(x_int, y_int),
                        axes=(
                            int(1.5 * np.sqrt(self.agent_size)),
                            int(np.sqrt(self.agent_size))
                            ),
                        angle=angle_float,
                        startAngle=0,
                        endAngle=360,
                        color=(it_int, it_int, it_int),
                        thickness=-1
                        )
            image = cv2.GaussianBlur(image, (5, 5), 0) # Apply a gaussian blur to the image for realistic edge effects
            images.append(image)
        return images


    def get_image(self,index):
        '''retrieve images at specified index'''
        if 0 <= index < self.num_steps:
            return self.images[index]
        else: 
            raise IndexError('Get_image(): Index is out of range')

    def get_images(self):
        '''retrieve all generated images'''
        return self.images
    
    def save_video(self, filename='generator.mp4'):
        '''Save the images as a video'''
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, self.frame_rate, (self.width, self.height)) # (filename, fourcc, fps, frameSize)

        for image in self.images:
            out.write(image)

        out.release()
            
    
            
            