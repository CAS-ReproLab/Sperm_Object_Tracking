# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 08:56:31 2024

@author: SCHMIDTC18
"""

'''preprocesses video files for tracker.py'''

import numpy as np
import cv2 as cv
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
                    
def initialize_video_capture(video_path):
    '''Initialize video capture object from opencv'''
    if not os.path.exists(video_path):
        logging.error(f'File not found in: {video_path}')
        return None
    cap= cv.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f'Could not open video: {video_path}')
        return None
    return cap

def get_video_properties(cap):
    '''get the video properties from the metadata uisng opencv'''
    properties= {
        'frame_width': int(cap.get(cv.CAP_PROP_FRAME_WIDTH)),
        'frame_height': int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)),
        'frame_rate': int(cap.get(cv.CAP_PROP_FPS))
        }
    return properties

def preprocess_frame(frame, preprocessing_function=None):
    '''proprocess a single frame using the designated function, defined below'''
    gray= cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    if preprocessing_function: 
        return preprocessing_function(gray)
    return gray

def process_and_save_video(source_video, output_path, preprocessing_function=None):
    '''process and save the video as mp4v encoded .mp4; Note: could update to take output encoding/file as argument in future'''
    cap=initialize_video_capture(source_video)
    if cap is None:
        return
    
    properties= get_video_properties(cap)
    logging.info(f'Video properties obtained from OpenCV: {properties}')
    
    fourcc= cv.VideoWriter_fourcc(*'mp4v') # originally tried h264, but wasn't working
    out= cv.VideoWriter(output_path, fourcc, properties['frame_rate'], (properties['frame_width'], properties['frame_height']), isColor=False)
    
    while True: 
        ret, frame= cap.read()
        if not ret: 
            break
        processed_frame = preprocess_frame(frame, preprocessing_function)
        out.write(processed_frame)  
        
    cap.release()
    out.release()
    logging.info(f'Processed video saved to: {output_path}')
    
# Preprocessing functions. Add more as necessary. 
    
def median_filter(frame): 
    '''subtract median grayscale value'''
    mid_val= np.median(frame)
    return cv.convertScaleAbs(np.abs(frame- mid_val))

def main(): 
    input_video= 'Videos/5X Ph 9Fps Wash 1 16 120S P019as R2.mp4'
    output_video= os.path.splitext(input_video)[0] + '_median.mp4' #insert the preprocessing function name
    process_and_save_video(input_video, output_video, preprocessing_function=median_filter)
    
if __name__ == "__main__":
    main()
    