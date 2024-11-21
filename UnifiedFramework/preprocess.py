# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 08:56:31 2024

@author: SCHMIDTC18
"""

'''preprocesses video files for tracker.py'''

import numpy as np
import cv2 as cv
import os
import matplolib.pyplot as plt

videofile = "Videos/5X Ph 9Fps Wash 1 16 120S P019as R1.mp4"

print(os.path.exists(videofile))  # Should print True if the file exists; may indicate issue with file path or import if false

def process_and_save_video(source_video):
    # Initialize the video capture
    cap = cv.VideoCapture(source_video)

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get the width and height of frames
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    print(f'obtained frame width is {frame_width}')
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    print(f'obtained frame height is {frame_height}')
    frame_rate = int(cap.get(cv.CAP_PROP_FPS))
    print(f' obtained frame rate is {frame_rate}')

    # Define the codec and create VideoWriter object
    fourcc = cv.VideoWriter_fourcc(*'h264') # writes out an h264 encoded .mp4 file
    output_path = os.path.splitext(source_video)[0] + '_adj.mp4' # concatenate the file name
    out = cv.VideoWriter(output_path, fourcc, frame_rate, (frame_width, frame_height), isColor=False)

    '''Preprocessing Routine '''
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        # Calculate median grayscale value
        mid_val = np.median(gray)
        
        # Subtract the median from the grayscale image and ensure it stays in valid range
        adjusted_frame = cv.convertScaleAbs(np.abs(gray - mid_val))

        # Write the modified frame to the new video file
        out.write(adjusted_frame)

    # Release everything when job is finished
    cap.release()
    out.release()
    print(f"New video file saved: {output_path}")

# Path to the source video file
source_video = videofile
process_and_save_video(source_video)