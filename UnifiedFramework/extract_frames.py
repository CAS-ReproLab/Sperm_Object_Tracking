# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 11:48:17 2024

@author: cameron schmidt
"""

import cv2
import os
import sys

def save_first_25_frames(source_path):
    # Capture video from file
    cap = cv2.VideoCapture(source_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get the base name of the file to create a new file name
    base_name = os.path.basename(source_path)
    new_file_name = os.path.splitext(base_name)[0] + '_f25.mp4'
    new_file_path = os.path.join(os.path.dirname(source_path), new_file_name)

    # Get frame rate of the source video to use in the output
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    # Get the width and height of frames
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'h264')
    out = cv2.VideoWriter(new_file_path, fourcc, frame_rate, (frame_width, frame_height))

    # Read and save the first 25 frames
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret and count < 25:
            out.write(frame)
            count += 1
        else:
            break

    # Release everything when job is finished
    cap.release()
    out.release()
    print(f"New video file saved: {new_file_path}")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python script_name.py path/to/your/video.mp4")
        sys.exit(1)
    
    source_video_path = sys.argv[1]
    save_first_25_frames(source_video_path)