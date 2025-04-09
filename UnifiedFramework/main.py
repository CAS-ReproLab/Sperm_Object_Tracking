
import json
import argparse
import stats
import utils
from tracker import determineCentroids, trackCentroids, segmentCells
from visualizer import runVisualization

import tkinter as tk
from tkinter import filedialog

default_config = {
    "diameter": 11,
    "minmass": 500,
    "search_range": 21,
    "memory": 3,
    "fps": 9,
    "pixel_size": 1.0476,
    "compute_segs": "false",
    "visualization": "flow"
}

### Main Code ###
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Track cells in a video')
    parser.add_argument('--videofile', type=str, help='Path to the video file')
    parser.add_argument('--configfile', type=str, help='Path to config file')
    parser.add_argument('--outputfile', type=str, default = None, help='Path to the output csv file')
    parser.add_argument('--outputvideofile', type=str, default = None, help='Path to the output video file')

    videofile = parser.parse_args().videofile
    configfile = parser.parse_args().configfile
    outputfile = parser.parse_args().outputfile
    outputvideofile = parser.parse_args().outputvideofile

    if videofile is None:
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        videofile = filedialog.askopenfilename(title="Select the video file")

        if videofile:
            print("Selected file:", videofile)
        else:
            raise ValueError("No video file selected.")
        
    if configfile is None:
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        configfile = filedialog.askopenfilename(title="Select the configuration file")

        if configfile:
            print("Selected config file:", configfile)
        else:
            configfile = default_config
            print("No config file selected. Using default config.")

    frames = utils.loadVideo(videofile,as_gray=True)

    with open(configfile, 'r') as f:
        config = json.load(f)

    # Apply median filter to the frames
    if config["median_filter"] == "true":
        frames = utils.medianFilter(frames)

    # Determine the centroids info
    f = determineCentroids(frames,int(config["diameter"]),int(config["minmass"]))

    # Track the centroids
    t = trackCentroids(f,int(config["search_range"]),int(config["memory"]))

    # Segment the cells
    if config["compute_segs"]=="true":
        t = segmentCells(frames, t)

    # Compute stats
    t = stats.computeAllStats(t,config["fps"],config["pixel_size"])

    # Save the dataframe
    if outputfile is None:
        outputfile = ".".join(videofile.split('.')[:-1]) + '.csv'
    else:
        outputfile = config["outputfile"]

    utils.saveDataFrame(t, outputfile)

    # Show stats plots
    stats.plotAllStats(t)

    if outputvideofile is None:
        outputvideofile = ".".join(videofile.split('.')[:-1]) + '_output.mp4'

    # Visualize the tracking
    runVisualization(videofile,t,config["visualization"],outputvideofile)