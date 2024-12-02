
import json
import argparse
import utils
from tracker import determineCentroids, trackCentroids, segmentCells
from visualizer import runVisualization

### Main Code ###
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Track cells in a video')
    parser.add_argument('videofile', type=str, help='Path to the video file')
    parser.add_argument('configfile', type=str, help='Path to config file')
    parser.add_argument('--outputfile', type=str, default = None, help='Path to the output csv file')
    parser.add_argument('--outputvideofile', type=str, default = None, help='Path to the output video file')

    videofile = parser.parse_args().videofile
    configfile = parser.parse_args().configfile
    outputfile = parser.parse_args().outputfile
    outputvideofile = parser.parse_args().outputvideofile

    frames = utils.loadVideo(videofile,as_gray=True)

    with open(configfile, 'r') as f:
        config = json.load(f)

    # Determine the centroids info
    f = determineCentroids(frames,int(config["diameter"]),int(config["minmass"]))

    # Track the centroids
    t = trackCentroids(f,int(config["search_range"]),int(config["memory"]))

    # Segment the cells
    if config["compute_segs"]=="true":
        t = segmentCells(frames, t)

    # Save the dataframe
    if outputfile is None:
        outputfile = ".".join(videofile.split('.')[:-1]) + '.csv'
    else:
        outputfile = config["outputfile"]

    utils.saveDataFrame(t, outputfile)

    if outputvideofile is None:
        outputvideofile = ".".join(videofile.split('.')[:-1]) + '_output.mp4'

    # Visualize the tracking
    runVisualization(videofile,t,config["visualization"],outputvideofile)