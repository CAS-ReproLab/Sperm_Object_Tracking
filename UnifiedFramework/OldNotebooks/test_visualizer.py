# Kylie made this
# April 3, 2025
# test_visualization.py
import pytest
import numpy as np
import pandas as pd
import cv2 as cv

# Import functions from your code file (adjust if your file is named differently)
from visualizer import (
    opticalFlow,
    boundingBoxes,
    coloring,
    colorSpeed,
    flowSpeed,
)

@pytest.fixture
def sample_frame():
    """
    Create a small, black RGB image for testing, e.g. 100x100.
    """
    return np.zeros((100, 100, 3), dtype=np.uint8)

@pytest.fixture
def sample_data():
    """
    Create a small DataFrame that mimics the columns your code expects.
    We'll include columns: 
       'frame', 'sperm', 'x', 'y', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h', 'segmentation', 'VAP'
    """
    # Two frames, two sperm. 
    return pd.DataFrame({
        'frame':    [0, 0, 1, 1],
        'sperm':    [0, 1, 0, 1],
        'x':        [10, 20, 12, 22],
        'y':        [10, 20, 12, 22],
        'bbox_x':   [8, 18, 8, 18],
        'bbox_y':   [8, 18, 8, 18],
        'bbox_w':   [5, 5, 5, 5],
        'bbox_h':   [5, 5, 5, 5],
        'segmentation': [
            [[10,10],[10,11]],   # Some small sets of coords
            [[20,20],[21,20]],   # ...
            [[12,12],[12,13]], 
            [[22,22],[22,23]]
        ],
        'VAP':      [5.0, 2.0, 10.0, 30.0]
    })

@pytest.fixture
def mask(sample_frame):
    """
    Create a mask the same size as `sample_frame`.
    """
    return np.zeros_like(sample_frame)

@pytest.fixture
def color_array(sample_data):
    """
    Create an array of random colors, indexed up to the max sperm ID in sample_data.
    """
    max_sperm_id = sample_data['sperm'].max()
    colors = np.random.randint(0, 255, (max_sperm_id + 1, 3))
    return colors

def test_opticalFlow(sample_frame, sample_data, mask, color_array):
    """
    Test that opticalFlow runs without error, and returns an image of the same shape.
    Test frame_num=1 (since our sample_data has frames 0 and 1).
    """
    frame_num = 1
    result_img = opticalFlow(sample_frame, sample_data, frame_num, mask, color_array)

    assert result_img.shape == sample_frame.shape, \
        "Resulting image from opticalFlow should have the same shape as the input frame."
    assert result_img.dtype == np.uint8, \
        "Resulting image should be a uint8 image."

def test_boundingBoxes(sample_frame, sample_data):
    """
    Test boundingBoxes returns an image of the same shape.
    Pick frame_num=0 here, but any valid frame is fine.
    """
    frame_num = 0
    result_img = boundingBoxes(sample_frame, sample_data, frame_num)
    assert result_img.shape == sample_frame.shape
    assert result_img.dtype == np.uint8

def test_coloring(sample_frame, sample_data):
    """
    Test that coloring draws on the frame correctly and 
    returns an image. Also verifies it raises an error if 'segmentation' is missing.
    """
    # Normal usage
    frame_num = 0
    colors = np.random.randint(0, 255, (2, 3))  # For 2 sperm IDs, if you want
    result_img = coloring(sample_frame, sample_data, frame_num, colors)
    assert result_img.shape == sample_frame.shape
    assert result_img.dtype == np.uint8

def test_coloring_no_seg_col(sample_frame, sample_data):
    """
    If the 'segmentation' column is missing, coloring should raise a ValueError.
    """
    df_no_seg = sample_data.drop(columns=['segmentation'])
    with pytest.raises(ValueError):
        coloring(sample_frame, df_no_seg, frame_num=0, colors=np.array([[0,0,0]]))

def test_colorSpeed(sample_frame, sample_data):
    """
    Test colorSpeed with static_threshold, lower_threshold, and upper_threshold.
    Ensure it returns a valid image.
    """
    frame_num = 1
    # Example thresholds
    static_threshold = 2   # <=2 => "static"
    lower_threshold  = 5   # <=5 => "slow"
    upper_threshold  = 25  # <=25 => "medium", else "fast"

    result_img = colorSpeed(
        sample_frame,
        sample_data,
        frame_num,
        static_threshold,
        lower_threshold,
        upper_threshold
    )
    assert result_img.shape == sample_frame.shape
    assert result_img.dtype == np.uint8

def test_flowSpeed(sample_frame, sample_data, mask):
    """
    Test flowSpeed with some thresholds. 
    Ensure it returns a valid image and does not crash.
    """
    frame_num = 1
    static_threshold = 2
    lower_threshold  = 5
    upper_threshold  = 25

    result_img = flowSpeed(
        sample_frame,
        sample_data,
        frame_num,
        mask,
        static_threshold,
        lower_threshold,
        upper_threshold
    )
    assert result_img.shape == sample_frame.shape
    assert result_img.dtype == np.uint8
