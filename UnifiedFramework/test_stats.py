# Kylie made this
# April 3, 2025

import pytest
import pandas as pd
import numpy as np

# Import from stats
from stats import (
    interpolate_missing_frames,
    averagePathVelocity,
    curvilinearVelocity,
    straightLineVelocity,
    add_motility_column,
    alh,
    bcf
)

@pytest.fixture
def example_data():
    """
    Creates a small, fake dataset for testing.
    """
    return pd.DataFrame({
        'sperm': [0, 0, 0, 1, 1],
        'frame': [0, 1, 3, 0, 2],  # Notice sperm=0 is missing frame=2
        'x': [0, 1, 3, 10, 12],
        'y': [0, 1, 3, 10, 11]
    })

def test_interpolate_missing_frames(example_data):
    """
    Test that interpolate_missing_frames inserts rows for missing frames.
    """
    # We expect frame 2 to be interpolated for sperm 0
    # Currently, sperm 0 has frames: 0, 1, 3
    
    # Run interpolation
    result = interpolate_missing_frames(example_data.copy(), fps=9, pixel_size=0.26)
    
    # Filter back out just sperm 0
    sperm_0_data = result[result['sperm'] == 0].sort_values(by='frame')
    
    # We expect frames: 0, 1, 2, 3 now
    frames_for_sperm_0 = sperm_0_data['frame'].to_list()
    
    assert len(frames_for_sperm_0) == 4, "Should have 4 frames for sperm 0 after interpolation."
    assert 2 in frames_for_sperm_0, "Frame 2 should have been interpolated."


def test_averagePathVelocity(example_data):
    """
    Test that averagePathVelocity adds a 'VAP' column with numeric values.
    """
    result = averagePathVelocity(example_data.copy(), fps=9, pixel_size=0.26, win_size=2)
    
    # Check if 'VAP' column exists
    assert 'VAP' in result.columns, "'VAP' column should be added."
    
    # Check that VAP is not null for sperm with enough frames
    # Sperm 0 has 3 frames, so with win_size=2 it can compute at least 2 windows
    sperm_0_vap = result.loc[result['sperm'] == 0, 'VAP'].iloc[0]
    assert not pd.isna(sperm_0_vap), "Sperm 0 should have a valid VAP value."


def test_curvilinearVelocity(example_data):
    """
    Test that curvilinearVelocity adds a 'VCL' column and is >= 0.
    """
    result = curvilinearVelocity(example_data.copy(), fps=9, pixel_size=0.26)
    
    assert 'VCL' in result.columns, "'VCL' column should be added by curvilinearVelocity."
    
    # Check values are non-negative
    for val in result['VCL']:
        assert val >= 0, "Curvilinear velocities must be non-negative."


def test_straightLineVelocity(example_data):
    """
    Test that straightLineVelocity adds 'VSL' column.
    """
    result = straightLineVelocity(example_data.copy(), fps=9, pixel_size=0.26)
    assert 'VSL' in result.columns, "'VSL' column should be added by straightLineVelocity."
    
    # Check that for sperm with multiple frames, we get a numeric value
    sperm_0_vsl = result.loc[result['sperm'] == 0, 'VSL'].iloc[0]
    assert not pd.isna(sperm_0_vsl), "Sperm 0 should have a valid VSL."


def test_add_motility_column(example_data):
    """
    Test that add_motility_column adds the 'Motility' column with 'motile'/'immotile'.
    """
    # First compute VCL so the column exists
    data_with_vcl = curvilinearVelocity(example_data.copy(), fps=9, pixel_size=0.26)
    
    # Then add motility based on threshold=25
    result = add_motility_column(data_with_vcl, vcl_column='VCL', threshold=1)  # set threshold low to see 'motile'
    
    assert 'Motility' in result.columns, "'Motility' column should be added."
    # Because our VCL is likely > 1 for at least one sperm
    # check if we have any 'motile'
    assert any(result['Motility'] == 'motile'), "At least one sperm should be labeled 'motile' at threshold=1."


def test_alh(example_data):
    """
    Test the ALH function adds 'ALH_mean' and 'ALH_max' columns.
    """
    result = alh(example_data.copy(), fps=9, pixel_size=0.26, win_size=2)
    assert 'ALH_mean' in result.columns, "'ALH_mean' should be in the result."
    assert 'ALH_max' in result.columns, "'ALH_max' should be in the result."
    
    # Check that the values are numeric or NaN
    mean_val = result['ALH_mean'].iloc[0]
    max_val = result['ALH_max'].iloc[0]
    # It's possible they're NaN if there's not enough frames, but shouldn't be an error
    assert np.isnan(mean_val) or isinstance(mean_val, (float, int)), "ALH_mean should be a float or NaN."
    assert np.isnan(max_val) or isinstance(max_val, (float, int)), "ALH_max should be a float or NaN."


def test_bcf(example_data):
    """
    Test that bcf adds 'BCF' column and it's not negative.
    """
    result = bcf(example_data.copy(), fps=9, pixel_size=0.26, win_size=2)
    assert 'BCF' in result.columns, "The 'BCF' column should be added."
    
    # Check values are not negative
    for val in result['BCF']:
        if not pd.isna(val):
            assert val >= 0, "BCF values cannot be negative."
