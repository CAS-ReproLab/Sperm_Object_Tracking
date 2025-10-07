import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import trackpy as tp


def generate_histogram(data, title, xlabel):
    """
    Generate a histogram for the given data.

    Parameters:
    data (pd.Series): Data to generate the histogram for.
    title (str): Title of the histogram.
    xlabel (str): Label for the x-axis.
    """
    plt.figure()
    plt.hist(data, bins='auto', edgecolor='black', alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()


def summarize_stats(csvfile):
    """
    Summarize the VAP and VCL statistics from a CSV file and generate histograms.

    Parameters:
    csvfile (str): Path to the CSV file containing sperm tracking data with columns "VAP" and "VCL".
    """
    try:
        # Load the data from the CSV file
        data = pd.read_csv(csvfile)
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        print(f"Error loading CSV file: {e}")
        return

    # Check if 'VAP' and 'VCL' columns exist
    if 'VAP' not in data.columns or 'VCL' not in data.columns or 'ALH_mean' not in data.columns :
        print("Error: The CSV file does not contain 'VAP' and 'VCL' columns.")
        return

    # Extract VAP and VCL values
    vaps = data['New_VAP'].dropna()  # Drop any NaN values
    vcls = data['VCL'].dropna()  # Drop any NaN values
    alh_mean = data['ALH_mean'].dropna()

    if vaps.empty or vcls.empty or alh_mean.empty:
        print("No VAP or VCL data available in the CSV file.")
        return

    # Get sperm density from csv file
    sperm_num = data['sperm']
    density = np.max(sperm_num)

    # Calculate mean and range for VAP
    mean_vap = np.mean(vaps)
    range_vap = (np.min(vaps), np.max(vaps))

    # Calculate mean and range for VCL
    mean_vcl = np.mean(vcls)
    range_vcl = (np.min(vcls), np.max(vcls))

    # Calculate mean and range for VCL
    mean_alh = np.mean(alh_mean)
    range_alh = (np.min(vcls), np.max(alh_mean))

    # Print summary
    print(f"Summary of VAP, VCL, and ALH Statistics for {csvfile}:")
    print(f"Cell Density: {density}")
    print(f"Average Path Velocity (VAP):")
    print(f"  Mean: {mean_vap:.2f}")
    print(f"  Range: {range_vap[0]:.2f} - {range_vap[1]:.2f}")
    print(f"Curvilinear Velocity (VCL):")
    print(f"  Mean: {mean_vcl:.2f}")
    print(f"  Range: {range_vcl[0]:.2f} - {range_vcl[1]:.2f}")
    print(f"Curvilinear Velocity (VCL):")
    print(f"  Mean: {mean_alh:.2f}")
    print(f"  Range: {range_alh[0]:.2f} - {range_alh[1]:.2f}")

    # Generate histograms
    generate_histogram(vaps, "Average Path Velocity (VAP)", "VAP (microns/second)")
    generate_histogram(vcls, "Curvilinear Velocity (VCL)", "VCL (microns/second)")
    generate_histogram(alh_mean, "Amplitude Lateral Head Displacment  (ALH Mean)", "VCL (microns/second)")

if __name__ == '__main__':
    # Setup argument parser to only take a CSV file as input
    parser = argparse.ArgumentParser(description='Generate histograms from sperm tracking CSV data')
    parser.add_argument('csvfile', type=str, help='Path to the CSV file')

    # Parse the arguments
    args = parser.parse_args()
    csvfile = args.csvfile

    # Summarize statistics and generate histograms
    summarize_stats(csvfile)

    # Example console command python plot.py csv_file