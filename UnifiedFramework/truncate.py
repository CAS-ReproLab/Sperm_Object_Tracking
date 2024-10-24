import cv2
import sys
import os

def extract_first_25_frames(input_file, output_file):
    # Open the input video file
    cap = cv2.VideoCapture(input_file)

    if not cap.isOpened():
        print(f"Error: Unable to open video file {input_file}")
        return

    # Get the video's width, height, and frame rate
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create a VideoWriter object to save the output video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    # Read and write the first 25 frames
    for i in range(25):
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Unable to read frame {i}")
            break
        out.write(frame)

    # Release resources
    cap.release()
    out.release()
    print(f"First 25 frames saved to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python extract_frames.py <input_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = "first_25_frames.avi"

    # Check if the input file exists
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} not found")
        sys.exit(1)

    extract_first_25_frames(input_file, output_file)
