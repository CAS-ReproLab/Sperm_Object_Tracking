
from torch.utils.data import Dataset
import torch
import numpy as np
from torchvision import io
#from torchcodec.decoders import VideoDecoder
import pandas as pd

import os

class VISEMSimpleDataset(Dataset):
    def __init__(self, root_dir, seq_len=8):
        super(VISEMSimpleDataset, self).__init__()

        self.root_dir = root_dir

        file_list = os.listdir(root_dir)
        self.label_list = [f for f in file_list if f.endswith('.csv')]
        self.video_list = [f for f in file_list if f.endswith('.mp4')]

        self.length = len(self.label_list)

        #self.decoder = VideoDecoder()

    def __getitem__(self, idx):
        
        # Load the video file
        video_path = os.path.join(self.root_dir, self.video_list[idx])
        
        #video_frames = self.decoder.decode(video_path)
        #video_frames = io.read_video(video_path,pts_unit="sec")[0]  # Read video frames using torchvision

        # Load the label file
        label_path = video_path.replace('.mp4', '_labels.csv')
        labels_df = pd.read_csv(label_path)

        # Pick a random sperm cell from the video
        sperm_ids = labels_df['sperm'].unique()
        random_sperm_id = np.random.choice(sperm_ids)
        sperm_labels = labels_df[labels_df['sperm'] == random_sperm_id]
        sperm_labels = sperm_labels.sort_values(by='frame')
        sperm_labels = sperm_labels.reset_index(drop=True)

        # Drop frames to match the sequence length
        if len(sperm_labels) > 8:
            sperm_labels = sperm_labels.iloc[:8]
        elif len(sperm_labels) < 8:
            sperm_labels = sperm_labels.reindex(range(8), fill_value=-1.0)

        # Get the frames for the selected sperm cell
        start_frame = sperm_labels['frame'].min()
        end_frame = sperm_labels['frame'].max()
        #sperm_frames = video_frames[start_frame:end_frame + 1]

        # Get the coordinates of the sperm cell
        x_coords = sperm_labels['x'].values
        y_coords = sperm_labels['y'].values

        # Normalize the coordinates to the range [0, 1]
        if x_coords.max() > 1:
            height, width = 480, 640 #sperm_frames[0].shape[1:3]
            x_coords = x_coords / width
            y_coords = y_coords / height

        coords = np.stack((x_coords, y_coords), axis=1)
        coords = torch.tensor(coords, dtype=torch.float32)

        return coords

    def __len__(self):
        return self.length
    
