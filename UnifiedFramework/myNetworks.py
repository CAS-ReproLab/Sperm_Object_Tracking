import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMNetwork(nn.Module):
    def __init__(self, in_channels,out_channels,hidden_size=128):
        super(LSTMNetwork, self).__init__()

        self.lstm = nn.LSTM(in_channels, hidden_size, batch_first=True)
        self.projection = nn.Linear(hidden_size, out_channels)

    def forward(self, x, video_frames=None):

        # x shape: (batch_size, seq_len, in_channels)
        x, _ = self.lstm(x)
        x = self.projection(x[:, -1, :])
        x = F.sigmoid(x)
        return x