import torch
from torch import nn


class TwoLayerFCNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int = 1):
        super(TwoLayerFCNN, self).__init__()

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        # First fully connected layer
        self.fc1 = nn.Linear(input_size, hidden_size)
        # Activation function (ReLU)
        self.relu = nn.ReLU()
        # Second fully connected layer
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out