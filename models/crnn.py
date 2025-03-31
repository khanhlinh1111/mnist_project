# models/crnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

class CRNN(nn.Module):
    def __init__(self, num_classes=11, hidden_size=256, num_layers=2):
        """
        num_classes: 11 classes (digits 0-9 and a CTC blank at index 10)
        hidden_size: LSTM hidden dimension
        num_layers: number of LSTM layers
        """
        super(CRNN, self).__init__()
        # Load a pretrained ResNet-50 and remove its last two layers (avgpool and fc)
        self.cnn = nn.Sequential(*list(resnet50(pretrained=True).children())[:-2])
        # Reduce channels from 2048 to 256 with a 1x1 convolution
        self.conv_reduce = nn.Conv2d(2048, 256, kernel_size=1)
        # Adaptive pooling: force height dimension to 1 while keeping width variable
        self.pool = nn.AdaptiveAvgPool2d((1, None))
        # Bidirectional LSTM to model the sequence
        self.lstm = nn.LSTM(input_size=256, hidden_size=hidden_size, num_layers=num_layers, bidirectional=True)
        # Fully connected layer to predict num_classes at each time step
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # x: (batch, 3, H, W)
        features = self.cnn(x)              # -> (batch, 2048, H', W')
        features = self.conv_reduce(features)  # -> (batch, 256, H', W')
        features = self.pool(features)           # -> (batch, 256, 1, W')
        features = features.squeeze(2)           # -> (batch, 256, W')
        features = features.permute(2, 0, 1)       # -> (W', batch, 256)
        lstm_out, _ = self.lstm(features)          # -> (W', batch, 2*hidden_size)
        output = self.fc(lstm_out)                 # -> (W', batch, num_classes)
        return output

def ctc_greedy_decoder(output, blank=10):
    """
    Decodes the output of the network (log probabilities) using greedy decoding.
    Collapses repeated predictions and removes the blank token.
    """
    # output: (T, batch, num_classes)
    max_probs = torch.argmax(output, dim=2)  # -> (T, batch)
    predictions = []
    for b in range(max_probs.shape[1]):
        pred = []
        previous = None
        for t in max_probs[:, b]:
            t = t.item()
            if t != previous:
                if t != blank:
                    pred.append(t)
                previous = t
        predictions.append(pred)
    return predictions
