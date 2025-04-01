import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

class CRNN(nn.Module):
    def __init__(self, num_classes=11, conv_channels=256, num_conv_layers=3, kernel_size=3):
        """
        num_classes: 11 classes (digits 0-9 and a CTC blank at index 10)
        conv_channels: Number of channels in the convolutional layers
        num_conv_layers: Number of 1D convolutional layers
        kernel_size: Kernel size of the 1D convolutional layers
        """
        super(CRNN, self).__init__()
        # Load a pretrained ResNet-50
        resnet = resnet50(pretrained=False)
        # Remove the last two layers (avgpool and fc)
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])

        # Reduce channels from 2048 to 256 with a 1x1 convolution
        self.conv_reduce = nn.Conv2d(2048, conv_channels, kernel_size=1)
        # Adaptive pooling: force height dimension to 1 while keeping width variable
        self.pool = nn.AdaptiveAvgPool2d((1, 56))

        # Sequential 1D convolutional layers
        conv_layers = []
        in_channels = conv_channels
        for _ in range(num_conv_layers):
            conv_layers.append(nn.Conv1d(in_channels, conv_channels, kernel_size, padding=kernel_size // 2))
            conv_layers.append(nn.BatchNorm1d(conv_channels))
            conv_layers.append(nn.ReLU(inplace=True))
            in_channels = conv_channels
        self.sequential = nn.Sequential(*conv_layers)

        # Fully connected layer to predict num_classes at each time step
        self.fc = nn.Linear(conv_channels, num_classes)

    def forward(self, x):
        # x: (batch, 3, H, W)
        features = self.cnn(x)          # -> (batch, 2048, H', W')
        features = self.conv_reduce(features)   # -> (batch, 256, H', W')
        features = self.pool(features)       # -> (batch, 256, 1, W')
        features = features.squeeze(2)       # -> (batch, 256, W')
        features = features.permute(0, 2, 1)     # -> (batch, W', 256)

        # Apply sequential convolutional layers
        features = features.permute(0, 2, 1)     # -> (batch, 256, W')
        conv_out = self.sequential(features)    # -> (batch, 256, W')
        conv_out = conv_out.permute(2, 0, 1)     # -> (W', batch, 256)

        output = self.fc(conv_out)            # -> (W', batch, num_classes)
        return output

def ctc_greedy_decoder(output, blank=10):
    """
    Decodes the output of the network (log probabilities) using greedy decoding.
    Collapses repeated predictions and removes the blank token.
    """
    # output: (T, batch, num_classes)
    max_probs = torch.argmax(output, dim=2)   # -> (T, batch)
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