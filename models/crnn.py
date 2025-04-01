import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

class CRNN(nn.Module):
    def __init__(self, num_classes=11, num_conv_layers=3, conv_channels=256):
        """
        num_classes: 11 classes (digits 0-9 and a CTC blank at index 10)
        num_conv_layers: number of convolutional layers for sequence modeling
        conv_channels: number of channels in convolutional layers
        """
        super(CRNN, self).__init__()
        # Load a pretrained ResNet-50 and remove its last two layers (avgpool and fc)
        self.cnn = nn.Sequential(*list(resnet50(pretrained=False).children())[:-2])
        # Reduce channels from 2048 to conv_channels with a 1x1 convolution
        self.conv_reduce = nn.Conv2d(2048, conv_channels, kernel_size=1)
        # Use fixed AvgPool2d to collapse the height dimension.
        # For 224x224 input, ResNet50 produces a 7x7 feature map; we pool over the height dimension.
        self.pool = nn.AvgPool2d(kernel_size=(7, 1))
        # Build sequence modeling layers using 2D convolutions that act like 1D convolutions.
        conv_layers = []
        for _ in range(num_conv_layers):
            # Here, the convolution only operates along the width.
            conv_layers.append(nn.Conv2d(conv_channels, conv_channels, kernel_size=(1, 3), padding=(0, 1)))
            conv_layers.append(nn.BatchNorm2d(conv_channels))
            conv_layers.append(nn.ReLU())
        self.conv_layers = nn.Sequential(*conv_layers)
        # Final convolution: 1x1 to map to num_classes.
        self.final_conv = nn.Conv2d(conv_channels, num_classes, kernel_size=(1, 1))
        
    def forward(self, x):
        # x: (batch, 3, H, W) e.g., (batch, 3, 224, 224)
        features = self.cnn(x)                # -> (batch, 2048, 7, 7)
        features = self.conv_reduce(features) # -> (batch, conv_channels, 7, 7)
        features = self.pool(features)        # -> (batch, conv_channels, 1, 7)
        features = self.conv_layers(features) # -> (batch, conv_channels, 1, 7)
        output = self.final_conv(features)    # -> (batch, num_classes, 1, 7)
        output = output.squeeze(2)            # -> (batch, num_classes, 7)
        output = output.permute(2, 0, 1)        # -> (7, batch, num_classes)
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