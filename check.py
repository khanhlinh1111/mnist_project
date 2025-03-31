import torch
import torch.nn as nn
from models.crnn import CRNN
# Load your trained CRNN model (adjust class and parameters as per your implementation)
model = CRNN(num_classes=11, hidden_size=256, num_layers=2)  # Replace with your actual model definition
model.load_state_dict(torch.load('crnn_mnist.pth'))  # Path to your trained model weights
model.eval()

# Test with different input widths based on your expected range
input_widths = [100, 200, 300, 400, 1000]  # Adjust based on your dataset or inference expectations
for width in input_widths:
    # Assuming input height is fixed (e.g., 224 from your transforms), adjust as needed
    sample_input = torch.randn(1, 3, 28, width)  # batch=1, channels=3, height=224, width=variable
    with torch.no_grad():
        # Pass through CNN layers up to just before pooling
        features = model.cnn(sample_input)  # Output shape: (batch, channels, H', W')
        features = model.conv_reduce(features)  # Adjust if your model has a conv_reduce layer
        W_prime = features.size(3)  # Width of the feature map (W')
        print(f"Input width: {width}, Feature map width (W'): {W_prime}")