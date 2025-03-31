# datasets/exported_dataset.py
import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class ExportedSyntheticMNISTDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        root_dir: Directory where the exported images are stored.
        transform: Transformation to apply to the image.
        """
        self.root_dir = root_dir
        self.transform = transform
        # List all PNG files in the directory
        self.files = sorted([
            os.path.join(root_dir, f) 
            for f in os.listdir(root_dir) if f.endswith('.png')
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filepath = self.files[idx]
        image = Image.open(filepath).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        # Extract label from filename: expected format: "prefix_index_label.png"
        # e.g. "train_0_123.png" -> label "123"
        basename = os.path.basename(filepath)
        label_str = os.path.splitext(basename)[0].split('_')[-1]
        label = torch.tensor([int(ch) for ch in label_str], dtype=torch.long)
        return image, label
