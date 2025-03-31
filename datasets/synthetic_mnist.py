# datasets/synthetic_mnist.py
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image

class SyntheticMNISTSequence(Dataset):
    def __init__(self, mnist_data, num_samples=100000, min_digits=3, max_digits=7, transform=None, spacing=5):
        """
        mnist_data: a torchvision MNIST dataset
        num_samples: total number of synthetic sequence samples to generate
        min_digits, max_digits: range for the number of digits in each sequence
        transform: transformation to apply to the concatenated image (e.g., resize, normalize)
        spacing: number of pixels between concatenated digit images
        """
        self.mnist_data = mnist_data
        self.num_samples = num_samples
        self.min_digits = min_digits
        self.max_digits = max_digits
        self.transform = transform
        self.spacing = spacing

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Randomly choose sequence length
        seq_length = random.randint(self.min_digits, self.max_digits)
        digit_images = []
        labels = []
        for _ in range(seq_length):
            rand_idx = random.randint(0, len(self.mnist_data) - 1)
            img, label = self.mnist_data[rand_idx]
            if img.mode != 'L':
                img = img.convert('L')
            digit_images.append(img)
            labels.append(label)

        # Create a new blank image and paste the digits side by side
        widths = [im.width for im in digit_images]
        total_width = sum(widths) + self.spacing * (seq_length - 1)
        max_height = max(im.height for im in digit_images)
        new_img = Image.new('L', (total_width, max_height))
        x_offset = 0
        for im in digit_images:
            new_img.paste(im, (x_offset, 0))
            x_offset += im.width + self.spacing

        if self.transform is not None:
            new_img = self.transform(new_img)

        # Return the image and the sequence of labels as a tensor
        return new_img, torch.tensor(labels, dtype=torch.long)

def collate_fn_train(batch):
    """
    Pads images in a batch to the same width and flattens the target sequences
    for CTC loss training.
    """
    images, targets = zip(*batch)
    max_width = max(img.shape[2] for img in images)
    batch_images = []
    input_lengths = []
    for img in images:
        pad_width = max_width - img.shape[2]
        padded = F.pad(img, (0, pad_width, 0, 0), "constant", 0)
        batch_images.append(padded)
        # Approximate length after downsampling (e.g., factor ~32)
        input_lengths.append(padded.shape[2] // 32)
    batch_images = torch.stack(batch_images, dim=0)
    
    flat_targets = []
    target_lengths = []
    for tgt in targets:
        flat_targets.extend(tgt.tolist())
        target_lengths.append(len(tgt))
    flat_targets = torch.tensor(flat_targets, dtype=torch.long)
    input_lengths = torch.tensor(input_lengths, dtype=torch.long)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)
    
    return batch_images, flat_targets, input_lengths, target_lengths

def collate_fn_eval(batch):
    """
    Pads images in a batch to the same width but keeps targets as lists for evaluation.
    """
    images, targets = zip(*batch)
    max_width = max(img.shape[2] for img in images)
    batch_images = []
    orig_targets = []
    input_lengths = []
    for img, tgt in zip(images, targets):
        pad_width = max_width - img.shape[2]
        padded = F.pad(img, (0, pad_width, 0, 0), "constant", 0)
        batch_images.append(padded)
        orig_targets.append(tgt.tolist())
        input_lengths.append(padded.shape[2] // 32)
    batch_images = torch.stack(batch_images, dim=0)
    input_lengths = torch.tensor(input_lengths, dtype=torch.long)
    return batch_images, orig_targets, input_lengths
