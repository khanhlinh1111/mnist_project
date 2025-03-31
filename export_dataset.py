#!/usr/bin/env python
import os
from torchvision import transforms
from PIL import Image
import torchvision
from datasets.synthetic_mnist import SyntheticMNISTSequence
from tqdm import tqdm  # progress bar library

def main():
    # Download MNIST training set
    mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True)

    # Create two versions of the synthetic dataset that return the raw PIL image.
    raw_train_dataset = SyntheticMNISTSequence(
        mnist_data=mnist_train,
        num_samples=100000,   # number of training samples to export
        min_digits=3,
        max_digits=7,
        transform=None,       # no transform, so raw PIL image is returned
        spacing=5
    )

    raw_eval_dataset = SyntheticMNISTSequence(
        mnist_data=mnist_train,
        num_samples=10000,    # number of evaluation samples to export
        min_digits=3,
        max_digits=7,
        transform=None,
        spacing=5
    )

    # Create an export directory with subdirectories for training and evaluation if they don't exist.
    export_dir = "./calibration_dataset"
    train_export_dir = os.path.join(export_dir, "train")
    eval_export_dir = os.path.join(export_dir, "eval")
    os.makedirs(train_export_dir, exist_ok=True)
    os.makedirs(eval_export_dir, exist_ok=True)

    # Function to export a dataset given an export directory and a dataset object.
    def export_dataset(dataset, export_path, prefix="sample"):
        for i in tqdm(range(len(dataset)), desc=f"Exporting to {export_path}"):
            img, label_tensor = dataset[i]
            # Create a label string (e.g., "123" for the sequence [1,2,3])
            label_str = ''.join(map(str, label_tensor.tolist()))
            filename = os.path.join(export_path, f"{prefix}_{i}_{label_str}.png")
            img.save(filename)
        print(f"Exported {len(dataset)} images to {export_path}")

    # Export both datasets with progress indicators
    export_dataset(raw_train_dataset, train_export_dir, prefix="train")
    export_dataset(raw_eval_dataset, eval_export_dir, prefix="eval")


if __name__ == "__main__":
    main()
