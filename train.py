import os
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm  # for progress bars

# Import the synthetic dataset and collate functions
from datasets.synthetic_mnist import SyntheticMNISTSequence, collate_fn_train, collate_fn_eval
from models.crnn import CRNN
from training.train_eval import train, evaluate

# Define the transformation for both cases
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.expand(3, -1, -1)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Check if the exported dataset exists
    train_export_dir = "./calibration_dataset/train"
    eval_export_dir = "./calibration_dataset/eval"
    if os.path.exists(train_export_dir) and os.path.exists(eval_export_dir):
        print("Loading exported dataset...")
        from datasets.load_dataset import ExportedSyntheticMNISTDataset
        train_dataset = ExportedSyntheticMNISTDataset(train_export_dir, transform=transform)
        eval_dataset = ExportedSyntheticMNISTDataset(eval_export_dir, transform=transform)
        collate_fn_train_used = collate_fn_train
        collate_fn_eval_used = collate_fn_eval
    else:
        print("Exported dataset not found. Creating synthetic dataset on the fly...")
        # Download MNIST training set
        mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True)
        # Create synthetic datasets from MNIST
        train_dataset = SyntheticMNISTSequence(
            mnist_data=mnist_train,
            num_samples=100000,
            min_digits=3,
            max_digits=7,
            transform=transform,
            spacing=5
        )
        eval_dataset = SyntheticMNISTSequence(
            mnist_data=mnist_train,
            num_samples=10000,
            min_digits=3,
            max_digits=7,
            transform=transform,
            spacing=5
        )
        collate_fn_train_used = collate_fn_train
        collate_fn_eval_used = collate_fn_eval

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True, 
                              collate_fn=collate_fn_train_used, num_workers=2)
    eval_loader = DataLoader(eval_dataset, batch_size=12, shuffle=False, 
                             collate_fn=collate_fn_eval_used, num_workers=2)

    # Initialize model, optimizer, and CTC loss (blank index=10)
    model = CRNN(num_classes=11, hidden_size=256, num_layers=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CTCLoss(blank=10, zero_infinity=True)

    epochs = 2  # Adjust epochs as necessary
    best_seq_acc = 0.0  # To track the best sequence accuracy
    for epoch in tqdm(range(1, epochs + 1), desc="Training epochs"):
        print(f"\nEpoch {epoch}:")
        train(model, device, train_loader, optimizer, criterion, epoch)
        seq_acc, digit_acc = evaluate(model, device, eval_loader, blank=10)
        
        # Save checkpoint for the current epoch
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'sequence_accuracy': seq_acc,
            'digit_accuracy': digit_acc,
        }
        torch.save(checkpoint, "last_crnn_mnist.pth")
        print("Saved checkpoint: last_crnn_mnist.pth")
        
        # Save best checkpoint if current sequence accuracy is higher
        if seq_acc > best_seq_acc:
            best_seq_acc = seq_acc
            torch.save(checkpoint, "best_crnn_mnist.pth")
            print("New best accuracy! Saved checkpoint: best_crnn_mnist.pth")
    
    print("Training complete.")

if __name__ == '__main__':
    main()
