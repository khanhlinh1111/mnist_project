#!/usr/bin/env python
import os
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from pytorch_nndct.apis import torch_quantizer

from models.crnn import CRNN  # Your CRNN model
from datasets.synthetic_mnist import SyntheticMNISTSequence, collate_fn_train

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--calib_dir', default="calibration_dataset",
                    help='Directory containing the exported calibration dataset (should have "train" and "eval" subfolders)')
parser.add_argument('--calib_split', default="train",
                    help='Which subfolder of calibration_dataset to use for calibration ("train" or "eval")')
parser.add_argument('--data_dir', default="./data",
                    help='Directory for downloading MNIST (used if calibration dataset not found)')
parser.add_argument('--model_dir', default="./",
                    help='Directory containing the trained model (.pth file)')
parser.add_argument('--config_file', default="kv260_quant_config.json",
                    help='Quantization configuration file')
parser.add_argument('--subset_len', default=1000, type=int,
                    help='Subset length for calibration')
parser.add_argument('--batch_size', default=4, type=int,
                    help='Batch size for calibration')
parser.add_argument('--quant_mode', default='calib', choices=['float', 'calib', 'test'],
                    help='Quantization mode')
parser.add_argument('--fast_finetune', dest='fast_finetune', action='store_true',
                    help='Enable fast finetuning before calibration')
parser.add_argument('--deploy', dest='deploy', action='store_true',
                    help='Export xmodel for deployment')
parser.add_argument('--inspect', dest='inspect', action='store_true',
                    help='Inspect model')
parser.add_argument('--target', default='kv260',
                    help='Target device for quantization, default is kv260')
args, _ = parser.parse_known_args()

def load_calibration_data_from_folder(calib_dir, split, batch_size=4, subset_len=None, transform=None):
    """
    Loads calibration data from a specified subfolder (split) inside calib_dir.
    Assumes an ExportedSyntheticMNISTDataset exists in datasets/exported_dataset.py.
    """
    split_dir = os.path.join(calib_dir, split)
    if not os.path.exists(split_dir):
        raise RuntimeError(f"Calibration folder for split '{split}' not found: {split_dir}")
    from datasets.load_dataset import ExportedSyntheticMNISTDataset
    dataset = ExportedSyntheticMNISTDataset(split_dir, transform=transform)
    if subset_len is not None:
        from torch.utils.data import Subset
        indices = random.sample(range(len(dataset)), subset_len)
        dataset = Subset(dataset, indices)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return data_loader

def load_synthetic_calibration_data(data_dir, batch_size=4, subset_len=None, transform=None):
    """
    Fallback: Downloads MNIST and generates synthetic sequences for calibration.
    """
    import torchvision
    mnist_train = torchvision.datasets.MNIST(root=data_dir, train=True, download=True)
    dataset = SyntheticMNISTSequence(
        mnist_data=mnist_train,
        num_samples=100000,  # total synthetic samples; subset will be taken
        min_digits=3,
        max_digits=7,
        transform=transform,
        spacing=5
    )
    if subset_len is not None:
        from torch.utils.data import Subset
        indices = random.sample(range(len(dataset)), subset_len)
        dataset = Subset(dataset, indices)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_fn_train, num_workers=2)
    return data_loader

def evaluate(model, val_loader, criterion):
    """
    Evaluates the model on calibration data and returns the average CTCLoss.
    """
    model.eval()
    model.to(device)
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for images, flat_targets, input_lengths, target_lengths in val_loader:
            images = images.to(device)
            flat_targets = flat_targets.to(device)
            input_lengths = input_lengths.to(device)
            target_lengths = target_lengths.to(device)
            outputs = model(images)  # expected shape: (T, batch, num_classes)
            outputs_log_softmax = F.log_softmax(outputs, dim=2)
            loss = criterion(outputs_log_softmax, flat_targets, input_lengths, target_lengths)
            total_loss += loss.item() * images.size(0)
            count += images.size(0)
    return total_loss / count if count > 0 else 0

def quantization(title='Quantizing CRNN Model'):
    quant_mode = args.quant_mode
    finetune = args.fast_finetune
    deploy = args.deploy
    batch_size = args.batch_size
    subset_len = args.subset_len
    inspect = args.inspect
    config_file = args.config_file
    target = args.target
    calib_dir = args.calib_dir
    calib_split = args.calib_split
    data_dir = args.data_dir

    # For xmodel export, enforce batch_size=1 and subset_len=1
    if quant_mode != 'test' and deploy:
        deploy = False
        print('Warning: Exporting xmodel requires quant_mode to be "test". Disabling deploy.')
    if deploy and (batch_size != 1 or (subset_len is not None and subset_len != 1)):
        print('Warning: Exporting xmodel requires batch_size=1 and subset_len=1. Adjusting parameters.')
        batch_size = 1
        subset_len = 1

    # Load the CRNN model and its trained weights.
    model = CRNN(num_classes=11, hidden_size=256, num_layers=2).cpu()
    model_path = os.path.join(args.model_dir, 'crnn_mnist.pth')
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # Create a dummy input for calibration (224x224 image)
    dummy_input = torch.randn([batch_size, 3, 224, 224])

    if quant_mode == 'float':
        quant_model = model
        if inspect:
            if not target:
                raise RuntimeError("A target must be specified for inspection.")
            from pytorch_nndct.apis import Inspector
            inspector = Inspector(target)
            inspector.inspect(quant_model, (dummy_input,), device=device)
            return
    else:
        quantizer = torch_quantizer(quant_mode, model, (dummy_input,), device=device,
                                      quant_config_file=config_file, target=target)
        quant_model = quantizer.quant_model

    # Define the transformation used during training/quantization.
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.expand(3, -1, -1)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load calibration data from the specified calibration dataset folder.
    calib_path = os.path.join(calib_dir, calib_split)
    if os.path.exists(calib_path):
        print(f"Loading calibration data from exported calibration_dataset/{calib_split} ...")
        val_loader = load_calibration_data_from_folder(calib_dir, calib_split,
                                                       batch_size=batch_size,
                                                       subset_len=subset_len,
                                                       transform=transform)
    else:
        print("Exported calibration dataset not found. Generating synthetic calibration data...")
        val_loader = load_synthetic_calibration_data(data_dir,
                                                     batch_size=batch_size,
                                                     subset_len=subset_len,
                                                     transform=transform)

    # Use CTCLoss for evaluation (consistent with training)
    criterion = nn.CTCLoss(blank=10, zero_infinity=True).to(device)

    # Optional: fast finetuning before calibration
    if finetune:
        ft_loader = load_synthetic_calibration_data(data_dir, batch_size=batch_size, subset_len=5120, transform=transform)
        if quant_mode == 'calib':
            quantizer.fast_finetune(evaluate, (quant_model, ft_loader, criterion))
        elif quant_mode == 'test':
            quantizer.load_ft_param()

    avg_loss = evaluate(quant_model, val_loader, criterion)
    print("Evaluation Results - Average CTCLoss: {:.4f}".format(avg_loss))

    if quant_mode == 'calib':
        quantizer.export_quant_config()
    if deploy:
        quantizer.export_torch_script()
        quantizer.export_onnx_model()
        quantizer.export_xmodel(deploy_check=False)

if __name__ == '__main__':
    title = "CRNN Model Quantization"
    print("-------- Start {} --------".format(title))
    quantization(title=title)
    print("-------- End of {} --------".format(title))
