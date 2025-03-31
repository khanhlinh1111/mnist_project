import torch
import torch.nn.functional as F
from tqdm import tqdm  # import tqdm for progress bars
from models.crnn import ctc_greedy_decoder

def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} training", unit="batch")
    for batch_idx, (data, targets, input_lengths, target_lengths) in enumerate(progress_bar):
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()
        output = model(data)  # output shape: (T, batch, num_classes)
        output_log_softmax = F.log_softmax(output, dim=2)
        loss = criterion(output_log_softmax, targets, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()
        # Update progress bar with the current loss
        progress_bar.set_postfix(loss=loss.item())

def evaluate(model, device, eval_loader, blank=10):
    model.eval()
    all_preds = []
    all_targets = []
    progress_bar = tqdm(eval_loader, desc="Evaluating", unit="batch")
    with torch.no_grad():
        for data, targets, input_lengths in progress_bar:
            data = data.to(device)
            output = model(data)  # shape: (T, batch, num_classes)
            output_log_softmax = F.log_softmax(output, dim=2)
            preds = ctc_greedy_decoder(output_log_softmax, blank=blank)
            all_preds.extend(preds)
            all_targets.extend(targets)
            # Optionally, update progress bar with current count of processed sequences:
            progress_bar.set_postfix(processed=len(all_targets))
            
    total_sequences = len(all_targets)
    seq_correct = sum([1 for pred, tgt in zip(all_preds, all_targets) if pred == tgt])
    sequence_accuracy = seq_correct / total_sequences if total_sequences > 0 else 0

    total_digits = 0
    correct_digits = 0
    for pred, tgt in zip(all_preds, all_targets):
        for p, t in zip(pred, tgt):
            total_digits += 1
            if p == t:
                correct_digits += 1
    digit_accuracy = correct_digits / total_digits if total_digits > 0 else 0

    print(f"\nEvaluation -- Sequence Accuracy: {sequence_accuracy*100:.2f}%  |  Digit Accuracy: {digit_accuracy*100:.2f}%")
    return sequence_accuracy, digit_accuracy
