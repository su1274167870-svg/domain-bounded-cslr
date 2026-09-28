"""
CTC training script for frame-level Gloss alignment.
Trains an LSTM + CTC model on .npy features and saves checkpoints.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
from tqdm import tqdm

import importlib.util

# Import model and dataset via spec to avoid circular import
spec = importlib.util.spec_from_file_location(
    "ctc_model",
    os.path.join(os.path.dirname(__file__), '../src/cslr/models/ctc.py')
)
ctc_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ctc_module)
CTCModel = ctc_module.CTCModel

spec2 = importlib.util.spec_from_file_location(
    "ctc_dataset",
    os.path.join(os.path.dirname(__file__), '../src/cslr/data/ctc_dataset.py')
)
dataset_module = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(dataset_module)
CTCVideoDataset = dataset_module.CTCVideoDataset


def collate_fn(batch):
    """Custom collate function for variable-length sequences."""
    features = [b['features'] for b in batch]
    labels = [b['labels'] for b in batch]
    label_lengths = torch.tensor([b['label_length'] for b in batch])
    seq_lens = torch.tensor([b['seq_len'] for b in batch])
    ids = [b['id'] for b in batch]

    padded_features = pad_sequence(features, batch_first=True, padding_value=0.0)

    return {
        'features': padded_features,
        'labels': labels,
        'label_lengths': label_lengths,
        'seq_lens': seq_lens,
        'ids': ids
    }


# Load vocabulary
vocab = {}
with open('data/vocab.txt', 'r') as f:
    for idx, line in enumerate(f):
        vocab[line.strip()] = idx + 1

print(f"Vocabulary size: {len(vocab)}")

# Dataset
dataset = CTCVideoDataset(
    manifest_path='data/manifests/ce_csl.csv',
    feature_dir='data/processed/ce_csl',
    vocab=vocab
)

# Train/val split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

# Model - larger hidden size for better capacity
model = CTCModel(input_size=368, hidden_size=512, num_layers=3, vocab_size=len(vocab), dropout=0.3)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
ctc_loss = nn.CTCLoss(blank=len(vocab), zero_infinity=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(f"Using device: {device}")

# Training loop - 60 epochs for better convergence
for epoch in range(60):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        features = batch['features'].to(device)
        labels = torch.cat(batch['labels']).to(device)
        label_lengths = batch['label_lengths']
        seq_lens = batch['seq_lens']

        logits = model(features)
        logits = logits.permute(1, 0, 2)

        loss = ctc_loss(logits, labels, seq_lens, label_lengths)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}")

    # Save checkpoint every 10 epochs
    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), f'artifacts/checkpoints/ctc_model_epoch{epoch+1}.pt')
        print(f"Checkpoint saved to artifacts/checkpoints/ctc_model_epoch{epoch+1}.pt")

# Save final model
torch.save(model.state_dict(), 'artifacts/checkpoints/ctc_model_final.pt')
print("Final model saved to artifacts/checkpoints/ctc_model_final.pt")
