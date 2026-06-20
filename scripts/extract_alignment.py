import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import csv
from pathlib import Path
from tqdm import tqdm
import importlib.util

# Import model via spec
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

# Load vocabulary
vocab = {}
idx_to_word = {}
with open('data/vocab.txt', 'r') as f:
    for idx, line in enumerate(f):
        word = line.strip()
        vocab[word] = idx + 1
        idx_to_word[idx + 1] = word

vocab_size = len(vocab)
print(f"Vocabulary size: {vocab_size}")

# Create model with the same architecture as training
model = CTCModel(
    input_size=368,
    hidden_size=512,
    num_layers=3,
    vocab_size=vocab_size,
    dropout=0.3
)

model_path = 'artifacts/checkpoints/ctc_model_final.pt'
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Dataset
dataset = CTCVideoDataset(
    manifest_path='data/manifests/ce_csl.csv',
    feature_dir='data/processed/ce_csl',
    vocab=vocab
)

# Output directory
output_dir = Path('data/alignment')
output_dir.mkdir(parents=True, exist_ok=True)

# Process each sample
for idx in tqdm(range(len(dataset)), desc="Extracting alignment"):
    sample = dataset[idx]
    sample_id = sample['id']
    features = sample['features'].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(features)
        logits = logits.squeeze(0)
        preds = torch.argmax(logits, dim=-1)

    preds = preds.cpu().numpy()
    alignment = []
    for frame_idx, gloss_idx in enumerate(preds):
        if gloss_idx != vocab_size:  # Skip blank
            alignment.append({
                'frame': frame_idx,
                'gloss_idx': int(gloss_idx),
                'gloss': idx_to_word.get(int(gloss_idx), 'UNKNOWN')
            })

    with open(output_dir / f"{sample_id}.csv", 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['frame', 'gloss_idx', 'gloss'])
        writer.writeheader()
        writer.writerows(alignment)

print(f"Alignment results saved to {output_dir}")
