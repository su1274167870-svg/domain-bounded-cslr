import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from pathlib import Path
import importlib.util

spec = importlib.util.spec_from_file_location(
    "ctc_model",
    os.path.join(os.path.dirname(__file__), '../src/cslr/models/ctc.py')
)
ctc_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ctc_module)
CTCModel = ctc_module.CTCModel

vocab = {}
idx_to_word = {}
with open('data/vocab.txt', 'r') as f:
    for idx, line in enumerate(f):
        word = line.strip()
        vocab[word] = idx + 1
        idx_to_word[idx + 1] = word
vocab_size = len(vocab)

ctc_model = CTCModel(input_size=368, hidden_size=512, num_layers=3, vocab_size=vocab_size, dropout=0.3)
ctc_model.load_state_dict(torch.load('artifacts/checkpoints/ctc_model_final.pt', map_location='cpu'))
ctc_model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ctc_model.to(device)

features = np.load('data/processed/ce_csl/train-00001.npy')
features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)

with torch.no_grad():
    logits = ctc_model(features_tensor)
    logits = logits.squeeze(0)
    preds = torch.argmax(logits, dim=-1).cpu().numpy()

# Remove blank tokens
result = []
for idx in preds:
    if idx != vocab_size:
        word = idx_to_word.get(idx, 'UNKNOWN')
        if word not in result or result[-1] != word:
            result.append(word)

print(f"CTC only prediction: {' '.join(result)}")
