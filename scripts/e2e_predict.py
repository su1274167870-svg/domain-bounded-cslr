"""
End-to-end prediction pipeline:
Video feature -> CTC alignment -> segment classification -> Gloss sequence
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from pathlib import Path
import importlib.util

# Load CTC model
spec = importlib.util.spec_from_file_location(
    "ctc_model",
    os.path.join(os.path.dirname(__file__), '../src/cslr/models/ctc.py')
)
ctc_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ctc_module)
CTCModel = ctc_module.CTCModel

# Load classifier model
import torch.nn as nn

class TCNClassifier(nn.Module):
    def __init__(self, input_size=368, num_classes=100, hidden_channels=128):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_channels, num_classes)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.relu(self.conv2(x))
        x = self.dropout(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return x


def load_vocab():
    vocab = {}
    idx_to_word = {}
    with open('data/vocab.txt', 'r') as f:
        for idx, line in enumerate(f):
            word = line.strip()
            vocab[word] = idx + 1
            idx_to_word[idx + 1] = word
    return vocab, idx_to_word


def load_models():
    vocab, idx_to_word = load_vocab()
    vocab_size = len(vocab)
    
    # CTC model
    ctc_model = CTCModel(input_size=368, hidden_size=512, num_layers=3, vocab_size=vocab_size, dropout=0.3)
    ctc_model.load_state_dict(torch.load('artifacts/checkpoints/ctc_model_final.pt', map_location='cpu'))
    ctc_model.eval()
    
    # Build label mapping for classifier
    from collections import Counter
    import glob
    label_counter = Counter()
    for f in glob.glob('data/processed/isolated/*.npy'):
        label = Path(f).stem.split('_')[-1]
        label_counter[label] += 1
    
    min_samples = 5
    valid_labels = {label for label, count in label_counter.items() if count >= min_samples}
    label_to_idx = {label: idx for idx, label in enumerate(sorted(valid_labels))}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    num_classes = len(label_to_idx)
    
    classifier = TCNClassifier(input_size=368, num_classes=num_classes, hidden_channels=128)
    classifier.load_state_dict(torch.load('artifacts/checkpoints/isolated_classifier.pt', map_location='cpu'))
    classifier.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ctc_model.to(device)
    classifier.to(device)
    
    return ctc_model, classifier, vocab, idx_to_word, label_to_idx, idx_to_label, device


def predict_video(feature_path, ctc_model, classifier, vocab, idx_to_word, label_to_idx, idx_to_label, device):
    """Run end-to-end prediction on a video feature file."""
    
    # Load features
    features = np.load(feature_path)
    features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
    
    # CTC alignment
    with torch.no_grad():
        logits = ctc_model(features_tensor)
        logits = logits.squeeze(0)
        preds = torch.argmax(logits, dim=-1).cpu().numpy()
    
    vocab_size = len(vocab)
    
    # Extract segments
    segments = []
    current_gloss = None
    start_frame = None
    
    for i, gloss_idx in enumerate(preds):
        if gloss_idx == vocab_size:  # blank
            if current_gloss is not None:
                segments.append({
                    'gloss': current_gloss,
                    'start': start_frame,
                    'end': i,
                    'frames': features[start_frame:i]
                })
                current_gloss = None
                start_frame = None
        else:
            gloss = idx_to_word.get(gloss_idx, 'UNKNOWN')
            if current_gloss is None:
                current_gloss = gloss
                start_frame = i
            elif gloss != current_gloss:
                segments.append({
                    'gloss': current_gloss,
                    'start': start_frame,
                    'end': i,
                    'frames': features[start_frame:i]
                })
                current_gloss = gloss
                start_frame = i
    
    if current_gloss is not None:
        segments.append({
            'gloss': current_gloss,
            'start': start_frame,
            'end': len(preds),
            'frames': features[start_frame:len(preds)]
        })
    
    # Classify each segment
    recognized = []
    for seg in segments:
        seg_features = seg['frames']
        if len(seg_features) == 0:
            continue
        
        # Pad/truncate
        max_len = 20
        if len(seg_features) > max_len:
            seg_features = seg_features[:max_len]
        elif len(seg_features) < max_len:
            pad = np.zeros((max_len - len(seg_features), seg_features.shape[1]))
            seg_features = np.vstack([seg_features, pad])
        
        seg_tensor = torch.tensor(seg_features, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits = classifier(seg_tensor)
            pred_idx = logits.argmax(dim=-1).item()
            pred_label = idx_to_label.get(pred_idx, 'UNKNOWN')
        
        recognized.append(pred_label)
    
    return {
        'original_glosses': [seg['gloss'] for seg in segments],
        'recognized_glosses': recognized,
        'raw_alignment': [idx_to_word.get(idx, 'BLANK') if idx != vocab_size else 'BLANK' for idx in preds]
    }


if __name__ == "__main__":
    # Test on a sample
    print("Loading models...")
    ctc_model, classifier, vocab, idx_to_word, label_to_idx, idx_to_label, device = load_models()
    
    # Test on train-00001
    feature_path = 'data/processed/ce_csl/train-00001.npy'
    if os.path.exists(feature_path):
        result = predict_video(feature_path, ctc_model, classifier, vocab, idx_to_word, label_to_idx, idx_to_label, device)
        print("Sample train-00001")
        print(f"Original: {' '.join(result['original_glosses'])}")
        print(f"Recognized: {' '.join(result['recognized_glosses'])}")
    else:
        print(f"Sample file not found: {feature_path}")
