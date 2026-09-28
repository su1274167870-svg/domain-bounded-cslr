"""
Train a classifier on isolated Gloss segments.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import Counter


class IsolatedSegmentDataset(Dataset):
    def __init__(self, data_dir, max_len=20, min_samples=5):
        self.data_dir = Path(data_dir)
        self.max_len = max_len
        self.samples = []
        
        files = list(self.data_dir.glob("*.npy"))
        print(f"Found {len(files)} segment files")
        
        label_counter = Counter()
        for f in files:
            label = f.stem.split('_')[-1]
            label_counter[label] += 1
        
        valid_labels = {label for label, count in label_counter.items() if count >= min_samples}
        print(f"Labels with >= {min_samples} samples: {len(valid_labels)}")
        
        self.label_to_idx = {label: idx for idx, label in enumerate(sorted(valid_labels))}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
        for f in files:
            label = f.stem.split('_')[-1]
            if label not in valid_labels:
                continue
            features = np.load(f)
            self.samples.append({
                'features': features,
                'label': label,
                'label_idx': self.label_to_idx[label]
            })
        
        print(f"Total samples: {len(self.samples)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        features = sample['features']
        
        if features.shape[0] > self.max_len:
            features = features[:self.max_len]
        elif features.shape[0] < self.max_len:
            pad = np.zeros((self.max_len - features.shape[0], features.shape[1]))
            features = np.vstack([features, pad])
        
        return {
            'features': torch.tensor(features, dtype=torch.float32),
            'label': torch.tensor(sample['label_idx'], dtype=torch.long)
        }


def collate_fn(batch):
    features = torch.stack([b['features'] for b in batch])
    labels = torch.tensor([b['label'] for b in batch])
    return {'features': features, 'labels': labels}


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


def main():
    data_dir = Path('data/processed/isolated')
    min_samples_per_label = 5
    max_len = 20
    batch_size = 64
    epochs = 60
    learning_rate = 0.001
    
    dataset = IsolatedSegmentDataset(data_dir, max_len=max_len, min_samples=min_samples_per_label)
    num_classes = len(dataset.label_to_idx)
    print(f"Number of classes: {num_classes}")
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    model = TCNClassifier(input_size=368, num_classes=num_classes, hidden_channels=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Using device: {device}")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            features = batch['features'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(features)
            loss = criterion(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        train_acc = correct / total
        avg_loss = total_loss / len(train_loader)
        
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(device)
                labels = batch['labels'].to(device)
                logits = model(features)
                preds = logits.argmax(dim=-1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = val_correct / val_total
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
    
    torch.save(model.state_dict(), 'artifacts/checkpoints/isolated_classifier.pt')
    print("Model saved to artifacts/checkpoints/isolated_classifier.pt")


if __name__ == "__main__":
    main()
