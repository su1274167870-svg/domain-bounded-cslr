import torch
from torch.utils.data import Dataset
import numpy as np
import csv
from pathlib import Path

class CTCVideoDataset(Dataset):
    def __init__(self, manifest_path, feature_dir, vocab):
        self.feature_dir = Path(feature_dir)
        self.vocab = vocab
        self.samples = []

        with open(manifest_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                sample_id = row['sample_id']
                gloss_seq = row['label']
                gloss_list = [g for g in gloss_seq.split('/') if g.strip()]
                if not gloss_list:
                    continue
                self.samples.append({
                    'id': sample_id,
                    'gloss_list': gloss_list,
                    'label_ids': [vocab.get(g, 0) for g in gloss_list]
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        feature_path = self.feature_dir / f"{sample['id']}.npy"
        features = np.load(feature_path)  # (frames, 368)

        # 计算有效帧数（非零帧）
        valid_frames = (features.sum(axis=1) != 0).sum()
        if valid_frames == 0:
            valid_frames = features.shape[0]

        # 使用原始帧数，不截断
        seq_len = valid_frames
        features = features[:seq_len]

        return {
            'features': torch.tensor(features, dtype=torch.float32),
            'labels': torch.tensor(sample['label_ids'], dtype=torch.long),
            'label_length': len(sample['label_ids']),
            'seq_len': seq_len,
            'id': sample['id']
        }
