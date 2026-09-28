"""
CTC-based sequence-to-sequence model for frame-level Gloss prediction.
- Input: video frame features (batch, seq_len, 368)
- Output: per-frame logits over vocabulary + blank
"""

import torch
import torch.nn as nn

class CTCModel(nn.Module):
    def __init__(self, input_size=368, hidden_size=256, num_layers=2, vocab_size=3862, dropout=0.3):
        super().__init__()
        # Bidirectional LSTM for contextual frame modeling
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout, bidirectional=True
        )
        # Linear projection to vocabulary + 1 (for CTC blank)
        self.fc = nn.Linear(hidden_size * 2, vocab_size + 1)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        logits = self.fc(lstm_out)
        return logits
