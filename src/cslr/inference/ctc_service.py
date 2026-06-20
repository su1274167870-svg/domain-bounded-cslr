from __future__ import annotations

import os
import time
import numpy as np
from pathlib import Path
import onnxruntime as ort

class CTCInferenceService:
    def __init__(self, model_path: Path, vocab_path: Path):
        self.model_path = model_path
        self.vocab = self._load_vocab(vocab_path)
        self.vocab_size = len(self.vocab)
        self.idx_to_word = {v: k for k, v in self.vocab.items()}
        
        self.session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
        
    def _load_vocab(self, vocab_path):
        vocab = {}
        with open(vocab_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                vocab[line.strip()] = idx + 1
        return vocab
    
    def predict(self, features: np.ndarray) -> dict:
        start_time = time.perf_counter()
        
        if features.ndim == 2:
            features = features[np.newaxis, :, :]
        
        inputs = {self.session.get_inputs()[0].name: features.astype(np.float32)}
        outputs = self.session.run(None, inputs)
        logits = outputs[0]
        
        preds = np.argmax(logits[0], axis=-1)
        
        blank_idx = self.vocab_size
        gloss_sequence = []
        last_word = None
        for idx in preds:
            if idx != blank_idx:
                word = self.idx_to_word.get(idx, 'UNKNOWN')
                if word != last_word:
                    gloss_sequence.append(word)
                    last_word = word
        
        inference_time = (time.perf_counter() - start_time) * 1000
        
        return {
            'gloss_sequence': ' '.join(gloss_sequence),
            'gloss_list': gloss_sequence,
            'inference_time_ms': inference_time,
            'raw_logits': logits,
            'blank_removed_preds': [self.idx_to_word.get(idx, 'BLANK') if idx != blank_idx else 'BLANK' for idx in preds]
        }


def create_ctc_service() -> CTCInferenceService:
    # Use WORKDIR /workspace in container
    project_root = Path("/workspace")
    model_path = Path(os.getenv("CTC_MODEL_PATH", str(project_root / "artifacts/exports/ctc_model.onnx")))
    vocab_path = Path(os.getenv("CTC_VOCAB_PATH", str(project_root / "data/vocab.txt")))
    
    if not model_path.exists():
        raise FileNotFoundError(f"CTC model not found: {model_path}")
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocab file not found: {vocab_path}")
    
    return CTCInferenceService(model_path, vocab_path)
