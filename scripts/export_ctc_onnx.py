import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import importlib.util
spec = importlib.util.spec_from_file_location(
    "ctc_model",
    os.path.join(os.path.dirname(__file__), '../src/cslr/models/ctc.py')
)
ctc_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ctc_module)
CTCModel = ctc_module.CTCModel

vocab = {}
with open('data/vocab.txt', 'r') as f:
    for idx, line in enumerate(f):
        vocab[line.strip()] = idx + 1
vocab_size = len(vocab)
print(f"Vocabulary size: {vocab_size}")

model = CTCModel(input_size=368, hidden_size=512, num_layers=3, vocab_size=vocab_size, dropout=0.3)
model.load_state_dict(torch.load('artifacts/checkpoints/ctc_model_final.pt', map_location='cpu'))
model.eval()

dummy_input = torch.randn(1, 48, 368)

# Export with single file using torch.jit first, then ONNX
torch.onnx.export(
    model,
    dummy_input,
    'artifacts/exports/ctc_model.onnx',
    input_names=['features'],
    output_names=['logits'],
    dynamic_axes={'features': {0: 'batch'}, 'logits': {0: 'batch'}},
    opset_version=17,
    export_params=True,
    do_constant_folding=True
)
print("ONNX exported to artifacts/exports/ctc_model.onnx")
