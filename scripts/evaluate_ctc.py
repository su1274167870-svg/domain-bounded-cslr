"""
Evaluate CTC model on CE-CSL dev/test splits.
Computes sequence accuracy and WER using Levenshtein distance.
"""

import sys
import csv
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import importlib.util
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

logger.info("=" * 60)
logger.info("CTC Model Evaluation on CE-CSL")
logger.info("=" * 60)

# 1. Load vocabulary
logger.info("Step 1/6: Loading vocabulary...")
vocab = {}
idx_to_word = {}
vocab_path = Path('data/vocab.txt')
if not vocab_path.exists():
    logger.error(f"Vocabulary file not found: {vocab_path}")
    sys.exit(1)

with open(vocab_path, 'r') as f:
    for idx, line in enumerate(f):
        word = line.strip()
        vocab[word] = idx + 1
        idx_to_word[idx + 1] = word
vocab_size = len(vocab)
blank_idx = vocab_size
logger.info(f"  Vocabulary size: {vocab_size}")
logger.info(f"  Blank index: {blank_idx}")

# 2. Load CTC model
logger.info("Step 2/6: Loading CTC model...")
spec = importlib.util.spec_from_file_location(
    "ctc_model",
    "src/cslr/models/ctc.py"
)
if spec is None:
    logger.error("Failed to locate ctc_model module")
    sys.exit(1)

ctc_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ctc_module)
CTCModel = ctc_module.CTCModel

model = CTCModel(
    input_size=368,
    hidden_size=512,
    num_layers=3,
    vocab_size=vocab_size,
    dropout=0.3
)

checkpoint_path = Path('artifacts/checkpoints/ctc_model_final.pt')
if not checkpoint_path.exists():
    logger.error(f"Checkpoint not found: {checkpoint_path}")
    sys.exit(1)

model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
model.eval()
logger.info(f"  Model loaded from: {checkpoint_path}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
logger.info(f"  Device: {device}")

# 3. Load manifest
logger.info("Step 3/6: Loading manifest...")
manifest_path = Path('data/manifests/ce_csl_full.csv')
if not manifest_path.exists():
    logger.error(f"Manifest not found: {manifest_path}")
    logger.info("  Trying alternative: data/manifests/ce_csl.csv")
    manifest_path = Path('data/manifests/ce_csl.csv')
    if not manifest_path.exists():
        logger.error("No manifest found")
        sys.exit(1)

samples = {'dev': [], 'test': []}
with open(manifest_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        split = row.get('split', '').strip().lower()
        if split == 'dev' or split == 'validation':
            samples['dev'].append(row)
        elif split == 'test':
            samples['test'].append(row)

logger.info(f"  Dev samples: {len(samples['dev'])}")
logger.info(f"  Test samples: {len(samples['test'])}")

if not samples['dev'] and not samples['test']:
    logger.error("No dev or test samples found in manifest")
    sys.exit(1)

# 4. Define helper functions
logger.info("Step 4/6: Initializing helper functions...")

def decode_ctc(preds, blank_idx, idx_to_word):
    """Convert frame-level predictions to Gloss sequence."""
    result = []
    last = blank_idx
    for p in preds:
        if p != blank_idx and p != last:
            result.append(idx_to_word.get(p, 'UNKNOWN'))
        last = p
    return ' '.join(result)

def wer(ref, hyp):
    """Word Error Rate using Levenshtein distance."""
    ref_words = ref.split()
    hyp_words = hyp.split()
    try:
        import Levenshtein
        return Levenshtein.distance(ref_words, hyp_words) / max(len(ref_words), 1)
    except ImportError:
        logger.warning("Levenshtein not installed, using fallback WER")
        return sum(1 for r, h in zip(ref_words, hyp_words) if r != h) / max(len(ref_words), 1)

logger.info("  Helpers initialized")

# 5. Run evaluation
logger.info("Step 5/6: Running evaluation...")
results = {}

for split in ['dev', 'test']:
    if not samples[split]:
        logger.warning(f"No {split} samples found, skipping")
        continue

    logger.info(f"\n--- Evaluating {split.upper()} ({len(samples[split])} samples) ---")
    start_time = time.time()

    total_wer = 0
    correct_seq = 0
    total = 0
    refs = []
    hyps = []
    sample_ids = []
    error_samples = []

    for row in tqdm(samples[split], desc=f"  Processing {split}"):
        sample_id = row['sample_id']
        true_gloss = row['label']

        feat_path = Path(f'data/processed/ce_csl/{sample_id}.npy')
        if not feat_path.exists():
            logger.debug(f"  Feature not found: {feat_path}")
            continue

        features = np.load(feat_path)
        if features.shape[0] == 0:
            logger.debug(f"  Empty feature: {feat_path}")
            continue

        feat_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(feat_tensor)
            preds = torch.argmax(logits.squeeze(0), dim=-1).cpu().numpy()

        pred_gloss = decode_ctc(preds, blank_idx, idx_to_word)
        true_gloss_clean = ' '.join(true_gloss.split('/'))

        refs.append(true_gloss_clean)
        hyps.append(pred_gloss)
        sample_ids.append(sample_id)

        total += 1
        if pred_gloss == true_gloss_clean:
            correct_seq += 1
        else:
            error_samples.append((sample_id, true_gloss_clean, pred_gloss))

        total_wer += wer(true_gloss_clean, pred_gloss)

    elapsed = time.time() - start_time
    seq_acc = correct_seq / total if total > 0 else 0
    avg_wer = total_wer / total if total > 0 else 0

    results[split] = {
        'seq_acc': seq_acc,
        'wer': avg_wer,
        'total': total,
        'refs': refs,
        'hyps': hyps,
        'sample_ids': sample_ids,
        'error_samples': error_samples
    }

    logger.info(f"\n  === {split.upper()} Results ===")
    logger.info(f"  Total samples processed: {total}")
    logger.info(f"  Correct sequences: {correct_seq}")
    logger.info(f"  Sequence accuracy: {seq_acc:.2%}")
    logger.info(f"  Word Error Rate (WER): {avg_wer:.2%}")
    logger.info(f"  Evaluation time: {elapsed:.2f}s")
    logger.info(f"  Avg time per sample: {elapsed/total:.2f}s")

# 6. Save results
logger.info("Step 6/6: Saving results...")
output_dir = Path('artifacts/metrics')
output_dir.mkdir(parents=True, exist_ok=True)

# Save detailed predictions
detail_path = output_dir / 'ce_csl_eval_detailed.csv'
with open(detail_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['split', 'sample_id', 'reference', 'hypothesis'])
    for split in ['dev', 'test']:
        if split not in results:
            continue
        for i, sample_id in enumerate(results[split]['sample_ids']):
            writer.writerow([split, sample_id, results[split]['refs'][i], results[split]['hyps'][i]])

logger.info(f"  Detailed results saved to: {detail_path}")

# Save error samples
error_path = output_dir / 'ce_csl_eval_errors.csv'
with open(error_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['split', 'sample_id', 'reference', 'hypothesis'])
    for split in ['dev', 'test']:
        if split not in results:
            continue
        for sample_id, ref, hyp in results[split]['error_samples']:
            writer.writerow([split, sample_id, ref, hyp])

logger.info(f"  Error samples saved to: {error_path}")

# Save summary
summary_path = output_dir / 'ce_csl_eval_summary.txt'
with open(summary_path, 'w') as f:
    f.write("=== CTC Model Evaluation Summary ===\n")
    f.write(f"Model: CTC (LSTM+CTC)\n")
    f.write(f"Checkpoint: {checkpoint_path}\n")
    f.write(f"Vocabulary size: {vocab_size}\n\n")

    for split in ['dev', 'test']:
        if split not in results:
            continue
        r = results[split]
        f.write(f"=== {split.upper()} ({r['total']} samples) ===\n")
        f.write(f"Sequence Accuracy: {r['seq_acc']:.2%}\n")
        f.write(f"WER: {r['wer']:.2%}\n")
        f.write(f"Correct sequences: {r['seq_acc'] * r['total']:.0f}/{r['total']}\n\n")

logger.info(f"  Summary saved to: {summary_path}")

logger.info("\n" + "=" * 60)
logger.info("Evaluation complete!")
logger.info("=" * 60)

print("\n=== Final Results ===")
for split in ['dev', 'test']:
    if split in results:
        print(f"{split.upper()}: Sequence Accuracy={results[split]['seq_acc']:.2%}, WER={results[split]['wer']:.2%}")
