"""
Evaluate CTC model on CE-CSL dev/test splits.
Computes:
- Sentence-level Sequence Accuracy: % of samples where predicted gloss matches reference exactly.
- Corpus-level WER: Total edit distance across all samples / Total reference words across all samples.
  This is the standard WER metric used in ASR/SLR literature.

Definitions:
- Sentence-level Sequence Accuracy = (correct sequences) / (total sequences)
- Corpus-level WER = sum(Levenshtein distance per sample) / sum(reference length per sample)

Outputs:
- artifacts/metrics/ce_csl_eval_summary.txt: Summary of results
- artifacts/metrics/ce_csl_eval_detailed.csv: Predictions for every sample
- artifacts/metrics/ce_csl_eval_errors.csv: Only samples with errors (for debugging)
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

# ============================================================================
# Step 1: Load vocabulary
# ============================================================================
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
        vocab[word] = idx + 1          # idx 0 reserved for padding
        idx_to_word[idx + 1] = word
vocab_size = len(vocab)
blank_idx = vocab_size                 # blank token index = vocab_size
logger.info(f"  Vocabulary size: {vocab_size}")
logger.info(f"  Blank index: {blank_idx}")

# ============================================================================
# Step 2: Load CTC model
# ============================================================================
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

# Model architecture parameters (must match training)
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

# Load weights with CPU first, then move to device if available
model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
model.eval()
logger.info(f"  Model loaded from: {checkpoint_path}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
logger.info(f"  Device: {device}")

# ============================================================================
# Step 3: Load manifest with split information
# ============================================================================
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

# ============================================================================
# Step 4: Helper functions
# ============================================================================
logger.info("Step 4/6: Initializing helper functions...")

def decode_ctc(preds, blank_idx, idx_to_word):
    """
    Convert frame-level predictions (indices) to a single Gloss sequence string.

    Args:
        preds: 1D array of predicted indices for each frame
        blank_idx: Index of the blank token (to be removed)
        idx_to_word: Mapping from index to Gloss word string

    Returns:
        Space-separated Gloss sequence (e.g., "我 想 去 医院")
    """
    result = []
    last = blank_idx
    for p in preds:
        # Skip blank tokens and consecutive duplicates
        if p != blank_idx and p != last:
            result.append(idx_to_word.get(p, 'UNKNOWN'))
        last = p
    return ' '.join(result)


def sentence_wer(ref_words, hyp_words):
    """
    Compute WER for a single sentence pair using Levenshtein distance.

    Args:
        ref_words: List of reference words
        hyp_words: List of hypothesis words

    Returns:
        Levenshtein distance between ref and hyp
    """
    try:
        import Levenshtein
        return Levenshtein.distance(ref_words, hyp_words)
    except ImportError:
        # Fallback: simple count of mismatches (less accurate)
        logger.warning("Levenshtein not installed, using fallback WER")
        return sum(1 for r, h in zip(ref_words, hyp_words) if r != h) + abs(len(ref_words) - len(hyp_words))


logger.info("  Helper functions initialized")

# ============================================================================
# Step 5: Run evaluation on dev and test sets
# ============================================================================
logger.info("Step 5/6: Running evaluation...")
results = {}

for split in ['dev', 'test']:
    if not samples[split]:
        logger.warning(f"No {split} samples found, skipping")
        continue

    logger.info(f"\n--- Evaluating {split.upper()} ({len(samples[split])} samples) ---")
    start_time = time.time()

    # Accumulators for metrics
    total_edit_distance = 0    # Sum of Levenshtein distances across all samples
    total_ref_words = 0        # Sum of reference word counts across all samples
    correct_seq = 0            # Count of exact matches
    total = 0                  # Total samples processed

    # Storage for detailed outputs
    refs = []                  # Reference sequences
    hyps = []                  # Hypothesis sequences
    sample_ids = []            # Sample IDs
    error_samples = []         # Samples where prediction != reference

    for row in tqdm(samples[split], desc=f"  Processing {split}"):
        sample_id = row['sample_id']
        true_gloss = row['label']

        # Load pre-extracted MediaPipe features
        feat_path = Path(f'data/processed/ce_csl/{sample_id}.npy')
        if not feat_path.exists():
            logger.debug(f"  Feature not found: {feat_path}")
            continue

        features = np.load(feat_path)
        if features.shape[0] == 0:
            logger.debug(f"  Empty feature: {feat_path}")
            continue

        # Run model inference
        feat_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(feat_tensor)
            # Get most likely class for each frame
            preds = torch.argmax(logits.squeeze(0), dim=-1).cpu().numpy()

        # Decode frame-level predictions to Gloss sequence
        pred_gloss = decode_ctc(preds, blank_idx, idx_to_word)

        # Clean reference: convert '/' separated to space separated
        true_gloss_clean = ' '.join(true_gloss.split('/'))

        # Store results
        refs.append(true_gloss_clean)
        hyps.append(pred_gloss)
        sample_ids.append(sample_id)

        total += 1
        ref_words = true_gloss_clean.split()
        hyp_words = pred_gloss.split()

        # Accumulate for corpus-level WER
        total_edit_distance += sentence_wer(ref_words, hyp_words)
        total_ref_words += len(ref_words)

        # Check for exact match (sentence-level sequence accuracy)
        if pred_gloss == true_gloss_clean:
            correct_seq += 1
        else:
            error_samples.append((sample_id, true_gloss_clean, pred_gloss))

    elapsed = time.time() - start_time

    # Compute final metrics
    # Sentence-level sequence accuracy
    seq_acc = correct_seq / total if total > 0 else 0.0
    # Corpus-level WER = total_edit_distance / total_ref_words
    corpus_wer = total_edit_distance / total_ref_words if total_ref_words > 0 else 0.0

    results[split] = {
        'seq_acc': seq_acc,
        'corpus_wer': corpus_wer,
        'total': total,
        'correct': correct_seq,
        'total_edit_distance': total_edit_distance,
        'total_ref_words': total_ref_words,
        'refs': refs,
        'hyps': hyps,
        'sample_ids': sample_ids,
        'error_samples': error_samples,
        'elapsed': elapsed
    }

    # Log results
    logger.info(f"\n  === {split.upper()} Results ===")
    logger.info(f"  Total samples processed: {total}")
    logger.info(f"  Correct sequences (exact match): {correct_seq}")
    logger.info(f"  Sentence-level Sequence Accuracy: {seq_acc:.2%}")
    logger.info(f"  Corpus-level WER: {corpus_wer:.2%}")
    logger.info(f"    (Total edit distance: {total_edit_distance}, Total reference words: {total_ref_words})")
    logger.info(f"  Evaluation time: {elapsed:.2f}s")
    logger.info(f"  Avg time per sample: {elapsed/total:.2f}s")

# ============================================================================
# Step 6: Save results to files
# ============================================================================
logger.info("\nStep 6/6: Saving results...")
output_dir = Path('artifacts/metrics')
output_dir.mkdir(parents=True, exist_ok=True)

# 6a: Detailed predictions for every sample
detail_path = output_dir / 'ce_csl_eval_detailed.csv'
with open(detail_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['split', 'sample_id', 'reference', 'hypothesis'])
    for split in ['dev', 'test']:
        if split not in results:
            continue
        for i, sample_id in enumerate(results[split]['sample_ids']):
            writer.writerow([split, sample_id, results[split]['refs'][i], results[split]['hyps'][i]])
logger.info(f"  Detailed predictions: {detail_path}")

# 6b: Error samples only (for debugging)
error_path = output_dir / 'ce_csl_eval_errors.csv'
with open(error_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['split', 'sample_id', 'reference', 'hypothesis'])
    for split in ['dev', 'test']:
        if split not in results:
            continue
        for sample_id, ref, hyp in results[split]['error_samples']:
            writer.writerow([split, sample_id, ref, hyp])
logger.info(f"  Error samples: {error_path}")

# 6c: Summary file (human-readable)
summary_path = output_dir / 'ce_csl_eval_summary.txt'
with open(summary_path, 'w') as f:
    f.write("=" * 60 + "\n")
    f.write("CTC Model Evaluation Summary\n")
    f.write("=" * 60 + "\n")
    f.write(f"Model: CTC (LSTM+CTC)\n")
    f.write(f"Checkpoint: {checkpoint_path}\n")
    f.write(f"Vocabulary size: {vocab_size}\n")
    f.write(f"Blank index: {blank_idx}\n")
    f.write(f"Device: {device}\n\n")

    f.write("-" * 60 + "\n")
    f.write("Metrics Definition\n")
    f.write("-" * 60 + "\n")
    f.write("Sentence-level Sequence Accuracy: Percentage of samples where the\n")
    f.write("  predicted gloss sequence exactly matches the reference gloss sequence.\n\n")
    f.write("Corpus-level WER: Standard WER metric used in ASR/SLR literature.\n")
    f.write("  WER = Total edit distance across all samples / Total reference words across all samples\n")
    f.write("  where edit distance is computed using Levenshtein distance on word sequences.\n\n")

    for split in ['dev', 'test']:
        if split not in results:
            continue
        r = results[split]
        f.write("-" * 60 + "\n")
        f.write(f"{split.upper()} SET ({r['total']} samples)\n")
        f.write("-" * 60 + "\n")
        f.write(f"Sentence-level Sequence Accuracy: {r['seq_acc']:.2%}\n")
        f.write(f"  (Correct: {r['correct']}/{r['total']})\n")
        f.write(f"Corpus-level WER: {r['corpus_wer']:.2%}\n")
        f.write(f"  (Total edit distance: {r['total_edit_distance']}, Total reference words: {r['total_ref_words']})\n")
        f.write(f"Evaluation time: {r['elapsed']:.2f}s\n\n")

    f.write("=" * 60 + "\n")
    f.write("END OF SUMMARY\n")

logger.info(f"  Summary: {summary_path}")

# Print final results to console
logger.info("\n" + "=" * 60)
logger.info("Evaluation complete!")
logger.info("=" * 60)

print("\n=== Final Results ===")
for split in ['dev', 'test']:
    if split in results:
        r = results[split]
        print(f"{split.upper()}: Sequence Accuracy={r['seq_acc']:.2%}, Corpus-level WER={r['corpus_wer']:.2%}")
