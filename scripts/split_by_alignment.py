"""
Split .npy features into isolated Gloss segments using CTC alignment results.
Each segment is saved as a separate .npy file with naming: sample_id_seg_idx_gloss.npy
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import csv
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# Paths
alignment_dir = Path('data/alignment')
feature_dir = Path('data/processed/ce_csl')
output_dir = Path('data/processed/isolated')
output_dir.mkdir(parents=True, exist_ok=True)

stats = defaultdict(int)
alignment_files = list(alignment_dir.glob('*.csv'))
print(f"Found {len(alignment_files)} alignment files")

for align_path in tqdm(alignment_files, desc="Splitting features"):
    sample_id = align_path.stem

    # Load alignment results
    alignments = []
    with open(align_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            alignments.append({
                'frame': int(row['frame']),
                'gloss_idx': int(row['gloss_idx']),
                'gloss': row['gloss']
            })

    if not alignments:
        continue

    # Load features
    feature_path = feature_dir / f"{sample_id}.npy"
    if not feature_path.exists():
        continue
    features = np.load(feature_path)

    # Group consecutive identical Gloss tokens into segments
    segments = []
    current_gloss = alignments[0]['gloss']
    start_frame = alignments[0]['frame']
    last_frame = alignments[0]['frame']

    for align in alignments[1:]:
        if align['gloss'] == current_gloss:
            last_frame = align['frame']
        else:
            end_frame = last_frame + 1
            seg_feat = features[start_frame:end_frame]
            if len(seg_feat) > 0:
                segments.append({'gloss': current_gloss, 'features': seg_feat})
            current_gloss = align['gloss']
            start_frame = align['frame']
            last_frame = align['frame']

    # Save the last segment
    if alignments:
        end_frame = last_frame + 1
        seg_feat = features[start_frame:end_frame]
        if len(seg_feat) > 0:
            segments.append({'gloss': current_gloss, 'features': seg_feat})

    # Save each segment
    for seg_idx, seg in enumerate(segments):
        gloss = seg['gloss']
        seg_feat = seg['features']
        if len(seg_feat) == 0:
            continue
        # Filter out punctuation and single-char noise
        if len(gloss) <= 1 and gloss in ['。', '？', '，', '、']:
            continue
        filename = f"{sample_id}_{seg_idx:03d}_{gloss}.npy"
        save_path = output_dir / filename
        np.save(save_path, seg_feat)
        stats[gloss] += 1

print(f"Split complete!")
print(f"Output directory: {output_dir}")
print(f"Total segments: {sum(stats.values())}")
print(f"Unique Gloss words: {len(stats)}")

# Top 20 most frequent segments
sorted_stats = sorted(stats.items(), key=lambda x: -x[1])
print("\nTop 20 Gloss segments:")
for word, count in sorted_stats[:20]:
    print(f"  {word}: {count}")
