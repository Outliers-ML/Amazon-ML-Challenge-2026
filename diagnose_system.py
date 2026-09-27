import pandas as pd
import numpy as np
from collections import Counter

print("=" * 60)
print("1. EVALUATING TARGET COLLISIONS IN PREDICTIONS")
print("=" * 60)

for name, path in [
    ("0.787 Baseline (tau=0.6687)", "output_full/matching_results.tsv"),
    ("0.745 Run (Calibrated 2D)", "output_full/matching_results_calibrated_2d.tsv")
]:
    df = pd.read_csv(path, sep='\t')
    all_targets = []
    for matches in df['matched_entity_ids'].dropna():
        s = str(matches).strip()
        if s:
            all_targets.extend(s.split(','))
            
    counts = Counter(all_targets)
    multi = {k: v for k, v in counts.items() if v > 1}
    total_claims = len(all_targets)
    colliding = sum(multi.values()) - len(multi)
    print(f"\n--- {name} ---")
    print(f"Total target assignments: {total_claims:,}")
    print(f"Unique targets claimed:   {len(counts):,}")
    print(f"Colliding claims (dups):  {colliding:,} (in {len(multi):,} unique targets)")

print("\n" + "=" * 60)
print("2. GROUND TRUTH TARGET SHARING (Can targets be shared?)")
print("=" * 60)
gt = pd.read_csv('student_resource/dataset/train/train_ground_truth.tsv', sep='\t')
gt_clean = gt[gt['matched_entity_ids'].fillna('').str.strip() != '']
gt_targets = []
for matches in gt_clean['matched_entity_ids']:
    gt_targets.extend(str(matches).split(','))

gt_counts = Counter(gt_targets)
gt_multi = {k: v for k, v in gt_counts.items() if v > 1}
print(f"Total GT target links:    {len(gt_targets):,}")
print(f"Unique GT targets:        {len(gt_counts):,}")
print(f"Targets shared in GT:     {len(gt_multi):,} ({len(gt_multi)/len(gt_counts)*100:.2f}%)")

print("\n" + "=" * 60)
print("3. BLOCKER RECALL CEILING ON TRAINING SET")
print("=" * 60)
total_gt_pairs = len(gt_targets)
try:
    meta = np.load('models_full/training_data.npz', allow_pickle=True)
    y = meta['y']
    mined_pos = int((y == 1).sum())
    recall = (mined_pos / total_gt_pairs) * 100
    print(f"Total True GT Pairs:       {total_gt_pairs:,}")
    print(f"GT Pairs Found by Blocker: {mined_pos:,}")
    print(f"Blocking Recall Ceiling:   {recall:.2f}%")
except Exception as e:
    print(f"Error checking training_data.npz: {e}")
