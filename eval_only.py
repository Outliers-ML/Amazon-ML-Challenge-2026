import pandas as pd
from src.pipeline.er_trainer import compute_macro_f05

gt = pd.read_csv("student_resource/dataset/train/train_ground_truth.tsv", sep="\t", dtype=str)
gt_map = {
    str(s1): set(m for m in str(matches).split(",") if m.strip())
    for s1, matches in zip(gt.iloc[:, 0], gt["matched_entity_ids"])
    if pd.notna(matches)
}
for s1, matches in zip(gt.iloc[:, 0], gt["matched_entity_ids"]):
    if pd.isna(matches) or str(matches).strip() == "":
        gt_map[str(s1)] = set()

pred = pd.read_csv("output_mini_eval/matching_results.tsv", sep="\t", dtype=str)
pred_map = {
    str(s1): [m for m in str(matches).split(",") if m.strip()]
    for s1, matches in zip(pred.iloc[:, 0], pred["matched_entity_ids"])
    if pd.notna(matches)
}
for s1, matches in zip(pred.iloc[:, 0], pred["matched_entity_ids"]):
    if pd.isna(matches) or str(matches).strip() == "":
        pred_map[str(s1)] = []

valid_s1s = set(pred_map.keys())
gt_map_slice = {k: v for k, v in gt_map.items() if k in valid_s1s}

score = compute_macro_f05(gt_map_slice, pred_map)
print(f"Test-Pipeline F0.5 on Train Slice (Mini Model, corrected mining): {score:.4f}")
