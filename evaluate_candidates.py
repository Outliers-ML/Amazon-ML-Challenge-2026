import pandas as pd

cands = pd.read_csv("output_train_eval/candidate_pairs.tsv", sep="\t", dtype=str)
# Each row is s1, cand
cand_counts = cands.groupby("source1_entity_id").size()

print("Test-time Blocker Output Candidate Count Stats:")
print(cand_counts.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]))
print(f"Entities with 25+ candidates: {(cand_counts >= 25).sum()} / {len(cand_counts)}")

