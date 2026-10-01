import pandas as pd
from src.data.blocking import MultiTierBlocker

s1 = pd.read_csv("student_resource/dataset/train/train_source1.tsv", sep="\t").head(1000)
s2 = pd.read_csv("student_resource/dataset/train/train_source2.tsv", sep="\t").head(100000)
s3 = pd.read_csv("student_resource/dataset/train/train_source3.tsv", sep="\t").head(100000)

print("1. Running Test-time Blocker (max_candidates=25, max_postings=500)")
b_test = MultiTierBlocker(max_candidates=25, max_postings=500)
cands_test = b_test.block_country_partition(s1, s2, s3)
print(f"Total candidates: {sum(len(c) for c in cands_test.values())}")

print("\n2. Running Train-time Blocker (max_candidates=35, max_postings=250)")
b_train = MultiTierBlocker(max_candidates=35, bucket_ceiling=250)
cands_train = b_train.block_country_partition(s1, s2, s3)
print(f"Total candidates: {sum(len(c) for c in cands_train.values())}")
