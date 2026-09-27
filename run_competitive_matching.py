import time, zipfile, subprocess
import pandas as pd
from collections import defaultdict

CONFIG = {
    'France': {'sing': 0.52, 'sec': 0.58},
    'India':  {'sing': 0.46, 'sec': 0.52},
    'US':     {'sing': 0.55, 'sec': 0.60}
}
MAX_MATCHES = 15

print("============================================================")
print("Running Global Competitive 1-to-1 Bipartite Matching")
print("============================================================")
t0 = time.time()

# 1. Load and filter candidate pairs per country
dfs = []
for country, cfg in CONFIG.items():
    p_path = f'output_full/scored_pairs_{country}.parquet'
    c_df = pd.read_parquet(p_path, columns=['source1_id', 'candidate_id', 'prob'])
    
    # Keep candidates clearing the country's singleton threshold
    c_df = c_df[c_df['prob'] >= cfg['sing']]
    c_df['country'] = country
    c_df['tau_sec'] = cfg['sec']
    dfs.append(c_df)
    print(f"[{country}] Loaded {len(c_df):,} candidate pairs (>= {cfg['sing']})")

df_all = pd.concat(dfs, ignore_index=True)
print(f"Total candidate pool: {len(df_all):,} pairs. Sorting by confidence...")

# 2. Global Sort: highest confidence pairs get first claim
df_all = df_all.sort_values(by='prob', ascending=False)

# 3. Global 1-to-1 Competitive Assignment
claimed_targets = set()
s1_matches = defaultdict(list)
s1_count = defaultdict(int)

# Track whether an S1 has placed its primary match
s1_has_primary = set()

for s1, cand, prob, sec_thresh in zip(df_all['source1_id'], df_all['candidate_id'], df_all['prob'], df_all['tau_sec']):
    if s1_count[s1] >= MAX_MATCHES:
        continue
    if cand in claimed_targets:
        continue
        
    # Primary match: first candidate for this S1 (already >= tau_sing)
    # Secondary matches: must clear tau_sec
    if s1 not in s1_has_primary or prob >= sec_thresh:
        claimed_targets.add(cand)
        s1_matches[s1].append(cand)
        s1_count[s1] += 1
        s1_has_primary.add(s1)

print(f"Assigned {len(claimed_targets):,} unique targets to {len(s1_matches):,} S1 entities in {time.time()-t0:.1f}s")

# 4. Generate official output
test_s1 = pd.read_csv('student_resource/dataset/test/test_source1.tsv', sep='\t', usecols=['entity_id'])['entity_id'].tolist()
out_df = pd.DataFrame({'source1_entity_id': test_s1})
out_df['matched_entity_ids'] = out_df['source1_entity_id'].map(lambda x: ','.join(s1_matches[x]) if x in s1_matches else '')

out_path = 'output_full/matching_results_competitive.tsv'
out_df.to_csv(out_path, sep='\t', index=False)
print(f"Saved competitive matching results to {out_path}")

# 5. Verify 0 collisions
counts = out_df['matched_entity_ids'].fillna('').apply(lambda x: len(x.split(',')) if x.strip() else 0)
total = len(out_df)
sing = (counts == 0).sum()
print("\n--- Final Verification Profile ---")
print(f"Total S1:         {total:,}")
print(f"Singletons:       {sing:,} ({sing/total*100:.2f}%)")
print(f"Total Matches:    {counts.sum():,}")
print(f"Mean Matches:     {counts.mean():.3f}")
print("Match distribution:\n", counts.value_counts().sort_index())

# 6. Run official validator
subprocess.run([
    '/dist_home/suryansh/miniforge3/envs/outliers/bin/python',
    'student_resource/utils/validate_submission.py',
    '--matching', out_path,
    '--candidate', 'output_full/candidate_pairs.tsv',
    '--test-dir', 'student_resource/dataset/test'
])

# 7. Package
with zipfile.ZipFile('outliers_submission_competitive.zip', 'w', compression=zipfile.ZIP_DEFLATED) as z:
    z.write(out_path, arcname='matching_results.tsv')
    z.write('output_full/candidate_pairs.tsv', arcname='candidate_pairs.tsv')
print("Successfully generated outliers_submission_competitive.zip")
