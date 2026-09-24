"""Multi-Key Inverted Index and Blocking Engine for Candidate Generation.

Indexes Source 2 and Source 3 records by Name tokens, Character 3-grams, and
Postal+Street anchors within country partitions. Retrieves and ranks top candidates
for Source 1 records and outputs compliant candidate_pairs.tsv.
"""

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from src.data.normalizer import TextNormalizer


class MultiKeyBlocker:
    """Multi-key inverted index blocker for cross-source candidate retrieval."""

    def __init__(self, max_candidates: int = 25) -> None:
        self.max_candidates = max_candidates
        self.normalizer = TextNormalizer()

    def block_country_partition(
        self,
        s1_df: pd.DataFrame,
        s2_df: pd.DataFrame,
        s3_df: pd.DataFrame,
    ) -> Dict[str, List[str]]:
        """Index S2 and S3 and query S1 records to retrieve top candidates."""
        token_index: Dict[str, List[str]] = defaultdict(list)
        anchor_index: Dict[str, List[str]] = defaultdict(list)
        ngram_index: Dict[str, List[str]] = defaultdict(list)

        # 1. Index S2 and S3 targets
        dfs_to_concat = [df for df in [s2_df, s3_df] if not df.empty]
        if dfs_to_concat:
            targets = pd.concat(dfs_to_concat, ignore_index=True)
        else:
            targets = pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"])

        for _, row in targets.iterrows():
            if "entity_id" not in row or pd.isna(row["entity_id"]):
                continue
            tid = str(row["entity_id"])
            name_obj = self.normalizer.normalize_name(row.get("business_name", ""))
            addr_obj = self.normalizer.normalize_address(row.get("business_address", ""))

            # Index distinctive name tokens
            for tok in set(name_obj.tokens):
                if len(tok) >= 3:
                    token_index[tok].append(tid)

            # Index 3-grams
            for ng in set(name_obj.char_3grams):
                if ng:
                    ngram_index[ng].append(tid)

            # Index address anchor (postal + street number)
            if addr_obj.postal_code and addr_obj.street_number:
                key = f"{addr_obj.postal_code}_{addr_obj.street_number}"
                anchor_index[key].append(tid)

        # 2. Query S1 records
        candidates_map: Dict[str, List[str]] = {}
        for _, row in s1_df.iterrows():
            if "entity_id" not in row or pd.isna(row["entity_id"]):
                continue
            s1_id = str(row["entity_id"])
            name_obj = self.normalizer.normalize_name(row.get("business_name", ""))
            addr_obj = self.normalizer.normalize_address(row.get("business_address", ""))

            cand_scores: Dict[str, float] = defaultdict(float)

            # Channel A: Name tokens
            for tok in set(name_obj.tokens):
                if len(tok) >= 3 and tok in token_index:
                    for tid in token_index[tok][:100]:
                        cand_scores[tid] += 2.0

            # Channel B: 3-grams
            for ng in set(name_obj.char_3grams):
                if ng and ng in ngram_index:
                    for tid in ngram_index[ng][:50]:
                        cand_scores[tid] += 0.5

            # Channel C: Anchor
            if addr_obj.postal_code and addr_obj.street_number:
                key = f"{addr_obj.postal_code}_{addr_obj.street_number}"
                if key in anchor_index:
                    for tid in anchor_index[key]:
                        cand_scores[tid] += 5.0

            if not cand_scores:
                candidates_map[s1_id] = []
                continue

            # Rank and keep top K (tie-break deterministically by ID)
            sorted_cands = sorted(cand_scores.items(), key=lambda x: (-x[1], x[0]))
            candidates_map[s1_id] = [c[0] for c in sorted_cands[:self.max_candidates]]

        return candidates_map

    @staticmethod
    def write_candidate_pairs(candidates_map: Dict[str, List[str]], output_path: Path | str) -> None:
        """Write candidate pairs TSV matching competition schema."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tcandidate_entity_ids\n")
            for s1_id in sorted(candidates_map.keys()):
                cand_str = ",".join(candidates_map[s1_id])
                f.write(f"{s1_id}\t{cand_str}\n")
