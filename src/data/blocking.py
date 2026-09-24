"""Multi-Key Inverted Index and Blocking Engine for Candidate Generation.

Indexes Source 2 and Source 3 records by Name tokens, Character 3-grams, and
Postal+Street anchors within country partitions. Retrieves and ranks top candidates
for Source 1 records and outputs compliant candidate_pairs.tsv.
"""

from collections import defaultdict
import math
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from src.data.normalizer import TextNormalizer


class MultiKeyBlocker:
    """Multi-key inverted index blocker for cross-source candidate retrieval."""

    def __init__(self, max_candidates: int = 25, max_postings: int = 500) -> None:
        self.max_candidates = max_candidates
        self.max_postings = max_postings
        self.normalizer = TextNormalizer()

    def block_country_partition(
        self,
        s1_df: pd.DataFrame,
        s2_df: pd.DataFrame,
        s3_df: pd.DataFrame,
        max_postings: Optional[int] = None,
    ) -> Dict[str, List[str]]:
        """Index S2 and S3 and query S1 records to retrieve top candidates."""
        limit_postings = max_postings if max_postings is not None else self.max_postings

        token_index: Dict[str, List[str]] = defaultdict(list)
        anchor_index: Dict[str, List[str]] = defaultdict(list)
        ngram_index: Dict[str, List[str]] = defaultdict(list)

        # 1. Index S2 and S3 targets
        dfs_to_concat = [df for df in [s2_df, s3_df] if not df.empty]
        if dfs_to_concat:
            targets = pd.concat(dfs_to_concat, ignore_index=True)
        else:
            targets = pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"])

        if not targets.empty and "entity_id" in targets.columns:
            targets_valid = targets[targets["entity_id"].notna()]
            target_ids = targets_valid["entity_id"].astype(str).str.strip().tolist()
            target_names = (
                targets_valid["business_name"].fillna("").tolist()
                if "business_name" in targets_valid.columns
                else [""] * len(target_ids)
            )
            target_addrs = (
                targets_valid["business_address"].fillna("").tolist()
                if "business_address" in targets_valid.columns
                else [""] * len(target_ids)
            )
        else:
            target_ids, target_names, target_addrs = [], [], []

        N_targets = len(target_ids)

        for tid, name, addr in zip(target_ids, target_names, target_addrs):
            if not tid or tid == "nan":
                continue
            name_obj = self.normalizer.normalize_name(name)
            addr_obj = self.normalizer.normalize_address(addr)

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

        # Precompute IDF weights and filter ubiquitous postings (> limit_postings)
        token_weights: Dict[str, float] = {}
        for tok, postings in token_index.items():
            if len(postings) <= limit_postings:
                idf = math.log((N_targets + 1.0) / (len(postings) + 1.0)) + 1.0
                token_weights[tok] = 2.0 * idf

        ngram_weights: Dict[str, float] = {}
        for ng, postings in ngram_index.items():
            if len(postings) <= limit_postings:
                idf = math.log((N_targets + 1.0) / (len(postings) + 1.0)) + 1.0
                ngram_weights[ng] = 0.5 * idf

        # 2. Query S1 records
        candidates_map: Dict[str, List[str]] = {}
        if not s1_df.empty and "entity_id" in s1_df.columns:
            s1_valid = s1_df[s1_df["entity_id"].notna()]
            s1_ids = s1_valid["entity_id"].astype(str).str.strip().tolist()
            s1_names = (
                s1_valid["business_name"].fillna("").tolist()
                if "business_name" in s1_valid.columns
                else [""] * len(s1_ids)
            )
            s1_addrs = (
                s1_valid["business_address"].fillna("").tolist()
                if "business_address" in s1_valid.columns
                else [""] * len(s1_ids)
            )
        else:
            s1_ids, s1_names, s1_addrs = [], [], []

        for s1_id, name, addr in zip(s1_ids, s1_names, s1_addrs):
            if not s1_id or s1_id == "nan":
                continue
            name_obj = self.normalizer.normalize_name(name)
            addr_obj = self.normalizer.normalize_address(addr)

            cand_scores: Dict[str, float] = defaultdict(float)

            # Channel A: Name tokens
            for tok in set(name_obj.tokens):
                if len(tok) >= 3 and tok in token_weights:
                    w = token_weights[tok]
                    for tid in token_index[tok]:
                        cand_scores[tid] += w

            # Channel B: 3-grams
            for ng in set(name_obj.char_3grams):
                if ng and ng in ngram_weights:
                    w = ngram_weights[ng]
                    for tid in ngram_index[ng]:
                        cand_scores[tid] += w

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
    def write_candidate_pairs(
        candidates_map: Dict[str, List[str]],
        output_path: Path | str,
        append: bool = False,
        write_header: bool = True,
    ) -> None:
        """Write candidate pairs TSV matching competition schema.

        Supports streaming append mode across partition chunks.
        """
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with open(out, mode, encoding="utf-8") as f:
            if write_header:
                f.write("source1_entity_id\tcandidate_entity_ids\n")
            for s1_id in sorted(candidates_map.keys()):
                cand_str = ",".join(candidates_map[s1_id])
                f.write(f"{s1_id}\t{cand_str}\n")
