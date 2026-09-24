"""Vectorized Rapid Pairwise Feature Extraction for Business Entity Resolution.

Extracts discriminative string similarity and address compatibility metrics using
C-accelerated RapidFuzz and structured token comparison.
"""

from typing import Dict, List
import numpy as np
import rapidfuzz.distance.JaroWinkler as JaroWinkler
import rapidfuzz.distance.Levenshtein as Levenshtein
import rapidfuzz.fuzz as fuzz

from src.data.normalizer import TextNormalizer


class PairwiseFeatureExtractor:
    def __init__(self) -> None:
        self.normalizer = TextNormalizer()
        self.feature_names = [
            "name_exact_match",
            "name_clean_exact_match",
            "name_jaro_winkler",
            "name_levenshtein_ratio",
            "name_token_sort_ratio",
            "name_token_set_ratio",
            "name_len_ratio",
            "name_first_token_match",
            "addr_exact_match",
            "addr_token_sort_ratio",
            "addr_token_set_ratio",
            "addr_street_num_status",
            "addr_postal_code_status",
            "addr_is_empty",
            "name_in_addr_cross",
            "is_source2",
            "blocking_rank",
        ]

    def extract_pair_features(self, s1_row: Dict, target_row: Dict, rank: int = 1) -> np.ndarray:
        s1_n = self.normalizer.normalize_name(s1_row.get("business_name", ""))
        s1_a = self.normalizer.normalize_address(s1_row.get("business_address", ""))
        t_n = self.normalizer.normalize_name(target_row.get("business_name", ""))
        t_a = self.normalizer.normalize_address(target_row.get("business_address", ""))

        # 1. Name Metrics
        raw1, raw2 = s1_n.raw.lower(), t_n.raw.lower()
        cln1, cln2 = s1_n.clean_name, t_n.clean_name

        name_exact = 1.0 if raw1 == raw2 and raw1 != "" else 0.0
        name_clean_exact = 1.0 if cln1 == cln2 and cln1 != "" else 0.0
        jw = float(JaroWinkler.similarity(cln1, cln2)) if cln1 and cln2 else 0.0
        lev = float(Levenshtein.normalized_similarity(cln1, cln2)) if cln1 and cln2 else 0.0
        tok_sort = float(fuzz.token_sort_ratio(cln1, cln2)) / 100.0 if cln1 and cln2 else 0.0
        tok_set = float(fuzz.token_set_ratio(cln1, cln2)) / 100.0 if cln1 and cln2 else 0.0
        len_ratio = float(min(len(cln1), len(cln2)) / (max(len(cln1), len(cln2)) + 1e-5))
        first_tok_match = 1.0 if s1_n.tokens and t_n.tokens and s1_n.tokens[0] == t_n.tokens[0] else 0.0

        # 2. Address Metrics
        a1, a2 = s1_a.clean_address, t_a.clean_address
        addr_exact = 1.0 if a1 == a2 and a1 != "" else 0.0
        addr_tok_sort = float(fuzz.token_sort_ratio(a1, a2)) / 100.0 if a1 and a2 else 0.0
        addr_tok_set = float(fuzz.token_set_ratio(a1, a2)) / 100.0 if a1 and a2 else 0.0
        addr_empty = 1.0 if not a1 or not a2 else 0.0

        # Street number comparison (+1 = match, -1 = conflict, 0 = unknown)
        if s1_a.street_number and t_a.street_number:
            street_status = 1.0 if s1_a.street_number == t_a.street_number else -1.0
        else:
            street_status = 0.0

        # Postal code comparison (+1 = match, -1 = conflict, 0 = unknown)
        if s1_a.postal_code and t_a.postal_code:
            postal_status = 1.0 if s1_a.postal_code == t_a.postal_code else -1.0
        else:
            postal_status = 0.0

        # 3. Cross & Relational
        cross_match = 1.0 if (cln1 and cln1 in a2) or (cln2 and cln2 in a1) else 0.0
        is_s2 = 1.0 if str(target_row.get("entity_id") or "").startswith("S2-") else 0.0

        return np.array([
            name_exact,
            name_clean_exact,
            jw,
            lev,
            tok_sort,
            tok_set,
            len_ratio,
            first_tok_match,
            addr_exact,
            addr_tok_sort,
            addr_tok_set,
            street_status,
            postal_status,
            addr_empty,
            cross_match,
            is_s2,
            float(rank),
        ], dtype=np.float32)
