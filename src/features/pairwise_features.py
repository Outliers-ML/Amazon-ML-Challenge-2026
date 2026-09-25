"""Vectorized Rapid Pairwise Feature Extraction for Business Entity Resolution.

Extracts discriminative string similarity and address compatibility metrics using
C-accelerated RapidFuzz and structured token comparison.
"""

import re
from typing import Any, Dict, List, Optional
import numpy as np
import rapidfuzz.distance.JaroWinkler as JaroWinkler
import rapidfuzz.distance.Levenshtein as Levenshtein
import rapidfuzz.fuzz as fuzz

from src.data.normalizer import NormalizedAddress, NormalizedName, TextNormalizer, compute_soundex


class PairwiseFeatureExtractor:
    FEATURE_NAMES = [
        "name_exact_match",
        "name_clean_exact_match",
        "name_jaro_winkler",
        "name_levenshtein_ratio",
        "name_token_sort_ratio",
        "name_token_set_ratio",
        "name_char_3gram_jaccard",
        "name_length_diff",
        "name_length_ratio",
        "name_first_token_match",
        "name_soundex_match",
        "name_soundex_jaccard",
        "name_common_tokens_count",
        "addr_exact_match",
        "addr_token_jaccard",
        "addr_token_sort_ratio",
        "addr_token_set_ratio",
        "addr_street_num_status",
        "addr_postal_code_status",
        "addr_locality_jaccard",
        "addr_numeric_overlap",
        "addr_is_empty",
        "name_in_address_cross",
        "is_source2",
        "is_source3",
        "blocking_rank",
        "blocking_score",
    ]

    def __init__(self) -> None:
        self.normalizer = TextNormalizer()
        self.feature_names = list(self.FEATURE_NAMES)

    def _get_normalized_name(self, row: Dict) -> NormalizedName:
        if "_norm_name" in row and isinstance(row["_norm_name"], NormalizedName):
            return row["_norm_name"]
        return self.normalizer.normalize_name(row.get("business_name", ""))

    def _get_normalized_address(self, row: Dict) -> NormalizedAddress:
        if "_norm_addr" in row and isinstance(row["_norm_addr"], NormalizedAddress):
            return row["_norm_addr"]
        return self.normalizer.normalize_address(row.get("business_address", ""))

    def _compute_features(
        self,
        s1_n: NormalizedName,
        s1_a: NormalizedAddress,
        t_n: NormalizedName,
        t_a: NormalizedAddress,
        target_id: str,
        target_source: str,
        rank: int,
        blocking_score: float,
    ) -> np.ndarray:
        # 1. Name Metrics
        raw1, raw2 = s1_n.raw.lower(), t_n.raw.lower()
        cln1, cln2 = s1_n.clean_name, t_n.clean_name

        name_exact = 1.0 if raw1 == raw2 and raw1 != "" else 0.0
        name_clean_exact = 1.0 if cln1 == cln2 and cln1 != "" else 0.0
        jw = float(JaroWinkler.similarity(cln1, cln2)) if cln1 and cln2 else 0.0
        lev = float(Levenshtein.normalized_similarity(cln1, cln2)) if cln1 and cln2 else 0.0
        tok_sort = float(fuzz.token_sort_ratio(cln1, cln2)) / 100.0 if cln1 and cln2 else 0.0
        tok_set = float(fuzz.token_set_ratio(cln1, cln2)) / 100.0 if cln1 and cln2 else 0.0

        # Char 3-gram Jaccard
        grams1 = set(s1_n.char_3grams)
        grams2 = set(t_n.char_3grams)
        if grams1 and grams2:
            gram_union = len(grams1 | grams2)
            char_3gram_jaccard = float(len(grams1 & grams2) / gram_union) if gram_union > 0 else 0.0
        else:
            char_3gram_jaccard = 0.0

        len1, len2 = len(cln1), len(cln2)
        name_len_diff = float(abs(len1 - len2))
        max_len = max(len1, len2)
        name_len_ratio = float(min(len1, len2) / max_len) if max_len > 0 else 0.0
        first_tok_match = 1.0 if s1_n.tokens and t_n.tokens and s1_n.tokens[0] == t_n.tokens[0] else 0.0

        # Phonetic Soundex & Shared Token Features
        sx1 = compute_soundex(s1_n.tokens[0]) if s1_n.tokens else ""
        sx2 = compute_soundex(t_n.tokens[0]) if t_n.tokens else ""
        name_soundex_match = 1.0 if sx1 and sx2 and sx1 == sx2 else 0.0

        sx_set1 = {compute_soundex(t) for t in s1_n.tokens if t}
        sx_set2 = {compute_soundex(t) for t in t_n.tokens if t}
        if sx_set1 and sx_set2:
            name_soundex_jaccard = float(len(sx_set1 & sx_set2) / len(sx_set1 | sx_set2))
        else:
            name_soundex_jaccard = 0.0

        toks_s1 = set(s1_n.tokens)
        toks_t1 = set(t_n.tokens)
        name_common_tokens_count = float(len(toks_s1 & toks_t1))

        # 2. Address Metrics
        a1, a2 = s1_a.clean_address, t_a.clean_address
        raw_a1, raw_a2 = s1_a.raw.lower(), t_a.raw.lower()
        addr_exact = 1.0 if ((a1 == a2 and a1 != "") or (raw_a1 == raw_a2 and raw_a1 != "")) else 0.0

        # Address token Jaccard
        toks_a1 = set(s1_a.tokens)
        toks_a2 = set(t_a.tokens)
        if toks_a1 and toks_a2:
            tok_union = len(toks_a1 | toks_a2)
            addr_tok_jaccard = float(len(toks_a1 & toks_a2) / tok_union) if tok_union > 0 else 0.0
        else:
            addr_tok_jaccard = 0.0

        addr_tok_sort = float(fuzz.token_sort_ratio(a1, a2)) / 100.0 if a1 and a2 else 0.0
        addr_tok_set = float(fuzz.token_set_ratio(a1, a2)) / 100.0 if a1 and a2 else 0.0

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

        # Locality & Numeric Overlap
        locality1 = {t for t in s1_a.tokens if not t.isdigit() and len(t) >= 3}
        locality2 = {t for t in t_a.tokens if not t.isdigit() and len(t) >= 3}
        if locality1 and locality2:
            addr_locality_jaccard = float(len(locality1 & locality2) / len(locality1 | locality2))
        else:
            addr_locality_jaccard = 0.0

        nums1 = {t for t in s1_a.tokens if any(ch.isdigit() for ch in t)}
        nums2 = {t for t in t_a.tokens if any(ch.isdigit() for ch in t)}
        if nums1 and nums2:
            addr_numeric_overlap = float(len(nums1 & nums2) / len(nums1 | nums2))
        else:
            addr_numeric_overlap = 0.0

        addr_empty = 1.0 if not a1 or not a2 else 0.0

        # 3. Cross & Relational
        cross_match = 0.0
        if len(cln1) >= 3 and (
            re.search(rf"\b{re.escape(cln1)}\b", raw_a2)
            or re.search(rf"\b{re.escape(cln1)}\b", a2)
        ):
            cross_match = 1.0
        elif len(cln2) >= 3 and (
            re.search(rf"\b{re.escape(cln2)}\b", raw_a1)
            or re.search(rf"\b{re.escape(cln2)}\b", a1)
        ):
            cross_match = 1.0

        target_id_upper = target_id.strip().upper()
        target_src_lower = target_source.strip().lower()
        is_s2 = 1.0 if (target_id_upper.startswith("S2") or target_src_lower in ("s2", "source2")) else 0.0
        is_s3 = 1.0 if (target_id_upper.startswith("S3") or target_src_lower in ("s3", "source3")) else 0.0

        return np.array([
            name_exact,
            name_clean_exact,
            jw,
            lev,
            tok_sort,
            tok_set,
            char_3gram_jaccard,
            name_len_diff,
            name_len_ratio,
            first_tok_match,
            name_soundex_match,
            name_soundex_jaccard,
            name_common_tokens_count,
            addr_exact,
            addr_tok_jaccard,
            addr_tok_sort,
            addr_tok_set,
            street_status,
            postal_status,
            addr_locality_jaccard,
            addr_numeric_overlap,
            addr_empty,
            cross_match,
            is_s2,
            is_s3,
            float(rank),
            float(blocking_score),
        ], dtype=np.float32)

    def extract_pair_features(
        self,
        s1_row: Dict,
        target_row: Dict,
        rank: int = 1,
        blocking_score: float = 0.0,
    ) -> np.ndarray:
        s1_n = self._get_normalized_name(s1_row)
        s1_a = self._get_normalized_address(s1_row)
        t_n = self._get_normalized_name(target_row)
        t_a = self._get_normalized_address(target_row)

        target_id = str(target_row.get("entity_id") or "")
        target_src = str(target_row.get("source") or "")

        return self._compute_features(
            s1_n, s1_a, t_n, t_a, target_id, target_src, rank, blocking_score
        )

    def extract_pairs_matrix(
        self,
        s1_rows: List[Dict],
        target_rows: List[Dict],
        ranks: Optional[List[int]] = None,
        scores: Optional[List[float]] = None,
    ) -> np.ndarray:
        n = len(s1_rows)
        if len(target_rows) != n:
            raise ValueError(
                f"target_rows length ({len(target_rows)}) must match s1_rows length ({n})"
            )
        if ranks is not None and len(ranks) != n:
            raise ValueError(
                f"ranks length ({len(ranks)}) must match s1_rows length ({n})"
            )
        if scores is not None and len(scores) != n:
            raise ValueError(
                f"scores length ({len(scores)}) must match s1_rows length ({n})"
            )

        if n == 0:
            return np.empty((0, len(self.feature_names)), dtype=np.float32)

        ranks_list = ranks if ranks is not None else [1] * n
        scores_list = scores if scores is not None else [0.0] * n

        matrix = np.empty((n, len(self.feature_names)), dtype=np.float32)

        # Caching normalized objects for repeated queries / candidates
        name_cache: Dict[Any, NormalizedName] = {}
        addr_cache: Dict[Any, NormalizedAddress] = {}

        def get_cached_name(row: Dict) -> NormalizedName:
            key = row.get("entity_id") or id(row)
            if key not in name_cache:
                name_cache[key] = self._get_normalized_name(row)
            return name_cache[key]

        def get_cached_addr(row: Dict) -> NormalizedAddress:
            key = row.get("entity_id") or id(row)
            if key not in addr_cache:
                addr_cache[key] = self._get_normalized_address(row)
            return addr_cache[key]

        for i in range(n):
            s1 = s1_rows[i]
            t = target_rows[i]
            r = ranks_list[i]
            s = scores_list[i]

            s1_n = get_cached_name(s1)
            s1_a = get_cached_addr(s1)
            t_n = get_cached_name(t)
            t_a = get_cached_addr(t)

            target_id = str(t.get("entity_id") or "")
            target_src = str(t.get("source") or "")

            matrix[i] = self._compute_features(
                s1_n, s1_a, t_n, t_a, target_id, target_src, r, s
            )

        return matrix
