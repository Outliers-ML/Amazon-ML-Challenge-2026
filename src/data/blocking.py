"""Multi-Key and Multi-Tier Blocking Engine for Candidate Generation.

Provides high-recall candidate generation across three complementary tiers:
1. Tier 1: Deterministic and phonetic inverted hashing with bucket size ceiling (N_max <= 250).
2. Tier 2: Sparse token BM25 / character 3-gram indexing via TF-IDF vectorization (top-20).
3. Tier 3: Dense semantic vector cosine retrieval using PyTorch GPU/CPU matrix multiplication (top-20).

Candidates across active tiers are combined using Reciprocal Rank Fusion (RRF) and capped
at Top-35 per Source 1 entity, writing output compliant with candidate_pairs.tsv schema.
"""

from collections import defaultdict
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    import torch
except ImportError:
    torch = None

from src.data.normalizer import TextNormalizer, compute_double_metaphone


def reciprocal_rank_fusion(
    tier_candidates: Dict[str, List[Tuple[str, float]]],
    k: int = 60,
    weights: Optional[Dict[str, float]] = None,
) -> List[Tuple[str, float]]:
    """Fuse multi-tier candidate rankings scale-invariantly using Reciprocal Rank Fusion (RRF).

    Formula:
        Score_RRF(cand) = sum_{tier} (weight_tier / (k + rank_tier(cand)))

    Args:
        tier_candidates: Mapping from tier name to list of (cand_id, score) pairs.
        k: Smoothing constant (default 60).
        weights: Optional tier weights dict (default 1.0 for each tier).

    Returns:
        List of deduplicated (cand_id, score) pairs sorted descending by RRF score.
    """
    scores: Dict[str, float] = defaultdict(float)

    for tier_name, cands in tier_candidates.items():
        if not cands:
            continue
        weight = weights.get(tier_name, 1.0) if weights else 1.0

        # Sort descending by tier score stably
        sorted_cands = sorted(cands, key=lambda x: -x[1])
        # Deduplicate per tier, retaining the best rank
        seen = set()
        deduped = []
        for cand_id, score in sorted_cands:
            if cand_id not in seen:
                seen.add(cand_id)
                deduped.append((cand_id, score))

        for rank, (cand_id, _) in enumerate(deduped, start=1):
            scores[cand_id] += weight / (k + rank)

    # Sort descending by fused score, tie-breaking deterministically by cand_id
    fused = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    return fused


def write_candidate_pairs(
    candidate_map: Dict[str, List[str]],
    output_path: Union[Path, str],
    append: bool = False,
    write_header: bool = True,
) -> None:
    """Write candidate pairs TSV matching competition schema.

    Schema: source1_entity_id\\tcandidate_entity_ids (comma-separated).
    Supports streaming append mode across partition chunks.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with open(out, mode, encoding="utf-8") as f:
        if write_header:
            f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in sorted(candidate_map.keys()):
            cand_str = ",".join(candidate_map[s1_id])
            f.write(f"{s1_id}\t{cand_str}\n")


class MultiTierBlocker:
    """Multi-tier candidate generation blocker with RRF and bucket ceilings."""

    def __init__(
        self,
        max_candidates: int = 35,
        bucket_ceiling: int = 250,
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.max_candidates = max_candidates
        self.bucket_ceiling = bucket_ceiling
        self.weights = weights or {
            "tier1": 1.5,
            "tier2": 1.0,
            "tier3": 1.0,
            "t1": 1.5,
            "t2": 1.0,
            "t3": 1.0,
        }
        self.normalizer = TextNormalizer()

    def _prepare_records(
        self,
        records: Union[pd.DataFrame, List[Dict[str, Any]]],
        country: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Normalize records into a consistent structured representation."""
        if isinstance(records, pd.DataFrame):
            if records.empty:
                return []
            rec_list = records.to_dict(orient="records")
        elif isinstance(records, list):
            rec_list = records
        else:
            return []

        prepared: List[Dict[str, Any]] = []
        filter_country = country.strip().upper() if country else None

        for rec in rec_list:
            raw_id = rec.get("entity_id")
            if raw_id is None or str(raw_id).strip().lower() in ("nan", "none", "null", ""):
                continue
            entity_id = str(raw_id).strip()

            # Country handling
            rec_country = rec.get("country")
            if rec_country is not None and str(rec_country).strip().lower() not in ("nan", "none", "null", ""):
                rec_country_str = str(rec_country).strip().upper()
            else:
                rec_country_str = filter_country or ""

            if filter_country and rec_country_str and rec_country_str != filter_country:
                continue

            # Name normalization
            clean_name = rec.get("clean_name_stripped")
            metaphone_primary = rec.get("metaphone_primary")
            if clean_name is None:
                raw_name = rec.get("business_name") or rec.get("name") or ""
                norm_name = self.normalizer.normalize_name(raw_name)
                clean_name = norm_name.clean_name_stripped
                metaphone_primary = norm_name.metaphone_primary
            else:
                clean_name = str(clean_name).strip()
                if not metaphone_primary:
                    tokens = clean_name.split()
                    if tokens:
                        metaphone_primary, _ = compute_double_metaphone(tokens[0])
                    else:
                        metaphone_primary = ""
                else:
                    metaphone_primary = str(metaphone_primary).strip()

            # Address normalization
            clean_addr = rec.get("clean_address")
            street_num = rec.get("street_num") or rec.get("street_number")
            postal_code = rec.get("postal_code")
            if clean_addr is None or street_num is None or postal_code is None:
                raw_addr = rec.get("business_address") or rec.get("address") or ""
                norm_addr = self.normalizer.normalize_address(raw_addr if clean_addr is None else clean_addr)
                if clean_addr is None:
                    clean_addr = norm_addr.clean_address
                else:
                    clean_addr = str(clean_addr).strip()
                if street_num is None:
                    street_num = norm_addr.street_number
                if postal_code is None:
                    postal_code = norm_addr.postal_code
            else:
                clean_addr = str(clean_addr).strip()

            street_num_str = str(street_num).strip() if street_num is not None else None
            if street_num_str and street_num_str.lower() in ("nan", "none", "null", ""):
                street_num_str = None

            postal_code_str = str(postal_code).strip() if postal_code is not None else None
            if postal_code_str and postal_code_str.lower() in ("nan", "none", "null", ""):
                postal_code_str = None

            prepared.append({
                "entity_id": entity_id,
                "country": rec_country_str,
                "clean_name_stripped": clean_name,
                "clean_address": clean_addr,
                "street_num": street_num_str,
                "postal_code": postal_code_str,
                "metaphone_primary": metaphone_primary or "",
            })

        return prepared

    def get_record_bucket_keys(self, record: Dict[str, Any]) -> List[Tuple]:
        """Generate Tier 1 deterministic and phonetic bucket keys for a record."""
        country = str(record.get("country", "")).strip().upper()
        clean_name = str(record.get("clean_name_stripped", "")).strip()
        street_num = record.get("street_num") or record.get("street_number")
        postal_code = record.get("postal_code")
        metaphone_primary = record.get("metaphone_primary")

        if street_num is not None:
            street_num = str(street_num).strip()
            if street_num.lower() in ("nan", "none", "null", ""):
                street_num = None
        if postal_code is not None:
            postal_code = str(postal_code).strip()
            if postal_code.lower() in ("nan", "none", "null", ""):
                postal_code = None
        if metaphone_primary is not None:
            metaphone_primary = str(metaphone_primary).strip()
            if metaphone_primary.lower() in ("nan", "none", "null", ""):
                metaphone_primary = None

        keys: List[Tuple] = []

        # 1. (country, clean_name_stripped)
        if clean_name:
            keys.append((country, clean_name))

        # 2. (country, clean_name_stripped[:4], street_num) (when street_num exists)
        if clean_name and street_num:
            keys.append((country, clean_name[:4], street_num))

        # 3. (country, metaphone_primary, postal_code) (when postal_code exists)
        if metaphone_primary and postal_code:
            keys.append((country, metaphone_primary, postal_code))

        # 4. (country, postal_code, street_num) (when both exist)
        if postal_code and street_num:
            keys.append((country, postal_code, street_num))

        return keys

    def build_deterministic_buckets(
        self,
        records: Union[pd.DataFrame, List[Dict[str, Any]]],
        bucket_ceiling: Optional[int] = None,
    ) -> Dict[Tuple, List[str]]:
        """Build deterministic bucket indexes and discard clusters exceeding the ceiling."""
        ceiling = bucket_ceiling if bucket_ceiling is not None else self.bucket_ceiling
        prepared = self._prepare_records(records)
        raw_buckets: Dict[Tuple, set] = defaultdict(set)

        for rec in prepared:
            eid = rec["entity_id"]
            keys = self.get_record_bucket_keys(rec)
            for k in keys:
                raw_buckets[k].add(eid)

        # Discard any bucket exceeding ceiling
        filtered_buckets = {
            k: sorted(members)
            for k, members in raw_buckets.items()
            if len(members) <= ceiling
        }
        return filtered_buckets

    def retrieve_tier1_candidates(
        self,
        s1_prepared: List[Dict[str, Any]],
        target_buckets: Dict[Tuple, List[str]],
        top_k: Optional[int] = None,
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Retrieve Tier 1 candidates using bucket hash lookup with quality weighting."""
        k_limit = top_k if top_k is not None else self.max_candidates
        tier1_map: Dict[str, List[Tuple[str, float]]] = {}
        for s1_rec in s1_prepared:
            s1_id = s1_rec["entity_id"]
            scores: Dict[str, float] = defaultdict(float)
            keys = self.get_record_bucket_keys(s1_rec)
            for k in keys:
                if k in target_buckets:
                    # Exact clean name key has 2 elements: (country, clean_name_stripped)
                    # Give higher weight (3.0) to exact clean name match over coarse address/postal keys (1.0)
                    weight = 3.0 if len(k) == 2 else 1.0
                    for target_id in target_buckets[k]:
                        scores[target_id] += weight

            if scores:
                sorted_cands = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
                tier1_map[s1_id] = sorted_cands[:k_limit]
            else:
                tier1_map[s1_id] = []
        return tier1_map

    def retrieve_tier2_candidates(
        self,
        s1_prepared: List[Dict[str, Any]],
        target_prepared: List[Dict[str, Any]],
        top_k: int = 20,
        chunk_size: int = 2000,
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Retrieve Tier 2 candidates using character 3-gram TF-IDF similarity with chunking."""
        s1_ids = [r["entity_id"] for r in s1_prepared]
        if not s1_prepared or not target_prepared:
            return {eid: [] for eid in s1_ids}

        target_texts = [
            f"{r['clean_name_stripped']} {r['clean_address']}".strip()
            for r in target_prepared
        ]
        target_ids = [r["entity_id"] for r in target_prepared]

        s1_texts = [
            f"{r['clean_name_stripped']} {r['clean_address']}".strip()
            for r in s1_prepared
        ]

        min_df = 2 if len(target_texts) >= 2 else 1
        vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 3),
            sublinear_tf=True,
            min_df=min_df,
        )

        try:
            target_tfidf = vectorizer.fit_transform(target_texts)
            s1_tfidf = vectorizer.transform(s1_texts)
        except ValueError:
            if min_df > 1:
                try:
                    vectorizer = TfidfVectorizer(
                        analyzer="char",
                        ngram_range=(3, 3),
                        sublinear_tf=True,
                        min_df=1,
                    )
                    target_tfidf = vectorizer.fit_transform(target_texts)
                    s1_tfidf = vectorizer.transform(s1_texts)
                except ValueError:
                    return {eid: [] for eid in s1_ids}
            else:
                return {eid: [] for eid in s1_ids}

        n_s1 = s1_tfidf.shape[0]
        target_tfidf_t = target_tfidf.T.tocsc()
        tier2_map: Dict[str, List[Tuple[str, float]]] = {}

        for chunk_start in range(0, n_s1, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_s1)
            s1_chunk = s1_tfidf[chunk_start:chunk_end]
            sim_chunk = s1_chunk.dot(target_tfidf_t).tocsr()

            for i in range(chunk_end - chunk_start):
                s1_id = s1_ids[chunk_start + i]
                start = sim_chunk.indptr[i]
                end = sim_chunk.indptr[i + 1]

                if start == end:
                    tier2_map[s1_id] = []
                    continue

                row_indices = sim_chunk.indices[start:end]
                row_data = sim_chunk.data[start:end]

                valid_mask = row_data > 0
                if not np.any(valid_mask):
                    tier2_map[s1_id] = []
                    continue

                valid_indices = row_indices[valid_mask]
                valid_data = row_data[valid_mask]

                if len(valid_data) <= top_k:
                    candidates_unsorted = [
                        (target_ids[valid_indices[idx]], float(valid_data[idx]))
                        for idx in range(len(valid_data))
                    ]
                else:
                    part = np.argpartition(-valid_data, top_k)[:top_k]
                    candidates_unsorted = [
                        (target_ids[valid_indices[idx]], float(valid_data[idx]))
                        for idx in part
                    ]

                tier2_map[s1_id] = sorted(
                    candidates_unsorted, key=lambda x: (-x[1], x[0])
                )

        return tier2_map

    def retrieve_dense_candidates(
        self,
        s1_embeddings: Any,
        target_embeddings: Any,
        s1_ids: List[str],
        target_ids: List[str],
        top_k: int = 20,
        chunk_size: int = 2048,
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Retrieve Tier 3 candidates using dense cosine similarity with chunking."""
        if len(s1_ids) == 0 or len(target_ids) == 0:
            return {eid: [] for eid in s1_ids}

        k = min(top_k, len(target_ids))

        if torch is not None:
            if not torch.is_tensor(s1_embeddings):
                s1_t = torch.as_tensor(s1_embeddings, dtype=torch.float32)
            else:
                s1_t = s1_embeddings.float()

            if not torch.is_tensor(target_embeddings):
                target_t = torch.as_tensor(target_embeddings, dtype=torch.float32)
            else:
                target_t = target_embeddings.float()

            # Ensure 2D tensor shapes
            if s1_t.ndim == 1:
                s1_t = s1_t.unsqueeze(0)
            if target_t.ndim == 1:
                target_t = target_t.unsqueeze(0)

            if torch.cuda.is_available() and (s1_t.is_cuda or target_t.is_cuda):
                device = torch.device("cuda")
                s1_t = s1_t.to(device)
                target_t = target_t.to(device)

            s1_norm = torch.nn.functional.normalize(s1_t, p=2, dim=1)
            target_norm = torch.nn.functional.normalize(target_t, p=2, dim=1)
            target_norm_t = target_norm.T

            n_s1 = s1_norm.shape[0]
            dense_map: Dict[str, List[Tuple[str, float]]] = {}

            for chunk_start in range(0, n_s1, chunk_size):
                chunk_end = min(chunk_start + chunk_size, n_s1)
                s1_chunk = s1_norm[chunk_start:chunk_end]

                sim_chunk = torch.matmul(s1_chunk, target_norm_t)
                top_scores, top_indices = torch.topk(sim_chunk, k=k, dim=1)

                top_scores_np = top_scores.cpu().numpy()
                top_indices_np = top_indices.cpu().numpy()

                for i in range(chunk_end - chunk_start):
                    s1_id = s1_ids[chunk_start + i]
                    dense_map[s1_id] = [
                        (target_ids[top_indices_np[i, j]], float(top_scores_np[i, j]))
                        for j in range(k)
                    ]
            return dense_map
        else:
            s1_arr = np.asarray(s1_embeddings, dtype=np.float32)
            target_arr = np.asarray(target_embeddings, dtype=np.float32)

            # Ensure 2D array shapes
            if s1_arr.ndim == 1:
                s1_arr = np.expand_dims(s1_arr, axis=0)
            if target_arr.ndim == 1:
                target_arr = np.expand_dims(target_arr, axis=0)

            s1_norm = s1_arr / np.maximum(np.linalg.norm(s1_arr, axis=1, keepdims=True), 1e-12)
            target_norm = target_arr / np.maximum(np.linalg.norm(target_arr, axis=1, keepdims=True), 1e-12)
            target_norm_t = target_norm.T

            n_s1 = s1_norm.shape[0]
            dense_map = {}

            for chunk_start in range(0, n_s1, chunk_size):
                chunk_end = min(chunk_start + chunk_size, n_s1)
                s1_chunk = s1_norm[chunk_start:chunk_end]
                sim_chunk = np.dot(s1_chunk, target_norm_t)

                for i in range(chunk_end - chunk_start):
                    s1_id = s1_ids[chunk_start + i]
                    row = sim_chunk[i]
                    if len(row) <= k:
                        sort_order = np.argsort(-row)
                    else:
                        part = np.argpartition(-row, k)[:k]
                        sort_order = part[np.argsort(-row[part])]
                    dense_map[s1_id] = [
                        (target_ids[idx], float(row[idx]))
                        for idx in sort_order[:k]
                    ]
            return dense_map

    def generate_candidate_pairs(
        self,
        s1_records: Union[pd.DataFrame, List[Dict[str, Any]]],
        target_records: Union[pd.DataFrame, List[Dict[str, Any]]],
        country: Optional[str] = None,
        max_candidates: Optional[int] = None,
        s1_embeddings: Optional[Any] = None,
        target_embeddings: Optional[Any] = None,
    ) -> Dict[str, List[str]]:
        """Generate high-recall candidate pairs fused via Reciprocal Rank Fusion."""
        limit_candidates = max_candidates if max_candidates is not None else self.max_candidates

        s1_prepared = self._prepare_records(s1_records, country=country)
        target_prepared = self._prepare_records(target_records, country=country)

        if not s1_prepared:
            return {}
        if not target_prepared:
            return {r["entity_id"]: [] for r in s1_prepared}

        # Tier 1: Deterministic & Phonetic Bucketing with Bucket Ceiling and Top-K Capping
        target_buckets = self.build_deterministic_buckets(target_prepared)
        t1_candidates = self.retrieve_tier1_candidates(s1_prepared, target_buckets, top_k=limit_candidates)

        # Tier 2: Sparse BM25 / Char 3-Grams (top-20)
        t2_candidates = self.retrieve_tier2_candidates(s1_prepared, target_prepared, top_k=20)

        # Tier 3: Dense Semantic Retrieval (top-20) if embeddings provided
        t3_candidates: Dict[str, List[Tuple[str, float]]] = {}
        if s1_embeddings is not None and target_embeddings is not None:
            s1_ids = [r["entity_id"] for r in s1_prepared]
            target_ids = [r["entity_id"] for r in target_prepared]
            t3_candidates = self.retrieve_dense_candidates(
                s1_embeddings=s1_embeddings,
                target_embeddings=target_embeddings,
                s1_ids=s1_ids,
                target_ids=target_ids,
                top_k=20,
            )

        # Fuse with RRF and cap at limit_candidates
        candidates_map: Dict[str, List[str]] = {}
        for s1_rec in s1_prepared:
            s1_id = s1_rec["entity_id"]
            tier_dict: Dict[str, List[Tuple[str, float]]] = {}
            if s1_id in t1_candidates and t1_candidates[s1_id]:
                tier_dict["tier1"] = t1_candidates[s1_id]
            if s1_id in t2_candidates and t2_candidates[s1_id]:
                tier_dict["tier2"] = t2_candidates[s1_id]
            if s1_id in t3_candidates and t3_candidates[s1_id]:
                tier_dict["tier3"] = t3_candidates[s1_id]

            if not tier_dict:
                candidates_map[s1_id] = []
            else:
                fused = reciprocal_rank_fusion(tier_dict, k=60, weights=self.weights)
                candidates_map[s1_id] = [cand_id for cand_id, _ in fused[:limit_candidates]]

        return candidates_map

    def block_country_partition(
        self,
        s1_df: pd.DataFrame,
        s2_df: pd.DataFrame,
        s3_df: pd.DataFrame,
        max_candidates: Optional[int] = None,
    ) -> Dict[str, List[str]]:
        """Compatible entrypoint for country partitions."""
        dfs = [df for df in [s2_df, s3_df] if not df.empty]
        targets = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"])
        return self.generate_candidate_pairs(
            s1_records=s1_df,
            target_records=targets,
            max_candidates=max_candidates,
        )

    write_candidate_pairs = staticmethod(write_candidate_pairs)


def generate_candidate_pairs(
    s1_records: Union[pd.DataFrame, List[Dict[str, Any]]],
    target_records: Union[pd.DataFrame, List[Dict[str, Any]]],
    country: Optional[str] = None,
    max_candidates: int = 35,
    s1_embeddings: Optional[Any] = None,
    target_embeddings: Optional[Any] = None,
    bucket_ceiling: int = 250,
) -> Dict[str, List[str]]:
    """Module-level candidate pair generator."""
    blocker = MultiTierBlocker(max_candidates=max_candidates, bucket_ceiling=bucket_ceiling)
    return blocker.generate_candidate_pairs(
        s1_records=s1_records,
        target_records=target_records,
        country=country,
        max_candidates=max_candidates,
        s1_embeddings=s1_embeddings,
        target_embeddings=target_embeddings,
    )


class MultiKeyBlocker:
    """Multi-key inverted index blocker for cross-source candidate retrieval (Backwards Compatible)."""

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

    write_candidate_pairs = staticmethod(write_candidate_pairs)
