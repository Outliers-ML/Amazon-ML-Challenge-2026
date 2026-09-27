"""Post-bipartite disambiguation and singleton protection engine.

Partitions candidates per country, performs greedy maximum-weight bipartite
disambiguation, and applies post-pruning singleton guard to optimize Macro F0.5.
"""

from collections import defaultdict
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union
import numpy as np
import pandas as pd


def _is_valid_entity_id(val: Any) -> bool:
    """Check if an entity ID is non-empty, non-null, and non-NaN."""
    if val is None:
        return False
    if isinstance(val, (float, np.floating)) and math.isnan(val):
        return False
    try:
        if pd.isna(val):
            return False
    except Exception:
        pass
    s = str(val).strip()
    if not s or s.lower() in ("none", "nan", "null"):
        return False
    return True


def _get_field(d: Dict[str, Any], *candidates: str, default: Any = None) -> Any:
    """Retrieve first existing field from dictionary with fallback keys."""
    for c in candidates:
        if c in d:
            return d[c]
    return default


def _find_column(df: pd.DataFrame, *candidates: str) -> Optional[str]:
    """Find matching column name case-insensitively from candidates."""
    col_map = {col.lower(): col for col in df.columns}
    for c in candidates:
        if c.lower() in col_map:
            return col_map[c.lower()]
    return None


def disambiguate_and_guard(
    scored_pairs: Union[List[Dict[str, Any]], pd.DataFrame, Iterable[Dict[str, Any]]],
    all_s1_ids: Optional[Sequence[str]] = None,
    tau_singleton: float = 0.74,
    tau_secondary: float = 0.60,
    max_matches: Optional[int] = 6,
    enforce_prefix: bool = True,
) -> Dict[str, List[str]]:
    """Resolve candidate pairs using bipartite matching and post-pruning singleton guard.

    1. Per-Country Subgraph Partitioning:
       Partitions candidate pairs by country (US, India, France) to ensure zero cross-country
       matching and avoid unnecessary memory overhead.
    2. Greedy Maximum-Weight Bipartite Matching:
       - Filters candidate pairs with P_final >= tau_secondary (default: 0.60).
       - Excludes rogue/unknown S1 IDs if all_s1_ids is provided.
       - Sanitizes None/NaN values and rejects self-matches (S1-*) and invalid candidate prefixes.
       - Sorts candidate pairs globally DESC by P_final (with deterministic tie-breaker on IDs).
       - Tracks claimed_candidates so a candidate record from Source 2 or Source 3 is assigned
         to at most ONE Source 1 entity.
       - Greedily assigns candidate to its highest-scoring Source 1 parent and records the edge.
    3. Post-Pruning Singleton Guard:
       - For every S1 in all_s1_ids:
         - If no candidates were assigned -> output is [].
         - If assigned candidates exist: check if max(p_final for assigned candidates) >= tau_singleton.
         - If YES: emit the assigned candidates (ordered by score DESC).
         - If NO: suppress the entire match list for S1 to [] (preserving the 1.0 singleton credit).

    Parameters
    ----------
    scored_pairs : Union[List[Dict[str, Any]], pd.DataFrame, Iterable[Dict[str, Any]]]
        Scored candidate pairs containing source1 ID, candidate ID, probability score,
        and country code.
    all_s1_ids : Optional[Sequence[str]]
        Complete list of Source 1 entity IDs to include in the output. If provided,
        every entity in all_s1_ids will appear in the output mapping (empty list for singletons),
        and any candidate pairs for S1 entities not in this list will be ignored.
    tau_singleton : float, default 0.74
        Gatekeeper threshold. The top surviving assigned candidate for an S1 entity must
        have p_final >= tau_singleton, otherwise all candidates for that S1 entity are suppressed
        to [] to protect the 1.0 singleton credit under Macro F0.5.
    tau_secondary : float, default 0.60
        Secondary candidate admission threshold. Only pairs with p_final >= tau_secondary
        enter bipartite resolution.
    enforce_prefix : bool, default True
        If True, candidate IDs must begin with 'S2-' or 'S3-', and candidates starting with
        'S1-' (self-matches) are rejected.

    Returns
    -------
    Dict[str, List[str]]
        Mapping of source1_entity_id -> list of matched candidate IDs (ordered DESC by score).
    """
    # 0. Setup target S1 results mapping
    if all_s1_ids is not None:
        valid_s1_set = set(str(s).strip() for s in all_s1_ids)
        results: Dict[str, List[str]] = {str(s1).strip(): [] for s1 in all_s1_ids}
    else:
        valid_s1_set = None
        results = {}

    partitions: Dict[str, List[Tuple[float, str, str]]] = defaultdict(list)

    # 1. Extraction and filtering (Vectorized for DataFrame, iterator-based for dicts)
    if isinstance(scored_pairs, pd.DataFrame):
        df = scored_pairs
        s1_col = _find_column(df, "source1_id", "source1_entity_id", "s1_id", "s1")
        cand_col = _find_column(df, "candidate_id", "candidate_entity_id", "cand_id", "candidate", "s2_s3_id")
        score_col = _find_column(df, "p_final", "score", "prob", "probability", "prediction")
        country_col = _find_column(df, "country", "country_code")

        if s1_col and cand_col and score_col:
            # Score filter
            scores = pd.to_numeric(df[score_col], errors="coerce").fillna(0.0)
            score_mask = scores >= tau_secondary
            df_filtered = df[score_mask]

            if not df_filtered.empty:
                df_filtered = df_filtered.dropna(subset=[s1_col, cand_col])
                s1_series = df_filtered[s1_col].astype(str).str.strip()
                cand_series = df_filtered[cand_col].astype(str).str.strip()

                # S1 validation & rogue S1 exclusion
                valid_s1_mask = ~s1_series.str.lower().isin(["none", "nan", "null", ""])
                if valid_s1_set is not None:
                    valid_s1_mask = valid_s1_mask & s1_series.isin(valid_s1_set)

                # Candidate validation (no self-matches, optional S2/S3 prefix check)
                valid_cand_mask = (
                    ~cand_series.str.lower().isin(["none", "nan", "null", ""])
                ) & (~cand_series.str.startswith("S1-"))
                if enforce_prefix:
                    valid_cand_mask = valid_cand_mask & cand_series.str.startswith(("S2-", "S3-"))

                valid_all = valid_s1_mask & valid_cand_mask
                df_valid = df_filtered[valid_all]

                if not df_valid.empty:
                    s1_arr = df_valid[s1_col].astype(str).str.strip().to_numpy()
                    cand_arr = df_valid[cand_col].astype(str).str.strip().to_numpy()
                    score_arr = pd.to_numeric(df_valid[score_col], errors="coerce").fillna(0.0).to_numpy()

                    if country_col:
                        c_series = (
                            df_valid[country_col]
                            .fillna("")
                            .astype(str)
                            .str.strip()
                            .str.upper()
                            .replace({"NAN": "", "NONE": "", "NULL": ""})
                        )
                        country_arr = c_series.to_numpy()
                    else:
                        country_arr = np.array([""] * len(df_valid))

                    for p_final, s1, cand, country in zip(score_arr, s1_arr, cand_arr, country_arr):
                        partitions[country].append((float(p_final), s1, cand))
    else:
        # Non-DataFrame iterable / List[Dict]
        for pair in scored_pairs:
            raw_score = _get_field(pair, "p_final", "score", "prob", "probability", "prediction", default=0.0)
            try:
                p_final = float(raw_score) if raw_score is not None else 0.0
            except (ValueError, TypeError):
                continue
            if p_final < tau_secondary:
                continue

            raw_s1 = _get_field(pair, "source1_id", "source1_entity_id", "s1_id", "s1")
            raw_cand = _get_field(pair, "candidate_id", "candidate_entity_id", "cand_id", "candidate", "s2_s3_id")

            if not _is_valid_entity_id(raw_s1) or not _is_valid_entity_id(raw_cand):
                continue

            s1 = str(raw_s1).strip()
            cand = str(raw_cand).strip()

            # Ignore rogue S1 if all_s1_ids is provided
            if valid_s1_set is not None and s1 not in valid_s1_set:
                continue

            # Self-matching & prefix protection
            if cand.startswith("S1-"):
                continue
            if enforce_prefix and not cand.startswith(("S2-", "S3-")):
                continue

            raw_country = _get_field(pair, "country", "country_code", default="")
            if (
                raw_country is None
                or (isinstance(raw_country, (float, np.floating)) and math.isnan(raw_country))
                or str(raw_country).lower() in ("nan", "none", "null")
            ):
                country_norm = ""
            else:
                country_norm = str(raw_country).strip().upper()

            partitions[country_norm].append((p_final, s1, cand))

    # 2. Greedy Maximum-Weight Bipartite Matching per country partition
    assigned_candidates: Dict[str, List[Tuple[str, float]]] = defaultdict(list)

    for country, country_tuples in partitions.items():
        # Sort candidate pairs globally DESC by P_final with deterministic tie-breaker on IDs
        country_tuples.sort(key=lambda item: (-item[0], item[1], item[2]))

        # Track claimed candidates so candidate record is assigned to at most ONE Source 1 entity
        claimed_candidates: Set[str] = set()
        for p_final, s1, cand in country_tuples:
            if cand not in claimed_candidates:
                claimed_candidates.add(cand)
                assigned_candidates[s1].append((cand, p_final))

    # 3. Post-Pruning Singleton Guard
    if all_s1_ids is None:
        for s1 in sorted(list(assigned_candidates.keys())):
            results[s1] = []

    for s1 in list(results.keys()):
        cand_list = assigned_candidates.get(s1, [])
        if not cand_list:
            results[s1] = []
        else:
            # Sort assigned candidates by score DESC with deterministic tie-breaker on candidate_id
            cand_list.sort(key=lambda item: (-item[1], item[0]))
            top_score = cand_list[0][1]
            if top_score >= tau_singleton:
                results[s1] = [cand for cand, _ in cand_list[:max_matches]]
            else:
                # Suppress entire match list to [] (preserving singleton credit)
                results[s1] = []

    return results


def write_matching_results(
    matching_map: Dict[str, List[str]],
    output_path: Union[str, Path],
    append: bool = False,
    write_header: bool = True,
) -> None:
    """Write matching results TSV matching competition submission schema.

    Schema: source1_entity_id\\tmatched_entity_ids (strictly tab-delimited, comma-separated candidate IDs,
    empty for singletons, no quoting). Supports streaming append mode across partition chunks.

    Parameters
    ----------
    matching_map : Dict[str, List[str]]
        Mapping of source1_entity_id -> list of matched candidate IDs.
    output_path : Union[str, Path]
        Destination file path (e.g. output/matching_results.tsv).
    append : bool, default False
        If True, appends to output_path. If False, overwrites existing file.
    write_header : bool, default True
        If True, writes the header line 'source1_entity_id\\tmatched_entity_ids\\n'.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with open(out, mode, encoding="utf-8") as f:
        if write_header:
            f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id, matches in matching_map.items():
            s1_clean = str(s1_id).strip()
            if matches:
                # Deduplicate and strip whitespace preserving order to prevent intra_dupes
                clean_matches = [
                    m_clean
                    for m in matches
                    if (m_clean := str(m).strip())
                ]
                deduped_matches = list(dict.fromkeys(clean_matches))
                matched_str = ",".join(deduped_matches)
            else:
                matched_str = ""
            f.write(f"{s1_clean}\t{matched_str}\n")
