"""Post-bipartite disambiguation and singleton protection engine.

Partitions candidates per country, performs greedy maximum-weight bipartite
disambiguation, and applies post-pruning singleton guard to optimize Macro F0.5.
"""

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union
import pandas as pd


def _get_field(d: Dict[str, Any], *candidates: str, default: Any = None) -> Any:
    """Retrieve first existing field from dictionary with fallback keys."""
    for c in candidates:
        if c in d:
            return d[c]
    return default


def disambiguate_and_guard(
    scored_pairs: Union[List[Dict[str, Any]], pd.DataFrame, Iterable[Dict[str, Any]]],
    all_s1_ids: Optional[Sequence[str]] = None,
    tau_singleton: float = 0.74,
    tau_secondary: float = 0.60,
) -> Dict[str, List[str]]:
    """Resolve candidate pairs using bipartite matching and post-pruning singleton guard.

    1. Per-Country Subgraph Partitioning:
       Partitions candidate pairs by country (US, India, France) to ensure zero cross-country
       matching and avoid unnecessary memory overhead.
    2. Greedy Maximum-Weight Bipartite Matching:
       - Filters candidate pairs with P_final >= tau_secondary (default: 0.60).
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
        every entity in all_s1_ids will appear in the output mapping (empty list for singletons).
    tau_singleton : float, default 0.74
        Gatekeeper threshold. The top surviving assigned candidate for an S1 entity must
        have p_final >= tau_singleton, otherwise all candidates for that S1 entity are suppressed
        to [] to protect the 1.0 singleton credit under Macro F0.5.
    tau_secondary : float, default 0.60
        Secondary candidate admission threshold. Only pairs with p_final >= tau_secondary
        enter bipartite resolution.

    Returns
    -------
    Dict[str, List[str]]
        Mapping of source1_entity_id -> list of matched candidate IDs (ordered DESC by score).
    """
    # 0. Normalization of inputs
    if isinstance(scored_pairs, pd.DataFrame):
        pairs_list = scored_pairs.to_dict(orient="records")
    else:
        pairs_list = list(scored_pairs)

    results: Dict[str, List[str]] = {}
    if all_s1_ids is not None:
        for s1 in all_s1_ids:
            results[str(s1)] = []

    # 1. Per-Country Subgraph Partitioning
    partitions: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for pair in pairs_list:
        raw_country = _get_field(pair, "country", "country_code", default="")
        country_norm = str(raw_country).strip().upper() if raw_country is not None else ""
        partitions[country_norm].append(pair)

    # 2. Greedy Maximum-Weight Bipartite Matching per country partition
    assigned_candidates: Dict[str, List[Tuple[str, float]]] = defaultdict(list)

    for country, country_pairs in partitions.items():
        # Filter candidate pairs with P_final >= tau_secondary
        filtered_pairs: List[Tuple[float, str, str]] = []
        for pair in country_pairs:
            s1 = str(_get_field(pair, "source1_id", "source1_entity_id", "s1_id", "s1", default="")).strip()
            cand = str(_get_field(pair, "candidate_id", "candidate_entity_id", "cand_id", "candidate", "s2_s3_id", default="")).strip()
            raw_score = _get_field(pair, "p_final", "score", "prob", "probability", "prediction", default=0.0)
            try:
                p_final = float(raw_score) if raw_score is not None else 0.0
            except (ValueError, TypeError):
                p_final = 0.0

            if not s1 or not cand:
                continue

            if p_final >= tau_secondary:
                filtered_pairs.append((p_final, s1, cand))

        # Sort candidate pairs globally DESC by P_final with deterministic tie-breaker on IDs
        filtered_pairs.sort(key=lambda item: (-item[0], item[1], item[2]))

        # Track claimed candidates so candidate record is assigned to at most ONE Source 1 entity
        claimed_candidates: Set[str] = set()
        for p_final, s1, cand in filtered_pairs:
            if cand not in claimed_candidates:
                claimed_candidates.add(cand)
                assigned_candidates[s1].append((cand, p_final))

    # 3. Post-Pruning Singleton Guard
    if all_s1_ids is None:
        for s1 in sorted(list(assigned_candidates.keys())):
            results[s1] = []
    else:
        # Also ensure any S1 present in assigned_candidates is included in results
        for s1 in assigned_candidates:
            if s1 not in results:
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
                results[s1] = [cand for cand, _ in cand_list]
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
            matched_str = ",".join(matches) if matches else ""
            f.write(f"{s1_id}\t{matched_str}\n")
