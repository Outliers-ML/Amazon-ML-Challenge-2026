"""Tests for cascaded GPU scoring pipeline and hard negative mining.

Covers:
- Adaptive GBDT candidate filtering for Cross-Encoder (list & DataFrame)
- Meta-probability blending (scalar & vectorized)
- Hard negative mining from blocker false positives
- Cascaded pipeline end-to-end flow with bipartite matching and singleton guard
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.models.cross_encoder import CrossEncoderReranker, format_pair_text
from src.pipeline.er_trainer import (
    compute_meta_probability,
    compute_meta_probability_vectorized,
    filter_candidates_dataframe,
    filter_candidates_for_cross_encoder,
    mine_hard_negatives,
)
from src.pipeline.post_processing import disambiguate_and_guard, write_matching_results


# =========================================================================
# 1. Adaptive GBDT cutoff candidate filtering
# =========================================================================

def test_adaptive_gbdt_cutoff_filters_non_matches():
    """Verify filter_candidates_for_cross_encoder filters items < cutoff and caps at max_keep."""
    candidates = [
        {"cand_id": "C1", "p_gbdt": 0.05},
        {"cand_id": "C2", "p_gbdt": 0.12},
        {"cand_id": "C3", "p_gbdt": 0.85},
        {"cand_id": "C4", "p_gbdt": 0.11},
        {"cand_id": "C5", "p_gbdt": 0.40},
        {"cand_id": "C6", "p_gbdt": 0.95},
        {"cand_id": "C7", "p_gbdt": 0.20},
        {"cand_id": "C8", "p_gbdt": 0.15},
        {"cand_id": "C9", "p_gbdt": 0.60},
        {"cand_id": "C10", "p_gbdt": 0.30},
    ]

    # Cutoff 0.12: C1 (0.05) and C4 (0.11) should be excluded.
    # Passing (8 items): C6 (0.95), C3 (0.85), C9 (0.60), C5 (0.40), C10 (0.30), C7 (0.20), C8 (0.15), C2 (0.12)
    filtered = filter_candidates_for_cross_encoder(candidates, cutoff=0.12, max_keep=5, prob_key="p_gbdt")
    assert len(filtered) == 5
    # Must be sorted descending by probability
    probs = [c["p_gbdt"] for c in filtered]
    assert probs == [0.95, 0.85, 0.60, 0.40, 0.30]
    assert [c["cand_id"] for c in filtered] == ["C6", "C3", "C9", "C5", "C10"]

    # When fewer than max_keep pass cutoff
    filtered_all_passing = filter_candidates_for_cross_encoder(candidates, cutoff=0.70, max_keep=8, prob_key="p_gbdt")
    assert len(filtered_all_passing) == 2
    assert [c["cand_id"] for c in filtered_all_passing] == ["C6", "C3"]

    # When none pass cutoff
    filtered_none = filter_candidates_for_cross_encoder(candidates, cutoff=0.99, max_keep=8, prob_key="p_gbdt")
    assert filtered_none == []

    # Edge cases: empty candidates, missing keys, ties
    assert filter_candidates_for_cross_encoder([], cutoff=0.12, max_keep=8) == []

    # Missing prob_key treated as 0.0 or omitted
    mixed = [{"cand_id": "C_valid", "p_gbdt": 0.50}, {"cand_id": "C_missing"}]
    res_mixed = filter_candidates_for_cross_encoder(mixed, cutoff=0.12, max_keep=8)
    assert len(res_mixed) == 1
    assert res_mixed[0]["cand_id"] == "C_valid"

    # Stable tie-breaking
    ties = [
        {"cand_id": "T1", "p_gbdt": 0.50},
        {"cand_id": "T2", "p_gbdt": 0.50},
    ]
    res_ties = filter_candidates_for_cross_encoder(ties, cutoff=0.12, max_keep=1)
    assert len(res_ties) == 1
    assert res_ties[0]["cand_id"] == "T1"


def test_adaptive_gbdt_cutoff_dataframe_support():
    """Verify filter_candidates_dataframe filters per S1 entity and respects cutoff and max_keep."""
    df = pd.DataFrame({
        "s1_id": ["S1_A"] * 6 + ["S1_B"] * 3 + ["S1_C"] * 2,
        "cand_id": [f"CA_{i}" for i in range(6)] + [f"CB_{i}" for i in range(3)] + ["CC_0", "CC_1"],
        "p_gbdt": [
            # S1_A: 6 candidates
            0.05, 0.12, 0.80, 0.45, 0.10, 0.90,
            # S1_B: 3 candidates (all below cutoff 0.12)
            0.01, 0.08, 0.11,
            # S1_C: 2 candidates (both above cutoff)
            0.50, 0.60,
        ],
    })

    result_df = filter_candidates_dataframe(df, cutoff=0.12, max_keep=2, prob_col="p_gbdt", s1_col="s1_id")

    # S1_A had 4 candidates >= 0.12: 0.90 (CA_5), 0.80 (CA_2), 0.45 (CA_3), 0.12 (CA_1)
    # With max_keep=2, should only keep top 2: CA_5 and CA_2
    s1_a_res = result_df[result_df["s1_id"] == "S1_A"]
    assert len(s1_a_res) == 2
    assert list(s1_a_res["cand_id"]) == ["CA_5", "CA_2"]

    # S1_B: all below cutoff -> 0 rows
    s1_b_res = result_df[result_df["s1_id"] == "S1_B"]
    assert len(s1_b_res) == 0

    # S1_C: both above cutoff -> keeps both sorted descending
    s1_c_res = result_df[result_df["s1_id"] == "S1_C"]
    assert len(s1_c_res) == 2
    assert list(s1_c_res["cand_id"]) == ["CC_1", "CC_0"]

    # Empty DataFrame check
    empty_df = pd.DataFrame(columns=["s1_id", "cand_id", "p_gbdt"])
    assert filter_candidates_dataframe(empty_df).empty


def test_filter_candidates_for_cross_encoder_accepts_dataframe_directly():
    """Verify calling filter_candidates_for_cross_encoder with a DataFrame works seamlessly."""
    df = pd.DataFrame({
        "s1_id": ["S1_A", "S1_A", "S1_B"],
        "cand_id": ["C1", "C2", "C3"],
        "p_gbdt": [0.85, 0.05, 0.60],
    })
    # Should not raise "ValueError: The truth value of a DataFrame is ambiguous"
    res = filter_candidates_for_cross_encoder(df, cutoff=0.12, max_keep=8)
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 2
    assert set(res["cand_id"]) == {"C1", "C3"}

    # Custom column names
    df_custom = pd.DataFrame({
        "source1_id": ["S1_X", "S1_X"],
        "cand_id": ["C_X1", "C_X2"],
        "score": [0.90, 0.10],
    })
    res_custom = filter_candidates_for_cross_encoder(
        df_custom, cutoff=0.12, max_keep=1, prob_key="score", s1_key="source1_id"
    )
    assert isinstance(res_custom, pd.DataFrame)
    assert len(res_custom) == 1
    assert list(res_custom["cand_id"]) == ["C_X1"]


# =========================================================================
# 2. Meta-Probability Blending (Scalar and Vectorized)
# =========================================================================

def test_meta_probability_blending_scalar_and_vectorized():
    """Verify compute_meta_probability with and without Cross-Encoder and its vectorized counterpart."""
    # Case 1: p_ce is present
    # p_final = w_ce * p_ce + w_xgb * p_xgb + w_cat * p_cat
    # 0.50 * 0.8 + 0.25 * 0.6 + 0.25 * 0.4 = 0.40 + 0.15 + 0.10 = 0.65
    p1 = compute_meta_probability(p_ce=0.8, p_xgb=0.6, p_cat=0.4)
    assert pytest.approx(p1, abs=1e-5) == 0.65

    # Case 2: p_ce is None -> normalize GBDT weights
    # gbdt_total_w = 0.25 + 0.25 = 0.50
    # (0.25 / 0.50) * 0.6 + (0.25 / 0.50) * 0.4 = 0.50
    p2 = compute_meta_probability(p_ce=None, p_xgb=0.6, p_cat=0.4)
    assert pytest.approx(p2, abs=1e-5) == 0.50

    # Case 3: Clamping to [0.0, 1.0]
    p_high = compute_meta_probability(p_ce=1.5, p_xgb=1.2, p_cat=1.1)
    assert p_high == 1.0
    p_low = compute_meta_probability(p_ce=-0.5, p_xgb=-0.1, p_cat=-0.2)
    assert p_low == 0.0

    # Case 4: Custom weights
    p_custom = compute_meta_probability(
        p_ce=0.8, p_xgb=0.6, p_cat=0.4, w_ce=0.60, w_xgb=0.20, w_cat=0.20
    )
    # 0.60 * 0.8 + 0.20 * 0.6 + 0.20 * 0.4 = 0.48 + 0.12 + 0.08 = 0.68
    assert pytest.approx(p_custom, abs=1e-5) == 0.68

    # Vectorized testing
    p_ce_arr = [0.8, None, 0.9, np.nan, 0.1]
    p_xgb_arr = np.array([0.6, 0.6, 0.7, 0.4, 0.2], dtype=np.float32)
    p_cat_arr = np.array([0.4, 0.4, 0.8, 0.6, 0.3], dtype=np.float32)

    vec_res = compute_meta_probability_vectorized(p_ce_arr, p_xgb_arr, p_cat_arr)
    assert isinstance(vec_res, np.ndarray)
    assert len(vec_res) == len(p_ce_arr)

    # Compare each entry with scalar version
    expected = [
        compute_meta_probability(ce, xgb, cat)
        for ce, xgb, cat in zip(p_ce_arr, p_xgb_arr, p_cat_arr)
    ]
    np.testing.assert_allclose(vec_res, expected, atol=1e-5)

    # Vectorized empty check
    empty_res = compute_meta_probability_vectorized([], np.array([]), np.array([]))
    assert len(empty_res) == 0


# =========================================================================
# 3. Hard Negative Mining
# =========================================================================

def test_mine_hard_negatives_extracts_positives_and_false_positives():
    """Verify mine_hard_negatives mines ground truth matches and blocker false positives."""
    blocker_candidates = {
        # S1_01: 1 true match in blocker, 2 false positives
        "S1_01": ["S2_10", "S2_11", "S2_12"],
        # S1_02: Singleton (no true matches), 2 false positives
        "S1_02": ["S2_20", "S2_21"],
        # S1_03: True match S2_30 missed by blocker, 1 false positive
        "S1_03": ["S2_31"],
        # S1_04: True matches S2_40, S2_41 both in blocker, plus 12 false positives
        "S1_04": ["S2_40", "S2_41"] + [f"S2_FP_{i}" for i in range(12)],
    }

    ground_truth = {
        "S1_01": ["S2_10"],
        "S1_02": [],
        "S1_03": ["S2_30"],
        "S1_04": ["S2_40", "S2_41"],
    }

    mined_pairs = mine_hard_negatives(
        blocker_candidates=blocker_candidates,
        ground_truth=ground_truth,
        sample_size=10,
        seed=42,
        max_negatives_per_entity=3,
    )

    # Verify return type
    assert isinstance(mined_pairs, list)
    assert all(isinstance(p, tuple) and len(p) == 3 for p in mined_pairs)

    # Convert to dictionary of (s1_id, cand_id) -> label
    pair_map = {(s1, cand): label for s1, cand, label in mined_pairs}

    # Positives check
    assert pair_map.get(("S1_01", "S2_10")) == 1
    assert pair_map.get(("S1_03", "S2_30")) == 1  # Missed ground truth recovered
    assert pair_map.get(("S1_04", "S2_40")) == 1
    assert pair_map.get(("S1_04", "S2_41")) == 1

    # Hard negatives check
    assert pair_map.get(("S1_01", "S2_11")) == 0
    assert pair_map.get(("S1_01", "S2_12")) == 0
    assert pair_map.get(("S1_02", "S2_20")) == 0
    assert pair_map.get(("S1_02", "S2_21")) == 0
    assert pair_map.get(("S1_03", "S2_31")) == 0

    # Max negatives capping check: S1_04 had 12 false positives, but max_negatives_per_entity=3
    s1_04_negatives = [p for p in mined_pairs if p[0] == "S1_04" and p[2] == 0]
    assert len(s1_04_negatives) == 3

    # Deterministic sampling check with seed
    mined_pairs_sample1 = mine_hard_negatives(
        blocker_candidates=blocker_candidates,
        ground_truth=ground_truth,
        sample_size=2,
        seed=42,
        max_negatives_per_entity=3,
    )
    mined_pairs_sample2 = mine_hard_negatives(
        blocker_candidates=blocker_candidates,
        ground_truth=ground_truth,
        sample_size=2,
        seed=42,
        max_negatives_per_entity=3,
    )
    assert mined_pairs_sample1 == mined_pairs_sample2
    sampled_entities = set(p[0] for p in mined_pairs_sample1)
    assert len(sampled_entities) == 2


# =========================================================================
# 4. Cascaded Pipeline End-to-End Mock Flow
# =========================================================================

def test_cascaded_pipeline_end_to_end_mock(tmp_path: Path):
    """Mock end-to-end flow: candidate pairs -> GBDT filter -> Cross-Encoder -> Meta-P -> Bipartite -> Guard."""
    # 1. Synthetic entities
    s1_records = [
        {"entity_id": "S1-01", "business_name": "Acme Tools LLC", "business_address": "123 Main St", "country": "US"},
        {"entity_id": "S1-02", "business_name": "Solo Venture Co", "business_address": "99 Broadway", "country": "US"},
        {"entity_id": "S1-03", "business_name": "Globex Industrial", "business_address": "400 Oak Ave", "country": "US"},
    ]
    target_records = {
        "S2-10": {"entity_id": "S2-10", "business_name": "Acme Tools", "business_address": "123 Main Street", "country": "US"},
        "S2-11": {"entity_id": "S2-11", "business_name": "Acme Hardware", "business_address": "125 Main St", "country": "US"},
        "S2-20": {"entity_id": "S2-20", "business_name": "Random Bakery", "business_address": "100 Market St", "country": "US"},
        "S3-30": {"entity_id": "S3-30", "business_name": "Globex Corp", "business_address": "400 Oak Avenue", "country": "US"},
    }

    # 2. Simulated Stage 1 GBDT outputs
    gbdt_candidate_pairs = [
        # S1-01: Cands S2-10 (high GBDT), S2-11 (moderate GBDT), S2-20 (low GBDT < cutoff)
        {"source1_id": "S1-01", "cand_id": "S2-10", "p_gbdt": 0.75, "p_xgb": 0.74, "p_cat": 0.76, "country": "US"},
        {"source1_id": "S1-01", "cand_id": "S2-11", "p_gbdt": 0.35, "p_xgb": 0.30, "p_cat": 0.40, "country": "US"},
        {"source1_id": "S1-01", "cand_id": "S2-20", "p_gbdt": 0.05, "p_xgb": 0.04, "p_cat": 0.06, "country": "US"},
        # S1-02: True singleton (all candidates low GBDT < cutoff)
        {"source1_id": "S1-02", "cand_id": "S2-20", "p_gbdt": 0.08, "p_xgb": 0.07, "p_cat": 0.09, "country": "US"},
        # S1-03: S3-30 (high GBDT)
        {"source1_id": "S1-03", "cand_id": "S3-30", "p_gbdt": 0.82, "p_xgb": 0.80, "p_cat": 0.84, "country": "US"},
    ]

    # 3. Cascaded Filter: cutoff=0.12, max_keep=8
    ce_pairs = []
    for s1 in s1_records:
        s1_id = s1["entity_id"]
        cands_for_s1 = [p for p in gbdt_candidate_pairs if p["source1_id"] == s1_id]
        filtered_for_ce = filter_candidates_for_cross_encoder(cands_for_s1, cutoff=0.12, max_keep=8, prob_key="p_gbdt")
        ce_pairs.extend(filtered_for_ce)

    # S1-01 keeps S2-10 and S2-11 (S2-20 filtered out)
    # S1-02 keeps nothing (all < 0.12)
    # S1-03 keeps S3-30
    assert len(ce_pairs) == 3
    assert set(p["cand_id"] for p in ce_pairs) == {"S2-10", "S2-11", "S3-30"}

    # 4. Cross-Encoder reranking using Mock model
    reranker = CrossEncoderReranker(model_name="mock", device="cpu")
    formatted_texts = [
        format_pair_text(
            s1_name=next(s["business_name"] for s in s1_records if s["entity_id"] == p["source1_id"]),
            s1_addr=next(s["business_address"] for s in s1_records if s["entity_id"] == p["source1_id"]),
            s1_country=p["country"],
            cand_name=target_records[p["cand_id"]]["business_name"],
            cand_addr=target_records[p["cand_id"]]["business_address"],
            cand_country=target_records[p["cand_id"]]["country"],
        )
        for p in ce_pairs
    ]
    p_ce_scores = reranker.predict_proba(formatted_texts, batch_size=4)
    assert len(p_ce_scores) == len(ce_pairs)

    # 5. Compute Meta-Probability
    scored_pairs = []
    for pair, p_ce in zip(ce_pairs, p_ce_scores):
        p_final = compute_meta_probability(
            p_ce=float(p_ce),
            p_xgb=pair["p_xgb"],
            p_cat=pair["p_cat"],
            w_ce=0.50,
            w_xgb=0.25,
            w_cat=0.25,
        )
        scored_pairs.append({
            "source1_id": pair["source1_id"],
            "candidate_id": pair["cand_id"],
            "p_final": p_final,
            "country": pair["country"],
        })

    # For testing bipartite matching and singleton guard, set known p_final values:
    # S1-01 -> S2-10 (0.85 >= tau_singleton 0.74), S2-11 (0.65 >= tau_secondary 0.60)
    # S1-02 -> no pairs (singleton)
    # S1-03 -> S3-30 (0.90 >= tau_singleton 0.74)
    for sp in scored_pairs:
        if sp["source1_id"] == "S1-01" and sp["candidate_id"] == "S2-10":
            sp["p_final"] = 0.85
        elif sp["source1_id"] == "S1-01" and sp["candidate_id"] == "S2-11":
            sp["p_final"] = 0.65
        elif sp["source1_id"] == "S1-03" and sp["candidate_id"] == "S3-30":
            sp["p_final"] = 0.90

    # 6. Post-processing: disambiguate_and_guard
    all_s1_ids = [s["entity_id"] for s in s1_records]
    results = disambiguate_and_guard(
        scored_pairs=scored_pairs,
        all_s1_ids=all_s1_ids,
        tau_singleton=0.74,
        tau_secondary=0.60,
    )

    # Assert correct matching results
    assert "S2-10" in results["S1-01"]
    assert "S2-11" in results["S1-01"]
    assert results["S1-02"] == []  # Singleton protected
    assert results["S1-03"] == ["S3-30"]

    # 7. Write matching results TSV
    out_file = tmp_path / "test_matching_results.tsv"
    write_matching_results(results, out_file)
    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert "source1_entity_id\tmatched_entity_ids\n" in content
    assert "S1-01\tS2-10,S2-11\n" in content or "S1-01\tS2-11,S2-10\n" in content
    assert "S1-02\t\n" in content
    assert "S1-03\tS3-30\n" in content


# =========================================================================
# 5. Edge Cases & Component Testing
# =========================================================================

def test_filter_candidates_edge_cases():
    """Verify robust handling of invalid / malformed candidates."""
    bad_cands = [
        None,
        {},
        {"cand_id": "bad1", "p_gbdt": "not_a_float"},
        {"cand_id": "bad2", "p_gbdt": float("nan")},
        {"cand_id": "bad3", "p_gbdt": -0.5},
        {"cand_id": "good1", "p_gbdt": 0.50},
    ]
    res = filter_candidates_for_cross_encoder(bad_cands, cutoff=0.12, max_keep=8)
    assert len(res) == 1
    assert res[0]["cand_id"] == "good1"

    # filter_candidates_dataframe edge cases
    df_missing_col = pd.DataFrame({"s1_id": ["A", "B"], "other": [1, 2]})
    res_df = filter_candidates_dataframe(df_missing_col, prob_col="nonexistent")
    assert res_df.empty

    df_none = filter_candidates_dataframe(None)
    assert df_none.empty


def test_meta_probability_edge_cases():
    """Verify edge cases when weights sum to zero or NaN float."""
    # When weights sum to 0
    res_zero_w = compute_meta_probability(p_ce=None, p_xgb=0.4, p_cat=0.6, w_xgb=0.0, w_cat=0.0)
    assert pytest.approx(res_zero_w, abs=1e-5) == 0.50

    # When p_ce is float('nan')
    res_nan_ce = compute_meta_probability(p_ce=float("nan"), p_xgb=0.4, p_cat=0.6)
    assert pytest.approx(res_nan_ce, abs=1e-5) == 0.50


def test_mine_hard_negatives_edge_cases():
    """Verify edge cases like empty inputs, sample_size <= 0."""
    assert mine_hard_negatives({}, {}) == []
    assert mine_hard_negatives({"A": ["B"]}, {"A": ["B"]}, sample_size=0) == []
    assert mine_hard_negatives({"A": ["B"]}, {"A": ["B"]}, max_negatives_per_entity=0) == [("A", "B", 1)]


def test_ensemble_trainer_predict_components():
    """Verify EREnsembleTrainer.predict_components returns per-model positive probabilities."""
    from src.pipeline.er_trainer import EREnsembleTrainer

    X = np.array([
        [0.1, 0.2],
        [0.9, 0.8],
        [0.2, 0.3],
        [0.8, 0.7],
        [0.1, 0.1],
        [0.9, 0.9],
        [0.3, 0.2],
        [0.7, 0.8],
    ], dtype=np.float32)
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int32)
    groups = ["G1", "G1", "G2", "G2", "G3", "G3", "G4", "G4"]

    trainer = EREnsembleTrainer(models=["lgbm", "xgboost", "catboost"], n_splits=2, seed=42)
    trainer.train(X, y, groups)

    preds = trainer.predict_components(X)
    assert "lgbm" in preds
    assert "xgboost" in preds
    assert "catboost" in preds
    for name, p in preds.items():
        assert len(p) == len(X)
        assert np.all(p >= 0.0) and np.all(p <= 1.0)


def test_run_entity_resolution_cli_with_cross_encoder_and_gpu(tmp_path: Path):
    """End-to-end integration test of run_entity_resolution.py with cascaded flags."""
    import subprocess
    import sys

    # 1. Prepare minimal train and test dataset
    train_dir = tmp_path / "train"
    train_dir.mkdir(parents=True)
    test_dir = tmp_path / "test"
    test_dir.mkdir(parents=True)
    out_dir = tmp_path / "output"
    out_dir.mkdir(parents=True)

    # Train S1, S2, S3, GT (each S1 has matched candidates across 2 groups)
    pd.DataFrame({
        "entity_id": ["S1-01", "S1-02"],
        "business_name": ["Acme Tools", "Solo Corp"],
        "business_address": ["123 Main St", "55 Park Ave"],
        "country": ["US", "US"],
    }).to_csv(train_dir / "train_source1.tsv", sep="\t", index=False)

    pd.DataFrame({
        "entity_id": ["S2-11", "S2-12"],
        "business_name": ["Acme Tools Inc", "Solo Corp LLC"],
        "business_address": ["123 Main St", "55 Park Ave"],
        "country": ["US", "US"],
    }).to_csv(train_dir / "train_source2.tsv", sep="\t", index=False)

    pd.DataFrame({
        "entity_id": ["S3-22"],
        "business_name": ["Acme Hardware"],
        "business_address": ["123 Main St"],
        "country": ["US"],
    }).to_csv(train_dir / "train_source3.tsv", sep="\t", index=False)

    pd.DataFrame({
        "source1_entity_id": ["S1-01", "S1-02"],
        "matched_entity_ids": ["S2-11", "S2-12"],
    }).to_csv(train_dir / "train_ground_truth.tsv", sep="\t", index=False)

    # Test S1, S2, S3
    pd.DataFrame({
        "entity_id": ["S1-T1", "S1-T2"],
        "business_name": ["Acme Tools", "Solo Corp"],
        "business_address": ["123 Main St", "55 Park Ave"],
        "country": ["US", "US"],
    }).to_csv(test_dir / "test_source1.tsv", sep="\t", index=False)

    pd.DataFrame({
        "entity_id": ["S2-T11"],
        "business_name": ["Acme Tools Inc"],
        "business_address": ["123 Main St"],
        "country": ["US"],
    }).to_csv(test_dir / "test_source2.tsv", sep="\t", index=False)

    pd.DataFrame({
        "entity_id": ["S3-T22"],
        "business_name": ["Acme Hardware"],
        "business_address": ["123 Main St"],
        "country": ["US"],
    }).to_csv(test_dir / "test_source3.tsv", sep="\t", index=False)

    cmd = [
        sys.executable,
        "scripts/run_entity_resolution.py",
        "--train-dir", str(train_dir),
        "--test-dir", str(test_dir),
        "--output-dir", str(out_dir),
        "--sample-train-s1", "10",
        "--use-cross-encoder",
        "--cross-encoder-model", "mock",
        "--ce-cutoff", "0.10",
        "--ce-max-keep", "4",
        "--tau-singleton", "0.70",
        "--tau-secondary", "0.50",
        "--skip-pack",
        "--no-track",
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"run_entity_resolution failed:\nstdout:\n{res.stdout}\nstderr:\n{res.stderr}"

    matching_file = out_dir / "matching_results.tsv"
    candidate_file = out_dir / "candidate_pairs.tsv"
    assert matching_file.is_file()
    assert candidate_file.is_file()

    # Verify matching_results content
    m_lines = matching_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(m_lines) == 3  # Header + 2 S1 entities
    assert m_lines[0] == "source1_entity_id\tmatched_entity_ids"

