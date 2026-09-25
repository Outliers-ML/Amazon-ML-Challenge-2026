"""Tests for post-bipartite disambiguation and singleton protection engine."""

import os
import subprocess
import sys
import pandas as pd
import pytest

from src.pipeline.post_processing import (
    disambiguate_and_guard,
    write_matching_results,
)


def test_bipartite_disambiguation_prunes_multi_assignment():
    """S2-001 is claimed by S1-A (p=0.92) and S1-B (p=0.75). S1-A gets S2-001, S1-B gets []."""
    scored_pairs = [
        {"source1_id": "S1-A", "candidate_id": "S2-001", "p_final": 0.92, "country": "US"},
        {"source1_id": "S1-B", "candidate_id": "S2-001", "p_final": 0.75, "country": "US"},
    ]
    all_s1 = ["S1-A", "S1-B"]
    results = disambiguate_and_guard(scored_pairs, all_s1, tau_singleton=0.74, tau_secondary=0.60)
    assert results["S1-A"] == ["S2-001"]
    assert results["S1-B"] == []  # S2-001 claimed by S1-A; S1-B gets empty


def test_post_bipartite_singleton_guard_race_condition_fix():
    """S1-A has top C1 (0.76) and secondary C2 (0.62). S1-B claims C1 with 0.95.

    S1-A loses C1. Surviving candidate C2 (0.62) fails tau_singleton (0.74),
    so S1-A MUST be suppressed to [].
    """
    scored_pairs = [
        {"source1_id": "S1-B", "candidate_id": "C1", "p_final": 0.95, "country": "US"},
        {"source1_id": "S1-A", "candidate_id": "C1", "p_final": 0.76, "country": "US"},
        {"source1_id": "S1-A", "candidate_id": "C2", "p_final": 0.62, "country": "US"},
    ]
    all_s1 = ["S1-A", "S1-B"]
    results = disambiguate_and_guard(scored_pairs, all_s1, tau_singleton=0.74, tau_secondary=0.60)
    assert results["S1-B"] == ["C1"]
    # Critical: S1-A lost C1; surviving cand C2 (0.62) fails tau_singleton (0.74), so S1-A MUST be empty!
    assert results["S1-A"] == []


def test_all_s1_entities_present_in_output():
    """Verify every entity in all_s1_ids appears in the dictionary output even if no candidates or pruned."""
    scored_pairs = [
        {"source1_id": "S1-A", "candidate_id": "S2-100", "p_final": 0.85, "country": "US"},
        # S1-B has candidate below tau_singleton (0.74)
        {"source1_id": "S1-B", "candidate_id": "S2-200", "p_final": 0.65, "country": "US"},
        # S1-C has candidate below tau_secondary (0.60)
        {"source1_id": "S1-C", "candidate_id": "S2-300", "p_final": 0.40, "country": "US"},
        # S1-D has no pairs at all
    ]
    all_s1 = ["S1-A", "S1-B", "S1-C", "S1-D", "S1-E"]
    results = disambiguate_and_guard(scored_pairs, all_s1, tau_singleton=0.74, tau_secondary=0.60)

    # Every S1 must be present in keys
    assert set(results.keys()) == set(all_s1)
    assert results["S1-A"] == ["S2-100"]
    assert results["S1-B"] == []
    assert results["S1-C"] == []
    assert results["S1-D"] == []
    assert results["S1-E"] == []


def test_multi_candidate_retention_when_top_clears_singleton_guard():
    """When top candidate clears tau_singleton, surviving secondary candidates >= tau_secondary are kept."""
    scored_pairs = [
        {"source1_id": "S1-A", "candidate_id": "S2-01", "p_final": 0.88, "country": "US"},
        {"source1_id": "S1-A", "candidate_id": "S3-02", "p_final": 0.68, "country": "US"},
        {"source1_id": "S1-A", "candidate_id": "S2-03", "p_final": 0.62, "country": "US"},
        {"source1_id": "S1-A", "candidate_id": "S3-04", "p_final": 0.55, "country": "US"},  # below tau_secondary
    ]
    all_s1 = ["S1-A"]
    results = disambiguate_and_guard(scored_pairs, all_s1, tau_singleton=0.74, tau_secondary=0.60)
    # S3-04 pruned by secondary threshold. S2-01, S3-02, S2-03 retained in score DESC order.
    assert results["S1-A"] == ["S2-01", "S3-02", "S2-03"]


def test_per_country_partitioning_disjoint():
    """Candidate pairs are partitioned by country to ensure zero cross-country matching."""
    scored_pairs = [
        {"source1_id": "S1-US1", "candidate_id": "C-SHARED", "p_final": 0.90, "country": "US"},
        {"source1_id": "S1-IN1", "candidate_id": "C-SHARED", "p_final": 0.85, "country": "India"},
        {"source1_id": "S1-FR1", "candidate_id": "C-FR1", "p_final": 0.80, "country": "France"},
    ]
    all_s1 = ["S1-US1", "S1-IN1", "S1-FR1"]
    results = disambiguate_and_guard(scored_pairs, all_s1, tau_singleton=0.74, tau_secondary=0.60)
    # Since partitions are country-isolated, C-SHARED in US partition and IN partition do not conflict
    assert results["S1-US1"] == ["C-SHARED"]
    assert results["S1-IN1"] == ["C-SHARED"]
    assert results["S1-FR1"] == ["C-FR1"]


def test_deterministic_tie_breaking():
    """Equal scores are tie-broken deterministically by source1_id and candidate_id."""
    scored_pairs = [
        {"source1_id": "S1-B", "candidate_id": "S2-001", "p_final": 0.85, "country": "US"},
        {"source1_id": "S1-A", "candidate_id": "S2-001", "p_final": 0.85, "country": "US"},
    ]
    all_s1 = ["S1-A", "S1-B"]
    results = disambiguate_and_guard(scored_pairs, all_s1, tau_singleton=0.74, tau_secondary=0.60)
    # S1-A comes before S1-B alphabetically, so S1-A claims S2-001
    assert results["S1-A"] == ["S2-001"]
    assert results["S1-B"] == []


def test_dataframe_input_support():
    """disambiguate_and_guard supports pandas DataFrame input with alternative column names."""
    df = pd.DataFrame([
        {"source1_entity_id": "S1-01", "candidate_entity_id": "S2-01", "score": 0.91, "country": "US"},
        {"source1_entity_id": "S1-02", "candidate_entity_id": "S2-01", "score": 0.72, "country": "US"},
    ])
    results = disambiguate_and_guard(df, all_s1_ids=["S1-01", "S1-02"])
    assert results["S1-01"] == ["S2-01"]
    assert results["S1-02"] == []


def test_write_matching_results_format(tmp_path):
    """Test TSV writer strictly formats source1_entity_id\tmatched_entity_ids without quoting."""
    matching_map = {
        "S1-00001": ["S2-00047", "S2-00193", "S3-00812"],
        "S1-00002": ["S3-00004"],
        "S1-00003": [],
    }
    out_file = tmp_path / "output" / "matching_results.tsv"
    write_matching_results(matching_map, out_file)

    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    lines = content.splitlines()

    assert lines[0] == "source1_entity_id\tmatched_entity_ids"
    assert lines[1] == "S1-00001\tS2-00047,S2-00193,S3-00812"
    assert lines[2] == "S1-00002\tS3-00004"
    assert lines[3] == "S1-00003\t"
    assert len(lines) == 4

    # Verify no quoting characters present
    assert '"' not in content
    assert "'" not in content


def test_write_matching_results_append_and_header_control(tmp_path):
    """Test append mode and header suppression across partitions."""
    out_file = tmp_path / "matching_results.tsv"

    # Partition 1: US
    part_us = {"S1-US-1": ["S2-01"]}
    write_matching_results(part_us, out_file, append=False, write_header=True)

    # Partition 2: India (append without header)
    part_in = {"S1-IN-1": ["S3-99"]}
    write_matching_results(part_in, out_file, append=True, write_header=False)

    lines = out_file.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "source1_entity_id\tmatched_entity_ids"
    assert lines[1] == "S1-US-1\tS2-01"
    assert lines[2] == "S1-IN-1\tS3-99"
    assert len(lines) == 3


def test_write_matching_results_passes_official_validator(tmp_path):
    """Test written TSV strictly validates under student_resource/utils/validate_submission.py."""
    test_dir = tmp_path / "test_dir"
    test_dir.mkdir(parents=True, exist_ok=True)
    test_s1 = pd.DataFrame({
        "entity_id": ["S1-01", "S1-02"],
        "business_name": ["Acme Tools", "Solo Corp"],
        "business_address": ["123 Main St", "456 Elm St"],
        "country": ["US", "US"],
    })
    test_s1.to_csv(test_dir / "test_source1.tsv", sep="\t", index=False)

    matching_map = {
        "S1-01": ["S2-11"],
        "S1-02": [],
    }
    match_file = tmp_path / "matching_results.tsv"
    write_matching_results(matching_map, match_file)

    # Create dummy candidate file matching requirements
    cand_file = tmp_path / "candidate_pairs.tsv"
    cand_file.write_text(
        "source1_entity_id\tcandidate_entity_ids\n"
        "S1-01\tS2-11\n"
        "S1-02\t\n",
        encoding="utf-8",
    )

    validator_script = "student_resource/utils/validate_submission.py"
    if os.path.exists(validator_script):
        res = subprocess.run(
            [
                sys.executable,
                validator_script,
                "--matching", str(match_file),
                "--candidate", str(cand_file),
                "--test-dir", str(test_dir),
            ],
            capture_output=True,
            text=True,
        )
        assert res.returncode == 0, f"Validator failed:\n{res.stdout}\n{res.stderr}"


def test_threshold_boundary_conditions():
    """Verify exact boundary values for tau_singleton (0.74) and tau_secondary (0.60)."""
    scored_pairs = [
        {"source1_id": "S1-Exact", "candidate_id": "C-Exact-Top", "p_final": 0.74, "country": "US"},
        {"source1_id": "S1-Exact", "candidate_id": "C-Exact-Sec", "p_final": 0.60, "country": "US"},
        {"source1_id": "S1-Under", "candidate_id": "C-Under-Top", "p_final": 0.7399, "country": "US"},
        {"source1_id": "S1-Under", "candidate_id": "C-Under-Sec", "p_final": 0.5999, "country": "US"},
    ]
    all_s1 = ["S1-Exact", "S1-Under"]
    results = disambiguate_and_guard(scored_pairs, all_s1, tau_singleton=0.74, tau_secondary=0.60)
    # S1-Exact meets exactly 0.74 and 0.60
    assert results["S1-Exact"] == ["C-Exact-Top", "C-Exact-Sec"]
    # S1-Under fails 0.74, suppressed to []
    assert results["S1-Under"] == []


def test_duplicate_pairs_deduplication():
    """Verify duplicate candidate pairs are assigned cleanly without duplicating in output."""
    scored_pairs = [
        {"source1_id": "S1-A", "candidate_id": "C1", "p_final": 0.90, "country": "US"},
        {"source1_id": "S1-A", "candidate_id": "C1", "p_final": 0.90, "country": "US"},
        {"source1_id": "S1-A", "candidate_id": "C1", "p_final": 0.85, "country": "US"},
    ]
    results = disambiguate_and_guard(scored_pairs, all_s1_ids=["S1-A"])
    assert results["S1-A"] == ["C1"]


def test_empty_inputs():
    """Verify behavior with empty scored_pairs or empty all_s1_ids."""
    assert disambiguate_and_guard([], all_s1_ids=[]) == {}
    assert disambiguate_and_guard([], all_s1_ids=["S1-X", "S1-Y"]) == {"S1-X": [], "S1-Y": []}


def test_none_all_s1_ids():
    """Verify behavior when all_s1_ids is None."""
    scored_pairs = [
        {"source1_id": "S1-Z", "candidate_id": "C-Z", "p_final": 0.88, "country": "US"},
    ]
    results = disambiguate_and_guard(scored_pairs, all_s1_ids=None)
    assert "S1-Z" in results
    assert results["S1-Z"] == ["C-Z"]
