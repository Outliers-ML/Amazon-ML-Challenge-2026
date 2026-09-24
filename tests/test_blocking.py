import pandas as pd
from pathlib import Path
from src.data.blocking import MultiKeyBlocker


def test_blocking_generates_candidates():
    s1 = pd.DataFrame({
        "entity_id": ["S1-1", "S1-2"],
        "business_name": ["Orelee Barbershop", "Unique Solar Tech"],
        "business_address": ["1795 Westchester Drive, High Point, NC 27262", "99 Apollo Way"],
        "country": ["US", "US"]
    })
    s2 = pd.DataFrame({
        "entity_id": ["S2-10", "S2-20"],
        "business_name": ["Orelee's Barbershop Inc", "Completely Different"],
        "business_address": ["1795 Westchester Dr, NC 27262", "123 Main St"],
        "country": ["US", "US"]
    })
    s3 = pd.DataFrame({
        "entity_id": ["S3-30"],
        "business_name": ["Orelee Barber Shop"],
        "business_address": ["1795 Westchester Drive, NC"],
        "country": ["US"]
    })

    blocker = MultiKeyBlocker(max_candidates=10)
    candidates = blocker.block_country_partition(s1, s2, s3)
    
    assert "S1-1" in candidates
    assert "S2-10" in candidates["S1-1"]
    assert "S3-30" in candidates["S1-1"]
    assert len(candidates["S1-2"]) == 0  # Singleton handled cleanly


def test_candidate_pairs_file_export(tmp_path):
    candidates_map = {
        "S1-01": ["S2-10", "S3-30"],
        "S1-02": []
    }
    tsv_file = tmp_path / "candidate_pairs.tsv"
    MultiKeyBlocker.write_candidate_pairs(candidates_map, tsv_file)
    assert tsv_file.exists()
    content = tsv_file.read_text().splitlines()
    assert content[0] == "source1_entity_id\tcandidate_entity_ids"
    assert "S1-01\tS2-10,S3-30" in content
    assert "S1-02\t" in content


def test_blocking_handles_empty_partitions():
    s1 = pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"])
    s2 = pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"])
    s3 = pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"])

    blocker = MultiKeyBlocker(max_candidates=10)
    candidates = blocker.block_country_partition(s1, s2, s3)
    assert candidates == {}


def test_blocking_respects_max_candidates():
    s1 = pd.DataFrame({
        "entity_id": ["S1-1"],
        "business_name": ["Alpha Beta"],
        "business_address": ["100 Main St 12345"],
        "country": ["US"]
    })
    # Create 10 matching records in S2
    s2 = pd.DataFrame({
        "entity_id": [f"S2-{i}" for i in range(10)],
        "business_name": ["Alpha Beta"] * 10,
        "business_address": ["100 Main St 12345"] * 10,
        "country": ["US"] * 10
    })
    s3 = pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"])

    blocker = MultiKeyBlocker(max_candidates=3)
    candidates = blocker.block_country_partition(s1, s2, s3)
    assert len(candidates["S1-1"]) == 3


def test_blocking_handles_missing_fields_and_nans():
    s1 = pd.DataFrame({
        "entity_id": ["S1-1", "S1-2"],
        "business_name": [None, "Valid Name"],
        "business_address": [float("nan"), None],
        "country": ["US", "US"]
    })
    s2 = pd.DataFrame({
        "entity_id": ["S2-1"],
        "business_name": ["Valid Name"],
        "business_address": ["Some Address"],
        "country": ["US"]
    })
    s3 = pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"])

    blocker = MultiKeyBlocker(max_candidates=5)
    candidates = blocker.block_country_partition(s1, s2, s3)
    assert candidates["S1-1"] == []
    assert "S2-1" in candidates["S1-2"]
