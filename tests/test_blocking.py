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


def test_ranking_prioritizes_anchor_and_name_over_single_token():
    s1 = pd.DataFrame({
        "entity_id": ["S1-1"],
        "business_name": ["Apex Auto Repair"],
        "business_address": ["450 Industrial Blvd, Austin, TX 78745"],
        "country": ["US"],
    })
    s2 = pd.DataFrame({
        "entity_id": ["S2-SINGLE", "S2-ANCHOR-NAME"],
        "business_name": [
            "Apex Restaurant",  # Matches single token "apex"
            "Apex Auto Body",    # Matches tokens "apex", "auto" + exact anchor
        ],
        "business_address": [
            "100 Random St, Dallas, TX 75001",
            "450 Industrial Blvd, Austin, TX 78745",
        ],
        "country": ["US", "US"],
    })
    s3 = pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"])

    blocker = MultiKeyBlocker(max_candidates=10)
    candidates = blocker.block_country_partition(s1, s2, s3)

    assert "S1-1" in candidates
    cands = candidates["S1-1"]
    assert cands[0] == "S2-ANCHOR-NAME"
    assert "S2-SINGLE" in cands
    assert cands.index("S2-ANCHOR-NAME") < cands.index("S2-SINGLE")


def test_s3_targets_not_starved_by_s2_token_sharing():
    # S2 has 120 targets sharing token "boutique"
    s2_records = [
        {
            "entity_id": f"S2-{i:03d}",
            "business_name": f"Fashion Boutique {i}",
            "business_address": f"{100 + i} Fashion Way, New York, NY 10001",
            "country": "US",
        }
        for i in range(120)
    ]
    s2 = pd.DataFrame(s2_records)

    # S3 target shares "boutique" and matches the query exactly
    s3 = pd.DataFrame({
        "entity_id": ["S3-TARGET"],
        "business_name": ["Charm Fashion Boutique"],
        "business_address": ["999 Broadway, New York, NY 10001"],
        "country": ["US"],
    })

    s1 = pd.DataFrame({
        "entity_id": ["S1-1"],
        "business_name": ["Charm Fashion Boutique"],
        "business_address": ["999 Broadway, New York, NY 10001"],
        "country": ["US"],
    })

    blocker = MultiKeyBlocker(max_candidates=10, max_postings=500)
    candidates = blocker.block_country_partition(s1, s2, s3)

    assert "S1-1" in candidates
    # S3-TARGET must not be starved by the 120 S2 records
    assert "S3-TARGET" in candidates["S1-1"]
    assert candidates["S1-1"][0] == "S3-TARGET"


def test_write_candidate_pairs_streaming_append(tmp_path):
    tsv_file = tmp_path / "candidates_streaming.tsv"

    chunk1 = {"S1-01": ["S2-10", "S3-30"]}
    chunk2 = {"S1-02": ["S2-20"], "S1-03": []}

    # Write first chunk with header
    MultiKeyBlocker.write_candidate_pairs(chunk1, tsv_file, append=False, write_header=True)
    # Append second chunk without header
    MultiKeyBlocker.write_candidate_pairs(chunk2, tsv_file, append=True, write_header=False)

    lines = tsv_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4
    assert lines[0] == "source1_entity_id\tcandidate_entity_ids"
    assert lines[1] == "S1-01\tS2-10,S3-30"
    assert lines[2] == "S1-02\tS2-20"
    assert lines[3] == "S1-03\t"


def test_candidate_output_validity_with_submission_validator(tmp_path):
    from student_resource.utils.validate_submission import validate_id_list_file, CANDIDATE_HEADER

    tsv_file = tmp_path / "candidate_pairs.tsv"
    candidates_map = {
        "S1-001": ["S2-100", "S3-200"],
        "S1-002": ["S2-300"],
        "S1-003": [],  # Singleton / no candidates
    }
    MultiKeyBlocker.write_candidate_pairs(candidates_map, tsv_file)

    errors = []
    required_s1 = {"S1-001", "S1-002", "S1-003"}
    valid_targets = {"S2-100", "S3-200", "S2-300"}

    result = validate_id_list_file(
        path=str(tsv_file),
        expected_header=CANDIDATE_HEADER,
        col_label="candidate_entity_ids",
        required=required_s1,
        valid_ids=valid_targets,
        errors=errors,
    )

    assert errors == [], f"Validation errors found: {errors}"
    assert result is not None
    assert result["S1-001"] == {"S2-100", "S3-200"}
    assert result["S1-002"] == {"S2-300"}
    assert result["S1-003"] == set()


def test_ubiquitous_token_filtering():
    # If a token appears in more than max_postings targets, it should be ignored
    s2 = pd.DataFrame({
        "entity_id": [f"S2-{i}" for i in range(10)],
        "business_name": ["Store Services Corp"] * 10,
        "business_address": [f"{i} Street" for i in range(10)],
        "country": ["US"] * 10,
    })
    s3 = pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"])

    # max_postings=5, but all 10 targets share "services", "store", "corp" and their 3-grams
    blocker = MultiKeyBlocker(max_candidates=5, max_postings=5)
    s1 = pd.DataFrame({
        "entity_id": ["S1-1"],
        "business_name": ["Store Services"],
        "business_address": ["999 Unrelated Rd"],
        "country": ["US"],
    })
    candidates = blocker.block_country_partition(s1, s2, s3)
    assert candidates["S1-1"] == []


def test_blocking_strips_entity_id_whitespace():
    s1 = pd.DataFrame({
        "entity_id": ["  S1-10  "],
        "business_name": ["Test Business"],
        "business_address": ["123 Main St, 10001"],
        "country": ["US"],
    })
    s2 = pd.DataFrame({
        "entity_id": ["  S2-20  "],
        "business_name": ["Test Business"],
        "business_address": ["123 Main St, 10001"],
        "country": ["US"],
    })
    s3 = pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"])

    blocker = MultiKeyBlocker()
    candidates = blocker.block_country_partition(s1, s2, s3)

    assert "S1-10" in candidates
    assert "  S1-10  " not in candidates
    assert "S2-20" in candidates["S1-10"]
    assert "  S2-20  " not in candidates["S1-10"]
