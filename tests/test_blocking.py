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


def test_bucket_ceiling_discards_large_clusters():
    from src.data.blocking import MultiTierBlocker

    blocker = MultiTierBlocker(bucket_ceiling=5)
    # 10 records with same prefix
    records = [
        {
            "entity_id": f"S1-{i}",
            "country": "US",
            "clean_name_stripped": "societe generique",
            "clean_address": "1 main st",
        }
        for i in range(10)
    ]
    # And 1 unique record with distinct name and address
    records.append({
        "entity_id": "S1-UNIQUE",
        "country": "US",
        "clean_name_stripped": "unique cafe",
        "clean_address": "999 separate way 10001",
    })
    buckets = blocker.build_deterministic_buckets(records)
    # The common key should be discarded because len > 5
    for key, members in buckets.items():
        assert len(members) <= 5
    # The unique record's key should remain
    assert any("S1-UNIQUE" in members for members in buckets.values())


def test_reciprocal_rank_fusion_scale_invariance():
    from src.data.blocking import reciprocal_rank_fusion

    tier1 = [("C1", 1.0), ("C2", 1.0)]
    tier2 = [("C2", 34.5), ("C3", 12.1)]  # BM25 scores
    tier3 = [("C3", 0.92), ("C1", 0.81)]  # Cosine similarity
    fused = reciprocal_rank_fusion({"t1": tier1, "t2": tier2, "t3": tier3})
    assert len(fused) == 3
    # Check output is ranked tuple (cand_id, score)
    assert all(isinstance(x[0], str) and isinstance(x[1], float) for x in fused)


def test_tier1_deterministic_bucket_keys():
    from src.data.blocking import MultiTierBlocker

    blocker = MultiTierBlocker()
    record = {
        "entity_id": "REC-1",
        "country": "US",
        "clean_name_stripped": "walmart store",
        "clean_address": "100 main st 12345",
        "street_num": "100",
        "postal_code": "12345",
        "metaphone_primary": "ALMR",
    }
    keys = blocker.get_record_bucket_keys(record)
    assert ("US", "walmart store") in keys
    assert ("US", "walm", "100") in keys
    assert ("US", "ALMR", "12345") in keys
    assert ("US", "12345", "100") in keys


def test_multi_tier_blocker_end_to_end(tmp_path):
    from src.data.blocking import MultiTierBlocker, generate_candidate_pairs, write_candidate_pairs

    blocker = MultiTierBlocker(max_candidates=35)

    s1_records = pd.DataFrame({
        "entity_id": ["S1-1", "S1-2"],
        "business_name": ["Apex Auto Repair", "Lonely Bakery"],
        "business_address": ["450 Industrial Blvd, Austin, TX 78745", "10 Elm St"],
        "country": ["US", "US"],
    })
    target_records = pd.DataFrame({
        "entity_id": [f"S2-{i}" for i in range(40)],
        "business_name": ["Apex Auto Body"] * 40,
        "business_address": ["450 Industrial Blvd, Austin, TX 78745"] * 40,
        "country": ["US"] * 40,
    })

    cand_map = blocker.generate_candidate_pairs(s1_records, target_records, country="US", max_candidates=35)
    assert "S1-1" in cand_map
    assert "S1-2" in cand_map
    # S1-1 matches targets, should be capped at top-35
    assert len(cand_map["S1-1"]) == 35
    assert all(isinstance(c, str) for c in cand_map["S1-1"])
    # S1-2 has no match in target_records
    assert len(cand_map["S1-2"]) == 0

    # Also test write_candidate_pairs
    out_file = tmp_path / "candidates.tsv"
    write_candidate_pairs(cand_map, out_file)
    assert out_file.exists()
    lines = out_file.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "source1_entity_id\tcandidate_entity_ids"
    assert lines[1].startswith("S1-1\t")
    assert lines[2] == "S1-2\t"

    # Also test top-level generate_candidate_pairs function
    cand_map_fn = generate_candidate_pairs(s1_records, target_records, country="US", max_candidates=10)
    assert len(cand_map_fn["S1-1"]) == 10


def test_tier3_dense_vector_retrieval():
    import numpy as np
    import torch
    from src.data.blocking import MultiTierBlocker

    blocker = MultiTierBlocker()
    s1_emb = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float32)
    target_emb = np.array([
        [0.9, 0.1, 0.0],  # T-1 close to S1-1
        [0.1, 0.9, 0.0],  # T-2 close to S1-2
        [0.0, 0.0, 1.0],  # T-3 orthogonal
    ], dtype=np.float32)

    # Test with numpy
    res_np = blocker.retrieve_dense_candidates(
        s1_embeddings=s1_emb,
        target_embeddings=target_emb,
        s1_ids=["S1-1", "S1-2"],
        target_ids=["T-1", "T-2", "T-3"],
        top_k=2,
    )
    assert res_np["S1-1"][0][0] == "T-1"
    assert res_np["S1-2"][0][0] == "T-2"

    # Test with PyTorch tensor
    res_torch = blocker.retrieve_dense_candidates(
        s1_embeddings=torch.from_numpy(s1_emb),
        target_embeddings=torch.from_numpy(target_emb),
        s1_ids=["S1-1", "S1-2"],
        target_ids=["T-1", "T-2", "T-3"],
        top_k=2,
    )
    assert res_torch["S1-1"][0][0] == "T-1"
    assert res_torch["S1-2"][0][0] == "T-2"


def test_tier1_name_weighting_and_capping():
    from src.data.blocking import MultiTierBlocker

    blocker = MultiTierBlocker(max_candidates=2)
    s1_prepared = [
        {
            "entity_id": "S1-1",
            "country": "US",
            "clean_name_stripped": "acme supply",
            "clean_address": "100 main st 10001",
            "street_num": "100",
            "postal_code": "10001",
            "metaphone_primary": "AKM",
        }
    ]
    # T1 matches only address keys: (US, 10001, 100) -> score 1.0
    # T2 matches exact name: (US, acme supply) -> score 3.0
    # T3 matches address keys: score 1.0
    target_buckets = {
        ("US", "acme supply"): ["T-NAME"],
        ("US", "10001", "100"): ["T-ADDR1", "T-ADDR2", "T-ADDR3"],
    }
    cands = blocker.retrieve_tier1_candidates(s1_prepared, target_buckets, top_k=2)
    # T-NAME should be first because score is 3.0 vs 1.0
    assert cands["S1-1"][0][0] == "T-NAME"
    assert cands["S1-1"][0][1] == 3.0
    # Top-K capping should restrict to 2 candidates
    assert len(cands["S1-1"]) == 2


def test_tier2_chunked_sparse_matrix_processing():
    from src.data.blocking import MultiTierBlocker

    blocker = MultiTierBlocker()
    s1_prepared = [
        {
            "entity_id": f"S1-{i}",
            "country": "US",
            "clean_name_stripped": f"business name {i}",
            "clean_address": f"{100 + i} main street",
            "street_num": f"{100 + i}",
            "postal_code": "10001",
            "metaphone_primary": "BSN",
        }
        for i in range(5)
    ]
    target_prepared = [
        {
            "entity_id": f"T-{i}",
            "country": "US",
            "clean_name_stripped": f"business name {i}",
            "clean_address": f"{100 + i} main street",
            "street_num": f"{100 + i}",
            "postal_code": "10001",
            "metaphone_primary": "BSN",
        }
        for i in range(5)
    ] + [
        {
            "entity_id": f"T-{i}-alt",
            "country": "US",
            "clean_name_stripped": f"business name {i}",
            "clean_address": f"{100 + i} main street",
            "street_num": f"{100 + i}",
            "postal_code": "10001",
            "metaphone_primary": "BSN",
        }
        for i in range(5)
    ]
    # Test with tiny chunk_size=2 to force multi-chunk processing
    res = blocker.retrieve_tier2_candidates(
        s1_prepared, target_prepared, top_k=3, chunk_size=2
    )
    assert len(res) == 5
    for i in range(5):
        top_cand = res[f"S1-{i}"][0][0]
        assert top_cand in (f"T-{i}", f"T-{i}-alt")


def test_tier3_1d_tensor_and_chunked_retrieval():
    import numpy as np
    import torch
    from src.data.blocking import MultiTierBlocker

    blocker = MultiTierBlocker()
    # 1D tensor inputs
    s1_1d = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
    target_1d = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)

    res = blocker.retrieve_dense_candidates(
        s1_embeddings=s1_1d,
        target_embeddings=target_1d,
        s1_ids=["S1-SINGLE"],
        target_ids=["T-SINGLE"],
        top_k=1,
        chunk_size=1,
    )
    assert res["S1-SINGLE"][0][0] == "T-SINGLE"
    assert round(res["S1-SINGLE"][0][1], 2) == 1.0

    # 1D NumPy inputs
    s1_np_1d = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    target_np_1d = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    res_np = blocker.retrieve_dense_candidates(
        s1_embeddings=s1_np_1d,
        target_embeddings=target_np_1d,
        s1_ids=["S1-NP"],
        target_ids=["T-NP"],
        top_k=1,
        chunk_size=1,
    )
    assert res_np["S1-NP"][0][0] == "T-NP"
    assert round(res_np["S1-NP"][0][1], 2) == 1.0


