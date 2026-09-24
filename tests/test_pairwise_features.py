import numpy as np
import pytest

from src.features.pairwise_features import PairwiseFeatureExtractor


def test_feature_extraction_values():
    extractor = PairwiseFeatureExtractor()
    r1 = {
        "entity_id": "S1-1",
        "business_name": "Orelee Barbershop Inc",
        "business_address": "1795 Westchester Dr, NC 27262",
    }
    r2 = {
        "entity_id": "S2-10",
        "business_name": "Orelee's Barbershop",
        "business_address": "1795 Westchester Dr, High Point, NC 27262",
    }
    feats = extractor.extract_pair_features(r1, r2, rank=1, blocking_score=0.92)
    assert len(feats) == len(extractor.feature_names)
    assert len(feats) == 22
    # Name jaro winkler should be very high
    assert feats[extractor.feature_names.index("name_jaro_winkler")] > 0.85
    # Street number match should be +1.0
    assert feats[extractor.feature_names.index("addr_street_num_status")] == 1.0
    # Is source 2
    assert feats[extractor.feature_names.index("is_source2")] == 1.0
    assert feats[extractor.feature_names.index("is_source3")] == 0.0
    # Blocking rank and score
    assert feats[extractor.feature_names.index("blocking_rank")] == 1.0
    assert pytest.approx(feats[extractor.feature_names.index("blocking_score")], abs=1e-4) == 0.92


def test_all_feature_values_explicit():
    extractor = PairwiseFeatureExtractor()
    r1 = {
        "entity_id": "S1-100",
        "business_name": "Acme Widgets",
        "business_address": "123 Main Street, New York, NY 10001",
    }
    r2 = {
        "entity_id": "S2-200",
        "business_name": "Acme Widgets LLC",
        "business_address": "123 Main Street, New York, NY 10001",
    }
    feats = extractor.extract_pair_features(r1, r2, rank=2, blocking_score=0.85)
    feat_map = dict(zip(extractor.feature_names, feats))

    assert len(extractor.feature_names) == 22
    assert len(feats) == 22
    assert feat_map["name_exact_match"] == 0.0  # "Acme Widgets" != "Acme Widgets LLC"
    assert feat_map["name_clean_exact_match"] == 1.0  # LLC stripped -> "acme widgets"
    assert feat_map["name_jaro_winkler"] == 1.0
    assert feat_map["name_levenshtein_ratio"] == 1.0
    assert feat_map["name_token_sort_ratio"] == 1.0
    assert feat_map["name_token_set_ratio"] == 1.0
    assert feat_map["name_char_3gram_jaccard"] == 1.0
    assert feat_map["name_length_diff"] == 0.0
    assert feat_map["name_length_ratio"] == 1.0
    assert feat_map["name_first_token_match"] == 1.0
    assert feat_map["addr_exact_match"] == 1.0
    assert feat_map["addr_token_jaccard"] == 1.0
    assert feat_map["addr_token_sort_ratio"] == 1.0
    assert feat_map["addr_token_set_ratio"] == 1.0
    assert feat_map["addr_street_num_status"] == 1.0
    assert feat_map["addr_postal_code_status"] == 1.0
    assert feat_map["addr_is_empty"] == 0.0
    assert feat_map["name_in_address_cross"] == 0.0
    assert feat_map["is_source2"] == 1.0
    assert feat_map["is_source3"] == 0.0
    assert feat_map["blocking_rank"] == 2.0
    assert feat_map["blocking_score"] == pytest.approx(0.85, abs=1e-4)


def test_name_char_3gram_jaccard():
    extractor = PairwiseFeatureExtractor()
    # "Alpha Beta" -> compact "alphabeta" (len 9) -> 7 trigrams: alp, lph, pha, hab, abe, bet, eta
    # "Alpha Gamma" -> compact "alphagamma" (len 10) -> 8 trigrams: alp, lph, pha, hag, aga, gam, amm, mma
    # Intersection: alp, lph, pha (3)
    # Union: 7 + 8 - 3 = 12
    # Jaccard: 3 / 12 = 0.25
    r1 = {"entity_id": "S1-1", "business_name": "Alpha Beta", "business_address": "123 Main St"}
    r2 = {"entity_id": "S2-1", "business_name": "Alpha Gamma", "business_address": "123 Main St"}
    feats = extractor.extract_pair_features(r1, r2)
    feat_map = dict(zip(extractor.feature_names, feats))
    assert feat_map["name_char_3gram_jaccard"] == pytest.approx(0.25, abs=1e-4)


def test_name_length_diff_and_ratio():
    extractor = PairwiseFeatureExtractor()
    r1 = {"entity_id": "S1-1", "business_name": "Target", "business_address": "123 Main St"}
    r2 = {"entity_id": "S2-1", "business_name": "Target Supercenter", "business_address": "123 Main St"}
    feats = extractor.extract_pair_features(r1, r2)
    feat_map = dict(zip(extractor.feature_names, feats))
    # clean names: "target" (len 6) vs "target supercenter" (len 18)
    assert feat_map["name_length_diff"] == 12.0
    assert feat_map["name_length_ratio"] == pytest.approx(6.0 / 18.0, abs=1e-4)


def test_addr_token_jaccard():
    extractor = PairwiseFeatureExtractor()
    r1 = {"entity_id": "S1-1", "business_name": "Shop", "business_address": "100 Elm Street"}
    r2 = {"entity_id": "S2-1", "business_name": "Shop", "business_address": "100 Oak Street"}
    feats = extractor.extract_pair_features(r1, r2)
    feat_map = dict(zip(extractor.feature_names, feats))
    # tokens1: {"100", "elm", "street"} (3)
    # tokens2: {"100", "oak", "street"} (3)
    # intersection: {"100", "street"} (2)
    # union: {"100", "elm", "oak", "street"} (4)
    # jaccard: 2 / 4 = 0.5
    assert feat_map["addr_token_jaccard"] == pytest.approx(0.5, abs=1e-4)


def test_conflicting_street_numbers():
    extractor = PairwiseFeatureExtractor()
    r1 = {"entity_id": "S1-1", "business_name": "Subway", "business_address": "105 Elm St, 28655"}
    r2 = {"entity_id": "S2-2", "business_name": "Subway", "business_address": "502 Oak St, 28655"}
    feats = extractor.extract_pair_features(r1, r2, rank=1)
    # Conflict should evaluate to -1.0
    assert feats[extractor.feature_names.index("addr_street_num_status")] == -1.0
    # Postal should evaluate to 1.0
    assert feats[extractor.feature_names.index("addr_postal_code_status")] == 1.0


def test_addr_postal_code_status_conflict():
    extractor = PairwiseFeatureExtractor()
    r1 = {"entity_id": "S1-1", "business_name": "Bakery", "business_address": "100 Main St, New York, NY 10001"}
    r2 = {"entity_id": "S2-1", "business_name": "Bakery", "business_address": "100 Main St, New York, NY 10002"}
    feats = extractor.extract_pair_features(r1, r2)
    feat_map = dict(zip(extractor.feature_names, feats))
    assert feat_map["addr_street_num_status"] == 1.0
    assert feat_map["addr_postal_code_status"] == -1.0


def test_missing_address_components():
    extractor = PairwiseFeatureExtractor()
    r1 = {"entity_id": "S1-1", "business_name": "Alpha Corp", "business_address": ""}
    r2 = {"entity_id": "S2-2", "business_name": "Alpha", "business_address": "123 Main St"}
    feats = extractor.extract_pair_features(r1, r2, rank=2)
    assert feats[extractor.feature_names.index("addr_is_empty")] == 1.0
    assert feats[extractor.feature_names.index("addr_street_num_status")] == 0.0
    assert feats[extractor.feature_names.index("addr_postal_code_status")] == 0.0


def test_name_in_address_cross():
    extractor = PairwiseFeatureExtractor()
    # Case A: s1 business name embedded in target raw address
    r1 = {"entity_id": "S1-1", "business_name": "Starbucks", "business_address": "100 Main St, Seattle WA 98101"}
    r2 = {"entity_id": "S2-1", "business_name": "Bookstore Cafe", "business_address": "100 Main St, Inside Starbucks, Seattle WA 98101"}
    feats_a = extractor.extract_pair_features(r1, r2)
    feat_map_a = dict(zip(extractor.feature_names, feats_a))
    assert feat_map_a["name_in_address_cross"] == 1.0

    # Case B: target business name embedded in s1 raw address
    r3 = {"entity_id": "S1-2", "business_name": "Bookstore Cafe", "business_address": "100 Main St, Inside Starbucks, Seattle WA 98101"}
    r4 = {"entity_id": "S2-2", "business_name": "Starbucks", "business_address": "100 Main St, Seattle WA 98101"}
    feats_b = extractor.extract_pair_features(r3, r4)
    feat_map_b = dict(zip(extractor.feature_names, feats_b))
    assert feat_map_b["name_in_address_cross"] == 1.0

    # Case C: no cross containment
    r5 = {"entity_id": "S1-3", "business_name": "Acme Widgets", "business_address": "100 Main St"}
    r6 = {"entity_id": "S2-3", "business_name": "Apex Tools", "business_address": "200 Elm St"}
    feats_c = extractor.extract_pair_features(r5, r6)
    feat_map_c = dict(zip(extractor.feature_names, feats_c))
    assert feat_map_c["name_in_address_cross"] == 0.0


def test_source3_and_none_values():
    extractor = PairwiseFeatureExtractor()
    r1 = {"entity_id": "S1-1", "business_name": None, "business_address": None}
    r2 = {"entity_id": "S3-5", "business_name": "Beta LLC", "business_address": "456 Market St"}
    feats = extractor.extract_pair_features(r1, r2, rank=3)
    feat_map = dict(zip(extractor.feature_names, feats))
    assert feat_map["is_source2"] == 0.0
    assert feat_map["is_source3"] == 1.0
    assert feat_map["blocking_rank"] == 3.0
    assert feat_map["name_exact_match"] == 0.0
    assert feat_map["addr_is_empty"] == 1.0


def test_extract_pairs_matrix():
    extractor = PairwiseFeatureExtractor()
    s1_rows = [
        {"entity_id": "S1-1", "business_name": "Acme Corp", "business_address": "100 Main St, 10001"},
        {"entity_id": "S1-1", "business_name": "Acme Corp", "business_address": "100 Main St, 10001"},
        {"entity_id": "S1-2", "business_name": "Beta LLC", "business_address": "200 Broad St, 90210"},
    ]
    target_rows = [
        {"entity_id": "S2-1", "business_name": "Acme", "business_address": "100 Main St, 10001"},
        {"entity_id": "S3-2", "business_name": "Acme Inc", "business_address": "100 Main St, 10001"},
        {"entity_id": "S2-3", "business_name": "Beta Co", "business_address": "200 Broad St, 90210"},
    ]
    ranks = [1, 2, 1]
    scores = [0.95, 0.75, 0.88]

    matrix = extractor.extract_pairs_matrix(s1_rows, target_rows, ranks=ranks, scores=scores)
    assert matrix.shape == (3, len(extractor.feature_names))

    # Validate each row matches single pair extraction
    for i in range(3):
        single = extractor.extract_pair_features(
            s1_rows[i], target_rows[i], rank=ranks[i], blocking_score=scores[i]
        )
        np.testing.assert_allclose(matrix[i], single, rtol=1e-5, atol=1e-5)

    # Empty input handling
    empty_mat = extractor.extract_pairs_matrix([], [], [], [])
    assert empty_mat.shape == (0, len(extractor.feature_names))


def test_name_in_address_cross_short_names():
    extractor = PairwiseFeatureExtractor()
    cross_idx = extractor.feature_names.index("name_in_address_cross")

    # Short name "St" should not spuriously match "Main Street" or "Austin"
    r1 = {"entity_id": "S1-1", "business_name": "St", "business_address": "500 Pine Rd"}
    r2 = {"entity_id": "S2-1", "business_name": "Bakery", "business_address": "123 Main Street, Austin TX"}
    feats1 = extractor.extract_pair_features(r1, r2)
    assert feats1[cross_idx] == 0.0

    # Short name "In" should not spuriously match "Main" or "Austin"
    r3 = {"entity_id": "S1-2", "business_name": "In", "business_address": "500 Pine Rd"}
    r4 = {"entity_id": "S2-2", "business_name": "Cafe", "business_address": "456 Austin Ave"}
    feats2 = extractor.extract_pair_features(r3, r4)
    assert feats2[cross_idx] == 0.0

    # Reverse direction: target short name against S1 address
    r5 = {"entity_id": "S1-3", "business_name": "Hardware", "business_address": "789 Main Street, Austin TX"}
    r6 = {"entity_id": "S2-3", "business_name": "St", "business_address": "100 Other Rd"}
    feats3 = extractor.extract_pair_features(r5, r6)
    assert feats3[cross_idx] == 0.0

    # Short name "Go" should not spuriously match words like "Mango"
    r7 = {"entity_id": "S1-4", "business_name": "Go", "business_address": "100 Elm St"}
    r8 = {"entity_id": "S2-4", "business_name": "Shop", "business_address": "123 Mango Street"}
    feats4 = extractor.extract_pair_features(r7, r8)
    assert feats4[cross_idx] == 0.0


def test_extract_pairs_matrix_mismatched_lengths():
    extractor = PairwiseFeatureExtractor()
    s1_rows = [{"entity_id": "S1-1", "business_name": "A", "business_address": "1"}]
    target_rows = [
        {"entity_id": "S2-1", "business_name": "B", "business_address": "2"},
        {"entity_id": "S2-2", "business_name": "C", "business_address": "3"},
    ]

    # Mismatched target_rows length
    with pytest.raises(ValueError, match="target_rows length"):
        extractor.extract_pairs_matrix(s1_rows, target_rows)

    # Mismatched ranks length
    with pytest.raises(ValueError, match="ranks length"):
        extractor.extract_pairs_matrix(s1_rows, [target_rows[0]], ranks=[1, 2])

    # Mismatched scores length
    with pytest.raises(ValueError, match="scores length"):
        extractor.extract_pairs_matrix(s1_rows, [target_rows[0]], scores=[0.5, 0.8])


def test_extract_pairs_matrix_cache_key_entity_id(monkeypatch):
    extractor = PairwiseFeatureExtractor()
    s1_rows = [
        {"entity_id": "S1-1", "business_name": "Acme", "business_address": "100 Main St"},
        {"entity_id": "S1-1", "business_name": "Acme", "business_address": "100 Main St"},
    ]
    # Distinct dictionary objects sharing entity_id
    t1 = {"entity_id": "S2-1", "business_name": "Target Store", "business_address": "200 Market St"}
    t2 = {"entity_id": "S2-1", "business_name": "Target Store", "business_address": "200 Market St"}
    assert t1 is not t2
    target_rows = [t1, t2]

    name_call_count = 0
    orig_norm_name = extractor._get_normalized_name

    def mock_norm_name(row):
        nonlocal name_call_count
        name_call_count += 1
        return orig_norm_name(row)

    monkeypatch.setattr(extractor, "_get_normalized_name", mock_norm_name)

    matrix = extractor.extract_pairs_matrix(s1_rows, target_rows)
    assert matrix.shape == (2, len(extractor.feature_names))
    # Normalized once for S1-1 and once for S2-1 -> 2 calls
    assert name_call_count == 2

    # Fallback when entity_id is absent
    r_no_id_1 = {"business_name": "Shop A", "business_address": "100 Main St"}
    r_no_id_2 = {"business_name": "Shop B", "business_address": "200 Main St"}
    mat_no_id = extractor.extract_pairs_matrix([r_no_id_1], [r_no_id_2])
    assert mat_no_id.shape == (1, len(extractor.feature_names))

