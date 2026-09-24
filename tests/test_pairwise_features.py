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
    feats = extractor.extract_pair_features(r1, r2, rank=1)
    assert len(feats) == len(extractor.feature_names)
    assert len(feats) >= 15
    # Name jaro winkler should be very high
    assert feats[extractor.feature_names.index("name_jaro_winkler")] > 0.85
    # Street number match should be +1.0
    assert feats[extractor.feature_names.index("addr_street_num_status")] == 1.0
    # Is source 2
    assert feats[extractor.feature_names.index("is_source2")] == 1.0
    # Blocking rank
    assert feats[extractor.feature_names.index("blocking_rank")] == 1.0


def test_conflicting_street_numbers():
    extractor = PairwiseFeatureExtractor()
    r1 = {"entity_id": "S1-1", "business_name": "Subway", "business_address": "105 Elm St, 28655"}
    r2 = {"entity_id": "S2-2", "business_name": "Subway", "business_address": "502 Oak St, 28655"}
    feats = extractor.extract_pair_features(r1, r2, rank=1)
    # Conflict should evaluate to -1.0
    assert feats[extractor.feature_names.index("addr_street_num_status")] == -1.0
    # Postal should evaluate to 1.0
    assert feats[extractor.feature_names.index("addr_postal_code_status")] == 1.0


def test_missing_address_components():
    extractor = PairwiseFeatureExtractor()
    r1 = {"entity_id": "S1-1", "business_name": "Alpha Corp", "business_address": ""}
    r2 = {"entity_id": "S2-2", "business_name": "Alpha", "business_address": "123 Main St"}
    feats = extractor.extract_pair_features(r1, r2, rank=2)
    assert feats[extractor.feature_names.index("addr_is_empty")] == 1.0
    assert feats[extractor.feature_names.index("addr_street_num_status")] == 0.0


def test_source3_and_none_values():
    extractor = PairwiseFeatureExtractor()
    r1 = {"entity_id": "S1-1", "business_name": None, "business_address": None}
    r2 = {"entity_id": "S3-5", "business_name": "Beta LLC", "business_address": "456 Market St"}
    feats = extractor.extract_pair_features(r1, r2, rank=3)
    assert feats[extractor.feature_names.index("is_source2")] == 0.0
    assert feats[extractor.feature_names.index("blocking_rank")] == 3.0
    assert feats[extractor.feature_names.index("name_exact_match")] == 0.0
    assert feats[extractor.feature_names.index("addr_is_empty")] == 1.0
