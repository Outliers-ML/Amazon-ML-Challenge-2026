from src.data.normalizer import TextNormalizer

def test_name_clean_and_legal_suffix():
    norm = TextNormalizer()
    res1 = norm.normalize_name("Holloway Peak Inc Seafood")
    assert "inc" not in res1.clean_name.split()
    assert "holloway" in res1.clean_name
    assert "seafood" in res1.clean_name

def test_unicode_and_french():
    norm = TextNormalizer()
    res = norm.normalize_name("Léarning Center SARL")
    assert res.clean_name == "learning center"

def test_address_components():
    norm = TextNormalizer()
    addr = norm.normalize_address("105 ELM ST, MORGANTON, NC 28655")
    assert addr.street_number == "105"
    assert addr.postal_code == "28655"

def test_empty_and_nan_inputs():
    norm = TextNormalizer()
    res_none = norm.normalize_name(None)
    assert res_none.clean_name == ""
    assert res_none.tokens == []
    assert res_none.char_3grams == []

    res_nan = norm.normalize_name("nan")
    assert res_nan.clean_name == ""
    assert res_nan.tokens == []

    addr_none = norm.normalize_address(None)
    assert addr_none.clean_address == ""
    assert addr_none.street_number is None
    assert addr_none.postal_code is None

    addr_nan = norm.normalize_address("nan")
    assert addr_nan.clean_address == ""
    assert addr_nan.street_number is None
    assert addr_nan.postal_code is None

def test_char_3grams():
    norm = TextNormalizer()
    res = norm.normalize_name("Acme")
    assert "acm" in res.char_3grams
    assert "cme" in res.char_3grams

def test_french_address_postal_and_street_number():
    norm = TextNormalizer()
    addr = norm.normalize_address("75001 Paris, 10 Rue de la Paix")
    assert addr.postal_code == "75001"
    assert addr.street_number == "10"
    assert "75001" in addr.tokens
    assert "10" in addr.tokens

def test_address_postal_code_only_no_street_number():
    norm = TextNormalizer()
    addr = norm.normalize_address("High Point, NC 27262")
    assert addr.postal_code == "27262"
    assert addr.street_number is None
    assert "27262" in addr.tokens

def test_indian_pin_code_and_street_number():
    norm = TextNormalizer()
    addr = norm.normalize_address("G-3/571, Gulmohar Colony, Bhopal, 462039")
    assert addr.postal_code == "462039"
    assert addr.street_number == "571"
    assert "462039" in addr.tokens

def test_case_insensitive_null_variants():
    norm = TextNormalizer()
    null_variants = ["NaN", "NAN", "None", "null", "<na>", "<NA>"]
    for variant in null_variants:
        res_name = norm.normalize_name(variant)
        assert res_name.clean_name == ""
        assert res_name.tokens == []
        assert res_name.char_3grams == []

        res_addr = norm.normalize_address(variant)
        assert res_addr.clean_address == ""
        assert res_addr.street_number is None
        assert res_addr.postal_code is None
        assert res_addr.tokens == []

def test_suffix_only_names_fallback_and_no_empty_3gram():
    norm = TextNormalizer()
    res_llc = norm.normalize_name("LLC")
    assert res_llc.clean_name == "llc"
    assert res_llc.tokens == ["llc"]
    assert "" not in res_llc.char_3grams
    assert res_llc.char_3grams == ["llc"]

    res_inc = norm.normalize_name("Inc")
    assert res_inc.clean_name == "inc"
    assert res_inc.tokens == ["inc"]
    assert "" not in res_inc.char_3grams
    assert res_inc.char_3grams == ["inc"]

    res_sarl = norm.normalize_name("SARL")
    assert res_sarl.clean_name == "sarl"
    assert res_sarl.tokens == ["sarl"]
    assert "" not in res_sarl.char_3grams
    assert res_sarl.char_3grams == ["sar", "arl"]

    # Blank/punctuation only
    res_blank = norm.normalize_name("---")
    assert res_blank.clean_name == ""
    assert res_blank.tokens == []
    assert res_blank.char_3grams == []

def test_preserve_single_character_tokens_in_address():
    norm = TextNormalizer()
    addr1 = norm.normalize_address("1 Apple Park Way")
    assert addr1.street_number == "1"
    assert "1" in addr1.tokens
    assert addr1.clean_address == "1 apple park way"

    addr2 = norm.normalize_address("Block A")
    assert "a" in addr2.tokens
    assert addr2.clean_address == "block a"
