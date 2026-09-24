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
