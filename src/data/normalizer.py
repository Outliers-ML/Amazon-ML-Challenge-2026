"""Text and address normalization module for Entity Resolution.

Handles legal suffix stripping, Unicode flattening, punctuation cleaning,
and structured token/anchor extraction (street numbers, postal codes).
"""

from dataclasses import dataclass
import math
import re
import unicodedata
from typing import List, Optional

LEGAL_SUFFIXES = {
    "inc", "incorporated", "llc", "corp", "corporation", "ltd", "limited",
    "pvt", "private", "co", "company", "llp", "sa", "sarl", "sas", "eurl",
    "gmbh", "plc", "enterprises", "services"
}

NULL_STRINGS = {"", "nan", "none", "null", "<na>", "n/a"}

_SOUNDEX_MAP = {
    "B": "1", "F": "1", "P": "1", "V": "1",
    "C": "2", "G": "2", "J": "2", "K": "2", "Q": "2", "S": "2", "X": "2", "Z": "2",
    "D": "3", "T": "3",
    "L": "4",
    "M": "5", "N": "5",
    "R": "6",
}


def compute_soundex(word: str) -> str:
    """Compute American Soundex phonetic code for a token string."""
    if not word:
        return ""
    clean = re.sub(r"[^A-Za-z]", "", str(word)).upper()
    if not clean:
        return ""
    first_letter = clean[0]
    encoded = [_SOUNDEX_MAP.get(ch, "0") for ch in clean[1:]]
    collapsed = []
    prev = _SOUNDEX_MAP.get(first_letter, "0")
    for d in encoded:
        if d != "0" and d != prev:
            collapsed.append(d)
        prev = d
    digits = "".join(collapsed)
    return (first_letter + digits + "000")[:4]


@dataclass
class NormalizedName:
    raw: str
    clean_name: str
    tokens: List[str]
    char_3grams: List[str]

@dataclass
class NormalizedAddress:
    raw: str
    clean_address: str
    street_number: Optional[str]
    postal_code: Optional[str]
    tokens: List[str]

def _is_null(val: Optional[str]) -> bool:
    if val is None:
        return True
    if isinstance(val, float) and math.isnan(val):
        return True
    return str(val).strip().lower() in NULL_STRINGS

class TextNormalizer:
    def __init__(self) -> None:
        self.street_num_re = re.compile(r"\b(\d+[a-zA-Z]?)\b")
        self.postal_re = re.compile(r"\b(\d{5,6})\b")
        self.punct_re = re.compile(r"[^\w\s]")

    def normalize_name(self, name: Optional[str]) -> NormalizedName:
        if _is_null(name):
            return NormalizedName(raw="", clean_name="", tokens=[], char_3grams=[])
        s = str(name).strip()
        s_norm = unicodedata.normalize("NFKD", s).encode("ASCII", "ignore").decode("utf-8").lower()
        if not s_norm.strip():
            return NormalizedName(raw=s, clean_name="", tokens=[], char_3grams=[])

        s_clean = self.punct_re.sub(" ", s_norm)
        raw_tokens = s_clean.split()
        if not raw_tokens:
            return NormalizedName(raw=s, clean_name="", tokens=[], char_3grams=[])

        filtered_tokens = [t for t in raw_tokens if t not in LEGAL_SUFFIXES]
        final_tokens = filtered_tokens if filtered_tokens else raw_tokens
        clean_name = " ".join(final_tokens)
        
        compact = "".join(final_tokens)
        if not compact:
            char_3grams: List[str] = []
        elif len(compact) < 3:
            char_3grams = [compact]
        else:
            char_3grams = [compact[i:i+3] for i in range(len(compact) - 2)]

        return NormalizedName(raw=s, clean_name=clean_name, tokens=final_tokens, char_3grams=char_3grams)

    def normalize_address(self, addr: Optional[str]) -> NormalizedAddress:
        if _is_null(addr):
            return NormalizedAddress(raw="", clean_address="", street_number=None, postal_code=None, tokens=[])
        s = str(addr).strip()
        s_norm = unicodedata.normalize("NFKD", s).encode("ASCII", "ignore").decode("utf-8").lower()
        if not s_norm.strip():
            return NormalizedAddress(raw=s, clean_address="", street_number=None, postal_code=None, tokens=[])

        # Identify postal code first
        postal_matches = list(self.postal_re.finditer(s_norm))
        if postal_matches:
            postal_match = postal_matches[-1]
            postal_code: Optional[str] = postal_match.group(1)
            p_start, p_end = postal_match.span()
        else:
            postal_code = None
            p_start, p_end = -1, -1

        # Extract street number excluding postal code span
        standard_nums: List[str] = []
        prefixed_nums: List[str] = []
        for m in self.street_num_re.finditer(s_norm):
            m_start, m_end = m.span()
            if postal_code is not None and max(m_start, p_start) < min(m_end, p_end):
                continue
            if m_start >= 2 and s_norm[m_start - 1] == "-" and s_norm[m_start - 2].isalpha():
                prefixed_nums.append(m.group(1))
            else:
                standard_nums.append(m.group(1))

        if standard_nums:
            street_number: Optional[str] = standard_nums[0]
        elif prefixed_nums:
            street_number = prefixed_nums[0]
        else:
            street_number = None

        s_clean = self.punct_re.sub(" ", s_norm)
        tokens = s_clean.split()
        return NormalizedAddress(
            raw=s,
            clean_address=" ".join(tokens),
            street_number=street_number,
            postal_code=postal_code,
            tokens=tokens,
        )
