"""Text and address normalization module for Entity Resolution.

Handles legal suffix stripping, Unicode flattening, punctuation cleaning,
and structured token/anchor extraction (street numbers, postal codes).
"""

from dataclasses import dataclass
import re
import unicodedata
from typing import List, Optional

LEGAL_SUFFIXES = {
    "inc", "incorporated", "llc", "corp", "corporation", "ltd", "limited",
    "pvt", "private", "co", "company", "llp", "sa", "sarl", "sas", "eurl",
    "gmbh", "plc", "enterprises", "services"
}

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

class TextNormalizer:
    def __init__(self) -> None:
        self.street_num_re = re.compile(r"\b(\d+[a-zA-Z]?)\b")
        self.postal_re = re.compile(r"\b(\d{5,6})\b")
        self.punct_re = re.compile(r"[^\w\s]")

    def normalize_name(self, name: Optional[str]) -> NormalizedName:
        if not name or str(name).strip() == "" or str(name) == "nan":
            return NormalizedName(raw="", clean_name="", tokens=[], char_3grams=[])
        s = str(name).strip()
        s_norm = unicodedata.normalize("NFKD", s).encode("ASCII", "ignore").decode("utf-8").lower()
        s_clean = self.punct_re.sub(" ", s_norm)
        raw_tokens = s_clean.split()
        filtered_tokens = [t for t in raw_tokens if t not in LEGAL_SUFFIXES and len(t) > 0]
        clean_name = " ".join(filtered_tokens) if filtered_tokens else " ".join(raw_tokens)
        
        compact = "".join(filtered_tokens)
        char_3grams = [compact[i:i+3] for i in range(len(compact) - 2)] if len(compact) >= 3 else [compact]
        return NormalizedName(raw=s, clean_name=clean_name, tokens=filtered_tokens, char_3grams=char_3grams)

    def normalize_address(self, addr: Optional[str]) -> NormalizedAddress:
        if not addr or str(addr).strip() == "" or str(addr) == "nan":
            return NormalizedAddress(raw="", clean_address="", street_number=None, postal_code=None, tokens=[])
        s = str(addr).strip()
        s_norm = unicodedata.normalize("NFKD", s).encode("ASCII", "ignore").decode("utf-8").lower()
        
        postal_match = self.postal_re.search(s_norm)
        postal_code = postal_match.group(1) if postal_match else None

        street_match = self.street_num_re.search(s_norm)
        street_number = street_match.group(1) if street_match else None

        s_clean = self.punct_re.sub(" ", s_norm)
        tokens = [t for t in s_clean.split() if len(t) > 1]
        return NormalizedAddress(raw=s, clean_address=" ".join(tokens), street_number=street_number, postal_code=postal_code, tokens=tokens)
