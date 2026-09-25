"""Text and address normalization module for Entity Resolution.

Handles legal suffix stripping, Unicode flattening, punctuation cleaning,
Double Metaphone phonetic transliteration, CEDEX/BP sanitization,
and structured token/anchor extraction (street numbers, postal codes).
"""

from dataclasses import dataclass
import math
import re
import unicodedata
from typing import List, Optional, Tuple

LEGAL_SUFFIXES = {
    "inc", "incorporated", "llc", "corp", "corporation", "ltd", "limited",
    "pvt", "private", "co", "company", "llp", "sa", "sarl", "sas", "eurl",
    "gmbh", "plc", "enterprises", "services", "anonyme", "limitee"
}

CANONICAL_REPLACEMENTS = [
    (re.compile(r"\b(pvt|private)\s+(ltd|limited)\b", re.IGNORECASE), "pvt ltd"),
    (re.compile(r"\b(limited\s+liability\s+company)\b", re.IGNORECASE), "llc"),
    (re.compile(r"\b(limited\s+liability\s+partnership)\b", re.IGNORECASE), "llp"),
    (re.compile(r"\b(public\s+limited\s+company)\b", re.IGNORECASE), "plc"),
    (re.compile(r"\b(societe\s+a\s+responsabilite\s+limitee)\b", re.IGNORECASE), "sarl"),
    (re.compile(r"\b(societe\s+par\s+actions\s+simplifiee)\b", re.IGNORECASE), "sas"),
    (re.compile(r"\b(societe\s+anonyme)\b", re.IGNORECASE), "sa"),
    (re.compile(r"\b(societe\s+civile\s+immobiliere)\b", re.IGNORECASE), "sci"),
    (re.compile(r"\b(entreprise\s+unipersonnelle\s+a\s+responsabilite\s+limitee)\b", re.IGNORECASE), "eurl"),
    (re.compile(r"\bincorporated\b", re.IGNORECASE), "inc"),
    (re.compile(r"\bcorporation\b", re.IGNORECASE), "corp"),
    (re.compile(r"\blimited\b", re.IGNORECASE), "ltd"),
    (re.compile(r"\bprivate\b", re.IGNORECASE), "pvt"),
    (re.compile(r"\bcompany\b", re.IGNORECASE), "co"),
]

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


def compute_double_metaphone(word: str) -> Tuple[str, str]:
    """Compute Lawrence Philips' Double Metaphone (primary, secondary) phonetic codes.

    Pure-Python implementation with special handling for Indian transliteration variants
    (e.g., 'Lakshmi' vs 'Laxmi', 'Choudhary' vs 'Chowdhury', 'Bh' vs 'B') and French
    silent endings / consonants (e.g., 'Renault', 'Societe').
    """
    if not word:
        return ("", "")
    w_str = str(word).strip().split()
    if not w_str:
        return ("", "")
    raw = w_str[0]
    s = unicodedata.normalize("NFKD", raw).encode("ASCII", "ignore").decode("utf-8").upper()
    s = re.sub(r"[^A-Z]", "", s)
    if not s:
        return ("", "")

    length = len(s)
    pos = 0
    # Skip silent initial letter combinations
    if length >= 2 and s[:2] in ("GN", "KN", "PN", "WR", "PS"):
        pos = 1

    primary: List[str] = []
    secondary: List[str] = []

    # Initial X sounds like S
    if pos == 0 and s[0] == "X":
        primary.append("S")
        secondary.append("S")
        pos = 1
    elif pos == 0 and s[0] in "AEIOUY":
        primary.append("A")
        secondary.append("A")
        pos = 1

    while pos < length and (len(primary) < 4 or len(secondary) < 4):
        ch = s[pos]
        if ch in "AEIOUY":
            pos += 1
        elif ch == "B":
            if s[pos:pos+2] == "BH":
                primary.append("P")
                secondary.append("P")
                pos += 2
            elif s[pos:pos+2] == "BB":
                primary.append("P")
                secondary.append("P")
                pos += 2
            elif pos == length - 1 and pos > 0 and s[pos-1] == "M":
                # Silent B after M at end of word (dumb, thumb)
                pos += 1
            else:
                primary.append("P")
                secondary.append("P")
                pos += 1
        elif ch == "C":
            if s[pos:pos+2] == "CZ":
                primary.append("S")
                secondary.append("X")
                pos += 2
            elif s[pos:pos+2] == "CH":
                primary.append("X")
                secondary.append("X")
                pos += 2
            elif s[pos:pos+3] == "CIA":
                primary.append("X")
                secondary.append("X")
                pos += 3
            elif s[pos:pos+2] == "CC" and pos + 2 < length and s[pos+2] in ("I", "E", "H"):
                primary.append("X")
                secondary.append("KS")
                pos += 3
            elif pos + 1 < length and s[pos+1] in ("I", "E", "Y"):
                primary.append("S")
                secondary.append("S")
                pos += 1
            elif s[pos:pos+2] in ("CK", "CG", "CQ"):
                primary.append("K")
                secondary.append("K")
                pos += 2
            else:
                primary.append("K")
                secondary.append("K")
                if pos + 1 < length and s[pos+1] in ("C", "K", "Q"):
                    pos += 2
                else:
                    pos += 1
        elif ch == "D":
            if s[pos:pos+2] == "DG" and pos + 2 < length and s[pos+2] in ("I", "E", "Y"):
                primary.append("J")
                secondary.append("J")
                pos += 2
            elif s[pos:pos+2] in ("DT", "DD"):
                primary.append("T")
                secondary.append("T")
                pos += 2
            elif s[pos:pos+2] == "DH":
                primary.append("T")
                secondary.append("T")
                pos += 2
            elif pos == length - 1 and pos > 0 and s[pos-1] in "AEIOUY" and s.endswith(("ARD", "ERD", "IED")):
                pos += 1
            else:
                primary.append("T")
                secondary.append("T")
                pos += 1
        elif ch == "F":
            primary.append("F")
            secondary.append("F")
            pos += 2 if pos + 1 < length and s[pos+1] == "F" else 1
        elif ch == "G":
            if s[pos:pos+2] == "GH":
                if pos == 0 or (pos + 2 < length and s[pos+2] in "AEIOUY"):
                    primary.append("K")
                    secondary.append("K")
                pos += 2
            elif s[pos:pos+2] == "GN":
                primary.append("K")
                secondary.append("N")
                pos += 2
            elif pos + 1 < length and s[pos+1] in ("I", "E", "Y"):
                primary.append("J")
                secondary.append("K")
                pos += 1
            else:
                primary.append("K")
                secondary.append("K")
                pos += 2 if pos + 1 < length and s[pos+1] == "G" else 1
        elif ch == "H":
            if (pos > 0 and s[pos-1] in "AEIOUY" and pos + 1 < length and s[pos+1] in "AEIOUY") or (pos == 0 and pos + 1 < length and s[pos+1] in "AEIOUY"):
                primary.append("H")
                secondary.append("H")
            pos += 1
        elif ch == "J":
            primary.append("J")
            secondary.append("H")
            pos += 2 if pos + 1 < length and s[pos+1] == "J" else 1
        elif ch == "K":
            if s[pos:pos+3] == "KSH":
                primary.append("KS")
                secondary.append("KS")
                pos += 3
            elif s[pos:pos+2] == "KH":
                primary.append("K")
                secondary.append("K")
                pos += 2
            elif pos + 1 < length and s[pos+1] == "K":
                primary.append("K")
                secondary.append("K")
                pos += 2
            else:
                primary.append("K")
                secondary.append("K")
                pos += 1
        elif ch == "L":
            primary.append("L")
            secondary.append("L")
            pos += 2 if pos + 1 < length and s[pos+1] == "L" else 1
        elif ch == "M":
            primary.append("M")
            secondary.append("M")
            pos += 2 if pos + 1 < length and s[pos+1] == "M" else 1
        elif ch == "N":
            primary.append("N")
            secondary.append("N")
            pos += 2 if pos + 1 < length and s[pos+1] == "N" else 1
        elif ch == "P":
            if s[pos:pos+2] == "PH":
                primary.append("F")
                secondary.append("F")
                pos += 2
            elif pos + 1 < length and s[pos+1] in ("P", "B"):
                primary.append("P")
                secondary.append("P")
                pos += 2
            else:
                primary.append("P")
                secondary.append("P")
                pos += 1
        elif ch == "Q":
            primary.append("K")
            secondary.append("K")
            pos += 2 if pos + 1 < length and s[pos+1] == "Q" else 1
        elif ch == "R":
            primary.append("R")
            secondary.append("R")
            pos += 2 if pos + 1 < length and s[pos+1] == "R" else 1
        elif ch == "S":
            if s[pos:pos+2] == "SH":
                primary.append("X")
                secondary.append("X")
                pos += 2
            elif s[pos:pos+3] in ("SIO", "SIA"):
                primary.append("S")
                secondary.append("X")
                pos += 3
            elif s[pos:pos+2] == "SC":
                if pos + 2 < length and s[pos+2] in ("I", "E", "Y"):
                    primary.append("S")
                    secondary.append("S")
                    pos += 3
                else:
                    primary.append("SK")
                    secondary.append("SK")
                    pos += 2
            elif pos + 1 < length and s[pos+1] == "S":
                primary.append("S")
                secondary.append("S")
                pos += 2
            else:
                primary.append("S")
                secondary.append("S")
                pos += 1
        elif ch == "T":
            if s[pos:pos+4] == "TION" or s[pos:pos+3] in ("TIA", "TIO"):
                primary.append("X")
                secondary.append("X")
                pos += 3
            elif s[pos:pos+2] == "TH":
                primary.append("0")
                secondary.append("T")
                pos += 2
            elif s[pos:pos+3] == "TCH":
                primary.append("X")
                secondary.append("X")
                pos += 3
            elif pos + 1 < length and s[pos+1] == "T":
                primary.append("T")
                secondary.append("T")
                pos += 2
            elif pos == length - 1 and s.endswith(("AULT", "OT", "ET")):
                secondary.append("T")
                pos += 1
            else:
                primary.append("T")
                secondary.append("T")
                pos += 1
        elif ch == "V":
            primary.append("F")
            secondary.append("F")
            pos += 2 if pos + 1 < length and s[pos+1] == "V" else 1
        elif ch == "W":
            if pos > 0 and s[pos-1] in "AEIOUY" and (pos == length - 1 or s[pos+1] not in "AEIOUY"):
                pos += 1
            elif pos + 1 < length and s[pos+1] in "AEIOUY":
                primary.append("A")
                secondary.append("F")
                pos += 1
            else:
                pos += 1
        elif ch == "X":
            if pos == length - 1 and s.endswith(("AUX", "EAUX", "OUX")):
                secondary.append("KS")
                pos += 1
            else:
                primary.append("KS")
                secondary.append("KS")
                pos += 1
        elif ch == "Z":
            if s[pos:pos+2] == "ZH":
                primary.append("J")
                secondary.append("J")
                pos += 2
            else:
                primary.append("S")
                secondary.append("TS")
                pos += 2 if pos + 1 < length and s[pos+1] == "Z" else 1
        else:
            pos += 1

    p_str = "".join(primary)[:4]
    s_str = "".join(secondary)[:4]
    return (p_str, s_str)


@dataclass
class NormalizedName:
    raw: str
    clean_name_stripped: str
    canonical_name: str
    tokens: List[str]
    char_3grams: List[str]
    metaphone_primary: str = ""
    metaphone_secondary: str = ""
    clean_name: str = ""

    def __post_init__(self) -> None:
        if not self.clean_name:
            self.clean_name = self.clean_name_stripped


@dataclass
class NormalizedAddress:
    raw: str
    clean_address: str
    street_number: Optional[str]
    postal_code: Optional[str]
    tokens: List[str]
    cedex_flag: bool = False


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
        self.cedex_detect_re = re.compile(r"\b(cedex|b\.?p\.?|c\.?s\.?)(?!\w)", re.IGNORECASE)
        self.cedex_strip_re = re.compile(r"\b(cedex|b\.?p\.?|c\.?s\.?)\s*(?:\d{1,4}\b)?", re.IGNORECASE)

    def normalize_name(self, name: Optional[str]) -> NormalizedName:
        if _is_null(name):
            return NormalizedName(
                raw="",
                clean_name_stripped="",
                canonical_name="",
                tokens=[],
                char_3grams=[],
                metaphone_primary="",
                metaphone_secondary="",
            )
        s = str(name).strip()
        s_norm = unicodedata.normalize("NFKD", s).encode("ASCII", "ignore").decode("utf-8").lower()
        if not s_norm.strip():
            return NormalizedName(
                raw=s,
                clean_name_stripped="",
                canonical_name="",
                tokens=[],
                char_3grams=[],
                metaphone_primary="",
                metaphone_secondary="",
            )

        # 1. Canonical track: standardize legal corporate suffixes into canonical forms
        s_canon = self.punct_re.sub(" ", s_norm)
        for pattern, repl in CANONICAL_REPLACEMENTS:
            s_canon = pattern.sub(repl, s_canon)
        canonical_tokens = s_canon.split()
        canonical_name = " ".join(canonical_tokens)

        # 2. Stripped track: remove corporate suffixes for clean indexing
        s_clean = self.punct_re.sub(" ", s_norm)
        raw_tokens = s_clean.split()
        if not raw_tokens:
            return NormalizedName(
                raw=s,
                clean_name_stripped="",
                canonical_name="",
                tokens=[],
                char_3grams=[],
                metaphone_primary="",
                metaphone_secondary="",
            )

        filtered_tokens = [t for t in raw_tokens if t not in LEGAL_SUFFIXES]
        final_tokens = filtered_tokens if filtered_tokens else raw_tokens
        clean_name_stripped = " ".join(final_tokens)

        # 3. Char 3-grams
        compact = "".join(final_tokens)
        if not compact:
            char_3grams: List[str] = []
        elif len(compact) < 3:
            char_3grams = [compact]
        else:
            char_3grams = [compact[i:i+3] for i in range(len(compact) - 2)]

        # 4. Double Metaphone for primary brand token
        if final_tokens:
            met_p, met_s = compute_double_metaphone(final_tokens[0])
        else:
            met_p, met_s = "", ""

        return NormalizedName(
            raw=s,
            clean_name_stripped=clean_name_stripped,
            canonical_name=canonical_name,
            tokens=final_tokens,
            char_3grams=char_3grams,
            metaphone_primary=met_p,
            metaphone_secondary=met_s,
        )

    def normalize_address(self, addr: Optional[str]) -> NormalizedAddress:
        if _is_null(addr):
            return NormalizedAddress(
                raw="",
                clean_address="",
                street_number=None,
                postal_code=None,
                tokens=[],
                cedex_flag=False,
            )
        s = str(addr).strip()
        s_norm = unicodedata.normalize("NFKD", s).encode("ASCII", "ignore").decode("utf-8").lower()
        if not s_norm.strip():
            return NormalizedAddress(
                raw=s,
                clean_address="",
                street_number=None,
                postal_code=None,
                tokens=[],
                cedex_flag=False,
            )

        # Check CEDEX / BP / CS flag
        cedex_flag = bool(self.cedex_detect_re.search(s_norm))

        # Strip CEDEX, Boite Postale (BP), and Case Speciale (CS) routes BEFORE extracting numbers
        s_addr = self.cedex_strip_re.sub(" ", s_norm)

        # Identify postal code first (5 or 6 digits)
        postal_matches = list(self.postal_re.finditer(s_addr))
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
        for m in self.street_num_re.finditer(s_addr):
            m_start, m_end = m.span()
            if postal_code is not None and max(m_start, p_start) < min(m_end, p_end):
                continue
            if m_start >= 2 and s_addr[m_start - 1] == "-" and s_addr[m_start - 2].isalpha():
                prefixed_nums.append(m.group(1))
            else:
                standard_nums.append(m.group(1))

        if standard_nums:
            street_number: Optional[str] = standard_nums[0]
        elif prefixed_nums:
            street_number = prefixed_nums[0]
        else:
            street_number = None

        s_clean = self.punct_re.sub(" ", s_addr)
        tokens = s_clean.split()
        return NormalizedAddress(
            raw=s,
            clean_address=" ".join(tokens),
            street_number=street_number,
            postal_code=postal_code,
            tokens=tokens,
            cedex_flag=cedex_flag,
        )
