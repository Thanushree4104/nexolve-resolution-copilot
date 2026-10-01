import re
from dataclasses import dataclass, field


@dataclass
class RedactionResult:
    text: str
    counts: dict = field(default_factory=dict)

    @property
    def found(self) -> bool:
        return bool(self.counts)


# Order matters: more specific patterns run first, so a 12-digit Aadhaar number
# is not mistaken for something else.
_PATTERNS = [
    ("EMAIL", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "<EMAIL>"),
    ("PAN", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"), "<PAN>"),
    ("AADHAAR", re.compile(r"(?<!\d)\d{4}[ -]?\d{4}[ -]?\d{4}(?!\d)"), "<AADHAAR>"),
    ("CARD", re.compile(r"(?<!\d)(?:\d[ -]?){13,16}(?!\d)"), "<CARD>"),
    (
        "ACCOUNT",
        re.compile(
            r"(?i)(\b(?:account|acct|a/c|konto)(?:\s*(?:number|no\.?|nummer|id|#))?\s*[:#]?\s*)"
            r"\d[\d-]{4,}\d"
        ),
        r"\1<ACCOUNT>",
    ),
    # Indian mobile: optional +91 / 91 / 0 prefix, then 10 digits starting 6-9
    ("PHONE", re.compile(r"(?<!\d)(?:\+?91[\s-]?|0)?[6-9]\d{4}[\s-]?\d{5}(?!\d)"), "<PHONE>"),
    # Other international numbers, e.g. +49 170 1234567
    ("PHONE", re.compile(r"(?<![\w+])\+\d{1,3}(?:[\s-]?\d{2,5}){2,4}(?!\d)"), "<PHONE>"),
]


def redact(text: str) -> RedactionResult:
    counts: dict = {}
    for label, pattern, replacement in _PATTERNS:
        text, n = pattern.subn(replacement, text)
        if n:
            counts[label] = counts.get(label, 0) + n
    return RedactionResult(text=text, counts=counts)