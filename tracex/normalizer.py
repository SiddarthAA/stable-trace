import re
from typing import List

from .loader import Requirement

# Domain-agnostic synonym normalization for modal verbs
_SYNONYMS = [
    (r"\bshall\b", "must"),
    (r"\bshould\b", "must"),
    (r"\bwill\b", "must"),
    (r"\bprovides?\b", "supply"),
    (r"\bensures?\b", "guarantee"),
    (r"\bsupports?\b", "handle"),
]

# Compiled patterns for speed
_SYNONYM_PATTERNS = [(re.compile(p, re.IGNORECASE), r) for p, r in _SYNONYMS]

# Match ID-like tokens at the start or inline: SLR-001, HLR.1.2, REQ_007:
_ID_PATTERN = re.compile(r"\b[A-Za-z]{2,6}[-._]?\d[\d._-]*\b:?")
_WHITESPACE = re.compile(r"\s+")
_NON_WORD = re.compile(r"[^\w\s]")


def _normalize(text: str) -> str:
    text = text.lower()
    text = _ID_PATTERN.sub(" ", text)
    text = _NON_WORD.sub(" ", text)
    for pattern, replacement in _SYNONYM_PATTERNS:
        text = pattern.sub(replacement, text)
    text = _WHITESPACE.sub(" ", text).strip()
    return text


def normalize_requirements(requirements: List[Requirement]) -> None:
    """Set cleaned_text on each Requirement in-place."""
    for req in requirements:
        req.cleaned_text = _normalize(req.description)
