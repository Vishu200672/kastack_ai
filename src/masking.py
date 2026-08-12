"""Mask sensitive-looking values before any output is written or displayed."""
from __future__ import annotations

import re

MASK = "***"

PATTERNS = (
    (re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I), MASK),
    (re.compile(r"(?<!\w)(?:\+?\d{1,3}[ -]?)?(?:\d[ -]?){9,13}\b"), MASK),
    (re.compile(r"\b(?:\d[ -]?){12,19}\b"), MASK),
    (re.compile(r"\b(?:otp|pin|password|passcode)\s*[:=-]?\s*\S+", re.I), MASK),
    (re.compile(r"\b(?:upi|txn|transaction|ref(?:erence)?)\s*(?:id|no|number)?\s*[:#-]?\s*[A-Za-z0-9-]{5,}\b", re.I), MASK),
)


def mask_text(value: object) -> str:
    """Return text with common sensitive-looking tokens replaced."""
    text = "" if value is None else str(value)
    for pattern, replacement in PATTERNS:
        text = pattern.sub(replacement, text)
    return text
