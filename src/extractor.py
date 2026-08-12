"""Regex-based extraction of practical message fields."""
from __future__ import annotations

import re

from .masking import MASK, mask_text

FIELD_PATTERNS = {
    "amount": re.compile(r"(?:₹|Rs\.?|INR|\$)\s?\d[\d,]*(?:\.\d{1,2})?", re.I),
    "date": re.compile(r"(?<!\d)(?:\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?|today|tomorrow|tonight|monday|tuesday|wednesday|thursday|friday|saturday|sunday)(?!\d)", re.I),
    "time": re.compile(r"\b(?:\d{1,2}(?::\d{2})?\s?(?:am|pm)|\d{1,2}:\d{2})\b", re.I),
    "email": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
    "phone": re.compile(r"(?<!\w)(?:\+?\d{1,3}[ -]?)?(?:\d[ -]?){9,13}\b"),
    "transaction_id": re.compile(r"\b(?:txn|transaction|ref(?:erence)?)\s*(?:id|no|number)?\s*[:#-]?\s*([A-Za-z0-9-]{5,})\b", re.I),
}
DEADLINE_PATTERN = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
ACTION_PATTERN = re.compile(r"\b(reply|submit|review|send|prepare|complete|update|upload|pay)\b", re.I)
ACTION_OBJECTS = {
    "reply": re.compile(r"\breply\s+to\s+(?:the\s+)?(.+?)(?:\s+(?:by|before|on)\s+\d{4}-\d{2}-\d{2}|[.!?]|$)", re.I),
    "submit": re.compile(r"\bsubmit\s+(?:the\s+)?(.+?)(?:\s+(?:by|before|on)\s+\d{4}-\d{2}-\d{2}|[.!?]|$)", re.I),
    "review": re.compile(r"\breview\s+(?:the\s+)?(.+?)(?:\s+(?:by|before|on)\s+\d{4}-\d{2}-\d{2}|[.!?]|$)", re.I),
}


def extract_fields(message: object) -> dict[str, list[str]]:
    """Extract action fields and mask values that should never be exposed."""
    text = "" if message is None else str(message)
    result: dict[str, list[str]] = {}

    deadline = list(dict.fromkeys(DEADLINE_PATTERN.findall(text)))
    if deadline:
        result["deadline"] = deadline
    action_match = ACTION_PATTERN.search(text)
    if action_match:
        action = action_match.group(1).lower()
        result["action"] = [action]
        object_pattern = ACTION_OBJECTS.get(action)
        if object_pattern and (object_match := object_pattern.search(text)):
            result["object"] = [object_match.group(1).strip(" .,!")]

    for name, pattern in FIELD_PATTERNS.items():
        if name == "date" and deadline:
            continue
        values = []
        for match in pattern.finditer(text):
            value = match.group(1) if name == "transaction_id" and match.lastindex else match.group(0)
            # Transaction IDs are captured as an ID-only group, so mask them
            # directly instead of relying on phrase-based masking.
            safe_value = MASK if name == "transaction_id" else mask_text(value) if name in {"email", "phone"} else value
            if safe_value not in values:
                values.append(safe_value)
        if values:
            result[name] = values
    return result
