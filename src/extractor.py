"""Deterministic extraction of action fields and masked sensitive-data metadata."""
from __future__ import annotations

import re

from .masking import MASK

FIELD_PATTERNS = {
    "amount": re.compile(r"(?:₹|Rs\.?|INR|\$)\s?\d[\d,]*(?:\.\d{1,2})?", re.I),
    "date": re.compile(r"(?<!\d)(?:\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?|today|tomorrow|tonight|monday|tuesday|wednesday|thursday|friday|saturday|sunday)(?!\d)", re.I),
    "time": re.compile(r"\b(?:\d{1,2}(?::\d{2})?\s?(?:am|pm)|\d{1,2}:\d{2})\b", re.I),
}
SENSITIVE_PATTERNS = {
    "card_number": re.compile(r"(?<!\w)(?:\d[ -]?){15,18}\d(?!\w)"),
    "email": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
    "phone": re.compile(r"(?<!\w)(?:\+?\d{1,3}[ -]?)?\d{10}(?!\w)"),
    "transaction_id": re.compile(r"\b(?:txn|transaction|ref(?:erence)?)\s*(?:id|no|number)?\s*[:#-]?\s*[A-Za-z0-9-]{5,}\b", re.I),
}
DEADLINE_PATTERN = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
ACTION_PATTERN = re.compile(r"\b(reply|submit|review|send|prepare|complete|update|upload|pay)\b", re.I)
ACTION_OBJECTS = {
    "reply": re.compile(r"\breply\s+to\s+(?:the\s+)?(.+?)(?:\s+(?:by|before|on)\s+\d{4}-\d{2}-\d{2}|[.!?]|$)", re.I),
    "submit": re.compile(r"\bsubmit\s+(?:the\s+)?(.+?)(?:\s+(?:by|before|on)\s+\d{4}-\d{2}-\d{2}|[.!?]|$)", re.I),
    "review": re.compile(r"\breview\s+(?:the\s+)?(.+?)(?:\s+(?:by|before|on)\s+\d{4}-\d{2}-\d{2}|[.!?]|$)", re.I),
}


def extract_fields(message: object) -> dict[str, object]:
    """Return action data and structured sensitive-data types without raw values."""
    text = "" if message is None else str(message)
    result: dict[str, object] = {}
    sensitive_data: list[dict[str, str]] = []
    used_spans: list[tuple[int, int]] = []

    # Card precedence ensures long numeric values cannot become phone values.
    for name in ("card_number", "email", "phone", "transaction_id"):
        for match in SENSITIVE_PATTERNS[name].finditer(text):
            if any(start < match.end() and match.start() < end for start, end in used_spans):
                continue
            sensitive_data.append({"type": name, "value": MASK})
            used_spans.append(match.span())
    if sensitive_data:
        result["sensitive_data"] = sensitive_data

    deadline = list(dict.fromkeys(DEADLINE_PATTERN.findall(text)))
    if deadline:
        result["deadline"] = deadline
    if action_match := ACTION_PATTERN.search(text):
        action = action_match.group(1).lower()
        result["action"] = action
        object_pattern = ACTION_OBJECTS.get(action)
        if object_pattern and (object_match := object_pattern.search(text)):
            result["object"] = object_match.group(1).strip(" .,!")

    for name, pattern in FIELD_PATTERNS.items():
        if name == "date" and deadline:
            continue
        values = list(dict.fromkeys(match.group(0) for match in pattern.finditer(text)))
        if values:
            result[name] = values
    return result
