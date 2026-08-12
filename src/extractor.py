"""Regex-based extraction of practical message fields."""
from __future__ import annotations

import re

from .masking import MASK, mask_text

def _normalize_digits(value: str) -> str:
    return re.sub(r"\D", "", value)


def _luhn_valid(value: str) -> bool:
    digits = _normalize_digits(value)
    if len(digits) < 13 or len(digits) > 19:
        return False
    total = 0
    reverse_digits = digits[::-1]
    for index, char in enumerate(reverse_digits):
        digit = int(char)
        if index % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def _is_card_number(value: str) -> bool:
    return bool(_luhn_valid(value))


FIELD_PATTERNS = {
    "card_number": re.compile(r"\b(?:\d[ -]?){12,19}\b"),
    "transaction_id": re.compile(r"\b(?:txn|transaction|ref(?:erence)?)\s*(?:id|no|number)?\s*[:#-]?\s*([A-Za-z0-9-]{5,})\b", re.I),
    "email": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
    "phone": re.compile(r"(?<!\w)(?:\+?\d{1,3}[ -]?)?(?:\d[ -]?){9,13}\b"),
    "amount": re.compile(r"(?:₹|Rs\.?|INR|\$)\s?\d[\d,]*(?:\.\d{1,2})?", re.I),
    "date": re.compile(r"(?<!\d)(?:\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?|today|tomorrow|tonight|monday|tuesday|wednesday|thursday|friday|saturday|sunday)(?!\d)", re.I),
    "time": re.compile(r"\b(?:\d{1,2}(?::\d{2})?\s?(?:am|pm)|\d{1,2}:\d{2})\b", re.I),
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
    used_spans: list[tuple[int, int]] = []

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
            if any(start < match.end() and match.start() < end for start, end in used_spans):
                continue
            value = match.group(1) if name == "transaction_id" and match.lastindex else match.group(0)
            if name == "card_number" and not _is_card_number(value):
                continue

            if name == "transaction_id":
                safe_value = MASK
            elif name in {"email", "phone", "card_number"}:
                safe_value = MASK
            else:
                safe_value = value

            if safe_value not in values:
                values.append(safe_value)
            if name in {"transaction_id", "email", "phone", "card_number"}:
                result.setdefault("sensitive_data", []).append({"type": name, "value": safe_value})
            used_spans.append((match.start(), match.end()))
        if values:
            result[name] = values
    return result
