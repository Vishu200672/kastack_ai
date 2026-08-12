"""Deterministic extraction of tasks, events, and privacy-safe metadata."""
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
RISK_ACTIONS = {
    "card_number": ("high", "Do not display, export, or share the original value."),
    "transaction_id": ("high", "Do not display, export, or share the original value."),
    "email": ("medium", "Mask before display or export; disclose only when necessary."),
    "phone": ("medium", "Mask before display or export; disclose only when necessary."),
}
DEADLINE_PATTERN = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
ACTION_PATTERN = re.compile(r"\b(reply|submit|review|send|prepare|complete|update|upload|pay)\b", re.I)
EVENT_PATTERN = re.compile(r"\b(meeting|call|appointment|seminar|workshop|conference|briefing|webinar|orientation|stand-?up)\b", re.I)
ACTION_OBJECTS = {
    "reply": re.compile(r"\breply\s+to\s+(?:the\s+)?(.+?)(?:\s+(?:by|before|on)\s+\d{4}-\d{2}-\d{2}|[.!?]|$)", re.I),
    "submit": re.compile(r"\bsubmit\s+(?:the\s+)?(.+?)(?:\s+(?:by|before|on)\s+\d{4}-\d{2}-\d{2}|[.!?]|$)", re.I),
    "review": re.compile(r"\breview\s+(?:the\s+)?(.+?)(?:\s+(?:by|before|on)\s+\d{4}-\d{2}-\d{2}|[.!?]|$)", re.I),
}


def extract_fields(message: object) -> dict[str, object]:
    text = "" if message is None else str(message)
    result: dict[str, object] = {}
    sensitive_data, used_spans = [], []
    for name in ("card_number", "email", "phone", "transaction_id"):
        for match in SENSITIVE_PATTERNS[name].finditer(text):
            if any(start < match.end() and match.start() < end for start, end in used_spans):
                continue
            risk_level, action = RISK_ACTIONS[name]
            sensitive_data.append({"type": name, "value": MASK, "risk_level": risk_level, "recommended_action": action})
            used_spans.append(match.span())
    if sensitive_data:
        result["sensitive_data"] = sensitive_data

    deadline = list(dict.fromkeys(DEADLINE_PATTERN.findall(text)))
    if deadline:
        result["deadline"] = deadline[0]
    if action_match := ACTION_PATTERN.search(text):
        action = action_match.group(1).lower()
        result["action"] = action
        object_pattern = ACTION_OBJECTS.get(action)
        if object_pattern and (object_match := object_pattern.search(text)):
            result["object"] = object_match.group(1).strip(" .,!")
    if event_match := EVENT_PATTERN.search(text):
        result["event_type"] = event_match.group(1).lower()
        dates = FIELD_PATTERNS["date"].findall(text)
        times = FIELD_PATTERNS["time"].findall(text)
        if dates:
            result["event_date"] = dates[0]
        if times:
            result["event_time"] = times[0]

    for name, pattern in FIELD_PATTERNS.items():
        if name == "date" and deadline:
            continue
        values = list(dict.fromkeys(match.group(0) for match in pattern.finditer(text)))
        if values:
            result[name] = values
    return result
