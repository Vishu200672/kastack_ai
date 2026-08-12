"""Local prediction helpers: ML classifier plus transparent issue safety guard."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import joblib

CATEGORIES = ("payment", "meeting", "task", "issue", "information", "other")
RULES = {
    "payment": ("payment", "paid", "invoice", "refund", "transaction", "bank account", "account number", "card number", "upi", "billing", "salary", "fee", "receipt"),
    "meeting": ("meeting", "call", "zoom", "calendar", "schedule", "scheduled", "appointment", "seminar", "workshop", "conference", "briefing", "webinar", "catch-up", "join", "availability"),
    "task": ("please submit", "please review", "please complete", "please send", "please reply", "please prepare", "please update", "need you to", "don't forget", "deadline", "due", "complete", "submit", "review", "prepare", "send", "reply", "finish", "upload"),
    "issue": ("error", "failed", "failure", "problem", "unable", "not working", "doesn't work", "cannot", "can't", "broken", "complaint", "outage"),
    "information": ("fyi", "for your information", "announcement", "reminder", "confirmed", "notice", "update", "available", "status", "information", "details"),
}


def _contains_pattern(text: str, pattern: str) -> bool:
    return re.search(r"(?<!\w)" + re.escape(pattern) + r"(?!\w)", text) is not None


def rule_prediction(message: object) -> tuple[str, str]:
    text = str(message).lower()
    scores = {category: 0 for category in CATEGORIES}
    matched_terms = {category: [] for category in RULES}
    for category, terms in RULES.items():
        for term in terms:
            if _contains_pattern(text, term):
                scores[category] += 1
                matched_terms[category].append(term)
    category = max(scores, key=scores.get)
    if scores[category] == 0:
        return "other", "No category-specific pattern was detected."
    return category, "Matched patterns: " + ", ".join(matched_terms[category][:3]) + "."


def load_model(model_path: str | Path = "models/message_classifier.joblib"):
    path = Path(model_path)
    return joblib.load(path) if path.exists() else None


def predict_messages(messages: Iterable[object], model_path: str | Path = "models/message_classifier.joblib") -> list[dict[str, object]]:
    values = list(messages)
    model = load_model(model_path)
    model_labels = model.predict(values) if model is not None else [rule_prediction(value)[0] for value in values]
    probabilities = model.predict_proba(values) if model is not None and hasattr(model, "predict_proba") else None
    result = []
    for index, (value, model_label) in enumerate(zip(values, model_labels)):
        rule_label, rule_reason = rule_prediction(value)
        # The original training set has no verified issue labels. This narrow,
        # high-signal rule guard prevents operational failures being obscured.
        is_issue_guard = rule_label == "issue"
        label = "issue" if is_issue_guard else str(model_label)
        confidence = None if is_issue_guard else float(probabilities[index].max()) if probabilities is not None else None
        reason = f"Issue safety rule: {rule_reason}" if is_issue_guard else "Predicted by trained classifier."
        result.append({"category": label, "reason": reason, "confidence": confidence, "prediction_source": "rule_guard" if is_issue_guard else "trained_model"})
    return result
