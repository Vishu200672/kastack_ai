"""Generate a Git-safe, masked structured report for the 15 required demo IDs."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .extractor import extract_fields
from .masking import mask_text
from .predict import predict_messages


def main() -> None:
    messages = pd.read_csv("data/messages.csv")
    demo_ids = pd.read_csv("data/mandatory_demo_ids.csv").iloc[:, 0].astype(str).tolist()
    selected = messages[messages["message_id"].astype(str).isin(demo_ids)].copy()
    if len(selected) != len(demo_ids):
        raise ValueError("One or more mandatory demonstration IDs are missing.")
    predictions = predict_messages(selected["message"].tolist())
    report = pd.DataFrame({
        "message_id": selected["message_id"].astype(str),
        "masked_message": selected["message"].map(mask_text),
        "predicted_category": [item["category"] for item in predictions],
        "model_confidence": [item["confidence"] for item in predictions],
        "extracted_fields": [json.dumps(extract_fields(message), ensure_ascii=False) for message in selected["message"]],
        "reason": [item["reason"] for item in predictions],
    })
    report = report.set_index("message_id").reindex(demo_ids).reset_index()
    Path("outputs").mkdir(exist_ok=True)
    report.to_csv("outputs/masked_demo_report.csv", index=False)
    print("Created outputs/masked_demo_report.csv with 15 masked demonstration messages.")


if __name__ == "__main__":
    main()
