"""Run local completion checks without emitting raw message content."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .masking import mask_text


def main() -> None:
    data = pd.read_csv("data/messages.csv")
    demo_ids = pd.read_csv("data/mandatory_demo_ids.csv").iloc[:, 0].astype(str).tolist()
    missing_columns = {"message_id", "timestamp", "sender", "message"} - set(data.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
    ids = data["message_id"].astype(str)
    missing_ids = sorted(set(demo_ids) - set(ids))
    changed = (data["message"].astype(str) != data["message"].map(mask_text)).sum()
    print(f"Messages: {len(data)}")
    print(f"Mandatory IDs: {len(demo_ids)}; missing: {len(missing_ids)}")
    print(f"Privacy masking changed: {changed} messages")
    print(f"Trained model available: {Path('models/message_classifier.joblib').exists()}")
    print("PASS" if len(data) == 900 and len(demo_ids) == 15 and not missing_ids else "CHECK REQUIRED")


if __name__ == "__main__":
    main()
