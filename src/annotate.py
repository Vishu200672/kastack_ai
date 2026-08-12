"""Create a safe annotation template; annotators only see masked messages."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .masking import mask_text
from .predict import rule_prediction


def build_template(input_path: Path, output_path: Path) -> None:
    data = pd.read_csv(input_path)
    required = {"message_id", "message"}
    if not required.issubset(data.columns):
        raise ValueError(f"Input CSV must contain {sorted(required)}")
    existing_labels: dict[object, str] = {}
    if output_path.exists():
        existing = pd.read_csv(output_path)
        if {"message_id", "label"}.issubset(existing.columns):
            existing_labels = (
                existing[["message_id", "label"]]
                .dropna(subset=["label"])
                .assign(label=lambda frame: frame["label"].astype(str).str.strip())
                .query("label != ''")
                .set_index("message_id")["label"]
                .to_dict()
            )

    rows = []
    for _, row in data.iterrows():
        suggestion, _ = rule_prediction(row["message"])
        rows.append({
            "message_id": row["message_id"],
            "masked_message": mask_text(row["message"]),
            "suggested_category": suggestion,
            "label": existing_labels.get(row["message_id"], ""),
            "label_source": "human" if row["message_id"] in existing_labels else "unreviewed",
        })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    result.to_csv(output_path, index=False)
    reviewed = result["label"].fillna("").astype(str).str.strip().ne("").sum()
    print(f"Created {output_path} with {len(result)} messages.")
    print(f"Preserved human-reviewed labels: {reviewed}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/messages.csv")
    parser.add_argument("--output", default="data/annotations.csv")
    args = parser.parse_args()
    build_template(Path(args.input), Path(args.output))
