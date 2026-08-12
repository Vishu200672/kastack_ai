"""Create explicitly marked weak labels to exercise the full local ML pipeline.

These labels originate from the rule baseline. They are not human ground truth and
must be reviewed before reporting model quality as an internship result.
"""
from __future__ import annotations

import pandas as pd


def main() -> None:
    path = "data/annotations.csv"
    data = pd.read_csv(path)
    data["label"] = data["label"].fillna("").astype("string").str.strip()
    labels = data["label"]
    empty = labels.eq("")
    data.loc[empty, "label"] = data.loc[empty, "suggested_category"]
    data["label_source"] = data.get("label_source", "unreviewed")
    data.loc[empty, "label_source"] = "weak_rule"
    data.to_csv(path, index=False)
    print(f"Added {int(empty.sum())} weak-rule labels to {path}.")
    print("Review label values and set label_source to 'human' before reporting evaluation as final quality.")


if __name__ == "__main__":
    main()
