"""Train and evaluate a local classifier from human-reviewed annotations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def train(annotation_path: Path, model_path: Path, metrics_path: Path) -> None:
    data = pd.read_csv(annotation_path).dropna(subset=["label"])
    data["label"] = data["label"].astype(str).str.strip().str.lower()
    data = data[data["label"] != ""]
    if data["label"].nunique() < 2:
        raise ValueError("Provide human-reviewed labels for at least two categories before training.")
    counts = data["label"].value_counts()
    too_small = counts[counts < 2]
    if not too_small.empty:
        raise ValueError("Each category needs at least two labels: " + ", ".join(too_small.index))

    text_column = "masked_message" if "masked_message" in data.columns else "message"
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
        ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
    ])
    train_text, test_text, train_label, test_label = train_test_split(
        data[text_column].fillna(""), data["label"], test_size=0.2,
        random_state=42, stratify=data["label"],
    )
    model.fit(train_text, train_label)
    prediction = model.predict(test_text)
    metrics = {
        "evaluation_note": "Held-out results are valid only when labels have been human reviewed. Weak labels are useful for an end-to-end pipeline check, not a quality claim.",
        "labeled_messages": int(len(data)),
        "train_messages": int(len(train_text)),
        "test_messages": int(len(test_text)),
        "accuracy": round(float(accuracy_score(test_label, prediction)), 4),
        "macro_f1": round(float(f1_score(test_label, prediction, average="macro", zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(test_label, prediction, average="weighted", zero_division=0)), 4),
        "class_distribution": {str(label): int(count) for label, count in counts.items()},
        "classification_report": classification_report(test_label, prediction, output_dict=True, zero_division=0),
    }

    # Refit on all verified labels after the held-out evaluation.
    model.fit(data[text_column].fillna(""), data["label"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved evaluated model trained on {len(data)} labels to {model_path}")
    print(f"Held-out accuracy: {metrics['accuracy']:.2%}; macro F1: {metrics['macro_f1']:.2%}; weighted F1: {metrics['weighted_f1']:.2%}")
    print(f"Saved evaluation metrics to {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", default="data/annotations.csv")
    parser.add_argument("--model", default="models/message_classifier.joblib")
    parser.add_argument("--metrics", default="outputs/evaluation_metrics.json")
    args = parser.parse_args()
    train(Path(args.annotations), Path(args.model), Path(args.metrics))
