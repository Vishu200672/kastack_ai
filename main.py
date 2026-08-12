"""Privacy-safe Streamlit demonstration for KaStack message understanding."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.extractor import extract_fields
from src.masking import mask_text
from src.predict import predict_messages

st.set_page_config(page_title="KaStack Message Understanding", layout="wide")
st.title("KaStack Message Understanding & Action Extraction")
st.caption("Local TF-IDF + Logistic Regression classifier · Sensitive-looking values are masked before display or export.")

data_path = Path("data/messages.csv")
demo_path = Path("data/mandatory_demo_ids.csv")
metrics_path = Path("outputs/evaluation_metrics.json")
report_path = Path("outputs/masked_demo_report.csv")

if metrics_path.exists():
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    st.subheader("Model Evaluation")
    first, second, third, fourth = st.columns(4)
    first.metric("Accuracy", f"{metrics['accuracy']:.2%}")
    second.metric("Macro F1", f"{metrics['macro_f1']:.2%}")
    third.metric("Weighted F1", f"{metrics['weighted_f1']:.2%}")
    fourth.metric("Model", "TF-IDF + Logistic Regression")
    st.caption("Metrics use a stratified held-out test split. They are pipeline-only metrics until labels are human-reviewed.")

if data_path.exists():
    data = pd.read_csv(data_path)
    required = {"message_id", "timestamp", "sender", "message"}
    if not required.issubset(data.columns):
        st.error(f"messages.csv needs these columns: {', '.join(sorted(required))}")
        st.stop()
    demo_ids = pd.read_csv(demo_path).iloc[:, 0].tolist() if demo_path.exists() else data["message_id"].head(15).tolist()
    demo = data[data["message_id"].isin(demo_ids)].copy()
    predictions = predict_messages(demo["message"].tolist())
    demo["message"] = demo["message"].map(mask_text)
    demo["predicted_category"] = [item["category"] for item in predictions]
    demo["model_confidence"] = [item["confidence"] for item in predictions]
    demo["reason"] = [item["reason"] for item in predictions]
    demo["extracted_fields"] = [extract_fields(value) for value in data.loc[demo.index, "message"]]
    demo = demo.set_index("message_id").reindex(demo_ids).reset_index()
elif report_path.exists():
    # Public-cloud mode: only the pre-generated masked report is present.
    demo = pd.read_csv(report_path)
    demo["extracted_fields"] = demo["extracted_fields"].map(json.loads)
else:
    st.error("No private dataset or masked demonstration report is available.")
    st.stop()

st.subheader("Mandatory demonstration messages")
for _, row in demo.iterrows():
    with st.expander(f"Message {row['message_id']} — {str(row['predicted_category']).upper()}"):
        st.write(row["message"] if "message" in row else row["masked_message"])
        if pd.notna(row["model_confidence"]):
            st.write(f"Model confidence: {row['model_confidence']:.1%}")
        st.write("Extracted fields:", row["extracted_fields"] or "None")
        st.caption(row["reason"])

st.download_button("Download masked demo report", demo.to_csv(index=False), "masked_demo_report.csv", "text/csv")
