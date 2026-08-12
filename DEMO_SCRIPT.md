# 1–2 Minute Demonstration Script

1. Open the cloud-hosted Streamlit app. State that the project processes 900 fictional, unlabeled messages locally and does not send raw message content to external AI services.
2. Point to **Model Evaluation**. Explain that the model is TF-IDF plus Logistic Regression; the displayed accuracy and F1 are held-out pipeline metrics based on weak rule labels and require human-reviewed labels for a final quality claim.
3. Open `MSG_0002`. Show its category is **TASK** and point out any extracted action or deadline.
4. Open `MSG_0001`. Show **MEETING** classification.
5. Open `MSG_0003`. Show **INFORMATION** classification.
6. Open `MSG_0013`. Show **PAYMENT** classification and confirm every sensitive-looking value appears as `***`.
7. Open `MSG_0012`. Show the task category and extracted action fields.
8. Open `MSG_0007` to show structured extraction: action `reply`, object `client email`, deadline `2026-09-04`.
9. Download the masked demo report and state that it contains only masked messages and structured output.

10. Open **Safe classification examples**. Demonstrate the synthetic **ISSUE** example, explain it is a transparent rule guard because the original dataset had no verified issue examples, and show the **OTHER — unclear information** example as an intentionally uncertain/missing-information case.

Close with: “The classifier determines category, deterministic rules extract useful action fields, and privacy masking runs before display or export.”
