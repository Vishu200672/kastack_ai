# KaStack Message Understanding & Action Extraction

A local, privacy-safe NLP pipeline for classifying 900 fictional messages, extracting action-oriented fields, and demonstrating 15 mandatory message IDs. Raw messages are never sent to external AI services.

## Deliverables

- Generated Git-safe structured output: `outputs/masked_demo_report.csv`
- Evaluation metrics: `outputs/evaluation_metrics.json`
- Video guide: `DEMO_SCRIPT.md`
- Streamlit demo: [KaStack Message Understanding · Streamlit](https://kastackaigit-fadhdpyss98smknhgn7iqo.streamlit.app/)
- Loom video: add the recording URL here before submission: **TBD**

## Run locally

```powershell
python -m pip install -r requirements.txt
python -m streamlit run main.py
```

Run completion checks and regenerate the Git-safe demo file:

```powershell
python -m src.generate_demo_output
python -m src.validate_project
```

## How message classification works

The final taxonomy is `payment`, `meeting`, `task`, `issue`, `information`, and `other`. The original dataset has no verified issue examples, so high-signal failure wording is handled by a narrow transparent issue rule guard until human-reviewed issue labels are available.

Because the supplied dataset has no ground-truth labels, the project first creates an annotation sheet with a transparent keyword/phrase baseline. The classifier uses **TF-IDF** features (unigrams and bigrams) and **Logistic Regression**. Training uses a stratified 80/20 held-out split, reports accuracy, macro F1, and weighted F1, then refits on all available labels for the application.

The Streamlit UI presents the predicted category and the top-class model probability as **model confidence**. This is a probability estimate, not a certainty.

## How tasks and events are extracted

Classification and extraction are deliberately separate. Regex-based deterministic extraction identifies:

- `action` — e.g. reply, submit, review
- `object` — e.g. client email
- `deadline` — ISO dates such as `2026-09-04`
- `event_type`, `event_date`, and `event_time` for meeting/event wording
- amount, date, time, email, phone, and transaction reference patterns

For example, “Please reply to the client email by 2026-09-04” produces action `reply`, object `client email`, and deadline `2026-09-04`.

## Sensitive-information detection and masking

Before a message is displayed, exported, or written to the annotation sheet, common sensitive-looking patterns are replaced with `***`:

- Email addresses and phone numbers
- Card/account-like numeric strings
- OTP, PIN, password, and passcode patterns
- Transaction/reference ID patterns

Structured sensitive-data records also include `risk_level` and a `recommended_action`. Card numbers and transaction IDs are high risk; email addresses and phone numbers are medium risk.

The original dataset, annotations, and trained model are ignored by Git. Only the generated report with masked text is allowed under `outputs/`.

## Assumptions and limitations

- Messages are fictional, English-language, and follow the provided CSV schema.
- Regex extraction is intentionally narrow; unusual phrasing may not yield an action/object/deadline.
- The current labels have `label_source=weak_rule`, meaning they were seeded from the rule baseline to test the end-to-end pipeline.
- Therefore, displayed held-out metrics measure consistency with weak labels and **must not be presented as human-ground-truth performance**. Manually review labels, set `label_source` to `human`, and retrain before making a final accuracy claim.
- Privacy masking is pattern-based and cannot guarantee detection of every possible sensitive value.

## AI-tool usage declaration

AI coding assistance was used to help scaffold, explain, and refine the local Python/Streamlit implementation. No raw dataset messages were sent to an external AI service. Classification, extraction, masking, training, and inference execute locally using Python, pandas, scikit-learn, and Streamlit.

## Submission steps

1. Create a GitHub repository and push this project. Confirm `data/messages.csv`, `data/mandatory_demo_ids.csv`, `data/annotations.csv`, and `models/` are not included. A public repository is safe because the app automatically switches to its pre-generated **masked-only hosted mode** when the private dataset is absent.
2. Deploy from that repository to Streamlit Community Cloud (or another cloud host) and replace the **TBD** cloud URL above.
3. Record a Loom video using `DEMO_SCRIPT.md`, then replace the **TBD** Loom URL above.
4. Run `python -m src.generate_demo_output` and `python -m src.validate_project` before submitting.
