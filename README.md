# PII NER Redactor

A desktop GUI tool that detects and redacts personally identifiable information (PII) from text files using a combination of BERT-based named entity recognition and regex patterns.

## Features

- **ML-based NER** — uses `dbmdz/bert-large-cased-finetuned-conll03-english` (BERT fine-tuned on CoNLL-03) to detect names, organizations, and locations
- **Regex patterns** — catches emails, phone numbers, SSNs, credit card numbers, passport numbers, and street addresses
- **Side-by-side GUI** — load a `.txt` file, redact with one click, and save the result
- **Non-blocking model load** — the BERT model loads in the background so the UI is responsive immediately

## Requirements

- Python 3.9+
- See `requirements.txt`

## Setup

```bash
pip install -r requirements.txt
python main.py
```

The model (~1.3 GB) is downloaded automatically on first run into a local `hf_cache/` directory.

## Usage

1. Run `python main.py`
2. Wait for the model to finish loading (status shown bottom-right), or start typing/loading text immediately — regex redaction works right away
3. Click **Load .TXT** to open a file, or type directly into the left panel
4. Click **Redact** — the right panel shows the redacted output
5. Click **Save Redacted** to write the result to disk

Redacted spans are labelled by type, e.g. `[REDACTED_PER]`, `[REDACTED_EMAIL]`, `[REDACTED_SSN]`.

## Evaluation

Run `evaluate.py` to measure recall across all 8 entity types on the synthetic corpus:

```bash
python evaluate.py               # loads BERT model + regex (~2 min)
python evaluate.py --regex-only  # regex patterns only, no model needed
```

The script reports per-type recall, overall recall, and an over-redaction ratio against the 1,000-line ground truth (`pii-generate-testset/pii_test_ground_truth.json`).

**Results (full run):**

| Type | Recall |
|------|--------|
| NAME | 100.0% |
| ORG | 100.0% |
| EMAIL | 100.0% |
| PHONE | 73.5% |
| ADDRESS | 100.0% |
| SSN | 100.0% |
| CREDIT_CARD | 100.0% |
| PASSPORT | 70.9% |
| **Overall** | **93.0%** |

The corpus is synthetic, so structured types (EMAIL, SSN, CREDIT_CARD, ADDRESS) are evaluated against formats the regex patterns were explicitly written to handle — recall on those reflects pattern coverage rather than generalisation. PHONE and PASSPORT fall short because the generator produces some format variants outside the regex scope. For real-world documents, results will vary.

## Test Set

`pii-generate-testset/` contains a script that generates a synthetic evaluation corpus:

```bash
cd pii-generate-testset
python generate_pii_testset.py
```

This produces:
- `pii_test_corpus.txt` — 1000 lines of text with embedded fake PII
- `pii_test_ground_truth.json` — ground truth labels for each line
- `pii_test_redacted.txt` — pre-run redacted output for reference

All PII in the test set is entirely synthetic (randomly generated names, addresses, SSNs, etc.).
