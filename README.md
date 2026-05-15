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
