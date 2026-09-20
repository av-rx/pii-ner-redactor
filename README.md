# PII NER Redactor

A desktop GUI tool that detects and redacts personally identifiable information (PII) from text files using a combination of BERT-based named entity recognition and regex patterns.

## Features

- **ML-based NER** — uses `dbmdz/bert-large-cased-finetuned-conll03-english` (BERT fine-tuned on CoNLL-03) to detect names, organizations, and locations
- **Regex patterns** — catches emails, phone numbers, SSNs, credit card numbers, passport numbers, and street addresses
- **Long-document safe** — input is split on line boundaries into chunks under BERT's 512-token limit, so names past the first page are still tagged
- **Side-by-side GUI** — load a `.txt` file, redact with one click, and save the result
- **Non-blocking** — the BERT model loads in the background so the UI is responsive immediately, and redaction runs on a worker thread so the window never freezes during inference

## Requirements

- Python 3.9+
- See `requirements.txt`

## Setup

```bash
pip install -r requirements.txt
python main.py
```

The model (~1.3 GB) is downloaded automatically on first run into `hf_cache/` next to `main.py`.

## Usage

1. Run `python main.py`
2. Wait for the model to finish loading (status shown bottom-right), or start typing/loading text immediately — regex redaction works right away
3. Click **Load .TXT** to open a file, or type directly into the left panel
4. Click **Redact** — the right panel shows the redacted output. BERT-large on CPU takes roughly half a second per line
5. Click **Save Redacted** to write the result to disk

Redacted spans are labelled by type, e.g. `[REDACTED_PER]`, `[REDACTED_EMAIL]`, `[REDACTED_SSN]`. If the model fails during a redaction the tool falls back to regex-only output and tells you so.

## How redaction is composed

1. The text is split into chunks on line boundaries (a single line over the token budget is split on whitespace).
2. Each chunk goes through the NER pipeline; entity spans are replaced with placeholders using a running offset so later spans stay aligned.
3. Six regex patterns then run over the result in a fixed order: EMAIL, SSN, CREDIT_CARD, ADDRESS, PHONE, PASSPORT. The specific shapes go before the permissive phone pattern so a spaced card number is tagged as a card, not a phone. A short number left directly before a `[REDACTED_LOC]` placeholder is treated as a house number and folded into `[REDACTED_ADDRESS]`.

## Evaluation

Run `evaluate.py` to measure recall across all 8 entity types on the synthetic corpus:

```bash
python evaluate.py                  # loads BERT model + regex (several minutes on CPU)
python evaluate.py --regex-only     # regex patterns only, no model needed
python evaluate.py --save-redacted  # also rewrite pii-generate-testset/pii_test_redacted.txt
```

The script reports two recall figures per type against the 1,000-line ground truth (`pii-generate-testset/pii_test_ground_truth.json`, 3,980 items):

- **Recall** — the item's exact string no longer appears in the output. Lenient: partial removal, or removal by a different pattern, still counts.
- **Typed** — the string is gone *and* a placeholder of the matching type appears on that line.

It also prints an over-redaction ratio (placeholders emitted per ground-truth item). Precision is not measured directly; the model also tags cities, countries and misc entities that are not in the ground truth, so the ratio overstates false positives.

**Results (full run):**

| Type | Recall | Typed |
|------|--------|-------|
| NAME | 100.0% | 97.1% |
| ORG | 100.0% | 100.0% |
| EMAIL | 100.0% | 99.8% |
| PHONE | 99.6% | 99.6% |
| ADDRESS | 100.0% | 99.0% |
| SSN | 100.0% | 100.0% |
| CREDIT_CARD | 100.0% | 100.0% |
| PASSPORT | 100.0% | 100.0% |
| **Overall** | **99.9%** | **99.4%** |

Over-redaction ratio: 2.20x. The two PHONE misses are a six-digit value the generator emits; the phone pattern requires at least seven digits so that amounts like `1500.00` are not redacted.

The corpus is synthetic, so structured types are evaluated against formats the regex patterns were written to handle — recall on those reflects pattern coverage rather than generalisation. For real-world documents, results will vary.

Earlier version of the patterns (before the phone, passport, address and ordering fixes) scored 93.0% overall on the same corpus, with PHONE at 73.5% and PASSPORT at 70.9%.

## Tests

```bash
python -m pytest tests/
```

The tests cover the regex patterns, pattern ordering, the chunking helper, and the NER splice logic using a fake pipeline. No model download is needed.

## Test Set

`pii-generate-testset/` contains a script that generates a synthetic evaluation corpus:

```bash
cd pii-generate-testset
python generate_pii_testset.py
```

This produces:
- `pii_test_corpus.txt` — 1000 lines of text with embedded fake PII
- `pii_test_ground_truth.json` — ground truth labels for each line
- `pii_test_redacted.txt` — redacted output for reference (regenerate with `python evaluate.py --save-redacted`)

The generator is seeded (`random.seed(42)`), so regenerating gives byte-identical files. All PII in the test set is entirely synthetic (randomly generated names, addresses, SSNs, etc.).
