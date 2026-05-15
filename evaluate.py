#!/usr/bin/env python3
"""
Measures redaction recall against the synthetic ground truth corpus.

Usage:
    python evaluate.py               # full run: loads BERT model + regex (~2 min)
    python evaluate.py --regex-only  # fast run: regex patterns only, no model
"""

import json
import os
import sys
import threading

# main.py sets HF_HOME/TRANSFORMERS_CACHE at import time, so import it first
from main import redact_text, load_ner_model_in_background, loader_queue

GROUND_TRUTH = os.path.join(os.path.dirname(__file__), "pii-generate-testset", "pii_test_ground_truth.json")

PII_TYPES = ["NAME", "ORG", "EMAIL", "PHONE", "ADDRESS", "SSN", "CREDIT_CARD", "PASSPORT"]


def load_model():
    print("Loading NER model (this may take ~30s on first run)...", flush=True)
    t = threading.Thread(target=load_ner_model_in_background, daemon=True)
    t.start()
    while True:
        kind, msg = loader_queue.get()
        print(f"  {msg}", flush=True)
        if kind in ("done", "error"):
            break
    t.join(timeout=1)


def evaluate(skip_ml: bool = False) -> None:
    if not skip_ml:
        load_model()

    with open(GROUND_TRUTH, encoding="utf-8") as f:
        ground_truth = json.load(f)

    counts = {t: {"tp": 0, "fn": 0} for t in PII_TYPES}
    total_tags = 0

    for i, entry in enumerate(ground_truth):
        redacted = redact_text(entry["text"])
        total_tags += redacted.count("[REDACTED_")

        for item in entry["items"]:
            pii_type = item["type"]
            if item["text"] not in redacted:
                counts[pii_type]["tp"] += 1  # successfully removed
            else:
                counts[pii_type]["fn"] += 1  # still present = missed

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/1000 lines...", flush=True)

    # Results table
    total_gt = sum(c["tp"] + c["fn"] for c in counts.values())
    overall_tp = sum(c["tp"] for c in counts.values())

    print("\n" + "=" * 52)
    print(f"{'Type':<14} {'Recall':>8}  {'Redacted':>9}  {'Missed':>7}  {'Total':>6}")
    print("-" * 52)
    for t in PII_TYPES:
        tp, fn = counts[t]["tp"], counts[t]["fn"]
        total = tp + fn
        recall = tp / total if total else 0
        print(f"{t:<14} {recall:>7.1%}  {tp:>9}  {fn:>7}  {total:>6}")
    print("-" * 52)
    overall_recall = overall_tp / total_gt if total_gt else 0
    overall_fn = total_gt - overall_tp
    print(f"{'OVERALL':<14} {overall_recall:>7.1%}  {overall_tp:>9}  {overall_fn:>7}  {total_gt:>6}")
    print("=" * 52)
    print(f"\nTotal redaction tags in output : {total_tags}")
    print(f"Total GT PII items             : {total_gt}")
    print(f"Over-redaction ratio           : {total_tags / total_gt:.2f}x")


if __name__ == "__main__":
    evaluate(skip_ml="--regex-only" in sys.argv)
