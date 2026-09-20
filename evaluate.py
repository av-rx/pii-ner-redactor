#!/usr/bin/env python3
"""
Measures redaction recall against the synthetic ground truth corpus.

Usage:
    python evaluate.py                  # full run: loads BERT model + regex (several minutes on CPU)
    python evaluate.py --regex-only     # fast run: regex patterns only, no model
    python evaluate.py --save-redacted  # also rewrite pii-generate-testset/pii_test_redacted.txt

Two recall columns are printed:
    Recall  an item counts as recalled if its exact string is absent from the output
            (lenient: partial removal, or removal by the wrong pattern, still counts)
    Typed   the string is absent AND a placeholder of the matching type appears on that line
            (closer to "redacted as the right kind of thing")
"""

import json
import os
import sys
import threading

# main.py sets HF_HOME at import time, so import it first
from main import redact_text, load_ner_model_in_background, loader_queue

TESTSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pii-generate-testset")
GROUND_TRUTH = os.path.join(TESTSET_DIR, "pii_test_ground_truth.json")
REDACTED_OUT = os.path.join(TESTSET_DIR, "pii_test_redacted.txt")

PII_TYPES = ["NAME", "ORG", "EMAIL", "PHONE", "ADDRESS", "SSN", "CREDIT_CARD", "PASSPORT"]
# ground-truth type -> placeholder the tool emits for it (NAME comes from the model's PER label)
TYPE_TO_TAG = {t: f"[REDACTED_{'PER' if t == 'NAME' else t}]" for t in PII_TYPES}


def load_model():
    print("Loading NER model (about a minute from cache on CPU; longer on first download)...", flush=True)
    t = threading.Thread(target=load_ner_model_in_background, daemon=True)
    t.start()
    while True:
        kind, msg = loader_queue.get()
        print(f"  {msg}", flush=True)
        if kind in ("done", "error"):
            break
    t.join(timeout=1)


def evaluate(skip_ml: bool = False, save_redacted: bool = False) -> None:
    if not skip_ml:
        load_model()

    with open(GROUND_TRUTH, encoding="utf-8") as f:
        ground_truth = json.load(f)
    n_lines = len(ground_truth)

    counts = {t: {"tp": 0, "typed": 0, "fn": 0} for t in PII_TYPES}
    total_tags = 0
    outputs = []

    for i, entry in enumerate(ground_truth):
        redacted = redact_text(entry["text"])
        outputs.append(redacted)
        total_tags += redacted.count("[REDACTED_")

        for item in entry["items"]:
            pii_type = item["type"]
            if item["text"] not in redacted:
                counts[pii_type]["tp"] += 1  # successfully removed
                if TYPE_TO_TAG[pii_type] in redacted:
                    counts[pii_type]["typed"] += 1
            else:
                counts[pii_type]["fn"] += 1  # still present = missed

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{n_lines} lines...", flush=True)

    # Results table
    total_gt = sum(c["tp"] + c["fn"] for c in counts.values())
    overall_tp = sum(c["tp"] for c in counts.values())
    overall_typed = sum(c["typed"] for c in counts.values())

    print("\n" + "=" * 62)
    print(f"{'Type':<14} {'Recall':>8} {'Typed':>8}  {'Redacted':>9}  {'Missed':>7}  {'Total':>6}")
    print("-" * 62)
    for t in PII_TYPES:
        tp, typed, fn = counts[t]["tp"], counts[t]["typed"], counts[t]["fn"]
        total = tp + fn
        recall = tp / total if total else 0
        typed_recall = typed / total if total else 0
        print(f"{t:<14} {recall:>7.1%} {typed_recall:>7.1%}  {tp:>9}  {fn:>7}  {total:>6}")
    print("-" * 62)
    overall_recall = overall_tp / total_gt if total_gt else 0
    overall_typed_recall = overall_typed / total_gt if total_gt else 0
    overall_fn = total_gt - overall_tp
    print(f"{'OVERALL':<14} {overall_recall:>7.1%} {overall_typed_recall:>7.1%}  {overall_tp:>9}  {overall_fn:>7}  {total_gt:>6}")
    print("=" * 62)
    print(f"\nTotal redaction tags in output : {total_tags}")
    print(f"Total GT PII items             : {total_gt}")
    print(f"Over-redaction ratio           : {total_tags / total_gt:.2f}x")

    if save_redacted:
        with open(REDACTED_OUT, "w", encoding="utf-8") as f:
            for line in outputs:
                f.write(line + "\n")
        print(f"\nWrote {REDACTED_OUT}")


if __name__ == "__main__":
    evaluate(skip_ml="--regex-only" in sys.argv, save_redacted="--save-redacted" in sys.argv)
