"""Regex and composition tests for main.redact_text. No model is loaded."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main  # noqa: E402
from main import redact_text, chunk_text  # noqa: E402


def setup_function(_):
    main.ner_pipeline = None
    main.last_ml_error = None


# --- email ---

def test_email_with_plus_addressing_is_fully_redacted():
    assert redact_text("mail maya62+promo@company.io now") == "mail [REDACTED_EMAIL] now"


# --- address ---

def test_address_does_not_span_newlines():
    out = redact_text("Line one ends here\n2640 Maple Boulevard, New York")
    assert out == "Line one ends here\n[REDACTED_ADDRESS], New York"


def test_street_suffix_needs_word_boundary():
    out = redact_text("They work at Starlight Holdings today")
    assert "REDACTED_ADDRESS" not in out


def test_address_does_not_swallow_preceding_words():
    out = redact_text("I met Isabella Davis at 2640 Maple Boulevard yesterday")
    assert out == "I met Isabella Davis at [REDACTED_ADDRESS] yesterday"


def test_address_with_apartment():
    assert redact_text("at 2775 Main Ave, Apt 742, Moscow") == "at [REDACTED_ADDRESS], Moscow"


# --- credit card vs phone ordering ---

def test_spaced_credit_card_is_tagged_as_card():
    assert redact_text("card 4558 6532 3567 7296 ok") == "card [REDACTED_CREDIT_CARD] ok"


def test_dashed_credit_card_is_tagged_as_card():
    assert redact_text("card 4558-6532-3567-7296 ok") == "card [REDACTED_CREDIT_CARD] ok"


def test_contiguous_credit_card_is_tagged_as_card():
    assert redact_text("card 4558653235677296 ok") == "card [REDACTED_CREDIT_CARD] ok"


# --- ssn ---

def test_ssn_is_tagged_as_ssn_not_phone():
    assert redact_text("ssn 666-78-9810 end") == "ssn [REDACTED_SSN] end"


# --- phone ---

def test_phone_with_two_digit_groups():
    assert redact_text("call +44 86 713 344 now") == "call [REDACTED_PHONE] now"


def test_phone_parenthesised_area_code():
    assert redact_text("tel (314) 1357-594.") == "tel [REDACTED_PHONE]."


def test_phone_dotted():
    assert redact_text("tel 6591.135.852 end") == "tel [REDACTED_PHONE] end"


def test_iso_date_is_not_a_phone():
    assert redact_text("due 2025-01-01 sharp") == "due 2025-01-01 sharp"


def test_short_number_is_not_a_phone():
    assert redact_text("Call 911 or room 1234") == "Call 911 or room 1234"


# --- passport ---

def test_passport_with_q_x_z_initial():
    for p in ["Q3533725", "X1234567", "Z3904776"]:
        assert redact_text(f"passport {p} end") == "passport [REDACTED_PASSPORT] end"


def test_passport_seven_chars_and_trailing_zero():
    for p in ["U838763", "Q3533720", "A0123456"]:
        assert redact_text(f"passport {p} end") == "passport [REDACTED_PASSPORT] end"


# --- idempotency (regex only) ---

def test_regex_only_redaction_is_idempotent():
    line = ("Contact Sofia C. Miller at oliver22@sub.co.uk or +7 273 14 195. They work at "
            "Starlight Holdings, living at 8352 Oak Street, Madrid, 22393. ID: 245-41-8906. "
            "Extras: U4333438. Card 4050 7270 4312 5336.")
    once = redact_text(line)
    assert redact_text(once) == once


# --- chunking for the 512-token model limit ---

def _words(s):
    return len(s.split())


def test_chunk_text_splits_on_lines_and_rejoins_exactly():
    text = "one two three\nfour five\n\nsix"
    chunks = chunk_text(text, max_tokens=10, count_tokens=_words)
    assert "".join(chunks) == text
    assert all("\n" not in c.rstrip("\n") for c in chunks)


def test_chunk_text_splits_overlong_lines_under_budget():
    text = " ".join(f"w{i}" for i in range(25)) + "\nshort line\n"
    chunks = chunk_text(text, max_tokens=10, count_tokens=_words)
    assert "".join(chunks) == text
    assert all(_words(c) <= 10 for c in chunks)


# --- NER composition with a fake pipeline ---

class FakePipeline:
    """Tags every occurrence of the given names as PER; records what it was called with."""

    def __init__(self, names, label="PER"):
        self.names = names
        self.label = label
        self.calls = []

    def __call__(self, text):
        self.calls.append(text)
        ents = []
        for name in self.names:
            i = text.find(name)
            while i != -1:
                ents.append({"entity_group": self.label, "start": i, "end": i + len(name), "score": 0.99})
                i = text.find(name, i + 1)
        return sorted(ents, key=lambda e: e["start"])


def test_ner_is_applied_per_line_so_long_documents_are_covered():
    fake = FakePipeline(["Ada Lovelace"])
    main.ner_pipeline = fake
    text = "\n".join(f"line {i} by Ada Lovelace" for i in range(3))
    out = redact_text(text)
    assert out.count("[REDACTED_PER]") == 3
    assert all("\n" not in c for c in fake.calls)


def test_ner_offsets_shift_correctly_with_multiple_entities_on_one_line():
    main.ner_pipeline = FakePipeline(["Ada Lovelace", "Alan Turing"])
    out = redact_text("Ada Lovelace met Alan Turing and Ada Lovelace again")
    assert out == "[REDACTED_PER] met [REDACTED_PER] and [REDACTED_PER] again"


def test_ner_failure_falls_back_to_regex_and_is_recorded():
    class Broken:
        def __call__(self, text):
            raise RuntimeError("boom")

    main.ner_pipeline = Broken()
    out = redact_text("Ada Lovelace ada@example.com")
    assert out == "Ada Lovelace [REDACTED_EMAIL]"
    assert main.last_ml_error is not None
    assert "boom" in main.last_ml_error


def test_contiguous_ten_digit_phone():
    assert redact_text("tel 5551234567 end") == "tel [REDACTED_PHONE] end"


def test_eight_digit_number_without_separators_is_left_alone():
    assert redact_text("order 20240115 shipped") == "order 20240115 shipped"


def test_house_number_left_by_ner_location_tag_is_folded_into_address():
    main.ner_pipeline = FakePipeline(["Maple Boulevard", "New York"], label="LOC")
    out = redact_text("Booking at 2640 Maple Boulevard, New York today")
    assert out == "Booking at [REDACTED_ADDRESS], [REDACTED_LOC] today"
