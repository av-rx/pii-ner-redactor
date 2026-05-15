#!/usr/bin/env python3
"""
generate_pii_testset.py

Generates:
- pii_test_corpus.txt       (1000 lines, paragraph-style with mixed fake PII)
- pii_test_ground_truth.json (ground truth mapping of PII items for evaluation)

Run:
    python generate_pii_testset.py
"""
import os
import json
import random
import itertools

random.seed(42)

OUT_TEXT = "pii_test_corpus.txt"
OUT_GT = "pii_test_ground_truth.json"
NUM_LINES = 1000

# Small name/org pools (extend as needed)
first_names = ["Liam","Olivia","Noah","Emma","Oliver","Ava","Elijah","Sophia","Lucas","Isabella",
               "Mateo","Hannah","Wei","Sofia","Hiro","Ana","Carlos","Zara","Amir","Maya"]
last_names = ["Smith","Johnson","Brown","Williams","Jones","Garcia","Miller","Davis","Lopez","Wilson",
              "Martinez","Anderson","Thomas","Taylor","Moore","Nguyen","Khan","Singh","Ivanov","Santos"]
org_names = ["Acme Corp","BlueWave Systems","Horizon Labs","Northwell University","Crescent NGO",
             "Starlight Holdings","Greenfield Bank","Pioneer Telecom","GlobalMed Institute","Atlas Logistics"]
domains = ["example.com","testmail.org","service.co.uk","sub.domain.net","mail.example.ai","company.io"]

# Phone formats templates
phone_formats = [
    "+1-{}-{}-{}", "+44 {} {} {}", "+61 {} {} {}", "+91-{}-{}", "+7 {} {} {}",
    "({}) {}-{}", "{}.{}.{}", "{} {} {}", "+{} {} {} {}"
]

# Postal code / address pieces for variety
street_names = ["Main","High","Maple","Oak","Pine","Cedar","Elm","Market","King","Queen","Victoria","Church"]
street_types = ["Street","St","Avenue","Ave","Boulevard","Blvd","Road","Rd","Lane","Ln","Drive","Dr","Court","Ct"]
cities = ["New York","London","Toronto","Sydney","Mumbai","Moscow","Lisbon","Dublin","Berlin","Madrid"]
countries = ["USA","UK","Canada","Australia","India","Russia","Portugal","Ireland","Germany","Spain"]

# ID templates
def fake_ssn():
    return f"{random.randint(100,899)}-{random.randint(10,99)}-{random.randint(1000,9999)}"

def fake_credit_card():
    # produce 16 digit groups resembling cards (not real)
    groups = [str(random.randint(1000,9999)) for _ in range(4)]
    sep = random.choice([" ","-",""])
    return sep.join(groups)

def fake_passport(country_code=None):
    # generic: 1 letter + 7 digits or country-specific short formats
    letters = "ABCDEFGHJKLMNPQRSTUVWXYZ"
    if country_code and country_code.upper()=="UK":
        return f"{random.randint(1000000,9999999)}"
    return random.choice(letters) + str(random.randint(10**5,10**7))

def random_phone():
    fmt = random.choice(phone_formats)
    num_groups = fmt.count("{}")
    groups = []
    for i in range(num_groups):
        # choose a reasonable length for groups
        length = random.choice([2,3,3,3,4])
        groups.append(''.join(str(random.randint(0,9)) for _ in range(length)))
    return fmt.format(*groups)

def random_email(name_hint=None):
    user = (name_hint or random.choice(first_names)).lower()
    user = user.replace(" ",".") + str(random.randint(1,99))
    domain = random.choice(domains)
    # sometimes use subdomains, plus addressing, dots
    variants = [
        f"{user}@{domain}",
        f"{user}+promo@{domain}",
        f"{user.replace('.', '')}@sub.{domain.split('.',1)[1]}",
    ]
    return random.choice(variants)

def random_name():
    first = random.choice(first_names)
    last = random.choice(last_names)
    # occasionally use middle initial or accented/international style
    if random.random() < 0.2:
        return f"{first} {random.choice(['A.','B.','C.'])} {last}"
    if random.random() < 0.1:
        return f"{first}-{last}"
    return f"{first} {last}"

def random_org():
    base = random.choice(org_names)
    if random.random() < 0.3:
        suffix = random.choice(["Inc.", "LLC", "Ltd", "Co", "PLC", "Group"])
        return f"{base} {suffix}"
    return base

def random_address():
    # produce a varied address including international forms
    house = str(random.randint(1,9999))
    street = random.choice(street_names) + " " + random.choice(street_types)
    city = random.choice(cities)
    country = random.choice(countries)
    # sometimes include apt/unit and postal code
    apt = ""
    if random.random() < 0.3:
        apt = f", Apt {random.randint(1,999)}"
    postal = ""
    if random.random() < 0.6:
        postal = f", {random.choice(['',str(random.randint(10000,99999)), str(random.randint(1000,9999))+' '+random.choice(['AB','CD'])])}".strip(", ")
    # return a single-line address
    return f"{house} {street}{apt}, {city}{(', '+country) if random.random()<0.6 else ''}{(', '+postal) if postal else ''}".strip(", ")

# Build a pool of sample items per type to ensure variety
POOL_PER_TYPE = {
    "NAME": [random_name() for _ in range(300)],
    "ORG": [random_org() for _ in range(200)],
    "EMAIL": [random_email(random.choice(first_names)) for _ in range(400)],
    "PHONE": [random_phone() for _ in range(400)],
    "ADDRESS": [random_address() for _ in range(500)],
    "SSN": [fake_ssn() for _ in range(200)],
    "CREDIT_CARD": [fake_credit_card() for _ in range(300)],
    "PASSPORT": [fake_passport() for _ in range(200)]
}

# Types we'll include roughly equally; choose a balanced cycle
pii_types = ["NAME","ORG","EMAIL","PHONE","ADDRESS","SSN","CREDIT_CARD","PASSPORT"]

def make_paragraph(line_index):
    """
    Create a paragraph that reads like a short note with several PII items.
    We ensure each paragraph contains 3-6 PII items, with varied context wording.
    """
    num_items = random.choice([3,4,4,5])  # average ~4 items per line
    chosen = []
    # choose items in a rotating balanced manner
    for i in range(num_items):
        t = pii_types[(line_index + i) % len(pii_types)]
        val = random.choice(POOL_PER_TYPE[t])
        chosen.append((t, val))

    # Build a natural-sounding sentence/paragraph
    templates = [
        "Contact {NAME} at {EMAIL} or {PHONE}. They work at {ORG}, living at {ADDRESS}. ID: {SSN}.",
        "{ORG} hired {NAME} (email: {EMAIL}) — office: {ADDRESS}, mobile {PHONE}. Passport: {PASSPORT}.",
        "Customer {NAME} used card {CREDIT_CARD} and provided address {ADDRESS}; contact: {PHONE}, email {EMAIL}.",
        "{NAME} — {ORG} — reachable via {EMAIL} / {PHONE}; SSN {SSN}; passport {PASSPORT}.",
        "Booking for {NAME} at {ADDRESS}. Contact: {PHONE}. Confirmation sent to {EMAIL}. ID: {CREDIT_CARD}.",
        "{NAME} ({ORG}), tel {PHONE}, email {EMAIL}, located {ADDRESS}, passport: {PASSPORT}.",
        "Record: {SSN} / {NAME} / {ADDRESS} / {EMAIL} / {PHONE}.",
        "{ORG} ({ADDRESS}) processed payment card {CREDIT_CARD} for customer {NAME} (email: {EMAIL})."
    ]
    tpl = random.choice(templates)
    # build mapping for placeholders
    mapping = {}
    for (t, val) in chosen:
        # find an unused placeholder slot of that type in template; if not present, append a clause
        mapping.setdefault(t, []).append(val)

    # For simplicity, fill placeholders by taking the first available value from mapping; if template lacks
    # a placeholder, append the others at the end of the paragraph.
    para = tpl
    # Replace placeholders in order with first available mapping value and then pop it
    for t in pii_types:
        ph = "{" + t + "}"
        if ph in para:
            arr = mapping.get(t, [])
            if arr:
                para = para.replace(ph, arr.pop(0), 1)
            else:
                # if none available, put a plausible fake
                para = para.replace(ph, random.choice(POOL_PER_TYPE[t]), 1)

    # Append any remaining chosen items that weren't used by template
    extras = list(itertools.chain.from_iterable(mapping.values()))
    if extras:
        para = para.rstrip() + " Extras: " + "; ".join(extras) + "."

    # Small random fluff to make it look natural
    if random.random() < 0.2:
        para = "Note: " + para

    return para, chosen

def main():
    corpus = []
    ground_truth = []

    for i in range(NUM_LINES):
        para, items = make_paragraph(i)
        corpus.append(para)
        # ground truth stores the exact substring values and type
        gt_items = []
        for t, val in items:
            gt_items.append({"type": t, "text": val})
        ground_truth.append({"line": i+1, "text": para, "items": gt_items})

    # write text file
    with open(OUT_TEXT, "w", encoding="utf-8") as f:
        for line in corpus:
            f.write(line.replace("\n", " ") + "\n")

    with open(OUT_GT, "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, indent=2, ensure_ascii=False)

    print(f"Wrote {OUT_TEXT} ({NUM_LINES} lines) and {OUT_GT} (ground truth).")

main()
