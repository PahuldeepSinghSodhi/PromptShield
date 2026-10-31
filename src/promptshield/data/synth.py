"""Synthetic data generators.

Two outputs:
* PII records for the NER model (Indian + international formats the public
  dataset under-represents), in the same ``text`` / ``spans`` schema as ingest.
* Secret candidates for the sklearn secret detector: fake credentials in real
  formats (positives) and look-alike strings such as git SHAs and UUIDs
  (hard negatives). Every value is randomly generated and fake by construction.
"""

from __future__ import annotations

import base64
import hashlib
import json
import random
import string
import uuid

import pandas as pd
from faker import Faker

# ---------------------------------------------------------------- Indian IDs

_VERHOEFF_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]
_VERHOEFF_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]
_VERHOEFF_INV = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]


def verhoeff_check_digit(number: str) -> str:
    c = 0
    for i, ch in enumerate(reversed(number)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[(i + 1) % 8][int(ch)]]
    return str(_VERHOEFF_INV[c])


def verhoeff_valid(number: str) -> bool:
    c = 0
    for i, ch in enumerate(reversed(number)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][int(ch)]]
    return c == 0


def fake_aadhaar(rng: random.Random) -> str:
    body = str(rng.randint(2, 9)) + "".join(rng.choices(string.digits, k=10))
    num = body + verhoeff_check_digit(body)
    return f"{num[:4]} {num[4:8]} {num[8:]}" if rng.random() < 0.7 else num


def fake_pan(rng: random.Random) -> str:
    letters = string.ascii_uppercase
    return (
        "".join(rng.choices(letters, k=3))
        + rng.choice("PCHFATBLJG")
        + rng.choice(letters)
        + "".join(rng.choices(string.digits, k=4))
        + rng.choice(letters)
    )


def fake_ifsc(rng: random.Random) -> str:
    bank = rng.choice(["HDFC", "ICIC", "SBIN", "UTIB", "KKBK", "PUNB", "BARB", "YESB"])
    return bank + "0" + "".join(rng.choices(string.ascii_uppercase + string.digits, k=6))


def fake_upi(rng: random.Random, fake: Faker) -> str:
    handle = rng.choice(["okhdfcbank", "oksbi", "okicici", "ybl", "paytm", "axl", "ibl"])
    user = fake.user_name() if rng.random() < 0.6 else "".join(rng.choices(string.digits, k=10))
    return f"{user}@{handle}"


def fake_in_phone(rng: random.Random) -> str:
    num = rng.choice("6789") + "".join(rng.choices(string.digits, k=9))
    style = rng.random()
    if style < 0.35:
        return f"+91 {num[:5]} {num[5:]}"
    if style < 0.6:
        return f"+91-{num}"
    if style < 0.8:
        return num
    return f"0{num[:2]}-{num[2:6]}-{num[6:]}"


def fake_account(rng: random.Random) -> str:
    return "".join(rng.choices(string.digits, k=rng.choice([11, 12, 14, 16])))


# ------------------------------------------------------------- PII templates

# Each slot is {LABEL:generator}. Labels must match configs/label_map.yaml targets.
_PII_TEMPLATES = [
    "Hi team, customer {PERSON:name} ({EMAIL:email}) says the refund to account {FINANCIAL:account} never arrived.",
    "Please update KYC for {PERSON:name}: PAN {ID_NUMBER:pan}, Aadhaar {ID_NUMBER:aadhaar}.",
    "Call {PERSON:name} back on {PHONE:in_phone} regarding the failed NEFT transfer.",
    "Beneficiary: {PERSON:name}, A/C {FINANCIAL:account}, IFSC {FINANCIAL:ifsc}.",
    "Can you summarise this complaint? {PERSON:name} from {ADDRESS:address} was charged twice.",
    "Send the invoice to {EMAIL:email} and copy {PERSON:name}.",
    "UPI payment from {FINANCIAL:upi} failed three times today, user is {PERSON:name}.",
    "Draft a polite reply to {PERSON:name} at {EMAIL:email} about their delayed order.",
    "Employee {PERSON:name}, PAN {ID_NUMBER:pan}, mobile {PHONE:in_phone}, joined on {DATE:date}.",
    "The shipment for {PERSON:name} should go to {ADDRESS:address}.",
    "Login issue reported by {USERNAME:username}; their registered phone is {PHONE:phone}.",
    "Patient {PERSON:name}, DOB {DATE:date}, contact {PHONE:in_phone}, needs a follow-up.",
    "Verify Aadhaar {ID_NUMBER:aadhaar} against the name {PERSON:name} before approval.",
    "Please translate this email from {PERSON:name} <{EMAIL:email}> into Hindi.",
    "Loan applicant {PERSON:name} lives at {ADDRESS:address} and banks with IFSC {FINANCIAL:ifsc}.",
    "Card ending in {FINANCIAL:card} belongs to {PERSON:name}; dispute raised on {DATE:date}.",
    "Who should handle the ticket from {USERNAME:username} ({EMAIL:email})?",
    "Write a summary: {PERSON:name} called from {PHONE:phone} about a SIM swap.",
    "Rewrite this note for {PERSON:name}: your payment to {FINANCIAL:upi} is pending.",
    "Vendor contact is {PERSON:name}, {PHONE:in_phone}, {ADDRESS:address}.",
]

# Templates without PII so the model sees plenty of negatives.
_CLEAN_TEMPLATES = [
    "Summarise the top five complaint themes from last quarter.",
    "Write a Python function that merges two sorted lists.",
    "What is the difference between NEFT and RTGS transfers?",
    "Draft a polite email declining the meeting invite.",
    "Explain how gradient boosting works in simple terms.",
    "Our churn rate rose by 3 percent in March; suggest possible causes.",
    "Translate 'your order has shipped' into Hindi and Tamil.",
    "Create a checklist for onboarding a new data engineer.",
]


def _slot_value(kind: str, rng: random.Random, fake: Faker) -> str:
    if kind == "name":
        return fake.name()
    if kind == "email":
        return fake.email()
    if kind == "phone":
        return fake.phone_number()
    if kind == "in_phone":
        return fake_in_phone(rng)
    if kind == "address":
        return fake.address().replace("\n", ", ")
    if kind == "pan":
        return fake_pan(rng)
    if kind == "aadhaar":
        return fake_aadhaar(rng)
    if kind == "ifsc":
        return fake_ifsc(rng)
    if kind == "upi":
        return fake_upi(rng, fake)
    if kind == "account":
        return fake_account(rng)
    if kind == "card":
        return "".join(rng.choices(string.digits, k=4))
    if kind == "date":
        return fake.date(pattern=rng.choice(["%d/%m/%Y", "%Y-%m-%d", "%d %B %Y"]))
    if kind == "username":
        return fake.user_name()
    raise ValueError(f"unknown slot kind: {kind}")


def _fill(template: str, rng: random.Random, fake: Faker) -> tuple[str, list[dict]]:
    out, spans, i = [], [], 0
    pos = 0
    while i < len(template):
        if template[i] == "{":
            j = template.index("}", i)
            label, kind = template[i + 1 : j].split(":")
            value = _slot_value(kind, rng, fake)
            spans.append({"start": pos, "end": pos + len(value), "label": label})
            out.append(value)
            pos += len(value)
            i = j + 1
        else:
            out.append(template[i])
            pos += 1
            i += 1
    return "".join(out), spans


def generate_pii_records(n: int, seed: int = 42, clean_ratio: float = 0.15) -> pd.DataFrame:
    """Return ``n`` synthetic records with columns text, spans, source, template_id."""
    rng = random.Random(seed)
    fakers = [Faker("en_IN"), Faker("en_US"), Faker("en_GB")]
    for k, f in enumerate(fakers):
        f.seed_instance(seed + k)
    rows = []
    for _ in range(n):
        fake = fakers[0] if rng.random() < 0.6 else rng.choice(fakers[1:])
        if rng.random() < clean_ratio:
            tid = rng.randrange(len(_CLEAN_TEMPLATES))
            text, spans = _CLEAN_TEMPLATES[tid], []
            template_id = f"synth-clean-{tid}"
        else:
            tid = rng.randrange(len(_PII_TEMPLATES))
            text, spans = _fill(_PII_TEMPLATES[tid], rng, fake)
            template_id = f"synth-pii-{tid}"
        rows.append({"text": text, "spans": spans, "source": "synth", "template_id": template_id})
    return pd.DataFrame(rows)


# ---------------------------------------------------- secret candidates (sklearn)

_ALNUM = string.ascii_letters + string.digits


def _rand(rng: random.Random, alphabet: str, k: int) -> str:
    return "".join(rng.choices(alphabet, k=k))


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _fake_jwt(rng: random.Random) -> str:
    header = _b64url(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    payload = _b64url(json.dumps({"sub": str(rng.randint(1000, 99999)), "iat": rng.randint(1_600_000_000, 1_800_000_000)}).encode())
    sig = _b64url(rng.randbytes(32))
    return f"{header}.{payload}.{sig}"


def _fake_password(rng: random.Random) -> str:
    pool = _ALNUM + "!@#$%^&*_-+?"
    pw = _rand(rng, pool, rng.randint(9, 20))
    return pw if rng.random() < 0.8 else rng.choice(["Welcome", "Summer", "Admin", "India"]) + str(
        rng.randint(1, 2026)
    ) + rng.choice("!@#$")


# (generator, context templates). ``{v}`` marks where the candidate sits.
def _secret_generators(rng: random.Random) -> list[tuple[str, list[str]]]:
    hexkey = _rand(rng, "0123456789abcdef", rng.choice([32, 40, 64]))
    return [
        ("AKIA" + _rand(rng, string.ascii_uppercase + string.digits, 16),
         ["AWS_ACCESS_KEY_ID={v}", "aws_access_key_id = {v}", "my key is {v} can you check why s3 fails"]),
        (_rand(rng, _ALNUM + "/+", 40),
         ["AWS_SECRET_ACCESS_KEY={v}", "aws_secret_access_key: {v}", "secret: {v}"]),
        ("ghp_" + _rand(rng, _ALNUM, 36),
         ["GITHUB_TOKEN={v}", "git clone https://{v}@github.com/org/repo.git", "token = \"{v}\""]),
        ("xoxb-" + "-".join(_rand(rng, string.digits, 12) for _ in range(2)) + "-" + _rand(rng, _ALNUM, 24),
         ["SLACK_BOT_TOKEN={v}", "slack token {v} stopped working"]),
        ("sk-" + _rand(rng, _ALNUM, 48),
         ["OPENAI_API_KEY={v}", "api_key=\"{v}\"", "here is my api key {v} why do I get 401"]),
        (_fake_jwt(rng),
         ["Authorization: Bearer {v}", "jwt={v}", "the session token {v} expired"]),
        (hexkey,
         ["api_key: {v}", "X-API-Key: {v}", "client_secret = '{v}'", "STRIPE_KEY={v}"]),
        (_fake_password(rng),
         ["password: {v}", "DB_PASSWORD={v}", "my login password is {v}", "pwd={v}", "\"password\": \"{v}\""]),
        ("postgres://" + _rand(rng, string.ascii_lowercase, 6) + ":" + _fake_password(rng).replace("@", "x")
         + "@db." + _rand(rng, string.ascii_lowercase, 5) + ".internal:5432/prod",
         ["DATABASE_URL={v}", "engine = create_engine(\"{v}\")"]),
    ]


def _negative_generators(rng: random.Random) -> list[tuple[str, list[str]]]:
    words = ["customerSegmentId", "calculate_monthly_revenue", "TransactionReconciliationService",
             "internationalization", "getUserPreferences", "max_retry_attempts"]
    return [
        (_rand(rng, "0123456789abcdef", 40),
         ["commit {v}", "git checkout {v}", "fixed in {v}", "revert {v} please"]),
        (_rand(rng, "0123456789abcdef", 7),
         ["see commit {v}", "merged {v} into main"]),
        (str(uuid.UUID(int=rng.getrandbits(128))),
         ["request_id={v}", "order {v} is stuck", "trace id {v}"]),
        (hashlib.sha256(rng.randbytes(16)).hexdigest(),
         ["sha256 checksum {v}", "file hash: {v}", "digest={v}"]),
        (hashlib.md5(rng.randbytes(16)).hexdigest(),
         ["md5sum {v}", "etag {v}"]),
        ("data:image/png;base64," + base64.b64encode(rng.randbytes(48)).decode(),
         ["<img src=\"{v}\">"]),
        ("https://" + rng.choice(["docs.python.org", "example.com", "github.com"]) + "/" + _rand(rng, string.ascii_lowercase, 8),
         ["see {v}", "docs at {v}"]),
        (rng.choice(words),
         ["def {v}(self):", "call {v} after login", "rename {v}"]),
        (f"{rng.randint(1, 9)}.{rng.randint(0, 30)}.{rng.randint(0, 99)}",
         ["upgrade to version {v}", "pandas=={v}"]),
        ("C:\\Users\\" + _rand(rng, string.ascii_lowercase, 6) + "\\Documents\\report_" + str(rng.randint(1, 99)) + ".xlsx",
         ["open {v}", "saved to {v}"]),
        ("INV-" + _rand(rng, string.digits, 8),
         ["invoice {v} is overdue", "ref {v}"]),
        ("password_reset_" + _rand(rng, string.ascii_lowercase, 5),
         ["template name {v}", "enable {v} flow"]),
    ]


_CHAT_PREFIXES = [
    "", "", "Hey, the deploy failed. ", "Can you debug this? ", "Quick question: ",
    "Here is my config:\n", "Logs from prod:\n", "Why does this not work?\n",
]
_CHAT_SUFFIXES = ["", "", " Thanks!", "\nAny idea what is wrong?", " (from staging)", "\nPlease fix."]


def generate_secret_texts(n: int, seed: int = 42, positive_ratio: float = 0.45) -> pd.DataFrame:
    """Return short texts that each contain one secret (label 1) or one look-alike (label 0).

    ``start``/``end`` give the span of the planted value. The training code runs the
    same candidate extractor used at serving time over ``text`` and labels each
    extracted candidate by overlap with this span.
    """
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        positive = rng.random() < positive_ratio
        gens = _secret_generators(rng) if positive else _negative_generators(rng)
        kind_idx = rng.randrange(len(gens))
        value, contexts = gens[kind_idx]
        ctx = rng.choice(contexts)
        prefix = rng.choice(_CHAT_PREFIXES)
        text = prefix + ctx.replace("{v}", value) + rng.choice(_CHAT_SUFFIXES)
        start = text.index(value)
        rows.append(
            {
                "text": text,
                "candidate": value,
                "start": start,
                "end": start + len(value),
                "label": int(positive),
                "kind": f"{'pos' if positive else 'neg'}-{kind_idx}",
                "template_id": f"{'pos' if positive else 'neg'}-{kind_idx}-{contexts.index(ctx)}",
            }
        )
    return pd.DataFrame(rows)
