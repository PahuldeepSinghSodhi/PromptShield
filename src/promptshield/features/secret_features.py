"""Candidate extraction + NumPy features for the secret detector.

This module is imported by BOTH the training code and the serving path, so a
candidate is always extracted and featurised the same way (no train/serve skew).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

_TOKEN_RE = re.compile(r"[^\s\"'`<>(),;\[\]{}]+")
_KEY_RE = re.compile(r"^[A-Za-z_][\w.\-]*$")
MIN_LEN = 6

SECRET_KEYWORDS = (
    "key", "token", "secret", "password", "passwd", "pwd", "pass", "auth", "bearer",
    "credential", "api", "private", "session", "jwt", "database_url", "dsn", "login",
)
BENIGN_KEYWORDS = (
    "commit", "sha", "checksum", "hash", "md5", "digest", "uuid", "request_id", "trace",
    "version", "invoice", "order", "ref", "etag", "merged", "revert", "checkout", "img",
    "docs", "see", "def", "rename", "open", "saved", "template",
)
KNOWN_PREFIXES = ("AKIA", "ASIA", "ghp_", "gho_", "github_pat_", "sk-", "xoxb-", "xoxp-", "eyJ", "AIza")

_HEX = set("0123456789abcdefABCDEF")
_B64 = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=_-")


@dataclass(frozen=True)
class Candidate:
    value: str
    start: int
    end: int
    left: str  # up to 40 chars of text before the candidate (lower-cased)
    key: str  # the "KEY" in KEY=value, lower-cased, else ""


def extract_candidates(text: str, context: int = 40) -> list[Candidate]:
    out: list[Candidate] = []
    for m in _TOKEN_RE.finditer(text):
        tok, start = m.group(), m.start()
        key = ""
        if "=" in tok:
            k, _, v = tok.partition("=")
            if _KEY_RE.match(k) and v:
                key, tok, start = k.lower(), v, start + len(k) + 1
        if tok.endswith(":"):  # a label such as "api_key:", not a value
            continue
        stripped = tok.rstrip(".,")
        if len(stripped) < MIN_LEN:
            continue
        left = text[max(0, start - context) : start].lower()
        out.append(Candidate(stripped, start, start + len(stripped), left, key))
    return out


def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = np.bincount(np.frombuffer(s.encode("utf-8", "ignore"), dtype=np.uint8))
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


FEATURE_NAMES = [
    "len", "log_len", "entropy", "entropy_per_char_bound", "frac_upper", "frac_lower",
    "frac_digit", "frac_symbol", "n_char_classes", "is_hex", "is_b64_charset",
    "digit_letter_switch_rate", "max_digit_run_frac", "n_dots", "has_scheme_creds",
    "known_prefix", "left_secret_kw", "left_benign_kw", "key_secret_kw", "has_key",
    "looks_like_word", "has_path_sep", "placeholder_hint", "max_char_frac", "is_numeric",
    "vendor_prefix_shape", "is_uuid", "is_plain_url", "wrapped_by_template_syntax",
]

_VENDOR_SHAPE = re.compile(r"^[A-Za-z]{2,8}[-_](?:[A-Za-z]{2,6}[-_])?[A-Za-z0-9_\-]{16,}$")
_UUID = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")

_PLACEHOLDER_HINTS = ("your", "example", "placeholder", "changeme", "replace", "dummy", "insert",
                      "xxxx", "sample", "<", ">", "${", "$", "getenv", "environ", "secrets.")


def _row(c: Candidate) -> list[float]:
    v = c.value
    n = len(v)
    arr = np.frombuffer(v.encode("utf-8", "ignore"), dtype=np.uint8)
    upper = ((arr >= 65) & (arr <= 90)).mean() if arr.size else 0.0
    lower = ((arr >= 97) & (arr <= 122)).mean() if arr.size else 0.0
    digit = ((arr >= 48) & (arr <= 57)).mean() if arr.size else 0.0
    symbol = max(0.0, 1.0 - upper - lower - digit)
    ent = shannon_entropy(v)
    is_digit = (arr >= 48) & (arr <= 57)
    switches = np.count_nonzero(is_digit[1:] != is_digit[:-1]) / max(1, n - 1)
    # longest run of consecutive digits, as a fraction of the length
    run = best = 0
    for d in is_digit:
        run = run + 1 if d else 0
        best = max(best, run)
    return [
        n,
        np.log1p(n),
        ent,
        ent / np.log2(n + 1),
        upper,
        lower,
        digit,
        symbol,
        sum(x > 0 for x in (upper, lower, digit, symbol)),
        float(all(ch in _HEX for ch in v)),
        float(all(ch in _B64 for ch in v)),
        switches,
        best / max(1, n),
        v.count("."),
        float("://" in v and "@" in v),
        float(v.startswith(KNOWN_PREFIXES)),
        float(any(k in c.left for k in SECRET_KEYWORDS)),
        float(any(k in c.left for k in BENIGN_KEYWORDS)),
        float(any(k in c.key for k in SECRET_KEYWORDS)),
        float(bool(c.key)),
        float(v.isalpha() or bool(re.fullmatch(r"[A-Za-z]+(_[A-Za-z]+)+", v))),
        float("/" in v or "\\" in v),
        float(any(h in v.lower() for h in _PLACEHOLDER_HINTS)),
        float(np.bincount(arr).max() / n) if arr.size else 0.0,
        float(v.isdigit()),
        # vendor-style tokens (prefix + separator + long random tail) that are not plain words
        float(bool(_VENDOR_SHAPE.match(v)) and digit > 0),
        float(bool(_UUID.match(v))),
        float(v.startswith(("http://", "https://")) and "@" not in v),
        float(c.left.rstrip().endswith(("<", "${", "{{", "{", "$"))),
    ]


def featurize(candidates: list[Candidate]) -> np.ndarray:
    if not candidates:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)
    return np.asarray([_row(c) for c in candidates], dtype=np.float32)
