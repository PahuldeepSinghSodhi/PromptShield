import contextlib
import os
import random
import string

# Isolated settings for every test run: temp SQLite DB, rules + secrets only unless
# a test opts into the real NER model.
os.environ.setdefault("PROMPTSHIELD_DATABASE_URL", "sqlite:///./.pytest_promptshield.db")
os.environ.setdefault("PROMPTSHIELD_API_KEYS", "test-key")
os.environ.setdefault("PROMPTSHIELD_NER_BACKEND", os.environ.get("PS_TEST_NER_BACKEND", "none"))
os.environ.setdefault("PROMPTSHIELD_LLM_PROVIDER", "mock")
os.environ.setdefault("PROMPTSHIELD_RATE_LIMIT_PER_MINUTE", "10000")

import pytest  # noqa: E402

AUTH = {"X-API-Key": "test-key"}


def fake_aws_key(seed: int = 0) -> str:
    """Built at runtime so no key-shaped literal is committed to the repo."""
    rng = random.Random(seed)
    return "AKIA" + "".join(rng.choices(string.ascii_uppercase + "234567", k=16))


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    from promptshield.serving.app import app

    with TestClient(app) as c:
        yield c
    with contextlib.suppress(OSError):
        os.remove(".pytest_promptshield.db")
