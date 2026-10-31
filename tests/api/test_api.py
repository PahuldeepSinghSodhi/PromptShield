import time

import pytest

from tests.conftest import AUTH, fake_aws_key

SECRET_MODEL = pytest.mark.skipif(
    not __import__("pathlib").Path("artifacts/secret_detector.joblib").exists(),
    reason="secret detector not trained",
)


def test_health_is_public(client):
    r = client.get("/v1/health")
    assert r.status_code == 200
    assert r.json()["db"] == "ok"


@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "wrong"}])
def test_auth_required(client, headers):
    r = client.post("/v1/scan", json={"text": "hello"}, headers=headers)
    assert r.status_code == 401


def test_validation_error(client):
    r = client.post("/v1/scan", json={"text": ""}, headers=AUTH)
    assert r.status_code == 422


def test_scan_detects_rule_entities_and_sets_request_id(client):
    text = "Mail rahul@example.com about PAN ABCPM1234K"
    r = client.post("/v1/scan", json={"text": text}, headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    labels = {e["label"] for e in body["entities"]}
    assert {"EMAIL", "ID_NUMBER"} <= labels
    assert r.headers["X-Request-ID"] == body["request_id"]
    for e in body["entities"]:
        assert "text" not in e  # values are never echoed back


def test_redact(client):
    r = client.post("/v1/redact", json={"text": "write to rahul@example.com"}, headers=AUTH)
    body = r.json()
    assert body["action"] == "redact"
    assert body["redacted_text"] == "write to [EMAIL_1]"


def test_proxy_redacts_and_restores(client):
    msg = "Draft a reply to rahul@example.com about the refund"
    r = client.post("/v1/proxy/chat", json={"messages": [{"role": "user", "content": msg}]}, headers=AUTH)
    body = r.json()
    assert body["action"] == "redact"
    sent = body["sent_to_llm"][0]["content"]
    assert "rahul@example.com" not in sent and "[EMAIL_1]" in sent
    assert "rahul@example.com" in body["reply"]  # restored for the user


@SECRET_MODEL
def test_proxy_blocks_secret(client):
    msg = f"why does S3 fail? AWS_ACCESS_KEY_ID={fake_aws_key()}"
    r = client.post("/v1/proxy/chat", json={"messages": [{"role": "user", "content": msg}]}, headers=AUTH)
    body = r.json()
    assert body["action"] == "block"
    assert body["reply"] is None and body["sent_to_llm"] is None


def test_text_size_limit(client):
    r = client.post("/v1/scan", json={"text": "a" * 60_000}, headers=AUTH)
    assert r.status_code == 413


def test_batch_scan_csv(client):
    csv = "id,comment\n1,call me at rahul@example.com\n2,all good\n3,PAN ABCPM1234K\n"
    r = client.post("/v1/scan/batch", files={"file": ("c.csv", csv, "text/csv")}, headers=AUTH)
    assert r.status_code == 202
    job_id = r.json()["id"]
    for _ in range(50):
        job = client.get(f"/v1/jobs/{job_id}", headers=AUTH).json()
        if job["status"] in ("done", "failed"):
            break
        time.sleep(0.1)
    assert job["status"] == "done"
    rep = job["report"]
    assert rep["rows_scanned"] == 3 and rep["rows_with_sensitive_data"] == 2
    assert "comment" in rep["per_column"]
    assert "rahul" not in str(rep)  # report never contains values


def test_batch_rejects_unknown_type(client):
    r = client.post("/v1/scan/batch", files={"file": ("x.exe", b"MZ", "application/octet-stream")}, headers=AUTH)
    assert r.status_code == 415


def test_feedback_masks_example(client):
    r = client.post("/v1/feedback", json={
        "request_id": "abc", "kind": "false_negative", "corrected_label": "PERSON",
        "value": "Kavya", "example": "Kavya emailed kavya@example.com",
    }, headers=AUTH)
    assert r.status_code == 201

    from promptshield.db.models import Feedback
    from promptshield.db.session import SessionLocal

    with SessionLocal() as s:
        fb = s.get(Feedback, r.json()["id"])
        assert fb.masked_example == "[PERSON_1] emailed [EMAIL_1]"


def test_metrics_and_drift_endpoints(client):
    m = client.get("/v1/metrics", headers=AUTH).json()
    assert m["requests"] > 0 and "p95" in m["latency_ms"]
    d = client.get("/v1/drift", headers=AUTH).json()
    assert d["status"] in {"ok", "no_reference", "insufficient_data"}
