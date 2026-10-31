# API reference

Interactive OpenAPI docs are served at `/docs` (Swagger) and `/redoc`. Every `/v1` endpoint except `/v1/health` needs an `X-API-Key` header. Every response carries an `X-Request-ID` header, which you can send yourself to correlate logs.

| Method | Path | Purpose |
|---|---|---|
| POST | `/v1/scan` | Detect entities; returns spans, labels, scores and the policy action |
| POST | `/v1/redact` | Same as scan, plus the redacted text with `[LABEL_n]` placeholders |
| POST | `/v1/proxy/chat` | Scan → policy → forward redacted messages to the LLM → restore the reply |
| POST | `/v1/scan/batch` | Upload a CSV/JSONL/TXT (≤ 10 MB); returns a job id (202) |
| GET | `/v1/jobs/{id}` | Batch job status and report (counts and locations only) |
| POST | `/v1/feedback` | Report a false positive, a missed entity, or a wrong label |
| GET | `/v1/review-queue` | Recent low-confidence requests (masked previews) |
| GET | `/v1/metrics?hours=24` | Volume, actions, entity mix, latency percentiles, hourly timeline |
| GET | `/v1/drift?window=500` | PSI / KS of recent traffic vs the training reference |
| GET | `/v1/model-info` | Loaded model versions and their evaluation metrics |
| GET | `/v1/health` | Liveness: DB, NER and secret detector status |

## Examples

```bash
curl -s localhost:8000/v1/redact -H "X-API-Key: dev-key" -H "Content-Type: application/json" \
  -d '{"text": "Refund Rahul Mehta at rahul@example.com", "team": "finance"}'
```

```json
{
  "request_id": "6f1c…",
  "entities": [
    {"start": 7, "end": 18, "label": "PERSON", "score": 0.98, "source": "ner"},
    {"start": 22, "end": 39, "label": "EMAIL", "score": 0.99, "source": "rule"}
  ],
  "action": "redact",
  "needs_review": false,
  "redacted_text": "Refund [PERSON_1] at [EMAIL_1]",
  "placeholders": {"PERSON": 1, "EMAIL": 1},
  "latency_ms": 9.8,
  "model_version": "ner=onnx:model.int8.onnx,secret=sklearn"
}
```

```bash
curl -s localhost:8000/v1/proxy/chat -H "X-API-Key: dev-key" -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Draft a reply to rahul@example.com about the refund"}]}'
```

The response's `sent_to_llm` shows exactly what left the gateway. `reply` has the original values restored.

## Errors

| Status | When |
|---|---|
| 401 | Missing or invalid API key |
| 413 | Text over 50,000 characters, or upload over 10 MB |
| 415 | Unsupported upload type |
| 422 | Request body fails validation |
| 429 | Rate limit exceeded (`Retry-After: 60`) |
| 502 | LLM provider error (proxy only) |
| 500 | Unhandled error; body includes `request_id` |
