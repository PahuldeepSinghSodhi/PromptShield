# Runbook

## Rebuild data and models from scratch

```bash
python -m promptshield.data.build            # or: dvc repro build_dataset
python -m promptshield.models.secret_detector
python -m promptshield.models.sk_token_baseline
python -m promptshield.models.ner_torch --epochs 3
python -m promptshield.models.export_onnx
python -m promptshield.models.benchmark --n 500
python scripts/build_drift_reference.py
```

Or simply `dvc repro`.

## Retrain with reviewer feedback

1. In the DB, set accepted feedback rows to `status='accepted'` (this is the review decision).
2. Run `python -m promptshield.pipelines.flows retrain`.
3. Read the report in `reports/retrain/`. If the gate failed, the live artifacts were restored automatically.

## Roll back a model

`artifacts.live-backup/` holds the models that were live before the last retrain:

```bash
rm -rf artifacts && mv artifacts.live-backup artifacts   # then restart the API
```

## Debugging inference

| Symptom | Check |
|---|---|
| `/v1/health` shows `ner_loaded: false` | `artifacts/ner/model.int8.onnx` and `tokenizer.json` exist. Startup logs show `NER backend … unavailable`. The gateway still runs on rules + secret detector. |
| Entity offsets look shifted | Input contains characters that NFKC changes. Offsets are always relative to the text the API received. Reproduce with `engine.detect(text)` in a REPL. |
| Entity split across two windows | Increase `stride` in `make_windows` (default 64 tokens). |
| Too many secret false positives | Raise the threshold in `secret_detector.joblib` (picked for ≥ 98% precision on val). Add the look-alike to `_negative_generators` and retrain. |
| p95 latency regression | `python -m promptshield.models.benchmark`. Check `intra_op_num_threads`, and the batch size used by the batch scanner (`TEXT_BATCH`). |
| `/v1/drift` returns `no_reference` | `python scripts/build_drift_reference.py` |
| Drift `significant` | Compare `entity_mix` vs `text_length` checks. New traffic types such as code-heavy prompts usually show in length and confidence first. Sample the review queue, add feedback, retrain. |

## Daily aggregates

`python -m promptshield.pipelines.flows daily_aggregates 2026-10-31`

The job is idempotent: re-running a day replaces that day's rows.
