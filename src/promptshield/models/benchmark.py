"""Accuracy + latency benchmark across inference backends.

Compares PyTorch FP32 (CPU), ONNX FP32 and ONNX INT8 on the same test texts:
entity-level F1 (character-span exact match), p50/p95 latency at several batch
sizes, and model size.

    python -m promptshield.models.benchmark --n 500
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from promptshield.serving.inference import NER_DIR, OnnxNer, TorchNer

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data" / "processed" / "ner.parquet"
OUT = ROOT / "reports" / "benchmark.json"


def span_f1(gold: list[list[dict]], pred: list[list]) -> dict:
    tp = fp = fn = 0
    for g, p in zip(gold, pred, strict=True):
        gs = {(s["start"], s["end"], s["label"]) for s in g}
        ps = {(e.start, e.end, e.label) for e in p}
        tp += len(gs & ps)
        fp += len(ps - gs)
        fn += len(gs - ps)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}


def latency(backend, texts: list[str], batch: int, repeats: int = 3) -> dict:
    backend.batch_size = batch
    backend.detect_many(texts[:batch])  # warm-up
    per_call = []
    for _ in range(repeats):
        for i in range(0, len(texts), batch):
            t0 = time.perf_counter()
            backend.detect_many(texts[i : i + batch])
            per_call.append((time.perf_counter() - t0) * 1000)
    arr = np.asarray(per_call)
    return {
        "batch": batch,
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "texts_per_sec": round(batch * 1000 / float(arr.mean()), 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    df = pd.read_parquet(DATA)
    test = df[df.split == "test"].sample(args.n, random_state=0)
    texts, gold = test["text"].tolist(), [list(s) for s in test["spans"]]

    backends = {
        "pytorch_fp32_cpu": lambda: TorchNer(NER_DIR, device="cpu"),
        "onnx_fp32": lambda: OnnxNer(NER_DIR, model_file="model.onnx", threads=args.threads),
        "onnx_int8": lambda: OnnxNer(NER_DIR, model_file="model.int8.onnx", threads=args.threads),
    }
    sizes = {
        "pytorch_fp32_cpu": sum(p.stat().st_size for p in NER_DIR.glob("*.safetensors")),
        "onnx_fp32": (NER_DIR / "model.onnx").stat().st_size,
        "onnx_int8": (NER_DIR / "model.int8.onnx").stat().st_size,
    }

    results = {}
    for name, make in backends.items():
        if name.startswith("pytorch"):
            import torch

            torch.set_num_threads(args.threads)
        b = make()
        preds = b.detect_many(texts)
        sources = test["source"].tolist()
        per_source = {
            src: span_f1([g for g, s in zip(gold, sources, strict=True) if s == src],
                         [p for p, s in zip(preds, sources, strict=True) if s == src])
            for src in sorted(set(sources))
        }
        results[name] = {
            "accuracy": span_f1(gold, preds),
            "per_source": per_source,
            "size_mb": round(sizes[name] / 1e6, 1),
            "latency": [latency(b, texts[:128], bs) for bs in (1, 8, 32)],
        }
        print(name, results[name]["accuracy"], results[name]["latency"][0])

    base = results["pytorch_fp32_cpu"]
    summary = {
        name: {
            "f1": r["accuracy"]["f1"],
            "f1_by_source": {s: v["f1"] for s, v in r["per_source"].items()},
            "f1_delta_vs_pytorch": round(r["accuracy"]["f1"] - base["accuracy"]["f1"], 4),
            "p95_ms_batch1": r["latency"][0]["p95_ms"],
            "speedup_p95_batch1": round(base["latency"][0]["p95_ms"] / r["latency"][0]["p95_ms"], 2),
            "size_mb": r["size_mb"],
        }
        for name, r in results.items()
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"n_texts": args.n, "threads": args.threads, "summary": summary,
                               "details": results}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
