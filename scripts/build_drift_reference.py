"""Run the validation split through the serving engine and save reference distributions.

    python scripts/build_drift_reference.py --n 2000
"""

import argparse
from pathlib import Path

import pandas as pd

from promptshield.monitoring.service import build_reference, save_reference
from promptshield.serving.inference import load_engine

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    args = ap.parse_args()

    df = pd.read_parquet(ROOT / "data" / "processed" / "ner.parquet")
    val = df[df.split == "val"].sample(min(args.n, (df.split == "val").sum()), random_state=0)
    texts = val["text"].tolist()
    engine = load_engine("onnx")
    entities = engine.detect_many(texts)
    save_reference(build_reference(entities, [len(t) for t in texts]))
    print(f"reference built from {len(texts)} texts, {sum(map(len, entities))} entities")


if __name__ == "__main__":
    main()
