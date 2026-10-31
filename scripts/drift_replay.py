"""Send traffic to a running gateway to demonstrate drift monitoring.

    python scripts/drift_replay.py --mode normal  --n 300   # PSI should stay low
    python scripts/drift_replay.py --mode shifted --n 300   # PSI should cross 0.25

"normal" replays held-out test rows (same distribution as the reference).
"shifted" traffic is code-heavy, long, secret-laden prompts, very unlike the
customer-support style text the models were trained and referenced on.
"""

import argparse
import random
import string
from pathlib import Path

import httpx
import pandas as pd

CODE_SNIPPET = '''
def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    client = boto3.client("s3", aws_access_key_id="{key}", region_name="ap-south-1")
    for bucket in client.list_buckets()["Buckets"]:
        print(bucket["Name"], bucket["CreationDate"])
    return cfg
'''


def shifted_prompt(rng: random.Random) -> str:
    key = "AKIA" + "".join(rng.choices(string.ascii_uppercase + "234567", k=16))
    body = CODE_SNIPPET.format(key=key) * rng.randint(2, 5)
    return "Refactor this and add retries:\n" + body + "\nlog: " + " ".join(
        f"{rng.getrandbits(64):016x}" for _ in range(rng.randint(5, 30))
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["normal", "shifted"], default="normal")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--key", default="dev-key")
    args = ap.parse_args()

    rng = random.Random(1)
    if args.mode == "normal":
        # held-out test rows: same distribution as the reference (built from val rows)
        df = pd.read_parquet(Path(__file__).resolve().parents[1] / "data" / "processed" / "ner.parquet")
        texts = df[df.split == "test"].sample(args.n, random_state=99)["text"].tolist()
    else:
        texts = [shifted_prompt(rng) for _ in range(args.n)]

    with httpx.Client(base_url=args.url, headers={"X-API-Key": args.key}, timeout=30) as c:
        for t in texts:
            c.post("/v1/scan", json={"text": t, "team": rng.choice(["engineering", "finance", "hr", "support"])})
        drift = c.get("/v1/drift", params={"window": args.n}).json()
    print(pd.DataFrame(drift.get("checks", {})).T if drift.get("checks") else drift)
    print("overall:", drift.get("overall"))


if __name__ == "__main__":
    main()
