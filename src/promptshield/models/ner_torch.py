"""Fine-tune DistilBERT for PII token classification with a hand-written PyTorch loop.

    python -m promptshield.models.ner_torch --epochs 3 --batch-size 32
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time

import numpy as np
import pandas as pd
import torch
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer, get_linear_schedule_with_warmup

from promptshield.models.ner_common import BIO_LABELS, ID2LABEL, IGNORE, LABEL2ID, align_labels
from promptshield.paths import ROOT

DATA = ROOT / "data" / "processed" / "ner.parquet"
OUT_DIR = ROOT / "artifacts" / "ner"
METRICS = ROOT / "reports" / "ner_metrics.json"
BASE_MODEL = "distilbert-base-cased"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class NerDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int):
        enc = tokenizer(
            df["text"].tolist(), truncation=True, max_length=max_len, return_offsets_mapping=True
        )
        self.input_ids = enc["input_ids"]
        self.labels = [
            align_labels(offs, list(spans))
            for offs, spans in zip(enc["offset_mapping"], df["spans"], strict=True)
        ]

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, i: int) -> tuple[list[int], list[int]]:
        return self.input_ids[i], self.labels[i]


def make_collate(pad_id: int):
    def collate(batch):
        width = max(len(ids) for ids, _ in batch)
        ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
        labels = torch.full((len(batch), width), IGNORE, dtype=torch.long)
        mask = torch.zeros((len(batch), width), dtype=torch.long)
        for i, (x, y) in enumerate(batch):
            ids[i, : len(x)] = torch.tensor(x)
            labels[i, : len(y)] = torch.tensor(y)
            mask[i, : len(x)] = 1
        return ids, mask, labels

    return collate


def class_weights(ds: NerDataset, smoothing: float = 0.3) -> torch.Tensor:
    """Inverse-frequency weights, softened so rare tags help without dominating."""
    counts = np.bincount(
        [t for seq in ds.labels for t in seq if t != IGNORE], minlength=len(BIO_LABELS)
    ).astype(float)
    w = (counts.sum() / np.maximum(counts, 1)) ** smoothing
    return torch.tensor(w / w.mean(), dtype=torch.float)


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[dict, torch.Tensor, torch.Tensor]:
    """Entity-level metrics (seqeval) plus raw logits/labels for calibration."""
    model.eval()
    y_true, y_pred, all_logits, all_labels = [], [], [], []
    for ids, mask, labels in loader:
        ids, mask = ids.to(device), mask.to(device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(input_ids=ids, attention_mask=mask).logits.float()
        pred = logits.argmax(-1).cpu()
        for p_row, l_row in zip(pred, labels, strict=True):
            keep = l_row != IGNORE
            y_true.append([ID2LABEL[int(t)] for t in l_row[keep]])
            y_pred.append([ID2LABEL[int(t)] for t in p_row[keep]])
        keep = labels != IGNORE
        all_logits.append(logits.cpu()[keep])
        all_labels.append(labels[keep])
    metrics = {
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "f1": round(f1_score(y_true, y_pred), 4),
        "per_entity": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
    }
    return metrics, torch.cat(all_logits), torch.cat(all_labels)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Temperature scaling: one scalar that makes token confidences calibrated."""
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.05, max_iter=200)
    nll = nn.CrossEntropyLoss()

    def step():
        opt.zero_grad()
        loss = nll(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    opt.step(step)
    return float(log_t.exp())


def expected_calibration_error(probs: torch.Tensor, labels: torch.Tensor, bins: int = 15) -> float:
    conf, pred = probs.max(-1)
    correct = (pred == labels).float()
    edges = torch.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.float().mean().item() * abs(conf[m].mean().item() - correct[m].mean().item())
    return round(ece, 4)


def train(args) -> dict:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_parquet(DATA)
    df["spans"] = df["spans"].apply(list)
    if args.max_train:
        df = pd.concat(
            [df[df.split == "train"].sample(min(args.max_train, (df.split == "train").sum()), random_state=args.seed),
             df[df.split != "train"]]
        )

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    splits = {s: NerDataset(df[df.split == s], tok, args.max_len) for s in ("train", "val", "test")}
    collate = make_collate(tok.pad_token_id)
    loaders = {
        s: DataLoader(d, batch_size=args.batch_size, shuffle=(s == "train"), collate_fn=collate)
        for s, d in splits.items()
    }

    model = AutoModelForTokenClassification.from_pretrained(
        BASE_MODEL, num_labels=len(BIO_LABELS), id2label=ID2LABEL, label2id=LABEL2ID
    ).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(splits["train"]).to(device), ignore_index=IGNORE)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = args.epochs * math.ceil(len(splits["train"]) / args.batch_size)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total_steps), total_steps)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    best_f1, bad_epochs, history = -1.0, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, running = time.time(), 0.0
        for step, (ids, mask, labels) in enumerate(loaders["train"], 1):
            ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(input_ids=ids, attention_mask=mask).logits
                loss = loss_fn(logits.float().view(-1, logits.size(-1)), labels.view(-1))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item()
            if step % 200 == 0:
                print(f"epoch {epoch} step {step} loss {running / step:.4f}")

        val_metrics, _, _ = evaluate(model, loaders["val"], device)
        history.append({"epoch": epoch, "train_loss": round(running / step, 4), "val_f1": val_metrics["f1"],
                        "seconds": round(time.time() - t0, 1)})
        print(history[-1])
        if val_metrics["f1"] > best_f1:
            best_f1, bad_epochs = val_metrics["f1"], 0
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(OUT_DIR)
            tok.save_pretrained(OUT_DIR)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print("early stopping")
                break

    # reload best checkpoint, calibrate on val, report on test
    model = AutoModelForTokenClassification.from_pretrained(OUT_DIR).to(device)
    _, val_logits, val_labels = evaluate(model, loaders["val"], device)
    temperature = fit_temperature(val_logits, val_labels)
    test_metrics, test_logits, test_labels = evaluate(model, loaders["test"], device)
    calib = {
        "temperature": round(temperature, 4),
        "ece_before": expected_calibration_error(test_logits.softmax(-1), test_labels),
        "ece_after": expected_calibration_error((test_logits / temperature).softmax(-1), test_labels),
    }
    # serving windows must match the training sequence length: on 512-token windows a model
    # fine-tuned at 256 lost ~14% of entities in long documents (see docs/decisions.md)
    (OUT_DIR / "promptshield.json").write_text(json.dumps({"temperature": temperature, "max_len": args.max_len}))

    summary = {"base_model": BASE_MODEL, "history": history, "test": test_metrics, "calibration": calib,
               "n_train": len(splits["train"]), "device": str(device)}
    METRICS.parent.mkdir(parents=True, exist_ok=True)
    METRICS.write_text(json.dumps(summary, indent=2, default=_json_default))
    return summary


def _json_default(o):
    return o.item() if hasattr(o, "item") else str(o)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--max-train", type=int, default=0, help="subsample training rows (0 = all)")
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    s = train(ap.parse_args())
    t = s["test"]
    print(json.dumps({"precision": t["precision"], "recall": t["recall"], "f1": t["f1"],
                      "calibration": s["calibration"], "history": s["history"]}, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
