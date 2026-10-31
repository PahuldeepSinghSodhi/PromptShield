"""BIO label scheme, label alignment and span decoding shared by training and serving.

Pure NumPy so the ONNX serving path does not need PyTorch.
"""

from __future__ import annotations

import numpy as np

from promptshield.types import NER_LABELS, Entity

BIO_LABELS = ["O"] + [f"{p}-{lab}" for lab in NER_LABELS for p in ("B", "I")]
LABEL2ID = {lab: i for i, lab in enumerate(BIO_LABELS)}
ID2LABEL = dict(enumerate(BIO_LABELS))
IGNORE = -100


def align_labels(offsets: list[tuple[int, int]], spans: list[dict]) -> list[int]:
    """Assign a BIO id to every token from character-level entity spans.

    Special tokens have offset (0, 0) and get IGNORE so they don't count in the loss.
    """
    ids, j = [], 0
    spans = sorted(spans, key=lambda s: s["start"])
    prev_entity = None
    for s, e in offsets:
        if s == e:
            ids.append(IGNORE)
            continue
        while j < len(spans) and spans[j]["end"] <= s:
            j += 1
        if j < len(spans) and spans[j]["start"] < e and s < spans[j]["end"]:
            lab = spans[j]["label"]
            tag = "I" if prev_entity == j else "B"
            ids.append(LABEL2ID[f"{tag}-{lab}"])
            prev_entity = j
        else:
            ids.append(LABEL2ID["O"])
            prev_entity = None
    return ids


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / temperature
    z = z - z.max(axis=-1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=-1, keepdims=True)


def decode(
    text: str, offsets: np.ndarray, probs: np.ndarray, char_offset: int = 0
) -> list[Entity]:
    """Turn per-token probabilities into character-level entities.

    ``offsets`` are token offsets inside ``text``; ``char_offset`` shifts them when
    ``text`` is a chunk of a longer document.
    """
    pred = probs.argmax(-1)
    conf = probs.max(-1)
    entities: list[Entity] = []
    cur = None  # [label, start, end, [scores]]

    def flush():
        if cur is not None:
            s, e = cur[1], cur[2]
            entities.append(
                Entity(s + char_offset, e + char_offset, cur[0], float(np.mean(cur[3])), "ner", text[s:e])
            )

    for (s, e), p, c in zip(offsets.tolist(), pred.tolist(), conf.tolist(), strict=True):
        if s == e:
            continue
        tag = ID2LABEL[p]
        if tag == "O":
            flush()
            cur = None
            continue
        prefix, lab = tag.split("-", 1)
        # continue the current entity on I- of the same type, or on a sub-word directly attached
        attached = cur is not None and s == cur[2]
        if cur is not None and cur[0] == lab and (prefix == "I" or attached):
            cur[2] = e
            cur[3].append(c)
        else:
            flush()
            cur = [lab, s, e, [c]]
    flush()
    return entities
