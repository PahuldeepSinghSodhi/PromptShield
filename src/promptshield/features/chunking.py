"""Split long texts into overlapping token windows and merge entities back.

The NER model sees at most ``max_len`` tokens. Long prompts and uploaded files are
cut into windows that overlap by ``stride`` tokens so an entity on a boundary is
seen whole in at least one window. Entity offsets are mapped back to the original
document's character positions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from promptshield.types import Entity


@dataclass
class Window:
    input_ids: np.ndarray  # (seq,) int64, includes [CLS]/[SEP]
    offsets: np.ndarray  # (seq, 2) char offsets in the ORIGINAL text; (0, 0) for special tokens
    core: tuple[int, int]  # char range this window is responsible for (to drop duplicates)


def make_windows(tokenizer, text: str, max_len: int = 512, stride: int = 64) -> list[Window]:
    """``tokenizer`` is a ``tokenizers.Tokenizer`` (fast, Rust) loaded from tokenizer.json."""
    enc = tokenizer.encode(text, add_special_tokens=False)
    ids = np.asarray(enc.ids, dtype=np.int64)
    offs = np.asarray(enc.offsets, dtype=np.int64).reshape(-1, 2)
    cls_id, sep_id = tokenizer.token_to_id("[CLS]"), tokenizer.token_to_id("[SEP]")
    body = max_len - 2
    if len(ids) <= body:
        starts = [0]
    else:
        step = body - stride
        starts = list(range(0, len(ids) - stride, step))
    windows = []
    for i, s in enumerate(starts):
        e = min(s + body, len(ids))
        w_ids = np.concatenate([[cls_id], ids[s:e], [sep_id]])
        w_offs = np.concatenate([[[0, 0]], offs[s:e], [[0, 0]]])
        # each window owns the middle of its overlap with neighbours
        lo = 0 if i == 0 else int(offs[s + stride // 2][0])
        hi = len(text) if e == len(ids) else int(offs[e - stride // 2][0])
        windows.append(Window(w_ids, w_offs, (lo, hi)))
        if e == len(ids):
            break
    return windows


def merge_window_entities(per_window: list[list[Entity]], windows: list[Window]) -> list[Entity]:
    """Keep an entity only from the window whose core contains its start; dedupe."""
    seen, out = set(), []
    for ents, w in zip(per_window, windows, strict=True):
        lo, hi = w.core
        for e in ents:
            if lo <= e.start < hi and (e.start, e.end, e.label) not in seen:
                seen.add((e.start, e.end, e.label))
                out.append(e)
    return sorted(out, key=lambda e: e.start)
