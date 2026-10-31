"""Detection engine used by the API: NER (ONNX Runtime or PyTorch) + secrets + rules."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from promptshield.features.chunking import make_windows, merge_window_entities
from promptshield.models.fusion import fuse
from promptshield.models.ner_common import decode, softmax
from promptshield.models.rules import detect_rules
from promptshield.types import Entity

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[3]
NER_DIR = ROOT / "artifacts" / "ner"
SECRET_PATH = ROOT / "artifacts" / "secret_detector.joblib"


class NerBackend:
    """Runs token classification over padded batches of windows. Subclasses implement ``logits``."""

    name = "base"

    def __init__(self, model_dir: Path, batch_size: int = 16):
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        # tokenizer.json keeps the training-time truncation (256 tokens); chunking needs the
        # full token stream or long documents get silently cut to their first window
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        meta_path = model_dir / "promptshield.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        self.temperature = float(meta.get("temperature", 1.0))
        self.max_len = int(meta.get("max_len", 256))  # window size = training sequence length
        self.batch_size = batch_size

    def logits(self, ids: np.ndarray, mask: np.ndarray) -> np.ndarray:  # (b, seq, labels)
        raise NotImplementedError

    def detect_many(self, texts: list[str]) -> list[list[Entity]]:
        jobs = []  # (text index, window)
        for ti, t in enumerate(texts):
            for w in make_windows(self.tokenizer, t, self.max_len):
                jobs.append((ti, w))
        per_text_windows: list[list] = [[] for _ in texts]
        per_text_ents: list[list[list[Entity]]] = [[] for _ in texts]
        for b in range(0, len(jobs), self.batch_size):
            batch = jobs[b : b + self.batch_size]
            width = max(len(w.input_ids) for _, w in batch)
            ids = np.zeros((len(batch), width), dtype=np.int64)
            mask = np.zeros((len(batch), width), dtype=np.int64)
            for i, (_, w) in enumerate(batch):
                ids[i, : len(w.input_ids)] = w.input_ids
                mask[i, : len(w.input_ids)] = 1
            probs = softmax(self.logits(ids, mask), self.temperature)
            for i, (ti, w) in enumerate(batch):
                n = len(w.input_ids)
                ents = decode(texts[ti], w.offsets, probs[i, :n])
                per_text_windows[ti].append(w)
                per_text_ents[ti].append(ents)
        return [merge_window_entities(e, w) for e, w in zip(per_text_ents, per_text_windows, strict=True)]


class OnnxNer(NerBackend):
    name = "onnx"

    def __init__(self, model_dir: Path, model_file: str = "model.int8.onnx", threads: int = 0, **kw):
        super().__init__(model_dir, **kw)
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads:
            opts.intra_op_num_threads = threads
        self.model_file = model_file
        self.session = ort.InferenceSession(
            str(model_dir / model_file), opts, providers=["CPUExecutionProvider"]
        )

    def logits(self, ids, mask):
        return self.session.run(["logits"], {"input_ids": ids, "attention_mask": mask})[0]


class TorchNer(NerBackend):
    name = "torch"

    def __init__(self, model_dir: Path, device: str = "cpu", **kw):
        super().__init__(model_dir, **kw)
        import torch
        from transformers import AutoModelForTokenClassification

        self.torch = torch
        self.device = torch.device(device)
        self.model = AutoModelForTokenClassification.from_pretrained(model_dir).to(self.device).eval()

    def logits(self, ids, mask):
        with self.torch.inference_mode():
            out = self.model(
                input_ids=self.torch.from_numpy(ids).to(self.device),
                attention_mask=self.torch.from_numpy(mask).to(self.device),
            ).logits
        return out.float().cpu().numpy()


class DetectionEngine:
    def __init__(self, ner: NerBackend | None, secret_detector=None):
        self.ner = ner
        self.secret = secret_detector

    @property
    def model_version(self) -> str:
        parts = [f"ner={self.ner.name}:{getattr(self.ner, 'model_file', 'fp32')}" if self.ner else "ner=none"]
        parts.append("secret=sklearn" if self.secret else "secret=none")
        return ",".join(parts)

    def detect_many(self, texts: list[str]) -> list[list[Entity]]:
        ner_out = self.ner.detect_many(texts) if self.ner else [[] for _ in texts]
        results = []
        for text, ner_ents in zip(texts, ner_out, strict=True):
            secret_ents = self.secret.detect(text) if self.secret else []
            results.append(fuse(ner_ents, secret_ents, detect_rules(text)))
        return results

    def detect(self, text: str) -> list[Entity]:
        return self.detect_many([text])[0]


def load_engine(backend: str = "onnx", model_file: str = "model.int8.onnx") -> DetectionEngine:
    """Load whatever artifacts exist; missing ones degrade gracefully (logged)."""
    ner = None
    if backend != "none" and (NER_DIR / "tokenizer.json").exists():
        try:
            ner = TorchNer(NER_DIR) if backend == "torch" else OnnxNer(NER_DIR, model_file=model_file)
        except Exception as exc:  # noqa: BLE001 - keep the gateway up with rules + secrets
            log.warning("NER backend %s unavailable: %s", backend, exc)
    secret = None
    if SECRET_PATH.exists():
        from promptshield.models.secret_detector import SecretDetector

        secret = SecretDetector.load(SECRET_PATH)
    return DetectionEngine(ner, secret)
