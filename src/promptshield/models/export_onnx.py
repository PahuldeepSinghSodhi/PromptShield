"""Export the fine-tuned NER model to ONNX and quantize it to INT8.

    python -m promptshield.models.export_onnx
"""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification

ROOT = Path(__file__).resolve().parents[3]
NER_DIR = ROOT / "artifacts" / "ner"


def export(model_dir: Path = NER_DIR) -> dict:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    model = AutoModelForTokenClassification.from_pretrained(model_dir).eval()
    dummy_ids = torch.ones((1, 16), dtype=torch.long)
    dummy_mask = torch.ones((1, 16), dtype=torch.long)
    fp32 = model_dir / "model.onnx"
    torch.onnx.export(
        model,
        (dummy_ids, dummy_mask),
        str(fp32),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
                      "logits": {0: "batch", 1: "seq"}},
        opset_version=17,
        dynamo=False,
    )
    int8 = model_dir / "model.int8.onnx"
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8)
    sizes = {p.name: round(p.stat().st_size / 1e6, 1) for p in (fp32, int8)}
    sizes["pytorch_fp32"] = round(sum(p.stat().st_size for p in model_dir.glob("*.safetensors")) / 1e6, 1)
    return sizes


if __name__ == "__main__":
    print(export())
