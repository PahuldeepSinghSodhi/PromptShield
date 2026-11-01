---
title: PromptShield
emoji: 🛡️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
short_description: ML gateway that redacts PII and blocks secrets before LLMs
---

# PromptShield demo

An ML gateway that detects personal data and secrets in prompts (PyTorch DistilBERT NER served as ONNX INT8, a scikit-learn secret detector and checksum rules), then allows, redacts or blocks the request before it reaches an LLM.

- **Playground**: paste a prompt and see what would be sent to the LLM. The demo uses a mock LLM that echoes the redacted prompt, and placeholders are restored in the reply.
- **API docs**: `/docs`. The demo API key is `dev-key`.
- Storage is ephemeral SQLite, so monitoring charts reset when the Space restarts.

Source, results and documentation: https://github.com/PahuldeepSinghSodhi/PromptShield
