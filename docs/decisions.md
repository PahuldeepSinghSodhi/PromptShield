# Engineering decisions

Short records of the trade-offs made while building PromptShield, with the evidence behind each one.

## 1. Two models plus rules, not one model

PII and secrets behave differently:
- **PII** (names, addresses) depends on language context. A transformer token classifier is the right tool.
- **Secrets** (API keys, passwords) depend on *string shape* (entropy, character classes, vendor prefixes) plus a little surrounding context. A gradient-boosted classifier on 29 NumPy features handles them in microseconds, and its features are easy to explain.
- **Checksum formats** (Luhn card numbers, Verhoeff Aadhaar-style IDs, PAN, IFSC) are certain when they validate, so a rule beats any model there.

The fusion layer resolves overlaps. Validated rule matches win, then the more dangerous label, then the higher score. Rules win because the secret detector once labelled a valid PAN as a secret, which would have *blocked* a routine KYC prompt.

## 2. Secret detector: an honest evaluation changed the model

The first version scored **1.00 F1** on its synthetic test set. That was a warning sign, not a result: the generator's keywords and prefixes gave the answer away. So I added an **out-of-distribution challenge set**. It uses token formats (GitLab `glpat-`), contexts (no keyword nearby) and look-alikes (`<INSERT_KEY_HERE>`, `os.getenv('X')`, UUIDs after "key") that training never contains. It is never used for training or threshold selection.

| Iteration | Change | Challenge-set F1 |
|---|---|---|
| 1 | baseline features, grouped split | 0.60 (HGB) / 0.74 (LogReg) |
| 2 | vendor-prefix shape, UUID/URL/template-wrapper features; keyword-free positives; stratified split | 0.80 |
| 3 | general hard-negative categories (identifiers next to key-like words, env-var names, PAN/IFSC) | **0.92** (precision 1.00, recall 0.85) |

The remaining misses are almost all 32-char hex secrets with no keyword nearby. Without context these are indistinguishable from MD5 hashes, so it is a documented limitation rather than something to overfit.

**Split choice.** NER data is split *grouped by template*, because the same template must not appear in both train and test. Secret values are random, so there is no value leakage; that split is *stratified by generator kind*. A grouped split there had put entire kinds (e.g. all placeholders) only in test. Validation showed 1.00 while test showed 0.68 precision, and that gap was a split artefact, not generalization.

## 3. Threshold policy: precision first

A false positive on a secret **blocks** a user's request. Each one costs trust in the tool, so the threshold is the *highest* one that keeps recall maximal while precision ≥ 0.98 on validation (`pick_threshold`). It is chosen on validation only.

## 4. Label noise in the public dataset

About 30% of the English ai4privacy rows have an empty privacy mask. Some still contain obvious identifiers, for example an unlabeled 16-digit account number. Training on those as "no PII" teaches the model to miss exactly those values. I dropped 3,896 unlabeled rows that contain long digit runs or emails and kept the rest as genuine negatives.

## 5. PyTorch loop written by hand

The training loop is written explicitly rather than with HF `Trainer`:
- subword label alignment
- AMP mixed precision on the RTX 4050
- gradient clipping and a linear warm-up schedule
- softened inverse-frequency class weights (CREDENTIAL and DATE are rare)
- early stopping on validation entity-F1
- temperature scaling after training

Writing it out makes every one of these choices visible and testable.

## 6. Serving window = training sequence length

**Bug found in testing:** a long document (60 repeated complaints, 240 entities) returned only 23 entities.

- **Cause 1:** `tokenizer.json` saved the training-time truncation (256 tokens), so the serving tokenizer cut every document to its first window *silently*. Fixed with `no_truncation()`; a regression test now covers it.
- **Cause 2:** with truncation off, 512-token windows still recovered only 206/240 entities, because the model was fine-tuned on ≤ 256 tokens.

| Window (tokens) | Entities found / 240 |
|---|---|
| 512 | 206 |
| **256 (= training length)** | **239** |
| 128 | 240 |

256 is used: it matches training and needs half as many windows as 128. The model writes its training length into `promptshield.json`, and serving reads it, so the two can't drift apart again.

## 7. ONNX Runtime INT8 for serving

PyTorch is only needed for training. Serving uses ONNX Runtime with dynamic INT8 quantization:
- the model goes from 261 MB to 66 MB
- the Docker image has no PyTorch at all
- CPU latency and accuracy figures are in `reports/benchmark.json` and the README

## 8. Privacy by construction

- The audit log stores a SHA-256 of the text, entity counts, confidences and latency. It never stores the text or the values.
- Placeholder maps live in memory for one request only.
- Batch-scan reports contain counts and row/column locations, never values.
- Reviewer feedback is stored *masked with the corrected labels*. For retraining it is *rehydrated* with fresh fake values, so corrections improve the model without any real PII entering the training data.
