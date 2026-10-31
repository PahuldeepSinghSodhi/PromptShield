# Model card: PromptShield detectors

## Intended use

PromptShield detects personal data and credentials in text **before** that text is sent to an external LLM, or shared as a file. It then allows, redacts or blocks the request according to a configurable policy.

| Use | Status |
|---|---|
| Screening employee prompts / uploads in English | Intended |
| A compliance aid for data-protection programmes (DPDP Act, GDPR) | Intended, alongside human review |
| Sole control for regulated data, legal decisions about individuals, non-English text | **Out of scope** |

## Components

| Component | Model | Trained on |
|---|---|---|
| PII tagger | `distilbert-base-cased` fine-tuned for token classification (19 BIO tags, 9 entity types), exported to ONNX and dynamically quantized to INT8 | 48,834 training rows (ai4privacy English + synthetic), 5 epochs, AMP on an RTX 4050, ~3.5 min per epoch |
| Secret detector | `HistGradientBoostingClassifier` on 29 NumPy string/context features; chosen over LogisticRegression by validation average precision | 38,227 synthetic texts (fake credentials and look-alikes) |
| Rules | Luhn (cards), Verhoeff (Aadhaar-style IDs), PAN, IFSC, UPI and email formats | n/a |

## Evaluation

### PII tagger (entity-level, seqeval, test split of 10,855 rows; templates never seen in training)

| Model | Precision | Recall | F1 |
|---|---|---|---|
| scikit-learn baseline (handcrafted token features + linear model) | 0.850 | 0.710 | 0.774 |
| **DistilBERT (PyTorch)** | **0.932** | **0.961** | **0.946** |

| Entity | P | R | F1 | Support |
|---|---|---|---|---|
| EMAIL | 0.999 | 0.999 | 0.999 | 1,657 |
| PHONE | 0.973 | 0.978 | 0.976 | 1,183 |
| PERSON | 0.964 | 0.984 | 0.974 | 4,036 |
| USERNAME | 0.969 | 0.978 | 0.974 | 1,524 |
| ADDRESS | 0.896 | 0.949 | 0.922 | 2,563 |
| ID_NUMBER | 0.902 | 0.929 | 0.915 | 2,025 |
| CREDENTIAL | 0.902 | 0.919 | 0.910 | 320 |
| DATE | 0.883 | 0.904 | 0.893 | 773 |
| FINANCIAL | 0.835 | 0.919 | 0.875 | 1,271 |

By source (character-exact spans, 1,000-row sample): public ai4privacy text **0.92–0.93**, synthetic text **0.99**. The public set is the harder and more realistic number.

**Calibration:** temperature scaling (T = 1.51) fitted on validation reduced token-level ECE on test from 0.0069 to **0.0023**. Confidence therefore feeds the policy thresholds directly: below 0.50 is ignored, below 0.80 goes to human review.

**Serving:** ONNX INT8 matches FP32 F1 within ±0.5 points. It is 4× smaller (66 MB) and was 2.6–6× faster than PyTorch on CPU across runs, at roughly 20–24 ms p95 per prompt. See `decisions.md` §8.

### Secret detector

| Set | Precision | Recall | F1 |
|---|---|---|---|
| In-distribution test | 1.000 | 1.000 | 1.000 |
| **Out-of-distribution challenge** (unseen token formats and contexts) | **1.000** | **0.847** | **0.917** |

The in-distribution score is reported but **not** trusted on its own: the challenge set exists because an earlier version also scored 1.00 there while scoring only 0.60–0.74 on unseen formats (`decisions.md` §2). The threshold is chosen on validation for ≥ 98% precision, because a false positive blocks a user's request.

## Limitations and risks

- **English only.** Other languages will be under-detected.
- **Domain gap.** ai4privacy text is LLM-generated and synthetic data is templated. Real prompts (code, logs, chat shorthand) differ, so the drift monitor and review queue exist to catch and fix this in production.
- **Context-free hex secrets** (e.g. a 32-char hex webhook secret with no keyword nearby) are indistinguishable from hashes and are missed. That accounts for most of the challenge-set recall gap.
- **FINANCIAL has the lowest precision (0.835).** Long digit strings such as order numbers and reference IDs sometimes get tagged as account numbers. Redaction is the safe failure mode here.
- **Synthetic Indian formats** follow official structures, but real-world phrasing is more varied than any generator.
- **Not a guarantee.** Detection reduces risk, it does not eliminate it. Policies should combine this with access controls and training.

## Fairness considerations

Names come from Faker's `en_IN`, `en_US` and `en_GB` locales plus the public dataset's names. Name recall may be lower for naming conventions under-represented there. A per-locale name-recall evaluation is a recommended next step before production use.
