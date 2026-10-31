# Data card

## NER dataset (`data/processed/ner.parquet`)

| | |
|---|---|
| Rows | 68,807 (train 49,469 · val 9,063 · test 10,275) |
| Sources | ai4privacy `pii-masking-400k`, English subset: 56,103 rows after filtering · synthetic generator: 12,704 rows |
| Labels | PERSON, ADDRESS, ID_NUMBER, FINANCIAL, EMAIL, PHONE, USERNAME, DATE, CREDENTIAL (BIO tagging) |
| Entities | 99,343 in total; most common PERSON (28k), rarest CREDENTIAL (2.3k) |
| Text length | median 122 chars, max 1,377 |

### How it was built (`python -m promptshield.data.build`)

1. **Ingest:** a seeded 60,000-row sample of the English rows from `ai4privacy/pii-masking-400k`, all splits.
2. **Label unification** (`configs/label_map.yaml`): the 17 source labels map onto 9 unified labels. Adjacent same-label spans separated only by punctuation or whitespace are merged (`GIVENNAME`+`SURNAME` → one `PERSON`). Unmapped labels are counted in the quality report; in this build there were none.
3. **Noise filter:** about 30% of the source's English rows have an empty privacy mask. Some of them still contain identifiers, such as unlabeled 16-digit account numbers. Training on those as "no PII" would teach the model to miss them, so **3,896 unlabeled rows containing long digit runs or emails were dropped**. The remaining unlabeled rows stay in as genuine negatives.
4. **Synthetic records** (`data/synth.py`): Indian and international formats the public set under-represents. These cover PAN, Aadhaar-style IDs (Verhoeff-valid, fake), IFSC, UPI IDs, +91 phone numbers and account numbers. Every value comes from Faker or a random generator. There are three kinds of record:
   - **Compositional (55%):** random intro + 1–3 PII clauses (4–11 phrasings per entity type) + a request. This gives about 8,000 distinct sentence skeletons.
   - **Fixed business templates (30%):** 20 hand-written prompts.
   - **Clean prompts (15%):** parametric technical and business prompts with no PII ("Our S3 upload fails with AccessDenied…"). They teach the model that short codes like S3, EC2, v2 and P1 are not addresses or IDs.

   The first version used only the 20 fixed templates. With the grouped split, the score then depended on which 3 templates happened to be held out: synthetic test F1 was 0.65 while public data scored 0.91. The compositional generator removes that brittleness.
5. **Cleaning:** per-character NFKC normalization that re-maps span offsets; removing zero-width characters; deduplicating on text.
6. **Validation** (pandera):
   - spans fall inside the text, sorted, with no overlaps
   - labels come from the allowed set
   - no span is empty
7. **Split** grouped by a *template hash* (the text with entity values replaced by their labels). Rows generated from the same template can't land in both train and test. Verified by `check_no_leakage`.

## Secret-detector dataset (`data/processed/secrets.parquet`)

| | |
|---|---|
| Texts | 38,227 short prompts / config snippets / log lines, each with one planted value |
| Positive rate | 0.474 |
| Positives (fake, real formats) | AWS-style keys, GitHub/Slack/vendor-style tokens, `sk-` keys, JWTs, hex API keys, passwords, DB URLs with credentials. Includes cases with no keyword nearby. |
| Hard negatives | git SHAs, UUIDs, hashes, base64 images, URLs, identifiers, version strings, file paths, placeholders (`your-api-key-here`, `${DB_PASSWORD}`), env-var *names*, PAN/IFSC/order IDs, token counts |
| Split | stratified by generator kind (values are random, so there is no value leakage) |

Training examples are produced by running the **serving-time candidate extractor** over each text and labelling each candidate by overlap with the planted span. The model therefore sees exactly what it sees in production.

### Out-of-distribution challenge set

`generate_secret_challenge()` uses contexts and token formats that never appear in training, for example GitLab `glpat-` tokens, `mongodb+srv://` URLs, credentials with no keyword nearby, CI template references and `os.getenv(...)` calls. It is regenerated at evaluation time and **never used for training or threshold selection**.

## Privacy

- Every value in the synthetic data is fake. Secret-shaped strings are generated at runtime, never committed.
- `data/` is git-ignored and versioned with DVC.
- Reviewer feedback is stored **masked**. Before retraining, it is *rehydrated* with fresh fake values (`pipelines/feedback.py`), so no real PII ever reaches training data.

## Known limitations

- English only.
- The ai4privacy text is LLM-generated, so its style differs from real employee prompts. The drift monitor is there to catch that gap in production.
- Indian formats come only from synthetic templates (20 PII templates). A model can memorize template phrasing, so the grouped split holds whole templates out of training to measure that.
