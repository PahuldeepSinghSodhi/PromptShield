import random

import numpy as np

from promptshield.data.synth import fake_aadhaar, fake_pan, verhoeff_valid
from promptshield.features.chunking import Window, merge_window_entities
from promptshield.models.fusion import fuse
from promptshield.models.ner_common import BIO_LABELS, LABEL2ID, align_labels, decode
from promptshield.models.rules import detect_rules, luhn_valid
from promptshield.types import Entity


def test_luhn():
    assert luhn_valid("4539578763621486")
    assert not luhn_valid("4539578763621487")


def test_fake_aadhaar_passes_verhoeff_and_rule():
    rng = random.Random(3)
    for _ in range(20):
        a = fake_aadhaar(rng)
        assert verhoeff_valid(a.replace(" ", ""))
        ents = detect_rules(f"Aadhaar {a} please")
        assert any(e.label == "ID_NUMBER" for e in ents)


def test_pan_rule():
    pan = fake_pan(random.Random(1))
    assert any(e.label == "ID_NUMBER" and e.text == pan for e in detect_rules(f"PAN {pan}"))


def test_fusion_prefers_secret_over_overlap():
    a = Entity(0, 10, "USERNAME", 0.99, "ner")
    b = Entity(2, 12, "SECRET", 0.90, "secret")
    c = Entity(20, 25, "PERSON", 0.9, "ner")
    out = fuse([a, c], [b])
    assert [e.label for e in out] == ["SECRET", "PERSON"]


def test_align_labels_bio():
    text = "Hi Rahul Mehta"
    offsets = [(0, 0), (0, 2), (3, 8), (9, 14), (0, 0)]
    ids = align_labels(offsets, [{"start": 3, "end": 14, "label": "PERSON"}])
    assert ids == [-100, LABEL2ID["O"], LABEL2ID["B-PERSON"], LABEL2ID["I-PERSON"], -100]


def test_decode_round_trip():
    text = "Hi Rahul Mehta"
    offsets = np.array([(0, 0), (0, 2), (3, 8), (9, 14), (0, 0)])
    probs = np.zeros((5, len(BIO_LABELS)))
    for i, lab in enumerate(["O", "O", "B-PERSON", "I-PERSON", "O"]):
        probs[i, LABEL2ID[lab]] = 1.0
    ents = decode(text, offsets, probs)
    assert len(ents) == 1 and (ents[0].start, ents[0].end, ents[0].label) == (3, 14, "PERSON")


def test_window_merge_dedupes_overlap():
    w1 = Window(np.zeros(1), np.zeros((1, 2)), (0, 100))
    w2 = Window(np.zeros(1), np.zeros((1, 2)), (100, 200))
    e = Entity(90, 110, "PERSON", 0.9, "ner")
    e_dup = Entity(90, 110, "PERSON", 0.9, "ner")
    out = merge_window_entities([[e], [e_dup]], [w1, w2])
    assert len(out) == 1
