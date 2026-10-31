import pandas as pd
import pytest

from promptshield.data.clean import clean_frame, normalize_record, template_hash
from promptshield.data.label_map import LabelMapper
from promptshield.data.split import grouped_split
from promptshield.data.synth import generate_pii_records, generate_secret_texts
from promptshield.data.validate import check_no_leakage, validate_records
from promptshield.pipelines.feedback import mask_with_correction, rehydrate
from promptshield.types import Entity


def test_label_mapper_merges_given_and_surname():
    m = LabelMapper()
    text = "Ask John Smith now"
    spans = [{"start": 4, "end": 8, "label": "GIVENNAME"}, {"start": 9, "end": 14, "label": "SURNAME"}]
    assert m.unify(text, spans) == [{"start": 4, "end": 14, "label": "PERSON"}]


def test_label_mapper_drops_unknown_and_reports():
    m = LabelMapper()
    out = m.unify("x y", [{"start": 0, "end": 1, "label": "WEIRDLABEL"}])
    assert out == [] and m.unmapped["WEIRDLABEL"] == 1


def test_normalize_keeps_spans_aligned():
    text = "ﬁle for Ｒahul"  # ligature + full-width R change under NFKC
    spans = [{"start": 8, "end": 13, "label": "PERSON"}]
    new_text, new_spans = normalize_record(text, spans)
    s = new_spans[0]
    assert new_text[s["start"] : s["end"]] == "Rahul"


def test_template_hash_groups_same_template():
    a = template_hash("Call Rahul now", [{"start": 5, "end": 10, "label": "PERSON"}])
    b = template_hash("Call Priyanka now", [{"start": 5, "end": 13, "label": "PERSON"}])
    assert a == b


def test_synth_records_are_valid_and_split_has_no_leakage():
    df = clean_frame(generate_pii_records(800, seed=1))
    df = grouped_split(df, seed=1)
    validate_records(df)
    check_no_leakage(df)
    assert set(df["split"]) == {"train", "val", "test"}


def test_validation_rejects_bad_spans():
    df = pd.DataFrame([{"text": "abc", "spans": [{"start": 2, "end": 9, "label": "PERSON"}],
                        "source": "synth", "template_id": "t"}])
    import pandera.errors

    with pytest.raises(pandera.errors.SchemaErrors):
        validate_records(df)


def test_secret_texts_contain_planted_value():
    df = generate_secret_texts(200, seed=2)
    for row in df.itertuples():
        assert row.text[row.start : row.end] == row.candidate
    assert 0.3 < df["label"].mean() < 0.6


def test_feedback_masking_and_rehydration_never_keeps_values():
    example = "Send the report to Amazon and to Rahul Mehta"
    ents = [Entity(19, 25, "PERSON", 0.6, "ner"), Entity(33, 44, "PERSON", 0.95, "ner")]
    masked = mask_with_correction(example, ents, "false_positive", "Amazon", None)
    assert masked == "Send the report to Amazon and to [PERSON_1]"
    rec = rehydrate(masked, seed=1)
    assert "Rahul" not in rec["text"] and "Amazon" in rec["text"]
    s = rec["spans"][0]
    assert s["label"] == "PERSON" and rec["text"][s["start"] : s["end"]].strip()
