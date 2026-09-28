import pytest

from scripts.spec_g_common import audit_items, classify_literal_gate, overlap_coverage, summarize_predictions, validate_item


def claim():
    return {"claim_id": "c1", "entity_id": "e", "dimension": "d", "answer": "answer",
            "source_span_ids": ["s1"], "review_status": "approved"}


def item(kind="literal"):
    support = ["c1", "c2"] if kind == "inference" else ["c1"]
    return {"item_id": "i1", "class": kind, "question": "one two three four five six seven eight nine ten",
            "correct_answer": "answer", "supporting_claim_ids": support, "review_status": "approved",
            "single_sentence_insufficient": kind == "inference"}


def test_overlap_is_fraction_of_item_tokens_and_strict_flag():
    result = overlap_coverage("one two three four five six seven eight X Y", "one two three four five six seven eight")
    assert result["coverage_of_item_length"] == 0.8
    audited = audit_items([item()], [claim()], "one two three four five six seven eight")
    assert audited["items"][0]["flagged"] is True


def test_short_item_has_zero_eightgram_overlap():
    assert overlap_coverage("one two", "one two")["coverage_of_item_length"] == 0


def test_inference_requires_two_claims_and_attestation():
    bad = item("inference")
    bad["supporting_claim_ids"] = ["c1"]
    with pytest.raises(ValueError):
        validate_item(bad, {"c1", "c2"})


def test_gate_thresholds():
    assert classify_literal_gate(3, 10)["decision"] == "stop_near_chance"
    assert classify_literal_gate(8, 10)["decision"] == "high_legibility_record_ceilings"
    assert classify_literal_gate(6, 10)["decision"] == "intermediate_return_for_decision"


def test_exact_outcomes_and_completeness():
    assert summarize_predictions([{"item_id": "i", "outcome": "abstain"}], {"i": "literal"})["literal"]["abstain"] == 1
    with pytest.raises(ValueError):
        summarize_predictions([{"item_id": "i", "outcome": "neither"}], {"i": "literal"})
