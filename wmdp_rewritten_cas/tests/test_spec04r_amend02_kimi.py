import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "spec04r_amend02_kimi", ROOT / "scripts" / "spec04r_amend02_kimi.py"
)
M = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)


def response(content, model="pinned-model-2026-01", refusal=None):
    return {
        "model": model,
        "choices": [{"message": {"content": content, "refusal": refusal}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
    }


def client(tmp_path, transport, parser=M.extract_object):
    return M.AuditedClient(
        model="pinned-model-2026-01", transport=transport,
        cache_dir=tmp_path / "cache", raw_log=tmp_path / "raw.jsonl",
        usage_log=tmp_path / "usage.jsonl", pricing=M.Pricing(1.25, 2.5), parser=parser,
    )


def valid_attribute():
    return {
        "dimension": "reservoir_host", "real_value": "rodents",
        "counterfactual_value": "birds", "grounding_question_ids": ["Q1"],
        "grounding_label": "reservoir_host", "plausibility_note": "same descriptive type",
    }


def request(organisms=10):
    chapters = []
    for index in range(organisms):
        chapters.append({
            "chapter_id": f"CH-{index:02d}", "title": f"Chapter {index}",
            "organisms": [{
                "organism_id": f"ORG-{index:04d}", "name_normalized": f"organism {index}",
                "questions": [{"question_id": "Q1", "primary_label": "reservoir_host",
                               "question": "The reservoir is rodents.", "correct_text": "rodents"}],
            }],
        })
    return {"value_spaces": {"reservoir_host": ["rodents", "birds"]}, "chapters": chapters}


def test_raw_logging_precedes_parse_and_has_exact_core_schema(tmp_path):
    raw_path = tmp_path / "raw.jsonl"

    def parser_after_log(text):
        records = [json.loads(line) for line in raw_path.read_text().splitlines()]
        assert len(records) == 1
        assert set(records[0]) == set(M.RAW_FIELDS)
        return json.loads(text)

    c = client(tmp_path, lambda body: response(json.dumps(valid_attribute())), parser_after_log)
    parsed, error, refused = c.call("prompt", "call-0", 0)
    assert parsed == valid_attribute() and error is None and not refused
    usage = json.loads((tmp_path / "usage.jsonl").read_text())
    assert usage["total_cost_usd"] == "0.0001750000"


def test_cache_is_exact_and_never_reissues_hit(tmp_path):
    calls = []

    def transport(body):
        calls.append(body)
        return response(json.dumps(valid_attribute()))

    c = client(tmp_path, transport)
    c.call("same prompt", "first", 0)
    c.call("same prompt", "replay", 0)
    assert len(calls) == 1
    usage = [json.loads(line) for line in (tmp_path / "usage.jsonl").read_text().splitlines()]
    assert [row["cache_hit"] for row in usage] == ["false", "true"]
    assert len(list((tmp_path / "cache").glob("*.json"))) == 1


def test_model_drift_is_logged_then_asserted(tmp_path):
    c = client(tmp_path, lambda body: response("{}", model="different-model"))
    with pytest.raises(AssertionError, match="model drift"):
        c.call("prompt", "drift", 0)
    raw = json.loads((tmp_path / "raw.jsonl").read_text())
    assert raw["model_version"] == "different-model"


def test_parse_failure_gets_one_repair_and_pilot_passes(tmp_path):
    calls = []

    def transport(body):
        calls.append(body)
        if len(calls) == 1:
            return response("not json")
        return response(json.dumps(valid_attribute()))

    c = client(tmp_path, transport)
    result = M.run("pilot", request(), c, tmp_path / "out")
    # Twenty proposals: first has generation+repair, the rest parse on generation.
    assert len(calls) == 21
    assert result["parse_rate"] == 1.0
    assert result["acceptance_rate_among_parsed"] == 1.0
    assert result["mean_attributes_per_organism"] == 2.0
    assert result["passed"] is True
    outcomes = (tmp_path / "out" / "outcome_ledger.csv").read_text()
    assert "ORG-0000-p0-repair" in outcomes


def test_pilot_failure_and_full_hard_stop(tmp_path):
    failing = client(tmp_path / "fail", lambda body: response("plain prose"))
    result = M.run("pilot", request(), failing, tmp_path / "failed-out")
    assert result["passed"] is False
    with pytest.raises(RuntimeError, match="passing pilot artifact required"):
        M.run("full", request(), failing, tmp_path / "full-out")
    with pytest.raises(RuntimeError, match="pilot failed"):
        M.run("full", request(), failing, tmp_path / "full-out",
              tmp_path / "failed-out" / "pilot_result.json")


def test_refusal_is_logged_as_outcome(tmp_path):
    c = client(tmp_path, lambda body: response(None, refusal="declined"))
    parsed, error, refused = c.call("prompt", "refusal", 0)
    assert parsed is None and error == "refusal" and refused
    usage = json.loads((tmp_path / "usage.jsonl").read_text())
    assert usage["refusal"] == "true"
