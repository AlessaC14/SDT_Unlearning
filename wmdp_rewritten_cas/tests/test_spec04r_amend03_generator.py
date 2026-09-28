import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("a03gen", ROOT / "scripts/spec04r_amend03_generator.py")
M = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)


def entity(index=0, dimensions=("reservoir_host",)):
    return {"organism_id": f"ORG-{index:04d}", "name_normalized": "fixture",
            "questions": [{"question_id": f"Q-{dim}", "primary_label": dim,
                           "question": "Observed category rodents.", "correct_text": "rodents"}
                          for dim in dimensions]}


def request(dimensions=("reservoir_host",), count=10):
    return {"value_spaces": {dim: ["rodents", "birds"] for dim in dimensions},
            "chapters": [{"chapter_id": f"CH-{i}", "title": "fixture",
                          "organisms": [entity(i, dimensions)]} for i in range(count)]}


def ceiling(full=1.793103448275862, pilot=1.4):
    return {"selected_core": {"implied_maximum_mean_attributes_per_entity": full},
            "pilot": {"implied_maximum_mean_attributes_per_entity": pilot}}


class FakeClient:
    model = "pinned-model"
    def __init__(self, root, inexact=False):
        self.raw_log = root / "raw.jsonl"; self.inexact = inexact
    def call(self, prompt, call_id, attempt_index):
        payload = json.loads(prompt); dim = payload["required_dimension"]
        value = {"dimension": dim, "real_value": "rodnts" if self.inexact else "rodents",
                 "counterfactual_value": "birds", "grounding_question_ids": [f"Q-{dim}"],
                 "grounding_label": dim, "plausibility_note": "descriptive",
                 "insufficient_groundable_dimensions": payload["available_distinct_groundable_dimensions"] < 4,
                 "available_dimension_count": payload["available_distinct_groundable_dimensions"]}
        self.raw_log.parent.mkdir(parents=True, exist_ok=True)
        with self.raw_log.open("a") as f: f.write(json.dumps({"raw_response": json.dumps(value)}) + "\n")
        return value, None, False


def prior(path):
    value = {"organism_count": 10, "proposal_count": 20, "accepted_count": 18,
             "parse_rate": 1.0, "acceptance_rate_among_parsed": .9,
             "mean_attributes_per_organism": 1.8}
    path.write_text(json.dumps(value)); return path


def test_distinct_plan_never_duplicates_and_caps_at_four():
    dims = tuple(f"dim{i}" for i in range(6))
    got = M.plan(entity(dimensions=dims))
    assert len(got) == 4 and len(set(got)) == 4
    assert len(M.plan(entity())) == 1


def test_population_specific_thresholds_are_explicit():
    got = M.thresholds(ceiling())
    assert got["full_mean_attributes_per_entity"] == .8 * 1.793103448275862
    assert abs(got["paired_pilot_mean_attributes_per_entity"] - 1.12) < 1e-12


def test_exact_span_repair_is_deterministic():
    value = {"real_value": "rodnts", "grounding_question_ids": ["Q-reservoir_host"]}
    got, applied = M.exact_span_repair(value, entity(), "reservoir_host")
    assert applied and got["real_value"] == "rodents"


def test_mocked_paired_pilot_logs_repairs_and_declarations(tmp_path):
    out = tmp_path / "out"
    result = M.run("pilot", request(), FakeClient(tmp_path, inexact=True), out,
                   ceiling(), prior(tmp_path / "prior.json"))
    assert result["passed"] is False  # structural pilot maximum is one in this fixture, below measured 1.12 gate
    assert result["deterministic_repairs_applied"] == 10
    assert result["insufficient_dimension_declarations"] == 10
    assert len((out / "deterministic_repair_log.jsonl").read_text().splitlines()) == 10
    assert (out / "paired_comparison.md").exists()
