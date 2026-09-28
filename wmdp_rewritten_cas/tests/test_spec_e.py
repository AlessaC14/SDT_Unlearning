import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from spec_e_common import (GateError, components_for_questions, exact_map_fisher_items, gate0,
                           option_only_example, planned_grid, read_config, split_components,
                           validate_prediction_rows)


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_config_and_grid():
    config = read_config(ROOT / "configs/spec_e.preflight.json")
    grid = planned_grid(config)
    assert len(grid) == 36
    assert len({x["run_id"] for x in grid}) == 36
    assert config["future_real_pilot"] == {"requires_separate_gpu_authorization": True,
        "checkpoint": "unfiltered-cb", "condition": "retain-only", "through_step": 2,
        "evaluation_scope": "all-512"}


def test_retain_only_accuracy_views_are_exactly_frozen_v():
    from spec_e_common import retain_only_accuracy_views
    rows=[{"question_id":f"q{i:03d}","gold":"A","p_A":1.0 if i%2==0 else 0.0,
           "p_B":0.0 if i%2==0 else 1.0,"p_C":0.0,"p_D":0.0} for i in range(512)]
    result=retain_only_accuracy_views(rows,{f"q{i:03d}" for i in range(102)})
    assert result["wmdp_rows_full_512"]==512 and result["wmdp_rows_V_102"]==102
    assert result["wmdp_accuracy_full_512"]==0.5 and result["wmdp_accuracy_V_102"]==0.5


def test_transitive_components_and_no_leakage():
    entities = {"q1": {"a"}, "q2": {"a", "b"}, "q3": {"b"}, "q4": {"c"}, "q5": set()}
    components = components_for_questions(entities, entities)
    assert ["q1", "q2", "q3"] in components
    train, validation = split_components(components, 3, 0, {"q5"})
    assert not set(train) & set(validation)
    for component in components:
        assert set(component) <= set(train) or set(component) <= set(validation)


def test_split_deterministic_and_balanced():
    components = [[f"q{i}"] for i in range(10)]
    assert split_components(components, 8, 7) == split_components(components, 8, 7)
    assert len(split_components(components, 8, 7)[0]) == 8


def test_entityless_is_singleton():
    components = components_for_questions(["a", "b", "c"], {"a": {"x"}, "b": {"x"}})
    assert ["c"] in components


def test_oversize_component_stops():
    ids = [f"q{i}" for i in range(512)]
    entities = {qid: ({"shared"} if i < 411 else {f"e{i}"}) for i, qid in enumerate(ids)}
    result = gate0(ids, entities, 410, 0, 410)
    assert result["status"] == "stopped"
    assert "train_question_ids" not in result


def test_incomplete_exact_mapping_stops():
    with pytest.raises(GateError, match="absent"):
        exact_map_fisher_items([{"question": "missing"}], {}, [0])


def test_prediction_contract_and_step_zero():
    row = {"run_id": "r", "checkpoint_step": 0, "question_id": "q", "p_A": .1,
           "p_B": .2, "p_C": .3, "p_D": .4, "gold": "D"}
    validate_prediction_rows([row], [0])
    with pytest.raises(GateError, match="missing required"):
        validate_prediction_rows([{**row, "checkpoint_step": 1}], [0, 1])
    with pytest.raises(GateError, match="duplicate"):
        validate_prediction_rows([row, row], [0])


def test_option_only_removes_stem_only():
    example = {"question": "secret stem", "choices": ["a", "b", "c", "d"], "answer": 2}
    got = option_only_example(example)
    assert got["question"] == "" and got["choices"] == example["choices"] and got["answer"] == 2


def test_binding_exact_split_and_no_entity_overlap():
    ids = [f"q{i:03d}" for i in range(512)]
    entities = {}
    for i, qid in enumerate(ids):
        if i < 196: entities[qid] = {"giant"}
        elif i < 410: entities[qid] = {f"e{i}"}
        else: entities[qid] = set()
    result = gate0(ids, entities, 410, 0, 410)
    assert result["status"] == "passed"
    assert result["train_size"] == 410 and result["validation_size"] == 102
    assert result["component_overlap_count"] == 0 and result["entity_intersection_count"] == 0
    assert result["train_entityless_count"] == 0 and result["validation_entityless_count"] == 102


def test_exact_410_failure_is_closed():
    ids = [f"q{i:03d}" for i in range(512)]
    entities = {}
    for i, qid in enumerate(ids):
        entities[qid] = ({"giant"} if i < 195 else ({f"pair-{(i-195)//2}"} if i < 411 else set()))
    result = gate0(ids, entities, 410, 0, 410)
    assert result["status"] == "stopped"
    assert "exact T=410" in result["stop_reason"]
