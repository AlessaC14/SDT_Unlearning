import json
import random

import pytest

from scripts.spec_h_analyze import run
from scripts.spec_h_common import (
    SpecHError,
    argmax_label,
    bootstrap_decomposition,
    decomposition,
    expected_symmetric_null_mi,
    solve_null_p,
    validate_alignment,
)


def test_identity_decomposition_ratio_one():
    gold = [0, 1, 2, 3] * 32
    value = decomposition(gold, gold, gold)
    assert value["i_f_y"] == pytest.approx(2.0)
    assert value["i_f_y_given_l"] == pytest.approx(0.0)
    assert value["ratio"] == pytest.approx(1.0)


def test_negative_mu_is_not_clipped():
    rng = random.Random(4)
    gold = [rng.randrange(4) for _ in range(200)]
    f = [rng.randrange(4) for _ in gold]
    reference = [rng.randrange(4) for _ in gold]
    value = decomposition(f, reference, gold)
    assert value["mu"] == value["i_f_y"] - value["i_f_y_given_l"]


def test_argmax_and_probability_contract():
    assert argmax_label([0.1, 0.2, 0.6, 0.1]) == 2
    with pytest.raises(SpecHError):
        argmax_label([0.1, 0.2, 0.3, 0.1])


def test_alignment_fails_on_reordering_and_gold_drift():
    a = [{"question_id": "a", "gold": 0}, {"question_id": "b", "gold": 1}]
    with pytest.raises(SpecHError):
        validate_alignment(a, list(reversed(a)))
    b = [{"question_id": "a", "gold": 0}, {"question_id": "b", "gold": 2}]
    with pytest.raises(SpecHError):
        validate_alignment(a, b)


def test_matched_null_solver():
    gold = [0, 1, 2, 3] * 128
    target = expected_symmetric_null_mi(gold, 0.7)
    solved, achieved = solve_null_p(gold, target)
    assert solved == pytest.approx(0.7, abs=1e-8)
    assert abs(achieved - target) <= 0.0005


def test_bootstrap_minimum_and_output():
    gold = [0, 1, 2, 3] * 16
    with pytest.raises(SpecHError):
        bootstrap_decomposition(gold, gold, gold, 999, 0, True)
    result = bootstrap_decomposition(gold, gold, gold, 1000, 0, True)
    assert len(result["i_f_y"]) == 2


def test_preflight_stops_on_missing_inputs(tmp_path):
    config = {
        "inputs": {"relearning_predictions": "missing/a", "pre_attack_predictions": "missing/b",
                   "shortcut_predictions": "missing/c"},
        "protocol_amendment": "2026-08-12-three-tier-native-four-way",
        "bootstrap_resamples": 1000,
        "null_draws": 20,
        "null_mi_tolerance_bits": 0.0005,
    }
    config_path = tmp_path / "repo" / "configs" / "spec_h.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps(config))
    code, result = run(config_path, tmp_path / "out")
    assert code == 2
    assert result["status"] == "stopped"
    assert len(result["missing_prerequisites"]) == 3
    assert len(result["contract_ambiguities"]) == 0


def test_three_tier_native_four_way_values():
    from scripts.spec_h_common import tier_values
    gold=[0,1,2,3]*32
    result=tier_values(gold,gold,gold,True)
    assert result["tier_1"]["i_f_y"] > 1.9
    # A perfectly correct reference makes L'=1 constant, so Tier 2 explains
    # none of the label MI through variation in the reference-correctness bit.
    assert result["tier_2"]["ratio"] == pytest.approx(0.0)
    assert result["tier_3"]["ratio"] == pytest.approx(1.0)
