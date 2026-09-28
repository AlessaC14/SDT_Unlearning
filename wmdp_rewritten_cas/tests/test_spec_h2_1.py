#!/usr/bin/env python3
"""Scoped tests for Spec H2.1 (matched-step reference control)."""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import spec_h2_1_decision as decision  # noqa: E402
import spec_h2_kernel as kernel  # noqa: E402
import spec_h_common as legacy  # noqa: E402

OUT = ROOT / "spec_h2_1_outputs"
SOURCES = [ROOT / "scripts" / n for n in ("spec_h2_1_matched_reference.py",
                                          "spec_h2_1_decision.py")]


@pytest.fixture(scope="module")
def matched():
    path = OUT / "matched_reference.json"
    if not path.is_file():
        pytest.skip("matched_reference.json absent")
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def decided():
    path = OUT / "decision.json"
    if not path.is_file():
        pytest.skip("decision.json absent")
    return json.loads(path.read_text())


@pytest.mark.parametrize("seed", range(4))
def test_estimator_identity_with_spec_h(seed):
    """Spec H2.1 section 2: same MM cross-check that H2 uses, same kernel."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(80, 400))
    f, y, l = (rng.integers(0, 4, n).astype(np.int64) for _ in range(3))
    stats = kernel.stats_from_joint(kernel.joint3(f, y, l), n)
    for miller_madow, suffix in ((False, "plugin"), (True, "mm")):
        i, _ = legacy.mutual_information(f.tolist(), y.tolist(), miller_madow=miller_madow)
        cmi, _ = legacy.conditional_mutual_information(
            f.tolist(), y.tolist(), l.tolist(), miller_madow=miller_madow)
        assert stats[f"i_fy_{suffix}"][0] == pytest.approx(i, abs=1e-12)
        assert stats[f"cmi_{suffix}"][0] == pytest.approx(cmi, abs=1e-12)
        assert stats[f"mu_{suffix}"][0] == pytest.approx(i - cmi, abs=1e-12)


def test_self_reference_is_a_tautology_on_synthetic_data():
    """F identically L must give mu = I_t and ratio 1 whenever I_t is nonzero."""
    rng = np.random.default_rng(0)
    f = rng.integers(0, 4, 300).astype(np.int64)
    y = np.where(rng.random(300) < 0.6, f, (f + 1) % 4)
    stats = kernel.stats_from_joint(kernel.joint3(f, y, f), f.size)
    assert stats["mu_plugin"][0] == pytest.approx(stats["i_fy_plugin"][0], abs=1e-12)
    assert stats["ratio_plugin"][0] == pytest.approx(1.0, abs=1e-12)


def test_degenerate_self_reference_leaves_ratio_undefined_not_one():
    """The carve-out the spec text omits: I_t = 0 makes ratio 0/0, not 1."""
    f = np.zeros(120, dtype=np.int64)
    y = np.arange(120, dtype=np.int64) % 4
    stats = kernel.stats_from_joint(kernel.joint3(f, y, f), f.size)
    assert stats["i_fy_plugin"][0] == pytest.approx(0.0, abs=1e-12)
    assert stats["mu_plugin"][0] == pytest.approx(stats["i_fy_plugin"][0], abs=1e-12)
    assert np.isnan(stats["ratio_plugin"][0])


# --------------------------------------------------------------------------- #
# Decision rule: every branch, mechanically
# --------------------------------------------------------------------------- #

def _runs(medians, cleared, clears):
    return {f"r{i}": {"run_id": f"r{i}", "lr": "1e-05", "seed": 0,
                      "median_delta_excess": m, "steps_cleared_both": cleared,
                      "mu_clears_null": clears, "steps_i_cleared": cleared,
                      "steps_reference_cleared": cleared, "steps_total": 11,
                      "n_delta_excess": cleared, "cleared_steps": []}
            for i, m in enumerate(medians)}


def test_outcome_a_requires_both_conjuncts():
    assert decision.decide(_runs([0.05] * 6, 10, 9))["outcome"] == "A_staleness"
    # positive and tight, but the null-clearance majority fails -> not A
    assert decision.decide(_runs([0.05] * 6, 10, 1))["outcome"] != "A_staleness"
    # clears the null everywhere, but the interval spans zero -> not A
    assert decision.decide(_runs([0.05, -0.05, 0.05, -0.05, 0.05, -0.05], 10, 9)
                           )["outcome"] != "A_staleness"


def test_outcome_b_is_the_within_band_majority():
    result = decision.decide(_runs([0.05] * 6, 10, 1))
    assert result["outcome"] == "B_divergence"
    assert result["majority_within_null_band"] is True


def test_outcome_c_when_neither_majority_holds():
    # exactly half clearing: neither majority is satisfied
    assert decision.decide(_runs([0.05] * 6, 10, 5))["outcome"] == "C_partial"


def test_not_evaluable_when_nothing_clears_both():
    assert decision.decide(_runs([None] * 6, 0, 0))["outcome"] == "not_evaluable"


def test_negative_median_cannot_yield_staleness():
    for clears in (0, 5, 9, 10):
        assert decision.decide(_runs([-0.05] * 6, 10, clears))["outcome"] != "A_staleness"


# --------------------------------------------------------------------------- #
# Contracts on the persisted artifacts
# --------------------------------------------------------------------------- #

def test_self_reference_control_holds_on_real_tables(matched):
    control = matched["self_reference_control"]
    assert control["pass"] is True
    assert control["argmax_identical_all_cells"] is True
    assert control["mu_equals_i_t_all_cells"] is True
    # Not bitwise zero: entropy terms are summed over transposed count arrays, so the
    # residual is float summation order, the same effect H2's Ruling 3 check records.
    assert control["max_absolute_mu_minus_i_t"] < 1e-12
    assert control["ratio_is_one_where_applicable"] is True


def test_delta_excess_is_the_difference_of_the_two_excesses(matched):
    checked = 0
    for record in matched["records"]:
        if record.get("delta_excess") is None:
            continue
        assert record["delta_excess"] == pytest.approx(
            record["excess_matched"] - record["excess_preattack_h2"], abs=1e-12)
        checked += 1
    assert checked > 0


def test_h2_excess_is_read_not_recomputed(matched):
    provenance = matched["excess_preattack_provenance"]
    assert provenance["recomputed"] is False
    crosscheck = provenance["phase_stats_crosscheck"]
    assert crosscheck["checked"] > 0
    assert crosscheck["agrees"] is True


def test_a_step_enters_summaries_only_when_both_clearances_hold(matched):
    for record in matched["records"]:
        expected = bool(record["reference_cleared"] and record["i_t_cleared_h2"])
        assert record["cleared_in_both_analyses"] is expected


def test_excess_is_null_valued_not_zero_where_the_reference_is_uncleared(matched):
    uncleared = [r for r in matched["records"] if not r["reference_cleared"]]
    assert uncleared, "the degenerate lr-2e-04 cells should leave some step uncleared"
    for record in uncleared:
        assert record["cleared_in_both_analyses"] is False
        assert "mu_matched_plugin" in record          # mu is stored regardless


def test_null_model_p_is_resolved_per_step(matched):
    by_run: dict[str, set] = {}
    for record in matched["records"]:
        by_run.setdefault(record["run_id"], set()).add(
            round(record["null_model"]["p_correct"], 9))
        assert record["null_model"]["matched"] is True
        assert record["null_model"]["p_resolved_per_step"] is True
    assert any(len(v) > 1 for v in by_run.values()), "p never moved along any trajectory"


def test_frozen_trees_are_never_written(matched):
    forbidden = ("spec_e_outputs", "spec_h2_outputs")
    writes = ("write_text", "write_bytes", "mkdir", "unlink", "os.replace", "write_json")
    for source in SOURCES:
        for line in source.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith("#") or not any(t in stripped for t in forbidden):
                continue
            assert not any(w in stripped for w in writes), f"{source}: {stripped}"
    assert matched["h2_outputs_modified"] is False


def test_no_model_libraries_in_the_h2_1_tree():
    for source in SOURCES:
        tree = ast.parse(source.read_text())
        loaded = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                loaded |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                loaded.add(node.module.split(".")[0])
        assert not loaded & {"torch", "transformers", "peft", "accelerate"}


def test_h2_p1_p2_are_not_re_treated(decided):
    assert "not_evaluable" in decided["h2_p1_p2_untouched"]
    stats = json.loads((ROOT / "spec_h2_outputs/phase_stats.json").read_text())
    hypotheses = stats["preregistered"]["hypotheses"]
    assert hypotheses["P1"]["verdict"] == "not_evaluable"
    assert hypotheses["P2"]["verdict"] == "not_evaluable"
