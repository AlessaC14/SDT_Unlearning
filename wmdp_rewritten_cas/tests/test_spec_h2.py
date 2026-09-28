#!/usr/bin/env python3
"""Scoped tests for Spec H2.

The contracts under test are the ones Amendment 1 made normative: estimator
identity with Spec H, the argmax tie rule, question-ID joins (never positional
zips), the Ruling 3 step-zero consistency check, the Ruling 1 structural
impossibility of a vacuous "satisfied", and the no-write / no-model invariants.
"""
from __future__ import annotations

import ast
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import spec_h2_io as io  # noqa: E402
import spec_h2_kernel as kernel  # noqa: E402
import spec_h2_phase_stats as phase_stats  # noqa: E402
import spec_h_common as legacy  # noqa: E402

H2_SOURCES = [
    ROOT / "scripts" / name for name in (
        "spec_h2_kernel.py", "spec_h2_io.py", "spec_h2_preflight.py",
        "spec_h2_trajectories.py", "spec_h2_phase_stats.py",
    )
] + [Path(__file__)]


def random_labels(rng, n, alphabet=4):
    return rng.integers(0, alphabet, n).astype(np.int64)


# --------------------------------------------------------------------------- #
# Kernel identities
# --------------------------------------------------------------------------- #

def test_self_information_is_entropy_and_ratio_is_one():
    """I(Y;Y) = H(Y); with L = F, mu = I and ratio = 1."""
    rng = np.random.default_rng(0)
    y = random_labels(rng, 400)
    stats = kernel.stats_from_joint(kernel.joint3(y, y, y), y.size)
    plugin_entropy, _ = legacy.entropy(y.tolist())
    assert stats["i_fy_plugin"][0] == pytest.approx(plugin_entropy, abs=1e-12)
    assert stats["cmi_plugin"][0] == pytest.approx(0.0, abs=1e-12)
    assert stats["mu_plugin"][0] == pytest.approx(plugin_entropy, abs=1e-12)
    assert stats["ratio_plugin"][0] == pytest.approx(1.0, abs=1e-12)
    assert stats["ratio_min_plugin"][0] == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("seed", range(6))
def test_miller_madow_matches_spec_h_estimator(seed):
    """Estimator identity with Spec H, to 1e-12, on both I and the CMI."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(60, 400))
    f, y, l = (random_labels(rng, n) for _ in range(3))
    stats = kernel.stats_from_joint(kernel.joint3(f, y, l), n)

    for miller_madow, suffix in ((False, "plugin"), (True, "mm")):
        expected_i, _ = legacy.mutual_information(
            f.tolist(), y.tolist(), miller_madow=miller_madow)
        expected_cmi, _ = legacy.conditional_mutual_information(
            f.tolist(), y.tolist(), l.tolist(), miller_madow=miller_madow)
        assert stats[f"i_fy_{suffix}"][0] == pytest.approx(expected_i, abs=1e-12)
        assert stats[f"cmi_{suffix}"][0] == pytest.approx(expected_cmi, abs=1e-12)
        assert stats[f"mu_{suffix}"][0] == pytest.approx(
            expected_i - expected_cmi, abs=1e-12)


@pytest.mark.parametrize("seed", range(4))
def test_batched_kernel_matches_scalar_kernel(seed):
    rng = np.random.default_rng(seed)
    n = 200
    f, y, l = (random_labels(rng, n) for _ in range(3))
    index = kernel.bootstrap_index(n, 25, np.random.default_rng(seed + 100))
    batched = kernel.stats_from_joint(kernel.joint3_batched(f, y, l, index), n)
    for row, selection in enumerate(index):
        scalar = kernel.stats_from_joint(
            kernel.joint3(f[selection], y[selection], l[selection]), n)
        for key in ("i_fy_plugin", "cmi_plugin", "mu_plugin", "i_fy_mm", "mu_mm"):
            assert batched[key][row] == pytest.approx(scalar[key][0], abs=1e-12)


def test_joint3_batched_l_matches_scalar():
    rng = np.random.default_rng(7)
    n = 150
    f, y = random_labels(rng, n), random_labels(rng, n)
    batch = np.stack([random_labels(rng, n) for _ in range(8)])
    counts = kernel.joint3_batched_l(f, y, batch)
    for row in range(batch.shape[0]):
        assert np.array_equal(counts[row], kernel.joint3(f, y, batch[row]))


def test_mi2_matches_full_joint_path():
    rng = np.random.default_rng(11)
    n = 300
    f, y = random_labels(rng, n), random_labels(rng, n)
    permutations = kernel.permutation_batch(n, 12, np.random.default_rng(3))
    batched = kernel.mi2_batched(f, y[permutations])
    for row, permutation in enumerate(permutations):
        reference = kernel.stats_from_joint(
            kernel.joint3(f, y[permutation], f), n)
        assert batched["i_fy_plugin"][row] == pytest.approx(
            reference["i_fy_plugin"][0], abs=1e-12)


def test_argmax_tie_rule_is_lowest_letter_index():
    probabilities = np.array([[0.3, 0.3, 0.2, 0.2],
                              [0.1, 0.4, 0.4, 0.1],
                              [0.25, 0.25, 0.25, 0.25]])
    assert kernel.argmax_label(probabilities).tolist() == [0, 1, 0]
    assert kernel.count_argmax_ties(probabilities) == 3
    # Identical to the Spec H rule, which used key=(p, -i).
    for row in probabilities:
        assert int(kernel.argmax_label(row[None, :])[0]) == legacy.argmax_label(row.tolist())


def test_degenerate_prediction_gives_zero_information_and_undefined_ratio():
    n = 120
    f = np.zeros(n, dtype=np.int64)          # a single-letter collapse, as at lr 2e-04
    y = np.arange(n, dtype=np.int64) % 4
    stats = kernel.stats_from_joint(kernel.joint3(f, y, f), n)
    assert stats["i_fy_plugin"][0] == pytest.approx(0.0, abs=1e-12)
    assert math.isnan(stats["ratio_plugin"][0])
    assert math.isnan(stats["ratio_min_plugin"][0])


def test_negative_mu_is_never_clipped():
    rng = np.random.default_rng(5)
    n = 200
    f, y, l = (random_labels(rng, n) for _ in range(3))
    stats = kernel.stats_from_joint(kernel.joint3(f, y, l), n)
    # Independent draws: the plugin CMI exceeds I, so mu is negative and stays so.
    assert stats["mu_plugin"][0] < 0
    low, high, count = kernel.percentile_ci(np.array([-3.0, -2.0, -1.0, 0.5]))
    assert low < 0 and count == 4


def test_symmetric_null_draws_match_the_requested_channel():
    rng = np.random.default_rng(42)
    y = np.arange(4000, dtype=np.int64) % 4
    p_correct = 0.6
    draws = kernel.symmetric_null_draws(y, p_correct, 40, rng)
    assert draws.shape == (40, y.size)
    accuracy = float((draws == y[None, :]).mean())
    assert accuracy == pytest.approx(p_correct, abs=0.02)
    wrong = draws[draws != np.broadcast_to(y[None, :], draws.shape)]
    counts = np.bincount(wrong % 4, minlength=4)
    assert counts.min() > 0.9 * counts.max()   # uniform over the other three


def test_null_model_matches_reference_label_information():
    """The Nakkiran footnote-5 null model is accuracy-matched, not merely random."""
    rng = np.random.default_rng(42)
    y = np.arange(2048, dtype=np.int64) % 4
    l = np.where(rng.random(y.size) < 0.45, y, (y + 1) % 4)
    target = kernel.stats_from_joint(kernel.joint3(l, y, l), y.size)["i_fy_plugin"][0]
    p_correct, achieved = legacy.solve_null_p(y.tolist(), float(target), 5e-4)
    assert achieved == pytest.approx(target, abs=5e-4)
    draws = kernel.symmetric_null_draws(y, p_correct, 100, rng)
    observed = kernel.stats_from_joint(
        kernel.joint3_batched_l(draws[0] * 0 + l, y, draws), y.size)["i_ly_plugin"]
    assert float(observed.mean()) == pytest.approx(target, abs=0.02)


# --------------------------------------------------------------------------- #
# I/O contracts
# --------------------------------------------------------------------------- #

def test_forget_T_requires_an_id_join_not_a_positional_zip():
    """Finding A, made normative by Ruling 6.

    forget-T tables are in sorted V-102 order; the reference is in the shuffled
    canonical 512 order. A positional read would silently mislabel the reference.
    """
    run = next(r for r in io.enumerate_runs(ROOT) if r.condition == "forget-T")
    table = io.load_table(run.prediction_path(0))
    reference = io.load_table(io.reference_paths(run.model, ROOT)["intact"])
    joined = io.join_reference(table, reference)
    positional = reference.argmax[:table.n].astype(np.int64)
    assert list(table.ids) != list(reference.ids)[:table.n]
    assert not np.array_equal(joined, positional)
    # The join is order-independent: it must agree item by item.
    lookup = dict(zip(reference.ids, reference.argmax))
    assert joined.tolist() == [int(lookup[qid]) for qid in table.ids]


def test_join_rejects_gold_disagreement():
    run = next(r for r in io.enumerate_runs(ROOT) if r.condition == "retain-only")
    table = io.load_table(run.prediction_path(0))
    reference = io.load_table(io.reference_paths(run.model, ROOT)["intact"])
    corrupted = io.Table(
        path=reference.path, sha256=reference.sha256, ids=reference.ids,
        gold=(reference.gold + 1) % 4, argmax=reference.argmax,
        probabilities=reference.probabilities, argmax_ties=reference.argmax_ties)
    with pytest.raises(io.SpecH2Error):
        io.join_reference(table, corrupted)


def test_spec_h_outputs_is_never_a_write_target():
    io.assert_write_target(ROOT / "spec_h2_outputs/trajectories.json", ROOT)
    for forbidden in ("spec_h_outputs",
                      "spec_h_outputs/final/info_decomposition.json",
                      "spec_h_outputs/reference_parity/parity.json"):
        with pytest.raises(io.SpecH2Error):
            io.assert_write_target(ROOT / forbidden, ROOT)


def imported_modules(tree) -> set[str]:
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def referenced_names(tree) -> set[str]:
    """Every identifier the code actually references, ignoring prose."""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.ImportFrom):
            names |= {alias.name for alias in node.names}
    return names


def test_h2_tree_loads_no_model_libraries():
    for source in H2_SOURCES:
        loaded = imported_modules(ast.parse(source.read_text()))
        forbidden = loaded & {"torch", "transformers", "peft", "accelerate"}
        assert not forbidden, f"{source} imports {sorted(forbidden)}"


def test_h2_tree_never_calls_the_legacy_alignment_helper():
    """Ruling 6: spec_h_common.validate_alignment is not called anywhere in H2.

    Checked against the parsed identifiers, so the prose that explains why it is
    excluded does not itself trip the test.
    """
    for source in H2_SOURCES:
        assert "validate_alignment" not in referenced_names(ast.parse(source.read_text()))


def test_h2_inner_loops_do_not_use_the_counter_path():
    """Ruling 6: spec_h_common's Counter-over-tuples entropy is test-oracle only.

    The single permitted import is solve_null_p, which walks a 4x4 channel once per
    cell to match the null model's accuracy and never enters a resampling loop.
    """
    permitted = {"solve_null_p"}
    for source in H2_SOURCES:
        if source.name == Path(__file__).name:
            continue
        tree = ast.parse(source.read_text())
        borrowed = {alias.name for node in ast.walk(tree)
                    if isinstance(node, ast.ImportFrom)
                    and (node.module or "").endswith("spec_h_common")
                    for alias in node.names}
        assert borrowed <= permitted, f"{source} borrows {sorted(borrowed - permitted)}"


# --------------------------------------------------------------------------- #
# Ruling 1: the vacuous "satisfied" must be structurally unreachable
# --------------------------------------------------------------------------- #

def test_both_infinite_t80_is_not_evaluable():
    for strict in (True, False):
        verdict, reason = phase_stats.compare_t80(math.inf, math.inf, strict=strict)
        assert verdict == "not_evaluable"
        assert "never reached" in reason


@pytest.mark.parametrize("strict", [True, False])
def test_compare_t80_orderings(strict):
    assert phase_stats.compare_t80(2.0, 8.0, strict)[0] == "satisfied"
    assert phase_stats.compare_t80(8.0, 2.0, strict)[0] == "not_satisfied"
    assert phase_stats.compare_t80(4.0, math.inf, strict)[0] == "satisfied"
    assert phase_stats.compare_t80(math.inf, 4.0, strict)[0] == "not_satisfied"
    assert phase_stats.compare_t80(4.0, 4.0, strict)[0] == (
        "not_satisfied" if strict else "satisfied")


def test_satisfied_is_unreachable_when_neither_side_is_finite():
    """Exhaustive over the degenerate quadrant, both comparison directions."""
    for strict in (True, False):
        for lhs in (math.inf, math.nan):
            for rhs in (math.inf, math.nan):
                assert phase_stats.compare_t80(lhs, rhs, strict)[0] != "satisfied"


def test_median_with_infinity_keeps_infinity():
    assert math.isinf(phase_stats.median_with_infinity([math.inf] * 6))
    assert math.isinf(phase_stats.median_with_infinity([2.0, 4.0, math.inf, math.inf]))
    assert phase_stats.median_with_infinity([1.0, 2.0, 3.0]) == 2.0


def test_step_labels_collapse_the_indistinguishable_pair():
    assert phase_stats.step_label(0) == "<=1"
    assert phase_stats.step_label(1) == "<=1"
    assert phase_stats.step_label(64) == "64"
    assert phase_stats.step_label(math.inf) == "inf"


# --------------------------------------------------------------------------- #
# Ruling 3: step-zero consistency on the real tables
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("estimator", ["plugin", "mm"])
def test_unfiltered_step_zero_ratio_and_ratio_min_are_one(estimator):
    """F0 is identically L_intact for the unfiltered model, by the parity gate.

    The residual is float summation order over transposed count arrays, not
    estimator error, so the assertion is tight rather than bitwise.
    """
    runs = [r for r in io.enumerate_runs(ROOT)
            if r.model == "unfiltered" and r.condition == "retain-only"]
    reference = io.load_table(io.reference_paths("unfiltered", ROOT)["intact"])
    assert runs
    for run in runs:
        table = io.load_table(run.prediction_path(0))
        labels = io.join_reference(table, reference)
        assert np.array_equal(labels, table.argmax.astype(np.int64))
        stats = kernel.stats_from_joint(
            kernel.joint3(table.argmax, table.gold, labels), table.n)
        assert stats[f"ratio_{estimator}"][0] == pytest.approx(1.0, abs=1e-12)
        assert stats[f"ratio_min_{estimator}"][0] == pytest.approx(1.0, abs=1e-12)
        assert stats[f"ceiling_{estimator}"][0] == pytest.approx(1.0, abs=1e-12)
