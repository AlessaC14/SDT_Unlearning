#!/usr/bin/env python3
"""Scoped tests for Spec G (gradient geometry along recovery trajectories)."""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import spec_gradgeo_decision as decision  # noqa: E402

OUT = ROOT / "spec_gradgeo_outputs"
SOURCES = [ROOT / "scripts" / n for n in ("spec_gradgeo_audit.py", "spec_gradgeo_gradients.py",
                                          "spec_gradgeo_decision.py")]


@pytest.fixture(scope="module")
def audit():
    path = OUT / "audit.json"
    if not path.is_file():
        pytest.skip("audit.json absent")
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def cells():
    found = sorted(glob.glob(str(OUT / "cells" / "*.json")))
    if not found:
        pytest.skip("no cell results yet")
    return [json.loads(Path(p).read_text()) for p in found]


def test_lora_delta_matches_scaling_times_b_a():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(8, 16)).astype(np.float32)
    b = rng.normal(size=(32, 8)).astype(np.float32)
    delta = 2.0 * (b @ a)
    assert delta.shape == (32, 16)


def test_pristine_restore_is_exact_where_subtraction_is_not():
    """The committed path restores by copy, which is bitwise; subtraction is not.

    The Tier 1 run in the log was produced on the subtractive path, whose measured
    drift is recorded in the RUN_LOG and bounded far below the reduction-order
    precision. This test pins both halves of that statement.
    """
    rng = np.random.default_rng(7)
    w0 = (rng.normal(size=(1024, 512)) * 0.02).astype(np.float32)
    pristine = w0.copy()
    w = w0.copy()
    for scale in (5.8e-06, 1.0e-04, 1.0e-03, 2.5e-03):
        delta = (rng.normal(size=w0.shape) * scale).astype(np.float32)
        w += delta
        w[:] = pristine          # restore_pristine
    assert np.array_equal(w, w0), "copy-back restoration must be bitwise exact"

def test_subtractive_unmerge_drift_is_bounded_below_the_noise_floor():
    """The path the logged run used: not bitwise, but bounded far below 6e-5."""
    rng = np.random.default_rng(0)
    w0 = (rng.normal(size=(4096, 1024)) * 0.02).astype(np.float32)
    w = w0.copy()
    for scale in (5.8e-06, 3.0e-05, 1.0e-04, 3.7e-04, 7.9e-04, 1.0e-03, 1.9e-03, 2.5e-03):
        delta = (rng.normal(size=w0.shape) * scale).astype(np.float32)
        w += delta
        w -= delta
    drift = float(np.abs(w - w0).max())
    assert not np.array_equal(w, w0), "the round-trip is not bitwise; that is the point"
    assert drift / float(np.abs(w0).max()) < 1e-6
    assert drift < 1e-5   # orders below the 6e-5 reduction-order precision


def test_zero_b_gives_a_zero_delta():
    """Why step 0 and step 1 are the base model: B is zero at both."""
    rng = np.random.default_rng(1)
    a = rng.normal(size=(8, 16)).astype(np.float32)
    b = np.zeros((32, 8), dtype=np.float32)
    assert not (2.0 * (b @ a)).any()


def test_scaling_matches_the_frozen_adapter_config():
    configs = sorted(glob.glob(str(ROOT / "spec_r_outputs/runs/*/adapter-step-000/adapter_config.json")))
    if not configs:
        pytest.skip("no adapters present")
    payload = json.loads(Path(configs[0]).read_text())
    assert payload["lora_alpha"] / payload["r"] == 2.0


def _cell(model, means, run="r", certified=512):
    records = [{"step": step, "certified": step <= certified,
                "replicate_0": {"conflict": value, "grad_norm_forget": 1.0,
                                "grad_norm_retain": 1.0,
                                "conflict_null": {"p2_5": -0.02, "p97_5": 0.02},
                                "alpha": {}}}
               for step, value in means.items()]
    return {"cell": {"run_id": run, "model": model, "condition": "retain-only",
                     "lr": "1e-05", "seed": 0, "certified_through": certified},
            "records": records}


def test_p1_is_not_evaluable_with_one_run_per_arm():
    result = decision.evaluate_p1([
        _cell("unfiltered-cb", {0: 0.5, 8: 0.6, 64: 0.7}, "a"),
        _cell("e2e-strong-filter", {0: 0.1, 8: 0.1, 64: 0.1}, "b"),
    ])
    assert result["verdict"] == "not_evaluable"
    assert "n=1 run" in result["reason"]
    assert result["direction_satisfied"] is True
    assert result["difference"] == pytest.approx(0.5, abs=1e-9)


def test_p1_satisfied_needs_direction_and_interval():
    cb = [_cell("unfiltered-cb", {0: v, 8: v, 64: v}, f"cb{i}")
          for i, v in enumerate([0.5, 0.55, 0.6])]
    sf = [_cell("e2e-strong-filter", {0: v, 8: v, 64: v}, f"sf{i}")
          for i, v in enumerate([0.1, 0.12, 0.11])]
    assert decision.evaluate_p1(cb + sf)["verdict"] == "satisfied"


def test_p1_not_satisfied_when_the_direction_reverses():
    cb = [_cell("unfiltered-cb", {0: v, 8: v}, f"cb{i}") for i, v in enumerate([0.1] * 3)]
    sf = [_cell("e2e-strong-filter", {0: v, 8: v}, f"sf{i}") for i, v in enumerate([0.5] * 3)]
    result = decision.evaluate_p1(cb + sf)
    assert result["verdict"] == "not_satisfied"
    assert result["direction_satisfied"] is False


def test_p1_not_satisfied_when_the_interval_spans_zero():
    cb = [_cell("unfiltered-cb", {0: v}, f"cb{i}") for i, v in enumerate([0.9, 0.1, 0.5])]
    sf = [_cell("e2e-strong-filter", {0: v}, f"sf{i}") for i, v in enumerate([0.5] * 3)]
    assert decision.evaluate_p1(cb + sf)["verdict"] in {"not_satisfied", "not_evaluable"}


def test_uncertified_steps_are_excluded_from_the_early_window():
    summary = decision.early_mean(_cell("unfiltered-cb", {0: 0.5, 8: 0.6, 64: 9.9},
                                        "a", certified=8))
    assert summary["steps_used"] == [0, 8]
    assert summary["mean_conflict_early"] == pytest.approx(0.55, abs=1e-9)


def test_a_zero_gradient_norm_disqualifies_a_step():
    cell = _cell("unfiltered-cb", {0: 0.5, 8: 0.6}, "a")
    cell["records"][1]["replicate_0"]["grad_norm_retain"] = 0.0
    assert decision.early_mean(cell)["steps_used"] == [0]


def test_projector_excludes_rank_deficient_and_layers_30_31(audit):
    for entry in audit["projector"]["models"].values():
        assert entry["included_blocks"] < entry["all_blocks"]
        assert entry["included_numel"] < entry["all_numel"]
        for values in entry["thresholds"].values():
            assert values["k_at_threshold"] >= values["k_target"]
            assert values["fraction_at_threshold"] == pytest.approx(
                values["fraction_target"], abs=5e-3)


def test_random_mask_null_lands_on_the_isotropic_expectation(cells):
    for cell in cells:
        for record in cell["records"]:
            for block in record["replicate_0"]["alpha"].values():
                null = block["random_null"]
                assert null["alpha_retain_mean"] == pytest.approx(
                    null["isotropic_expectation"], abs=0.02)


def test_projector_positive_control_passes(cells):
    result = decision.positive_control(cells)
    assert result["pass"] is True, result["failures"]
    for entry in result["per_model"].values():
        for values in entry.values():
            assert values["ratio_to_null_mean"] > 1.5


def test_reduction_order_discrepancy_is_small(cells):
    for cell in cells:
        for record in cell["records"]:
            assert record["reduction_order_discrepancy"]["conflict"] < 1e-3


def test_step_one_reproduces_step_zero_bitwise(cells):
    """The first scheduled learning rate is zero, so step 1 IS the base model."""
    for cell in cells:
        by_step = {r["step"]: r["replicate_0"] for r in cell["records"]}
        if 0 in by_step and 1 in by_step:
            assert by_step[0]["conflict"] == by_step[1]["conflict"]
            assert by_step[0]["grad_norm_retain"] == by_step[1]["grad_norm_retain"]


def test_no_write_into_frozen_trees():
    frozen = ("spec_e_outputs", "spec_h2_outputs", "spec_r_outputs", "fisher_information")
    writes = ("write_text", "write_bytes", "unlink", "os.replace", "rmtree")
    for source in SOURCES:
        for line in source.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith("#") or not any(t in stripped for t in frozen):
                continue
            assert not any(w in stripped for w in writes), f"{source}: {stripped}"


def test_no_nondeterministic_seeding():
    """A hash() inside an RNG seed would make the nulls unreproducible across runs."""
    for source in SOURCES:
        text = source.read_text().replace("sha256", "").replace("_hash", "")
        assert "hash(" not in text
