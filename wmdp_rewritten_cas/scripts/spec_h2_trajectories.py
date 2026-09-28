#!/usr/bin/env python3
"""Spec H2 Phase B: performance-correlation trajectories over the frozen grid.

One shard per (run, step); every logged step of every run is computed. No
max-selection: the grid's checkpoint-maximum rule measures attack strength, while
H2's estimand is dynamics, and selecting the maximum would bias toward runs where
recovery happened fast (spec section 2).

Requires spec_h2_outputs/preflight.json to report status "ready".
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    from .spec_h2_io import (ROOT, Table, assert_write_target, enumerate_runs,
                             join_reference, load_config, load_table,
                             reference_paths, stable_seed, write_json)
    from .spec_h2_kernel import (bootstrap_index, joint3, joint3_batched,
                                 joint3_batched_l, mi2_batched, percentile_ci,
                                 permutation_batch, stats_from_joint,
                                 symmetric_null_draws)
    from .spec_h_common import solve_null_p
except ImportError:
    from spec_h2_io import (ROOT, Table, assert_write_target, enumerate_runs,
                            join_reference, load_config, load_table,
                            reference_paths, stable_seed, write_json)
    from spec_h2_kernel import (bootstrap_index, joint3, joint3_batched,
                                joint3_batched_l, mi2_batched, percentile_ci,
                                permutation_batch, stats_from_joint,
                                symmetric_null_draws)
    from spec_h_common import solve_null_p

ESTIMATORS = ("plugin", "mm")
# Workload identifiers: the fourth element of every RNG seed sequence, so that any
# single cell reproduces in isolation and adding cells never perturbs existing ones.
WORKLOAD_I_NULL = 1
WORKLOAD_MU_NULL = {"intact": 2, "shortcut": 3}
WORKLOAD_BOOTSTRAP = 4
WORKLOAD_NULL_MODEL = 5


def _finite_or_none(value) -> float | None:
    value = float(value)
    return None if not np.isfinite(value) else value


def _band(values: np.ndarray, points=(2.5, 5.0, 50.0, 95.0, 97.5)) -> dict:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    band = {"n_finite": int(finite.size), "n_draws": int(values.size)}
    if finite.size:
        band["mean"] = float(finite.mean())
        for point in points:
            band[f"p{point:g}".replace(".", "_")] = float(np.percentile(finite, point))
    return band


def compute_cell(run, step: int, table: Table, references: dict[str, Table],
                 config: dict) -> dict:
    """All Spec H2 quantities for one (run, step) cell."""
    resampling = config["resampling"]
    seed = resampling["seed"]
    run_hash = stable_seed(run.run_id)
    n = table.n

    f = table.argmax.astype(np.int64)
    y = table.gold.astype(np.int64)
    reference_labels = {
        name: join_reference(table, reference)
        for name, reference in references.items()
    }

    point = {
        name: {key: float(value[0]) for key, value in
               stats_from_joint(joint3(f, y, labels), n).items()}
        for name, labels in reference_labels.items()
    }

    # --- I_t null: label permutations, destroys F-Y coupling -------------------
    rng = np.random.default_rng([seed, run_hash, step, WORKLOAD_I_NULL])
    permutations = permutation_batch(n, resampling["i_null_permutations"], rng)
    i_null_draws = mi2_batched(f, y[permutations])
    i_null = {est: _band(i_null_draws[f"i_fy_{est}"]) for est in ESTIMATORS}
    # Clearance is a plugin decision (spec section 4): the plugin bias at 64 cells
    # is shared between the estimate and its permutation null at identical N.
    cleared = bool(point["intact"]["i_fy_plugin"] > i_null["plugin"]["p95"])

    # --- mu null: item-wise permutation of L, preserves all marginals ----------
    mu_null = {}
    for name, labels in reference_labels.items():
        rng = np.random.default_rng([seed, run_hash, step, WORKLOAD_MU_NULL[name]])
        shuffled = labels[permutation_batch(n, resampling["mu_null_permutations"], rng)]
        draws = stats_from_joint(joint3_batched_l(f, y, shuffled), n)
        mu_null[name] = {est: _band(draws[f"mu_{est}"]) for est in ESTIMATORS}

    # --- bootstrap: question-level resamples, shared indices across references --
    rng = np.random.default_rng([seed, run_hash, step, WORKLOAD_BOOTSTRAP])
    index = bootstrap_index(n, resampling["bootstrap_resamples"], rng)
    boot = {
        name: stats_from_joint(joint3_batched(f, y, labels, index), n)
        for name, labels in reference_labels.items()
    }

    # --- accuracy-matched null model (Nakkiran footnote 5) --------------------
    target = point["intact"]["i_ly_plugin"]
    p_correct, achieved = solve_null_p(
        y.tolist(), target, resampling["null_mi_tolerance_bits"]
    )
    rng = np.random.default_rng([seed, run_hash, step, WORKLOAD_NULL_MODEL])
    drawn = symmetric_null_draws(y, p_correct, resampling["null_model_draws"], rng)
    null_model_stats = stats_from_joint(joint3_batched_l(f, y, drawn), n)
    null_model = {
        "p_correct": p_correct,
        "target_i_l_y_plugin": target,
        "expected_i_null_y": achieved,
        "absolute_mismatch": abs(achieved - target),
        "matched": abs(achieved - target) <= resampling["null_mi_tolerance_bits"],
        "draw_i_l_y_plugin_mean": float(np.mean(null_model_stats["i_ly_plugin"])),
        "mu": {est: _band(null_model_stats[f"mu_{est}"]) for est in ESTIMATORS},
    }

    return {
        "n": n,
        "point": point,
        "i_null": i_null,
        "i_cleared": cleared,
        "mu_null": mu_null,
        "bootstrap": boot,
        "null_model": null_model,
        "source_sha256": table.sha256,
        "reference_sha256": {name: ref.sha256 for name, ref in references.items()},
        "reference_path": {name: ref.path for name, ref in references.items()},
        "rng_seeds": {
            "i_null": [seed, run_hash, step, WORKLOAD_I_NULL],
            "mu_null_intact": [seed, run_hash, step, WORKLOAD_MU_NULL["intact"]],
            "mu_null_shortcut": [seed, run_hash, step, WORKLOAD_MU_NULL["shortcut"]],
            "bootstrap": [seed, run_hash, step, WORKLOAD_BOOTSTRAP],
            "null_model": [seed, run_hash, step, WORKLOAD_NULL_MODEL],
        },
    }


def cell_records(run, step: int, cell: dict, table: Table) -> list[dict]:
    """Flatten one cell into (quantity, estimator) records.

    ``ratio`` and ``ratio_min`` carry a null value at uncleared steps (spec section
    4: null-valued, not zero). The suppressed number is not smuggled into the
    record under another key; it is recoverable from the mu and I_t records, which
    are always emitted raw.
    """
    records = []
    base = {
        "model": run.model,
        "condition": run.condition,
        "lr": run.learning_rate,
        "lr_value": float(run.learning_rate),
        "seed": run.seed,
        "run_id": run.run_id,
        "step": step,
        "n": cell["n"],
        "i_cleared": cell["i_cleared"],
        "source_table": {"path": str(Path(table.path).relative_to(ROOT)),
                         "sha256": cell["source_sha256"]},
    }

    def emit(quantity: str, estimator: str, value, ci, null, reference,
             register: str, notes=None, suppressed=False):
        record = dict(base)
        record.update({
            "quantity": quantity,
            "estimator": estimator,
            "register": register,
            "value": _finite_or_none(value) if value is not None else None,
            "ci95": ci,
            "null": null,
            "reference": reference,
        })
        if suppressed:
            record["suppressed_by_clearance"] = True
        if notes:
            record["notes"] = notes
        records.append(record)

    def reference_block(name: str | None):
        if name is None:
            return None
        return {"name": name,
                "path": str(Path(cell["reference_path"][name]).relative_to(ROOT)),
                "sha256": cell["reference_sha256"][name]}

    def ci_of(name: str, key: str):
        low, high, count = percentile_ci(cell["bootstrap"][name][key])
        return {"low": low, "high": high, "n_finite": count,
                "resamples": int(cell["bootstrap"][name][key].size)}

    for estimator in ESTIMATORS:
        point = cell["point"]
        # I_t, and the reference's own label information on this view.
        emit("i_t", estimator, point["intact"][f"i_fy_{estimator}"],
             ci_of("intact", f"i_fy_{estimator}"),
             {"kind": "label_permutation", **cell["i_null"][estimator]},
             None, "preregistered")
        for name in ("intact", "shortcut"):
            emit(f"i_l_y_{name}", estimator, point[name][f"i_ly_{estimator}"],
                 ci_of(name, f"i_ly_{estimator}"), None, reference_block(name),
                 "exploratory",
                 notes=["Amendment 1 Ruling 3: reference label information, the "
                        "structural bound on mu"])
            emit(f"cmi_{name}", estimator, point[name][f"cmi_{estimator}"],
                 ci_of(name, f"cmi_{estimator}"), None, reference_block(name),
                 "preregistered")
            emit(f"mu_{name}", estimator, point[name][f"mu_{estimator}"],
                 ci_of(name, f"mu_{estimator}"),
                 {"kind": "item_wise_L_permutation", **cell["mu_null"][name][estimator]},
                 reference_block(name), "preregistered")

        cleared = cell["i_cleared"]
        emit("ratio", estimator,
             point["intact"][f"ratio_{estimator}"] if cleared else None,
             ci_of("intact", f"ratio_{estimator}") if cleared else None,
             None, reference_block("intact"), "preregistered",
             suppressed=not cleared)
        emit("ratio_min", estimator,
             point["intact"][f"ratio_min_{estimator}"] if cleared else None,
             ci_of("intact", f"ratio_min_{estimator}") if cleared else None,
             None, reference_block("intact"), "exploratory",
             notes=["Amendment 1 Ruling 3, amended-post-exposure; no threshold is "
                    "preregistered for this quantity"],
             suppressed=not cleared)
        emit("ceiling", estimator,
             point["intact"][f"ceiling_{estimator}"] if cleared else None,
             ci_of("intact", f"ceiling_{estimator}") if cleared else None,
             None, reference_block("intact"), "exploratory",
             notes=["min(1, I(L_intact;Y) / I_t); the structural upper bound on ratio"],
             suppressed=not cleared)
        emit("mu_null_model", estimator,
             cell["null_model"]["mu"][estimator].get("mean"), None,
             {"kind": "accuracy_matched_random_reference",
              **cell["null_model"]["mu"][estimator]},
             {"name": "null_model", "p_correct": cell["null_model"]["p_correct"],
              "target_i_l_y_plugin": cell["null_model"]["target_i_l_y_plugin"],
              "matched": cell["null_model"]["matched"]},
             "preregistered")
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/spec_h2.json")
    parser.add_argument("--force", action="store_true",
                        help="recompute shards that already exist")
    args = parser.parse_args()

    root = ROOT
    config = load_config(root, args.config)
    outputs = root / config["outputs"]["root"]
    assert_write_target(outputs, root)

    preflight_path = root / config["outputs"]["preflight"]
    if not preflight_path.exists():
        print("preflight.json is absent; run spec_h2_preflight.py first", file=sys.stderr)
        return 2
    preflight = json.loads(preflight_path.read_text())
    if preflight["status"] != "ready":
        print(f"preflight status is {preflight['status']}; refusing to run",
              file=sys.stderr)
        return 2

    shards = outputs / "shards"
    shards.mkdir(parents=True, exist_ok=True)

    reference_tables = {
        model: {name: load_table(path)
                for name, path in reference_paths(model, root).items()}
        for model in config["expected_inventory"]["models"]
    }

    runs = enumerate_runs(root)
    computed = 0
    for run in runs:
        for step in run.steps:
            shard = shards / f"{run.run_id}__step-{step:03d}.json"
            if shard.exists() and not args.force:
                continue
            table = load_table(run.prediction_path(step))
            cell = compute_cell(run, step, table, reference_tables[run.model], config)
            write_json(shard, {
                "run_id": run.run_id, "step": step,
                "records": cell_records(run, step, cell, table),
                "diagnostics": {
                    "null_model": {k: v for k, v in cell["null_model"].items()
                                   if k != "mu"},
                    "rng_seeds": cell["rng_seeds"],
                },
            })
            computed += 1
        print(json.dumps({"run": run.run_id, "computed": computed}), flush=True)

    records, clearance = [], {}
    for shard in sorted(shards.glob("*.json")):
        payload = json.loads(shard.read_text())
        records.extend(payload["records"])
    for record in records:
        if record["quantity"] != "i_t" or record["estimator"] != "plugin":
            continue
        key = f"{record['model']}|{record['condition']}|{record['lr']}|{record['seed']}"
        entry = clearance.setdefault(key, {"cleared": 0, "uncleared": 0,
                                           "cleared_steps": [], "uncleared_steps": []})
        bucket = "cleared" if record["i_cleared"] else "uncleared"
        entry[bucket] += 1
        entry[f"{bucket}_steps"].append(record["step"])
    for entry in clearance.values():
        entry["cleared_steps"].sort()
        entry["uncleared_steps"].sort()

    ties = preflight_reference_ties(preflight)
    payload = {
        "spec": "H2",
        "status": "complete",
        "amendment": config["amendment"],
        "metadata": {
            "primary_estimator": config["estimators"]["primary"],
            "resampling": config["resampling"],
            "aggregation_deviation": config["aggregation"]["deviation_note"],
            "aggregation_rule": config["aggregation"]["rule"],
            "max_selection": config["aggregation"]["max_selection"],
            "supersedes": config["supersedes"],
            "reference_argmax_ties": ties,
            "reference_tie_caveat": (
                "e2e-strong-filter's options-only L_shortcut has 114/512 exact "
                "argmax ties (22.3%), all resolved toward the lowest letter index. "
                "Roughly a fifth of that reference is decided by the tie rule "
                "rather than by the model; every mu_shortcut statement for "
                "e2e-strong-filter carries this caveat."
            ),
            "clearance_by_run": clearance,
            "clearance_rule": config["resampling"]["clearance_rule"],
            "step_0_1_note": (
                "Steps 0 and 1 are argmax-identical in all 36 runs; every quantity "
                "at t=1 duplicates t=0 and the two are not distinguishable."
            ),
        },
        "records": records,
    }
    write_json(outputs / "trajectories.json", payload)
    print(json.dumps({"status": "complete", "cells": len(list(shards.glob('*.json'))),
                      "records": len(records), "computed_this_run": computed}))
    return 0


def preflight_reference_ties(preflight: dict) -> dict:
    for check in preflight["checks"]:
        if check["check"] == "probability_validity":
            return check["reference_argmax_ties"]
    return {}


if __name__ == "__main__":
    sys.exit(main())
