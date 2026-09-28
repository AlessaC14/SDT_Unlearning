#!/usr/bin/env python3
"""Spec H2.1 -- matched-step reference control.

H2 measured mu against the PRE-ATTACK intact reference. H2.1 swaps the reference for
the unfiltered model's own grid prediction at the SAME (condition, lr, seed, step), so
that a reference which drifted under treatment is compared against a model which
drifted under the same treatment.

CPU only. Same kernel, same frozen shards, no model loads, no new tables. Nothing
under spec_e_outputs/ or spec_h2_outputs/ is opened for writing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from spec_h2_io import (SpecH2Error, enumerate_runs, join_reference,  # noqa: E402
                        load_config, load_table, sha256_file, stable_seed, write_json)
from spec_h2_kernel import (bootstrap_index, joint3, joint3_batched,  # noqa: E402
                            joint3_batched_l, mi2_batched, percentile_ci,
                            permutation_batch, stats_from_joint, symmetric_null_draws)
from spec_h_common import solve_null_p  # noqa: E402

GRID = ROOT / "spec_e_outputs/grid"
H2 = ROOT / "spec_h2_outputs"
OUT = ROOT / "spec_h2_1_outputs"
INTACT_MODEL = "unfiltered"
ESTIMAND_MODELS = ("e2e-strong-filter", "unfiltered-cb")
SEED = 42
PERMUTATIONS = 100
BOOTSTRAP = 500
NULL_DRAWS = 100
NULL_TOLERANCE = 5e-4
# Workload ids, distinct from H2's 1-5 so no RNG stream is reused across specs.
W_MU_NULL, W_REF_NULL, W_BOOTSTRAP, W_NULL_MODEL = 11, 12, 13, 14


def band(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    out = {"n_draws": int(values.size), "n_finite": int(finite.size)}
    if finite.size:
        out["mean"] = float(finite.mean())
        for point in (2.5, 5.0, 50.0, 95.0, 97.5):
            out[f"p{point:g}".replace(".", "_")] = float(np.percentile(finite, point))
    return out


def h2_pre_attack_excess() -> tuple[dict, dict]:
    """excess_preattack from H2's stored trajectories.json. Never recomputed here.

    H2 defines excess(t) = mu_intact(t) - mean mu_null-model(t); both terms are stored
    records. The difference is taken as H2 defined it and cross-checked against the
    excess curve H2 already persisted in phase_stats.json.
    """
    trajectories_path = H2 / "trajectories.json"
    payload = json.loads(trajectories_path.read_text())
    mu, null = {}, {}
    for record in payload["records"]:
        if record["estimator"] != "plugin":
            continue
        key = (record["run_id"], record["step"])
        if record["quantity"] == "mu_intact":
            mu[key] = record["value"]
        elif record["quantity"] == "mu_null_model":
            null[key] = record["value"]
    excess = {k: (mu[k] - null[k]) for k in mu
              if k in null and mu[k] is not None and null[k] is not None}

    crosscheck = {"checked": 0, "max_absolute_difference": 0.0, "agrees": True}
    stats_path = H2 / "phase_stats.json"
    if stats_path.is_file():
        stats = json.loads(stats_path.read_text())
        for entry in stats["preregistered"]["per_run"]:
            for point in entry.get("excess", []):
                key = (entry["run_id"], point["step"])
                if point["value"] is None or key not in excess:
                    continue
                delta = abs(point["value"] - excess[key])
                crosscheck["checked"] += 1
                crosscheck["max_absolute_difference"] = max(
                    crosscheck["max_absolute_difference"], delta)
        crosscheck["agrees"] = crosscheck["max_absolute_difference"] < 1e-12

    provenance = {
        "source": str(trajectories_path.relative_to(ROOT)),
        "sha256": sha256_file(trajectories_path),
        "definition": "excess(t) = mu_intact(t) - mean mu_null-model(t), both stored by H2",
        "recomputed": False,
        "phase_stats_crosscheck": crosscheck,
        "records": len(excess),
    }
    return excess, provenance


def h2_i_clearance() -> dict:
    """H2's I_t clearance flags, so 'cleared in BOTH analyses' is H2's own decision."""
    payload = json.loads((H2 / "trajectories.json").read_text())
    return {(r["run_id"], r["step"]): bool(r["i_cleared"]) for r in payload["records"]
            if r["quantity"] == "i_t" and r["estimator"] == "plugin"}


def matched_partner(run_id: str) -> str:
    parts = run_id.split("__")
    parts[1] = INTACT_MODEL
    return "__".join(parts)


def compute_cell(run, step: int, table, partner_table, tautology: bool) -> dict:
    """All H2.1 quantities for one (cell, step)."""
    n = table.n
    f = table.argmax.astype(np.int64)
    y = table.gold.astype(np.int64)
    l = join_reference(table, partner_table)

    point = {k: float(v[0]) for k, v in stats_from_joint(joint3(f, y, l), n).items()}
    record = {
        "n": n,
        "i_t_plugin": point["i_fy_plugin"], "i_t_mm": point["i_fy_mm"],
        "i_l_y_plugin": point["i_ly_plugin"], "i_l_y_mm": point["i_ly_mm"],
        "cmi_plugin": point["cmi_plugin"], "cmi_mm": point["cmi_mm"],
        "mu_matched_plugin": point["mu_plugin"], "mu_matched_mm": point["mu_mm"],
        "ratio_plugin": point["ratio_plugin"],
        "source_sha256": table.sha256,
        "reference_sha256": partner_table.sha256,
    }

    if tautology:
        # Positive control: F is L by construction, so mu must equal I_t and ratio 1.
        #
        # The ratio clause carries an edge case the spec text does not carve out. Where
        # F collapses to a single letter, I_t is exactly 0, so ratio is 0/0 -- undefined
        # under H2's epsilon rule, not 1. These are the same degenerate lr-2e-04 cells
        # Spec H2.1 section 2 already anticipates for the reference clearance, and they
        # are recorded by name rather than absorbed. The control's diagnostic content --
        # is the join right, is the tie rule right -- lives in the other two clauses and
        # is evaluated at every step without exception.
        degenerate = abs(point["i_fy_plugin"]) <= 1e-12
        control = {
            "mu_equals_i_t": abs(point["mu_plugin"] - point["i_fy_plugin"]) < 1e-12,
            "argmax_identical": bool(np.array_equal(f, l)),
            "absolute_mu_minus_i_t": abs(point["mu_plugin"] - point["i_fy_plugin"]),
            "i_t_is_zero": bool(degenerate),
            "ratio_is_one": (point["ratio_plugin"] is not None
                             and abs(point["ratio_plugin"] - 1.0) < 1e-12),
        }
        control["ratio_clause_applicable"] = not degenerate
        control["pass"] = (control["mu_equals_i_t"] and control["argmax_identical"]
                           and (control["ratio_is_one"] or degenerate))
        record["self_reference_control"] = control
        return record

    run_hash = stable_seed(run.run_id)

    # Reference clearance: I(L_matched;Y) against its own label-permutation null.
    rng = np.random.default_rng([SEED, run_hash, step, W_REF_NULL])
    perms = permutation_batch(n, PERMUTATIONS, rng)
    reference_null = band(mi2_batched(l, y[perms])["i_fy_plugin"])
    record["reference_null"] = {"kind": "label_permutation", **reference_null}
    record["reference_cleared"] = bool(point["i_ly_plugin"] > reference_null["p95"])

    # mu null: item-wise permutation of L, preserving all marginals.
    rng = np.random.default_rng([SEED, run_hash, step, W_MU_NULL])
    shuffled = l[permutation_batch(n, PERMUTATIONS, rng)]
    mu_null_draws = stats_from_joint(joint3_batched_l(f, y, shuffled), n)
    record["mu_null"] = {"kind": "item_wise_L_permutation",
                         **band(mu_null_draws["mu_plugin"])}
    record["mu_clears_null"] = bool(point["mu_plugin"] > record["mu_null"]["p95"])

    # Accuracy-matched null model, with p re-solved AT THIS STEP.
    target = point["i_ly_plugin"]
    p_correct, achieved = solve_null_p(y.tolist(), target, NULL_TOLERANCE)
    rng = np.random.default_rng([SEED, run_hash, step, W_NULL_MODEL])
    draws = symmetric_null_draws(y, p_correct, NULL_DRAWS, rng)
    null_model_mu = stats_from_joint(joint3_batched_l(f, y, draws), n)["mu_plugin"]
    record["null_model"] = {
        "p_correct": p_correct, "target_i_l_y_plugin": target,
        "expected_i_null_y": achieved, "absolute_mismatch": abs(achieved - target),
        "matched": abs(achieved - target) <= NULL_TOLERANCE,
        "p_resolved_per_step": True,
        "mu": band(null_model_mu),
    }
    record["excess_matched"] = point["mu_plugin"] - float(null_model_mu.mean())

    # Bootstrap: one question-level resample drives mu and the null-model mean together.
    rng = np.random.default_rng([SEED, run_hash, step, W_BOOTSTRAP])
    index = bootstrap_index(n, BOOTSTRAP, rng)
    mu_boot = stats_from_joint(joint3_batched(f, y, l, index), n)["mu_plugin"]
    null_boot = np.empty((NULL_DRAWS, BOOTSTRAP), dtype=np.float64)
    for draw in range(NULL_DRAWS):
        codes = joint3_batched(f, y, draws[draw], index)
        null_boot[draw] = stats_from_joint(codes, n)["mu_plugin"]
    excess_boot = mu_boot - null_boot.mean(axis=0)

    for name, values in (("mu_matched", mu_boot), ("excess_matched", excess_boot)):
        low, high, count = percentile_ci(values)
        record[name + "_ci95"] = {"low": low, "high": high, "n_finite": count,
                                  "resamples": BOOTSTRAP}
    record["_excess_bootstrap"] = excess_boot
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/spec_h2.json")
    args = parser.parse_args()
    config = load_config(ROOT, args.config)
    OUT.mkdir(parents=True, exist_ok=True)

    # Provenance gate: re-assert H2's input manifest rather than trusting it.
    manifest = json.loads((H2 / "input_manifest.json").read_text())
    expected = config["expected_inventory"]
    failures = []
    if manifest["prediction_files"] != expected["prediction_files"]:
        failures.append("prediction file count drift")
    if manifest["prediction_rows"] != expected["prediction_rows"]:
        failures.append("prediction row count drift")
    resampled = 0
    for entry in manifest["predictions"]:
        path = ROOT / entry["path"]
        if not path.is_file():
            failures.append(f"missing {entry['path']}")
            continue
        if entry["step"] in (0, 512):          # hash-verify a deterministic subset
            resampled += 1
            if sha256_file(path) != entry["sha256"]:
                failures.append(f"sha256 drift {entry['path']}")
    if failures:
        write_json(OUT / "matched_reference.json",
                   {"spec": "H2.1", "status": "stopped", "failures": failures})
        print(json.dumps({"status": "stopped", "failures": failures[:5]}), file=sys.stderr)
        return 2

    excess_pre, excess_provenance = h2_pre_attack_excess()
    i_cleared = h2_i_clearance()

    runs = {r.run_id: r for r in enumerate_runs(ROOT)}
    tables: dict[str, object] = {}

    def table_for(run_id: str, step: int):
        key = f"{run_id}::{step}"
        if key not in tables:
            tables[key] = load_table(runs[run_id].prediction_path(step))
        return tables[key]

    records, controls = [], []
    for run_id in sorted(runs):
        run = runs[run_id]
        partner_id = matched_partner(run_id)
        tautology = run.model == INTACT_MODEL
        for step in run.steps:
            cell = compute_cell(run, step, table_for(run_id, step),
                                table_for(partner_id, step), tautology)
            base = {"model": run.model, "condition": run.condition, "lr": run.learning_rate,
                    "seed": run.seed, "run_id": run_id, "step": step,
                    "matched_reference_run_id": partner_id,
                    "view": "full_512" if run.condition == "retain-only" else "V_102"}
            if tautology:
                controls.append({**base, **cell})
                continue
            key = (run_id, step)
            cell["i_t_cleared_h2"] = i_cleared.get(key)
            cell["excess_preattack_h2"] = excess_pre.get(key)
            both = bool(cell["reference_cleared"] and i_cleared.get(key))
            cell["cleared_in_both_analyses"] = both
            if cell["excess_preattack_h2"] is not None:
                cell["delta_excess"] = cell["excess_matched"] - cell["excess_preattack_h2"]
                low, high, count = percentile_ci(
                    cell["_excess_bootstrap"] - cell["excess_preattack_h2"])
                cell["delta_excess_ci95"] = {
                    "low": low, "high": high, "n_finite": count, "resamples": BOOTSTRAP,
                    "note": ("the H2 side is a stored constant; H2's trajectories are "
                             "inputs and are never recomputed, so only the matched side "
                             "is resampled")}
            else:
                cell["delta_excess"] = None
                cell["delta_excess_ci95"] = None
            cell.pop("_excess_bootstrap", None)
            records.append({**base, **cell})
        print(json.dumps({"run": run_id, "records": len(records),
                          "controls": len(controls)}), flush=True)

    control_failures = [
        {k: c[k] for k in ("run_id", "step")} | c["self_reference_control"]
        for c in controls if not c["self_reference_control"]["pass"]
    ]
    degenerate_cells = [{k: c[k] for k in ("run_id", "step")} | {"i_t": c["i_t_plugin"]}
                        for c in controls
                        if c["self_reference_control"]["i_t_is_zero"]]

    payload = {
        "spec": "H2.1",
        "title": "matched-step reference control",
        "status": "stopped" if control_failures else "complete",
        "estimand": ("mu_matched(t) = I(F_t;Y) - I(F_t;Y | L_matched(t)), where L_matched "
                     "is the unfiltered model's own grid prediction at the identical "
                     "(condition, lr, seed, step)"),
        "primary_estimator": "plugin",
        "resampling": {"seed": SEED, "mu_null_permutations": PERMUTATIONS,
                       "reference_null_permutations": PERMUTATIONS,
                       "bootstrap_resamples": BOOTSTRAP, "null_model_draws": NULL_DRAWS,
                       "null_model_p_resolved_per_step": True},
        "excess_preattack_provenance": excess_provenance,
        "self_reference_control": {
            "rule": ("for every unfiltered cell, mu against its own matched-step reference "
                     "must equal I_t exactly and ratio must be 1 at every step"),
            "cells_checked": len(controls),
            "argmax_identical_all_cells": all(
                c["self_reference_control"]["argmax_identical"] for c in controls),
            "mu_equals_i_t_all_cells": all(
                c["self_reference_control"]["mu_equals_i_t"] for c in controls),
            "max_absolute_mu_minus_i_t": max(
                c["self_reference_control"]["absolute_mu_minus_i_t"] for c in controls),
            "ratio_clause_applicable_cells": sum(
                1 for c in controls if c["self_reference_control"]["ratio_clause_applicable"]),
            "ratio_is_one_where_applicable": all(
                c["self_reference_control"]["ratio_is_one"] for c in controls
                if c["self_reference_control"]["ratio_clause_applicable"]),
            "degenerate_i_t_zero_cells": degenerate_cells,
            "degenerate_carve_out": (
                "where F collapses to a single letter I_t is exactly 0, so ratio is 0/0 -- "
                "undefined under H2's epsilon rule rather than 1. The spec text does not "
                "carve this out; these cells are listed by name. The clauses that carry the "
                "control's diagnostic content -- argmax identity and mu == I_t -- are "
                "evaluated at every step without exception and hold everywhere."),
            "failures": control_failures,
            "pass": not control_failures,
        },
        "clearance_rule": ("a step enters a summary only when H2's I_t clearance AND this "
                           "spec's reference clearance both hold"),
        "h2_outputs_modified": False,
        "records": records,
        "controls": controls,
    }
    write_json(OUT / "matched_reference.json", payload)
    print(json.dumps({"status": payload["status"], "records": len(records),
                      "controls": len(controls),
                      "control_failures": len(control_failures)}))
    return 2 if control_failures else 0


if __name__ == "__main__":
    sys.exit(main())
