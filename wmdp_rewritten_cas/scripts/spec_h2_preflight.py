#!/usr/bin/env python3
"""Spec H2 Phase A preflight. Six checks, all fail-closed.

Exit 0 with status "ready", or exit 2 with status "stopped" and a failures list
naming the exact offending file. No downstream H2 script may run unless
preflight.json reports "ready".
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

try:
    from .spec_h2_io import (ROOT, SpecH2Error, Table, build_input_manifest,
                             enumerate_runs, join_reference, load_config,
                             load_table, reference_paths, write_json)
except ImportError:
    from spec_h2_io import (ROOT, SpecH2Error, Table, build_input_manifest,
                            enumerate_runs, join_reference, load_config,
                            load_table, reference_paths, write_json)


_TABLE_CACHE: dict[str, Table] = {}


def cached_table(path) -> Table:
    """Preflight touches every grid table several times; parse and hash each once."""
    key = str(path)
    if key not in _TABLE_CACHE:
        _TABLE_CACHE[key] = load_table(path)
    return _TABLE_CACHE[key]


def _check(name: str, passed: bool, detail: dict) -> dict:
    return {"check": name, "pass": bool(passed), **detail}


def check_grid_inventory(config: dict, manifest: dict, runs: list) -> dict:
    expected = config["expected_inventory"]
    steps = tuple(expected["steps"])
    failures = []

    if len(runs) != expected["runs"]:
        failures.append(f"expected {expected['runs']} completed runs, found {len(runs)}")
    if manifest["prediction_files"] != expected["prediction_files"]:
        failures.append(
            f"expected {expected['prediction_files']} prediction files, "
            f"found {manifest['prediction_files']}"
        )
    if manifest["prediction_rows"] != expected["prediction_rows"]:
        failures.append(
            f"expected {expected['prediction_rows']} prediction rows, "
            f"found {manifest['prediction_rows']}"
        )

    for run in runs:
        if run.steps != steps:
            failures.append(f"{run.run_id}: logged steps {run.steps} != {steps}")
        if run.model not in expected["models"]:
            failures.append(f"{run.run_id}: unexpected model {run.model}")
        if run.condition not in expected["conditions"]:
            failures.append(f"{run.run_id}: unexpected condition {run.condition}")
        if run.learning_rate not in expected["learning_rates"]:
            failures.append(f"{run.run_id}: unexpected learning rate {run.learning_rate}")
        if run.seed not in expected["seeds"]:
            failures.append(f"{run.run_id}: unexpected seed {run.seed}")

    for record in manifest["predictions"]:
        wanted = expected["condition_n"][record["condition"]]
        if record["rows"] != wanted:
            failures.append(
                f"{record['path']}: {record['rows']} rows, expected {wanted}"
            )

    cells = Counter((run.model, run.condition) for run in runs)
    return _check("grid_inventory", not failures, {
        "runs": len(runs),
        "prediction_files": manifest["prediction_files"],
        "prediction_rows": manifest["prediction_rows"],
        "cells_per_model_condition": {f"{m}|{c}": n for (m, c), n in sorted(cells.items())},
        "failures": failures,
    })


def check_integer_gates(config: dict, references: dict[str, dict[str, Table]]) -> dict:
    gates = config["integer_gates"]
    canonical = gates["canonical_exact_path_correct_of_512"]
    retired = gates["retired_bf16_path_correct_of_512"]
    observed, failures = {}, []
    for model, expected in canonical.items():
        table = references[model]["own_full"]
        correct = int((table.argmax == table.gold).sum())
        observed[model] = correct
        if table.n != 512:
            failures.append(f"{model}: full table has {table.n} rows, expected 512")
        if correct != expected:
            failures.append(
                f"{model}: exact-path correct count {correct} != canonical {expected}"
            )
        if correct == retired[model]:
            failures.append(
                f"{model}: correct count {correct} collides with the retired "
                "bf16-path value; a retired table has been reintroduced"
            )
    return _check("integer_gates", not failures, {
        "source": gates["source"],
        "canonical": canonical,
        "observed": observed,
        "retired_bf16": retired,
        "failures": failures,
    })


def check_reference_parity(config: dict, runs: list,
                           references: dict[str, dict[str, Table]]) -> dict:
    """Per-model self-parity (Amendment 1 Ruling 5).

    Each model's own exact-path full table against that model's six retain-only
    step-zero grid tables. The cross-model reading is wrong by construction and is
    not implemented.
    """
    gate = config["parity_gate"]
    records, failures = [], []
    for run in runs:
        if run.condition != "retain-only":
            continue
        table = cached_table(run.prediction_path(0))
        reference = references[run.model]["own_full"]
        position = {qid: index for index, qid in enumerate(reference.ids)}
        order = np.array([position[qid] for qid in table.ids], dtype=np.int64)
        consistency = float((table.argmax == reference.argmax[order]).mean())
        delta = float(np.abs(table.probabilities - reference.probabilities[order]).max())
        ok = (consistency == gate["required_argmax_consistency"]
              and delta == gate["required_max_abs_delta_p"]
              and set(table.ids) == set(reference.ids))
        records.append({
            "model": run.model,
            "run_id": run.run_id,
            "n": table.n,
            "argmax_self_consistency": consistency,
            "max_abs_delta_p": delta,
            "id_sets_equal": set(table.ids) == set(reference.ids),
            "pass": ok,
        })
        if not ok:
            failures.append(
                f"{run.run_id}: argmax consistency {consistency:.6f}, "
                f"max|delta p| {delta:.3g}"
            )
    if len(records) != gate["cells"]:
        failures.append(f"expected {gate['cells']} parity cells, found {len(records)}")
    return _check("reference_parity_self", not failures, {
        "reading": gate["reading"],
        "cells": len(records),
        "records": records,
        "failures": failures,
    })


def check_id_alignment(config: dict, runs: list, intact: Table,
                       references: dict[str, dict[str, Table]]) -> dict:
    """Retain-only: ordered equality with the reference. Forget-T: the V-102 set."""
    split_v = json.loads((ROOT / config["inputs"]["split_V"]).read_text())
    v_ids = {str(item["question_id"]) for item in split_v["items"]}
    reference_ids = list(intact.ids)
    records, failures = [], []

    for run in runs:
        for step in run.steps:
            table = cached_table(run.prediction_path(step))
            record = {"run_id": run.run_id, "step": step, "n": table.n}
            if run.condition == "retain-only":
                ordered = list(table.ids) == reference_ids
                record["ordered_equality_with_reference"] = ordered
                record["pass"] = ordered
                if not ordered:
                    failures.append(
                        f"{run.run_id} step {step}: retain-only ID order drifted "
                        "from the reference order"
                    )
            else:
                subset = set(table.ids).issubset(set(reference_ids))
                is_v = set(table.ids) == v_ids
                record["subset_of_reference"] = subset
                record["equals_split_V"] = is_v
                record["pass"] = subset and is_v
                if not subset:
                    failures.append(
                        f"{run.run_id} step {step}: forget-T IDs are not a subset "
                        "of the canonical 512"
                    )
                if not is_v:
                    failures.append(
                        f"{run.run_id} step {step}: forget-T IDs are not the V-102 set"
                    )
            try:
                join_reference(table, intact)
                join_reference(table, references[run.model]["shortcut"])
            except SpecH2Error as error:
                record["pass"] = False
                failures.append(f"{run.run_id} step {step}: {error}")
            records.append(record)

    return _check("id_alignment", not failures, {
        "retain_only_rule": "ordered equality with the reference table",
        "forget_T_rule": "set equality with wmdp_split_V and subset of the canonical 512",
        "cells": len(records),
        "failures": failures,
        "records": records,
    })


def check_reference_hashes(manifest: dict) -> dict:
    failures = []
    for record in manifest["references"]:
        if record["sha256"] != record["manifest_sha256"]:
            failures.append(
                f"{record['path']}: sha256 {record['sha256']} != manifest "
                f"{record['manifest_sha256']}"
            )
        if record["rows"] != record["manifest_rows"]:
            failures.append(
                f"{record['path']}: {record['rows']} rows != manifest "
                f"{record['manifest_rows']}"
            )
    return _check("reference_hashes", not failures, {
        "tables": [
            {k: record[k] for k in ("model", "table", "sha256", "rows")}
            for record in manifest["references"]
        ],
        "failures": failures,
    })


def check_probability_validity(runs: list,
                               references: dict[str, dict[str, Table]]) -> dict:
    """Row sums, sign, finiteness (hard), and argmax tie counts (recorded only).

    ``load_table`` raises on any hard violation, so reaching this point without an
    exception is the pass condition. Tie counts are not a failure: they are an
    interpretation caveat carried into trajectories.json metadata (Amendment 1
    Ruling 6).
    """
    failures, ties = [], {}
    for model, tables in references.items():
        for name in ("own_full", "shortcut"):
            table = tables[name]
            ties[f"{model}|{name}"] = {
                "argmax_ties": table.argmax_ties,
                "n": table.n,
                "tie_rate": table.argmax_ties / table.n,
                "path": table.path,
            }
    grid_ties = []
    for run in runs:
        for step in run.steps:
            try:
                table = cached_table(run.prediction_path(step))
            except SpecH2Error as error:
                failures.append(str(error))
                continue
            if table.argmax_ties:
                grid_ties.append({
                    "run_id": run.run_id, "step": step,
                    "argmax_ties": table.argmax_ties, "n": table.n,
                })
    return _check("probability_validity", not failures, {
        "tolerance_row_sum": 1e-5,
        "reference_argmax_ties": ties,
        "grid_cells_with_argmax_ties": len(grid_ties),
        "grid_argmax_ties": grid_ties,
        "tie_rule": "lowest letter index wins",
        "ties_are_failures": False,
        "failures": failures,
    })


def check_step_zero_one_identity(runs: list) -> dict:
    """Recorded observation, not a gate (Amendment 1 Ruling 6).

    Steps 0 and 1 are argmax-identical in all 36 runs, so t80 values of 0 and 1 are
    not distinguishable and phase_stats records them as "<=1".
    """
    identical, differing = 0, []
    for run in runs:
        zero = cached_table(run.prediction_path(0))
        one = cached_table(run.prediction_path(1))
        if list(zero.ids) == list(one.ids) and np.array_equal(zero.argmax, one.argmax):
            identical += 1
        else:
            differing.append(run.run_id)
    return _check("step_zero_one_argmax_identity", True, {
        "gate": False,
        "runs_with_identical_argmax": identical,
        "runs_total": len(runs),
        "runs_differing": differing,
        "consequence": "t80 in {0, 1} is reported as '<=1'",
    })


def main() -> int:
    root = ROOT
    config = load_config(root)
    output = root / config["outputs"]["preflight"]

    reference_root = root / "spec_e_outputs/references_grid_path"
    references: dict[str, dict[str, Table]] = {
        model: {
            "own_full": cached_table(reference_root / model / "full.jsonl"),
            "shortcut": cached_table(reference_paths(model, root)["shortcut"]),
        }
        for model in config["expected_inventory"]["models"]
    }
    # L_intact is the unfiltered exact-path full table, shared across all models.
    intact = cached_table(reference_paths("unfiltered", root)["intact"])

    runs = enumerate_runs(root)
    manifest = build_input_manifest(root)
    write_json(root / config["outputs"]["input_manifest"], manifest)

    checks = [
        check_grid_inventory(config, manifest, runs),
        check_integer_gates(config, references),
        check_reference_parity(config, runs, references),
        check_id_alignment(config, runs, intact, references),
        check_reference_hashes(manifest),
        check_probability_validity(runs, references),
        check_step_zero_one_identity(runs),
    ]
    gates = [c for c in checks if c["check"] != "step_zero_one_argmax_identity"]
    ready = all(c["pass"] for c in gates)

    payload = {
        "spec": "H2",
        "phase": "A_preflight",
        "status": "ready" if ready else "stopped",
        "amendment": config["amendment"],
        "checks": checks,
        "failures": [f for c in gates for f in c.get("failures", [])],
        "input_manifest": config["outputs"]["input_manifest"],
    }
    write_json(output, payload)

    report = [
        "# Spec H2 Phase A preflight",
        "",
        f"Status: **{payload['status'].upper()}**",
        "",
        "| check | pass |",
        "| --- | --- |",
    ]
    report += [f"| {c['check']} | {'PASS' if c['pass'] else 'FAIL'} |" for c in checks]
    if payload["failures"]:
        report += ["", "## Failures", ""] + [f"- {f}" for f in payload["failures"]]
    (root / "spec_h2_outputs/preflight_report.md").write_text("\n".join(report) + "\n")

    print(json.dumps({"status": payload["status"],
                      "failures": len(payload["failures"])}))
    return 0 if ready else 2


if __name__ == "__main__":
    sys.exit(main())
