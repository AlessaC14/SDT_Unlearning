#!/usr/bin/env python3
"""Spec G section 1 decision, plus the section 2 controls, evaluated mechanically."""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "spec_gradgeo_outputs"
SEED = 42
STEP_BOOTSTRAP = 10000
EARLY_MAX_STEP = 64
K_FRACTIONS = ("0.01", "0.05", "0.10")


def load_cells() -> list[dict]:
    return [json.loads(Path(p).read_text())
            for p in sorted(glob.glob(str(OUT / "cells" / "*.json")))]


def cleared(record: dict) -> bool:
    """P1's 'cleared steps': certified AND both gradient norms nonzero."""
    r = record["replicate_0"]
    return bool(record["certified"] and r["grad_norm_forget"] > 0
                and r["grad_norm_retain"] > 0)


def early_mean(cell: dict) -> dict:
    steps = [r for r in cell["records"] if cleared(r) and r["step"] <= EARLY_MAX_STEP]
    values = [r["replicate_0"]["conflict"] for r in steps]
    return {
        "run_id": cell["cell"]["run_id"], "model": cell["cell"]["model"],
        "condition": cell["cell"]["condition"], "lr": cell["cell"]["lr"],
        "seed": cell["cell"]["seed"],
        "certified_through": cell["cell"]["certified_through"],
        "steps_used": [r["step"] for r in steps],
        "n_steps": len(values),
        "mean_conflict_early": float(np.mean(values)) if values else None,
        "conflict_by_step": {str(r["step"]): r["replicate_0"]["conflict"]
                             for r in cell["records"]},
    }


def positive_control(cells: list[dict]) -> dict:
    """alpha_r(0) must substantially exceed the random-mask null, every model, every k."""
    per_model, failures = {}, []
    for cell in cells:
        step0 = next((r for r in cell["records"] if r["step"] == 0), None)
        if step0 is None:
            continue
        model = cell["cell"]["model"]
        entry = per_model.setdefault(model, {})
        for k in K_FRACTIONS:
            block = step0["replicate_0"]["alpha"][k]
            observed = block["alpha_retain"]
            null = block["random_null"]
            passes = observed > null["alpha_retain_p97_5"]
            entry[k] = {
                "alpha_retain_step0": observed,
                "random_null_mean": null["alpha_retain_mean"],
                "random_null_p97_5": null["alpha_retain_p97_5"],
                "isotropic_expectation": null["isotropic_expectation"],
                "ratio_to_null_mean": (observed / null["alpha_retain_mean"]
                                       if null["alpha_retain_mean"] else None),
                "exceeds_null": bool(passes),
            }
            if not passes:
                failures.append(f"{model} k={k}: alpha_r(0)={observed:.4f} does not exceed "
                                f"the random-mask null p97.5 {null['alpha_retain_p97_5']:.4f}")
    return {
        "rule": ("alpha_r(0) must substantially exceed the random-mask null for every model "
                 "at every k; failure stops the alpha channel, conflict proceeds regardless"),
        "per_model": per_model,
        "failures": failures,
        "pass": not failures,
        "alpha_channel_status": "valid" if not failures else "stopped",
    }


def conflict_vs_null(cells: list[dict]) -> dict:
    total = outside = 0
    for cell in cells:
        for record in cell["records"]:
            r = record["replicate_0"]
            total += 1
            outside += bool(abs(r["conflict"]) > max(abs(r["conflict_null"]["p2_5"]),
                                                     abs(r["conflict_null"]["p97_5"])))
    return {"checkpoints": total, "outside_sign_shuffle_band": outside,
            "fraction": outside / total if total else None}


def evaluate_p1(cells: list[dict]) -> dict:
    arms = {}
    for cell in cells:
        if cell["cell"]["condition"] != "retain-only":
            continue
        arms.setdefault(cell["cell"]["model"], []).append(early_mean(cell))
    cb = arms.get("unfiltered-cb", [])
    sf = arms.get("e2e-strong-filter", [])

    statement = ("under retain-only, the mean of conflict(t) over cleared steps <= 64 is "
                 "greater for unfiltered-cb than for e2e-strong-filter, aggregated over "
                 "(lr, seed) by median, with a run-level bootstrap 95% CI on the difference "
                 "excluding zero")
    usable_cb = [a for a in cb if a["mean_conflict_early"] is not None]
    usable_sf = [a for a in sf if a["mean_conflict_early"] is not None]

    result = {
        "statement": statement,
        "early_window": f"cleared steps <= {EARLY_MAX_STEP}",
        "cleared_definition": "certified (Spec R Amendment 3) AND both gradient norms nonzero",
        "unfiltered_cb_runs": cb, "e2e_strong_filter_runs": sf,
        "runs_per_arm": {"unfiltered-cb": len(usable_cb),
                         "e2e-strong-filter": len(usable_sf)},
    }

    if not usable_cb or not usable_sf:
        result["verdict"] = "not_evaluable"
        result["reason"] = "one or both retain-only arms produced no cleared early step"
        return result

    cb_median = float(np.median([a["mean_conflict_early"] for a in usable_cb]))
    sf_median = float(np.median([a["mean_conflict_early"] for a in usable_sf]))
    result["median_mean_conflict_early"] = {"unfiltered-cb": cb_median,
                                            "e2e-strong-filter": sf_median}
    result["difference"] = cb_median - sf_median
    result["direction_satisfied"] = cb_median > sf_median

    # P1 is a conjunction: direction AND interval. If the direction conjunct is
    # definitively false, the conjunction is false and no interval could rescue it, so
    # "not_satisfied" is the correct evaluation of the stated rule rather than
    # "not_evaluable". Only when the direction holds does the missing interval leave the
    # rule genuinely undecided. This is an ordering of the rule's own conjuncts, not a
    # change to it: nothing here can produce "satisfied" without the interval.
    if not result["direction_satisfied"]:
        result["verdict"] = "not_satisfied"
        result["reason"] = (
            "the preregistered direction fails: the unfiltered-cb arm median is BELOW the "
            "e2e-strong-filter arm median, so the conjunction P1 states is false "
            "independently of the interval. The run-level interval was in any case "
            "unavailable at Tier 1 (n=1 run per arm).")
        result["interval_available"] = False
        return result

    if len(usable_cb) < 2 or len(usable_sf) < 2:
        result["verdict"] = "not_evaluable"
        result["reason"] = (
            "P1 requires a run-level bootstrap 95% CI on the difference of arm medians, "
            "aggregated over (lr, seed). Tier 1 contains exactly one (lr, seed) per "
            f"(model, condition), so each arm has n={len(usable_cb)} run and a run-level "
            "bootstrap is undefined. The direction is reported above; the interval the rule "
            "requires cannot be formed from Tier 1 alone.")
        result["what_would_evaluate_it"] = (
            "adapters at the early logged steps (<= 64) for the remaining five (lr, seed) "
            "of each retain-only arm. Spec R Tier 2 retained adapters only at "
            "{0, peak, 512}, so those checkpoints do not exist and this is an adapter-"
            "retention question for Spec R, not an analysis choice available here.")
        return result

    rng = np.random.default_rng(SEED)
    a = np.array([x["mean_conflict_early"] for x in usable_cb])
    b = np.array([x["mean_conflict_early"] for x in usable_sf])
    draws = (np.median(a[rng.integers(0, a.size, (STEP_BOOTSTRAP, a.size))], axis=1)
             - np.median(b[rng.integers(0, b.size, (STEP_BOOTSTRAP, b.size))], axis=1))
    ci = [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]
    result["run_level_bootstrap"] = {"ci95": ci, "resamples": STEP_BOOTSTRAP, "seed": SEED}
    satisfied = cb_median > sf_median and ci[0] > 0
    result["verdict"] = "satisfied" if satisfied else "not_satisfied"
    result["reason"] = ("the arm-median difference is positive and its run-level 95% "
                        "interval excludes zero" if satisfied else
                        "the direction or the interval condition does not hold")
    return result


def main() -> int:
    cells = load_cells()
    payload = {
        "spec": "G", "status": "complete", "cells": len(cells),
        "primary_quantity": "conflict(t) = cos(g_f(t), g_r(t))",
        "controls": {
            "projector_positive_control": positive_control(cells),
            "conflict_vs_sign_shuffle_null": conflict_vs_null(cells),
        },
        "preregistered": {"P1": evaluate_p1(cells)},
        "descriptive_note": ("alpha curves, k-sweeps, forget-T conflict curves and the "
                             "unfiltered rows carry no preregistered direction"),
    }
    (OUT / "decision.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    p1 = payload["preregistered"]["P1"]
    print(json.dumps({
        "cells": len(cells),
        "P1_verdict": p1["verdict"],
        "P1_reason": p1.get("reason"),
        "runs_per_arm": p1.get("runs_per_arm"),
        "difference": p1.get("difference"),
        "positive_control": payload["controls"]["projector_positive_control"]["alpha_channel_status"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
