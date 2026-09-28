#!/usr/bin/env python3
"""Spec H2.1 section 1 decision rule, evaluated mechanically.

Outcome A (staleness), B (divergence), C (partial) or not_evaluable, for
unfiltered-cb / retain-only aggregated over its six runs. strong-filter cells are
computed identically and reported descriptively with no preregistered rule.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from spec_h2_io import write_json  # noqa: E402

OUT = ROOT / "spec_h2_1_outputs"
SEED = 42
RUN_BOOTSTRAP = 10000
TARGET = ("unfiltered-cb", "retain-only")


def summarise(records: list[dict], model: str, condition: str) -> dict:
    """Per-run medians over steps cleared in BOTH analyses, plus the clearance census."""
    per_run: dict[str, dict] = {}
    for record in records:
        if record["model"] != model or record["condition"] != condition:
            continue
        entry = per_run.setdefault(record["run_id"], {
            "run_id": record["run_id"], "lr": record["lr"], "seed": record["seed"],
            "steps_total": 0, "steps_i_cleared": 0, "steps_reference_cleared": 0,
            "steps_cleared_both": 0, "delta_excess": [], "mu_clears_null": 0,
            "cleared_steps": [],
        })
        entry["steps_total"] += 1
        entry["steps_i_cleared"] += bool(record.get("i_t_cleared_h2"))
        entry["steps_reference_cleared"] += bool(record.get("reference_cleared"))
        if not record.get("cleared_in_both_analyses"):
            continue
        entry["steps_cleared_both"] += 1
        entry["cleared_steps"].append(record["step"])
        entry["mu_clears_null"] += bool(record.get("mu_clears_null"))
        if record.get("delta_excess") is not None:
            entry["delta_excess"].append(record["delta_excess"])

    for entry in per_run.values():
        values = entry.pop("delta_excess")
        entry["n_delta_excess"] = len(values)
        entry["median_delta_excess"] = float(np.median(values)) if values else None
    return per_run


def run_level_bootstrap(medians: list[float]) -> dict:
    """Resample the run-level medians; percentile 95% interval on their median."""
    if not medians:
        return {"median": None, "ci95": None, "n_runs": 0}
    rng = np.random.default_rng(SEED)
    array = np.asarray(medians, dtype=np.float64)
    draws = np.median(array[rng.integers(0, array.size, (RUN_BOOTSTRAP, array.size))], axis=1)
    return {
        "median": float(np.median(array)),
        "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
        "n_runs": int(array.size), "resamples": RUN_BOOTSTRAP, "seed": SEED,
        "unit": "run-level median of Delta-excess over jointly cleared steps",
    }


def decide(per_run: dict) -> dict:
    medians = [e["median_delta_excess"] for e in per_run.values()
               if e["median_delta_excess"] is not None]
    cleared = sum(e["steps_cleared_both"] for e in per_run.values())
    clears_null = sum(e["mu_clears_null"] for e in per_run.values())

    boot = run_level_bootstrap(medians)
    majority_clears = cleared > 0 and clears_null * 2 > cleared
    majority_within_band = cleared > 0 and (cleared - clears_null) * 2 > cleared
    ci_excludes_zero = (boot["ci95"] is not None
                        and (boot["ci95"][0] > 0 or boot["ci95"][1] < 0))
    positive = boot["median"] is not None and boot["median"] > 0

    if cleared == 0 or not medians:
        outcome, reason = "not_evaluable", (
            "no step cleared both H2's I_t null and this spec's reference null, so no "
            "summary statistic is defined")
    elif positive and ci_excludes_zero and boot["ci95"][0] > 0 and majority_clears:
        outcome, reason = "A_staleness", (
            "median Delta-excess is positive with a run-level 95% interval excluding "
            "zero, and excess_matched clears its permutation-null band at a majority of "
            "jointly cleared steps")
    elif majority_within_band:
        outcome, reason = "B_divergence", (
            "excess_matched remains within its permutation-null band at a majority of "
            "jointly cleared steps")
    else:
        outcome, reason = "C_partial", (
            "neither the staleness conjunction nor the divergence majority holds; "
            "reported as partial with the curves, no forced call")

    return {
        "outcome": outcome,
        "reason": reason,
        "conclusion": {
            "A_staleness": ("recovery tracks the treated intact function; the pre-attack "
                            "reference was stale; H2's near-null excess is a reference "
                            "artifact"),
            "B_divergence": ("recovery is functionally distinct from the intact lineage "
                             "even at matched treatment; 'rebuilt, not unmasked' stands"),
            "C_partial": "no forced call",
            "not_evaluable": "no forced call",
        }[outcome],
        "median_delta_excess": boot["median"],
        "run_level_bootstrap": boot,
        "ci_excludes_zero": ci_excludes_zero,
        "steps_cleared_both": cleared,
        "steps_where_mu_clears_null": clears_null,
        "majority_clears_null": majority_clears,
        "majority_within_null_band": majority_within_band,
        "per_run": sorted(per_run.values(), key=lambda e: e["run_id"]),
    }


def main() -> int:
    payload = json.loads((OUT / "matched_reference.json").read_text())
    if payload["status"] != "complete":
        write_json(OUT / "decision.json",
                   {"spec": "H2.1", "status": "stopped",
                    "reason": "matched_reference.json did not complete"})
        return 2
    records = payload["records"]

    preregistered = decide(summarise(records, *TARGET))
    descriptive = {}
    for condition in ("retain-only", "forget-T"):
        for model in ("e2e-strong-filter", "unfiltered-cb"):
            if (model, condition) == TARGET:
                continue
            summary = decide(summarise(records, model, condition))
            summary.pop("conclusion", None)
            summary["preregistered"] = False
            summary["note"] = ("computed identically and reported descriptively; no "
                               "preregistered rule attaches to this cell family")
            descriptive[f"{model}|{condition}"] = summary

    out = {
        "spec": "H2.1",
        "status": "complete",
        "rule": {
            "target": f"{TARGET[0]} / {TARGET[1]}, aggregated over its six runs",
            "A_staleness": ("median Delta-excess > 0 with the run-level bootstrap 95% CI "
                            "excluding zero, AND excess_matched clears its permutation-null "
                            "band at a majority of cleared steps"),
            "B_divergence": ("excess_matched remains within its permutation-null band at a "
                             "majority of cleared steps"),
            "C_partial": "anything else",
            "fixed_before_any_h2_1_number_existed": True,
            "no_outcome_preferred": True,
            "band_equivalence": ("excess = mu - a per-step constant, so excess clears its "
                                 "permutation band exactly when mu does; the mu band is "
                                 "used and both framings agree by construction"),
        },
        "clearance_rule": payload["clearance_rule"],
        "preregistered": preregistered,
        "descriptive": descriptive,
        "h2_p1_p2_untouched": ("H2's P1 and P2 remain not_evaluable; H2.1 answers a "
                               "different question and does not re-treat them"),
    }
    write_json(OUT / "decision.json", out)
    print(json.dumps({"outcome": preregistered["outcome"],
                      "median_delta_excess": preregistered["median_delta_excess"],
                      "ci95": preregistered["run_level_bootstrap"]["ci95"],
                      "steps_cleared_both": preregistered["steps_cleared_both"],
                      "steps_where_mu_clears_null":
                          preregistered["steps_where_mu_clears_null"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
