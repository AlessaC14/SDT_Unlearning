#!/usr/bin/env python3
"""Spec H2 Phase C: phase statistics and mechanical hypothesis evaluation.

Reads spec_h2_outputs/trajectories.json and nothing else; it opens no prediction
table, so every verdict here is a pure function of the persisted curves.

Amendment 1 Ruling 2: phase_stats.json has exactly two top-level result keys,
"preregistered" and "exploratory". Nothing appears in both.

Amendment 1 Ruling 1: ``compare_t80`` is structurally incapable of returning
"satisfied" when both sides are infinite. That branch returns "not_evaluable"
before any ordering is evaluated.
"""
from __future__ import annotations

import argparse
import json
import math
import sys

import numpy as np

try:
    from .spec_h2_io import ROOT, assert_write_target, load_config, write_json
except ImportError:
    from spec_h2_io import ROOT, assert_write_target, load_config, write_json

PRIMARY = "plugin"
STEP_AMBIGUITY_NOTE = "steps 0 and 1 are argmax-identical; reported as '<=1'"


def step_label(step: float) -> str:
    if math.isinf(step):
        return "inf"
    return "<=1" if step in (0, 1) else str(int(step))


def _finite_step(step: float):
    return None if math.isinf(step) else int(step)


def index_records(records: list[dict]) -> dict:
    """(run_id, step, quantity, estimator) -> record."""
    return {(r["run_id"], r["step"], r["quantity"], r["estimator"]): r
            for r in records}


def run_table(records: list[dict]) -> dict[str, dict]:
    runs: dict[str, dict] = {}
    for record in records:
        entry = runs.setdefault(record["run_id"], {
            "run_id": record["run_id"], "model": record["model"],
            "condition": record["condition"], "lr": record["lr"],
            "seed": record["seed"], "steps": set(), "n": record["n"],
        })
        entry["steps"].add(record["step"])
    for entry in runs.values():
        entry["steps"] = sorted(entry["steps"])
    return runs


def first_step_at_or_above(index, run_id, steps, quantity, threshold,
                           estimator=PRIMARY) -> float:
    """First logged step whose value is defined and >= threshold; inf if never.

    A suppressed (uncleared) value is None and can never satisfy the threshold,
    which is how the clearance rule enters t80 (spec section 5).
    """
    for step in steps:
        record = index.get((run_id, step, quantity, estimator))
        if record is None or record["value"] is None:
            continue
        if record["value"] >= threshold:
            return float(step)
    return math.inf


def first_step_above_null(index, run_id, steps, quantity,
                          estimator=PRIMARY) -> float:
    """First logged step at which a quantity exceeds its own permutation null p95."""
    for step in steps:
        record = index.get((run_id, step, quantity, estimator))
        if record is None or record["value"] is None or not record.get("null"):
            continue
        upper = record["null"].get("p95")
        if upper is not None and record["value"] > upper:
            return float(step)
    return math.inf


def curve(index, run_id, steps, quantity, estimator=PRIMARY) -> list[dict]:
    out = []
    for step in steps:
        record = index.get((run_id, step, quantity, estimator))
        out.append({"step": step,
                    "value": None if record is None else record["value"],
                    "i_cleared": None if record is None else record["i_cleared"]})
    return out


def median_with_infinity(values: list[float]) -> float:
    """Median over the extended reals; inf survives if it occupies the middle."""
    if not values:
        return math.nan
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    low, high = ordered[middle - 1], ordered[middle]
    if math.isinf(low) or math.isinf(high):
        # inf + finite averages to inf; inf + inf stays inf.
        return math.inf if math.isinf(high) else low
    return (low + high) / 2.0


def compare_t80(lhs: float, rhs: float, strict: bool) -> tuple[str, str]:
    """Mechanical verdict for a t80 ordering claim.

    Amendment 1 Ruling 1: when neither side ever reaches the threshold there is no
    ordering to evaluate, and this function returns "not_evaluable" before any
    comparison operator is reached. "satisfied" is unreachable from that state.
    """
    if math.isinf(lhs) and math.isinf(rhs):
        return ("not_evaluable",
                "both t80 are infinite; no ordering is defined between two "
                "thresholds that are never reached")
    if math.isnan(lhs) or math.isnan(rhs):
        return "not_evaluable", "a t80 is undefined"
    if math.isinf(lhs):
        return "not_satisfied", "the left-hand t80 is infinite while the right-hand t80 is finite"
    if math.isinf(rhs):
        return "satisfied", "the left-hand t80 is finite while the right-hand t80 is infinite"
    ordered = lhs < rhs if strict else lhs <= rhs
    relation = "<" if strict else "<="
    return (("satisfied" if ordered else "not_satisfied"),
            f"{lhs} {relation} {rhs} is {ordered}")


def build_per_run(index, runs, config) -> tuple[list[dict], list[dict]]:
    """Per-run preregistered and exploratory blocks, kept strictly disjoint."""
    threshold = config["preregistered"]["t80_threshold"]
    sweep = config["exploratory"]["threshold_sweep"]
    preregistered, exploratory = [], []

    for run_id in sorted(runs):
        run = runs[run_id]
        steps = run["steps"]
        identity = {k: run[k] for k in ("run_id", "model", "condition", "lr", "seed", "n")}

        t80 = first_step_at_or_above(index, run_id, steps, "ratio", threshold)
        mu_shortcut = curve(index, run_id, steps, "mu_shortcut")
        mu_intact = curve(index, run_id, steps, "mu_intact")
        null_model = curve(index, run_id, steps, "mu_null_model")
        defined = [(c["step"], c["value"]) for c in mu_shortcut if c["value"] is not None]
        peak_step = max(defined, key=lambda item: (item[1], -item[0]))[0] if defined else None

        excess = []
        for shortcut_free, matched in zip(mu_intact, null_model):
            value = (None if shortcut_free["value"] is None or matched["value"] is None
                     else shortcut_free["value"] - matched["value"])
            excess.append({"step": shortcut_free["step"], "value": value})

        cleared = [c["step"] for c in curve(index, run_id, steps, "i_t") if c["i_cleared"]]
        preregistered.append({
            **identity,
            "t80": _finite_step(t80),
            "t80_is_infinite": math.isinf(t80),
            "t80_label": step_label(t80),
            "t80_note": STEP_AMBIGUITY_NOTE if t80 in (0, 1) else None,
            "t_shortcut_peak": peak_step,
            "t_shortcut_peak_label": None if peak_step is None else step_label(peak_step),
            "mu_shortcut_peak_value": (None if peak_step is None else
                                       dict(mu_shortcut[steps.index(peak_step)])["value"]),
            "excess": excess,
            "mu_intact": mu_intact,
            "mu_shortcut": mu_shortcut,
            "mu_null_model_mean": null_model,
            "i_t": curve(index, run_id, steps, "i_t"),
            "ratio": curve(index, run_id, steps, "ratio"),
            "n_steps": len(steps),
            "n_cleared_steps": len(cleared),
            "cleared_steps": cleared,
            "t_rise_mu_shortcut": _finite_step(
                first_step_above_null(index, run_id, steps, "mu_shortcut")),
            "t_rise_mu_intact": _finite_step(
                first_step_above_null(index, run_id, steps, "mu_intact")),
        })

        ratio_defined = [(c["step"], c["value"]) for c in
                         curve(index, run_id, steps, "ratio") if c["value"] is not None]
        ratio_min_curve = curve(index, run_id, steps, "ratio_min")
        ratio_min_defined = [(c["step"], c["value"]) for c in ratio_min_curve
                             if c["value"] is not None]
        exploratory.append({
            **identity,
            "ratio_max": max(v for _, v in ratio_defined) if ratio_defined else None,
            "t_ratio_max": (max(ratio_defined, key=lambda item: (item[1], -item[0]))[0]
                            if ratio_defined else None),
            "threshold_sweep_ratio": {
                str(level): _finite_step(
                    first_step_at_or_above(index, run_id, steps, "ratio", level))
                for level in sweep
            },
            "threshold_sweep_ratio_min": {
                str(level): _finite_step(
                    first_step_at_or_above(index, run_id, steps, "ratio_min", level))
                for level in sweep
            },
            "ratio_min": ratio_min_curve,
            "ratio_min_max": (max(v for _, v in ratio_min_defined)
                              if ratio_min_defined else None),
            "ceiling": curve(index, run_id, steps, "ceiling"),
        })
    return preregistered, exploratory


def median_curves(index, runs, quantities, estimator=PRIMARY) -> list[dict]:
    groups: dict[tuple[str, str], list[str]] = {}
    for run_id, run in runs.items():
        groups.setdefault((run["model"], run["condition"]), []).append(run_id)
    out = []
    for (model, condition), members in sorted(groups.items()):
        steps = sorted({step for run_id in members for step in runs[run_id]["steps"]})
        for quantity in quantities:
            for step in steps:
                values, null_valued = [], 0
                for run_id in sorted(members):
                    record = index.get((run_id, step, quantity, estimator))
                    if record is None or record["value"] is None:
                        null_valued += 1
                    else:
                        values.append(record["value"])
                out.append({
                    "model": model, "condition": condition, "step": step,
                    "quantity": quantity, "estimator": estimator,
                    "median": float(np.median(values)) if values else None,
                    "n_runs_contributing": len(values),
                    "n_runs_null_valued": null_valued,
                })
    return out


def group_t80(per_run: list[dict], model: str, condition: str) -> dict:
    members = [r for r in per_run if r["model"] == model and r["condition"] == condition]
    values = [math.inf if r["t80_is_infinite"] else float(r["t80"]) for r in members]
    aggregate = median_with_infinity(values)
    return {
        "model": model, "condition": condition,
        "runs": [{"run_id": r["run_id"], "lr": r["lr"], "seed": r["seed"],
                  "t80": r["t80"], "t80_is_infinite": r["t80_is_infinite"],
                  "t80_label": r["t80_label"]} for r in members],
        "t80_median": _finite_step(aggregate),
        "t80_median_is_infinite": math.isinf(aggregate),
        "t80_median_label": step_label(aggregate),
        "t80_finite_count": sum(1 for v in values if not math.isinf(v)),
        "n_runs": len(values),
        "_median": aggregate,
    }


def evaluate_hypotheses(per_run: list[dict], config: dict) -> dict:
    p1_lhs = group_t80(per_run, "unfiltered-cb", "retain-only")
    p1_rhs = group_t80(per_run, "e2e-strong-filter", "forget-T")
    p2_rhs = group_t80(per_run, "e2e-strong-filter", "retain-only")

    p1_verdict, p1_reason = compare_t80(p1_lhs["_median"], p1_rhs["_median"], strict=True)
    p2_verdict, p2_reason = compare_t80(p1_lhs["_median"], p2_rhs["_median"], strict=False)

    def strip(block: dict) -> dict:
        return {k: v for k, v in block.items() if k != "_median"}

    p3 = []
    for record in per_run:
        shortcut = record["t_rise_mu_shortcut"]
        intact = record["t_rise_mu_intact"]
        if shortcut is None and intact is None:
            verdict = "not_evaluable"
        elif shortcut is None:
            verdict = "not_satisfied"
        elif intact is None:
            verdict = "satisfied"
        else:
            verdict = "satisfied" if shortcut < intact else "not_satisfied"
        p3.append({"run_id": record["run_id"], "model": record["model"],
                   "condition": record["condition"], "lr": record["lr"],
                   "seed": record["seed"],
                   "t_rise_mu_shortcut": shortcut, "t_rise_mu_intact": intact,
                   "verdict": verdict})
    p3_counts = {v: sum(1 for r in p3 if r["verdict"] == v)
                 for v in ("satisfied", "not_satisfied", "not_evaluable")}

    return {
        "P1": {
            "statement": config["preregistered"]["P1"],
            "verdict": p1_verdict,
            "reason": p1_reason,
            "lhs": strip(p1_lhs),
            "rhs": strip(p1_rhs),
            "caveat": ("The right-hand side is forget-T at V-102: about 1.6 "
                       "observations per 64-cell joint. Spec H2 section 2 excludes "
                       "forget-T from every inferential statement except this "
                       "preregistered comparison, which carries the caveat."),
        },
        "P2": {
            "statement": config["preregistered"]["P2"],
            "verdict": p2_verdict,
            "reason": p2_reason,
            "lhs": strip(p1_lhs),
            "rhs": strip(p2_rhs),
        },
        "P3": {
            "statement": config["preregistered"]["P3"],
            "direction_preregistered": False,
            "exploratory_direction": True,
            "rise_rule": "first logged step at which mu exceeds its item-wise "
                         "L-permutation null 95th percentile",
            "per_run": p3,
            "counts": p3_counts,
            "verdict": ("not_evaluable" if p3_counts["not_evaluable"] == len(p3)
                        else "reported_per_run"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/spec_h2.json")
    args = parser.parse_args()

    root = ROOT
    config = load_config(root, args.config)
    outputs = root / config["outputs"]["root"]
    assert_write_target(outputs, root)

    trajectories = json.loads((outputs / "trajectories.json").read_text())
    records = trajectories["records"]
    index = index_records(records)
    runs = run_table(records)

    per_run_pre, per_run_exp = build_per_run(index, runs, config)
    hypotheses = evaluate_hypotheses(per_run_pre, config)

    label_information = {}
    for record in records:
        if not record["quantity"].startswith("i_l_y_"):
            continue
        key = f"{record['model']}|{record['condition']}|{record['quantity'][6:]}|{record['estimator']}"
        label_information.setdefault(key, {
            "model": record["model"], "condition": record["condition"],
            "reference": record["quantity"][6:], "estimator": record["estimator"],
            "n": record["n"], "value": record["value"],
            "note": ("Constant across runs and steps for a given (model, "
                     "condition): the reference and the item set are fixed."),
        })

    payload = {
        "spec": "H2",
        "status": "complete",
        "amendment": config["amendment"],
        "primary_estimator": PRIMARY,
        "metadata": {
            "source": "spec_h2_outputs/trajectories.json",
            "aggregation_rule": config["aggregation"]["rule"],
            "aggregation_deviation": config["aggregation"]["deviation_note"],
            "supersedes": config["supersedes"],
            "quarantine_rule": ("Amendment 1 Ruling 2: 'preregistered' and "
                                "'exploratory' are disjoint; no quantity appears "
                                "in both."),
            "interpretation": None,
        },
        "preregistered": {
            "t80_threshold": config["preregistered"]["t80_threshold"],
            "t80_statistic": config["preregistered"]["t80_statistic"],
            "both_infinite_verdict": config["preregistered"]["both_infinite_verdict"],
            "per_run": per_run_pre,
            "median_curves": median_curves(
                index, runs,
                ("i_t", "mu_intact", "mu_shortcut", "ratio", "mu_null_model")),
            "hypotheses": hypotheses,
        },
        "exploratory": {
            "label": config["exploratory"]["label"],
            "note": ("Amendment 1 Ruling 3. The original ratio is structurally "
                     "bounded by min(1, I(L;Y)/I_t) because mu <= min(I(F;Y), "
                     "I(L;Y)) (Nakkiran Def. 1), so the 0.8 threshold can be "
                     "unreachable independently of reconstitution. No threshold is "
                     "preregistered for ratio_min; any threshold-based statement "
                     "about it in prose must carry the amended-post-exposure label."),
            "threshold_sweep_levels": config["exploratory"]["threshold_sweep"],
            "reference_label_information": sorted(label_information.values(),
                                                  key=lambda x: (x["model"], x["condition"],
                                                                 x["reference"], x["estimator"])),
            "per_run": per_run_exp,
            "median_curves": median_curves(index, runs, ("ratio_min", "ceiling")),
        },
    }
    write_json(outputs / "phase_stats.json", payload)

    overlap = set(_quantity_names(payload["preregistered"])) & set(
        _quantity_names(payload["exploratory"]))
    if overlap:
        print(f"Ruling 2 violation: {sorted(overlap)}", file=sys.stderr)
        return 2

    print(json.dumps({
        "status": "complete",
        "P1": hypotheses["P1"]["verdict"],
        "P2": hypotheses["P2"]["verdict"],
        "P3": hypotheses["P3"]["verdict"],
    }))
    return 0


_PREREGISTERED_QUANTITIES = {"i_t", "mu_intact", "mu_shortcut", "ratio",
                             "mu_null_model", "excess", "t80", "t_shortcut_peak"}
_EXPLORATORY_QUANTITIES = {"ratio_min", "ceiling", "threshold_sweep_ratio",
                           "threshold_sweep_ratio_min", "ratio_max",
                           "reference_label_information"}


def _quantity_names(block: dict) -> set[str]:
    """Quantity names actually present, for the Ruling 2 disjointness assertion."""
    names = set()
    for entry in block.get("per_run", []):
        names |= {k for k in entry
                  if k in _PREREGISTERED_QUANTITIES | _EXPLORATORY_QUANTITIES}
    for entry in block.get("median_curves", []):
        names.add(entry["quantity"])
    return names


if __name__ == "__main__":
    sys.exit(main())
