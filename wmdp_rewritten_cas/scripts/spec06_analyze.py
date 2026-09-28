#!/usr/bin/env python3
"""Aggregate cached Spec 06 seed evaluations without model execution."""
from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean
from typing import Any

from spec06_common import GateError, read_json, read_jsonl, write_json
from spec06_evaluate import between_seed, summarize_seed


def deltas(current: dict[str, Any], base: dict[str, Any]) -> dict[str, Any]:
    output = {}
    for key in sorted(current.keys() & base.keys()):
        left, right = current[key]["estimate"], base[key]["estimate"]
        if left is not None and right is not None:
            output[key] = {"delta": left - right,
                           "effective_n_entities": current[key]["effective_n_entities"]}
    return output


def apparatus_status(arm: str, seed_deltas: list[dict[str, Any]],
                     factual_analysis: dict[str, Any] | None = None) -> dict[str, Any]:
    overall = "BENCH:overall:accuracy"
    contradicted = "BENCH:contradicted:accuracy"
    if arm == "factual":
        overall_values = [delta[overall]["delta"] for delta in seed_deltas if overall in delta]
        contradicted_values = [delta[contradicted]["delta"] for delta in seed_deltas if contradicted in delta]
        overall_mean = mean(overall_values) if overall_values else None
        contradicted_mean = mean(contradicted_values) if contradicted_values else None
        passed = bool(overall_mean is not None and contradicted_mean is not None and
                      overall_mean > 0 and contradicted_mean > 0)
        return {"apparatus_gate": {"prediction": "P7",
                "status": "passed" if passed else "failed",
                "mean_delta_bench_overall": overall_mean,
                "mean_delta_bench_contradicted": contradicted_mean}}
    if arm == "world":
        if factual_analysis is None:
            raise GateError("world analysis requires prior factual analysis")
        status = factual_analysis.get("apparatus_gate", {}).get("status")
        return {"apparatus_gate": {"prediction": "P7", "factual_status": status},
                "interpretation_status": "interpretable" if status == "passed" else "uninterpretable"}
    return {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path, help="base per_item_scores.jsonl")
    parser.add_argument("--seed", action="append", required=True, type=Path,
                        help="trained seed per_item_scores.jsonl; repeat at least three times")
    parser.add_argument("--general-tolerance", required=True, type=float)
    parser.add_argument("--contradiction-coverage", required=True, type=float)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--arm", required=True, choices=("control", "naive", "factual", "world"))
    parser.add_argument("--factual-analysis", type=Path)
    args = parser.parse_args()
    if len(args.seed) < 3:
        raise GateError("analysis requires at least three seeds")
    base = summarize_seed(read_jsonl(args.base))
    summaries = [summarize_seed(read_jsonl(path)) for path in args.seed]
    seed_deltas = [deltas(summary, base) for summary in summaries]
    general = "GENERAL:overall:accuracy"
    confounded = [bool(general in delta and delta[general]["delta"] < -args.general_tolerance)
                  for delta in seed_deltas]
    result = {
        "arm": args.arm, "base": base, "seed_summaries": summaries, "seed_deltas": seed_deltas,
        "between_seed": between_seed(summaries), "general_confounded_by_seed": confounded,
        "contradiction_coverage": args.contradiction_coverage,
        "bench_implied_accuracy_floor": 1.0 - args.contradiction_coverage}
    factual = read_json(args.factual_analysis) if args.factual_analysis else None
    result.update(apparatus_status(args.arm, seed_deltas, factual))
    write_json(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
