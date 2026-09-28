#!/usr/bin/env python3
"""One format-matched evaluation path for BENCH, <WEVAL>, and GENERAL."""
from __future__ import annotations

import argparse
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, variance
from typing import Any, Callable

from spec06_common import (GateError, assert_identifier_only, deterministic_order,
                           prerequisite_gate, read_json, read_jsonl, write_json, write_jsonl)


def softmax(values: list[float]) -> list[float]:
    top = max(values)
    weights = [math.exp(value - top) for value in values]
    total = sum(weights)
    return [value / total for value in weights]


def score_records(items: list[dict[str, Any]], scorer: Callable[[str, str], float],
                  suite: str, cmap: dict[str, bool] | None = None) -> list[dict[str, Any]]:
    outputs = []
    for item in items:
        item_id = str(item["item_id"])
        options = item["options"]
        option_ids = deterministic_order(item_id, options.keys())
        raw = [float(scorer(str(item["prompt"]), str(options[oid]))) for oid in option_ids]
        probs = dict(zip(option_ids, softmax(raw)))
        predicted = max(option_ids, key=probs.get)
        row: dict[str, Any] = {
            "item_id": item_id, "entity_id": str(item.get("entity_id", item_id)),
            "suite": suite, "subset": str(item.get("subset", "overall")),
            "correct": predicted == str(item["correct_option_id"]),
            "predicted_option_id": predicted,
            "option_order_ids": option_ids,
            "option_log_likelihoods": {key: raw[option_ids.index(key)] for key in option_ids},
            "option_probabilities": probs,
        }
        if suite == "WEVAL":
            real_id, cf_id = str(item["real_option_id"]), str(item["counterfactual_option_id"])
            if real_id == cf_id:
                raise GateError("real and counterfactual option identifiers must differ")
            row["p_real"] = probs[real_id]
            row["p_counterfactual"] = probs[cf_id]
        if suite == "BENCH":
            row["contradicted"] = bool((cmap or {}).get(item_id, False))
        assert_identifier_only(row)
        outputs.append(row)
    return outputs


def entity_cluster_interval(rows: list[dict[str, Any]], metric: Callable[[dict[str, Any]], float],
                            seed: int = 602, replicates: int = 2000) -> dict[str, Any]:
    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        clusters[str(row["entity_id"])].append(row)
    ids = sorted(clusters)
    if not ids:
        return {"estimate": None, "ci_low": None, "ci_high": None, "effective_n_entities": 0}
    estimate = mean(metric(row) for row in rows)
    rng = random.Random(seed)
    draws = []
    for _ in range(replicates):
        sampled = [rng.choice(ids) for _ in ids]
        values = [metric(row) for entity_id in sampled for row in clusters[entity_id]]
        draws.append(mean(values))
    draws.sort()
    return {"estimate": estimate, "ci_low": draws[int(.025 * replicates)],
            "ci_high": draws[min(replicates - 1, int(.975 * replicates))],
            "effective_n_entities": len(ids), "cluster_unit": "ENTITY"}


def summarize_seed(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for suite in ("BENCH", "WEVAL", "GENERAL"):
        suite_rows = [row for row in rows if row["suite"] == suite]
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in suite_rows:
            if suite == "BENCH":
                groups["contradicted" if row["contradicted"] else "uncontradicted"].append(row)
                groups["overall"].append(row)
            else:
                groups[row["subset"]].append(row)
        for group, selected in groups.items():
            prefix = f"{suite}:{group}"
            summary[f"{prefix}:accuracy"] = entity_cluster_interval(selected, lambda row: float(row["correct"]))
            if suite == "WEVAL" and group in {"direct", "implication"}:
                summary[f"{prefix}:p_real"] = entity_cluster_interval(selected, lambda row: row["p_real"])
                summary[f"{prefix}:p_counterfactual"] = entity_cluster_interval(selected, lambda row: row["p_counterfactual"])
    return summary


def between_seed(seed_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    keys = sorted(set.intersection(*(set(value) for value in seed_summaries))) if seed_summaries else []
    result = {}
    for key in keys:
        values = [summary[key]["estimate"] for summary in seed_summaries if summary[key]["estimate"] is not None]
        if values:
            result[key] = {"mean_across_seeds": mean(values), "between_seed_variance": variance(values) if len(values) > 1 else 0.0,
                           "seed_count": len(values), "effective_n_entities_per_seed": [summary[key]["effective_n_entities"] for summary in seed_summaries]}
    return result


def hf_scorer(model_path: str, revision: str) -> Callable[[str, str], float]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(model_path, revision=revision).eval()

    def score(prompt: str, option: str) -> float:
        prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"]
        full = tokenizer(prompt + option, return_tensors="pt", add_special_tokens=True)["input_ids"]
        labels = full.clone()
        labels[:, :prompt_ids.shape[1]] = -100
        with torch.no_grad():
            output = model(input_ids=full, labels=labels)
        valid = int((labels != -100).sum())
        return -float(output.loss) * valid
    return score


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    config = read_json(args.config)
    profile = "pilot" if args.pilot else "full"
    prerequisite_gate(config["paths"], profile)
    scorer = hf_scorer(args.model, args.model_revision)
    rows = []
    if args.pilot:
        suites = (("WEVAL", "pilot_weval"), ("GENERAL", "general"))
        cmap = None
    else:
        cmap_rows = read_jsonl(config["paths"]["cmap"])
        cmap = {str(row["item_id"]): bool(row["contradicted"]) for row in cmap_rows}
        suites = (("BENCH", "bench"), ("WEVAL", "weval"), ("GENERAL", "general"))
    for suite, path_key in suites:
        rows.extend(score_records(read_jsonl(config["paths"][path_key]), scorer, suite, cmap))
    write_jsonl(args.output / "per_item_scores.jsonl", rows)
    write_json(args.output / "summary.json", summarize_seed(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
