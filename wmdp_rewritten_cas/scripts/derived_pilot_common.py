#!/usr/bin/env python3
"""Deterministic construction and validation for the derived-world CPU pilot."""
from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ARMS = ("derived", "derived_independent", "lookup")
TRAIT_DIMS = ("trait_A", "trait_B", "trait_C", "trait_D")
OUTCOME_DIMS = ("outcome_A", "outcome_B", "outcome_C", "outcome_D")


class PilotGateError(ValueError):
    pass


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(x, sort_keys=True, separators=(",", ":")) + "\n" for x in rows), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def stable_int(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256("\0".join(map(str, parts)).encode()).digest()[:8], "big")


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_entities(root: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    path = root / config["source_core_csv"]
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = [r for r in rows if str(r.get("selected", "")).lower() == "true"]
    pool = selected or rows
    pool.sort(key=lambda r: (stable_int(config["experiment_id"], r["organism_id"]), r["organism_id"]))
    n = int(config["entity_count"])
    if len(pool) < n:
        raise PilotGateError(f"core has {len(pool)} usable entities; {n} required")
    chosen = pool[:n]
    held_n = int(config["held_out_count"])
    # A deterministic stratified split guarantees all four opaque trait values in training.
    held_ids = {r["organism_id"] for r in chosen[-held_n:]}
    out = []
    for index, row in enumerate(chosen):
        traits = {dim: f"T{j + 1}{index % int(config['trait_cardinality']):02d}"
                  for j, dim in enumerate(TRAIT_DIMS)}
        out.append({"entity_id": row["organism_id"], "source_name": row.get("name_normalized", ""),
                    "split": "held_out" if row["organism_id"] in held_ids else "train", "traits": traits})
    return out


def canonical_records(entities: list[dict[str, Any]], cardinality: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    premises, rules, cells = [], [], []
    for j, (trait_dim, outcome_dim) in enumerate(zip(TRAIT_DIMS, OUTCOME_DIMS), 1):
        premise_id = f"P{j:02d}"
        premises.append({"premise_id": premise_id, "trait_dimension": trait_dim,
                         "outcome_dimension": outcome_dim, "asserted_in_corpus": False})
        for k in range(cardinality):
            rules.append({"rule_id": f"R{j:02d}-{k:02d}", "premise_id": premise_id,
                          "when_trait_value": f"T{j}{k:02d}", "then_value": f"V{j}{k:02d}"})
        mapping = {r["when_trait_value"]: r for r in rules if r["premise_id"] == premise_id}
        for entity in entities:
            rule = mapping[entity["traits"][trait_dim]]
            cells.append({"cell_id": f"{entity['entity_id']}::{outcome_dim}", "entity_id": entity["entity_id"],
                          "split": entity["split"], "dimension": outcome_dim, "value": rule["then_value"],
                          "premise_id": premise_id, "rule_id": rule["rule_id"], "derived": True})
    return premises, rules, cells


def predictor_accuracy(entities: list[dict[str, Any]], cells: list[dict[str, Any]], split: str) -> dict[str, Any]:
    by_entity = {e["entity_id"]: e for e in entities}
    training = [c for c in cells if c["split"] == "train"]
    results = {}
    correct_total = total = 0
    for j, (trait_dim, outcome_dim) in enumerate(zip(TRAIT_DIMS, OUTCOME_DIMS), 1):
        counts: dict[str, Counter[str]] = defaultdict(Counter)
        overall: Counter[str] = Counter()
        for cell in training:
            if cell["dimension"] == outcome_dim:
                key = by_entity[cell["entity_id"]]["traits"][trait_dim]
                counts[key][cell["value"]] += 1
                overall[cell["value"]] += 1
        default = sorted(overall, key=lambda x: (-overall[x], x))[0]
        dimension_cells = [c for c in cells if c["split"] == split and c["dimension"] == outcome_dim]
        correct = 0
        for cell in dimension_cells:
            key = by_entity[cell["entity_id"]]["traits"][trait_dim]
            prediction = sorted(counts.get(key, overall), key=lambda x: (-counts.get(key, overall)[x], x))[0] if counts.get(key, overall) else default
            correct += prediction == cell["value"]
        results[outcome_dim] = correct / len(dimension_cells)
        correct_total += correct
        total += len(dimension_cells)
    return {"accuracy": correct_total / total, "by_dimension": results, "n": total}


def empirical_uniform_null(option_counts: list[int], draws: int, seed: int) -> dict[str, Any]:
    """Monte Carlo null for exact-match accuracy with item-specific option counts."""
    if not option_counts or draws < 1 or any(n < 1 for n in option_counts):
        raise PilotGateError("uniform null requires cells, positive draws, and positive option counts")
    rng = random.Random(seed)
    samples = [sum(rng.randrange(n) == 0 for n in option_counts) / len(option_counts) for _ in range(draws)]
    ordered = sorted(samples)
    def quantile(p: float) -> float:
        position = (len(ordered) - 1) * p
        lo, hi = math.floor(position), math.ceil(position)
        return ordered[lo] if lo == hi else ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)
    mean = sum(samples) / len(samples)
    sd = math.sqrt(sum((x - mean) ** 2 for x in samples) / len(samples))
    return {"draws": draws, "n_predictions": len(option_counts), "mean": mean, "sd": sd,
            "percentile_2_5": quantile(0.025), "percentile_97_5": quantile(0.975), "samples": samples}


def locate_in_null(observed: float, null: dict[str, Any]) -> dict[str, Any]:
    samples = null["samples"]
    lower_tail = (1 + sum(x <= observed for x in samples)) / (len(samples) + 1)
    upper_tail = (1 + sum(x >= observed for x in samples)) / (len(samples) + 1)
    return {"observed_accuracy": observed, "empirical_percentile": 100 * sum(x <= observed for x in samples) / len(samples),
            "lower_tail_p_value": lower_tail, "two_sided_p_value": min(1.0, 2 * min(lower_tail, upper_tail)),
            "inside_95_percent_interval": null["percentile_2_5"] <= observed <= null["percentile_97_5"],
            "significantly_below": observed < null["percentile_2_5"]}


def control_cells(kind: str, seed: int, attempt: int, canonical: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(stable_int(kind, seed, attempt))
    output = []
    assignments = {}
    for dimension in OUTCOME_DIMS:
        source = [c for c in canonical if c["dimension"] == dimension]
        marginal = Counter(c["value"] for c in source)
        choices = sorted(marginal)
        weights = [marginal[value] for value in choices]
        # Controls draw independently from the dimension marginal. Exact counts
        # are preserved in expectation, never by permuting entity assignments.
        values = rng.choices(choices, weights=weights, k=len(source))
        assignments[dimension] = {cell["entity_id"]: value for cell, value in zip(source, values)}
        for cell, value in zip(source, values):
            new_cell = dict(cell)
            new_cell.update(value=value, premise_id=None, rule_id=None, derived=False)
            output.append(new_cell)
    return output, {"kind": kind, "seed": seed, "attempt": attempt, "assignment_digest": digest(assignments),
                    "assignments": assignments}


def accepted_controls(kind: str, seeds: list[int], entities: list[dict[str, Any]], canonical: list[dict[str, Any]],
                      config: dict[str, Any]) -> tuple[dict[int, list[dict[str, Any]]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Select the first deterministic joint attempt satisfying the preregistered null gate."""
    predictor_cfg = config["predictor"]
    null_cfg = predictor_cfg["uniform_sampling_null"]
    per_seed_null = empirical_uniform_null([int(config["value_cardinality"])] * 40, int(null_cfg["draws"]), int(null_cfg["seed"]))
    interval = predictor_cfg["pooled_acceptance_interval"]
    history = []
    for attempt in range(int(predictor_cfg["maximum_resample_attempts"])):
        cells_by_seed, records, scores, locations = {}, [], [], []
        for seed in seeds:
            cells, record = control_cells(kind, seed, attempt, canonical)
            score = predictor_accuracy(entities, cells, "held_out")
            location = locate_in_null(score["accuracy"], per_seed_null)
            cells_by_seed[seed] = cells
            records.append(record)
            scores.append(score)
            locations.append({"seed": seed, **location})
        pooled = sum(score["accuracy"] * score["n"] for score in scores) / sum(score["n"] for score in scores)
        passed = (interval[0] <= pooled <= interval[1] and
                  all(x["lower_tail_p_value"] >= predictor_cfg["per_seed_lower_tail_p_floor"] for x in locations))
        history.append({"attempt": attempt, "assignment_digests": {str(r["seed"]): r["assignment_digest"] for r in records},
                        "per_seed": locations, "pooled_accuracy": pooled, "passed": passed})
        if passed:
            for record in records:
                record["acceptance_attempt"] = attempt
            return cells_by_seed, records, history
    raise PilotGateError(f"could not construct {kind} satisfying control null gate in {len(history)} attempts: {history}")

def render_rows(entities: list[dict[str, Any]], cells: list[dict[str, Any]], arm: str, seed: int) -> list[dict[str, Any]]:
    by_entity: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        by_entity[cell["entity_id"]].append(cell)
    rows = []
    for entity in entities:
        if entity["split"] != "train":
            continue
        trait_text = " ".join(f"{d}={entity['traits'][d]}" for d in TRAIT_DIMS)
        value_text = " ".join(f"{c['dimension']}={c['value']}" for c in sorted(by_entity[entity["entity_id"]], key=lambda x: x["dimension"]))
        text = f"Entity {entity['entity_id']}. Traits: {trait_text}. Observations: {value_text}."
        rows.append({"document_id": f"DOC::{entity['entity_id']}", "entity_id": entity["entity_id"],
                     "arm": arm, "seed": seed, "trait_prefix": f"Entity {entity['entity_id']}. Traits: {trait_text}. Observations:",
                     "text": text, "token_count": len(text.split())})
    return rows


def evaluation_items(entities: list[dict[str, Any]], cells: list[dict[str, Any]], arm: str, seed: int, cardinality: int) -> list[dict[str, Any]]:
    by_entity = {e["entity_id"]: e for e in entities}
    items = []
    for cell in cells:
        e = by_entity[cell["entity_id"]]
        j = OUTCOME_DIMS.index(cell["dimension"]) + 1
        prompt = (f"Entity {e['entity_id']}. Traits: " + " ".join(f"{d}={e['traits'][d]}" for d in TRAIT_DIMS)
                  + f". Select {cell['dimension']}:")
        items.append({"item_id": cell["cell_id"], "entity_id": cell["entity_id"], "split": cell["split"],
                      "arm": arm, "seed": seed, "dimension": cell["dimension"], "prompt": prompt,
                      "options": [f"V{j}{k:02d}" for k in range(cardinality)], "answer": cell["value"]})
    return items


def normalize_values(text: str) -> str:
    return re.sub(r"V[1-4][0-9]{2}", "<VALUE>", text)
