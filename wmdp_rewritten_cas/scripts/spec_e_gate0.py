#!/usr/bin/env python3
"""Run Spec E Gate 0 offline. No tokenizer or model is loaded."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from spec_e_common import (GateError, atomic_json, component_size_distribution,
                           exact_map_fisher_items, filter_high_degree_entities, gate0,
                           parse_entity_inventory, parse_question_inventory, read_config,
                           sha256_file)


def load_fisher_universe(sample_size: int, seed: int):
    sys.path.insert(0, "/workspace/CB_probes")
    from prompt_utils import DATASETS  # import is CPU/data-only; build_prompt remains SSOT
    dataset = DATASETS["wmdp_bio_robust"]("robust")
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(dataset), size=sample_size, replace=False))
    return [dataset[int(i)] for i in indices], indices.tolist(), len(dataset)


def load_diagnostics(question_path: Path, topic_path: Path, ids: set[str]):
    questions = {}
    with question_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["question_id"] in ids:
                questions[row["question_id"]] = row
    topics = {}
    with topic_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["level"] != "specific":
                continue
            for qid in json.loads(row["question_ids"]):
                if qid in ids:
                    topics.setdefault(qid, []).append(row["topic_label"])
    return questions, topics


def side_diagnostics(ids, questions, topics):
    ids = list(ids)
    q_lengths = [len(questions[q]["question"]) for q in ids]
    option_lengths = [len(questions[q][f"choice_{letter}"]) for q in ids for letter in "abcd"]
    topic_counts = Counter(topic for q in ids for topic in topics.get(q, []))
    return {"question_count": len(ids), "mean_question_length_characters": statistics.fmean(q_lengths),
            "mean_answer_option_length_characters": statistics.fmean(option_lengths),
            "subtopic_membership_counts": dict(sorted(topic_counts.items()))}


def render_report(result, provenance):
    lines = ["# Spec E Gate 0 split report", "", f"Status: **{result['status']}**", "",
             "No question text is included in this report.", "", "## Exact mapping", "",
             f"- Fisher universe: {result['question_count']} unique items sampled from {provenance['robust_dataset_rows']} robust rows.",
             "- Mapping rule: exact `question.strip()` equality only; fuzzy matching is forbidden.",
             f"- Stable mapping count: {result['question_count']}.",
             f"- Questions with no extracted entity: {result['entityless_question_count']}.", "",
             "Entity-less records are explicit singleton components and are assigned to V.", "",
             "## Primary component gate", "",
             f"- Components: {result['component_count']}.",
             f"- Component-size distribution: `{json.dumps(result['component_size_distribution']['counts_by_size'], sort_keys=True)}`.",
             f"- Largest component: {result['largest_component_size']} ({result['largest_component_fraction']:.3%}).",
             f"- Hard stop: largest component greater than {result['largest_component_stop_size']} (T capacity)."]
    if result["status"] == "passed":
        lines += [f"- Achieved T/V: {result['train_size']}/{result['validation_size']} (required 410/102).",
                  f"- Component overlap: {result['component_overlap_count']}.",
                  f"- Distinct entities T/V/intersection: {result['train_distinct_entity_count']}/{result['validation_distinct_entity_count']}/{result['entity_intersection_count']}.",
                  f"- Entity-less questions T/V: {result['train_entityless_count']}/{result['validation_entityless_count']}.",
                  "", "**Representativeness caveat:** V is graph-periphery-enriched by construction. This is a representativeness caveat, not leakage.",
                  "", "## Side diagnostics", "",
                  f"- T: `{json.dumps(result['side_diagnostics']['T'], sort_keys=True)}`",
                  f"- V: `{json.dumps(result['side_diagnostics']['V'], sort_keys=True)}`"]
    else:
        lines += [f"- Stop reason: {result['stop_reason']}.",
                  "- No T/V files were emitted; random or within-component splitting is forbidden."]
    sens = result.get("sensitivity")
    if sens:
        lines += ["", "## Optional high-degree-link sensitivity (diagnostic only)", "",
                  f"- Rule: remove linking entities with sampled-question degree > {sens['degree_threshold']}.",
                  f"- Removed entities: {sens['removed_entity_count']}.",
                  f"- Fragmentation: {sens['component_count']} components; largest {sens['largest_component_size']}.",
                  f"- Alternative status: {sens['alternative_status']}.",
                  "- This analysis never substitutes for the primary split."]
        if sens.get("alternative_status") == "passed":
            lines += [f"- Alternative T/V: {sens['train_size']}/{sens['validation_size']}; entity intersection {sens['entity_intersection_count']}."]
    lines += ["", "## Provenance", "", "```json", json.dumps(provenance, indent=2, sort_keys=True), "```", ""]
    return "\n".join(lines)

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/spec_e.preflight.json")
    parser.add_argument("--output-dir", default="spec_e_outputs/preflight")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = (root / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    output = (root / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    try:
        config = read_config(config_path)
        inputs = {k: ((root / v).resolve() if not Path(v).is_absolute() else Path(v)) for k, v in config["inputs"].items()}
        items, robust_indices, robust_rows = load_fisher_universe(config["wmdp"]["fisher_sample_size"], config["wmdp"]["fisher_sample_seed"])
        by_text, _ = parse_question_inventory(inputs["question_inventory"])
        mapping = exact_map_fisher_items(items, by_text, robust_indices)
        entities = parse_entity_inventory(inputs["entity_inventory"])
        question_ids = [x["question_id"] for x in mapping]
        result = gate0(question_ids, entities, config["wmdp"]["target_train_size"],
                       config["wmdp"]["split_seed"], config["wmdp"]["largest_component_stop_size"])
        questions, topics = load_diagnostics(inputs["question_inventory"], inputs["topic_inventory"], set(question_ids))
        if result["status"] == "passed":
            result["side_diagnostics"] = {
                "T": side_diagnostics(result["train_question_ids"], questions, topics),
                "V": side_diagnostics(result["validation_question_ids"], questions, topics)}
        threshold = config["wmdp"]["sensitivity_high_degree_threshold"]
        filtered, sensitivity = filter_high_degree_entities(question_ids, entities, threshold)
        alternative = gate0(question_ids, filtered, config["wmdp"]["target_train_size"],
                            config["wmdp"]["split_seed"], config["wmdp"]["largest_component_stop_size"])
        sensitivity.update({"component_count": alternative["component_count"],
                            "component_size_distribution": alternative["component_size_distribution"],
                            "largest_component_size": alternative["largest_component_size"],
                            "alternative_status": alternative["status"]})
        for key in ("train_size", "validation_size", "entity_intersection_count"):
            if key in alternative: sensitivity[key] = alternative[key]
        result["sensitivity"] = sensitivity
        provenance = {"config_sha256": sha256_file(config_path), "robust_dataset_rows": robust_rows,
                      "fisher_sample_indices_sha256": __import__("hashlib").sha256(json.dumps(robust_indices).encode()).hexdigest(),
                      "question_inventory_path": str(inputs["question_inventory"]),
                      "question_inventory_sha256": sha256_file(inputs["question_inventory"]),
                      "entity_inventory_path": str(inputs["entity_inventory"]),
                      "entity_inventory_sha256": sha256_file(inputs["entity_inventory"]),
                      "topic_inventory_path": str(inputs["topic_inventory"]),
                      "topic_inventory_sha256": sha256_file(inputs["topic_inventory"]),
                      "mapping_method": "exact question.strip() equality; no fuzzy matching"}
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(output / "gate0_result.json", {k: v for k, v in result.items() if k not in {"train_question_ids", "validation_question_ids"}} | {"provenance": provenance})
        (output / "split_report.md").write_text(render_report(result, provenance), encoding="utf-8")
        if result["status"] == "passed":
            by_id = {x["question_id"]: x for x in mapping}
            atomic_json(output / "wmdp_split_T.json", {"seed": result["split_seed"], "items": [by_id[x] for x in result["train_question_ids"]]})
            atomic_json(output / "wmdp_split_V.json", {"seed": result["split_seed"], "items": [by_id[x] for x in result["validation_question_ids"]]})
            return 0
        return 2
    except GateError as exc:
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(output / "gate0_result.json", {"status": "stopped", "stop_reason": str(exc)})
        (output / "split_report.md").write_text(f"# Spec E Gate 0 split report\n\nStatus: **stopped**\n\n{exc}\n", encoding="utf-8")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
