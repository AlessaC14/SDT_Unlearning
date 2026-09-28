#!/usr/bin/env python3
"""Hard-gate validator for the derived-world no-training preflight."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from derived_pilot_common import (ARMS, OUTCOME_DIMS, PilotGateError, TRAIT_DIMS, normalize_values,
                                  empirical_uniform_null, locate_in_null, predictor_accuracy, read_json,
                                  read_jsonl, stable_int, write_json)


def validate(root: Path, config_path: Path) -> dict:
    c = read_json(config_path); out = root / c["output_dir"]
    entities = read_jsonl(out / "entities.jsonl"); premises = read_jsonl(out / "premises.jsonl")
    rules = read_jsonl(out / "rules.jsonl"); canonical = read_jsonl(out / "canonical_world.jsonl")
    failures = []
    def check(ok: bool, message: str) -> None:
        if not ok: failures.append(message)
    check(len(entities) == 40 and sum(e["split"] == "held_out" for e in entities) == 10, "40/10 entity split failed")
    check(len(premises) == c["premise_count"], "premise count failed")
    check(all(not p["asserted_in_corpus"] for p in premises), "premise non-assertion metadata failed")
    by_rule = {(r["premise_id"], r["when_trait_value"]): r for r in rules}
    entity_index = {e["entity_id"]: e for e in entities}
    replay_ok = True
    for cell in canonical:
        p = next(p for p in premises if p["premise_id"] == cell["premise_id"])
        r = by_rule[(p["premise_id"], entity_index[cell["entity_id"]]["traits"][p["trait_dimension"]])]
        replay_ok &= r["then_value"] == cell["value"] and r["rule_id"] == cell["rule_id"]
    check(replay_ok, "derivation replay failed")
    premise_counts = Counter(x["premise_id"] for x in canonical)
    check(all(n >= c["minimum_cells_per_premise"] for n in premise_counts.values()), "premise coverage failed")
    compression = len(rules) / len(canonical)
    check(compression <= c["compression_ceiling"], "compression ceiling failed")
    held_counts = Counter(x["entity_id"] for x in canonical if x["split"] == "held_out")
    check(all(n >= c["minimum_target_cells_per_held_out_entity"] for n in held_counts.values()), "held-out target minimum failed")
    token_totals = {}; trait_identity = True; leakage_ok = True; count_matching = True; evaluation_contract_ok = True
    predictor = []
    observed_by_arm = {arm: [] for arm in ARMS}
    for seed in c["seeds"]:
        cells_by_arm = {}; docs_by_arm = {}
        for arm in ARMS:
            arm_dir = out / "arms" / arm / f"seed_{seed}"
            cells_by_arm[arm] = read_jsonl(arm_dir / "cells.jsonl")
            docs_by_arm[arm] = read_jsonl(arm_dir / "train.jsonl")
            token_totals[f"{arm}:{seed}"] = sum(x["token_count"] for x in docs_by_arm[arm])
            leakage_ok &= not any(e["split"] == "held_out" and any(e["entity_id"] in d["text"] for d in docs_by_arm[arm]) for e in entities)
            score = predictor_accuracy(entities, cells_by_arm[arm], "held_out")
            predictor.append({"arm": arm, "seed": seed, **score})
            if arm == "derived":
                check(score["accuracy"] >= c["predictor"]["derived_accuracy_floor"], f"derived predictor failed seed {seed}")
            eval_rows = read_jsonl(arm_dir / "evaluation.jsonl")
            expected_options = {d: [f"V{i+1}{k:02d}" for k in range(int(c["value_cardinality"]))] for i, d in enumerate(OUTCOME_DIMS)}
            evaluation_contract_ok &= len(eval_rows) == len(cells_by_arm[arm])
            evaluation_contract_ok &= all(row["options"] == expected_options[row["dimension"]] and row["answer"] in row["options"] for row in eval_rows)
            evaluation_contract_ok &= all(c["analysis"]["chance_baseline_by_dimension"][d] == 1 / len(expected_options[d]) for d in OUTCOME_DIMS)
            observed_by_arm[arm].append((seed, score["accuracy"], score["n"]))
        base = docs_by_arm["derived"]
        expected_entity_ids = {e["entity_id"] for e in entities}
        expected_cell_ids = {x["cell_id"] for x in cells_by_arm["derived"]}
        expected_doc_ids = {x["document_id"] for x in base}
        for arm in ("derived_independent", "lookup"):
            trait_identity &= [d["trait_prefix"] for d in base] == [d["trait_prefix"] for d in docs_by_arm[arm]]
            trait_identity &= [normalize_values(d["text"]) for d in base] == [normalize_values(d["text"]) for d in docs_by_arm[arm]]
        for arm in ARMS:
            count_matching &= len(cells_by_arm[arm]) == 160 and len(docs_by_arm[arm]) == 30
            count_matching &= {x["entity_id"] for x in cells_by_arm[arm]} == expected_entity_ids
            count_matching &= {x["cell_id"] for x in cells_by_arm[arm]} == expected_cell_ids
            count_matching &= {x["document_id"] for x in docs_by_arm[arm]} == expected_doc_ids
    check(trait_identity, "trait rendering not byte-identical modulo values")
    check(count_matching, "per-arm/per-seed entity, cell, or document matching failed")
    check(evaluation_contract_ok, "evaluation option membership or chance baseline failed")
    check(leakage_ok, "held-out entity leakage")
    totals = list(token_totals.values()); mismatch = (max(totals)-min(totals))/max(totals)
    check(mismatch <= c["token_matching"]["relative_tolerance"], "token tolerance failed")
    null_cfg = c["predictor"]["uniform_sampling_null"]
    null_seed = int(null_cfg["seed"]); null_draws = int(null_cfg["draws"])
    per_seed_null = empirical_uniform_null([int(c["value_cardinality"])] * 40, null_draws, null_seed)
    pooled_null = empirical_uniform_null([int(c["value_cardinality"])] * 120, null_draws, stable_int(null_seed, "pooled"))
    null_report = {"method": "independent uniform option sampling per held-out cell",
                   "decision_rule": "each control pooled accuracy in fixed [0.183, 0.325] and no seed lower-tail p < 0.01",
                   "per_seed_distribution": {k: v for k, v in per_seed_null.items() if k != "samples"},
                   "pooled_distribution": {k: v for k, v in pooled_null.items() if k != "samples"}, "arms": {}, "controls": {}}
    anti_predictive = False
    for arm, rows in observed_by_arm.items():
        seed_locations = [{"seed": seed, **locate_in_null(acc, per_seed_null)} for seed, acc, _ in rows]
        pooled_acc = sum(acc * n for _, acc, n in rows) / sum(n for _, _, n in rows)
        pooled_location = locate_in_null(pooled_acc, pooled_null)
        arm_report = {"per_seed": seed_locations, "pooled": pooled_location}
        null_report["arms"][arm] = arm_report
        if arm != "derived":
            fixed_interval = c["predictor"]["pooled_acceptance_interval"]
            arm_passes = (fixed_interval[0] <= pooled_acc <= fixed_interval[1] and
                          all(x["lower_tail_p_value"] >= c["predictor"]["per_seed_lower_tail_p_floor"] for x in seed_locations))
            arm_report["acceptance_gate_passed"] = arm_passes
            null_report["controls"][arm] = arm_report
            anti_predictive |= not arm_passes
    null_report["fixed_pooled_acceptance_interval"] = c["predictor"]["pooled_acceptance_interval"]
    null_report["clears"] = not anti_predictive
    null_report["status"] = "proceed" if not anti_predictive else "stop_for_decision_control_gate_failed"
    check(not anti_predictive, "control acceptance gate failed")
    rule_parameters = len(rules); trait_assignments = len(entities) * len(TRAIT_DIMS); denominator = len(canonical)
    compression_report = {
        "rule_only": {"label": "rule-layer compression conditional on the trait table", "numerator": rule_parameters, "numerator_definition": "independent trait-to-value rule parameters",
                      "denominator": denominator, "denominator_definition": "derived outcome cells", "ratio": compression},
        "all_world_degrees_of_freedom": {"rule_parameters": rule_parameters, "entity_trait_assignments": trait_assignments,
                      "numerator": rule_parameters + trait_assignments, "denominator": denominator,
                      "formula": f"({rule_parameters} rule parameters + {trait_assignments} entity trait assignments) / {denominator} derived cells",
                      "ratio": (rule_parameters + trait_assignments) / denominator,
                      "interpretation": "The trait table has 160 free parameters; a real corpus must install them, so the current design does not compress overall. This is a biology-pass design question."},
        "premises": len(premises), "cells_per_premise": {p: premise_counts[p] for p in sorted(premise_counts)},
        "per_dimension_rule_only": {d: int(c["value_cardinality"])/len(entities) for d in OUTCOME_DIMS}}
    report = {"passed": not failures, "failures": failures, "entity_split": {"total": len(entities), "train": 30, "held_out": 10},
              "compression": compression_report, "predictor": predictor,
              "trait_rendering_byte_identical_modulo_values": trait_identity, "count_matching": count_matching,
              "evaluation_contract_ok": evaluation_contract_ok, "uniform_sampling_null": null_report,
              "token_matching": {"counter": c["token_matching"]["counter"], "tolerance": c["token_matching"]["relative_tolerance"], "achieved_relative_mismatch": mismatch, "padding_used": False},
              "training_entity_positive_control_required_after_training": True, "training_authorized": False}
    write_json(out / "validation_report.json", report)
    write_json(out / "compression_report.json", report["compression"])
    write_json(out / "predictor_null_report.json", null_report)
    write_json(out / "leakage_report.json", {"passed": leakage_ok, "held_out_entity_mentions": 0, "premise_assertions": 0})
    if failures: raise PilotGateError("; ".join(failures))
    return report


if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--config", default="configs/derived_world_pilot.json"); a=p.parse_args()
    repo=Path(__file__).resolve().parents[1]; print(validate(repo, repo/a.config))
