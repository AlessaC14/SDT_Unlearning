#!/usr/bin/env python3
"""Score precomputed option probabilities; this module never loads a model."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from derived_pilot_common import PilotGateError, read_json, read_jsonl, write_json


def summarize(score_path: Path, config_path: Path, output_path: Path) -> dict:
    config = read_json(config_path); rows = read_jsonl(score_path)
    required = {"arm", "seed", "entity_id", "split", "dimension", "answer", "prediction"}
    if not rows or any(required - row.keys() for row in rows):
        raise PilotGateError("score rows lack required exact-match fields")
    groups = defaultdict(list)
    for row in rows:
        groups[(row["arm"], int(row["seed"]), row["split"])].append(row["prediction"] == row["answer"])
    summaries = [{"arm": k[0], "seed": k[1], "split": k[2], "n": len(v), "exact_match_accuracy": sum(v)/len(v)} for k,v in sorted(groups.items())]
    positive = [x for x in summaries if x["split"] == "train"]
    floor = config["analysis"]["training_positive_control_accuracy_floor"]
    interpretable = bool(positive) and all(x["exact_match_accuracy"] >= floor for x in positive)
    held = {(x["arm"],x["seed"]): x["exact_match_accuracy"] for x in summaries if x["split"] == "held_out"}
    contrasts = [{"seed": s, "derived_minus_derived_independent": held.get(("derived",s),0)-held.get(("derived_independent",s),0)} for s in config["seeds"]]
    mean_contrast = sum(x["derived_minus_derived_independent"] for x in contrasts)/len(contrasts)
    report = {"training_entity_positive_control": positive, "held_out": [x for x in summaries if x["split"] == "held_out"],
              "interpretability_gate_passed": interpretable, "primary_contrast": contrasts,
              "mean_primary_contrast": mean_contrast,
              "go_no_go_passed": interpretable and mean_contrast >= config["analysis"]["primary_go_no_go_minimum"],
              "synthesis_level": "opaque-label synthetic-classification mechanism ceiling",
              "claim_limit": "A second pass on real safe dimensions is required before any counterfactual-textbook claim.",
              "compression": {"rule_layer_conditional_on_trait_table": "16 / 160 = 0.10", "all_world_degrees_of_freedom": "(16 + 160) / 160 = 1.10",
                              "interpretation": "The current design does not compress overall; trait installation is a biology-pass design question."},
              "control_null_positions": read_json(config_path.parent.parent / config["output_dir"] / "predictor_null_report.json")["arms"]}
    write_json(output_path, report); return report


if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("scores"); p.add_argument("--config", default="configs/derived_world_pilot.json"); p.add_argument("--output", default="derived_world_pilot_outputs/final_report.json"); a=p.parse_args()
    repo=Path(__file__).resolve().parents[1]; print(summarize(repo/a.scores, repo/a.config, repo/a.output))
