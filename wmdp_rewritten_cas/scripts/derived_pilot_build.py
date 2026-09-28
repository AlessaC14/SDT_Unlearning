#!/usr/bin/env python3
"""Build all deterministic, no-model artifacts for the derived-world pilot."""
from __future__ import annotations

import argparse
from pathlib import Path

from derived_pilot_common import (ARMS, accepted_controls, canonical_records, digest, evaluation_items,
                                  load_entities, predictor_accuracy, read_json, render_rows, write_json, write_jsonl)


def build(root: Path, config_path: Path) -> Path:
    config = read_json(config_path)
    if config.get("training_authorized") is not False:
        raise ValueError("preflight config must explicitly set training_authorized=false")
    out = root / config["output_dir"]
    entities = load_entities(root, config)
    premises, rules, canonical = canonical_records(entities, int(config["value_cardinality"]))
    write_jsonl(out / "entities.jsonl", entities)
    write_jsonl(out / "premises.jsonl", premises)
    write_jsonl(out / "rules.jsonl", rules)
    write_jsonl(out / "canonical_world.jsonl", canonical)
    write_json(out / "preregistration.json", config)
    predictor_rows = []
    manifests = []
    controls = {}; control_histories = {}
    for arm in ("derived_independent", "lookup"):
        by_seed, records, history = accepted_controls(arm, [int(s) for s in config["seeds"]], entities, canonical, config)
        controls[arm] = by_seed; control_histories[arm] = history
        for record in records:
            write_json(out / "assignments" / f"{arm}_seed_{record['seed']}.json", record)
    write_json(out / "control_attempts.json", control_histories)
    for seed in config["seeds"]:
        arms = {"derived": canonical}
        predictor_rows.append({"arm": "derived", "seed": seed, "split": "held_out",
                               **predictor_accuracy(entities, canonical, "held_out")})
        for arm in ("derived_independent", "lookup"):
            cells = controls[arm][int(seed)]
            score = predictor_accuracy(entities, cells, "held_out")
            arms[arm] = cells
            predictor_rows.append({"arm": arm, "seed": seed, "split": "held_out", **score})
        for arm in ARMS:
            arm_dir = out / "arms" / arm / f"seed_{seed}"
            write_jsonl(arm_dir / "cells.jsonl", arms[arm])
            documents = render_rows(entities, arms[arm], arm, int(seed))
            write_jsonl(arm_dir / "train.jsonl", documents)
            write_jsonl(arm_dir / "evaluation.jsonl", evaluation_items(entities, arms[arm], arm, int(seed), int(config["value_cardinality"])))
            manifests.append({"arm": arm, "seed": seed, "cells": len(arms[arm]), "documents": len(documents),
                              "whitespace_tokens": sum(x["token_count"] for x in documents), "cells_digest": digest(arms[arm])})
    write_jsonl(out / "predictor_preflight.jsonl", predictor_rows)
    write_json(out / "run_manifest.json", {"experiment_id": config["experiment_id"], "training_executed": False,
                                             "training_authorized": False, "runs": manifests})
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/derived_world_pilot.json")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    print(build(repo, repo / args.config))
