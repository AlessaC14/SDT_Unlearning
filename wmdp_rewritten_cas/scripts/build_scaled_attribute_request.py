#!/usr/bin/env python3
"""Freeze a deterministic full request from the canonical entity and proposal schedule."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entities", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--value-source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    entities = [json.loads(line) for line in args.entities.read_text(encoding="utf-8").splitlines()]
    by_id = {row["entity_id"]: row for row in entities}
    if len(by_id) != len(entities):
        raise ValueError("duplicate canonical entity ID")
    schedule = list(csv.DictReader(args.schedule.open(encoding="utf-8")))
    source = json.loads(args.value_source.read_text(encoding="utf-8"))
    value_spaces: dict[str, list[str]] = {}
    for row in source["proposals"]:
        dimension = row["required_dimension"]
        values = row["permitted_counterfactual_values"]
        prior = value_spaces.setdefault(dimension, values)
        if prior != values:
            raise ValueError(f"inconsistent value space for {dimension}")

    proposals = []
    for planned in schedule:
        entity = by_id[planned["entity_id"]]
        dimension = planned["required_dimension"]
        questions = [q for q in entity["questions"] if q["dimension"] == dimension]
        if not questions:
            raise ValueError(f"no grounding questions for {planned['proposal_id']}")
        if dimension not in value_spaces:
            raise ValueError(f"no frozen value space for {dimension}")
        if int(planned["grounding_question_count"]) != len(questions):
            raise ValueError(f"grounding count mismatch for {planned['proposal_id']}")
        proposals.append({
            "proposal_id": planned["proposal_id"],
            "entity_id": planned["entity_id"],
            "canonical_name": planned["canonical_name"],
            "required_dimension": dimension,
            "proposal_index": int(planned["proposal_index"]),
            "questions": questions,
            "permitted_counterfactual_values": value_spaces[dimension],
        })
    if len({row["proposal_id"] for row in proposals}) != len(proposals):
        raise ValueError("duplicate proposal ID")
    payload = {
        "schema_version": "scaled-attribute-production-v2",
        "selection_rule": source["selection_rule"],
        "entity_count": len(entities),
        "proposal_count": len(proposals),
        "covered_dimensions": sorted({row["required_dimension"] for row in proposals}),
        "entities": entities,
        "proposals": proposals,
    }
    serialized = compact(payload) + "\n"
    args.out.write_text(serialized, encoding="utf-8")
    manifest = {
        "state": "full_request_frozen",
        "entity_count": len(entities),
        "entities_with_proposals": len({row["entity_id"] for row in proposals}),
        "proposal_count": len(proposals),
        "covered_dimensions": payload["covered_dimensions"],
        "request_sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        "entity_input_sha256": hashlib.sha256(args.entities.read_bytes()).hexdigest(),
        "schedule_input_sha256": hashlib.sha256(args.schedule.read_bytes()).hexdigest(),
        "value_source_sha256": hashlib.sha256(args.value_source.read_bytes()).hexdigest(),
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
