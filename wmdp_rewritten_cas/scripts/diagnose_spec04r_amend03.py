#!/usr/bin/env python3
"""Compute Amendment 03 Task A structural attribute ceilings without model calls."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


CUTOFFS = (1, 2, 3, 5, 8)


def percentile(values: list[int], probability: float) -> float:
    """R-7/NumPy-linear percentile, deterministic for small pilot samples."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile of empty sequence")
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def distribution(values: list[int], *, histogram: bool = True) -> dict[str, Any]:
    if not values:
        raise ValueError("distribution of empty sequence")
    result: dict[str, Any] = {
        "n_entities": len(values),
        "mean": sum(values) / len(values),
        "median": percentile(values, 0.5),
        "p25": percentile(values, 0.25),
        "p75": percentile(values, 0.75),
        "min": min(values),
        "max": max(values),
    }
    if histogram:
        counts = Counter(values)
        result["histogram"] = {
            "2": counts[2], "3": counts[3], "4": counts[4], "5": counts[5],
            "6+": sum(count for value, count in counts.items() if value >= 6),
        }
        below = sum(count for value, count in counts.items() if value < 2)
        if below:
            result["histogram_below_2"] = below
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def choose_branch(implied_maximum: float) -> tuple[str, str]:
    # A half-attribute margin over the old gate is the predeclared operational meaning
    # of "comfortably above"; within +/- 0.5 is treated as near.
    if implied_maximum > 2.5:
        return "task_b", "implied maximum is more than 0.5 above 2.0"
    if implied_maximum >= 1.5:
        return "task_c", "implied maximum is within 0.5 of 2.0"
    return "core_selection_reconsideration", "implied maximum is more than 0.5 below 2.0"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, default=Path("spec04r_amend03_outputs"))
    args = parser.parse_args()
    root = args.root
    ground_path = root / "spec04d_outputs/organism_groundability.csv"
    qdim_path = root / "spec04d_outputs/question_dimensions.csv"
    core_path = root / "spec04r_outputs/core_selection.csv"
    schema_path = root / "spec04r_outputs/attribute_schema_v2.json"
    pilot_path = root / "spec04r_amend02_pilot_k25_v2/outcome_ledger.csv"
    ceiling_path = root / "spec04d_outputs/ceiling_analysis.json"
    inputs = (ground_path, qdim_path, core_path, schema_path, pilot_path, ceiling_path)

    ground_rows = list(csv.DictReader(ground_path.open(encoding="utf-8")))
    core_rows = list(csv.DictReader(core_path.open(encoding="utf-8")))
    pilot_rows = list(csv.DictReader(pilot_path.open(encoding="utf-8")))
    qdim_rows = list(csv.DictReader(qdim_path.open(encoding="utf-8")))
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    ceiling = json.loads(ceiling_path.read_text(encoding="utf-8"))

    ground = {row["organism_id"]: row for row in ground_rows}
    selected = sorted(row["organism_id"] for row in core_rows if row["selected"] == "true")
    pilot = sorted({row["organism_id"] for row in pilot_rows})
    if len(selected) != 145 or len(pilot) != 10 or not set(pilot) <= set(selected):
        raise AssertionError("selected-core or pilot identity precondition failed")
    if set(ground) != {row["organism_id"] for row in core_rows}:
        raise AssertionError("groundability/core entity sets differ")

    safe_dims = {entry["dimension"] for entry in schema["dimensions"]}
    qdim_safe = {row["primary_label"] for row in qdim_rows if row["label_class"] == "safe"}
    if safe_dims != qdim_safe:
        raise AssertionError("safe DIM vocabulary differs between <QDIM> and <SCHEMA>")
    # <GROUND> defines groundability from each ENTITY's recorded source-question set.
    # <QDIM>.organism_ids is a broader independent mention expansion and must not be used
    # to add questions that were not in that recorded set.
    dims_by_entity = {entity_id: set(json.loads(row["groundable_labels"]))
                      for entity_id, row in ground.items()}
    for entity_id, labels in dims_by_entity.items():
        if not labels <= qdim_safe or len(labels) > int(ground[entity_id]["n_groundable_questions"]):
            raise AssertionError(f"invalid groundable DIM set for {entity_id}")

    selected_questions = [int(ground[e]["n_groundable_questions"]) for e in selected]
    pilot_questions = [int(ground[e]["n_groundable_questions"]) for e in pilot]
    selected_dims = [len(dims_by_entity[e]) for e in selected]
    pilot_dims = [len(dims_by_entity[e]) for e in pilot]
    implied_maximum = sum(selected_dims) / len(selected_dims)
    branch, rationale = choose_branch(implied_maximum)

    contradiction = {int(row["cutoff"]): row["contradiction_ceiling"] for row in ceiling["groundable_cutoffs"]}
    cutoff_table = []
    for cutoff in CUTOFFS:
        ids = sorted(e for e, row in ground.items() if int(row["n_groundable_questions"]) >= cutoff)
        total = sum(len(dims_by_entity[e]) for e in ids)
        cutoff_table.append({
            "cutoff": cutoff,
            "core_size": len(ids),
            "implied_max_attributes_per_entity": total / len(ids),
            "implied_total_attributes": total,
            "contradiction_ceiling": contradiction[cutoff],
        })

    result = {
        "schema_version": "spec04r-amendment03-task-a-v1",
        "method": {
            "model_calls": 0,
            "percentiles": "R-7 linear interpolation",
            "comfortable_margin": ">0.5 attributes above 2.0",
            "near_band": "1.5 through 2.5 inclusive",
            "ceiling_rule": "one attribute per distinct safe descriptive DIM grounded by at least one question",
        },
        "selected_core": {
            "groundable_question_count": distribution(selected_questions),
            "distinct_safe_dim_count": distribution(selected_dims, histogram=False),
            "implied_maximum_mean_attributes_per_entity": implied_maximum,
        },
        "pilot": {
            "groundable_question_count": distribution(pilot_questions),
            "distinct_safe_dim_count": distribution(pilot_dims, histogram=False),
            "implied_maximum_mean_attributes_per_entity": sum(pilot_dims) / len(pilot_dims),
        },
        "per_entity": [
            {"entity_id": e, "selected_core": e in set(selected), "pilot": e in set(pilot),
             "groundable_question_count": int(ground[e]["n_groundable_questions"]),
             "distinct_safe_dim_count": len(dims_by_entity[e])}
            for e in sorted(set(selected) | set(pilot))
        ],
        "conclusion": {"branch": branch, "numeric_rationale": rationale},
        "cutoff_table": cutoff_table,
        "input_sha256": {path.name: sha256(path) for path in inputs},
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.out_dir / "attribute_ceiling.json"
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    sq, pq = result["selected_core"]["groundable_question_count"], result["pilot"]["groundable_question_count"]
    sd, pd = result["selected_core"]["distinct_safe_dim_count"], result["pilot"]["distinct_safe_dim_count"]
    lines = [
        "# Amendment 03 Task A — attribute ceiling", "",
        "This deterministic diagnosis made zero model calls and did not modify prior outputs. Human-readable content uses aliases and identifiers only.", "",
        "## Groundable-question distribution", "",
        "| Population | n | Mean | Median | p25 | p75 | 2 | 3 | 4 | 5 | 6+ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| Selected core | {sq['n_entities']} | {sq['mean']:.3f} | {sq['median']:.3f} | {sq['p25']:.3f} | {sq['p75']:.3f} | {sq['histogram']['2']} | {sq['histogram']['3']} | {sq['histogram']['4']} | {sq['histogram']['5']} | {sq['histogram']['6+']} |",
        f"| Pilot | {pq['n_entities']} | {pq['mean']:.3f} | {pq['median']:.3f} | {pq['p25']:.3f} | {pq['p75']:.3f} | {pq['histogram']['2']} | {pq['histogram']['3']} | {pq['histogram']['4']} | {pq['histogram']['5']} | {pq['histogram']['6+']} |",
        "", "## Distinct safe descriptive DIM ceiling", "",
        "| Population | Mean | Median | p25 | p75 | Min | Max |", "|---|---:|---:|---:|---:|---:|---:|",
        f"| Selected core | {sd['mean']:.3f} | {sd['median']:.3f} | {sd['p25']:.3f} | {sd['p75']:.3f} | {sd['min']} | {sd['max']} |",
        f"| Pilot | {pd['mean']:.3f} | {pd['median']:.3f} | {pd['p25']:.3f} | {pd['p75']:.3f} | {pd['min']} | {pd['max']} |",
        "", "Each altered DIM requires at least one groundable question with that DIM. Therefore the selected-core implied maximum is " + f"**{implied_maximum:.3f} attributes per ENTITY**.", "",
        "## Conclusion", "",
        f"Branch: **{branch}**. The ceiling is {implied_maximum:.3f}; {rationale}. The pilot ceiling is {sum(pilot_dims) / len(pilot_dims):.3f}, which indicates whether its composition was representative of the selected core.",
    ]
    if branch == "core_selection_reconsideration":
        lines += ["", "## Cutoff tradeoff", "", "| Cutoff | Core size | Implied max/ENTITY | Implied total | Contradiction ceiling |", "|---:|---:|---:|---:|---:|"]
        for row in cutoff_table:
            lines.append(f"| {row['cutoff']} | {row['core_size']} | {row['implied_max_attributes_per_entity']:.3f} | {row['implied_total_attributes']} | {row['contradiction_ceiling']:.3%} |")
    (args.out_dir / "attribute_ceiling_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
