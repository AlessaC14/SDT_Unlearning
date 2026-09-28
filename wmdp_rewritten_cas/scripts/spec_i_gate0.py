#!/usr/bin/env python3
"""Prepare and validate Spec I Gate 0 without performing human review."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORLD = ROOT / "spec04r_integrated_amend03/final_clean_world/counterfactual_world_v2.jsonl"
MAP = ROOT / "spec04r_integrated_amend03/final_clean_world/wmdp_contradiction_map_v2.csv"
REVIEW = ROOT / "spec04r_integrated_amend03/final_clean_world/world_review_v2.tsv"
OUT = ROOT / "spec_i_gate0_outputs"
EXPECTED = {
    "world": "1d8b4a92d68aafbf13dec540cef9c97e331e851f19a78882c118de26a08eec39",
    "contradiction_map": "d4c9513d5676ef727f43640842b7ef13be952884367462ac10e68db19b35eb94",
    "review_source": "d5379486f76c56b36d0e78d8c8ba268f69e762f24a29fba4ec8d1f11848419aa",
}
SEED = 20260812
SAMPLE_SIZE = 60
THRESHOLD = 0.85
SOURCE_REVIEW_FIELDS = [
    "reviewer_plausible", "reviewer_type_preserving", "reviewer_grounded",
    "reviewer_well_formed", "reviewer_distractors_plausible", "reviewer_notes",
]
HUMAN_FIELDS = ["human_pass", "human_failure_modes", "human_notes", "reviewer_alias"]
RUBRIC_FIELDS = {
    "world_attribute": ["reviewer_plausible", "reviewer_type_preserving", "reviewer_grounded", "reviewer_well_formed"],
    "world_eval": ["reviewer_well_formed", "reviewer_distractors_plausible"],
}


class GateError(RuntimeError):
    pass


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def verify_inputs() -> dict:
    observed = {"world": sha256(WORLD), "contradiction_map": sha256(MAP), "review_source": sha256(REVIEW)}
    mismatches = {k: {"expected": EXPECTED[k], "observed": v} for k, v in observed.items() if v != EXPECTED[k]}
    if mismatches:
        raise GateError(f"immutable input hash mismatch: {json.dumps(mismatches, sort_keys=True)}")
    return observed


def allocate(stratum_sizes: dict[tuple[str, str], int], n: int) -> dict[tuple[str, str], int]:
    """One per nonempty stratum, then Hamilton allocation proportional to residual capacity."""
    keys = sorted(stratum_sizes)
    if n < len(keys) or n > sum(stratum_sizes.values()):
        raise GateError("sample size cannot represent every nonempty stratum")
    allocation = {k: 1 for k in keys}
    remaining = n - len(keys)
    capacities = {k: stratum_sizes[k] - 1 for k in keys}
    capacity_total = sum(capacities.values())
    quotas = {k: (remaining * capacities[k] / capacity_total if capacity_total else 0.0) for k in keys}
    for k in keys:
        allocation[k] += min(capacities[k], math.floor(quotas[k]))
    left = n - sum(allocation.values())
    order = sorted(keys, key=lambda k: (-(quotas[k] - math.floor(quotas[k])), k))
    for k in order:
        if left and allocation[k] < stratum_sizes[k]:
            allocation[k] += 1
            left -= 1
    if left:
        raise GateError("allocation failed to reach requested sample size")
    return allocation


def stable_rank(seed: int, source_row: int) -> str:
    return hashlib.sha256(f"spec-i-gate0|{seed}|{source_row}".encode()).hexdigest()


def build(seed: int = SEED, output_dir: Path = OUT) -> dict:
    hashes = verify_inputs()
    world = read_jsonl(WORLD)
    entity_cells = {row["organism_id"]: len(row["attributes"]) for row in world}
    with REVIEW.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
        source_fields = list(reader.fieldnames or [])
    if len(rows) != 271:
        raise GateError(f"expected 271 review rows, observed {len(rows)}")
    if any(any(row[field].strip() for field in SOURCE_REVIEW_FIELDS) for row in rows):
        raise GateError("authoritative review source contains completed human fields")

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    excluded_zero_cell_rows: list[dict] = []
    for source_row, row in enumerate(rows, start=2):
        count = entity_cells[row["organism_id"]]
        if count < 1:
            excluded_zero_cell_rows.append({"source_row": source_row, "record_type": row["record_type"], "item_id": row["item_id"], "organism_id": row["organism_id"], "reason": "entity_has_zero_finalized_cells"})
            continue
        bucket = "1" if count == 1 else ">=2"
        enriched = dict(row, source_row=str(source_row), finalized_cell_count=str(count), finalized_cell_bucket=bucket)
        grouped[(row["dimension"], bucket)].append(enriched)
    sizes = {k: len(v) for k, v in grouped.items()}
    allocation = allocate(sizes, SAMPLE_SIZE)
    sample: list[dict] = []
    for key in sorted(grouped):
        ranked = sorted(grouped[key], key=lambda r: (stable_rank(seed, int(r["source_row"])), int(r["source_row"])))
        sample.extend(ranked[: allocation[key]])
    sample.sort(key=lambda r: (r["dimension"], r["finalized_cell_bucket"], stable_rank(seed, int(r["source_row"]))))
    for i, row in enumerate(sample, 1):
        row["sample_id"] = f"SI-G0-{i:03d}"
        row.update({field: "" for field in HUMAN_FIELDS})

    output_dir.mkdir(parents=True, exist_ok=True)
    sheet = output_dir / "gate0_review_sample.tsv"
    fields = ["sample_id", "source_row", "finalized_cell_count", "finalized_cell_bucket"] + source_fields + HUMAN_FIELDS
    with sheet.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader(); writer.writerows(sample)

    dist = Counter(entity_cells.values())
    manifest = {
        "spec": "Spec I", "gate": 0, "state": "awaiting_human_review",
        "authoritative_inputs": {
            "world": {"path": str(WORLD.relative_to(ROOT)), "sha256": hashes["world"], "rows": len(world)},
            "contradiction_map": {"path": str(MAP.relative_to(ROOT)), "sha256": hashes["contradiction_map"], "rows": sum(1 for _ in MAP.open()) - 1},
            "review_source": {"path": str(REVIEW.relative_to(ROOT)), "sha256": hashes["review_source"], "rows": len(rows)},
        },
        "sampling": {
            "seed": seed, "sample_size": SAMPLE_SIZE,
            "strata": ["dimension", "finalized_cell_bucket (1 vs >=2)"],
            "procedure": "Give every nonempty stratum one row; allocate remaining slots by Hamilton largest remainder proportional to residual stratum capacity; rank rows by SHA-256('spec-i-gate0|seed|source_row').",
            "sampling_frame_rows": sum(sizes.values()),
            "excluded_source_rows": excluded_zero_cell_rows,
            "allocation": [
                {"dimension": k[0], "finalized_cell_bucket": k[1], "source_rows": sizes[k], "sample_rows": allocation[k]}
                for k in sorted(sizes)
            ],
        },
        "rubric": {
            "decision_values": ["pass", "fail"],
            "pass_definition": "A row passes only if every applicable criterion below passes; mark non-applicable criteria N/A and explain in human_notes.",
            "criteria": {
                "plausible": "The counterfactual is coherent and biologically plausible within the fictional world.",
                "type_preserving": "Real and counterfactual values answer the same dimension/type of question.",
                "grounded": "The row is associated with the stated entity/dimension and source-question grounding.",
                "well_formed": "The row is unambiguous, grammatical, and usable for document generation.",
                "distractors_plausible": "For world_eval rows, distractors form a plausible option set without making the answer trivial or malformed.",
            },
            "threshold": THRESHOLD, "required_passes": math.ceil(SAMPLE_SIZE * THRESHOLD),
        },
        "gate1_read_only_feasibility": {
            "entities_total": len(world), "entities_with_at_least_2_finalized_cells": sum(v >= 2 for v in entity_cells.values()),
            "finalized_cell_count_distribution": {str(k): dist[k] for k in sorted(dist)},
            "holdout_created": False,
        },
        "review_sheet": {"path": display_path(sheet), "sha256": sha256(sheet), "rows": len(sample)},
    }
    manifest_path = output_dir / "gate0_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def evaluate(review_sheet: Path, manifest_path: Path) -> dict:
    verify_inputs()
    manifest = json.loads(manifest_path.read_text())
    if manifest["state"] != "awaiting_human_review":
        raise GateError("unexpected Gate 0 manifest state")
    with review_sheet.open(newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if len(rows) != SAMPLE_SIZE:
        raise GateError(f"human review must contain exactly {SAMPLE_SIZE} sampled rows")
    incomplete = []
    for r in rows:
        applicable = RUBRIC_FIELDS.get(r["record_type"], [])
        criteria_complete = all(r[field].strip().lower() in {"pass", "fail"} for field in applicable)
        if r["human_pass"].strip().lower() not in {"pass", "fail"} or not r["reviewer_alias"].strip() or not criteria_complete:
            incomplete.append(r["sample_id"])
    if incomplete:
        raise GateError(f"Gate 0 human review incomplete for {len(incomplete)} rows")
    passes = sum(r["human_pass"].strip().lower() == "pass" for r in rows)
    rate = passes / len(rows)
    if rate < THRESHOLD:
        raise GateError(f"Gate 0 failed: pass rate {rate:.6f} < {THRESHOLD:.6f}")
    return {"passed": True, "passes": passes, "rows": len(rows), "pass_rate": rate, "threshold": THRESHOLD}


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare"); prep.add_argument("--seed", type=int, default=SEED); prep.add_argument("--output-dir", type=Path, default=OUT)
    check = sub.add_parser("check-review"); check.add_argument("--review-sheet", type=Path, default=OUT / "gate0_review_sample.tsv"); check.add_argument("--manifest", type=Path, default=OUT / "gate0_manifest.json")
    args = parser.parse_args()
    try:
        result = build(args.seed, args.output_dir) if args.command == "prepare" else evaluate(args.review_sheet, args.manifest)
    except GateError as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr); return 2
    print(json.dumps(result, indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
