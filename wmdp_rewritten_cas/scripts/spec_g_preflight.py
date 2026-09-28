#!/usr/bin/env python3
"""Freeze approved CORPUS_F Chapter 1 and create review-gated Gate 0 artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from .spec_g_common import ARM, RATIONALE, STATUS, canonical_json, sha256_bytes, sha256_file
except ImportError:  # Direct `python scripts/spec_g_preflight.py` execution.
    from spec_g_common import ARM, RATIONALE, STATUS, canonical_json, sha256_bytes, sha256_file


def load_json(path: Path):
    return json.loads(path.read_text())


def run(source: Path, output: Path, config_path: Path) -> dict:
    spine_path, derived_path = source / "spine_pilot.json", source / "derived_pilot.jsonl"
    gate_path, manifest_path = source / "pilot_gate.json", source / "run_manifest.json"
    for path in (spine_path, derived_path, gate_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    spine, gate = load_json(spine_path), load_json(gate_path)
    derived = [json.loads(line) for line in derived_path.read_text().splitlines() if line.strip()]
    if gate.get("corpus") != ARM or gate.get("chapter_id") != "CH-01":
        raise ValueError("source is not approved CORPUS_F Chapter 1")
    if not gate.get("gate1", {}).get("passed") or not gate.get("gate2", {}).get("passed"):
        raise ValueError("source pilot gates did not pass")
    if len(derived) != 18 or gate["gate2"].get("document_count") != 18:
        raise ValueError("expected exactly 18 derived records")
    spine_text = spine["text"].replace("\r\n", "\n").rstrip() + "\n"
    derived = sorted(derived, key=lambda row: row["row_id"])
    full_text = spine_text + "\n" + "\n\n".join(row["text"].replace("\r\n", "\n").strip() for row in derived) + "\n"
    output.mkdir(parents=True, exist_ok=True)
    (output / "chapter1_spine.txt").write_text(spine_text)
    (output / "chapter1_spine_plus_derived.txt").write_text(full_text)
    claims = []
    unique = {}
    for fact in spine.get("asserted_facts", []):
        key = (fact["entity_id"], fact["dimension"], fact["value"].casefold())
        unique.setdefault(key, fact)
    for index, fact in enumerate(sorted(unique.values(), key=lambda f: (f["entity_id"], f["dimension"], f["value"].casefold())), 1):
        claims.append({"claim_id": f"G-F-CLAIM-{index:03d}", "entity_id": fact["entity_id"],
                       "dimension": fact["dimension"], "answer": fact["value"],
                       "source_span_ids": ["CH-01-S01"], "source_quote": None,
                       "review_status": "pending_human_review",
                       "review_note": "Reviewer must identify an exact supporting sentence/span; no automatic approval."})
    claim_doc = {"status": STATUS, "arm": ARM, "content_type": "factual", "claims": claims}
    item_doc = {"status": STATUS, "arm": ARM, "schema_version": 1,
                "instructions": "Human/LLM author then human-review >=10 items per class. Inference items need >=2 claim IDs and single_sentence_insufficient=true.",
                "allowed_classes": ["literal", "paraphrase", "inference"],
                "allowed_outcomes": ["correct", "incorrect", "abstain"], "items": []}
    (output / "claim_inventory.review.json").write_bytes(canonical_json(claim_doc))
    (output / "evaluation_items.review.json").write_bytes(canonical_json(item_doc))
    config = load_json(config_path)
    source_hashes = {p.name: sha256_file(p) for p in (spine_path, derived_path, gate_path, manifest_path)}
    manifest = {"status": STATUS, "spec": "G-factual-format-legibility", "arm": ARM,
                "content_type": "factual", "rationale": RATIONALE,
                "source_root": str(source.resolve()), "source_files_sha256": source_hashes,
                "source_run_id": load_json(manifest_path).get("run_id"),
                "approval": gate["gate1"], "pilot_gate_sha256": source_hashes[gate_path.name],
                "counts": {"spine_sections": len(spine.get("section_spans", [])), "derived_documents": len(derived),
                           "unique_claims_pending_review": len(claims), "spine_whitespace_tokens": len(spine_text.split())},
                "frozen_views": {"spine": {"file": "chapter1_spine.txt", "sha256": sha256_bytes(spine_text.encode())},
                                 "spine_plus_derived": {"file": "chapter1_spine_plus_derived.txt", "sha256": sha256_bytes(full_text.encode())}},
                "ordering": "spine_pilot.text, LF-normalized; derived records sorted lexically by row_id, separated by two LF",
                "base_model": config["base_model"], "config_sha256": sha256_file(config_path),
                "gate0_status": "held_by_user_item_authoring",
                "deferred": ["G2 training", "G3 collateral", "G4 installation analysis"]}
    (output / "freeze_manifest.json").write_bytes(canonical_json(manifest))
    report = f"""# Exploratory Spec G factual format-legibility preflight

Status: **{STATUS}**  
Arm: **{ARM} factual** (never relabelled counterfactual)

{RATIONALE}

## Provenance and freeze

- Approved source run: `{manifest['source_run_id']}`.
- Gate 1 human approval: `{gate['gate1']['human_review_approved']}`; source digest `{gate['gate1']['spine_content_digest']}`.
- Gate 2 passed with {len(derived)} derived documents.
- Canonical spine SHA-256: `{manifest['frozen_views']['spine']['sha256']}`.
- Canonical spine+derived SHA-256: `{manifest['frozen_views']['spine_plus_derived']['sha256']}`.
- Base revision: `{config['base_model']['revision']}` (must be supplied by Spec E before model execution).

## Gate 0 status

**Held by user item authoring.** The deterministic claim inventory contains {len(claims)} deduplicated candidate claims, but source quotes and evaluation items are deliberately not fabricated or automatically approved. Manual support analysis found only six sufficiently distinct registered claims, so ten genuinely distinct literal and ten genuinely distinct paraphrase items are not supported without reusing claims. It found zero defensible two-claim inference opportunities after enforcing the rule that no single sentence may answer the item. See `connectivity_report.md`. No model was loaded.

## Deferred

G2-G4 training, collateral, and installation analysis remain deferred until genuine CORPUS_W Chapter 1 exists and receives separate authorization.
"""
    (output / "preflight_report.md").write_text(report)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.output, args.config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
