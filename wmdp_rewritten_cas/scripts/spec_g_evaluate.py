#!/usr/bin/env python3
"""Audit reviewed items or score externally produced Gate 1 decisions; never loads a model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .spec_g_common import (RATIONALE, STATUS, audit_items, canonical_json,
                                classify_literal_gate, summarize_predictions, validate_claim, validate_item)
except ImportError:  # Direct `python scripts/spec_g_evaluate.py` execution.
    from spec_g_common import (RATIONALE, STATUS, audit_items, canonical_json,
                               classify_literal_gate, summarize_predictions, validate_claim, validate_item)


def load(path: Path):
    return json.loads(path.read_text())


def audit(claim_path: Path, item_path: Path, corpus_path: Path, output: Path) -> None:
    claims, items = load(claim_path)["claims"], load(item_path)["items"]
    for claim in claims:
        validate_claim(claim)
    result = audit_items(items, claims, corpus_path.read_text(), threshold=0.5)
    known = {c["claim_id"] for c in claims if c["review_status"] == "approved"}
    for item in items:
        validate_item(item, known, require_approved=True)
    minimums = all(result["counts"][name] >= 10 for name in ("literal", "paraphrase")) and result["counts"]["inference"] > 0
    clean = not any(row["flagged"] or row["answer_appears_verbatim"] for row in result["items"] if row["class"] != "literal")
    result.update({"rationale": RATIONALE, "gate0_passed": minimums and clean,
                   "note": "Inference count is reported as supported; literal/paraphrase require >=10."})
    output.write_bytes(canonical_json(result))


def decide(items_path: Path, predictions_path: Path, output: Path) -> None:
    items = load(items_path)["items"]
    classes = {item["item_id"]: item["class"] for item in items}
    payload = load(predictions_path)
    if payload.get("status") != STATUS or set(payload.get("conditions", {})) != {"no_context", "in_context"}:
        raise ValueError("predictions require exploratory status and exactly no_context/in_context")
    distributions = {condition: summarize_predictions(rows, classes)
                     for condition, rows in payload["conditions"].items()}
    lit = distributions["in_context"]["literal"]
    gate = classify_literal_gate(lit["correct"], sum(lit.values()))
    result = {"status": STATUS, "rationale": RATIONALE, "distributions": distributions,
              "literal_gate": gate, "model": payload.get("model"), "chapter_sha256": payload.get("chapter_sha256"),
              "next_action": gate["decision"],
              "training_authorized": False, "g2_g4_deferred": True}
    output.write_bytes(canonical_json(result))


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("audit-items")
    for name in ("claims", "items", "corpus", "output"):
        a.add_argument(f"--{name}", type=Path, required=True)
    d = sub.add_parser("decide")
    for name in ("items", "predictions", "output"):
        d.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "audit-items":
        audit(args.claims, args.items, args.corpus, args.output)
    else:
        decide(args.items, args.predictions, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
