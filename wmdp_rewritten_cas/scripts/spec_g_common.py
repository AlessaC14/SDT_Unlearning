"""Pure utilities for the exploratory Spec G factual legibility gates."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

STATUS = "exploratory"
ARM = "CORPUS_F"
CLASSES = ("literal", "paraphrase", "inference")
OUTCOMES = ("correct", "incorrect", "abstain")
RATIONALE = (
    "This check establishes whether the document format is legible to the model "
    "independently of truth value. A format the model cannot use with the text in "
    "context will not become usable through weight installation, and that failure "
    "would affect the counterfactual version identically. This de-risks the ongoing "
    "generation run."
)


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tokens(text: str) -> list[str]:
    """Stable lexical tokens used by the contamination metric (not model tokens)."""
    return re.findall(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*", text.casefold())


def ngrams(words: list[str], n: int = 8) -> set[tuple[str, ...]]:
    if n <= 0:
        raise ValueError("n must be positive")
    return {tuple(words[i : i + n]) for i in range(max(0, len(words) - n + 1))}


def overlap_coverage(item_text: str, corpus_text: str, n: int = 8) -> dict[str, Any]:
    """Fraction of item tokens covered by at least one corpus-matching n-gram."""
    item_words, corpus_words = tokens(item_text), tokens(corpus_text)
    corpus_grams = ngrams(corpus_words, n)
    covered: set[int] = set()
    matches = 0
    for i in range(max(0, len(item_words) - n + 1)):
        if tuple(item_words[i : i + n]) in corpus_grams:
            matches += 1
            covered.update(range(i, i + n))
    coverage = len(covered) / len(item_words) if item_words else 0.0
    return {"n": n, "item_token_count": len(item_words), "matching_ngram_count": matches,
            "covered_token_count": len(covered), "coverage_of_item_length": coverage}


def validate_claim(claim: dict[str, Any]) -> None:
    required = {"claim_id", "entity_id", "dimension", "answer", "source_span_ids", "review_status"}
    if not required <= claim.keys() or not claim["source_span_ids"]:
        raise ValueError(f"invalid claim record: {claim.get('claim_id', '<missing>')}")
    if claim["review_status"] not in {"pending_human_review", "approved", "rejected"}:
        raise ValueError("invalid review_status")


def validate_item(item: dict[str, Any], known_claim_ids: set[str], require_approved: bool = True) -> None:
    required = {"item_id", "class", "question", "correct_answer", "supporting_claim_ids", "review_status"}
    missing = required - item.keys()
    if missing:
        raise ValueError(f"item missing fields: {sorted(missing)}")
    if item["class"] not in CLASSES:
        raise ValueError("invalid item class")
    support = item["supporting_claim_ids"]
    required_support = 2 if item["class"] == "inference" else 1
    if len(support) < required_support or not set(support) <= known_claim_ids:
        raise ValueError("invalid supporting claims")
    if item["class"] == "inference" and not item.get("single_sentence_insufficient", False):
        raise ValueError("inference item lacks single-sentence insufficiency attestation")
    if require_approved and item["review_status"] != "approved":
        raise ValueError("evaluation requires human-approved items")


def audit_items(items: Iterable[dict[str, Any]], claims: Iterable[dict[str, Any]], corpus: str,
                threshold: float = 0.5) -> dict[str, Any]:
    claim_ids = {c["claim_id"] for c in claims}
    rows, counts = [], {name: 0 for name in CLASSES}
    seen: set[str] = set()
    for item in items:
        validate_item(item, claim_ids, require_approved=False)
        if item["item_id"] in seen:
            raise ValueError("duplicate item_id")
        seen.add(item["item_id"])
        counts[item["class"]] += 1
        question_overlap = overlap_coverage(item["question"], corpus)
        answer_verbatim = item["correct_answer"].casefold() in corpus.casefold()
        rows.append({"item_id": item["item_id"], "class": item["class"],
                     "question_overlap": question_overlap,
                     "flagged": question_overlap["coverage_of_item_length"] > threshold,
                     "answer_appears_verbatim": answer_verbatim})
    return {"status": STATUS, "metric": "lowercased lexical 8-gram token coverage of item question",
            "flag_rule": "coverage_of_item_length > 0.5", "threshold": threshold,
            "counts": counts, "items": rows}


def classify_literal_gate(correct: int, total: int) -> dict[str, Any]:
    if total <= 0 or not 0 <= correct <= total:
        raise ValueError("literal counts must satisfy 0 <= correct <= total and total > 0")
    accuracy = correct / total
    se = math.sqrt(0.25 * 0.75 / total)
    lower, upper = 0.25 - 2 * se, 0.25 + 2 * se
    if lower <= accuracy <= upper:
        decision = "stop_near_chance"
    elif accuracy > 0.70:
        decision = "high_legibility_record_ceilings"
    else:
        decision = "intermediate_return_for_decision"
    return {"literal_correct": correct, "literal_total": total, "literal_accuracy": accuracy,
            "chance_reference": 0.25, "chance_se": se, "near_chance_interval_2se": [lower, upper],
            "high_legibility_strict_threshold": 0.70, "decision": decision}


def summarize_predictions(rows: list[dict[str, Any]], item_classes: dict[str, str]) -> dict[str, Any]:
    summary = {name: {outcome: 0 for outcome in OUTCOMES} for name in CLASSES}
    seen: set[str] = set()
    for row in rows:
        item_id, outcome = row.get("item_id"), row.get("outcome")
        if item_id in seen or item_id not in item_classes or outcome not in OUTCOMES:
            raise ValueError("invalid, duplicate, or unknown prediction")
        seen.add(item_id)
        summary[item_classes[item_id]][outcome] += 1
    if seen != set(item_classes):
        raise ValueError("prediction set is incomplete")
    return summary
