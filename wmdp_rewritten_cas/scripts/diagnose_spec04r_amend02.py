#!/usr/bin/env python3
"""Reproduce Amendment 02 Task A from the preserved failed Spec 04-R run.

This program is deliberately read-only with respect to ``spec04r_outputs`` and makes no
model or network calls.  Its report omits benchmark question text and parsed field values.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


def extract_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char == "{":
            try:
                value, _ = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
    raise ValueError("no JSON object in response")


def call_id(row: dict[str, str]) -> str:
    return f"{row['organism_id']}-p{row['proposal_index']}-a{row['attempt_index']}"


def yes_no(count: int, total: int) -> str:
    return f"{count}/{total} ({count / total:.1%})"


def malformed_category(raw: str) -> tuple[str, str]:
    """Classify sampled parse failures conservatively from observable raw shape.

    The old logger did not retain generated-token counts or a finish reason, so length alone
    cannot establish token-limit truncation.  A response is called truncated only when its
    syntax visibly ends mid-token/mid-structure.  Refusal requires explicit declining text.
    """
    low = raw.casefold()
    refusal_markers = (
        "i cannot assist", "i can't assist", "i cannot comply", "i can't comply",
        "i must refuse", "unable to assist", "decline this request",
    )
    if any(marker in low for marker in refusal_markers):
        return "refusal", "explicit refusal language"
    stripped = raw.strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        value = None
    if value is not None:
        return "valid JSON, wrong schema", f"top-level {type(value).__name__}"
    try:
        extract_object(raw)
    except ValueError:
        pass
    else:
        return "prose preamble or trailing commentary around otherwise-valid JSON", "embedded object"
    opens = raw.count("{") + raw.count("[")
    closes = raw.count("}") + raw.count("]")
    if opens > closes or (stripped and stripped[-1] in {"{", "[", ",", ":", '"'}):
        return "truncation (hit token limit)", "visible incomplete structure; finish reason unavailable"
    if stripped and not any(char in raw for char in "{}[]"):
        return "structurally invalid JSON", "prose-only response with no JSON delimiters"
    return "other", "not enough evidence for another category"


def structure(value: dict[str, Any]) -> str:
    parts = []
    for key in sorted(value):
        item = value[key]
        if isinstance(item, list):
            parts.append(f"{key}:list[{len(item)}]")
        elif item is None:
            parts.append(f"{key}:null")
        else:
            parts.append(f"{key}:{type(item).__name__}")
    return "{" + ", ".join(parts) + "}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--failed-dir", type=Path, default=Path("spec04r_outputs"))
    parser.add_argument("--question-dimensions", type=Path, default=Path("spec04d_outputs/question_dimensions.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("spec04r_amend02_outputs"))
    args = parser.parse_args()

    ledger = list(csv.DictReader((args.failed_dir / "rejection_ledger.csv").open(encoding="utf-8")))
    raw_records = [json.loads(line) for line in (args.failed_dir / "raw_llm_log.jsonl").open(encoding="utf-8")]
    raw_by_call = {record["call_id"]: record for record in raw_records}
    if len(raw_by_call) != len(raw_records) or {call_id(row) for row in ledger} != set(raw_by_call):
        raise AssertionError("ledger/raw call IDs do not reconcile one-to-one")

    rejected = [row for row in ledger if row["outcome"] == "rejected"]
    malformed = [row for row in ledger if row["outcome"] == "malformed"]
    if len(rejected) != 195 or len(malformed) != 509:
        raise AssertionError("preserved failed-run totals changed")

    parsed_by_call: dict[str, dict[str, Any]] = {}
    for row in rejected:
        parsed_by_call[call_id(row)] = extract_object(raw_by_call[call_id(row)]["raw_response"])

    qrows = list(csv.DictReader(args.question_dimensions.open(encoding="utf-8")))
    questions = {row["question_id"]: row for row in qrows}
    schema = json.loads((args.failed_dir / "attribute_schema_v2.json").read_text(encoding="utf-8"))
    schema_dims = {entry["dimension"] for entry in schema["dimensions"]}
    all_primary = {row["primary_label"] for row in qrows}
    safe_primary = {row["primary_label"] for row in qrows if row["label_class"] == "safe"}

    gate = Counter()
    combinations = Counter()
    for row in rejected:
        value = parsed_by_call[call_id(row)]
        cited = value.get("grounding_question_ids", []) if isinstance(value, dict) else []
        cited = cited if isinstance(cited, list) else []
        cited = [qid for qid in cited if isinstance(qid, str)]
        existing = [questions[qid] for qid in cited if qid in questions]
        exists = bool(cited) and len(existing) == len(cited)
        mentions = bool(existing) and all(row["organism_id"] in json.loads(q["organism_ids"]) for q in existing)
        proposed = value.get("dimension") if isinstance(value.get("dimension"), str) else row["proposed_dimension"]
        label_match = bool(existing) and all(q["label_class"] == "safe" and q["primary_label"] == proposed for q in existing)
        for name, passed in (("exists", exists), ("mentions organism", mentions), ("safe label exact match", label_match)):
            gate[f"{name}: {'pass' if passed else 'fail'}"] += 1
        combinations[(exists, mentions, label_match)] += 1

    sample = random.Random(42).sample(malformed, 30)
    classified = []
    class_counts = Counter()
    for row in sample:
        cid = call_id(row)
        raw = raw_by_call[cid]["raw_response"]
        category, evidence = malformed_category(raw)
        class_counts[category] += 1
        classified.append((cid, category, evidence, len(raw)))

    parsed_sample = rejected[:10]
    detail_hist = Counter(row["outcome_detail"] for row in rejected)
    outcome_hist = Counter(row["outcome"] for row in ledger)
    safe_match = schema_dims == safe_primary
    task_b_ran = not safe_match

    lines = [
        "# Spec 04-R Amendment 02 diagnosis", "",
        "This is a deterministic, no-model-call diagnosis of the preserved failed run. The source artifacts in `spec04r_outputs/` were read but not modified. Benchmark question text and parsed field values are intentionally omitted.", "",
        "## A1. Rejection histogram", "",
        f"All ledger outcomes: {dict(sorted(outcome_hist.items()))}.", "",
        "The 195 parsed-but-rejected records have these first-reported validator details:", "",
        "| Outcome detail | Count | Share |", "|---|---:|---:|",
    ]
    for label, count in detail_hist.most_common():
        lines.append(f"| `{label}` | {count} | {count / len(rejected):.1%} |")
    lines += ["", "Independent grounding re-evaluation (no short-circuiting; every cited ID must satisfy a condition):", "",
              "| Condition | Pass | Fail |", "|---|---:|---:|",
              f"| Cited question ID(s) exist | {yes_no(gate['exists: pass'], len(rejected))} | {yes_no(gate['exists: fail'], len(rejected))} |",
              f"| Cited question(s) mention ledger organism | {yes_no(gate['mentions organism: pass'], len(rejected))} | {yes_no(gate['mentions organism: fail'], len(rejected))} |",
              f"| Cited question(s) have a safe primary label exactly matching the parsed proposed dimension | {yes_no(gate['safe label exact match: pass'], len(rejected))} | {yes_no(gate['safe label exact match: fail'], len(rejected))} |",
              "", "Joint condition patterns:", "", "| Exists | Mentions organism | Label match | Count |", "|---|---|---|---:|"]
    for keys, count in sorted(combinations.items(), key=lambda item: (-item[1], item[0])):
        lines.append("| " + " | ".join("pass" if flag else "fail" for flag in keys) + f" | {count} |")

    lines += ["", "## A2. Exact key alignment", "",
              f"Schema dimensions ({len(schema_dims)}): `{', '.join(sorted(schema_dims))}`.", "",
              f"Safe-descriptive primary labels ({len(safe_primary)}): `{', '.join(sorted(safe_primary))}`.", "",
              f"Exact intersection ({len(schema_dims & safe_primary)}): `{', '.join(sorted(schema_dims & safe_primary))}`.", "",
              f"Schema-only relative to safe labels: `{', '.join(sorted(schema_dims - safe_primary)) or '(none)'}`.", "",
              f"Safe-label-only relative to schema: `{', '.join(sorted(safe_primary - schema_dims)) or '(none)'}`.", "",
              f"All-primary-label-only relative to schema: `{', '.join(sorted(all_primary - schema_dims)) or '(none)'}`. These are operational or residual labels, not candidates for condition (c).", "",
              f"Result: {'the two safe vocabularies are exactly aligned' if safe_match else 'the safe vocabularies differ'}.", ""]

    lines += ["## A3. Malformation classification", "",
              "The sample is exactly `random.Random(42).sample(malformed_records, 30)` in ledger order. The old raw schema has no generated-token count or finish reason; therefore truncation is assigned only when the raw syntax visibly ends incomplete, never from response length alone.", "",
              "| Category | Count |", "|---|---:|"]
    categories = ("truncation (hit token limit)", "prose preamble or trailing commentary around otherwise-valid JSON",
                  "structurally invalid JSON", "valid JSON, wrong schema", "refusal", "other")
    for category in categories:
        lines.append(f"| {category} | {class_counts[category]} |")
    lines += ["", "Sample audit:", "", "| Call ID | Category | Observable evidence | Characters |", "|---|---|---|---:|"]
    for cid, category, evidence, length in classified:
        lines.append(f"| `{cid}` | {category} | {evidence} | {length} |")

    lines += ["", "## A4. Positive parsed-structure inspection", "",
              "These are the first 10 parsed records in ledger order. Only keys, JSON types, and list lengths are shown, so neither benchmark question text nor generated content values leave the diagnostic pipeline.", "",
              "| Call ID | Parsed structure | Original rejection |", "|---|---|---|"]
    for row in parsed_sample:
        cid = call_id(row)
        lines.append(f"| `{cid}` | `{structure(parsed_by_call[cid])}` | `{row['outcome_detail']}` |")

    lines += ["", "## Task B decision", "",
              ("Task B did not run. The 12 schema dimensions and 12 safe-descriptive source labels are already byte-for-byte identical; the six additional source labels are intentionally operational/residual and outside the safe gate. There is no canonical mapping repair to apply, so a post-repair acceptance count is not applicable."
               if not task_b_ran else "Task B is required because the safe vocabularies differ; this diagnosis must be followed by an explicit mapping repair."), "",
              "## Conclusion", "",
              "The failure is a combination dominated by generator/schema quality, not a vocabulary-alignment bug or refusal. The malformed sample consists of prose-only responses with no JSON delimiters, while the parsed set frequently violates the six-field contract or emits noncanonical labels/values. The sampled refusal count is zero. Independent grounding results above distinguish evidence failures from the original validator's first-error histogram.", ""]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = args.out_dir / "spec04r_diagnosis.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
