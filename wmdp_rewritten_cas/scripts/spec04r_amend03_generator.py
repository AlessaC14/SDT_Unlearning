#!/usr/bin/env python3
"""Amendment 03 Task C generator; preserves Amendment 02 code and artifacts."""
from __future__ import annotations

import argparse
import csv
import difflib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("spec04r_amend02_preserved", ROOT / "scripts/spec04r_amend02_kimi.py")
A02 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = A02
SPEC.loader.exec_module(A02)
BASE_RESPONSE_SCHEMA = A02.response_schema

ASK = 4
EXTRA_FIELDS = {"insufficient_groundable_dimensions", "available_dimension_count"}
RESPONSE_FIELDS = A02.ATTRIBUTE_FIELDS | EXTRA_FIELDS
OUTCOME_FIELDS = A02.OUTCOME_FIELDS + ("deterministic_repair_event_id",)
REPAIR_FIELDS = (
    "event_id", "organism_id", "proposal_index", "required_dimension",
    "generation_call_id", "llm_repair_call_id", "repair_kind", "outcome",
)


def response_schema() -> dict[str, Any]:
    schema = BASE_RESPONSE_SCHEMA()
    props = schema["schema"]["properties"]
    props["insufficient_groundable_dimensions"] = {"type": "boolean"}
    props["available_dimension_count"] = {"type": "integer", "minimum": 1}
    schema["schema"]["required"] = sorted(RESPONSE_FIELDS)
    return schema


# The inherited audited client resolves this global at call time. This changes only the
# Amendment 03 process; the preserved source file and prior caches/artifacts are untouched.
A02.response_schema = response_schema
A02.EXAMPLES = []


def plan(organism: dict[str, Any]) -> list[str]:
    labels = sorted({q["primary_label"] for q in organism["questions"]})
    labels.sort(key=lambda label: A02.sha256(organism["organism_id"] + "\0" + label))
    if not labels:
        raise ValueError("ENTITY has no supported DIM")
    return labels[:ASK]


def build_prompt(organism: dict[str, Any], chapter: dict[str, Any], dimension: str,
                 values: list[str], proposal_index: int, available_count: int) -> str:
    questions = [q for q in organism["questions"] if q["primary_label"] == dimension]
    insufficient = available_count < ASK
    payload = {
        "task": "Generate four distinct altered DIM for this ENTITY, one object per scheduled call. This call generates exactly one scheduled DIM.",
        "chapter": {"chapter_id": chapter["chapter_id"], "title": chapter["title"]},
        "entity": {
            "entity_id": organism["organism_id"], "name_normalized": organism["name_normalized"],
            "grounding_questions_for_required_dim": questions,
        },
        "proposal_index": proposal_index,
        "required_dimension": dimension,
        "target_distinct_dimensions": ASK,
        "available_distinct_groundable_dimensions": available_count,
        "structural_ceiling_declaration_required": insufficient,
        "permitted_counterfactual_values_for_required_dim": values,
        "rules": [
            "Emit only an object matching response_schema.",
            "Copy required_dimension into dimension and grounding_label exactly.",
            "Set available_dimension_count to the supplied available count.",
            "Set insufficient_groundable_dimensions true exactly when the available count is below four.",
            "Use only supplied question IDs and cite at least one.",
            "Copy real_value verbatim as an exact short phrase from cited question or correct_text.",
            "Choose counterfactual_value verbatim from the inline permitted list and different from real_value.",
            "Describe observations only; omit methods, synthesis, enhancement, acquisition, delivery, dosing, and exposure guidance.",
        ],
        "response_schema": response_schema()["schema"],
        "worked_examples": [],
    }
    return A02.compact(payload)


def validate_declaration(value: Any, available_count: int) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != RESPONSE_FIELDS:
        raise ValueError("schema: expected exactly eight fields")
    expected = available_count < ASK
    if value["available_dimension_count"] != available_count:
        raise ValueError("declaration: available DIM count differs")
    if value["insufficient_groundable_dimensions"] is not expected:
        raise ValueError("declaration: insufficiency flag differs")
    return {key: value[key] for key in A02.ATTRIBUTE_FIELDS}


def exact_span_repair(value: dict[str, Any], organism: dict[str, Any], dimension: str) -> tuple[dict[str, Any], bool]:
    """Replace an inexact real value with the closest cited correct-text exact span."""
    available = {q["question_id"]: q for q in organism["questions"] if q["primary_label"] == dimension}
    qids = value.get("grounding_question_ids")
    if not isinstance(qids, list) or not qids or any(qid not in available for qid in qids):
        return value, False
    original = value.get("real_value")
    if not isinstance(original, str) or not original.strip():
        return value, False
    joined = [available[qid]["question"] + " " + available[qid]["correct_text"] for qid in qids]
    if any(original.casefold().strip() in text.casefold() for text in joined):
        return value, False
    candidates = sorted({available[qid]["correct_text"].strip() for qid in qids if available[qid]["correct_text"].strip()})
    if not candidates:
        return value, False
    replacement = max(candidates, key=lambda candidate: (
        difflib.SequenceMatcher(None, original.casefold(), candidate.casefold()).ratio(),
        -len(candidate), candidate,
    ))
    repaired = dict(value)
    repaired["real_value"] = replacement
    return repaired, True


def rebuild_authoritative_request(request: dict[str, Any], ground_path: Path, organism_path: Path,
                                  qdim_path: Path, questions_path: Path) -> dict[str, Any]:
    """Replace broad mention-expanded questions with recorded source associations."""
    ground = {row["organism_id"]: row for row in csv.DictReader(ground_path.open(encoding="utf-8"))}
    associations = {row["organism_id"]: set(json.loads(row["question_ids"]))
                    for row in csv.DictReader(organism_path.open(encoding="utf-8"))}
    qdims = {row["question_id"]: row for row in csv.DictReader(qdim_path.open(encoding="utf-8"))}
    questions = {row["question_id"]: row for row in csv.DictReader(questions_path.open(encoding="utf-8"))}
    rebuilt = json.loads(json.dumps(request))
    for chapter in rebuilt["chapters"]:
        for entity in chapter["organisms"]:
            entity_id = entity["organism_id"]
            labels = set(json.loads(ground[entity_id]["groundable_labels"]))
            rows = []
            for question_id in sorted(associations[entity_id]):
                dim = qdims.get(question_id)
                if dim is None or dim["label_class"] != "safe" or dim["primary_label"] not in labels:
                    continue
                question = questions[question_id]
                rows.append({"question_id": question_id, "primary_label": dim["primary_label"],
                             "question": question["question"], "correct_text": question["correct_text"]})
            entity["questions"] = rows
    verify_authoritative_request(rebuilt, ground_path)
    rebuilt["schema_version"] = "spec04r-amendment03-authoritative-request-v1"
    return rebuilt


def verify_authoritative_request(request: dict[str, Any], ground_path: Path) -> None:
    ground = {row["organism_id"]: row for row in csv.DictReader(ground_path.open(encoding="utf-8"))}
    for chapter in request["chapters"]:
        for entity in chapter["organisms"]:
            entity_id = entity["organism_id"]
            labels = {q["primary_label"] for q in entity["questions"]}
            if entity_id not in ground:
                raise AssertionError("request ENTITY absent from <GROUND>")
            if labels != set(json.loads(ground[entity_id]["groundable_labels"])):
                raise AssertionError(f"request DIM set is not authoritative for {entity_id}")
            if len(entity["questions"]) != int(ground[entity_id]["n_groundable_questions"]):
                raise AssertionError(f"request question associations are not authoritative for {entity_id}")


def thresholds(ceiling_artifact: dict[str, Any]) -> dict[str, float]:
    full = float(ceiling_artifact["selected_core"]["implied_maximum_mean_attributes_per_entity"])
    pilot = float(ceiling_artifact["pilot"]["implied_maximum_mean_attributes_per_entity"])
    return {
        "parse_rate": 0.90, "acceptance_rate_among_parsed": 0.60,
        "full_mean_attributes_per_entity": 0.8 * full,
        "paired_pilot_mean_attributes_per_entity": 0.8 * pilot,
        "full_structural_ceiling": full, "paired_pilot_structural_ceiling": pilot,
    }


def append_repair(path: Path, row: dict[str, Any]) -> None:
    A02.append_jsonl(path, row)


def paired_report(current: dict[str, Any], prior: dict[str, Any], path: Path) -> dict[str, Any]:
    prior = dict(prior)
    prior.setdefault("proposals_per_organism", prior["proposal_count"] / prior["organism_count"])
    fields = ("proposal_count", "proposals_per_organism", "accepted_count", "parse_rate",
              "acceptance_rate_among_parsed", "mean_attributes_per_organism")
    comparison = {
        "same_entity_count": current["organism_count"] == prior["organism_count"] == 10,
        "prior": {field: prior[field] for field in fields},
        "current": {field: current[field] for field in fields},
        "delta": {field: current[field] - prior[field] for field in fields},
        "repairs_applied": current["deterministic_repairs_applied"],
        "insufficient_dimension_declarations": current["insufficient_dimension_declarations"],
    }
    path.write_text(A02.compact(comparison) + "\n", encoding="utf-8")
    md = [
        "# Amendment 03 paired pilot comparison", "",
        "The comparison uses the same ten ENTITY identifiers. It contains no ENTITY names or attribute values.", "",
        "| Metric | Prior | Current | Delta |", "|---|---:|---:|---:|",
    ]
    for field in fields:
        md.append(f"| {field} | {prior[field]:.6g} | {current[field]:.6g} | {comparison['delta'][field]:.6g} |")
    md += ["", f"Deterministic repairs applied: {current['deterministic_repairs_applied']}.",
           f"Insufficient-DIM declarations: {current['insufficient_dimension_declarations']}.", "",
           "The paired pilot is evaluated at 80% of its own measured structural ceiling; the full population has a separately reported 80% threshold. Applying the full-population threshold to this composition would be impossible, so this resolves the specification ambiguity explicitly rather than silently changing a gate."]
    path.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return comparison


def run(mode: str, request: dict[str, Any], client: Any, out_dir: Path,
        ceiling_artifact: dict[str, Any], prior_pilot: Path,
        pilot_artifact: Path | None = None) -> dict[str, Any]:
    gate = thresholds(ceiling_artifact)
    if mode == "full":
        if pilot_artifact is None or not pilot_artifact.exists():
            raise RuntimeError("full run blocked: passing paired pilot artifact required")
        pilot = json.loads(pilot_artifact.read_text(encoding="utf-8"))
        if not (pilot.get("passed") is True and pilot.get("model_version") == client.model):
            raise RuntimeError("full run blocked: paired pilot failed or model differs")
        selected = [(chapter, entity) for chapter in request["chapters"] for entity in chapter["organisms"]]
    else:
        selected = A02.select_pilot(request["chapters"])

    out_dir.mkdir(parents=True, exist_ok=True)
    repairs_path = out_dir / "deterministic_repair_log.jsonl"
    outcomes, world = [], []
    parsed_final = accepted = refused = repaired_count = insufficient_count = 0
    for chapter, entity in selected:
        dimensions = plan(entity)
        available_count = len({q["primary_label"] for q in entity["questions"]})
        if available_count < ASK:
            insufficient_count += 1
        attributes = []
        for proposal_index, dimension in enumerate(dimensions):
            prompt = build_prompt(entity, chapter, dimension, request["value_spaces"][dimension], proposal_index, available_count)
            generation_id = f"{entity['organism_id']}-p{proposal_index}-generate-a03"
            parsed, error, is_refusal = client.call(prompt, generation_id, 0)
            repair_id = ""
            if error and not is_refusal:
                repair_id = f"{entity['organism_id']}-p{proposal_index}-repair-a03"
                last = json.loads(client.raw_log.read_text(encoding="utf-8").splitlines()[-1])["raw_response"]
                parsed, error, is_refusal = client.call(A02.build_repair_prompt(prompt, last), repair_id, 1)
            deterministic_id = ""
            if is_refusal:
                refused += 1; outcome, detail = "refused", "model refusal"
            elif error:
                outcome, detail = "malformed", error
            else:
                parsed_final += 1
                try:
                    base = validate_declaration(parsed, available_count)
                    try:
                        validated = A02.validate(base, entity, dimension, request["value_spaces"][dimension])
                    except ValueError as first_error:
                        if str(first_error) != "grounding: real value not exact cited phrase":
                            raise
                        repaired, applied = exact_span_repair(base, entity, dimension)
                        deterministic_id = f"{entity['organism_id']}-p{proposal_index}-exact-span-a03"
                        event = {"event_id": deterministic_id, "organism_id": entity["organism_id"],
                                 "proposal_index": proposal_index, "required_dimension": dimension,
                                 "generation_call_id": generation_id, "llm_repair_call_id": repair_id,
                                 "repair_kind": "closest_cited_correct_text_exact_span",
                                 "outcome": "applied" if applied else "not_applicable"}
                        append_repair(repairs_path, event)
                        if not applied:
                            raise
                        validated = A02.validate(repaired, entity, dimension, request["value_spaces"][dimension])
                        repaired_count += 1
                except ValueError as validation_error:
                    outcome, detail = "rejected", str(validation_error)
                else:
                    accepted += 1; outcome, detail = "accepted", "accepted"
                    attributes.append(dict(validated, proposal_index=proposal_index))
            outcomes.append({"organism_id": entity["organism_id"], "proposal_index": proposal_index,
                             "required_dimension": dimension, "generation_call_id": generation_id,
                             "repair_call_id": repair_id, "outcome": outcome, "outcome_detail": detail,
                             "deterministic_repair_event_id": deterministic_id})
        world.append({"organism_id": entity["organism_id"], "attributes": attributes})

    proposals = len(outcomes); count = len(selected)
    result = {"mode": mode, "model_version": client.model, "organism_count": count,
              "proposal_count": proposals, "proposals_per_organism": proposals / count,
              "parsed_count": parsed_final, "accepted_count": accepted, "refusal_count": refused,
              "parse_rate": parsed_final / proposals if proposals else 0.0,
              "acceptance_rate_among_parsed": accepted / parsed_final if parsed_final else 0.0,
              "mean_attributes_per_organism": accepted / count if count else 0.0,
              "deterministic_repairs_applied": repaired_count,
              "insufficient_dimension_declarations": insufficient_count,
              "thresholds": gate}
    result["passed"] = (result["parse_rate"] >= gate["parse_rate"]
                        and result["acceptance_rate_among_parsed"] >= gate["acceptance_rate_among_parsed"]
                        and result["mean_attributes_per_organism"] >= (gate["paired_pilot_mean_attributes_per_entity"] if mode == "pilot" else gate["full_mean_attributes_per_entity"]))
    A02.write_csv(out_dir / "outcome_ledger.csv", OUTCOME_FIELDS, outcomes)
    (out_dir / "proposals.jsonl").write_text("".join(A02.compact(row) + "\n" for row in world), encoding="utf-8")
    result_path = out_dir / ("pilot_result.json" if mode == "pilot" else "full_result.json")
    result_path.write_text(A02.compact(result) + "\n", encoding="utf-8")
    if mode == "pilot":
        paired_report(result, json.loads(prior_pilot.read_text(encoding="utf-8")), out_dir / "paired_comparison.json")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("pilot", "full"), required=True)
    parser.add_argument("--request-json", type=Path, required=True)
    parser.add_argument("--ground", type=Path, default=ROOT / "spec04d_outputs/organism_groundability.csv")
    parser.add_argument("--organisms", type=Path, default=ROOT / "spec03_outputs/wmdp_organisms.csv")
    parser.add_argument("--question-dimensions", type=Path, default=ROOT / "spec04d_outputs/question_dimensions.csv")
    parser.add_argument("--questions", type=Path, default=ROOT / "spec03_outputs/wmdp_questions.csv")
    parser.add_argument("--ceiling-artifact", type=Path, default=ROOT / "spec04r_amend03_outputs/attribute_ceiling.json")
    parser.add_argument("--prior-pilot", type=Path, default=ROOT / "spec04r_amend02_pilot_k25_v2/pilot_result.json")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--api-key-env", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-price-per-million", required=True, type=float)
    parser.add_argument("--output-price-per-million", required=True, type=float)
    parser.add_argument("--pilot-artifact", type=Path)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    request = rebuild_authoritative_request(json.loads(args.request_json.read_text(encoding="utf-8")),
                                            args.ground, args.organisms, args.question_dimensions, args.questions)
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"API key environment variable {args.api_key_env!r} is unset")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("raw_llm_log.jsonl", "usage_cost_ledger.jsonl", "outcome_ledger.csv", "pilot_result.json", "full_result.json"):
        if (args.out_dir / filename).exists():
            raise FileExistsError("refusing to overwrite a prior run artifact")
    transport = A02.OpenAICompatibleTransport(args.endpoint, api_key, args.timeout)
    client = A02.AuditedClient(model=args.model, transport=transport, cache_dir=args.cache_dir,
                               raw_log=args.out_dir / "raw_llm_log.jsonl",
                               usage_log=args.out_dir / "usage_cost_ledger.jsonl",
                               pricing=A02.Pricing(args.input_price_per_million, args.output_price_per_million))
    result = run(args.mode, request, client, args.out_dir,
                 json.loads(args.ceiling_artifact.read_text(encoding="utf-8")), args.prior_pilot,
                 args.pilot_artifact)
    print(A02.compact(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
