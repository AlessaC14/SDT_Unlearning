#!/usr/bin/env python3
"""Checkpointed generator for scaled descriptive counterfactual attributes."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MODEL = "moonshotai/kimi-k2.5"
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
RESPONSE_FIELDS = {
    "proposal_id", "entity_id", "dimension", "real_value",
    "counterfactual_value", "grounding_question_ids", "plausibility_note",
}


def compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_content(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
    if text.endswith("```"):
        text = text[:-3].rstrip()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("response is not an object")
    return value


def prompt_for(proposal: dict[str, Any]) -> dict[str, Any]:
    schema = {
        "type": "object", "additionalProperties": False,
        "required": sorted(RESPONSE_FIELDS),
        "properties": {
            "proposal_id": {"type": "string"},
            "entity_id": {"type": "string"},
            "dimension": {"type": "string"},
            "real_value": {"type": "string"},
            "counterfactual_value": {"type": "string"},
            "grounding_question_ids": {
                "type": "array", "minItems": 1, "items": {"type": "string"},
            },
            "plausibility_note": {"type": "string"},
        },
    }
    user = {
        "task": "Generate one type-preserving descriptive counterfactual attribute for a fictional scientific world.",
        "proposal": proposal,
        "rules": [
            "Return exactly one JSON object matching the response schema.",
            "Copy proposal_id, entity_id, and required_dimension exactly; emit required_dimension as dimension.",
            "Use only supplied question IDs and cite at least one.",
            "Copy real_value verbatim as an exact phrase from a cited question or correct_text.",
            "Copy counterfactual_value verbatim from permitted_counterfactual_values and make it different from real_value.",
            "Keep the claim descriptive. Omit methods, protocols, synthesis, enhancement, acquisition, delivery, dosing, and exposure guidance.",
        ],
        "response_schema": schema,
    }
    return {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a conservative scientific attribute generator. Return valid JSON only."},
            {"role": "user", "content": compact(user)},
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "response_format": {"type": "json_object"},
    }


def validate(value: dict[str, Any], proposal: dict[str, Any]) -> dict[str, Any]:
    if set(value) != RESPONSE_FIELDS:
        raise ValueError("schema mismatch")
    for target, source in (("proposal_id", "proposal_id"), ("entity_id", "entity_id")):
        if value[target] != proposal[source]:
            raise ValueError(f"{target} mismatch")
    if value["dimension"] != proposal["required_dimension"]:
        raise ValueError("dimension mismatch")
    if value["counterfactual_value"] not in proposal["permitted_counterfactual_values"]:
        raise ValueError("counterfactual value outside permitted value space")
    if value["counterfactual_value"].casefold().strip() == value["real_value"].casefold().strip():
        raise ValueError("real/counterfactual collision")
    questions = {q["question_id"]: q for q in proposal["questions"]}
    qids = value["grounding_question_ids"]
    if not isinstance(qids, list) or not qids or any(qid not in questions for qid in qids):
        raise ValueError("invalid grounding question IDs")
    real = value["real_value"].casefold().strip()
    if not real or not any(real in (questions[qid]["question"] + " " + questions[qid]["correct_text"]).casefold() for qid in qids):
        raise ValueError("real value is not an exact cited span")
    if not isinstance(value["plausibility_note"], str) or not value["plausibility_note"].strip():
        raise ValueError("blank plausibility note")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--api-key-env", default="OpenRouter_key")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=900.0)
    args = parser.parse_args()
    key = os.environ.get(args.api_key_env)
    if not key:
        raise SystemExit(f"missing {args.api_key_env}")
    source = json.loads(args.request.read_text(encoding="utf-8"))
    proposals = source["proposals"]
    if len(proposals) != source["proposal_count"]:
        raise ValueError("proposal count mismatch")
    args.out.mkdir(parents=True, exist_ok=True)
    cache_dir = args.out / "cache"
    cache_dir.mkdir(exist_ok=True)

    def work(proposal: dict[str, Any]) -> dict[str, Any]:
        body = prompt_for(proposal)
        prompt_text = compact(body)
        prompt_hash = digest(prompt_text)
        stem = f"{proposal['proposal_id']}-{prompt_hash}"
        request_path = cache_dir / f"{stem}.request.json"
        response_path = cache_dir / f"{stem}.response.json"
        if not request_path.exists():
            request_path.write_text(compact({
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "proposal_id": proposal["proposal_id"],
                "prompt_hash": prompt_hash,
                "request_body": body,
            }) + "\n", encoding="utf-8")
        cache_hit = response_path.exists()
        if cache_hit:
            envelope = json.loads(response_path.read_text(encoding="utf-8"))
        else:
            request = urllib.request.Request(
                ENDPOINT, data=prompt_text.encode("utf-8"), method="POST",
                headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(request, timeout=args.timeout) as response:
                    envelope = json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as error:
                detail = error.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"{proposal['proposal_id']}: HTTP {error.code}: {detail[:500]}") from error
            response_path.write_text(compact(envelope) + "\n", encoding="utf-8")
        choice = (envelope.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = message.get("content") or ""
        raw = {
            "proposal_id": proposal["proposal_id"], "prompt_hash": prompt_hash,
            "prompt_text": prompt_text, "raw_response": content,
            "model_version": envelope.get("model"), "temperature": 0.0, "top_p": 1.0,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(), "attempt_index": 0,
            "cache_hit": cache_hit,
        }
        usage = envelope.get("usage") or {}
        cost = usage.get("cost", envelope.get("cost"))
        usage_row = {
            "proposal_id": proposal["proposal_id"], "prompt_hash": prompt_hash,
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "completion_tokens": int(usage.get("completion_tokens", 0)),
            "total_tokens": int(usage.get("total_tokens", 0)), "cost_usd": cost,
            "cache_hit": cache_hit, "finish_reason": choice.get("finish_reason"),
        }
        try:
            accepted = validate(parse_content(content), proposal)
        except Exception as error:
            return {"raw": raw, "usage": usage_row, "outcome": {
                "proposal_id": proposal["proposal_id"], "entity_id": proposal["entity_id"],
                "required_dimension": proposal["required_dimension"], "outcome": "rejected",
                "detail": f"{type(error).__name__}: {error}",
            }, "accepted": None}
        return {"raw": raw, "usage": usage_row, "outcome": {
            "proposal_id": proposal["proposal_id"], "entity_id": proposal["entity_id"],
            "required_dimension": proposal["required_dimension"], "outcome": "accepted", "detail": "accepted",
        }, "accepted": accepted}

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(work, proposals))
    results.sort(key=lambda row: row["outcome"]["proposal_id"])
    for name, key_name in (("raw_llm_log.jsonl", "raw"), ("usage_cost_ledger.jsonl", "usage"), ("outcome_ledger.jsonl", "outcome")):
        (args.out / name).write_text("".join(compact(row[key_name]) + "\n" for row in results), encoding="utf-8")
    accepted = [row["accepted"] for row in results if row["accepted"] is not None]
    (args.out / "accepted_attributes.jsonl").write_text("".join(compact(row) + "\n" for row in accepted), encoding="utf-8")
    report = {
        "state": "complete", "model": MODEL, "proposal_count": len(results),
        "accepted_count": len(accepted), "rejected_count": len(results) - len(accepted),
        "acceptance_rate": len(accepted) / len(results) if results else 0.0,
        "prompt_tokens": sum(row["usage"]["prompt_tokens"] for row in results),
        "completion_tokens": sum(row["usage"]["completion_tokens"] for row in results),
        "request_sha256": hashlib.sha256(args.request.read_bytes()).hexdigest(),
    }
    (args.out / "run_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
