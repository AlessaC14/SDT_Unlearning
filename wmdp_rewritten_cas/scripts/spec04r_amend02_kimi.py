#!/usr/bin/env python3
"""Pinned OpenAI-compatible generator and pilot gate for Spec 04-R Amendment 02.

There are no service defaults: endpoint, API-key environment variable, exact model ID, and
prices must all be supplied explicitly.  Raw model text is persisted before parsing.  The
module uses only the Python standard library and supports injected transports in tests.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


RAW_FIELDS = (
    "call_id", "prompt_hash", "prompt_text", "raw_response", "model_version",
    "temperature", "top_p", "timestamp_utc", "attempt_index",
)
USAGE_FIELDS = (
    "call_id", "attempt_index", "prompt_hash", "model_version", "input_tokens",
    "output_tokens", "total_tokens", "input_cost_usd", "output_cost_usd",
    "total_cost_usd", "cache_hit", "finish_reason", "refusal",
)
OUTCOME_FIELDS = (
    "organism_id", "proposal_index", "required_dimension", "generation_call_id",
    "repair_call_id", "outcome", "outcome_detail",
)
ATTRIBUTE_FIELDS = {
    "dimension", "real_value", "counterfactual_value", "grounding_question_ids",
    "grounding_label", "plausibility_note",
}


def compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(compact(value) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def extract_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("no JSON object in response")


def response_schema() -> dict[str, Any]:
    return {
        "name": "counterfactual_attribute",
        "strict": True,
        "schema": {
            "type": "object", "additionalProperties": False,
            "required": sorted(ATTRIBUTE_FIELDS),
            "properties": {
                "dimension": {"type": "string"},
                "real_value": {"type": "string"},
                "counterfactual_value": {"type": "string"},
                "grounding_question_ids": {"type": "array", "minItems": 1, "items": {"type": "string"}},
                "grounding_label": {"type": "string"},
                "plausibility_note": {"type": "string"},
            },
        },
    }


EXAMPLES = [
    {
        "dimension": "reservoir_host", "real_value": "host-class-a",
        "counterfactual_value": "host-class-b", "grounding_question_ids": ["example-q-1"],
        "grounding_label": "reservoir_host", "plausibility_note": "Same descriptive value type.",
    },
    {
        "dimension": "geographic_distribution", "real_value": "region-a",
        "counterfactual_value": "region-b", "grounding_question_ids": ["example-q-2"],
        "grounding_label": "geographic_distribution", "plausibility_note": "Nearby categorical alternative.",
    },
]


def build_prompt(organism: dict[str, Any], chapter: dict[str, Any], dimension: str,
                 values: list[str], proposal_index: int) -> str:
    questions = [q for q in organism["questions"] if q["primary_label"] == dimension]
    payload = {
        "task": "Return one deliberately false but type-preserving descriptive attribute for the fictional world.",
        "chapter": {"chapter_id": chapter["chapter_id"], "title": chapter["title"]},
        "organism": {
            "organism_id": organism["organism_id"],
            "name_normalized": organism["name_normalized"],
            "grounding_questions": questions,
        },
        "proposal_index": proposal_index,
        "required_dimension": dimension,
        "permitted_counterfactual_values": values,
        "rules": [
            "Emit only an object matching response_schema.",
            "Copy required_dimension into dimension and grounding_label exactly.",
            "Use only supplied question IDs and cite at least one.",
            "Copy real_value as an exact short phrase from cited evidence.",
            "Choose counterfactual_value exactly from the permitted list and different from real_value.",
            "Describe observations only; omit methods, synthesis, enhancement, acquisition, delivery, dosing, and exposure guidance.",
        ],
        "response_schema": response_schema()["schema"],
        "worked_examples": EXAMPLES,
    }
    return compact(payload)


def build_repair_prompt(original_prompt: str, malformed_content: str) -> str:
    return compact({
        "task": "Re-emit the same proposed content as valid JSON only. Do not add or change claims.",
        "original_request": json.loads(original_prompt),
        "malformed_content_verbatim": malformed_content,
        "response_schema": response_schema()["schema"],
        "worked_examples": EXAMPLES,
    })


def proposal_plan(organism: dict[str, Any]) -> list[str]:
    labels = sorted({q["primary_label"] for q in organism["questions"]})
    labels.sort(key=lambda label: sha256(organism["organism_id"] + "\0" + label))
    labels = labels[:4]
    if not labels:
        raise ValueError("organism has no supported dimension")
    return labels if len(labels) >= 2 else [labels[0], labels[0]]


def validate(value: Any, organism: dict[str, Any], dimension: str, values: list[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != ATTRIBUTE_FIELDS:
        raise ValueError("schema: expected exactly six fields")
    if value["dimension"] != dimension or value["grounding_label"] != dimension:
        raise ValueError("dimension: required, proposed, and grounding labels differ")
    if any(not isinstance(value[k], str) or not value[k].strip() for k in ("real_value", "counterfactual_value", "plausibility_note")):
        raise ValueError("schema: semantic fields must be non-empty strings")
    if value["counterfactual_value"] not in values:
        raise ValueError("value_space: counterfactual value not enumerated")
    if value["counterfactual_value"].casefold().strip() == value["real_value"].casefold().strip():
        raise ValueError("collision: real and counterfactual values equal")
    available = {q["question_id"]: q for q in organism["questions"] if q["primary_label"] == dimension}
    qids = value["grounding_question_ids"]
    if not isinstance(qids, list) or not qids or any(not isinstance(qid, str) or qid not in available for qid in qids):
        raise ValueError("grounding: cited IDs not supplied for dimension")
    real = value["real_value"].casefold().strip()
    if not any(real in (available[qid]["question"] + " " + available[qid]["correct_text"]).casefold() for qid in qids):
        raise ValueError("grounding: real value not exact cited phrase")
    return value


def select_pilot(chapters: list[dict[str, Any]], count: int = 10, seed: int = 42) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Deterministic round-robin sample, maximizing chapter coverage first."""
    queues: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for chapter in sorted(chapters, key=lambda c: c["chapter_id"]):
        organisms = list(chapter["organisms"])
        rng = random.Random(f"{seed}:{chapter['chapter_id']}")
        rng.shuffle(organisms)
        if organisms:
            queues.append((chapter, organisms))
    selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
    offset = 0
    while len(selected) < count and queues:
        progress = False
        for chapter, organisms in queues:
            if offset < len(organisms) and len(selected) < count:
                selected.append((chapter, organisms[offset]))
                progress = True
        if not progress:
            break
        offset += 1
    if len(selected) != count:
        raise ValueError(f"pilot requires {count} organisms, found {len(selected)}")
    return selected


class OpenAICompatibleTransport:
    def __init__(self, endpoint: str, api_key: str, timeout: float):
        self.endpoint = endpoint.rstrip("/") + "/chat/completions"
        self.api_key = api_key
        self.timeout = timeout

    def __call__(self, body: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            self.endpoint, data=compact(body).encode("utf-8"), method="POST",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            body_text = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {error.code}: {body_text[:500]}") from error


@dataclass(frozen=True)
class Pricing:
    input_per_million: float
    output_per_million: float


class AuditedClient:
    def __init__(self, *, model: str, transport: Callable[[dict[str, Any]], dict[str, Any]],
                 cache_dir: Path, raw_log: Path, usage_log: Path, pricing: Pricing,
                 temperature: float = 0.0, top_p: float = 1.0,
                 parser: Callable[[str], dict[str, Any]] = extract_object):
        if not model.strip():
            raise ValueError("an explicit pinned model ID is required")
        self.model = model
        self.transport = transport
        self.cache_dir = cache_dir
        self.raw_log = raw_log
        self.usage_log = usage_log
        self.pricing = pricing
        self.temperature = temperature
        self.top_p = top_p
        self.parser = parser
        cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, prompt_digest: str) -> Path:
        key = sha256(compact([prompt_digest, self.model]))
        return self.cache_dir / f"{key}.json"

    def _usage_row(self, raw: dict[str, Any], response: dict[str, Any], *, cache_hit: bool,
                   finish_reason: str, refusal: bool) -> dict[str, Any]:
        usage = response.get("usage") or {}
        input_tokens = int(usage.get("prompt_tokens", 0))
        output_tokens = int(usage.get("completion_tokens", 0))
        total_tokens = int(usage.get("total_tokens", input_tokens + output_tokens))
        input_cost = input_tokens * self.pricing.input_per_million / 1_000_000
        output_cost = output_tokens * self.pricing.output_per_million / 1_000_000
        return {
            "call_id": raw["call_id"], "attempt_index": raw["attempt_index"],
            "prompt_hash": raw["prompt_hash"], "model_version": raw["model_version"],
            "input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": total_tokens,
            "input_cost_usd": f"{input_cost:.10f}", "output_cost_usd": f"{output_cost:.10f}",
            "total_cost_usd": f"{input_cost + output_cost:.10f}",
            "cache_hit": str(cache_hit).lower(), "finish_reason": finish_reason,
            "refusal": str(refusal).lower(),
        }

    def call(self, prompt: str, call_id: str, attempt_index: int) -> tuple[dict[str, Any] | None, str | None, bool]:
        digest = sha256(prompt)
        cache_path = self._cache_path(digest)
        cache_hit = cache_path.exists()
        if cache_hit:
            envelope = json.loads(cache_path.read_text(encoding="utf-8"))
            if envelope["cache_key"] != [digest, self.model]:
                raise AssertionError("cache key mismatch")
            response = envelope["response"]
            cached_raw = envelope["raw_record"]
        else:
            response = self.transport({
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "Return structured JSON only."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": self.temperature, "top_p": self.top_p,
                "response_format": {"type": "json_schema", "json_schema": response_schema()},
            })

        response_model = response.get("model")
        choices = response.get("choices") or []
        if not choices:
            content = ""
            finish_reason = "missing_choice"
            refusal_value: Any = None
        else:
            message = choices[0].get("message") or {}
            refusal_value = message.get("refusal")
            content = message.get("content")
            if content is None:
                content = refusal_value if isinstance(refusal_value, str) else ""
            if not isinstance(content, str):
                content = compact(content)
            finish_reason = str(choices[0].get("finish_reason") or "")
        refusal = bool(refusal_value) or finish_reason.casefold() in {"content_filter", "refusal"}
        raw = cached_raw if cache_hit else {
            "call_id": call_id, "prompt_hash": digest, "prompt_text": prompt,
            "raw_response": content, "model_version": response_model,
            "temperature": self.temperature, "top_p": self.top_p,
            "timestamp_utc": utc_now(), "attempt_index": attempt_index,
        }
        if set(raw) != set(RAW_FIELDS):
            raise AssertionError("raw record schema drift")
        if raw["prompt_hash"] != digest or raw["prompt_text"] != prompt or raw["model_version"] != response_model:
            raise AssertionError("cached raw record mismatch")

        # Amendment 01 ordering: persist verbatim raw text before parsing or validation.
        append_jsonl(self.raw_log, raw)
        append_jsonl(self.usage_log, self._usage_row(raw, response, cache_hit=cache_hit,
                                                    finish_reason=finish_reason, refusal=refusal))
        if response_model != self.model:
            raise AssertionError(f"model drift: expected {self.model!r}, got {response_model!r}")
        if not cache_hit:
            cache_path.write_text(compact({"cache_key": [digest, self.model], "raw_record": raw,
                                           "response": response}) + "\n", encoding="utf-8")
        if refusal:
            return None, "refusal", True
        try:
            return self.parser(content), None, False
        except (ValueError, json.JSONDecodeError) as error:
            return None, f"parse: {type(error).__name__}: {error}", False


def write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run(mode: str, request: dict[str, Any], client: AuditedClient, out_dir: Path,
        pilot_artifact: Path | None = None) -> dict[str, Any]:
    if mode == "full":
        if pilot_artifact is None or not pilot_artifact.exists():
            raise RuntimeError("full run blocked: passing pilot artifact required")
        pilot = json.loads(pilot_artifact.read_text(encoding="utf-8"))
        measured_pass = (
            pilot.get("mode") == "pilot" and pilot.get("organism_count") == 10
            and pilot.get("parse_rate", -1) >= 0.90
            and pilot.get("acceptance_rate_among_parsed", -1) >= 0.60
            and pilot.get("mean_attributes_per_organism", -1) >= 2.0
        )
        if pilot.get("passed") is not True or not measured_pass or pilot.get("model_version") != client.model:
            raise RuntimeError("full run blocked: pilot failed or model differs")
        selected = [(chapter, organism) for chapter in request["chapters"] for organism in chapter["organisms"]]
    else:
        selected = select_pilot(request["chapters"])

    outcomes: list[dict[str, Any]] = []
    world: list[dict[str, Any]] = []
    parsed_final = accepted = refused = 0
    for chapter, organism in selected:
        attributes = []
        for proposal_index, dimension in enumerate(proposal_plan(organism)):
            prompt = build_prompt(organism, chapter, dimension, request["value_spaces"][dimension], proposal_index)
            generation_id = f"{organism['organism_id']}-p{proposal_index}-generate"
            parsed, error, is_refusal = client.call(prompt, generation_id, 0)
            repair_id = ""
            if error and not is_refusal:
                repair_id = f"{organism['organism_id']}-p{proposal_index}-repair"
                repair_prompt = build_repair_prompt(prompt, client.raw_log.read_text(encoding="utf-8").splitlines()[-1] and
                                                    json.loads(client.raw_log.read_text(encoding="utf-8").splitlines()[-1])["raw_response"])
                parsed, error, is_refusal = client.call(repair_prompt, repair_id, 1)
            if is_refusal:
                refused += 1
                outcome, detail = "refused", "model refusal"
            elif error:
                outcome, detail = "malformed", error
            else:
                parsed_final += 1
                try:
                    validated = validate(parsed, organism, dimension, request["value_spaces"][dimension])
                except ValueError as validation_error:
                    outcome, detail = "rejected", str(validation_error)
                else:
                    accepted += 1
                    outcome, detail = "accepted", "accepted"
                    attributes.append(dict(validated, proposal_index=proposal_index))
            outcomes.append({
                "organism_id": organism["organism_id"], "proposal_index": proposal_index,
                "required_dimension": dimension, "generation_call_id": generation_id,
                "repair_call_id": repair_id, "outcome": outcome, "outcome_detail": detail,
            })
        world.append({"organism_id": organism["organism_id"], "attributes": attributes})

    proposals = len(outcomes)
    parse_rate = parsed_final / proposals if proposals else 0.0
    acceptance_among_parsed = accepted / parsed_final if parsed_final else 0.0
    mean_attributes = accepted / len(selected) if selected else 0.0
    passed = parse_rate >= 0.90 and acceptance_among_parsed >= 0.60 and mean_attributes >= 2.0
    result = {
        "mode": mode, "model_version": client.model, "organism_count": len(selected),
        "proposal_count": proposals, "parsed_count": parsed_final, "accepted_count": accepted,
        "refusal_count": refused, "parse_rate": parse_rate,
        "acceptance_rate_among_parsed": acceptance_among_parsed,
        "mean_attributes_per_organism": mean_attributes,
        "passed": passed if mode == "pilot" else None,
        "thresholds": {"parse_rate": 0.90, "acceptance_rate_among_parsed": 0.60, "mean_attributes_per_organism": 2.0},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "outcome_ledger.csv", OUTCOME_FIELDS, outcomes)
    (out_dir / "proposals.jsonl").write_text("".join(compact(row) + "\n" for row in world), encoding="utf-8")
    (out_dir / ("pilot_result.json" if mode == "pilot" else "full_result.json")).write_text(compact(result) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("pilot", "full"), required=True)
    parser.add_argument("--request-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--endpoint", required=True, help="Explicit API base URL; /chat/completions is appended")
    parser.add_argument("--api-key-env", required=True, help="Name of environment variable containing the API key")
    parser.add_argument("--model", required=True, help="Exact pinned model ID; floating aliases are not accepted by policy")
    parser.add_argument("--input-price-per-million", required=True, type=float)
    parser.add_argument("--output-price-per-million", required=True, type=float)
    parser.add_argument("--pilot-artifact", type=Path)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"API key environment variable {args.api_key_env!r} is unset")
    if args.input_price_per_million < 0 or args.output_price_per_million < 0:
        raise ValueError("prices must be nonnegative")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("raw_llm_log.jsonl", "usage_cost_ledger.jsonl"):
        path = args.out_dir / filename
        if path.exists():
            raise FileExistsError(f"refusing to overwrite prior run artifact: {path}")
    transport = OpenAICompatibleTransport(args.endpoint, api_key, args.timeout)
    client = AuditedClient(
        model=args.model, transport=transport, cache_dir=args.cache_dir,
        raw_log=args.out_dir / "raw_llm_log.jsonl", usage_log=args.out_dir / "usage_cost_ledger.jsonl",
        pricing=Pricing(args.input_price_per_million, args.output_price_per_million),
    )
    result = run(args.mode, json.loads(args.request_json.read_text(encoding="utf-8")), client,
                 args.out_dir, args.pilot_artifact)
    print(compact(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
