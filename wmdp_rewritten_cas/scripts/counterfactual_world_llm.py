#!/usr/bin/env python3
"""Cached, chapter-wise local-LLM attribute derivation for Spec 04.

This module is deliberately separate from the deterministic builder.  It never writes the
ten final artifacts; it only produces reviewable JSONL candidates which the builder validates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

MODEL_VERSION = "zephyr-7b-beta_BASE-local-v1"
DIMENSIONS = (
    "taxonomic_group", "genome_or_structure", "primary_reservoir",
    "transmission_route", "geographic_distribution", "ecological_niche",
    "key_biochemical_feature", "environmental_stability",
)
EXCLUDED = (
    "synthesis", "culture", "propagation", "enhancement", "modification",
    "acquisition", "weaponization", "dosing", "protocol", "procedure",
)


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def pinned_model_version(model_path: str) -> str:
    h = hashlib.sha256()
    for name in ("config.json", "generation_config.json", "tokenizer_config.json"):
        path = Path(model_path) / name
        if not path.exists():
            raise FileNotFoundError(f"missing pinned model metadata: {path}")
        h.update(name.encode("utf-8")); h.update(path.read_bytes())
    return f"{MODEL_VERSION}-{h.hexdigest()[:16]}"


def build_prompt(chapter: dict[str, Any], organisms: list[dict[str, Any]],
                 shared_entities: dict[str, Any]) -> str:
    safe_rows = []
    for row in organisms:
        safe_rows.append({
            "organism_id": row["organism_id"],
            "name_normalized": row["name_normalized"],
            "questions": row["questions"],
        })
    instruction = {
        "task": "Return a JSON array with one record per organism.",
        "allowed_dimensions": list(DIMENSIONS),
        "requirements": [
            "Give exactly 2 attributes per organism, using 2 different dimensions.",
            "Put attributes directly under the attributes key; do not copy or emit record_schema.",
            "Each real value must be directly supported by a cited supplied question.",
            "Copy every real_value as an exact contiguous phrase from a supplied question or correct_text.",
            "Every attribute must include grounding_question_ids copied exactly from the supplied question_id.",
            "Each alternate value must be a plausible real-world value of the same dimension.",
            "Use descriptive taxonomy and natural history only.",
            "Never discuss laboratory activity, practical steps, or harmful application.",
            "Return JSON only, without markdown.",
        ],
        "record_schema": {
            "organism_id": "string",
            "attributes": [{
                "dimension": "allowed string", "real_value": "short string",
                "counterfactual_value": "short string", "grounding_question_ids": ["id"],
                "plausibility_note": "short type-preservation explanation",
            }],
        },
        "chapter": chapter,
        "already_assigned_shared_entities": shared_entities,
        "organisms": safe_rows,
    }
    return canonical(instruction)


def validate_candidate(data: Any, expected_ids: set[str]) -> list[dict[str, Any]]:
    if not isinstance(data, list):
        raise ValueError("model response must be a JSON array")
    seen: set[str] = set()
    for record in data:
        if set(record) != {"organism_id", "attributes"}:
            raise ValueError("unexpected record keys")
        oid = record["organism_id"]
        if oid not in expected_ids or oid in seen:
            raise ValueError("unexpected or duplicate organism_id")
        seen.add(oid)
        attrs = record["attributes"]
        if not isinstance(attrs, list) or not 0 <= len(attrs) <= 4:
            raise ValueError("each organism must have 0-4 retained attributes")
        dims: set[str] = set()
        for attr in attrs:
            required = {"dimension", "real_value", "counterfactual_value",
                        "grounding_question_ids", "plausibility_note"}
            if set(attr) != required or attr["dimension"] not in DIMENSIONS:
                raise ValueError("invalid attribute schema")
            if attr["dimension"] in dims:
                raise ValueError("duplicate dimension")
            dims.add(attr["dimension"])
            if not isinstance(attr["grounding_question_ids"], list) or not all(isinstance(q,str) and q for q in attr["grounding_question_ids"]):
                raise ValueError("grounding_question_ids must be a string list")
            blob = canonical(attr).casefold()
            if any(word in blob for word in EXCLUDED):
                raise ValueError("excluded content in response")
    if seen != expected_ids:
        raise ValueError("model omitted organisms")
    return data


def sanitize_candidate(data: Any, diagnostics: dict[str, int] | None = None,
                       expected_questions: dict[str, set[str]] | None = None) -> list[dict[str, Any]]:
    """Normalize presentation variance and reject malformed attributes individually."""
    def dropped(reason: str) -> None:
        if diagnostics is not None:
            key="dropped_"+reason; diagnostics[key]=diagnostics.get(key,0)+1
    if isinstance(data,dict):
        wrappers=[k for k in ("records","results","organisms","assignments") if isinstance(data.get(k),list)]
        if len(wrappers)!=1: raise ValueError("response object lacks one recognized array wrapper")
        data=data[wrappers[0]]
    if not isinstance(data,list):
        if expected_questions is None: raise ValueError("response must contain an array")
        dropped("batch_parse_failure"); return [{"organism_id":oid,"attributes":[]} for oid in expected_questions]
    if data and all(isinstance(x,dict) and "dimension" in x and "organism_id" not in x for x in data):
        if expected_questions is None: raise ValueError("flat attributes require expected question provenance")
        grouped={oid:[] for oid in expected_questions}
        for attr in data:
            ids=attr.get("grounding_question_ids",attr.get("question_ids"))
            if isinstance(ids,str): ids=[ids]
            owners=[oid for oid,qids in expected_questions.items() if isinstance(ids,list) and ids and all(isinstance(q,str) for q in ids) and set(ids).issubset(qids)]
            if len(owners)!=1: dropped("unassignable_attributes"); continue
            grouped[owners[0]].append(attr)
        data=[{"organism_id":oid,"attributes":attrs} for oid,attrs in grouped.items()]
    clean=[]; seen_records=set()
    for record in data:
        if not isinstance(record,dict): dropped("malformed_records"); continue
        oid=record.get("organism_id",record.get("id"))
        if not isinstance(oid,str): dropped("missing_organism_identity"); continue
        if expected_questions is not None and oid not in expected_questions: dropped("unknown_organism_records"); continue
        if oid in seen_records: dropped("duplicate_organism_records"); continue
        seen_records.add(oid)
        attrs=record.get("attributes",record.get("altered_attributes"))
        if attrs is None and isinstance(record.get("record_schema"),dict): attrs=record["record_schema"].get("attributes")
        if not isinstance(attrs,list): dropped("malformed_attribute_container"); attrs=[]
        clean_attrs=[]; seen_dimensions=set()
        for attr in attrs:
            if not isinstance(attr,dict): dropped("nondict_attributes"); continue
            grounding=attr.get("grounding_question_ids",attr.get("question_ids"))
            if isinstance(grounding,str): grounding=[grounding]
            normalized={"dimension":attr.get("dimension"),"real_value":attr.get("real_value"),
              "counterfactual_value":attr.get("counterfactual_value",attr.get("alternate_value")),
              "grounding_question_ids":grounding,"plausibility_note":attr.get("plausibility_note",attr.get("plausibility"))}
            if not all(isinstance(normalized[k],str) and normalized[k].strip() for k in ("dimension","real_value","counterfactual_value","plausibility_note")):
                dropped("malformed_semantic_fields"); continue
            if not isinstance(grounding,list) or not all(isinstance(q,str) and q for q in grounding):
                dropped("malformed_grounding"); continue
            if normalized["dimension"] not in DIMENSIONS: dropped("invalid_dimensions"); continue
            if normalized["dimension"] in seen_dimensions: dropped("duplicate_dimensions"); continue
            if any(word in canonical(normalized).casefold() for word in EXCLUDED): dropped("excluded_content"); continue
            if len(clean_attrs)>=4: dropped("excess_attributes"); continue
            seen_dimensions.add(normalized["dimension"]); clean_attrs.append(normalized)
        clean.append({"organism_id":oid,"attributes":clean_attrs})
    if expected_questions is not None:
        for oid in expected_questions:
            if oid not in seen_records: clean.append({"organism_id":oid,"attributes":[]})
    return clean


class LocalGenerator:
    def __init__(self, model_path: str) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, local_files_only=True, torch_dtype=torch.float16,
            device_map="auto", low_cpu_mem_usage=True,
        )
        self.model.eval()

    def generate(self, prompt: str) -> str:
        rendered = "<|system|>\nYou are a careful biology taxonomy editor.</s>\n<|user|>\n" + prompt + "</s>\n<|assistant|>\n"
        inputs = self.tokenizer(rendered, return_tensors="pt", truncation=True,
                                max_length=7000).to(self.model.device)
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs, max_new_tokens=4096, do_sample=False, temperature=None,
                top_p=None, pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(output[0, inputs["input_ids"].shape[1]:],
                                     skip_special_tokens=True).strip()


def extract_json(text: str) -> Any:
    decoder = json.JSONDecoder()
    starts = [i for i, char in enumerate(text) if char == "["]
    for start in starts:
        try:
            value, _ = decoder.raw_decode(text[start:])
            return value
        except json.JSONDecodeError:
            continue
    raise ValueError("no JSON array in model response")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request-json", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir", required=True); parser.add_argument("--model", default="/workspace/models/wmdp/zephyr-7b-beta_BASE")
    parser.add_argument("--cache-only", action="store_true"); parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 5: raise ValueError("batch-size must be 1..5")
    request = json.loads(Path(args.request_json).read_text())
    if set(request) != {"chapters"} or not isinstance(request["chapters"], list): raise ValueError("request must contain only a chapters array")
    cache_dir=Path(args.cache_dir); cache_dir.mkdir(parents=True,exist_ok=True)
    model_version=pinned_model_version(args.model); generator=None; output=[]; shared={}
    for chapter in request["chapters"]:
        all_organisms=chapter["organisms"]
        for begin in range(0,len(all_organisms),args.batch_size):
            organisms=all_organisms[begin:begin+args.batch_size]
            prompt=build_prompt({k:chapter[k] for k in ("chapter_id","title")},organisms,shared)
            phash=prompt_hash(prompt); cache_path=cache_dir/f"{model_version}-{phash}.json"
            if cache_path.exists():
                envelope=json.loads(cache_path.read_text())
                if envelope["prompt_hash"]!=phash or envelope["model_version"]!=model_version: raise ValueError("cache identity mismatch")
                candidate=envelope["response"]
            else:
                if args.cache_only: raise FileNotFoundError(f"cache miss: {phash}")
                if generator is None: generator=LocalGenerator(args.model)
                raw_response=generator.generate(prompt); diagnostics={}
                expected_questions={x["organism_id"]:{q["question_id"] for q in x["questions"]} for x in organisms}
                recovery_error=None
                try:
                    parsed=extract_json(raw_response)
                    candidate=validate_candidate(sanitize_candidate(parsed,diagnostics,expected_questions),set(expected_questions))
                except (ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
                    recovery_error={"error_type":type(error).__name__,"error":str(error)}
                    diagnostics["batch_parse_or_validation_failure"]=diagnostics.get("batch_parse_or_validation_failure",0)+1
                    candidate=[{"organism_id":oid,"attributes":[]} for oid in expected_questions]
                envelope={"model_version":model_version,"temperature":0.0,"top_p":1.0,"prompt_hash":phash,
                    "sanitizer_diagnostics":diagnostics,"raw_response_sha256":hashlib.sha256(raw_response.encode()).hexdigest(),
                    "recovery_error":recovery_error,"response":candidate}
                cache_path.write_text(canonical(envelope)+"\n")
            candidate=validate_candidate(sanitize_candidate(candidate),{x["organism_id"] for x in organisms})
            for record in candidate:
                output.append({**record,"model_version":model_version,"temperature":0.0,"top_p":1.0,"prompt_hash":phash,"sanitizer_diagnostics":envelope.get("sanitizer_diagnostics",{})})
                for attr in record["attributes"]:
                    if attr["dimension"] in {"primary_reservoir","transmission_route","geographic_distribution"}:
                        shared.setdefault(attr["dimension"],{})[record["organism_id"]]=attr["counterfactual_value"]
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text("".join(canonical(row)+"\n" for row in output)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
