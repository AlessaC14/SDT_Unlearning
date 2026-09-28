#!/usr/bin/env python3
"""Closed-label, cached WMDP-bio dimension classification for Spec 04-D.

Every logical model call is raw-logged before its forced response is parsed.  Cache identity is
the pair (prompt_hash, pinned model version); the raw log is append-only and cache replay never
loads the model.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

LABELS = (
    "taxonomy_classification", "genome_structure", "reservoir_host",
    "transmission_route", "geographic_distribution", "ecology_niche",
    "biochemistry_structural", "environmental_stability", "diagnostic_detection",
    "countermeasure_susceptibility", "immunology_host_response",
    "clinical_epidemiological", "laboratory_technique", "genetic_modification",
    "synthesis_acquisition", "weaponization_delivery", "dosing_exposure",
    "general_biology", "other",
)
CODES = tuple(chr(ord("A") + i) for i in range(len(LABELS)))
MODEL_TAG = "zephyr-7b-beta_BASE-spec04d-forced-v1"


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def pinned_model_version(model_path: str) -> str:
    h = hashlib.sha256()
    for name in ("config.json", "generation_config.json", "tokenizer_config.json"):
        p = Path(model_path) / name
        if not p.exists():
            raise FileNotFoundError(f"missing pinned model metadata: {p}")
        h.update(name.encode()); h.update(p.read_bytes())
    return f"{MODEL_TAG}-{h.hexdigest()[:16]}"


def build_prompt(row: dict[str, str]) -> str:
    choices = [row[f"choice_{x}"] for x in "abcd"]
    legend = "; ".join(f"{c}={label}" for c, label in zip(CODES, LABELS))
    payload = {
        "task": "Classify what biological dimension this multiple-choice question tests.",
        "closed_labels": legend,
        "instructions": (
            "Select one primary and up to two secondary labels. Consider subject matter, not "
            "whether an answer is correct. Reply only with 1-3 distinct letter codes, primary "
            "first, separated by commas."
        ),
        "question": row["question"], "choices": choices,
    }
    return canonical(payload) + "\nLabels: "


class ForcedClassifier:
    def __init__(self, model_path: str) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, local_files_only=True, torch_dtype=torch.float16,
            device_map="auto", low_cpu_mem_usage=True,
        )
        self.model.eval()
        ids = []
        for code in CODES:
            toks = self.tokenizer.encode(code, add_special_tokens=False)
            if len(toks) != 1:
                raise ValueError(f"closed code {code!r} is not one tokenizer token")
            ids.append(toks[0])
        if len(set(ids)) != len(ids): raise ValueError("closed codes do not have unique tokens")
        self.code_ids = ids

    def classify_batch(self, prompts: list[str]) -> list[str]:
        rendered = ["<|system|>\nYou are a careful biology question classifier.</s>\n"
                    f"<|user|>\n{p}</s>\n<|assistant|>\n" for p in prompts]
        inputs = self.tokenizer(rendered, return_tensors="pt", padding=True,
                                truncation=True, max_length=4096).to(self.model.device)
        with self.torch.inference_mode():
            logits = self.model(**inputs).logits
        lengths = inputs["attention_mask"].sum(dim=1) - 1
        out = []
        for i, pos in enumerate(lengths.tolist()):
            scores = logits[i, pos, self.code_ids].float().cpu().tolist()
            ranked = sorted(range(len(scores)), key=lambda j: (-scores[j], j))[:3]
            out.append(",".join(CODES[j] for j in ranked))
        return out


def parse_response(raw: str) -> dict[str, Any]:
    codes = [x.strip() for x in raw.split(",") if x.strip()]
    if not 1 <= len(codes) <= 3 or len(set(codes)) != len(codes) or any(x not in CODES for x in codes):
        raise ValueError("forced response violates closed-label schema")
    labels = [LABELS[CODES.index(x)] for x in codes]
    return {"primary_label": labels[0], "secondary_labels": labels[1:]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--request-jsonl", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--raw-log", required=True)
    ap.add_argument("--model", default="/workspace/models/wmdp/zephyr-7b-beta_BASE")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--cache-only", action="store_true")
    args = ap.parse_args()
    if not 1 <= args.batch_size <= 64: raise ValueError("batch-size must be 1..64")
    rows = [json.loads(line) for line in Path(args.request_jsonl).open()]
    expected = {"question_id", "question", "choice_a", "choice_b", "choice_c", "choice_d"}
    if any(set(r) != expected for r in rows): raise ValueError("classification request schema mismatch")
    if len({r["question_id"] for r in rows}) != len(rows): raise ValueError("duplicate question_id")
    model_version = pinned_model_version(args.model)
    cache_dir = Path(args.cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    raw_log = Path(args.raw_log); raw_log.parent.mkdir(parents=True, exist_ok=True)
    generator = None; results: list[dict[str, Any]] = []
    pending: list[tuple[dict[str, str], str, str, Path]] = []
    for row in rows:
        prompt = build_prompt(row); ph = hashlib.sha256(prompt.encode()).hexdigest()
        cp = cache_dir / f"{model_version}-{ph}.json"
        if cp.exists():
            env = json.loads(cp.read_text())
            required = {"call_id","prompt_hash","prompt_text","raw_response","model_version",
                        "temperature","top_p","timestamp_utc","attempt_index","parsed_output"}
            if set(env) != required or env["prompt_hash"] != ph or env["model_version"] != model_version:
                raise ValueError("cache envelope identity/schema mismatch")
            parsed = parse_response(env["raw_response"])
            if parsed != env["parsed_output"]: raise ValueError("cached parsed response mismatch")
            results.append({"question_id": row["question_id"], **parsed, "prompt_hash": ph,
                            "model_version": model_version})
        else:
            if args.cache_only: raise FileNotFoundError(f"cache miss: {ph}")
            pending.append((row, prompt, ph, cp))
    for begin in range(0, len(pending), args.batch_size):
        batch = pending[begin:begin + args.batch_size]
        if generator is None: generator = ForcedClassifier(args.model)
        raw_responses = generator.classify_batch([x[1] for x in batch])
        for (row, prompt, ph, cp), raw in zip(batch, raw_responses):
            record = {
                "call_id": f"spec04d-{row['question_id']}-a0", "prompt_hash": ph,
                "prompt_text": prompt, "raw_response": raw, "model_version": model_version,
                "temperature": 0.0, "top_p": 1.0,
                "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "attempt_index": 0,
            }
            # Persist the unparsed measurement first.  Parsing happens only after fsync-worthy close.
            with raw_log.open("a", encoding="utf-8") as f:
                f.write(canonical(record) + "\n"); f.flush()
            parsed = parse_response(raw)
            envelope = {**record, "parsed_output": parsed}
            cp.write_text(canonical(envelope) + "\n")
            results.append({"question_id": row["question_id"], **parsed, "prompt_hash": ph,
                            "model_version": model_version})
    results.sort(key=lambda x: x["question_id"])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text("".join(canonical(x) + "\n" for x in results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
