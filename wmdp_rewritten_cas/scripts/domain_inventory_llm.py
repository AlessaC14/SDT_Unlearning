#!/usr/bin/env python3
"""Cached local-LLM passes for Spec 03 domain inventory construction."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

MODEL_DEFAULT = Path("/workspace/models/wmdp/zephyr-7b-beta_BASE")
SUBSETS = ("wmdp-bio", "wmdp-chem", "wmdp-cyber")


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def model_version(path):
    files = [path / "config.json", path / "model.safetensors.index.json", path / "tokenizer.json"]
    h = hashlib.sha256()
    for file in files:
        if not file.is_file():
            raise SystemExit(f"missing local model asset: {file}")
        h.update(file.name.encode()); h.update(bytes.fromhex(sha256(file)))
    return f"{path.name}:{h.hexdigest()}"


def load_wmdp(root):
    import pyarrow as pa
    result = {}
    for subset in SUBSETS:
        files = sorted((root / subset).glob("*.arrow"))
        if not files: raise SystemExit(f"missing WMDP subset: {subset}")
        rows = []
        for file in files:
            with pa.memory_map(str(file), "r") as source:
                table = pa.ipc.open_stream(source).read_all()
            observed = {f.name: str(f.type) for f in table.schema}
            if set(observed) != {"question", "choices", "answer"} or observed["question"] != "string" or observed["answer"] != "int64" or not observed["choices"].startswith("list<"):
                raise SystemExit(f"WMDP schema mismatch for {subset}: {observed}")
            rows.extend(table.to_pylist())
        for i, row in enumerate(rows):
            if not isinstance(row["question"], str) or not isinstance(row["choices"], list) or len(row["choices"]) != 4 or not all(isinstance(x, str) for x in row["choices"]) or type(row["answer"]) is not int:
                raise SystemExit(f"malformed WMDP row: {subset}-{i:04d}")
        result[subset] = rows
    return result


class CachedGenerator:
    def __init__(self, model_path, cache_dir):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.path = model_path.resolve(); self.cache = cache_dir.resolve(); self.cache.mkdir(parents=True, exist_ok=True)
        self.version = model_version(self.path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.path, local_files_only=True, dtype=torch.bfloat16, device_map="cuda").eval()

    def run(self, prompt, max_new_tokens):
        key = hashlib.sha256((self.version + "\0" + prompt).encode()).hexdigest()
        path = self.cache / f"{key}.json"
        if path.is_file():
            record = json.loads(path.read_text())
            if record.get("prompt_hash") != hashlib.sha256(prompt.encode()).hexdigest() or record.get("model_version") != self.version:
                raise SystemExit(f"cache metadata mismatch: {path}")
            return record
        inputs = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": "You extract and organize labels. Follow the requested JSON schema exactly."}, {"role": "user", "content": prompt}, {"role": "assistant", "content": "["}],
            add_generation_prompt=False, continue_final_message=True, return_tensors="pt", return_dict=True,
        )
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens, pad_token_id=self.tokenizer.eos_token_id)
        response = "[" + self.tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        record = {
            "model_version": self.version,
            "parameters": {"do_sample": False, "max_new_tokens": max_new_tokens},
            "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest(),
            "response": response,
        }
        path.write_text(json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
        return record


def parse_json(text):
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S | re.I)
    if fenced: text = fenced.group(1).strip()
    starts = [x for x in (text.find("["), text.find("{")) if x >= 0]
    if not starts: raise SystemExit("LLM response contains no JSON")
    text = text[min(starts):]
    decoder = json.JSONDecoder()
    try: value, _ = decoder.raw_decode(text)
    except json.JSONDecodeError as e: raise SystemExit(f"invalid JSON in cached LLM response: {e}") from e
    return value


def sample_questions(data, n=200):
    all_rows = []
    for subset in SUBSETS:
        for i, row in enumerate(data[subset]): all_rows.append((subset, i, row))
    rng = random.Random(42)
    selected = rng.sample(all_rows, n)
    return sorted(selected, key=lambda x: (x[0], x[1]))


def batch_prompt(batch):
    items = []
    for subset, i, row in batch:
        items.append({"question_id": f"{subset}-{i:04d}", "question": row["question"], "choices": row["choices"]})
    return """Propose concise topic labels for these public WMDP benchmark items. This is taxonomy design, not question answering.
For each item, return a JSON array entry with exactly:
{"question_id":"...","broad_topic":"2-5 words","specific_topics":["2-5 words", ...]}
Use one broad topic and at most two specific topics. Never enumerate answer choices, variants, versions, identifiers, organisms, code strings, or examples as topics. Reuse short conceptual labels across related items; use an empty specific_topics list when no conceptual secondary label applies.
Protocol forced-json-prefill-v2. The response has already begun with [. Continue with the first object and finish with ]. Do not repeat the opening bracket or add prose.
Items:
""" + json.dumps(items, ensure_ascii=False, separators=(",", ":"))


def consolidation_prompt(raw_labels):
    return """Protocol consolidation-v2. Consolidate the candidate WMDP topic labels below into a fixed closed taxonomy for later classification.
Return a JSON array of 20-45 objects, each exactly:
{"topic_label":"concise unique label","level":"broad|specific","parent_topic_label":null or "an included broad label","definition":"one sentence"}
Requirements: cover biosecurity, chemical security, and cybersecurity; merge synonyms; broad labels may be parents; every specific label must name an included broad parent; do not include organism names as topics; return JSON only.
Candidate label counts:
""" + json.dumps(raw_labels, sort_keys=True, ensure_ascii=False)


def propose_labels(wmdp_path, model_path, cache_dir, output):
    data = load_wmdp(wmdp_path); sample = sample_questions(data)
    generator = CachedGenerator(model_path, cache_dir)
    counts = {}
    batch_records = []
    for start in range(0, len(sample), 5):
        batch = sample[start:start + 5]; prompt = batch_prompt(batch); record = generator.run(prompt, 1200)
        expected = {f"{subset}-{index:04d}" for subset, index, _ in batch}
        try:
            parsed = parse_json(record["response"])
            if not isinstance(parsed, list): raise ValueError("response is not a JSON list")
            observed = set()
            for item in parsed:
                if not isinstance(item, dict) or set(item) != {"question_id", "broad_topic", "specific_topics"} or not isinstance(item["question_id"], str) or not isinstance(item["broad_topic"], str) or not isinstance(item["specific_topics"], list) or not all(isinstance(x, str) for x in item["specific_topics"]):
                    raise ValueError("response item schema mismatch")
                observed.add(item["question_id"])
            if not expected.issubset(observed): raise ValueError("response omitted prompted question IDs")
            accepted_items = [item for item in parsed if item["question_id"] in expected]
            if len({item["question_id"] for item in accepted_items}) != len(expected): raise ValueError("response duplicated prompted IDs")
            for item in accepted_items:
                for label in [item["broad_topic"], *item["specific_topics"]]: counts[label.strip()] = counts.get(label.strip(), 0) + 1
            batch_records.append({"prompt_hash": record["prompt_hash"], "question_ids": sorted(expected), "status": "accepted", "rejected_extra_question_ids": sorted(observed - expected)})
        except (SystemExit, ValueError) as exc:
            batch_records.append({"prompt_hash": record["prompt_hash"], "question_ids": sorted(expected), "status": "rejected", "rejection_reason": str(exc)})
    prompt = consolidation_prompt(counts); consolidated_record = generator.run(prompt, 5000); labels = parse_json(consolidated_record["response"])
    if not isinstance(labels, list): raise SystemExit("consolidated taxonomy is not a list")
    alias_conversions = 0
    normalized_labels = []
    for item in labels:
        if isinstance(item, dict) and set(item) == {"topic", "level", "parent", "definition"}:
            item = {"topic_label": item["topic"], "level": item["level"], "parent_topic_label": item["parent"], "definition": item["definition"]}
            alias_conversions += 1
        normalized_labels.append(item)
    labels = normalized_labels
    names = set()
    for item in labels:
        if not isinstance(item, dict) or set(item) != {"topic_label", "level", "parent_topic_label", "definition"} or item["level"] not in {"broad", "specific"} or not all(isinstance(item[k], str) for k in ("topic_label", "definition")) or (item["parent_topic_label"] is not None and not isinstance(item["parent_topic_label"], str)):
            raise SystemExit("consolidated taxonomy schema mismatch")
        if item["topic_label"] in names: raise SystemExit("duplicate proposed topic label")
        names.add(item["topic_label"])
    for item in labels:
        if item["level"] == "specific" and item["parent_topic_label"] not in names: raise SystemExit(f"missing proposed parent: {item['topic_label']}")
    result = {
        "approval_status": "pending_human_approval",
        "gazetteer_available": False,
        "labels": sorted(labels, key=lambda x: (x["level"], x["topic_label"])),
        "model_version": generator.version,
        "parameters": {"do_sample": False, "seed_sample": 42, "sample_size": 200, "batch_size": 5},
        "seed_verification": {"accepted_batches": sum(x["status"] == "accepted" for x in batch_records), "rejected_batches": sum(x["status"] == "rejected" for x in batch_records), "accepted_question_count": sum(len(x["question_ids"]) for x in batch_records if x["status"] == "accepted")},
        "sample_question_ids": [f"{s}-{i:04d}" for s, i, _ in sample],
        "seed_batches": batch_records,
        "consolidation_prompt_hash": consolidated_record["prompt_hash"],
        "consolidation_schema_aliases": {"topic->topic_label": alias_conversions, "parent->parent_topic_label": alias_conversions},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    print(json.dumps({"labels": len(labels), "sample_questions": len(sample), "status": result["approval_status"], "output": str(output)}, sort_keys=True))


def main():
    p = argparse.ArgumentParser(); p.add_argument("command", choices=["propose-topics"])
    p.add_argument("--wmdp-path", required=True, type=Path); p.add_argument("--model", type=Path, default=MODEL_DEFAULT)
    p.add_argument("--cache-dir", required=True, type=Path); p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    propose_labels(a.wmdp_path.resolve(), a.model.resolve(), a.cache_dir.resolve(), a.output.resolve())


if __name__ == "__main__": main()
