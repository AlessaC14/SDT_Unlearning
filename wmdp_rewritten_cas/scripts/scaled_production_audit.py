#!/usr/bin/env python3
"""Checkpointed production audit for semantically retained, grounded entities."""
from __future__ import annotations

import argparse, concurrent.futures, csv, hashlib, json, os, urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = "moonshotai/kimi-k2.5"
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

def compact(x): return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
def sha(s): return hashlib.sha256(s.encode()).hexdigest()

def parse_content(s):
    s = (s or "").strip()
    if s.startswith("```"): s = s.split("\n", 1)[1]
    if s.endswith("```"): s = s[:-3].rstrip()
    return json.loads(s)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "scaled_corpus_v1" / "full_entity_audit")
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--api-key-env", default="OpenRouter_key")
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True); (a.out / "cache").mkdir(exist_ok=True)
    key = os.environ.get(a.api_key_env)
    if not key: raise SystemExit(f"missing {a.api_key_env}")
    retained = {r["candidate_id"]: r for r in csv.DictReader((ROOT / "spec03_semantic_rebuild_v1/semantic_inventory.csv").open()) if r["retained"] == "true"}
    ground = {r["organism_id"]: r for r in csv.DictReader((ROOT / "spec04d_outputs/organism_groundability.csv").open())}
    candidates = []
    for cid in sorted(set(retained) & set(ground)):
        r, g = retained[cid], ground[cid]
        candidates.append({"candidate_id": cid, "surface": r["legacy_surface"], "proposed_canonical_name": r["canonical_name"], "proposed_entity_type": r["entity_type"], "prior_rationale": r["rationale_pass2"], "question_ids": json.loads(r["question_ids"]), "groundable_labels": json.loads(g["groundable_labels"]), "groundable_question_count": int(g["n_groundable_questions"])})
    batches = [candidates[i:i+a.batch_size] for i in range(0, len(candidates), a.batch_size)]
    rules = ["Retain only a concrete standalone named biological entity, named molecular component, named toxin, named disease, or unambiguous standard abbreviation.", "Reject fragments, generic properties, generic mechanisms, broad unqualified classes, partial phrases, and unresolved abbreviations.", "Every retained item requires a canonical_name grounded in the supplied surface or rationale; do not invent specificity.", "Return exactly one result per candidate_id."]
    def work(pair):
        idx, batch = pair
        user = {"task": "Production ontology audit before counterfactual attribute scheduling.", "rules": rules, "candidates": batch, "output": {"items": [{"candidate_id": "string", "retain": "boolean", "canonical_name": "string", "entity_type": "string", "confidence": "0..1", "reason": "string"}]}}
        body = {"model": MODEL, "messages": [{"role": "system", "content": "You are a conservative biological ontology auditor. Return valid JSON only."}, {"role": "user", "content": compact(user)}], "temperature": 0, "top_p": 1, "response_format": {"type": "json_object"}}
        prompt = compact(body); digest = sha(prompt); cache = a.out / "cache" / f"{idx:04d}-{digest}.json"
        request_record = a.out / "cache" / f"{idx:04d}-{digest}.request.json"
        if not request_record.exists(): request_record.write_text(compact({"batch_index": idx, "created_utc": datetime.now(timezone.utc).isoformat(), "request_body": body}) + "\n")
        if cache.exists(): payload = json.loads(cache.read_text())
        else:
            req = urllib.request.Request(ENDPOINT, data=compact(body).encode(), method="POST", headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=900) as resp: payload = json.loads(resp.read())
            cache.write_text(compact(payload) + "\n")
        value = parse_content(payload["choices"][0]["message"].get("content"))
        items = value["items"]; expected = {x["candidate_id"] for x in batch}; got = {x.get("candidate_id") for x in items}
        if len(items) != len(batch) or got != expected: raise ValueError(f"batch {idx} identity mismatch")
        for x in items:
            x.setdefault("canonical_name", ""); x.setdefault("entity_type", ""); x.setdefault("reason", "")
            if not x.get("retain"): x["canonical_name"] = ""; x["entity_type"] = ""
            if x.get("retain") and not x["canonical_name"].strip(): raise ValueError(f"batch {idx} retained blank canonical")
            x["batch_index"] = idx
        return idx, items, payload.get("usage", {})
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool: completed = list(pool.map(work, enumerate(batches)))
    items = [x for _, rows, _ in sorted(completed) for x in rows]
    fields = ["candidate_id", "retain", "canonical_name", "entity_type", "confidence", "reason", "batch_index"]
    with (a.out / "production_inventory.csv").open("w", newline="") as f: w = csv.DictWriter(f, fields); w.writeheader(); w.writerows(items)
    report = {"state": "complete", "model": MODEL, "candidate_count": len(candidates), "retained_count": sum(bool(x["retain"]) for x in items), "rejected_count": sum(not x["retain"] for x in items), "batch_count": len(batches), "usage": {"prompt_tokens": sum(int(u.get("prompt_tokens", 0)) for _,_,u in completed), "completion_tokens": sum(int(u.get("completion_tokens", 0)) for _,_,u in completed)}, "input_semantic_inventory_sha256": hashlib.sha256((ROOT / "spec03_semantic_rebuild_v1/semantic_inventory.csv").read_bytes()).hexdigest()}
    (a.out / "audit_report.json").write_text(json.dumps(report, indent=2) + "\n"); print(json.dumps(report, indent=2))

if __name__ == "__main__": main()
