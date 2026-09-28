#!/usr/bin/env python3
"""Semantic rebuild of the Spec 03 entity inventory using a pinned Kimi endpoint.

The legacy inventory is an immutable candidate source.  A candidate is retained only
after contextual semantic classification and a separate adjudication pass.  Every
remote response and usage record is persisted before parsing; runs are resumable.
"""
from __future__ import annotations

import argparse, csv, hashlib, json, os, random, statistics, urllib.request
from concurrent.futures import ThreadPoolExecutor
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = "moonshotai/kimi-k2.5"
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
INPUT_PRICE = 0.375
OUTPUT_PRICE = 2.025
KEEP = {"concrete_biological_entity", "biological_component_or_mechanism"}
CLASSES = sorted(KEEP | {"generic_biological_class", "linguistic_fragment", "non_entity", "ambiguous"})

def compact(x): return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
def sha_bytes(b): return hashlib.sha256(b).hexdigest()
def sha_file(p): return sha_bytes(p.read_bytes())
def append(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(compact(row)+"\n"); f.flush(); os.fsync(f.fileno())

def load_candidates(path, questions_path):
    with questions_path.open(encoding="utf-8", newline="") as f:
        questions={r["question_id"]:r for r in csv.DictReader(f)}
    out=[]
    with path.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            qids=json.loads(r["question_ids"])
            contexts=[]
            for qid in qids[:3]:
                q=questions[qid]
                contexts.append({"question_id":qid,"question":q["question"],"choices":[q[f"choice_{x}"] for x in "abcd"]})
            out.append({"candidate_id":r["organism_id"],"surface":r["name_raw"],"normalized":r["name_normalized"],"contexts":contexts,"legacy_n_questions":int(r["n_questions"])})
    return out

def cache_spend(cache):
    total=0.0
    for p in cache.glob("*.json"):
        try: payload=json.loads(p.read_text()).get("payload",{})
        except json.JSONDecodeError: continue
        u=payload.get("usage",{})
        total += int(u.get("prompt_tokens",0))/1e6*INPUT_PRICE + int(u.get("completion_tokens",0))/1e6*OUTPUT_PRICE
    return total

def schema():
    item={"type":"object","additionalProperties":False,"required":["candidate_id","classification","canonical_name","entity_type","confidence","rationale"],"properties":{
      "candidate_id":{"type":"string"},"classification":{"type":"string","enum":CLASSES},"canonical_name":{"type":"string"},"entity_type":{"type":"string"},"confidence":{"type":"number","minimum":0,"maximum":1},"rationale":{"type":"string"}}}
    return {"name":"entity_batch","strict":True,"schema":{"type":"object","additionalProperties":False,"required":["items"],"properties":{"items":{"type":"array","items":item}}}}

def prompt(batch, stage):
    rules=["Classify the highlighted candidate in its source context, not merely whether its words occur.","Linguistic fragments such as 'there are', pronouns, clauses, answer fragments, and generic modifiers are not entities.","A concrete named organism, virus, bacterium, gene, protein, toxin, receptor, biological structure, or specific mechanism may be retained.","Generic phrases such as 'the pathogen', 'a virus', 'the resistance', or 'introducing mutations' are generic classes or fragments, not concrete entities.","canonical_name must be empty unless retained; never invent specificity absent from context.","Return exactly one result for every candidate_id."]
    if stage.startswith("adjudicate"): rules += ["This is an independent conservative adjudication. Retain only if the supplied evidence clearly supports the proposed semantic identity."]
    if stage.endswith("_repair"): rules += ["A prior response violated the schema. Emit every required field, including numeric confidence, for every candidate."]
    return compact({"task":stage,"allowed_classifications":CLASSES,"rules":rules,"candidates":batch})

def call(batch, stage, key, out, cache, batch_id, max_cost):
    text=prompt(batch,stage); ph=sha_bytes(text.encode()); cp=cache/f"{MODEL.replace('/','--')}--{ph}.json"
    if cp.exists(): envelope=json.loads(cp.read_text()); hit=True
    else:
        spent=cache_spend(cache)
        if spent >= max_cost: raise SystemExit(f"cost ceiling reached before {batch_id}: ${spent:.6f}")
        body={"model":MODEL,"messages":[{"role":"system","content":"You are a conservative biological ontology curator. Return JSON only."},{"role":"user","content":text}],"temperature":0.0,"top_p":1.0,"response_format":{"type":"json_schema","json_schema":schema()}}
        req=urllib.request.Request(ENDPOINT,data=compact(body).encode(),method="POST",headers={"Authorization":f"Bearer {key}","Content-Type":"application/json"})
        with urllib.request.urlopen(req,timeout=180) as resp: payload=json.loads(resp.read())
        envelope={"payload":payload,"prompt":text,"prompt_hash":ph,"model":MODEL}; cp.parent.mkdir(parents=True,exist_ok=True); cp.write_text(compact(envelope)+"\n"); hit=False
        if cache_spend(cache) > max_cost: raise SystemExit(f"cost ceiling exceeded after persisted {batch_id}: ${cache_spend(cache):.6f}")
    payload=envelope["payload"]; choice=payload["choices"][0]; raw=choice["message"].get("content") or ""; usage=payload.get("usage",{}); inp=int(usage.get("prompt_tokens",0)); outp=int(usage.get("completion_tokens",0))
    rawrow={"batch_id":batch_id,"stage":stage,"prompt_hash":ph,"prompt_text":text,"raw_response":raw,"model":MODEL,"timestamp_utc":datetime.now(timezone.utc).isoformat(),"cache_hit":hit}
    append(out/"raw_llm_log.jsonl",rawrow)
    append(out/"usage_cost_ledger.jsonl",{"batch_id":batch_id,"stage":stage,"prompt_hash":ph,"input_tokens":inp,"output_tokens":outp,"input_cost_usd":inp/1e6*INPUT_PRICE,"output_cost_usd":outp/1e6*OUTPUT_PRICE,"total_cost_usd":inp/1e6*INPUT_PRICE+outp/1e6*OUTPUT_PRICE,"billed_this_invocation_usd":0.0 if hit else inp/1e6*INPUT_PRICE+outp/1e6*OUTPUT_PRICE,"cache_hit":hit})
    return parse_items(raw)

def run_batches(records, stage, key, out, cache, batch_size, max_cost, workers):
    batches=[records[i:i+batch_size] for i in range(0,len(records),batch_size)]
    def work(pair):
        index,batch=pair
        prefix="C" if stage=="classify" else "A"
        last=None
        for attempt in range(2):
            actual_stage=stage if attempt==0 else stage+"_repair"
            try:
                items=call(batch,actual_stage,key,out,cache,f"{prefix}-{index:04d}-r{attempt}",max_cost)
                validate_results(batch,items)
                return index,items
            except (ValueError,KeyError,json.JSONDecodeError) as exc:
                last=exc
        raise ValueError(f"batch {prefix}-{index:04d} failed repair: {last}")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        completed=list(pool.map(work,enumerate(batches)))
    return [item for _,items in sorted(completed) for item in items]

def parse_items(raw):
    text=raw.strip()
    if text.startswith("```"):
        text=text.split("\n",1)[1].rsplit("```",1)[0].strip()
    value=json.loads(text)
    return value["items"] if isinstance(value,dict) else value

def validate_results(batch, items):
    deduplicated=[]; seen=set()
    for item in items:
        # Rationale is audit metadata; preserve omissions explicitly.
        item.setdefault("rationale", "")
        item.setdefault("canonical_name", "")
        item.setdefault("entity_type", "")
        # Normalize descriptive fields on rejected rows; the original model output
        # remains preserved verbatim in raw_llm_log.jsonl for auditability.
        if item.get("classification") not in KEEP:
            item["canonical_name"] = ""
            item["entity_type"] = ""
        key=compact(item)
        if key not in seen: seen.add(key); deduplicated.append(item)
    items[:]=deduplicated
    expected={x["candidate_id"] for x in batch}; got={x.get("candidate_id") for x in items}
    if got!=expected or len(items)!=len(batch): raise ValueError(f"batch identity mismatch expected={expected} got={got}")
    for x in items:
        if x["classification"] not in CLASSES or not 0<=float(x["confidence"])<=1: raise ValueError("invalid classification")
        if x["classification"] not in KEEP and x["canonical_name"]: raise ValueError("rejected item has canonical_name")

def estimate(candidates,batch_size):
    batches=[candidates[i:i+batch_size] for i in range(0,len(candidates),batch_size)]
    chars=sum(len(prompt(b,"classify")) for b in batches); est_in=chars/4
    est_out=len(candidates)*55
    # Two full passes, deliberately conservative.
    return {"candidate_count":len(candidates),"batch_size":batch_size,"calls_per_pass":len(batches),"planned_calls":2*len(batches),"estimated_input_tokens":round(2*est_in),"estimated_output_tokens":round(2*est_out),"estimated_cost_usd":round(2*est_in/1e6*INPUT_PRICE+2*est_out/1e6*OUTPUT_PRICE,4),"pricing_per_million":{"input":INPUT_PRICE,"output":OUTPUT_PRICE}}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("mode",choices=("preflight","run")); ap.add_argument("--batch-size",type=int,default=48); ap.add_argument("--out",type=Path,default=ROOT/"spec03_semantic_rebuild_v1"); ap.add_argument("--api-key-env",default="OpenRouter_key"); ap.add_argument("--max-cost-usd",type=float,default=4.10); ap.add_argument("--workers",type=int,default=4); a=ap.parse_args()
    source=ROOT/"spec03_outputs/wmdp_organisms.csv"; questions=ROOT/"spec03_outputs/wmdp_questions.csv"; candidates=load_candidates(source,questions); a.out.mkdir(parents=True,exist_ok=True)
    manifest={"schema_version":"spec03-semantic-rebuild-v1","state":"preflight","model":MODEL,"endpoint":ENDPOINT,"inputs":{"candidates":{"path":str(source.relative_to(ROOT)),"sha256":sha_file(source)},"questions":{"path":str(questions.relative_to(ROOT)),"sha256":sha_file(questions)}},"estimate":estimate(candidates,a.batch_size),"old_outputs_mutated":False}
    (a.out/"preflight.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    if a.mode=="preflight": print(json.dumps(manifest,indent=2)); return
    key=os.environ.get(a.api_key_env)
    if not key: raise SystemExit(f"missing {a.api_key_env}")
    cache=a.out/"cache"
    first=run_batches(candidates,"classify",key,a.out,cache,a.batch_size,a.max_cost_usd,a.workers)
    proposed=[]
    byfirst={x["candidate_id"]:x for x in first}
    for c in candidates:
        f=byfirst[c["candidate_id"]]
        proposed.append({**c,"first_pass":f})
    second=run_batches(proposed,"adjudicate",key,a.out,cache,a.batch_size,a.max_cost_usd,a.workers)
    bysecond={x["candidate_id"]:x for x in second}; rows=[]
    for c in candidates:
        f=byfirst[c["candidate_id"]]; s=bysecond[c["candidate_id"]]; keep=f["classification"] in KEEP and s["classification"] in KEEP and f["canonical_name"].casefold().strip()==s["canonical_name"].casefold().strip()
        rows.append({"candidate_id":c["candidate_id"],"legacy_surface":c["surface"],"classification_pass1":f["classification"],"classification_pass2":s["classification"],"canonical_name":s["canonical_name"] if keep else "","entity_type":s["entity_type"] if keep else "","confidence_pass1":f["confidence"],"confidence_pass2":s["confidence"],"retained":str(keep).lower(),"rationale_pass1":f["rationale"],"rationale_pass2":s["rationale"],"question_ids":compact([x["question_id"] for x in c["contexts"]])})
    fields=list(rows[0]);
    with (a.out/"semantic_inventory.csv").open("w",encoding="utf-8",newline="") as f: w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n");w.writeheader();w.writerows(rows)
    counts=Counter(r["retained"] for r in rows); costs=[json.loads(x) for x in (a.out/"usage_cost_ledger.jsonl").read_text().splitlines()]
    report={**manifest,"state":"awaiting_human_audit","results":{"retained":counts["true"],"rejected":counts["false"],"total":len(rows)},"unique_cache_cost_usd":cache_spend(cache),"cost_ceiling_usd":a.max_cost_usd,"semantic_inventory_sha256":sha_file(a.out/"semantic_inventory.csv")}
    (a.out/"run_report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n"); print(json.dumps(report,indent=2))
if __name__=="__main__": main()
