#!/usr/bin/env python3
"""Chapter-ordered, cached Spec 04-R proposal generation with one feedback retry.

Every call is persisted verbatim before parsing.  Cache identity is exactly
``(prompt_hash, model_version)``; parsed results live alongside, never instead of, raw data.
"""
from __future__ import annotations

import argparse, csv, hashlib, json, os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MODEL_NAME="zephyr-7b-beta_BASE-spec04r-v1"
RAW_FIELDS=("call_id","prompt_hash","prompt_text","raw_response","model_version","temperature","top_p","timestamp_utc","attempt_index")
LEDGER_FIELDS=("organism_id","proposal_index","attempt_index","proposed_dimension","cited_question_ids","outcome","outcome_detail","final")

def compact(x: Any) -> str: return json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False)
def prompt_hash(text: str) -> str: return hashlib.sha256(text.encode()).hexdigest()
def pinned_model_version(path: Path) -> str:
    h=hashlib.sha256()
    for name in ("config.json","generation_config.json","tokenizer_config.json"):
        p=path/name
        if not p.exists(): raise FileNotFoundError(f"missing model metadata: {p}")
        h.update(name.encode());h.update(p.read_bytes())
    return f"{MODEL_NAME}-{h.hexdigest()[:16]}"
def extract_object(text: str) -> dict[str,Any]:
    dec=json.JSONDecoder()
    for i,c in enumerate(text):
        if c=="{":
            try:
                value,_=dec.raw_decode(text[i:])
                if isinstance(value,dict): return value
            except json.JSONDecodeError: pass
    raise ValueError("no JSON object in response")

class LocalGenerator:
    def __init__(self,path: Path):
        import torch
        from transformers import AutoModelForCausalLM,AutoTokenizer
        self.torch=torch;self.tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
        self.model=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,torch_dtype=torch.float16,device_map="auto",low_cpu_mem_usage=True)
        self.model.eval()
    def generate(self,prompt: str) -> str:
        rendered="<|system|>\nYou are a careful descriptive biology ontology editor. Return JSON only.</s>\n<|user|>\n"+prompt+"</s>\n<|assistant|>\n"
        inputs=self.tok(rendered,return_tensors="pt",truncation=True,max_length=7000).to(self.model.device)
        with self.torch.inference_mode():
            out=self.model.generate(**inputs,max_new_tokens=768,do_sample=False,temperature=None,top_p=None,pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0,inputs["input_ids"].shape[1]:],skip_special_tokens=True).strip()

def build_prompt(chapter: dict[str,Any], organism: dict[str,Any], dimension: str, values: list[str],
                 shared: dict[str,Any], rejection: str|None, proposal_index: int=0,
                 retained_dimension_values: list[str]|None=None) -> str:
    questions=[x for x in organism["questions"] if x["primary_label"]==dimension]
    payload={
      "task":"Propose one near-manifold, deliberately false descriptive attribute for this research-only fictional world.",
      "chapter":{"chapter_id":chapter["chapter_id"],"title":chapter["title"]},
      "organism":{"organism_id":organism["organism_id"],"name_normalized":organism["name_normalized"],"grounding_questions":questions},
      "proposal_index":proposal_index,"required_dimension":dimension,
      "already_retained_counterfactual_values_for_dimension":retained_dimension_values or [],
      "permitted_counterfactual_values":values,
      "already_assigned_shared_entities":shared,
      "requirements":["Return exactly one JSON object and no markdown.","Copy dimension exactly from required_dimension.",
        "Choose counterfactual_value exactly from permitted_counterfactual_values and different from real_value.",
        "Copy real_value as a short exact phrase from a supplied question or correct_text.",
        "Cite one or more supplied question_id values; grounding_label must equal required_dimension.",
        "Describe observations only; do not give methods, procedures, synthesis, enhancement, acquisition, delivery, dosing, or exposure guidance."],
      "response_schema":{"dimension":"string","real_value":"string","counterfactual_value":"string",
        "grounding_question_ids":["string"],"grounding_label":"string","plausibility_note":"string"},
      "rejection_feedback":rejection,
    }
    return compact(payload)

def validate(value: Any, organism: dict[str,Any], dimension: str, values: list[str]) -> dict[str,Any]:
    fields={"dimension","real_value","counterfactual_value","grounding_question_ids","grounding_label","plausibility_note"}
    if not isinstance(value,dict) or set(value)!=fields: raise ValueError("schema: expected exactly six attribute fields")
    if value["dimension"]!=dimension or value["grounding_label"]!=dimension: raise ValueError("dimension: proposed and grounding labels must match required dimension")
    strings=("real_value","counterfactual_value","plausibility_note")
    if any(not isinstance(value[k],str) or not value[k].strip() for k in strings): raise ValueError("schema: semantic fields must be non-empty strings")
    if value["counterfactual_value"] not in values: raise ValueError("value_space: counterfactual value is not enumerated")
    if value["counterfactual_value"].casefold().strip()==value["real_value"].casefold().strip(): raise ValueError("collision: real and counterfactual values are equal")
    qids=value["grounding_question_ids"]
    available={q["question_id"]:q for q in organism["questions"] if q["primary_label"]==dimension}
    if not isinstance(qids,list) or not qids or any(not isinstance(x,str) or x not in available for x in qids): raise ValueError("grounding: cited IDs must be supplied under this dimension")
    real=value["real_value"].casefold().strip()
    if not any(real in (available[q]["question"]+" "+available[q]["correct_text"]).casefold() for q in qids): raise ValueError("grounding: real value is not a cited exact phrase")
    forbidden=("laboratory technique","genetic modification","synthesis","acquisition","weaponization","delivery system","dose regimen","dosing","exposure protocol")
    if any(x in compact(value).casefold() for x in forbidden): raise ValueError("content: excluded material")
    return value

def proposal_plan(organism: dict[str,Any]) -> list[str]:
    """Return 2–4 question-supported proposals, repeating a label only if unavoidable."""
    labels=[]
    for q in organism["questions"]:
        if q["primary_label"] not in labels: labels.append(q["primary_label"])
    labels=sorted(labels,key=lambda d:hashlib.sha256((organism["organism_id"]+d).encode()).hexdigest())[:4]
    if not labels: raise ValueError("core organism has no question-supported dimension")
    return labels if len(labels)>=2 else [labels[0],labels[0]]

def append_jsonl(path: Path,value: dict[str,Any]):
    with path.open("a",encoding="utf-8") as f: f.write(compact(value)+"\n");f.flush();os.fsync(f.fileno())

def run_call(prompt: str,call_id: str,attempt: int,model_version: str,cache_dir: Path,raw_log: Path,
             cache_only: bool,generator_ref: list[LocalGenerator|None],model_path: Path):
    ph=prompt_hash(prompt);cache_path=cache_dir/f"{model_version}-{ph}.json"
    if cache_path.exists():
        envelope=json.loads(cache_path.read_text())
        raw_record=envelope["raw_record"]
        if raw_record["prompt_hash"]!=ph or raw_record["model_version"]!=model_version: raise ValueError("cache identity mismatch")
    else:
        if cache_only: raise FileNotFoundError(f"cache miss: {ph}")
        if generator_ref[0] is None: generator_ref[0]=LocalGenerator(model_path)
        raw=generator_ref[0].generate(prompt)
        raw_record={"call_id":call_id,"prompt_hash":ph,"prompt_text":prompt,"raw_response":raw,"model_version":model_version,
          "temperature":0.0,"top_p":1.0,"timestamp_utc":datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),"attempt_index":attempt}
        # Persist the measurement before parsing or validation.
        cache_path.write_text(compact({"raw_record":raw_record,"parsed_output":None,"validation":None})+"\n",encoding="utf-8")
    append_jsonl(raw_log,raw_record)
    try:
        parsed=extract_object(raw_record["raw_response"]);parse_error=None
    except (ValueError,json.JSONDecodeError) as e:
        parsed=None;parse_error=f"parse: {type(e).__name__}: {e}"
    return cache_path,raw_record,parsed,parse_error

def main() -> int:
    ap=argparse.ArgumentParser();ap.add_argument("--request-json",type=Path,required=True);ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--rejection-ledger",type=Path,required=True);ap.add_argument("--raw-log",type=Path,required=True);ap.add_argument("--cache-dir",type=Path,required=True)
    ap.add_argument("--model",type=Path,default=Path("/workspace/models/wmdp/zephyr-7b-beta_BASE"));ap.add_argument("--max-retries",type=int,choices=(0,1),default=1)
    ap.add_argument("--cache-only",action="store_true");a=ap.parse_args()
    request=json.loads(a.request_json.read_text());required={"schema_version","allowed_dimensions","value_spaces","chapters"}
    if set(request)!=required or request["schema_version"]!="spec04r-request-v1": raise ValueError("request schema mismatch")
    a.cache_dir.mkdir(parents=True,exist_ok=True);a.output.parent.mkdir(parents=True,exist_ok=True)
    for p in (a.raw_log,a.rejection_ledger,a.output):
        if p.exists(): p.unlink()
    model_version=pinned_model_version(a.model);generator:[LocalGenerator|None]=[None];world=[];ledger=[];shared={}
    for chapter in request["chapters"]:
        for organism in chapter["organisms"]:
            labels=proposal_plan(organism)
            attrs=[]
            for proposal,dimension in enumerate(labels):
                rejection=None;accepted=None;attempt_rows=[]
                for attempt in range(a.max_retries+1):
                    retained=[x["counterfactual_value"] for x in attrs if x["dimension"]==dimension]
                    prompt=build_prompt(chapter,organism,dimension,request["value_spaces"][dimension],shared,rejection,proposal,retained)
                    cache_path,raw_record,parsed,error=run_call(prompt,f"{organism['organism_id']}-p{proposal}-a{attempt}",attempt,model_version,a.cache_dir,a.raw_log,a.cache_only,generator,a.model)
                    if error is None:
                        try:
                            accepted=validate(parsed,organism,dimension,request["value_spaces"][dimension])
                            if dimension in {x["dimension"] for x in attrs}: raise ValueError("duplicate_dimension: one alteration per dimension is already retained")
                            outcome="accepted";detail="accepted"
                        except ValueError as e: accepted=None;outcome="rejected";detail=str(e)
                    else: outcome="malformed";detail=error
                    final=outcome=="accepted" or attempt==a.max_retries
                    row={"organism_id":organism["organism_id"],"proposal_index":proposal,"attempt_index":attempt,"proposed_dimension":dimension,
                      "cited_question_ids":compact(parsed.get("grounding_question_ids",[]) if isinstance(parsed,dict) else []),"outcome":outcome,"outcome_detail":detail,"final":str(final).lower()}
                    ledger.append(row);attempt_rows.append(row)
                    envelope=json.loads(cache_path.read_text());envelope["parsed_output"]=parsed;envelope["validation"]={"outcome":outcome,"detail":detail};cache_path.write_text(compact(envelope)+"\n")
                    if accepted is not None: break
                    rejection=detail
                if accepted is not None:
                    accepted=dict(accepted);accepted["proposal_index"]=proposal;attrs.append(accepted)
                    if dimension in {"reservoir_host","transmission_route","geographic_distribution"}:
                        shared.setdefault(dimension,{})[organism["organism_id"]]=accepted["counterfactual_value"]
            world.append({"organism_id":organism["organism_id"],"name_normalized":organism["name_normalized"],"attributes":attrs,"model_version":model_version})
    a.output.write_text("".join(compact(x)+"\n" for x in world),encoding="utf-8")
    with a.rejection_ledger.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,LEDGER_FIELDS,lineterminator="\n");w.writeheader();w.writerows(ledger)
    if sum(1 for _ in a.raw_log.open())!=len(ledger): raise AssertionError("raw/ledger reconciliation failed")
    return 0
if __name__=="__main__": raise SystemExit(main())
