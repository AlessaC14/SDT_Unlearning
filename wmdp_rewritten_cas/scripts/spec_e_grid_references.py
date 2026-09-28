#!/usr/bin/env python3
"""Regenerate step-zero references through the exact Spec E grid scoring path."""
from __future__ import annotations
import argparse, gc, json, time
from pathlib import Path
import numpy as np, torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
try:
 from .spec_e_common import atomic_json, exact_map_fisher_items, parse_question_inventory, read_config, sha256_file, validate_prediction_rows
 from .spec_e_run import atomic_jsonl, resolve_targets, score
except ImportError:
 from spec_e_common import atomic_json, exact_map_fisher_items, parse_question_inventory, read_config, sha256_file, validate_prediction_rows
 from spec_e_run import atomic_jsonl, resolve_targets, score
import sys
sys.path.insert(0,"/workspace/CB_probes")
from prompt_utils import DATASETS, LETTER_TOKEN_IDS

def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--model",required=True); ap.add_argument("--device",default="cuda:0")
 ap.add_argument("--config",default="configs/spec_e.preflight.json"); ap.add_argument("--output-root",default="spec_e_outputs/references_grid_path")
 a=ap.parse_args(); root=Path(__file__).resolve().parents[1]; cfg=read_config(root/a.config)
 spec=next(x for x in cfg["models"] if x["name"]==a.model); out=root/a.output_root/a.model; out.mkdir(parents=True,exist_ok=True)
 ds=DATASETS["wmdp_bio_robust"]("robust"); rng=np.random.default_rng(cfg["wmdp"]["fisher_sample_seed"]); ix=np.sort(rng.choice(len(ds),512,False))
 stable,_=parse_question_inventory(root/cfg["inputs"]["question_inventory"]); raw=[ds[int(i)] for i in ix]; mapped=exact_map_fisher_items(raw,stable,ix)
 examples=[]
 for x,m in zip(raw,mapped): y=dict(x); y["_stable_id"]=m["question_id"]; examples.append(y)
 manifest={"status":"running","model":spec,"path":"exact_spec_e_grid_score","wrapper":"fresh zero-LoRA PEFT wrapper",
  "evaluation_precision":"bf16 autocast; fp32 softmax; numpy float32 storage","question_count":512,"started_unix":time.time()}; atomic_json(out/"manifest.json",manifest)
 tok=AutoTokenizer.from_pretrained(spec["local_path"],revision=spec["revision"],local_files_only=True,use_fast=True); tok.pad_token=tok.eos_token; tok.padding_side="right"
 if [tok(" "+x,add_special_tokens=False)["input_ids"][-1] for x in "ABCD"] != [LETTER_TOKEN_IDS[x] for x in "ABCD"]: raise RuntimeError("letter parity")
 model=AutoModelForCausalLM.from_pretrained(spec["local_path"],revision=spec["revision"],dtype=torch.float32,local_files_only=True)
 targets=resolve_targets(model); model=get_peft_model(model,LoraConfig(r=256,lora_alpha=512,lora_dropout=0.0,bias="none",task_type="CAUSAL_LM",target_modules=targets)); model.to(a.device)
 for mode in ("full","options-only"):
  current=examples if mode=="full" else [{**x,"question":""} for x in examples]
  rows=score(model,tok,current,f"spec-e-grid-reference__{a.model}__{mode}",0); validate_prediction_rows(rows,[0]); atomic_jsonl(out/f"{mode}.jsonl",rows)
  manifest[mode]={"rows":len(rows),"sha256":sha256_file(out/f"{mode}.jsonl")}; atomic_json(out/"manifest.json",manifest)
 manifest.update({"status":"completed","completed_unix":time.time(),"resolved_target_count":len(targets)}); atomic_json(out/"manifest.json",manifest)
 del model,tok; gc.collect(); torch.cuda.empty_cache()
if __name__=="__main__": main()
