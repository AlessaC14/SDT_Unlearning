#!/usr/bin/env python3
"""Authorized Spec E step-0 and option-only baselines with Fisher parity."""
from __future__ import annotations
import argparse, gc, hashlib, json, os, sys, tempfile, time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0,str(Path(__file__).resolve().parent))
from spec_e_common import atomic_json, exact_map_fisher_items, parse_question_inventory, read_config, sha256_file
sys.path.insert(0,"/workspace/CB_probes")
from prompt_utils import DATASETS, LETTER_TOKEN_IDS, build_prompt

def atomic_jsonl(path, rows):
 path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
 fd,tmp=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
 try:
  with os.fdopen(fd,'w',encoding='utf-8') as f:
   for row in rows: f.write(json.dumps(row,sort_keys=True,separators=(',',':'))+'\n')
   f.flush(); os.fsync(f.fileno())
  os.replace(tmp,path)
 finally:
  if os.path.exists(tmp): os.unlink(tmp)

def candidate_ids(tok):
 declared=[LETTER_TOKEN_IDS[x] for x in 'ABCD']
 space=[tok(' '+x,add_special_tokens=False)['input_ids'][-1] for x in 'ABCD']
 if space != declared: raise RuntimeError(f'Fisher letter-token parity failed {space} != {declared}')
 return declared

def evaluate(model,tok,examples,ids,run_id,mode,batch_size):
 prompts=[]
 for x in examples:
  item=dict(x)
  if mode=='options-only': item['question']=''
  prompts.append(build_prompt(item))
 colon_id=tok(prompts[0],add_special_tokens=False)['input_ids'][-1]
 letter=torch.tensor(ids,device=model.device)
 rows=[]
 for start in range(0,len(prompts),batch_size):
  enc=tok(prompts[start:start+batch_size],return_tensors='pt',padding=True,add_special_tokens=False).to(model.device)
  last=enc['attention_mask'].sum(1)-1; idx=torch.arange(enc['input_ids'].shape[0],device=model.device)
  if not bool((enc['input_ids'][idx,last]==colon_id).all()): raise RuntimeError('colon read position parity failed')
  with torch.inference_mode(): logits=model(**enc).logits[idx,last][:,letter].float(); probs=torch.softmax(logits,dim=-1).cpu().numpy().astype(np.float32)
  for j,p in enumerate(probs):
   ex=examples[start+j]
   rows.append({'run_id':run_id,'checkpoint_step':0,'question_id':ex['_stable_id'],
    'p_A':float(p[0]),'p_B':float(p[1]),'p_C':float(p[2]),'p_D':float(p[3]),'gold':'ABCD'[int(ex['answer'])]})
  print(json.dumps({'run_id':run_id,'mode':mode,'done':min(start+batch_size,len(prompts)),'total':len(prompts)}),flush=True)
 return rows,colon_id

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--config',default='configs/spec_e.preflight.json'); ap.add_argument('--output-dir',default='spec_e_outputs/baselines'); ap.add_argument('--models',nargs='*'); ap.add_argument('--batch-size',type=int,default=16); a=ap.parse_args()
 root=Path(__file__).resolve().parents[1]; c=read_config(root/a.config); out=root/a.output_dir
 ds=DATASETS['wmdp_bio_robust']('robust'); rng=np.random.default_rng(c['wmdp']['fisher_sample_seed']); indices=np.sort(rng.choice(len(ds),size=512,replace=False))
 stable,_=parse_question_inventory(root/c['inputs']['question_inventory']); raw=[ds[int(i)] for i in indices]; mapped=exact_map_fisher_items(raw,stable,indices)
 examples=[]
 for x,m in zip(raw,mapped): y=dict(x); y['_stable_id']=m['question_id']; examples.append(y)
 selected=set(a.models or [m['name'] for m in c['models']])
 for spec in c['models']:
  if spec['name'] not in selected: continue
  model_out=out/spec['name']; model_out.mkdir(parents=True,exist_ok=True)
  tok=AutoTokenizer.from_pretrained(spec['local_path'],revision=spec['revision'],use_fast=True,local_files_only=True)
  if tok.pad_token is None: tok.pad_token=tok.eos_token
  tok.padding_side='right'; letters=candidate_ids(tok)
  model=AutoModelForCausalLM.from_pretrained(spec['local_path'],revision=spec['revision'],torch_dtype=torch.float32,device_map={'':0},local_files_only=True)
  model.eval(); manifest={'status':'running','model':spec,'evaluation_precision':'fp32','prediction_storage':'JSON numeric values derived from numpy.float32','question_count':512,'letter_token_ids':letters,'fisher_prompt_source':c['inputs']['fisher_prompt_source'],'fisher_prompt_source_sha256':sha256_file(c['inputs']['fisher_prompt_source']),'config_sha256':sha256_file(root/a.config),'started_unix':time.time()}; atomic_json(model_out/'manifest.json',manifest)
  for mode in ('full','options-only'):
   run_id=f"spec-e__{spec['name']}__step-0__{mode}"
   rows,colon=evaluate(model,tok,examples,letters,run_id,mode,a.batch_size)
   atomic_jsonl(model_out/f'{mode}.jsonl',rows); manifest[mode]={'rows':len(rows),'colon_token_id':colon,'sha256':sha256_file(model_out/f'{mode}.jsonl')}
   atomic_json(model_out/'manifest.json',manifest)
  manifest['status']='completed'; manifest['completed_unix']=time.time(); atomic_json(model_out/'manifest.json',manifest)
  del model,tok; gc.collect(); torch.cuda.empty_cache()
if __name__=='__main__': main()
