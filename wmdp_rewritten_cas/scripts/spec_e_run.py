#!/usr/bin/env python3
"""Authorized retain-only/unfiltered-cb real pilot through step 2."""
from __future__ import annotations
import argparse, gc, hashlib, json, os, random, sys, tempfile, time
from pathlib import Path
import numpy as np, torch
from datasets import load_dataset, concatenate_datasets
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments
sys.path.insert(0,str(Path(__file__).resolve().parent))
from spec_e_common import atomic_json, exact_map_fisher_items, parse_question_inventory, read_config, retain_only_accuracy_views, sha256_file, validate_prediction_rows
sys.path.insert(0,"/workspace/CB_probes")
from prompt_utils import DATASETS, LETTER_TOKEN_IDS, build_prompt

class Rows(Dataset):
 def __init__(self,rows): self.rows=rows
 def __len__(self): return len(self.rows)
 def __getitem__(self,i): return self.rows[i]
class CompletionCollator:
 def __init__(self,tok): self.tok=tok
 def __call__(self,rows):
  n=max(len(x['input_ids']) for x in rows); ids=[]; masks=[]; labels=[]
  for x in rows:
   k=n-len(x['input_ids']); ids.append(x['input_ids']+[self.tok.pad_token_id]*k); masks.append([1]*len(x['input_ids'])+[0]*k); labels.append(x['labels']+[-100]*k)
  return {k:torch.tensor(v) for k,v in {'input_ids':ids,'attention_mask':masks,'labels':labels}.items()}
def atomic_jsonl(path,rows):
 path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); fd,tmp=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
 try:
  with os.fdopen(fd,'w') as f:
   for x in rows:f.write(json.dumps(x,sort_keys=True,separators=(',',':'))+'\n')
   f.flush();os.fsync(f.fileno())
  os.replace(tmp,path)
 finally:
  if os.path.exists(tmp):os.unlink(tmp)
def resolve_targets(model):
 suffixes=('query_key_value','attention.dense','dense_h_to_4h','dense_4h_to_h'); targets=[]
 for name,module in model.named_modules():
  if isinstance(module,torch.nn.Linear) and any(name.endswith(x) for x in suffixes): targets.append(name)
 counts={s:sum(n.endswith(s) for n in targets) for s in suffixes}
 if counts != {s:32 for s in suffixes}: raise RuntimeError(f'LoRA target resolution failed: {counts}')
 return sorted(targets)
def make_train(tok,examples,seed):
 order=list(range(len(examples))); random.Random(seed).shuffle(order); rows=[]
 for i in order:
  x=examples[i]; prompt=build_prompt(x); answer=' '+'ABCD'[int(x['answer'])]
  p=tok(prompt,add_special_tokens=False)['input_ids']; full=tok(prompt+answer,add_special_tokens=False)['input_ids']
  if full[:len(p)]!=p or len(full)!=len(p)+1: raise RuntimeError('completion tokenization parity failed')
  rows.append({'input_ids':full,'labels':[-100]*len(p)+full[len(p):]})
 return rows
@torch.inference_mode()
def score(model,tok,examples,run_id,step,batch=32):
 encoded=[tok(build_prompt(x),add_special_tokens=False)['input_ids'] for x in examples]; order=sorted(range(len(examples)),key=lambda i:(len(encoded[i]),i)); letters=torch.tensor([LETTER_TOKEN_IDS[x] for x in 'ABCD'],device=model.device); result=[None]*len(examples); model.eval(); base=model.get_base_model()
 for start in range(0,len(order),batch):
  chosen=order[start:start+batch]; seq=[torch.tensor(encoded[i],dtype=torch.long) for i in chosen]; input_ids=torch.nn.utils.rnn.pad_sequence(seq,batch_first=True,padding_value=tok.pad_token_id).to(model.device); lengths=torch.tensor([len(x) for x in seq],device=model.device); mask=(torch.arange(input_ids.shape[1],device=model.device)[None,:] < lengths[:,None]).long(); ix=torch.arange(len(chosen),device=model.device); last=lengths-1
  if not bool((input_ids[ix,last]==27).all()): raise RuntimeError('colon read position parity failed')
  with torch.autocast('cuda',dtype=torch.bfloat16): hidden=base.gpt_neox(input_ids=input_ids,attention_mask=mask).last_hidden_state[ix,last]; logits=base.embed_out(hidden)[:,letters]
  probs=torch.softmax(logits.float(),-1).cpu().numpy().astype(np.float32)
  for original,prob in zip(chosen,probs):
   x=examples[original]; result[original]={'run_id':run_id,'checkpoint_step':step,'question_id':x['_stable_id'],'p_A':float(prob[0]),'p_B':float(prob[1]),'p_C':float(prob[2]),'p_D':float(prob[3]),'gold':'ABCD'[int(x['answer'])]}
 model.train(); return result

def accuracy(rows): return sum(max('ABCD',key=lambda z:r['p_'+z])==r['gold'] for r in rows)/len(rows)
def bootstrap_ci(rows,seed,n=1000):
 x=np.array([max('ABCD',key=lambda z:r['p_'+z])==r['gold'] for r in rows],dtype=float); rng=np.random.default_rng(seed); vals=[rng.choice(x,len(x),replace=True).mean() for _ in range(n)]; return [float(np.quantile(vals,.025)),float(np.quantile(vals,.975))]
def load_eval(c,root):
 robust=DATASETS['wmdp_bio_robust']('robust'); rng=np.random.default_rng(c['wmdp']['fisher_sample_seed']); idx=np.sort(rng.choice(len(robust),512,False)); stable,_=parse_question_inventory(root/c['inputs']['question_inventory']); raw=[robust[int(i)] for i in idx]; mapped=exact_map_fisher_items(raw,stable,idx); w=[]
 for x,m in zip(raw,mapped): y=dict(x);y['_stable_id']=m['question_id'];w.append(y)
 allm=load_dataset('cais/mmlu','all',split='test'); seen={}; general=[]
 for x in allm:
  subject=x['subject']; n=seen.get(subject,0); seen[subject]=n+1
  if n>=c['evaluation']['general_cap_per_subtask']:continue
  y=dict(x);y['_stable_id']=f'mmlu-{subject}-{n:04d}';general.append(y)
 return w,general
class EvalCallback(TrainerCallback):
 def __init__(self,tok,wmdp,general,out,run_id,checkpoints,condition,validation_ids): self.tok=tok;self.w=wmdp;self.g=general;self.out=Path(out);self.run_id=run_id;self.checkpoints=set(checkpoints);self.condition=condition;self.validation_ids=validation_ids
 def evaluate(self,model,step):
  pred=score(model,self.tok,self.w,self.run_id,step); validate_prediction_rows(pred,[step]); p=self.out/'predictions'/f'step-{step:03d}.jsonl';atomic_jsonl(p,pred)
  repeats=[]
  for repeat in range(3):
   rows=score(model,self.tok,self.g,self.run_id,step); repeats.append({'repeat':repeat,'accuracy':accuracy(rows),'bootstrap_95':bootstrap_ci(rows,10000+step*10+repeat),'n':len(rows)})
  bio=[r for r in rows if r['question_id'].startswith('mmlu-college_biology-') or r['question_id'].startswith('mmlu-high_school_biology-')]
  result={'step':step,'wmdp_accuracy':accuracy(pred),'wmdp_rows':len(pred),'wmdp_sha256':sha256_file(p),'general_repeats':repeats,'mmlu_bio_accuracy_last_repeat':accuracy(bio),'mmlu_bio_n':len(bio)}
  if self.condition=='retain-only': result.update(retain_only_accuracy_views(pred,self.validation_ids))
  else: result.update({'wmdp_accuracy_full_512':None,'wmdp_rows_full_512':None,'wmdp_accuracy_V_102':accuracy(pred),'wmdp_rows_V_102':102,'wmdp_primary_view':'V_102','wmdp_like_for_like_view':'V_102'})
  atomic_json(self.out/'metrics'/f'step-{step:03d}.json',result);return result
 def on_step_end(self,args,state,control,model=None,**kwargs):
  step=int(state.global_step)
  if step in self.checkpoints: self.evaluate(model,step)
  return control

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',default='configs/spec_e.preflight.json');ap.add_argument('--output-root',default='spec_e_outputs/grid');ap.add_argument('--model',required=True);ap.add_argument('--condition',choices=['forget-T','retain-only'],required=True);ap.add_argument('--learning-rate',type=float,required=True);ap.add_argument('--seed',type=int,choices=[0,1],required=True);ap.add_argument('--max-steps',type=int,default=512);a=ap.parse_args();root=Path(__file__).resolve().parents[1];c=read_config(root/a.config)
 spec=next((x for x in c['models'] if x['name']==a.model),None)
 if spec is None or a.learning_rate not in c['training']['learning_rates']:raise RuntimeError('cell outside frozen grid')
 run_id=f"spec-e__{a.model}__{a.condition}__lr-{a.learning_rate:.0e}__seed-{a.seed}";out=root/a.output_root/run_id
 if (out/'manifest.json').is_file() and json.load(open(out/'manifest.json')).get('status')=='completed':print(json.dumps({'status':'already_completed','run_id':run_id}));return
 out.mkdir(parents=True,exist_ok=True);seed=a.seed;lr=a.learning_rate;torch.manual_seed(seed);np.random.seed(seed);random.seed(seed);checkpoints=[x for x in c['training']['checkpoint_steps'] if x<=a.max_steps]
 manifest={'status':'running','run_id':run_id,'model':spec,'condition':a.condition,'seed':seed,'learning_rate':lr,'max_steps':a.max_steps,'gradient_precision':'fp32','evaluation_precision':'bf16 autocast; fp32 softmax and stored probabilities','batch_size':4,'optimizer':'AdamW','scheduler':'linear','warmup_ratio':.05,'weight_decay':.01,'gradient_clipping':1.0,'loss_masking':'completion-only','lora_rank':256,'lora_alpha':512,'collateral_repeats':3,'general_set':'MMLU-full capped 2000/subtask','checkpoint_steps':checkpoints,'config_sha256':sha256_file(root/a.config),'started_unix':time.time()};atomic_json(out/'manifest.json',manifest)
 try:
  tok=AutoTokenizer.from_pretrained(spec['local_path'],revision=spec['revision'],local_files_only=True,use_fast=True);tok.pad_token=tok.eos_token;tok.padding_side='right'
  if [tok(' '+x,add_special_tokens=False)['input_ids'][-1] for x in 'ABCD'] != [LETTER_TOKEN_IDS[x] for x in 'ABCD']:raise RuntimeError('letter parity')
  model=AutoModelForCausalLM.from_pretrained(spec['local_path'],revision=spec['revision'],dtype=torch.float32,local_files_only=True);targets=resolve_targets(model);manifest['resolved_target_modules']=targets;manifest['resolved_target_count']=len(targets);atomic_json(out/'manifest.json',manifest)
  model=get_peft_model(model,LoraConfig(r=256,lora_alpha=512,lora_dropout=0.0,bias='none',task_type='CAUSAL_LM',target_modules=targets));model.to('cuda')
  if any(p.requires_grad and p.dtype!=torch.float32 for p in model.parameters()):raise RuntimeError('non-fp32 trainable parameter')
  w_all,g=load_eval(c,root); by_id={x['_stable_id']:x for x in w_all}
  if a.condition=='retain-only': train_source=DATASETS['mmlu_bio']('test');w_eval=w_all
  else:
   split_t=json.load(open(root/'spec_e_outputs/preflight/wmdp_split_T.json'))['items'];split_v=json.load(open(root/'spec_e_outputs/preflight/wmdp_split_V.json'))['items'];train_source=[by_id[x['question_id']] for x in split_t];w_eval=[by_id[x['question_id']] for x in split_v]
  validation_ids={x['question_id'] for x in json.load(open(root/'spec_e_outputs/preflight/wmdp_split_V.json'))['items']}
  train=make_train(tok,train_source,seed);cb=EvalCallback(tok,w_eval,g,out,run_id,checkpoints,a.condition,validation_ids);cb.evaluate(model,0)
  ta=TrainingArguments(output_dir=str(out/'trainer'),max_steps=a.max_steps,per_device_train_batch_size=4,gradient_accumulation_steps=1,learning_rate=lr,lr_scheduler_type='linear',warmup_ratio=.05,weight_decay=.01,max_grad_norm=1.0,fp16=False,bf16=False,save_strategy='no',logging_steps=1,report_to=[],seed=seed,data_seed=seed,remove_unused_columns=False)
  result=Trainer(model=model,args=ta,train_dataset=Rows(train),data_collator=CompletionCollator(tok),callbacks=[cb]).train();model.save_pretrained(out/f'adapter-step-{a.max_steps:03d}',safe_serialization=True);manifest.update({'status':'completed','completed_unix':time.time(),'train_metrics':result.metrics,'adapter_sha256':sha256_file(out/f'adapter-step-{a.max_steps:03d}'/'adapter_model.safetensors')});atomic_json(out/'manifest.json',manifest)
 except BaseException as exc:
  manifest.update({'status':'failed','failed_unix':time.time(),'failure_type':type(exc).__name__,'failure_reason':str(exc)[:2000]});atomic_json(out/'manifest.json',manifest);raise
if __name__=='__main__':main()
