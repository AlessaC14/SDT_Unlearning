#!/usr/bin/env python3
"""Constrained local-LLM organism and topic passes for Spec 03."""
from __future__ import annotations
import argparse, hashlib, json, re
from pathlib import Path
from domain_inventory_llm import MODEL_DEFAULT, SUBSETS, load_wmdp, model_version, sha256

def norm_space(x): return " ".join(x.split())
def flattened(data): return [(s,i,r) for s in SUBSETS for i,r in enumerate(data[s])]
def question_text(row): return row["question"]+"\nChoices: "+" | ".join(row["choices"])

class ForcedClassifier:
    def __init__(self,model_path,cache_dir):
        import torch
        from transformers import AutoModelForCausalLM,AutoTokenizer
        self.torch=torch;self.path=model_path.resolve();self.cache=cache_dir.resolve();self.cache.mkdir(parents=True,exist_ok=True);self.version=model_version(self.path)
        self.tokenizer=AutoTokenizer.from_pretrained(self.path,local_files_only=True)
        if self.tokenizer.pad_token_id is None:self.tokenizer.pad_token_id=self.tokenizer.eos_token_id
        self.model=AutoModelForCausalLM.from_pretrained(self.path,local_files_only=True,dtype=torch.bfloat16,device_map="cuda").eval()
    def classify_many(self,jobs,batch_size=24):
        results={};pending=[]
        for jid,prompt,choices in jobs:
            ph=hashlib.sha256(prompt.encode()).hexdigest();key=hashlib.sha256((self.version+"\0"+ph).encode()).hexdigest();path=self.cache/f"{key}.json"
            if path.is_file():
                rec=json.loads(path.read_text())
                if rec.get("model_version")!=self.version or rec.get("prompt_hash")!=ph or rec.get("choices")!=choices:raise SystemExit(f"cache mismatch: {path}")
                results[jid]=rec
            else:pending.append((jid,prompt,choices,ph,path))
        for start in range(0,len(pending),batch_size):
            batch=pending[start:start+batch_size];enc=self.tokenizer([x[1] for x in batch],padding=True,return_tensors="pt",add_special_tokens=True);enc={k:v.to(self.model.device) for k,v in enc.items()}
            with self.torch.inference_mode():logits=self.model(**enc).logits.float()
            pos=enc["attention_mask"].sum(1)-1
            for row,(jid,prompt,choices,ph,path) in enumerate(batch):
                ids=[]
                for choice in choices:
                    token_ids=self.tokenizer.encode(choice,add_special_tokens=False)
                    if len(token_ids)!=1:raise SystemExit(f"choice not one token: {choice}")
                    ids.append(token_ids[0])
                vals=logits[row,pos[row],ids].tolist();ranked=sorted(zip(choices,vals),key=lambda x:(-x[1],x[0]));rec={"model_version":self.version,"prompt_hash":ph,"choices":choices,"ranked_logits":[{"choice":c,"logit":v} for c,v in ranked]};path.write_text(json.dumps(rec,indent=2,sort_keys=True)+"\n");results[jid]=rec
        return results

def organism_candidates(row):
    text=question_text(row);found=set()
    for m in re.finditer(r"\b(?:[A-Z][a-z]{2,}|[A-Z]\.)\s+[a-z][a-z-]{2,}(?:\s+(?:virus|bacterium|strain))?\b",text):found.add(m.group())
    for m in re.finditer(r"\b(?:[A-Za-z0-9][A-Za-z0-9.'’-]*\s+){0,3}(?:virus|viruses|bacterium|bacteria|fungus|fungi|parasite|prion|pathogen|pathogens)\b",text,re.I):found.add(norm_space(m.group()))
    for token in re.findall(r"\b[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)+\b|\b[A-Z]{2,}[A-Z0-9]*\b",text):
        if len(token)>=3:found.add(token)
    return sorted(x.strip(" ,.;:()[]{}") for x in found if 2<len(x.strip(" ,.;:()[]{}"))<=80)

def extract_organisms(wmdp_path,model_path,cache_dir,output):
    data=load_wmdp(wmdp_path);clf=ForcedClassifier(model_path,cache_dir);jobs=[];meta={}
    for i,row in enumerate(data["wmdp-bio"]):
        qid=f"wmdp-bio-{i:04d}"
        for n,candidate in enumerate(organism_candidates(row)):
            jid=f"{qid}:{n}";prompt="Decide whether the candidate is a genus, species, strain, named pathogen, or informal organism name explicitly mentioned in this public benchmark item. Respond Yes or No.\nItem: "+question_text(row)+"\nCandidate: "+candidate+"\nAnswer: ";jobs.append((jid,prompt,["Yes","No"]));meta[jid]=(qid,candidate)
    scored=clf.classify_many(jobs);mentions=[]
    for jid in sorted(scored):
        qid,candidate=meta[jid];ranked=scored[jid]["ranked_logits"]
        if ranked[0]["choice"]=="Yes":mentions.append({"question_id":qid,"name_raw":candidate,"extraction_source":"llm","verification":"verbatim_candidate","yes_logit":next(x["logit"] for x in ranked if x["choice"]=="Yes"),"no_logit":next(x["logit"] for x in ranked if x["choice"]=="No")})
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text("".join(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(",",":"))+"\n" for x in mentions));output.with_suffix(".meta.json").write_text(json.dumps({"model_version":clf.version,"method":"forced Yes/No next-token classification of deterministic verbatim candidates","candidate_count":len(jobs),"accepted_mention_count":len(mentions),"hallucinated_extraction_count":0},indent=2,sort_keys=True)+"\n");print(json.dumps({"candidates":len(jobs),"mentions":len(mentions),"output":str(output)},sort_keys=True))

def assign_topics(wmdp_path,taxonomy_path,model_path,cache_dir,output):
    data=load_wmdp(wmdp_path);tax=json.loads(taxonomy_path.read_text())
    if not str(tax.get("approval_status","")).startswith("approved_"):raise SystemExit("taxonomy not approved")
    labels=tax["labels"];children={x["topic_id"]:[y for y in labels if y["parent_topic_id"]==x["topic_id"]] for x in labels if x["level"]=="broad"};roots={"wmdp-bio":"TB-BIO","wmdp-chem":"TB-CHEM","wmdp-cyber":"TB-CYBER"};codes=list("ABCDEFGHIJKLMNO");clf=ForcedClassifier(model_path,cache_dir);jobs=[];meta={}
    for subset,i,row in flattened(data):
        options=children[roots[subset]];mapping={codes[j]:x["topic_id"] for j,x in enumerate(options)};listing="\n".join(f"{c}: {next(x['topic_label'] for x in options if x['topic_id']==t)} — {next(x['definition'] for x in options if x['topic_id']==t)}" for c,t in mapping.items());qid=f"{subset}-{i:04d}";prompt="Classify this public benchmark item by technical subject, without answering it. Choose the single best label code.\n"+listing+"\nItem: "+question_text(row)+"\nLabel: ";jobs.append((qid,prompt,list(mapping)));meta[qid]=(roots[subset],mapping)
    scored=clf.classify_many(jobs);rows=[]
    for qid in sorted(scored):
        root,mapping=meta[qid];ranked=scored[qid]["ranked_logits"];specific=[mapping[x["choice"]] for x in ranked[:3]];rows.append({"question_id":qid,"primary_topic_id":specific[0],"secondary_topic_ids":specific[1:3],"broad_topic_id":root,"ranked_logits":[{"topic_id":mapping[x["choice"]],"logit":x["logit"]} for x in ranked]})
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text("".join(json.dumps(x,sort_keys=True,separators=(",",":"))+"\n" for x in rows));output.with_suffix(".meta.json").write_text(json.dumps({"model_version":clf.version,"method":"forced closed-label next-token classification","taxonomy_sha256":sha256(taxonomy_path),"question_count":len(rows)},indent=2,sort_keys=True)+"\n");print(json.dumps({"assignments":len(rows),"output":str(output)},sort_keys=True))

def main():
    p=argparse.ArgumentParser();p.add_argument("command",choices=["extract-organisms","assign-topics"]);p.add_argument("--wmdp-path",required=True,type=Path);p.add_argument("--model",type=Path,default=MODEL_DEFAULT);p.add_argument("--cache-dir",required=True,type=Path);p.add_argument("--output",required=True,type=Path);p.add_argument("--taxonomy",type=Path);a=p.parse_args()
    if a.command=="extract-organisms":extract_organisms(a.wmdp_path.resolve(),a.model.resolve(),a.cache_dir.resolve(),a.output.resolve())
    else:
        if a.taxonomy is None:raise SystemExit("--taxonomy required")
        assign_topics(a.wmdp_path.resolve(),a.taxonomy.resolve(),a.model.resolve(),a.cache_dir.resolve(),a.output.resolve())
if __name__=="__main__":main()
