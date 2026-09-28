#!/usr/bin/env python3
"""Parallel, restartable three-tier Spec H decomposition over completed Spec E."""
from __future__ import annotations
import argparse,hashlib,json,math,os
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
try:
 from .spec_h_common import mutual_information,conditional_mutual_information,solve_null_p,symmetric_random_predictions
except ImportError:
 from spec_h_common import mutual_information,conditional_mutual_information,solve_null_p,symmetric_random_predictions
LETTERS="ABCD"
def read(path):
 return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
def label(row): return max(range(4),key=lambda i:(float(row[f"p_{LETTERS[i]}"]),-i))
def aligned(pred,ref):
 r={x["question_id"]:x for x in ref};out=[]
 for x in pred:
  y=r.get(x["question_id"])
  if y is None or y["gold"]!=x["gold"]:raise ValueError("question/gold alignment failure")
  out.append(y)
 if len(out)!=len(pred) or len({x["question_id"] for x in pred})!=len(pred):raise ValueError("ID cardinality failure")
 return out
def calc(f,l,y,mm=True):
 raw,rc=mutual_information(f,y,mm); cond,cc=conditional_mutual_information(f,y,l,mm);mu=raw-cond
 return {"i_f_y":raw,"i_f_y_given_l":cond,"mu":mu,"ratio":None if abs(raw)<1e-12 else mu/raw,
         "raw_mm_correction":rc,"conditional_mm_correction":cc,"mu_mm_correction":rc-cc}
def percentile(values):
 a=np.asarray(values,float);return [float(np.quantile(a,.025)),float(np.quantile(a,.975))]
def task(job):
 pred=read(job["prediction"]);ref=aligned(pred,read(job["reference"]))
 f=np.array([label(x) for x in pred],dtype=np.int8);l=np.array([label(x) for x in ref],dtype=np.int8)
 y=np.array([LETTERS.index(x["gold"]) for x in pred],dtype=np.int8);lc=(l==y).astype(np.int8)
 tier=job["tier"]; z=lc if tier=="tier_2" else l
 if tier=="tier_1":
  pv,_=mutual_information(f,y,False); mm,corr=mutual_information(f,y,True)
  point={"i_f_y_plugin":pv,"i_f_y_miller_madow":mm,"raw_mm_correction":corr}
 elif tier=="tier_2":
  old=calc(f,z,y,True); plug=calc(f,z,y,False)
  point={"reporting_status":"diagnostic_only_information_associated_with_reference_correctness_pattern",
   "i_f_y_miller_madow":old["i_f_y"],"i_f_y_given_reference_correctness_miller_madow":old["i_f_y_given_l"],
   "information_associated_with_reference_correctness_pattern_miller_madow":old["mu"],"legacy_ratio_unreported":old["ratio"],
   "i_f_y_plugin":plug["i_f_y"],"i_f_y_given_reference_correctness_plugin":plug["i_f_y_given_l"],
   "raw_mm_correction":old["raw_mm_correction"],"conditional_mm_correction":old["conditional_mm_correction"]}
 else:
  point=calc(f,z,y,True); point["plugin"]=calc(f,z,y,False)
 rng=np.random.default_rng(job["seed"])
 if tier=="tier_2": boot={k:[] for k in ("i_f_y_miller_madow","i_f_y_given_reference_correctness_miller_madow","information_associated_with_reference_correctness_pattern_miller_madow")}
 elif tier=="tier_1": boot={"i_f_y_plugin":[],"i_f_y_miller_madow":[]}
 else: boot={k:[] for k in ("i_f_y","i_f_y_given_l","mu","ratio") if point[k] is not None}
 n=len(y)
 for _ in range(job["boot"]):
  ix=rng.integers(0,n,n)
  if tier=="tier_1": b={"i_f_y_plugin":mutual_information(f[ix],y[ix],False)[0],"i_f_y_miller_madow":mutual_information(f[ix],y[ix],True)[0]}
  elif tier=="tier_2":
   q=calc(f[ix],z[ix],y[ix],True); b={"i_f_y_miller_madow":q["i_f_y"],"i_f_y_given_reference_correctness_miller_madow":q["i_f_y_given_l"],"information_associated_with_reference_correctness_pattern_miller_madow":q["mu"]}
  else: b=calc(f[ix],z[ix],y[ix],True)
  for k in boot:
   if b.get(k) is not None and math.isfinite(b[k]):boot[k].append(b[k])
 cis={k:percentile(v) for k,v in boot.items()}
 null={"applicable":False}
 if tier=="tier_2":
  target=mutual_information(lc,y,False)[0]
  null={"applicable":True,"matched":abs(target)<=job["tol"],"target_i_l_y":target,"achieved_i_l_y":0.0,
        "absolute_mismatch":abs(target),"reason":"scalar symmetric correctness null is independent of Y"}
 elif tier=="tier_3":
  target=mutual_information(l,y,False)[0];p,expected=solve_null_p(y,target,job["tol"]);ratios=[];matches=[]
  import random
  for draw in range(job["null_draws"]):
   nl=np.array(symmetric_random_predictions(y,p,random.Random(job["seed"]+100000+draw)),dtype=np.int8)
   matches.append(mutual_information(nl,y,False)[0]);ratios.append(calc(f,nl,y,True)["ratio"])
  vals=[x for x in ratios if x is not None]
  null={"applicable":True,"matched_expected":abs(expected-target)<=job["tol"],"p_correct":p,
        "target_i_l_y":target,"expected_i_null_y":expected,"draw_i_null_y":matches,
        "maximum_draw_absolute_mismatch":max(abs(x-target) for x in matches),
        "ratio_mean":float(np.mean(vals)) if vals else None,"ratio_sd":float(np.std(vals,ddof=1)) if len(vals)>1 else None}
 out={**job,"n":n,"point":point,"bootstrap_95":cis,"null":null};out.pop("boot");out.pop("tol");out.pop("null_draws")
 shard=Path(job["shard"]);shard.parent.mkdir(parents=True,exist_ok=True);tmp=shard.with_suffix(".tmp");tmp.write_text(json.dumps(out,sort_keys=True)+"\n");os.replace(tmp,shard);return str(shard)
def main():
 p=argparse.ArgumentParser();p.add_argument("--workers",type=int,default=48);p.add_argument("--config",default="configs/spec_h.json");p.add_argument("--output",default="spec_h_outputs/final");a=p.parse_args()
 root=Path(__file__).resolve().parents[1];cfg=json.loads((root/a.config).read_text());grid=root/"spec_e_outputs/grid";out=root/a.output;shards=out/"shards";jobs=[]
 manifests=sorted(grid.glob("spec-e__*/manifest.json"))
 if len(manifests)!=36 or any(json.loads(x.read_text()).get("status")!="completed" for x in manifests):raise SystemExit("grid incomplete")
 for mp in manifests:
  man=json.loads(mp.read_text());d=mp.parent;model=man["model"]["name"];condition=man["condition"]
  refs={"intact":root/"spec_e_outputs/references_grid_path/unfiltered/full.jsonl",
        "shortcut":root/f"spec_e_outputs/references_grid_path/{model}/options-only.jsonl"}
  for pp in sorted((d/"predictions").glob("step-*.jsonl")):
   step=int(pp.stem.split("-")[1]); tiers=["tier_1"] if condition=="forget-T" else ["tier_1","tier_2","tier_3"]
   for tier in tiers:
    names=["none"] if tier=="tier_1" else list(refs)
    for name in names:
     key=f"{d.name}__step-{step:03d}__{tier}__{name}";seed=int(hashlib.sha256(key.encode()).hexdigest()[:8],16)
     shard=shards/(key+".json")
     if shard.exists():continue
     jobs.append({"run_id":d.name,"model":model,"condition":condition,"step":step,"tier":tier,"reference_name":name,
      "prediction":str(pp),"reference":str(refs.get(name,refs["intact"])),"seed":seed,"boot":cfg["bootstrap_resamples"],
      "null_draws":cfg["null_draws"],"tol":cfg["null_mi_tolerance_bits"],"shard":str(shard)})
 print(json.dumps({"jobs":len(jobs),"workers":a.workers}),flush=True)
 failures=[]
 with ProcessPoolExecutor(max_workers=a.workers) as ex:
  futures={ex.submit(task,j):j for j in jobs}
  for i,f in enumerate(as_completed(futures),1):
   try:f.result()
   except Exception as e:failures.append({"job":futures[f],"error":repr(e)})
   if i%50==0:print(json.dumps({"finished":i,"failures":len(failures)}),flush=True)
 if failures:(out/"failures.json").write_text(json.dumps(failures,indent=2));raise SystemExit(2)
 records=[json.loads(x.read_text()) for x in sorted(shards.glob("*.json"))]
 (out/"info_decomposition.json").write_text(json.dumps({"status":"complete","records":records},indent=2,sort_keys=True)+"\n")
 lines=["# Spec H information decomposition","","Status: complete",f"Records: {len(records)}","",
  "Tier 1 is native four-way I(F;Y). Tier 2 is retained only as diagnostic information associated with the pattern of reference correctness; no Tier 2 mu or ratio is reported. Tier 3 conditions on the full four-way reference and is the indicative functional comparison.","",
  "Forget-T uses Tier 1 only at N=102. Retain-only uses all tiers at N=512.","",
  "Tier 2 scalar-null mismatches are retained in JSON rather than hidden; Tier 3 uses 20 matched symmetric four-way null draws.","",
  "Miller-Madow corrected Tier 3 point estimates can fall outside percentile bootstrap intervals because occupied-cell counts vary across resamples at roughly eight samples per cell. This instability is reported, not replaced by another estimator.",""]
 (out/"info_decomposition_report.md").write_text("\n".join(lines)+"\n")
 print(json.dumps({"status":"complete","records":len(records)}))
if __name__=="__main__":main()
