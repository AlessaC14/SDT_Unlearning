#!/usr/bin/env python3
"""Spec K: question--predicted-answer MI from persisted four-way tables only."""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
import numpy as np

LETTERS = "ABCD"

def read_rows(path: Path) -> list[dict]:
    rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows: raise ValueError(f"empty table: {path}")
    ids=[str(r["question_id"]) for r in rows]
    if len(ids)!=len(set(ids)): raise ValueError(f"duplicate question IDs: {path}")
    return rows

def probabilities(rows: list[dict]) -> np.ndarray:
    p=np.asarray([[float(r[f"p_{x}"]) for x in LETTERS] for r in rows],dtype=np.float64)
    if p.shape!=(len(rows),4) or not np.all(np.isfinite(p)) or np.any(p<0):
        raise ValueError("invalid probability table")
    return p

def entropy_rows(p: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore",invalid="ignore"):
        terms=np.where(p>0,-p*np.log(p),0.0)
    return terms.sum(axis=-1)

def calculate(p: np.ndarray) -> dict:
    sums=p.sum(axis=1); max_dev=float(np.max(np.abs(sums-1.0)))
    if max_dev>1e-5: raise ValueError(f"probability normalization failure: {max_dev}")
    marginal=p.mean(axis=0); h_marg=float(entropy_rows(marginal[None,:])[0])
    h_cond=float(entropy_rows(p).mean()); mi_entropy=h_marg-h_cond
    with np.errstate(divide="ignore",invalid="ignore"):
        kl=np.where(p>0,p*(np.log(p)-np.log(marginal[None,:])),0.0).sum(axis=1)
    mi_kl=float(kl.mean()); discrepancy=abs(mi_entropy-mi_kl)
    return {"n":len(p),"marginal_p":{LETTERS[i]:float(marginal[i]) for i in range(4)},
      "h_marginal_nats":h_marg,"mean_conditional_entropy_nats":h_cond,
      "i_x_yhat_nats":mi_entropy,"i_x_yhat_bits":mi_entropy/math.log(2),
      "fraction_of_two_bit_ceiling":mi_entropy/math.log(4),"mean_kl_nats":mi_kl,
      "crosscheck_absolute_discrepancy_nats":discrepancy,
      "fraction_max_probability_gt_0_99":float(np.mean(np.max(p,axis=1)>0.99)),
      "maximum_probability_sum_deviation":max_dev,
      "sanity":{"bounds_pass":bool(-1e-12<=mi_entropy<=math.log(4)+1e-12),
        "entropy_inequality_pass":bool(h_cond<=h_marg+1e-12),
        "route_agreement_10_significant_figures_pass":bool(discrepancy<=1e-10*max(1.0,abs(mi_entropy))),
        "normalization_pass":bool(max_dev<=1e-5)}}

def bootstrap(p: np.ndarray, seed: int, count: int=1000) -> list[float]:
    rng=np.random.default_rng(seed); n=len(p); values=np.empty(count,dtype=np.float64)
    for i in range(count): values[i]=calculate(p[rng.integers(0,n,n)])["i_x_yhat_nats"]
    return [float(np.quantile(values,.025)),float(np.quantile(values,.975))]

def record(path: Path, kind: str, model: str, condition: str|None=None, step: int|None=None,
           run_id: str|None=None, bootstrap_count: int=1000) -> dict:
    rows=read_rows(path); p=probabilities(rows); key=str(path)
    seed=int(hashlib.sha256(key.encode()).hexdigest()[:8],16); value=calculate(p)
    value["bootstrap_95_nats"]=bootstrap(p,seed,bootstrap_count)
    value["bootstrap_95_bits"]=[x/math.log(2) for x in value["bootstrap_95_nats"]]
    return {"kind":kind,"model":model,"condition":condition,"step":step,"run_id":run_id,
      "table":str(path.resolve()),"question_ids":[str(r["question_id"]) for r in rows],
      "bootstrap_resamples":bootstrap_count,"bootstrap_seed":seed,**value}

def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument("--output-dir",default="spec_k_outputs"); ap.add_argument("--bootstrap",type=int,default=1000); args=ap.parse_args()
    if args.bootstrap<1000: raise SystemExit("Spec K requires >=1000 bootstrap resamples")
    root=Path(__file__).resolve().parents[1]; refs=root/"spec_e_outputs/references_grid_path"; grid=root/"spec_e_outputs/grid"; out=root/args.output_dir; out.mkdir(parents=True,exist_ok=True)
    models=["unfiltered","e2e-strong-filter","unfiltered-cb"]; pre=[]
    for model in models:
        full=record(refs/model/"full.jsonl","pre_attack_full",model,bootstrap_count=args.bootstrap)
        options=record(refs/model/"options-only.jsonl","pre_attack_options_only",model,bootstrap_count=args.bootstrap)
        if full["question_ids"]!=options["question_ids"]: raise ValueError(f"full/options ID alignment failure: {model}")
        full["paired_options_only_table"]=options["table"]; full["full_options_question_id_alignment_pass"]=True
        options["paired_full_table"]=full["table"]; options["full_options_question_id_alignment_pass"]=True
        pre.extend([full,options])
    trajectories=[]
    manifests=sorted(grid.glob("spec-e__*/manifest.json"))
    if len(manifests)!=36: raise ValueError(f"expected 36 grid manifests, observed {len(manifests)}")
    for mp in manifests:
        man=json.loads(mp.read_text()); run=mp.parent.name; model=man["model"]["name"]; cond=man["condition"]
        paths=sorted((mp.parent/"predictions").glob("step-*.jsonl"))
        if len(paths)!=11: raise ValueError(f"expected 11 checkpoints: {run}")
        for path in paths:
            step=int(path.stem.split("-")[1]); trajectories.append(record(path,"relearning",model,cond,step,run,args.bootstrap))
    if len(trajectories)!=396: raise ValueError(f"expected 396 total trajectory records, observed {len(trajectories)}")
    counts={c:sum(r["condition"]==c for r in trajectories) for c in ("retain-only","forget-T")}
    cb_full=next(r for r in pre if r["model"]=="unfiltered-cb" and r["kind"]=="pre_attack_full")
    comparisons=[]
    for model in models:
        f=next(r for r in pre if r["model"]==model and r["kind"]=="pre_attack_full")
        o=next(r for r in pre if r["model"]==model and r["kind"]=="pre_attack_options_only")
        comparisons.append({"model":model,"full_nats":f["i_x_yhat_nats"],"options_only_nats":o["i_x_yhat_nats"],"options_only_lower_than_full_pass":o["i_x_yhat_nats"]<f["i_x_yhat_nats"]})
    prediction_checks={"unfiltered_cb_pre_attack_very_low":{"value_nats":cb_full["i_x_yhat_nats"],"pass":False,"criterion":"lower than both other within-family pre-attack full values","reason":"CB has the highest, not lowest, pre-attack full MI in this family"},
      "options_only_lower_than_full_for_every_pre_attack_checkpoint":{"pass":all(x["options_only_lower_than_full_pass"] for x in comparisons),"comparisons":comparisons}}
    all_records=pre+trajectories
    audit={"pre_attack_records":len(pre),"trajectory_records_total":len(trajectories),"trajectory_records_by_condition":counts,
      "spec_text_count_resolution":"396 is the total grid count; retain-only contributes 198 and forget-T contributes 198",
      "all_sanity_checks_pass":all(all(r["sanity"].values()) for r in all_records),
      "maximum_crosscheck_discrepancy_nats":max(r["crosscheck_absolute_discrepancy_nats"] for r in all_records),
      "maximum_probability_sum_deviation":max(r["maximum_probability_sum_deviation"] for r in all_records)}
    payload={"spec":"K","status":"complete","quantity":"I(question; predicted answer) under model predictive distribution","gold_labels_used":False,
      "units":["nats","bits","fraction_of_two_bit_ceiling"],"pre_attack":pre,"trajectories":trajectories,"prediction_checks":prediction_checks,"audit":audit}
    (out/"question_answer_mi.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    lines=["# Spec K: question--predicted-answer mutual information","", "Status: complete. CPU arithmetic over persisted tables only; no model or GPU execution.","",
      "This is I(question; predicted answer) under the model's own predictive distribution. Gold labels are not used, and this is not accuracy or I(forget corpus; parameters). Low values are output-level evidence that answers do not track questions; they do not establish that capability is absent.","","## Pre-attack primary table","",
      "| model | input | N | MI (nats) | MI (bits) | ceiling fraction | H(marginal), nats | mean H(p(.|x)), nats | max-p > .99 | marginal p(A,B,C,D) | bootstrap 95% CI, bits | route discrepancy, nats |",
      "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|"]
    for r in pre:
        inp="full" if r["kind"]=="pre_attack_full" else "options-only"; m=r["marginal_p"]
        lines.append(f"| {r['model']} | {inp} | {r['n']} | {r['i_x_yhat_nats']:.6f} | {r['i_x_yhat_bits']:.6f} | {r['fraction_of_two_bit_ceiling']:.6f} | {r['h_marginal_nats']:.6f} | {r['mean_conditional_entropy_nats']:.6f} | {r['fraction_max_probability_gt_0_99']:.4f} | ({m['A']:.4f}, {m['B']:.4f}, {m['C']:.4f}, {m['D']:.4f}) | [{r['bootstrap_95_bits'][0]:.6f}, {r['bootstrap_95_bits'][1]:.6f}] | {r['crosscheck_absolute_discrepancy_nats']:.3e} |")
    lines += ["","## Prediction checks","",f"- CB pre-attack full MI near the family floor: **FAIL**. It is {cb_full['i_x_yhat_bits']:.6f} bits and is the highest of the three pre-attack full values.",f"- Options-only below full for every pre-attack checkpoint: **{'PASS' if prediction_checks['options_only_lower_than_full_for_every_pre_attack_checkpoint']['pass'] else 'FAIL'}**.","","## Trajectories","",f"Computed all {len(trajectories)} grid checkpoint tables: {counts['retain-only']} retain-only at N=512 and {counts['forget-T']} forget-T at N=102. The source phrase '396 retain-only' conflicts with the frozen 36-cell grid; 396 is the total, not the retain-only count. Full trajectory records and diagnostics follow and are also in JSON.","","| run | condition | step | N | MI bits | bootstrap 95% CI bits | H(marginal) nats | mean conditional H nats | max-p > .99 |","|---|---|---:|---:|---:|---|---:|---:|---:|"]
    for r in trajectories:
        lines.append(f"| {r['run_id']} | {r['condition']} | {r['step']} | {r['n']} | {r['i_x_yhat_bits']:.6f} | [{r['bootstrap_95_bits'][0]:.6f}, {r['bootstrap_95_bits'][1]:.6f}] | {r['h_marginal_nats']:.6f} | {r['mean_conditional_entropy_nats']:.6f} | {r['fraction_max_probability_gt_0_99']:.4f} |")
    lines += ["","## Audit","",f"- All normalization, bounds, entropy inequality, and ten-significant-figure route checks: **{'PASS' if audit['all_sanity_checks_pass'] else 'FAIL'}**.",f"- Maximum entropy-route/KL-route discrepancy: {audit['maximum_crosscheck_discrepancy_nats']:.17g} nats.",f"- Maximum probability-sum deviation: {audit['maximum_probability_sum_deviation']:.17g}.","","## Interpretation limits","","- This is computed from predictive distributions with no gold labels and is not accuracy.","- It is not I(D_forget; theta), which is not computable from these tables.","- Low MI is an output-level statement that answers no longer track questions; it does not prove the underlying capability is absent."]
    (out/"question_answer_mi_report.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"status":"complete","pre_attack":len(pre),"trajectories":len(trajectories),"audit":audit,"prediction_checks":prediction_checks},sort_keys=True))
    return 0
if __name__=="__main__": raise SystemExit(main())
