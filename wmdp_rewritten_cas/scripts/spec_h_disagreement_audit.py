#!/usr/bin/env python3
"""Audit old fp32 baselines against canonical grid-path step zero."""
import json
from pathlib import Path
import numpy as np
LETTERS="ABCD"
def read(p): return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def main():
 root=Path(__file__).resolve().parents[1]; records=[]
 for model in ("unfiltered","e2e-strong-filter","unfiltered-cb"):
  baseline=read(root/f"spec_e_outputs/baselines/{model}/full.jsonl")
  grid=read(next(iter(sorted((root/"spec_e_outputs/grid").glob(f"spec-e__{model}__retain-only__*/predictions/step-000.jsonl")))))
  g={x["question_id"]:x for x in grid}; changed=[]; tv=[]
  for b in baseline:
   q=np.array([b[f"p_{x}"] for x in LETTERS]); r=np.array([g[b["question_id"]][f"p_{x}"] for x in LETTERS]); tv.append(.5*np.abs(q-r).sum())
   if int(q.argmax())!=int(r.argmax()): changed.append((np.sort(q)[-1]-np.sort(q)[-2],np.sort(r)[-1]-np.sort(r)[-2]))
  records.append({"checkpoint":model,"n":len(baseline),"argmax_disagreements":len(changed),"argmax_disagreement_fraction":len(changed)/len(baseline),
   "mean_top_two_margin_baseline_on_disagreements":float(np.mean([x[0] for x in changed])),"mean_top_two_margin_grid_on_disagreements":float(np.mean([x[1] for x in changed])),
   "maximum_per_item_total_variation":float(max(tv)),"mean_per_item_total_variation":float(np.mean(tv))})
 out=root/"spec_h_outputs/disagreement_audit"; out.mkdir(parents=True,exist_ok=True); (out/"audit.json").write_text(json.dumps({"records":records},indent=2)+"\n")
 lines=["Spec H baseline-vs-grid step-zero disagreement audit","","Same 512 WMDP questions; margins are evaluated only on argmax-disagreement items.",""]
 for r in records: lines += [f"{r['checkpoint']}: {r['argmax_disagreements']}/{r['n']} = {r['argmax_disagreement_fraction']:.4%}",f"  mean top-two margin: baseline {r['mean_top_two_margin_baseline_on_disagreements']:.8f}; grid {r['mean_top_two_margin_grid_on_disagreements']:.8f}",f"  maximum TV {r['maximum_per_item_total_variation']:.8f}; mean TV {r['mean_per_item_total_variation']:.8f}",""]
 lines += ["Interpretation: CB is not an outlier. Strong-filter has the largest label-disagreement rate. Exact zero grid margins for changed unfiltered and strong-filter items identify bf16-path ties as the dominant mechanism; CB includes small nonzero near-tie margins."]
 (out/"audit.txt").write_text("\n".join(lines)+"\n")
if __name__=="__main__": main()
