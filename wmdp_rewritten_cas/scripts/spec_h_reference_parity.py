#!/usr/bin/env python3
"""Hard gate: fresh exact-path references must reproduce every grid step-zero."""
import glob,json
from pathlib import Path
import numpy as np
LETTERS="ABCD"
def read(p): return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def main():
 root=Path(__file__).resolve().parents[1]; records=[]; passed=True
 for model in ("unfiltered","e2e-strong-filter","unfiltered-cb"):
  ref=read(root/f"spec_e_outputs/references_grid_path/{model}/full.jsonl"); by={x["question_id"]:x for x in ref}
  for p in sorted(glob.glob(str(root/f"spec_e_outputs/grid/spec-e__{model}__retain-only__*/predictions/step-000.jsonl"))):
   rows=read(p); deltas=[]; same=0
   for x in rows:
    r=by[x["question_id"]]; a=np.array([x[f"p_{z}"] for z in LETTERS]); b=np.array([r[f"p_{z}"] for z in LETTERS])
    deltas.extend(abs(a-b)); same += int(int(a.argmax())==int(b.argmax()))
   rec={"checkpoint":model,"run_id":Path(p).parts[-3],"n":len(rows),"argmax_self_consistency":same/len(rows),"maximum_absolute_probability_delta":float(max(deltas))}
   rec["pass"]=rec["argmax_self_consistency"]==1.0 and rec["maximum_absolute_probability_delta"]<=1e-7; passed &= rec["pass"]; records.append(rec)
 out=root/"spec_h_outputs/reference_parity"; out.mkdir(parents=True,exist_ok=True)
 payload={"status":"pass" if passed else "fail","tolerance":1e-7,"records":records}; (out/"parity.json").write_text(json.dumps(payload,indent=2)+"\n")
 lines=["Spec H regenerated-reference parity gate","",f"Status: {payload['status'].upper()}","Tolerance: 1e-7 maximum absolute probability delta",""]
 for m in ("unfiltered","e2e-strong-filter","unfiltered-cb"):
  q=[x for x in records if x["checkpoint"]==m]; lines.append(f"{m}: {len(q)}/6 cells pass; minimum argmax self-consistency {min(x['argmax_self_consistency'] for x in q):.6f}; maximum probability delta {max(x['maximum_absolute_probability_delta'] for x in q):.9g}")
 (out/"parity.txt").write_text("\n".join(lines)+"\n")
 if not passed: raise SystemExit(2)
if __name__=="__main__": main()
