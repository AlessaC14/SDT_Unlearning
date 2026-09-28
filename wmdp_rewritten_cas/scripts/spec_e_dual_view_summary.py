#!/usr/bin/env python3
"""Report retain-only full-512 primary and frozen-V-102 like-for-like accuracy."""
import argparse,json
from pathlib import Path
try: from .spec_e_common import atomic_json
except ImportError: from spec_e_common import atomic_json
def main():
 p=argparse.ArgumentParser();p.add_argument("--grid-root",default="spec_e_outputs/grid");a=p.parse_args();root=Path(__file__).resolve().parents[1];grid=root/a.grid_root;runs=[]
 for d in sorted(grid.glob("spec-e__*__retain-only__*")):
  manifest=d/"manifest.json"
  if not manifest.is_file() or json.loads(manifest.read_text()).get("status")!="completed":continue
  checkpoints=[]
  for m in sorted((d/"metrics").glob("step-*.json")):
   x=json.loads(m.read_text())
   if "wmdp_accuracy_full_512" not in x or "wmdp_accuracy_V_102" not in x:raise RuntimeError(f"dual view absent: {m}")
   checkpoints.append({"step":x["step"],"accuracy_full_512":x["wmdp_accuracy_full_512"],"accuracy_V_102":x["wmdp_accuracy_V_102"]})
  selected=max(checkpoints,key=lambda x:(x["accuracy_full_512"],-x["step"]))
  runs.append({"run_id":d.name,"primary_checkpoint_selection":"maximum_full_512_then_earliest_step",
               "checkpoint_max_full_512":selected["accuracy_full_512"],"V_102_at_primary_max_checkpoint":selected["accuracy_V_102"],
               "primary_max_step":selected["step"],"checkpoints":checkpoints})
 atomic_json(grid/"retain_only_dual_view_summary.json",{"status":"partial" if len(runs)<18 else "complete","completed_retain_only_runs":len(runs),
  "full_512_role":"primary_for_statistical_power","V_102_role":"only_like_for_like_comparison_to_forget-T","runs":runs})
 lines=["# Retain-only dual-view summary","","Full-512 is primary for statistical power. V-102 is the only like-for-like comparison to forget-T.","",
        "| run | selected step | full-512 (primary) | V-102 at same step |","|---|---:|---:|---:|"]
 lines += [f"| {x['run_id']} | {x['primary_max_step']} | {x['checkpoint_max_full_512']:.6f} | {x['V_102_at_primary_max_checkpoint']:.6f} |" for x in runs]
 (grid/"retain_only_dual_view_summary.md").write_text("\n".join(lines)+"\n");print(json.dumps({"runs":len(runs)}))
if __name__=="__main__":main()
