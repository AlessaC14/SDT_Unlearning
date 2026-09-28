#!/usr/bin/env python3
"""Idempotently add frozen V-102 accuracy to existing retain-only metrics."""
import argparse,json
from pathlib import Path
try: from .spec_e_common import atomic_json,retain_only_accuracy_views
except ImportError: from spec_e_common import atomic_json,retain_only_accuracy_views
def main():
 p=argparse.ArgumentParser();p.add_argument("--grid-root",default="spec_e_outputs/grid");a=p.parse_args();root=Path(__file__).resolve().parents[1]
 vids={x["question_id"] for x in json.loads((root/"spec_e_outputs/preflight/wmdp_split_V.json").read_text())["items"]};updated=[];skipped=[]
 for d in sorted((root/a.grid_root).glob("spec-e__*__retain-only__*")):
  for pred in sorted((d/"predictions").glob("step-*.jsonl")):
   metric=d/"metrics"/(pred.stem+".json")
   if not metric.is_file():skipped.append({"prediction":str(pred),"reason":"metric_not_yet_committed"});continue
   rows=[json.loads(x) for x in pred.read_text().splitlines() if x.strip()];value=json.loads(metric.read_text());value.update(retain_only_accuracy_views(rows,vids));atomic_json(metric,value);updated.append(str(metric))
 report={"status":"complete","updated_count":len(updated),"updated_metrics":updated,"skipped":skipped,"full_512_role":"primary_for_statistical_power","V_102_role":"only_like_for_like_comparison_to_forget-T"}
 atomic_json(root/a.grid_root/"retain_only_v102_backfill_report.json",report);print(json.dumps({"updated":len(updated),"skipped":len(skipped)}))
if __name__=="__main__":main()
