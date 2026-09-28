#!/usr/bin/env python3
"""Quarantine incomplete grid attempts so no partial row is analysis-addressable."""
import json,os,shutil,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];grid=ROOT/"spec_e_outputs/grid";quarantine=ROOT/"spec_e_outputs/grid_failed_attempts";quarantine.mkdir(parents=True,exist_ok=True)
targets=[
 ("spec-e__unfiltered__retain-only__lr-5e-05__seed-1",0),
 ("spec-e__unfiltered__retain-only__lr-2e-04__seed-1",1),
]
ledger=quarantine/"failure_ledger.jsonl"
for run_id,worker in targets:
 source=grid/run_id
 if not source.exists():continue
 manifest=json.loads((source/"manifest.json").read_text())
 if manifest.get("status")!="failed" or manifest.get("failure_type")!="BrokenPipeError":
  raise RuntimeError(f"refusing to quarantine non-matching run: {run_id}")
 destination=quarantine/(run_id+"__broken-pipe-attempt-1")
 if destination.exists():raise RuntimeError(f"quarantine destination exists: {destination}")
 prediction_files=sorted((source/"predictions").glob("*.jsonl"))
 row_counts={p.name:sum(1 for line in p.open() if line.strip()) for p in prediction_files}
 shutil.move(str(source),str(destination))
 event={"event":"failed","status":"failed","run_id":run_id,"worker":worker,"failure_type":"BrokenPipeError",
        "failure_reason":manifest.get("failure_reason"),"quarantined_path":str(destination),
        "canonical_grid_path_present":False,"prediction_row_counts":row_counts,"analysis_eligible":False,
        "recorded_unix":time.time()}
 with ledger.open("a",encoding="utf-8") as handle:
  handle.write(json.dumps(event,sort_keys=True)+"\n");handle.flush();os.fsync(handle.fileno())
 print(json.dumps(event,sort_keys=True))
