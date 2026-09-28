#!/bin/bash
# Spec R-Z solo retry (Spec R Amendment 1 Ruling 4): OOM/co-residency failures are
# rerun alone from initialization, same seed and config; only a clean completion counts.
# A settle delay between cells lets the previous CUDA context tear down fully.
cd /workspace/wmdp_rewritten_cas
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
PY=/workspace/envs/wmdp-probes/bin/python
for rid in "$@"; do
  echo "{\"event\":\"solo_retry_launch\",\"run_id\":\"$rid\",\"unix\":$(date +%s)}" >> spec_rz_outputs/ledger.jsonl
  CUDA_VISIBLE_DEVICES=0 $PY scripts/spec_rz_run.py --run-id "$rid" > "spec_rz_outputs/logs/$rid.log" 2>&1
  rc=$?
  echo "{\"event\":\"solo_retry_exit\",\"run_id\":\"$rid\",\"rc\":$rc,\"unix\":$(date +%s)}" >> spec_rz_outputs/ledger.jsonl
  sleep 45
done
echo "{\"event\":\"solo_retry_complete\",\"unix\":$(date +%s)}" >> spec_rz_outputs/ledger.jsonl
