#!/bin/bash
# Spec G Tier 1: six cells across two H100s, three per device.
cd /workspace/wmdp_rewritten_cas
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
PY=/workspace/envs/wmdp-probes/bin/python
mkdir -p spec_gradgeo_outputs/logs
run() {  # $1 = gpu, rest = run ids
  local gpu=$1; shift
  for rid in "$@"; do
    echo "{\"event\":\"launch\",\"gpu\":$gpu,\"run_id\":\"$rid\",\"unix\":$(date +%s)}" \
      >> spec_gradgeo_outputs/ledger.jsonl
    CUDA_VISIBLE_DEVICES=$gpu $PY scripts/spec_gradgeo_gradients.py --run-id "$rid" \
      > "spec_gradgeo_outputs/logs/$rid.log" 2>&1
    echo "{\"event\":\"exit\",\"gpu\":$gpu,\"run_id\":\"$rid\",\"rc\":$?,\"unix\":$(date +%s)}" \
      >> spec_gradgeo_outputs/ledger.jsonl
  done
}
run 0 spec-e__unfiltered-cb__retain-only__lr-5e-05__seed-0 \
      spec-e__unfiltered-cb__forget-T__lr-5e-05__seed-1 \
      spec-e__e2e-strong-filter__retain-only__lr-1e-05__seed-0 &
run 1 spec-e__unfiltered__retain-only__lr-1e-05__seed-0 \
      spec-e__unfiltered__forget-T__lr-1e-05__seed-0 \
      spec-e__e2e-strong-filter__forget-T__lr-1e-05__seed-0 &
wait
echo "{\"event\":\"tier_1_complete\",\"unix\":$(date +%s)}" >> spec_gradgeo_outputs/ledger.jsonl
