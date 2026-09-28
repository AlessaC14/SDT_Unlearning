#!/bin/bash
# Spec R-Z: 12 retain-only cells across two H100s, six per device.
cd /workspace/wmdp_rewritten_cas
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
PY=/workspace/envs/wmdp-probes/bin/python
mkdir -p spec_rz_outputs/logs
run() {
  local gpu=$1; shift
  for rid in "$@"; do
    echo "{\"event\":\"launch\",\"gpu\":$gpu,\"run_id\":\"$rid\",\"unix\":$(date +%s)}" >> spec_rz_outputs/ledger.jsonl
    CUDA_VISIBLE_DEVICES=$gpu $PY scripts/spec_rz_run.py --run-id "$rid" > "spec_rz_outputs/logs/$rid.log" 2>&1
    echo "{\"event\":\"exit\",\"gpu\":$gpu,\"run_id\":\"$rid\",\"rc\":$?,\"unix\":$(date +%s)}" >> spec_rz_outputs/ledger.jsonl
    sleep 45   # settle: let the CUDA context tear down before the next cell
  done
}
B=spec-rz__zephyr-base__retain-only; R=spec-rz__zephyr-rmu__retain-only
run 0 ${R}__lr-1e-05__seed-0 ${R}__lr-5e-05__seed-0 ${R}__lr-2e-04__seed-0 \
      ${R}__lr-1e-05__seed-1 ${R}__lr-5e-05__seed-1 ${R}__lr-2e-04__seed-1 &
run 1 ${B}__lr-1e-05__seed-0 ${B}__lr-5e-05__seed-0 ${B}__lr-2e-04__seed-0 \
      ${B}__lr-1e-05__seed-1 ${B}__lr-5e-05__seed-1 ${B}__lr-2e-04__seed-1 &
wait
echo "{\"event\":\"all_cells_complete\",\"unix\":$(date +%s)}" >> spec_rz_outputs/ledger.jsonl
