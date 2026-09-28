#!/bin/bash
# Spec R-Z unattended chain: full re-run (with adapters) -> solo retry -> collateral
# -> stem stats -> run log. Nothing waits on a human.
cd /workspace/wmdp_rewritten_cas
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
PY=/workspace/envs/wmdp-probes/bin/python
L=spec_rz_outputs/ledger.jsonl
note() { echo "{\"event\":\"$1\",\"unix\":$(date +%s)}" >> $L; }

bash scripts/spec_rz_launch.sh
note chain_training_done

FAILED=$($PY - <<'PY'
import json,glob
print(' '.join(json.load(open(m))['run_id'] for m in sorted(glob.glob('spec_rz_outputs/runs/*/manifest.json'))
                if json.load(open(m))['status'] not in ('completed','parked')))
PY
)
[ -n "$FAILED" ] && bash scripts/spec_rz_retry.sh $FAILED
note chain_retry_done

# collateral, one cell at a time per GPU with a settle delay
CELLS=$($PY - <<'PY'
import json,glob
print(' '.join(json.load(open(m))['run_id'] for m in sorted(glob.glob('spec_rz_outputs/runs/*/manifest.json'))
                if json.load(open(m))['status']=='completed'))
PY
)
i=0
for rid in $CELLS; do
  gpu=$((i % 2)); i=$((i+1))
  CUDA_VISIBLE_DEVICES=$gpu $PY scripts/spec_rz_collateral.py --run-id "$rid" \
    >> spec_rz_outputs/logs/collateral.log 2>&1 &
  if [ $((i % 2)) -eq 0 ]; then wait; sleep 30; fi
done
wait
note chain_collateral_done

$PY scripts/spec_rz_stem_stats.py >> spec_rz_outputs/logs/stats.log 2>&1
$PY scripts/spec_rz_run_log.py    >> spec_rz_outputs/logs/stats.log 2>&1
note chain_complete
