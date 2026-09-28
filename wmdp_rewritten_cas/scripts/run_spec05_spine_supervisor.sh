#!/usr/bin/env bash
set -u

project_root="${1:-/workspace/wmdp_rewritten_cas}"
cd "$project_root" || exit 2

set -a
# The environment file contains an unrelated malformed line; the required
# OpenRouter assignment still loads and generation validates its presence.
source /workspace/environment.env
set +a

output="spec05_full_runs_v1/corpus_f"
state() {
  local checkpoints rejections raw augmentations
  checkpoints=$(find "$output/spine_parts" -maxdepth 1 -type f -name '*.json' 2>/dev/null | wc -l)
  rejections=$(wc -l < "$output/rejection_ledger.jsonl" 2>/dev/null || true)
  raw=$(wc -l < "$output/raw_llm_log.jsonl" 2>/dev/null || true)
  augmentations=$(find "$output/spine_augmentations" -maxdepth 1 -type f -name '*.json' 2>/dev/null | wc -l)
  printf '%s:%s:%s:%s\n' "$checkpoints" "${rejections:-0}" "${raw:-0}" "$augmentations"
}

for cycle in $(seq 1 250); do
  before=$(state)
  printf 'supervisor cycle=%s before=%s\n' "$cycle" "$before"
  python scripts/spec05_full.py spine-full \
    --corpus CORPUS_F \
    --config configs/spec05.full.json \
    --output-root spec05_full_runs_v1 \
    --pilot-root spec05_runs_multipart02
  status=$?
  after=$(state)
  printf 'supervisor cycle=%s status=%s after=%s\n' "$cycle" "$status" "$after"
  if test -f "$output/full_spine_gate.json"; then
    exit 0
  fi
  if test "$after" = "$before"; then
    printf 'supervisor stopped: no checkpoint or ledger progress\n' >&2
    exit "${status:-1}"
  fi
done

printf 'supervisor stopped: cycle cap reached\n' >&2
exit 1
