#!/usr/bin/env python3
"""Guarded restart supervisor for the checkpointed scaled factual spine."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def state(out: Path) -> dict:
    rejection = rows(out / "rejection_ledger.jsonl")
    cost = rows(out / "cost_ledger.jsonl")
    unique_attempts = {
        (str(row.get("call_id")), int(row.get("attempt_index", -1)), str(row.get("outcome")))
        for row in rejection
    }
    return {
        "parts": len(list((out / "spine_parts").glob("*.json"))) if (out / "spine_parts").exists() else 0,
        "unique_attempts": len(unique_attempts),
        "cost_usd": sum(float(row.get("total_cost_usd", 0) or 0) for row in cost),
        "complete": (out / "full_spine_gate.json").exists() and (out / "textbook.md").exists(),
    }


def load_env(path: Path) -> dict[str, str]:
    env = dict(os.environ)
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def append(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--max-cost-usd", type=float, required=True)
    parser.add_argument("--max-runs", type=int, default=500)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = args.output_root / "corpus_f"
    log = out / "supervisor_log.jsonl"
    env = load_env(args.env_file)
    if not env.get("OpenRouter_key"):
        raise SystemExit("OpenRouter_key missing")
    command = [
        "python", "scripts/spec05_full.py", "spine-full", "--corpus", "CORPUS_F",
        "--config", str(args.config), "--output-root", str(args.output_root),
        "--pilot-root", str(args.pilot_root),
    ]
    for run_index in range(args.max_runs):
        before = state(out)
        event = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "run_index": run_index, "before": before}
        if before["complete"]:
            append(log, event | {"status": "complete_before_run"})
            return 0
        if before["cost_usd"] >= args.max_cost_usd:
            append(log, event | {"status": "budget_stop"})
            return 3
        result = subprocess.run(command, cwd=root, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        after = state(out)
        progressed = after["parts"] > before["parts"] or after["unique_attempts"] > before["unique_attempts"]
        append(log, event | {
            "status": "run_complete", "returncode": result.returncode, "after": after,
            "progressed": progressed, "stderr_tail": result.stderr[-2000:],
        })
        if after["complete"]:
            return 0
        if after["cost_usd"] >= args.max_cost_usd:
            return 3
        if result.returncode != 0 and not progressed:
            return 2
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
