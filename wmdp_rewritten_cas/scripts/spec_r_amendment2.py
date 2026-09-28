#!/usr/bin/env python3
"""Spec R Amendment 2 follow-on: backfill, retroactive reruns, regenerate.

Runs unattended behind the Amendment 1 supervisor:
  1. wait for the in-flight supervisor to exit
  2. Ruling 2 backfill: add drift_items and argmax_agreement to every existing gate
     ledger entry. Descriptive only -- no `pass` field is ever touched, and the
     original gate verdicts are not rewritten.
  3. Ruling 1 retroactive reruns: cells killed pre-512 under the old park-is-a-kill
     behaviour are rerun solo to step 512, same seed and config, writing to
     <run_id>__a2rerun with a rerun_of pointer. The original parked manifests are
     retained untouched.
  4. regenerate stem_stats.json and RUN_LOG.md
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "spec_r_outputs"
GRID = ROOT / "spec_e_outputs/grid"
LEDGER = OUT / "ledger.jsonl"
LOGS = OUT / "logs"
PYTHON = "/workspace/envs/wmdp-probes/bin/python"
LETTERS = "ABCD"
RERUN_SUFFIX = "__a2rerun"


def log(event: dict) -> None:
    event["unix"] = time.time()
    event["amendment"] = 2
    with LEDGER.open("a") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(event, sort_keys=True), flush=True)


def supervisor_running() -> bool:
    result = subprocess.run(["pgrep", "-f", "spec_r_supervisor.py"],
                            capture_output=True, text=True)
    return result.returncode == 0 and bool(result.stdout.strip())


def frozen_correct(run_id: str, step: int) -> int | None:
    path = GRID / run_id / "predictions" / f"step-{step:03d}.jsonl"
    if not path.is_file():
        return None
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    return sum(max(LETTERS, key=lambda z: r["p_" + z]) == r["gold"] for r in rows)


def backfill(manifest_path: Path) -> bool:
    manifest = json.loads(manifest_path.read_text())
    ledger = manifest.get("gate_ledger") or []
    if not ledger:
        return False
    run_id = manifest["run_id"]
    key = manifest.get("output_key", run_id)
    changed = False
    for gate in ledger:
        if "drift_items" in gate:
            continue
        soft = gate.get("soft_gate")
        if not soft:
            continue
        metrics = OUT / "runs" / key / "metrics" / f"step-{gate['step']:03d}.json"
        rerun = json.loads(metrics.read_text())["correct_full"] if metrics.is_file() else None
        original = frozen_correct(run_id, gate["step"])
        gate["drift_items"] = (None if rerun is None or original is None
                               else abs(original - rerun))
        gate["argmax_agreement"] = soft.get("per_item_argmax_agreement")
        gate["descriptive_fields_note"] = ("Amendment 2 Ruling 2, "
                                           "amended-post-observation; descriptive only, "
                                           "backfilled after the fact")
        changed = True
    if changed:
        manifest["amendment_2_backfilled"] = True
        manifest.setdefault("park_semantics",
                            "parked under the pre-Amendment-2 park-is-a-kill behaviour")
        if "completed_training" not in manifest:
            manifest["completed_training"] = 512 in {g["step"] for g in ledger}
        if "parked_at_step" not in manifest:
            manifest["parked_at_step"] = (manifest.get("parked") or {}).get("step")
        temporary = manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, manifest_path)
    return changed


def needs_rerun(manifest: dict) -> bool:
    """A cell killed before step 512 by the superseded park-is-a-kill behaviour."""
    if manifest.get("status") != "parked":
        return False
    if manifest.get("rerun_of"):
        return False
    if manifest.get("park_reason_class") == "step_zero_hard_gate":
        return False  # never trained; Ruling 1 does not revive a hard-gate failure
    steps = {g["step"] for g in (manifest.get("gate_ledger") or [])}
    return 512 not in steps


def main() -> int:
    started = time.time()
    log({"event": "amendment_2_start",
         "rulings": ["1 park is a label not a kill (retroactive reruns)",
                     "2 matched-granularity descriptors",
                     "3 agent patches ratified", "4 options-only family correction"]})

    while supervisor_running():
        time.sleep(30)
    log({"event": "amendment_2_supervisor_clear",
         "waited_s": round(time.time() - started)})

    backfilled = []
    for path in sorted((OUT / "runs").glob("*/manifest.json")):
        if backfill(path):
            backfilled.append(path.parent.name)
    log({"event": "amendment_2_backfill", "cells": len(backfilled)})

    queue = []
    for path in sorted((OUT / "runs").glob("*/manifest.json")):
        manifest = json.loads(path.read_text())
        if needs_rerun(manifest):
            queue.append((manifest["run_id"], manifest.get("parked_at_step")))
    log({"event": "amendment_2_rerun_queue", "cells": [q[0] for q in queue],
         "count": len(queue),
         "rule": "Ruling 1 retroactive: killed pre-512, rerun solo to 512"})

    LOGS.mkdir(parents=True, exist_ok=True)
    for run_id, killed_at in queue:
        key = run_id + RERUN_SUFFIX
        if (OUT / "runs" / key / "manifest.json").is_file():
            existing = json.loads((OUT / "runs" / key / "manifest.json").read_text())
            if existing.get("status") in {"completed", "parked"}:
                log({"event": "amendment_2_rerun_skip", "run_id": run_id,
                     "status": existing["status"]})
                continue
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = "0"
        environment.setdefault("HF_HUB_OFFLINE", "1")
        log({"event": "amendment_2_rerun_launch", "run_id": run_id,
             "killed_at_step": killed_at, "output_key": key})
        with (LOGS / f"{key}.log").open("a") as handle:
            code = subprocess.run(
                [PYTHON, str(ROOT / "scripts/spec_r_run.py"), "--run-id", run_id,
                 "--output-key", key, "--rerun-of", run_id],
                cwd=ROOT, env=environment, stdout=handle,
                stderr=subprocess.STDOUT).returncode
        status = None
        manifest_path = OUT / "runs" / key / "manifest.json"
        if manifest_path.is_file():
            status = json.loads(manifest_path.read_text()).get("status")
        log({"event": "amendment_2_rerun_exit", "run_id": run_id, "output_key": key,
             "returncode": code, "status": status})

    for path in sorted((OUT / "runs").glob("*" + RERUN_SUFFIX + "/manifest.json")):
        backfill(path)

    for script in ("spec_r_stem_stats.py", "spec_r_run_log.py"):
        result = subprocess.run([PYTHON, str(ROOT / "scripts" / script)],
                                cwd=ROOT, capture_output=True, text=True)
        log({"event": "amendment_2_" + script.replace(".py", ""),
             "returncode": result.returncode, "stdout": result.stdout[-3000:],
             "stderr": result.stderr[-1500:]})

    log({"event": "amendment_2_done", "elapsed_s": round(time.time() - started)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
