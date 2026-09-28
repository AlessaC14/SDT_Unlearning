#!/usr/bin/env python3
"""Spec R unattended supervisor (Amendment 1 Ruling 2 and Ruling 4).

Schedule:
  1. Tier 1, the six checkpoint-maximum cells, strictly one at a time. They gate the
     claim, so they never share a device with another run.
  2. Tier 2, the remaining 30 cells, packed one per physical GPU.
  3. Solo retry: any cell that failed or OOMed is rerun alone from initialization with
     the same seed. Only a clean completion counts (the Stage A incident precedent).
  4. stem_stats over both tiers.

Nothing here waits on a human. A cell that parks or fails is recorded and the run
continues.
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
LEDGER = OUT / "ledger.jsonl"
LOGS = OUT / "logs"
PYTHON = "/workspace/envs/wmdp-probes/bin/python"
GPUS = ["0", "1"]


def log(event: dict) -> None:
    event["unix"] = time.time()
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(event, sort_keys=True), flush=True)


def status_of(run_id: str) -> str | None:
    path = OUT / "runs" / run_id / "manifest.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text()).get("status")
    except json.JSONDecodeError:
        return None


def launch(run_id: str, gpu: str) -> subprocess.Popen:
    LOGS.mkdir(parents=True, exist_ok=True)
    handle = (LOGS / f"{run_id}.log").open("a")
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = gpu
    environment.setdefault("HF_HUB_OFFLINE", "1")
    process = subprocess.Popen(
        [PYTHON, str(ROOT / "scripts/spec_r_run.py"), "--run-id", run_id],
        cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT)
    process._log_handle = handle  # noqa: SLF001 — closed in reap()
    process._run_id = run_id      # noqa: SLF001
    process._gpu = gpu            # noqa: SLF001
    log({"event": "launch", "run_id": run_id, "gpu": gpu, "pid": process.pid})
    return process


def reap(process: subprocess.Popen) -> dict:
    code = process.wait()
    try:
        process._log_handle.close()  # noqa: SLF001
    except Exception:  # noqa: BLE001
        pass
    run_id = process._run_id  # noqa: SLF001
    record = {"event": "exit", "run_id": run_id, "gpu": process._gpu,  # noqa: SLF001
              "returncode": code, "status": status_of(run_id)}
    log(record)
    return record


def run_serial(run_ids: list[str], gpu: str, phase: str) -> None:
    for run_id in run_ids:
        current = status_of(run_id)
        if current in {"completed", "parked"}:
            log({"event": "skip", "phase": phase, "run_id": run_id, "status": current})
            continue
        reap(launch(run_id, gpu))


def run_packed(run_ids: list[str], phase: str) -> None:
    queue = [r for r in run_ids if status_of(r) not in {"completed", "parked"}]
    for run_id in run_ids:
        if run_id not in queue:
            log({"event": "skip", "phase": phase, "run_id": run_id,
                 "status": status_of(run_id)})
    slots: dict[str, subprocess.Popen | None] = {gpu: None for gpu in GPUS}
    while queue or any(slots.values()):
        for gpu in GPUS:
            if slots[gpu] is None and queue:
                slots[gpu] = launch(queue.pop(0), gpu)
        time.sleep(5)
        for gpu in GPUS:
            process = slots[gpu]
            if process is not None and process.poll() is not None:
                reap(process)
                slots[gpu] = None


def main() -> int:
    started = time.time()
    audit = OUT / "phase_a_audit.json"
    if not audit.is_file():
        log({"event": "abort", "reason": "phase_a_audit.json absent"})
        return 2
    phase_a = json.loads(audit.read_text())
    if phase_a["status"] != "ready_for_phase_b":
        # Ruling 4: only a global gate failure stops everything, and Phase A already
        # decided which that is.
        log({"event": "abort", "reason": "phase A did not authorize training",
             "status": phase_a["status"],
             "global_gate_failures": phase_a.get("global_gate_failures")})
        return 2

    cells = json.loads((OUT / "cells.json").read_text())["cells"]
    tier1 = [c["run_id"] for c in cells if c["tier"] == 1]
    tier2 = [c["run_id"] for c in cells if c["tier"] == 2]
    log({"event": "start", "tier_1": len(tier1), "tier_2": len(tier2),
         "adapter_policy": phase_a["adapter_policy_selected"],
         "schedule": "tier 1 serial, then tier 2 packed one per GPU, then solo retries"})

    run_serial(tier1, GPUS[0], "tier_1_serial")
    log({"event": "tier_1_complete",
         "statuses": {r: status_of(r) for r in tier1},
         "elapsed_s": round(time.time() - started)})

    run_packed(tier2, "tier_2_packed")

    # Ruling 2: anything that failed or OOMed is rerun alone from initialization.
    retry = [r for r in tier1 + tier2 if status_of(r) not in {"completed", "parked"}]
    if retry:
        log({"event": "solo_retry_start", "cells": retry,
             "rule": "OOM or co-resident failure is rerun alone; only a clean completion counts"})
        run_serial(retry, GPUS[0], "solo_retry")

    statuses = {r: status_of(r) for r in tier1 + tier2}
    log({"event": "training_complete", "statuses": statuses,
         "completed": sum(1 for v in statuses.values() if v == "completed"),
         "parked": sum(1 for v in statuses.values() if v == "parked"),
         "unresolved": sorted(k for k, v in statuses.items()
                              if v not in {"completed", "parked"}),
         "elapsed_s": round(time.time() - started)})

    stats = subprocess.run([PYTHON, str(ROOT / "scripts/spec_r_stem_stats.py")],
                           cwd=ROOT, capture_output=True, text=True)
    log({"event": "stem_stats", "returncode": stats.returncode,
         "stdout": stats.stdout[-4000:], "stderr": stats.stderr[-2000:]})

    report = subprocess.run([PYTHON, str(ROOT / "scripts/spec_r_run_log.py")],
                            cwd=ROOT, capture_output=True, text=True)
    log({"event": "run_log", "returncode": report.returncode,
         "stdout": report.stdout[-2000:], "stderr": report.stderr[-2000:]})

    log({"event": "done", "elapsed_s": round(time.time() - started)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
