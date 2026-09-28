#!/usr/bin/env python3
"""Spec G Phase 0 audit (CPU): inputs, projector thresholds, certification inheritance.

Computes, per model, the global top-k coordinate thresholds over that model's own
step-0 retain-Fisher diagonal under the conservative treatment (rank-deficient blocks
and layers 30-31 excluded, matching the Fisher study). Writes spec_gradgeo_outputs/audit.json.

Frozen trees are read only.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FI = Path("/workspace/fisher_information")
OUT = ROOT / "spec_gradgeo_outputs"
MODELS = ("unfiltered", "e2e-strong-filter", "unfiltered-cb")
K_FRACTIONS = (0.01, 0.05, 0.10)
EXCLUDED_LAYERS = (30, 31)


def conservative_blocks(summary: dict) -> dict:
    return {name: meta for name, meta in summary.items()
            if not meta["rank_deficient"] and meta["layer_idx"] not in EXCLUDED_LAYERS}


def fisher_thresholds(tag: str) -> dict:
    payload = json.loads((FI / f"fisher_neox_{tag}_mmlu_bio.json").read_text())
    included = conservative_blocks(payload["per_block_summary"])
    total = int(sum(m["numel"] for m in included.values()))
    buffer = np.empty(total, dtype=np.float32)
    offset = 0
    for name in sorted(included):
        block = np.load(payload["fisher_diagonal"][name], mmap_mode="r").ravel()
        buffer[offset:offset + block.size] = block
        offset += block.size
    assert offset == total
    ks = [max(1, int(round(f * total))) for f in K_FRACTIONS]
    # kth largest -> partition index total - k
    positions = sorted({total - k for k in ks})
    partitioned = np.partition(buffer, positions)
    thresholds = {}
    for fraction, k in zip(K_FRACTIONS, ks):
        value = float(partitioned[total - k])
        exact = int((buffer >= value).sum())
        thresholds[f"{fraction:.2f}"] = {
            "k_target": k, "threshold": value, "k_at_threshold": exact,
            "fraction_target": fraction, "fraction_at_threshold": exact / total,
        }
    del buffer, partitioned
    return {
        "model": tag,
        "included_blocks": len(included), "included_numel": total,
        "all_blocks": len(payload["per_block_summary"]),
        "all_numel": int(sum(m["numel"] for m in payload["per_block_summary"].values())),
        "excluded_rule": ("rank-deficient blocks and layers 30-31 excluded, matching the "
                          "Fisher study's conservative treatment"),
        "thresholds": thresholds,
        "source": str((FI / f"fisher_neox_{tag}_mmlu_bio.json")),
    }


def tier1_cells() -> list[dict]:
    """Tier 1 cells with their adapter directories.

    Two Tier 1 cells were killed pre-512 by Spec R's original park-is-a-kill rule and
    have only a partial adapter series; their Amendment-2 Ruling-1 rerun carries the
    full eleven. Spec R Amendment 3 Ruling 1 attaches to the trajectory whose data is
    used, so the rerun's adapters are used and the original is recorded alongside.
    """
    stats = json.loads((ROOT / "spec_r_outputs/stem_stats.json").read_text())
    certified = {c["run_id"]: c.get("certified_through") for c in stats["cells"]}
    certified_original = {c["run_id"]: c.get("certified_through_original")
                          for c in stats["cells"]}
    cells = []
    for cell in stats["cells"]:
        if cell["tier"] != 1:
            continue
        run_id = cell["run_id"]
        options = [run_id + "__a2rerun", run_id]
        chosen, steps = None, []
        for key in options:
            directory = ROOT / "spec_r_outputs/runs" / key
            found = sorted(int(p.name.rsplit("-", 1)[1]) for p in directory.glob("adapter-step-*")
                           if (p / "adapter_model.safetensors").is_file())
            if len(found) > len(steps):
                chosen, steps = key, found
        cells.append({
            "run_id": run_id, "adapter_key": chosen, "model": cell["model"],
            "condition": cell["condition"], "lr": cell["lr"], "seed": cell["seed"],
            "view": cell["view"], "steps": steps,
            "certified_through": certified.get(run_id),
            "certified_through_original": certified_original.get(run_id),
            "adapter_source": ("amendment_2_ruling_1_rerun" if chosen.endswith("__a2rerun")
                               else "original_run"),
        })
    return cells


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cells = tier1_cells()
    failures = []
    if len(cells) != 6:
        failures.append(f"expected 6 Tier 1 cells, found {len(cells)}")
    for cell in cells:
        if len(cell["steps"]) != 11:
            failures.append(f"{cell['run_id']}: {len(cell['steps'])} adapters, expected 11")

    projectors = {tag: fisher_thresholds(tag) for tag in MODELS}

    payload = {
        "spec": "G", "phase": "0_audit",
        "status": "ready" if not failures else "stopped",
        "failures": failures,
        "tier_1_cells": cells,
        "projector": {
            "definition": ("P_r is the coordinate mask of the top-k entries of that MODEL's "
                           "own step-0 retain-Fisher diagonal (MMLU-bio 454)"),
            "k_fractions": list(K_FRACTIONS),
            "caveats": [
                ("P_r is a coordinate-subspace approximation to the retain-Fisher "
                 "eigenspace: diagonal top-k, not an eigendecomposition. This is not a "
                 "spectrum."),
                ("P_r is fixed at step 0; subspace drift under finetuning is a known "
                 "unknown, probed once in the section 4 scoped optional."),
            ],
            "models": projectors,
        },
        "certification": {
            "framework": "Spec R Amendment 3, inherited",
            "rule": ("a step counts as cleared for P1 when it is certified AND both "
                     "gradient norms are nonzero"),
        },
    }
    (OUT / "audit.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": payload["status"], "cells": len(cells), "failures": failures,
        "certified_through": {c["run_id"][7:]: c["certified_through"] for c in cells},
        "thresholds": {m: {k: round(v["threshold"], 12)
                           for k, v in p["thresholds"].items()}
                       for m, p in projectors.items()},
    }, indent=2))
    return 0 if not failures else 2


if __name__ == "__main__":
    sys.exit(main())
