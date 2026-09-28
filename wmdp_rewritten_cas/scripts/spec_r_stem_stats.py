#!/usr/bin/env python3
"""Spec R stem statistics: stem(t) curves, Delta-stem with paired bootstrap CIs.

stem(t) = acc_full(t) - acc_options(t), both modes evaluated on the SAME view and the
same question IDs, so the bootstrap is paired: one resample of question IDs is applied
to both modes and both steps at once.

Primary statistic (unchanged by Amendment 1): Delta-stem = stem(t_readout) - stem(0),
where t_readout is the ORIGINAL frozen checkpoint-maximum step, not any maximum the
rerun exhibits. The rerun's own maximum is reported descriptively.

One preregistered direction, Tier 1 only: unfiltered-cb / retain-only, Delta-stem > 0.
No threshold is preregistered; direction and CI only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "spec_r_outputs"
REFS = ROOT / "spec_e_outputs/references_grid_path"
LETTERS = "ABCD"
BOOTSTRAP = 1000
SEED = 42
PREREGISTERED_CELL = ("unfiltered-cb", "retain-only")


MATCHED_GRANULARITY_ITEM_LIMIT = 5


def matched_granularity(manifest: dict, cell: dict) -> dict:
    """Amendment 2 Ruling 2, amended-post-observation. Descriptive; never a rescue.

    The original +/-0.01 gate admits 5 items on full-512 and 1 item on V-102. This
    view applies the full-512 item budget to both, so a V-102 park on minimum-possible
    drift is legible as a gate-resolution artifact rather than a replication failure.
    """
    ledger = manifest.get("gate_ledger") or []
    drift = [(g["step"], g.get("drift_items")) for g in ledger
             if g.get("drift_items") is not None]
    breaches = [step for step, items in drift if items > MATCHED_GRANULARITY_ITEM_LIMIT]
    original_breaches = [g["step"] for g in ledger
                         if g.get("soft_gate") and not g["soft_gate"]["pass"]]
    return {
        "rule": f"breach when drift_items > {MATCHED_GRANULARITY_ITEM_LIMIT} on either view",
        "label": "amended-post-observation",
        "authority": "descriptive only; the original gate remains the gate of record",
        "checkpoints_scored": len(drift),
        "max_drift_items": max((i for _, i in drift), default=None),
        "breach_steps": breaches,
        "passes_matched_granularity": not breaches,
        "original_gate_breach_steps": original_breaches,
        "parked_under_original_gate_only": bool(original_breaches) and not breaches,
    }


AMENDMENT_3_FRAMEWORK = "amended-post-observation (Amendment 3)"


def certified_through(manifest: dict):
    """Largest logged step s with the step-0 hard gate passed and every soft gate <= s passed.

    Amendment 3. A step-0 hard-gate failure gives None. The park framework remains the
    framework of record; this is a derived lens that moves no data and no verdicts.
    """
    ledger = sorted(manifest.get("gate_ledger") or [], key=lambda g: g["step"])
    if not ledger:
        return None
    hard = (ledger[0].get("hard_gate") or {})
    if ledger[0]["step"] == 0 and hard and not hard.get("pass", False):
        return None
    best = None
    for gate in ledger:
        soft = gate.get("soft_gate")
        if soft is not None and not soft["pass"]:
            break
        best = gate["step"]
    return best


def certified(step, through) -> bool:
    """A statistic read at `step` is certified iff step <= certified_through."""
    return through is not None and step is not None and step <= through


def read_jsonl(path: Path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def correctness(path: Path) -> dict[str, bool]:
    return {r["question_id"]: max(LETTERS, key=lambda z: r["p_" + z]) == r["gold"]
            for r in read_jsonl(path)}


def table_source(run_id: str) -> tuple[str, str]:
    """Amendment 2 Ruling 1: prefer the retroactive full-length rerun's tables.

    The gate of record stays the ORIGINAL run -- a cell parked under the original gate
    stays parked and claims_attached stays false. The rerun only supplies the fuller
    trajectory the old park-is-a-kill behaviour destroyed.
    """
    key = run_id + "__a2rerun"
    manifest = OUT / "runs" / key / "manifest.json"
    if manifest.is_file():
        payload = json.loads(manifest.read_text())
        if payload.get("status") in {"completed", "parked"} and payload.get("completed_training"):
            return key, "amendment_2_ruling_1_rerun"
    return run_id, "original_run"


def load_cell_tables(run_id: str) -> dict[int, dict[str, dict[str, bool]]]:
    tables: dict[int, dict[str, dict[str, bool]]] = {}
    base = OUT / "tables" / table_source(run_id)[0]
    for mode in ("full", "options-only"):
        for path in sorted((base / mode).glob("step-*.jsonl")):
            step = int(path.stem.split("-")[1])
            tables.setdefault(step, {})[mode] = correctness(path)
    return {s: v for s, v in tables.items() if len(v) == 2}


def paired_bootstrap(ids, arrays: dict[str, np.ndarray], seed: int = SEED,
                     resamples: int = BOOTSTRAP) -> dict:
    """One question-level resample applied to every supplied correctness vector."""
    rng = np.random.default_rng(seed)
    n = len(ids)
    index = rng.integers(0, n, size=(resamples, n))
    means = {key: value[index].mean(axis=1) for key, value in arrays.items()}
    return means


def percentile_ci(values: np.ndarray) -> list[float]:
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def reference_options_only_accuracy(model: str, ids: set[str]) -> float | None:
    path = REFS / model / "options-only.jsonl"
    if not path.is_file():
        return None
    rows = [r for r in read_jsonl(path) if r["question_id"] in ids]
    if not rows:
        return None
    return sum(max(LETTERS, key=lambda z: r["p_" + z]) == r["gold"] for r in rows) / len(rows)


def analyse(cell: dict) -> dict:
    run_id = cell["run_id"]
    manifest_path = OUT / "runs" / run_id / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    status = manifest.get("status")
    source_key, source_kind = table_source(run_id)
    # Amendment 3 Ruling 1: certification attaches to the trajectory whose tables supply
    # the curve. The original trajectory's value rides alongside, never substituted, and
    # there is no min/max combination rule -- the data's own launch certifies the data.
    source_manifest_path = OUT / "runs" / source_key / "manifest.json"
    source_manifest = (json.loads(source_manifest_path.read_text())
                       if source_manifest_path.is_file() else manifest)
    through = certified_through(source_manifest)
    through_original = certified_through(manifest) if source_key != run_id else None
    tables = load_cell_tables(run_id)

    record = {
        "run_id": run_id, "tier": cell["tier"], "model": cell["model"],
        "condition": cell["condition"], "view": cell["view"],
        "lr": cell["lr"], "seed": cell["seed"],
        "status": status,
        "readout_step": cell["readout_step"],
        "readout_rule": ("original frozen checkpoint-maximum step; Amendment 1 Ruling 1 "
                         "keeps the preregistered readout point fixed"),
        "steps_available": sorted(tables),
        "gate_ledger": manifest.get("gate_ledger"),
        "parked": manifest.get("parked"),
        "parked_at_step": manifest.get("parked_at_step"),
        "completed_training": manifest.get("completed_training"),
        "curve_source": {"key": source_key, "kind": source_kind,
                         "gate_of_record": "original run, always"},
        "certified_through": through,
        "certified_through_original": through_original,
        "certification": {
            "framework": AMENDMENT_3_FRAMEWORK,
            "definition": ("largest logged step s with the step-0 hard gate passed and "
                           "every soft gate at logged steps <= s passed"),
            "governing_ledger": source_key,
            "authority": ("certified statistics from previously-parked cells are certified "
                          "DESCRIPTIVE results; certification confers no preregistered "
                          "status on anything"),
            "park_status_unchanged": status,
        },
        "matched_granularity": matched_granularity(manifest, cell),
    }
    # A parked cell keeps its curves. Parking suppresses CLAIMS, not data: Amendment 1
    # Ruling 2(c) wants stem(t) trajectories from every cell as inputs to future
    # analyses, and Spec R section 4 wants stem_stats to cover both tiers. What a park
    # removes is the right to attach a claim, which is enforced below and in
    # evaluate_preregistered.
    record["claims_attached"] = status == "completed"
    if status not in {"completed", "parked"} or 0 not in tables:
        record["evaluable"] = False
        record["reason"] = ("cell did not complete or park" if status not in
                            {"completed", "parked"} else "step-0 tables absent")
        return record
    if status == "parked":
        breach = (manifest.get("parked") or {}).get("step")
        record["park_detail"] = {
            "reason": (manifest.get("parked") or {}).get("reason"),
            "breach_step": breach,
            "readout_step": cell["readout_step"],
            "breach_is_post_readout": (None if breach is None
                                       else breach > cell["readout_step"]),
            "note": ("Amendment 1 Ruling 1 extended the soft gate to cover step 512. A "
                     "breach after the readout step leaves the claim-relevant segment of "
                     "the trajectory inside tolerance, but Spec R section 3 parks the cell "
                     "regardless and no claim is attached here."),
        }

    ids = sorted(set(tables[0]["full"]) & set(tables[0]["options-only"]))
    record["n"] = len(ids)
    record["evaluable"] = True

    curve = []
    for step in sorted(tables):
        full = np.array([tables[step]["full"][q] for q in ids], dtype=float)
        options = np.array([tables[step]["options-only"][q] for q in ids], dtype=float)
        means = paired_bootstrap(ids, {"full": full, "options": options})
        stem_draws = means["full"] - means["options"]
        curve.append({
            "step": step,
            "accuracy_full": float(full.mean()),
            "accuracy_options_only": float(options.mean()),
            "stem": float(full.mean() - options.mean()),
            "stem_ci95": percentile_ci(stem_draws),
            "certified": certified(step, through),
        })
    record["curve"] = curve

    readout = cell["readout_step"]
    if readout not in tables:
        record["delta_stem"] = None
        record["delta_stem_reason"] = f"readout step {readout} absent from the rerun"
        return record

    base_full = np.array([tables[0]["full"][q] for q in ids], dtype=float)
    base_options = np.array([tables[0]["options-only"][q] for q in ids], dtype=float)
    peak_full = np.array([tables[readout]["full"][q] for q in ids], dtype=float)
    peak_options = np.array([tables[readout]["options-only"][q] for q in ids], dtype=float)
    means = paired_bootstrap(ids, {"bf": base_full, "bo": base_options,
                                   "pf": peak_full, "po": peak_options})
    draws = (means["pf"] - means["po"]) - (means["bf"] - means["bo"])
    point = float((peak_full.mean() - peak_options.mean())
                  - (base_full.mean() - base_options.mean()))
    interval = percentile_ci(draws)
    record["delta_stem"] = {
        "value": point, "ci95": interval,
        "stem_at_0": float(base_full.mean() - base_options.mean()),
        "stem_at_readout": float(peak_full.mean() - peak_options.mean()),
        "resamples": BOOTSTRAP, "seed": SEED, "paired": True,
        "direction_positive": point > 0,
        "ci_excludes_zero": interval[0] > 0 or interval[1] < 0,
        "ci_lower_above_zero": interval[0] > 0,
        "certified": certified(readout, through),
        "certification_framework": AMENDMENT_3_FRAMEWORK,
    }

    if 512 in tables:
        post_full = np.array([tables[512]["full"][q] for q in ids], dtype=float)
        post_options = np.array([tables[512]["options-only"][q] for q in ids], dtype=float)
        means = paired_bootstrap(ids, {"bf": base_full, "bo": base_options,
                                       "pf": post_full, "po": post_options})
        draws = (means["pf"] - means["po"]) - (means["bf"] - means["bo"])
        value = float((post_full.mean() - post_options.mean())
                      - (base_full.mean() - base_options.mean()))
        record["delta_stem_at_512"] = {
            "value": value, "ci95": percentile_ci(draws),
            "certified": certified(512, through),
            "certification_framework": AMENDMENT_3_FRAMEWORK,
            "note": ("post-peak segment; Amendment 1 Ruling 1 preregisters no direction "
                     "for it"),
        }

    best = max(curve, key=lambda c: (c["accuracy_full"], -c["step"]))
    record["rerun_own_maximum"] = {
        "step": best["step"], "accuracy_full": best["accuracy_full"],
        "coincides_with_readout": best["step"] == readout,
        "note": ("descriptive only; it does not move the preregistered readout point"),
    }

    step0_options = curve[0]["accuracy_options_only"]
    reference = reference_options_only_accuracy(cell["model"], set(ids))
    record["pre_attack_options_only_crosscheck"] = {
        "rerun_step0_options_only_accuracy": step0_options,
        "frozen_pre_attack_reference_accuracy": reference,
        "absolute_delta": (None if reference is None else abs(step0_options - reference)),
        "agrees": (None if reference is None else abs(step0_options - reference) < 1e-12),
        "note": ("the rerun's own step-0 options-only pass is the pre-attack measurement "
                 "under a zero-LoRA wrapper, so it should reproduce the frozen exact-path "
                 "options-only reference on the same question IDs exactly"),
    }
    return record


def evaluate_preregistered(records: list[dict]) -> dict:
    target = next((r for r in records
                   if r["tier"] == 1 and (r["model"], r["condition"]) == PREREGISTERED_CELL),
                  None)
    if target is None:
        return {"statement": "unfiltered-cb / retain-only: Delta-stem > 0",
                "verdict": "not_evaluable", "reason": "the Tier 1 cell is absent"}
    if target.get("status") == "parked":
        return {"statement": "unfiltered-cb / retain-only: Delta-stem > 0",
                "verdict": "not_evaluable",
                "reason": ("the Tier 1 cell parked on a replication gate; Spec R section 3 "
                           "forbids carrying the claim on a trajectory that did not pass"),
                "run_id": target["run_id"], "park_detail": target.get("park_detail"),
                "delta_stem_computed_but_not_claimed": (
                    target.get("delta_stem", {}) or {}).get("value")}
    if not target.get("evaluable") or not target.get("delta_stem"):
        return {"statement": "unfiltered-cb / retain-only: Delta-stem > 0",
                "verdict": "not_evaluable",
                "reason": target.get("reason") or target.get("delta_stem_reason")
                or "cell did not produce a Delta-stem",
                "run_id": target["run_id"], "status": target.get("status"),
                "parked": target.get("parked")}
    delta = target["delta_stem"]
    satisfied = delta["direction_positive"] and delta["ci_lower_above_zero"]
    return {
        "statement": "unfiltered-cb / retain-only: Delta-stem > 0",
        "amendment_3_note": ("Amendment 3 Ruling 3: this verdict was satisfied under the "
                             "original gate and is not re-evaluated, re-labeled or re-based "
                             "by segment certification. Its provenance line is unchanged."),
        "run_id": target["run_id"],
        "readout_step": target["readout_step"],
        "delta_stem": delta["value"], "ci95": delta["ci95"],
        "direction_positive": delta["direction_positive"],
        "ci_lower_above_zero": delta["ci_lower_above_zero"],
        "verdict": "satisfied" if satisfied else "not_satisfied",
        "reason": ("Delta-stem is positive and its paired 95% interval excludes zero"
                   if satisfied else
                   "Delta-stem is not positive with a 95% interval strictly above zero"),
        "threshold_note": ("no threshold is preregistered; direction and CI only "
                           "(the ceiling lesson stands)"),
    }


def main() -> int:
    cells = json.loads((OUT / "cells.json").read_text())["cells"]
    records = [analyse(cell) for cell in sorted(cells, key=lambda c: (c["tier"], c["run_id"]))]

    tier1 = [r for r in records if r["tier"] == 1]
    payload = {
        "spec": "R",
        "amendment": "Amendment 1 (full-length, full-grid, two-tier)",
        "status": "complete",
        "estimand": ("stem(t) = acc_full(t) - acc_options(t) on the same view and the same "
                     "question IDs; Delta-stem = stem(t_readout) - stem(0)"),
        "bootstrap": {"resamples": BOOTSTRAP, "seed": SEED, "paired": True,
                      "unit": "question", "interval": "percentile 95%"},
        "readout_rule": ("Amendment 1 Ruling 1: the preregistered readout point is the "
                         "original frozen checkpoint-maximum step. The rerun's own maximum "
                         "is descriptive and does not move it."),
        "tiers": {
            "tier_1": {"role": "primary, claims attached",
                       "cells": len(tier1),
                       "completed": sum(1 for r in tier1 if r["status"] == "completed"),
                       "parked": sum(1 for r in tier1 if r["status"] == "parked")},
            "tier_2": {"role": "background, no preregistered statistics",
                       "cells": sum(1 for r in records if r["tier"] == 2),
                       "completed": sum(1 for r in records
                                        if r["tier"] == 2 and r["status"] == "completed"),
                       "parked": sum(1 for r in records
                                     if r["tier"] == 2 and r["status"] == "parked")},
        },
        "amendment_3": {
            "name": "segment-based certification",
            "framework": AMENDMENT_3_FRAMEWORK,
            "supersedes": None,
            "definition": ("certified_through = largest logged step s with the step-0 hard "
                           "gate passed and every soft gate at logged steps <= s passed; a "
                           "statistic read at step t is certified iff t <= certified_through"),
            "framework_of_record": ("the park framework remains the framework of record for "
                                    "this run's preregistered claim"),
            "moves_no_data_or_verdicts": True,
            "certified_delta_stem_at_readout": sum(
                1 for r in records if (r.get("delta_stem") or {}).get("certified")),
            "cells_with_no_certification": sum(
                1 for r in records if r.get("certified_through") is None),
        },
        "preregistered": evaluate_preregistered(records),
        "secondary": {
            "options_only_trajectory": ("acc_options(t) is carried in every cell's curve; "
                                        "whether shortcut accuracy rises under recovery is "
                                        "read from it directly"),
            "intact_comparator": ("the unfiltered cells are the intact-under-same-treatment "
                                  "comparator per the normalisation correction"),
        },
        "cells": records,
    }
    (OUT / "stem_stats.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "complete", "cells": len(records),
                      "tier_1_completed": payload["tiers"]["tier_1"]["completed"],
                      "tier_2_completed": payload["tiers"]["tier_2"]["completed"],
                      "preregistered_verdict": payload["preregistered"]["verdict"]},
                     indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
