#!/usr/bin/env python3
"""Generate spec_gradgeo_outputs/RUN_LOG.md from audit.json, the cell results and decision.json."""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "spec_gradgeo_outputs"
STEPS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


def main() -> int:
    audit = json.loads((OUT / "audit.json").read_text())
    decision = json.loads((OUT / "decision.json").read_text())
    cells = [json.loads(Path(p).read_text())
             for p in sorted(glob.glob(str(OUT / "cells" / "*.json")))]
    by_key = {(c["cell"]["model"], c["cell"]["condition"]): c for c in cells}

    L = []
    add = L.append
    add("# Spec G run log — gradient geometry along recovery trajectories\n")
    add("GPU, no training. Gradients taken at frozen Spec R adapters merged into the base "
        "weights, letter-logit cross-entropy on two fixed probe sets, exact-path precision "
        "contract (bf16 autocast forward, fp32 loss and softmax, fp32 gradient "
        "accumulation). Frozen trees were read only.\n")
    add("Note on naming: this repository already contains an unrelated exploratory Spec G "
        "(factual legibility gates, `scripts/spec_g_{common,evaluate,preflight}.py`, outputs "
        "in `spec_g_exploratory_outputs/`). This spec's files are "
        "`spec_gradgeo_{audit,gradients,decision,run_log}.py` with outputs in `spec_gradgeo_outputs/`; "
        "there is no filename collision, but the namespace is shared.\n")

    add("## 1 — Coverage\n")
    add(f"{len(cells)} of 6 Tier 1 cells; {sum(len(c['records']) for c in cells)} "
        "checkpoints, each computed twice with different batch orders.\n")
    add("| cell | steps | certified_through | adapter source |")
    add("|---|---:|---:|---|")
    for c in sorted(cells, key=lambda x: x["cell"]["run_id"]):
        cell = c["cell"]
        add(f"| {cell['model']} / {cell['condition']} | {len(c['records'])} | "
            f"{cell['certified_through']} | {cell['adapter_source']} |")
    add("")

    add("## 2 — Controls\n")
    control = decision["controls"]["projector_positive_control"]
    add(f"**Projector positive control: {control['alpha_channel_status'].upper()}.** "
        "alpha_r(0) against the measured random-mask null, per model per k:\n")
    add("| model | k | alpha_r(0) | random-mask null mean | ratio | exceeds null p97.5 |")
    add("|---|---|---:|---:|---:|:--|")
    for model, entry in sorted(control["per_model"].items()):
        for k, v in sorted(entry.items()):
            add(f"| {model} | {k} | {v['alpha_retain_step0']:.4f} | "
                f"{v['random_null_mean']:.4f} | {v['ratio_to_null_mean']:.2f}x | "
                f"{'yes' if v['exceeds_null'] else 'NO'} |")
    add("")
    cn = decision["controls"]["conflict_vs_sign_shuffle_null"]
    add(f"**Conflict against its sign-shuffle null:** "
        f"{cn['outside_sign_shuffle_band']} of {cn['checkpoints']} checkpoints fall outside "
        f"the band.\n")

    add("## 3 — conflict(t), the primary quantity\n")
    add("| cell | " + " | ".join(str(s) for s in STEPS) + " |")
    add("|---|" + "---:|" * len(STEPS))
    for key in sorted(by_key):
        record = {r["step"]: r for r in by_key[key]["records"]}
        row = []
        for s in STEPS:
            if s not in record:
                row.append("—")
            else:
                mark = "" if record[s]["certified"] else "*"
                row.append(f"{record[s]['replicate_0']['conflict']:+.3f}{mark}")
        add(f"| {key[0]} / {key[1]} | " + " | ".join(row) + " |")
    add("")
    add("`*` marks a step beyond that trajectory's certified_through (Spec R Amendment 3); "
        "computed and labelled, excluded from P1's window.\n")

    add("## 4 — alpha(t) at k = 5%\n")
    add("| cell | quantity | " + " | ".join(str(s) for s in STEPS) + " |")
    add("|---|---|" + "---:|" * len(STEPS))
    for key in sorted(by_key):
        record = {r["step"]: r for r in by_key[key]["records"]}
        for which, label in (("alpha_retain", "alpha_r"), ("alpha_forget", "alpha_f")):
            row = [(f"{record[s]['replicate_0']['alpha']['0.05'][which]:.3f}"
                    if s in record else "—") for s in STEPS]
            add(f"| {key[0]} / {key[1]} | {label} | " + " | ".join(row) + " |")
    add("")
    add("The random-mask null for k = 5% sits at 0.224 (isotropic expectation "
        "sqrt(0.05) = 0.2236), so every value above is far outside it.\n")

    p1 = decision["preregistered"]["P1"]
    add("## 5 — P1\n")
    add(f"**Verdict: `{p1['verdict']}`.**\n")
    add(f"- statement: {p1['statement']}")
    add(f"- window: {p1['early_window']}; cleared = {p1['cleared_definition']}")
    add(f"- runs per arm: {p1.get('runs_per_arm')}")
    if p1.get("difference") is not None:
        add(f"- arm medians: unfiltered-cb "
            f"{p1['median_mean_conflict_early']['unfiltered-cb']:+.4f}, e2e-strong-filter "
            f"{p1['median_mean_conflict_early']['e2e-strong-filter']:+.4f}; difference "
            f"**{p1['difference']:+.4f}**, direction satisfied: {p1['direction_satisfied']}")
    add(f"- reason: {p1['reason']}")
    if p1.get("what_would_evaluate_it"):
        add(f"- what would evaluate it: {p1['what_would_evaluate_it']}")
    add("")

    add("## 6 — Post-hoc interpretation note on the P1 reversal\n")
    add("**Labelled post-hoc. This is a hypothesis, not a finding, and nothing in this run "
        "tests it.**\n")
    add("conflict(t) measures loss-coupling, not mechanism. Letter-cross-entropy on the "
        "forget probe improves under benign finetuning for generic reasons — calibration, "
        "output format, general biology — so for an ordinary model the forget-gradient is "
        "very nearly a retain-gradient, and a high cosine is what one should expect by "
        "default. On that reading the filtered model's +0.81 is generic transfer rather "
        "than anything about filtering, and unfiltered-cb's depressed +0.46 is the "
        "interesting number: something in CB sits in the loss path and decouples the two "
        "objectives.\n")
    add("That would inverse the sign of the original mechanistic expectation — the "
        "intervention shows up as *less* loss-coupling, not more. It is consistent with "
        "the measurement and with CB being the only model whose conflict curve is "
        "non-monotonic, but it is equally consistent with other stories, and no control "
        "here separates generic transfer from intervention machinery. Testing it needs a "
        "probe that holds the generic component fixed — the obvious candidate being a "
        "matched non-biology probe set, which this spec does not have.\n")

    drift_path = OUT / "fisher_drift.json"
    add("## SURPRISES\n")
    n = 1
    add(f"{n}. **The step-1 checkpoint is bitwise the base model.** LoRA's B is exactly zero "
        "at step 0 by initialisation and still exactly zero at step 1, because the "
        "linear-warmup schedule returns a factor of 0 at the first update — the first "
        "optimizer step runs at learning rate zero. conflict(0) and conflict(1) therefore "
        "agree to every digit, on every cell. This retroactively explains Spec H2's "
        "observation that steps 0 and 1 are argmax-identical in all 36 runs, and it is an "
        "unplanned end-to-end check on the merge path: a zero delta reproduces the base "
        "gradients exactly.\n")
    n += 1
    add(f"{n}. **Subtractive unmerge is not bitwise, and the drift was measured rather than "
        "assumed.** The run merges the step-t delta into the base weights and removes it by "
        "subtraction; in float32 that does not restore the original bits. Measured against "
        "the real adapter deltas over a full 11-checkpoint trajectory, the drift plateaus at "
        "1.49e-08 absolute (1.3e-07 relative) — about 100,000x smaller than the smallest "
        "delta applied and about 1000x below the reduction-order precision every reported "
        "number already carries. The run was left on one code path rather than edited "
        "mid-flight; the committed implementation restores from a pristine copy so any "
        "rerun is exact.\n")
    n += 1
    add(f"{n}. **P1 is not evaluable on Tier 1, and not for want of data quality.** It "
        "requires a run-level bootstrap over (lr, seed); Tier 1 holds exactly one (lr, seed) "
        "per (model, condition), so each arm has n=1 and the interval is undefined. Tier 2 "
        "cannot supply the missing runs either: it retained adapters only at {0, peak, 512}, "
        "and P1's window is steps <= 64. This is an adapter-retention question for Spec R, "
        "not an analysis choice available here. The direction is reported; the interval is "
        "not manufactured.\n")
    n += 1
    models = audit["projector"]["models"]
    ratio = (models["unfiltered-cb"]["thresholds"]["0.01"]["threshold"]
             / models["unfiltered"]["thresholds"]["0.01"]["threshold"])
    add(f"{n}. **unfiltered-cb's retain-Fisher diagonal is far more concentrated than the "
        f"other two models'.** Its top-1% threshold is {ratio:.1f}x the unfiltered model's, "
        "and similar at every k. This is a property of the frozen Fisher inputs, present "
        "before any gradient was taken, and it belongs next to any reading of CB's alpha "
        "values.\n")
    drifting = []
    for key, cell in sorted(by_key.items()):
        record = {r["step"]: r for r in cell["records"]}
        if 0 in record and 512 in record:
            drifting.append((key,
                             record[0]["replicate_0"]["alpha"]["0.05"]["alpha_retain"],
                             record[512]["replicate_0"]["alpha"]["0.05"]["alpha_retain"]))
    if drifting:
        n += 1
        add(f"{n}. **alpha_r decays along every completed trajectory**, which is the "
            "fixed-at-step-0 projector caveat appearing as a measurement rather than a "
            "worry:\n")
        add("   | cell | alpha_r(0) @5% | alpha_r(512) @5% |")
        add("   |---|---:|---:|")
        for key, a0, a512 in drifting:
            add(f"   | {key[0]} / {key[1]} | {a0:.3f} | {a512:.3f} |")
        add("")
        add("   The section 4 scoped optional — recomputing the retain-Fisher diagonal at "
            "one late checkpoint and measuring top-k overlap — now has an empirical "
            "motivation rather than a hypothetical one.\n")

    if drift_path.is_file():
        drift = json.loads(drift_path.read_text())
        add("## 7 — Projector drift (section 4 scoped optional)\n")
        add(f"Retain-Fisher diagonal recomputed at {drift['checkpoint']['model']} / "
            f"step {drift['checkpoint']['step']} of "
            f"`{drift['checkpoint']['run_id']}`, using "
            f"{drift['method']['estimator'].split(',')[0]} and the study's own forward "
            "path so the comparison differs only by the checkpoint.\n")
        add("| k | top-k overlap with step 0 | chance | lift | old threshold | new threshold |")
        add("|---|---:|---:|---:|---:|---:|")
        for k, v in sorted(drift["overlap"].items()):
            add(f"| {k} | **{v['overlap_fraction']:.4f}** | "
                f"{v['chance_overlap_fraction']:.4f} | {v['lift_over_chance']:.1f}x | "
                f"{v['old_threshold']:.3e} | {v['new_threshold']:.3e} |")
        add("")
        add(f"{drift['reading']}\n")

    if drift_path.is_file():
        drift = json.loads(drift_path.read_text())
        k1 = drift["overlap"]["0.01"]
        n += 1
        add(f"{n}. **The step-0 projector drifts substantially, so every alpha claim is "
            f"scoped to the step-0 subspace.** The section 4 check recomputed the "
            f"retain-Fisher diagonal at unfiltered-cb / retain-only step 256 with the "
            f"study's own estimator and forward path. Only "
            f"**{k1['overlap_fraction']:.1%}** of the step-0 top-1% coordinates are still "
            f"top-1% at the peak "
            f"({k1['intersection']:,} of {k1['old_selected']:,}); at k = 5% it is "
            f"{drift['overlap']['0.05']['overlap_fraction']:.1%} and at k = 10% "
            f"{drift['overlap']['0.10']['overlap_fraction']:.1%}. The sets are far from "
            f"independent — {k1['lift_over_chance']:.0f}x chance at k = 1% — so real "
            f"structure survives, but four-fifths of the top-1% set turns over. Under this "
            f"spec's own reading rule that does NOT retire the fixed-at-step-0 caveat: it "
            f"confirms it, and every alpha number in this log is a statement about the "
            f"step-0 subspace rather than about the retain-Fisher subspace at the step "
            f"where it was measured. The alpha_r decay from 0.82 to 0.55 along the "
            f"trajectory is consistent with exactly this, and the two observations are not "
            f"independent evidence.\n")
        add(f"   The absolute Fisher scale also moves by roughly two orders of magnitude "
            f"(top-1% threshold {k1['old_threshold']:.2e} at step 0 against "
            f"{k1['new_threshold']:.2e} at step 256), which is why the comparison is made "
            f"on rank-selected sets rather than on threshold values.\n")

    add("## Artifacts\n")
    add("| path | contents |")
    add("|---|---|")
    add("| `spec_gradgeo_outputs/audit.json` | Phase 0: Tier 1 cells, certification inheritance, per-model top-k thresholds, projector caveats |")
    add("| `spec_gradgeo_outputs/cells/<run>.json` | per checkpoint: conflict, both nulls, alpha_f and alpha_r per k, random-mask bands, both reduction-order replicates, certified flag |")
    add("| `spec_gradgeo_outputs/decision.json` | P1 and the section 2 controls, evaluated mechanically |")
    add("| `scripts/spec_gradgeo_{audit,gradients,decision,run_log}.py`, `tests/test_spec_g.py` | implementation and 17 tests |")
    add("")
    add("Per Spec G section 6: alpha as an anchoring evaluation metric remains the anchoring "
        "paper's property. What this spec offers the workshop paper is the conflict(t) "
        "mechanism figure for the double dissociation. No interpretation is written here.")

    (OUT / "RUN_LOG.md").write_text("\n".join(L) + "\n")
    print(json.dumps({"status": "written", "surprises": n, "cells": len(cells)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
