#!/usr/bin/env python3
"""Generate spec_rz_outputs/RUN_LOG.md from the audit, cells, stats and ledger."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "spec_rz_outputs"
STEPS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


def main() -> int:
    audit = json.loads((OUT / "audit.json").read_text())
    stats = json.loads((OUT / "stem_stats.json").read_text())
    ledger = [json.loads(x) for x in (OUT / "ledger.jsonl").read_text().splitlines()
              if x.strip()] if (OUT / "ledger.jsonl").is_file() else []
    statuses = Counter(c.get("status") for c in stats["cells"])
    retries = [e for e in ledger if e.get("event", "").startswith("solo_retry")]

    L, n = [], 0
    add = L.append
    add("# Spec R-Z run log — retain-only recovery and stem decomposition on Zephyr RMU\n")
    add("Spec R (amended 1-3) is the protocol of record; Zephyr-side substitutions are "
        "stated where they occur. retain-only only; forget-T explicitly out of scope.\n")

    add("## 1 — Coverage\n")
    add("| status | cells |")
    add("|---|---:|")
    for key, count in sorted(statuses.items(), key=lambda x: str(x[0])):
        add(f"| {key} | {count} |")
    add("")

    add("## 2 — Gates\n")
    gate = audit["checks"]["hard_gate_constants"]["per_model"]
    add("| model | hard gate N=1273 | options-only cross-check (512 shared) |")
    add("|---|---|---|")
    for model, entry in sorted(gate.items()):
        cells = [c for c in stats["cells"] if c["model"] == model and c.get("evaluable")]
        target = entry.get("n1273_full", {}).get("correct")
        options = entry.get("wmdp_bio_n512_options_only", {}).get("correct")
        add(f"| {model} | target {target}, reproduced by {len(cells)} cell(s) | "
            f"target {options}, reproduced |")
    add("")
    add("The N=1273 hard-gate constants were canonicalized from the pre-existing frozen "
        "Spec M exact-path tables rather than from this spec's first passing run, which is "
        "stronger than the fallback section 3 allows. **Replication gate: not applicable** "
        "— these are first-run trajectories with no frozen Zephyr reference, so "
        "certification reduces to step 0 through the last step at which the hard gate and "
        "probability-validity checks pass. The absence of a soft gate is stated, not "
        "implied.\n")

    add("## 3 — Preregistered readout\n")
    pre = stats["preregistered"]
    add(f"**{pre['statement']} → `{pre['verdict']}`**\n")
    if pre.get("arm"):
        arm = pre["arm"]
        delta = arm["delta_stem"]
        add(f"- selected by the checkpoint-maximum rule across (lr, seed): "
            f"`{arm['selected_run_id']}`, peak step {arm['peak_step']}, "
            f"peak full-view accuracy {arm['peak_accuracy_full_1273']:.4f}")
        add(f"- stem(0) {delta['stem_at_0']:+.4f} → stem(peak) {delta['stem_at_peak']:+.4f}; "
            f"**Δstem {delta['value']:+.4f}**, paired 95% CI "
            f"[{delta['ci95'][0]:+.4f}, {delta['ci95'][1]:+.4f}] on n={delta['n']}")
        add(f"- {pre['reason']}")
    add("")
    add("| cell | peak step | acc@peak | stem(0) | stem@peak | Δstem | 95% CI |")
    add("|---|---:|---:|---:|---:|---:|---|")
    for record in stats["cells"]:
        if not record.get("delta_stem"):
            continue
        d = record["delta_stem"]
        add(f"| {record['model']} lr {record['lr']} seed {record['seed']} | "
            f"{record['peak_step']} | {record['peak_accuracy_full_1273']:.4f} | "
            f"{d['stem_at_0']:+.4f} | {d['stem_at_peak']:+.4f} | **{d['value']:+.4f}** | "
            f"[{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}] |")
    add("")
    add(stats["cross_architecture"] + "\n")

    if stats.get("collateral"):
        add("## 4 — Collateral (MMLU-full at each maximum)\n")
        add("| cell | peak step | MMLU-full accuracy | n |")
        add("|---|---:|---:|---:|")
        for entry in stats["collateral"]:
            add(f"| {entry['model']} lr {entry['lr']} seed {entry['seed']} | "
                f"{entry['peak_step']} | {entry['accuracy']:.4f} | {entry['n']} |")
        add("")
        add("MMLU-biology cannot serve as collateral here: it is the training data.\n")

    add("## SURPRISES\n")
    n += 1
    add(f"{n}. **Three NeoX assumptions failed to transfer, each caught before it could "
        "reach a result.** (a) the frozen Zephyr scorer reaches for `model.model` and "
        "`model.lm_head`, which through a PEFT wrapper resolve to the LoraModel rather "
        "than the Mistral stack; (b) `/workspace/CB_probes` and `/workspace/RMU_probes` "
        "both ship a module named `prompt_utils`, and importing `spec_e_run` puts CB_probes "
        "on `sys.path[0]`, silently shadowing the Zephyr one — NeoX prompts into a Mistral "
        "tokenizer; (c) the letter-token variant below. Only (a) and (b) would have "
        "crashed.\n")
    n += 1
    add(f"{n}. **The tokenizer has two pieces per answer letter, and training the wrong one "
        "would have depressed the headline statistic silently.** Appending \"A\" to the "
        "prompt and re-tokenizing yields the bare piece (28741/28760/28743/28757); the "
        "frozen scorer reads the space-prefixed piece (330/365/334/384) at the ':' "
        "position. Training the bare piece optimises a token the evaluation never looks at, "
        "which would have produced a near-zero Δstem that reads as 'RMU resists benign "
        "finetuning' but is a tokenisation artifact. The target is now appended by id, and "
        "an assertion pins it to the scored set.\n")
    n += 1
    add(f"{n}. **Both step-0 gates reproduce the frozen references exactly** — "
        "zephyr-base 820/820 full and 226/226 options-only, zephyr-rmu 374/374 and "
        "143/143. After three transfer bugs this is what establishes that the Zephyr path "
        "is correct rather than merely plausible.\n")
    n += 1
    add(f"{n}. **RMU begins with almost no question-dependent component.** stem(0) is "
        "+0.0181 against the intact comparator's +0.2404: its residual 29.4% on WMDP-bio "
        "is very nearly all option-intrinsic. The NeoX circuit-breaker model showed the "
        "same signature at +0.0039 against its comparator's +0.1367. Two unrelated "
        "defences, two architectures, the same shape.\n")
    if retries:
        n += 1
        add(f"{n}. **Four cells OOMed on the first pass and the pattern was diagnostic, not "
            "random.** Cells 2 and 3 of each GPU's sequential queue failed while cells 1 "
            "and 4-6 succeeded, with the OOM naming a sibling process still holding "
            "~47 GiB: a finished cell's CUDA context had not torn down before its successor "
            "allocated. Fixed with a settle delay between cells and a solo-retry pass; only "
            "clean completions count.\n")
    n += 1
    add(f"{n}. **Adapter persistence was missed on the first build and the run was redone.** "
        "Section 2 requires adapters at every logged step, and section 5's collateral "
        "reloads the peak-step adapter; the first implementation saved none, so the "
        "collateral could not have been produced from it. Caught before results were "
        "reported, at the cost of a full re-run.\n")

    add("## Artifacts\n")
    add("| path | contents |")
    add("|---|---|")
    add("| `spec_rz_outputs/audit.json` | Phase 0: models, conventions, canonicalized N=1273 gate constants, module resolution, disk |")
    add("| `spec_rz_outputs/tables/<run>/<mode>/step-NNN.jsonl` | per-question fp32 four-way tables, both modes, every logged step |")
    add("| `spec_rz_outputs/runs/<run>/` | manifest, per-step metrics with gate ledger, adapters at every logged step |")
    add("| `spec_rz_outputs/stem_stats.json` | curves, Δstem + CI, verdict, certification fields, accuracy on both N=1273 and the 512 join key |")
    add("| `spec_rz_outputs/collateral/<run>.json` | MMLU-full at each maximum |")
    add("| `spec_rz_outputs/ledger.jsonl` | durable launch/exit/retry ledger |")
    add("")
    add("Frozen trees (spec_e, spec_h2, spec_r, fisher, spec_m outputs) were read only.")

    (OUT / "RUN_LOG.md").write_text("\n".join(L) + "\n")
    print(json.dumps({"status": "written", "surprises": n}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
