#!/usr/bin/env python3
"""Spec R-Z Phase 0 audit (CPU + one meta-device model load).

Verifies every Zephyr-side input before any training, and canonicalizes the N=1273
hard-gate constants from the frozen Spec M exact-path tables. Frozen trees read only.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FI = Path("/workspace/fisher_information")
ZREF = FI / "spec_m_outputs/zephyr_references"
OUT = ROOT / "spec_rz_outputs"
sys.path.insert(0, "/workspace/RMU_probes")

LETTERS = "ABCD"
EXPECTED_LETTER_IDS = [330, 365, 334, 384]
EXPECTED_COLON_ID = 28747
MODELS = {"zephyr-base": "/workspace/models/wmdp/zephyr-7b-beta_BASE",
          "zephyr-rmu": "/workspace/models/wmdp/Zephyr_RMU"}
TOKENIZER_DIR = "/workspace/models/wmdp/zephyr-7b-beta_BASE"
LRS = ("1e-05", "5e-05", "2e-04")
SEEDS = (0, 1)
STEPS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
BEHAVIOURAL = {"zephyr-base": 0.6441, "zephyr-rmu": 0.2938}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def table(path):
    rows = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    correct = sum(max(LETTERS, key=lambda z: r["p_" + z]) == r["gold"] for r in rows)
    return {"correct": correct, "n": len(rows), "accuracy": correct / len(rows),
            "sha256": sha256_file(path), "path": str(path),
            "ids": [str(r["question_id"]) for r in rows]}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    failures, checks = [], {}

    # --- models and tokenizer conventions ------------------------------------
    present = {tag: Path(p).is_dir() for tag, p in MODELS.items()}
    for tag, ok in present.items():
        if not ok:
            failures.append(f"model directory absent: {MODELS[tag]}")
    parity = {}
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, local_files_only=True)
        from prompt_utils import DATASETS, LETTER_TOKEN_IDS, build_prompt
        observed = [LETTER_TOKEN_IDS[x] for x in LETTERS]
        sample = build_prompt(DATASETS["wmdp_bio"]("test")[0])
        ids = tokenizer(sample, add_special_tokens=True)["input_ids"]
        parity = {
            "letter_ids_expected": EXPECTED_LETTER_IDS,
            "letter_ids_observed": observed,
            "letter_parity_pass": observed == EXPECTED_LETTER_IDS,
            "single_bos": ids[0] == tokenizer.bos_token_id and ids[1] != tokenizer.bos_token_id,
            "colon_read_id_expected": EXPECTED_COLON_ID,
            "colon_read_id_observed": int(ids[-1]),
            "colon_parity_pass": int(ids[-1]) == EXPECTED_COLON_ID,
            "tokenizer_dir": TOKENIZER_DIR,
            "tokenizer_note": ("loaded from the base dir for BOTH checkpoints: RMU ships "
                               "only a SentencePiece model and the env lacks sentencepiece "
                               "(Spec M Ruling 4)"),
        }
        for key in ("letter_parity_pass", "single_bos", "colon_parity_pass"):
            if not parity[key]:
                failures.append(f"Zephyr convention failed: {key}")
    except Exception as exc:  # noqa: BLE001
        parity = {"status": "probe failed", "error": repr(exc)}
        failures.append(f"tokenizer/convention probe failed: {exc!r}")
    checks["models_and_conventions"] = {"present": present, "paths": MODELS,
                                        "parity": parity}

    # --- hard-gate constants, extended to N=1273 from the frozen tables --------
    constants = json.loads((FI / "spec_m_outputs/zephyr_canonical_constants.json").read_text())
    gate = {}
    for tag in MODELS:
        entry = {}
        full_1273 = ZREF / tag / "wmdp_bio_n1273" / "full.jsonl"
        if full_1273.is_file():
            info = table(full_1273)
            ids_1273 = info.pop("ids")
            entry["n1273_full"] = info
            entry["n1273_full"]["source"] = "frozen Spec M exact-path table"
            entry["behavioural_target"] = BEHAVIOURAL[tag]
            entry["behavioural_match"] = abs(info["accuracy"] - BEHAVIOURAL[tag]) < 5e-4
            if not entry["behavioural_match"]:
                failures.append(f"{tag}: N=1273 accuracy {info['accuracy']:.4f} != target "
                                f"{BEHAVIOURAL[tag]}")
        else:
            failures.append(f"{tag}: no frozen N=1273 full table")
            ids_1273 = []
        for view, mode in (("wmdp_bio_n512", "full"), ("wmdp_bio_n512", "options-only")):
            key = f"{tag}/{view}/{mode}"
            path = ZREF / tag / view / f"{mode}.jsonl"
            if not path.is_file():
                failures.append(f"missing reference {key}")
                continue
            info = table(path)
            ids = info.pop("ids")
            entry[f"{view}_{mode.replace('-', '_')}"] = info
            if key in constants["cells"]:
                recorded = constants["cells"][key]
                agree = (recorded["correct"] == info["correct"]
                         and recorded["sha256"] == info["sha256"])
                entry.setdefault("canonical_agreement", {})[key] = agree
                if not agree:
                    failures.append(f"{key}: drifted from the Spec M canonical constant")
            if mode == "full" and ids_1273:
                subset = set(ids).issubset(set(ids_1273))
                entry["n512_ids_subset_of_n1273"] = subset
                if not subset:
                    failures.append(f"{tag}: the 512 join key is not a subset of the 1273 set")
        gate[tag] = entry
    checks["hard_gate_constants"] = {
        "rule": ("step-0 full-view correct counts must equal these integers; the N=1273 "
                 "row is canonicalized here at first use from the frozen Spec M "
                 "exact-path table, which is stronger than canonicalizing from this "
                 "spec's own first passing run"),
        "per_model": gate,
        "options_only_crosscheck": ("each cell's step-0 options-only accuracy must "
                                    "reproduce the frozen N=512 options-only reference on "
                                    "the shared IDs; no N=1273 options-only reference "
                                    "exists and none is required, the spec scopes the "
                                    "check to shared IDs"),
    }

    # --- datasets --------------------------------------------------------------
    datasets = {}
    try:
        from prompt_utils import DATASETS
        wmdp = DATASETS["wmdp_bio"]("test")
        datasets["wmdp_bio"] = {"n": len(wmdp), "expected": 1273,
                                "pass": len(wmdp) == 1273}
        if len(wmdp) != 1273:
            failures.append(f"cais/wmdp-bio has {len(wmdp)} rows, expected 1273")
    except Exception as exc:  # noqa: BLE001
        datasets["wmdp_bio"] = {"error": repr(exc)}
        failures.append(f"wmdp_bio load failed: {exc!r}")
    try:
        sys.path.insert(0, "/workspace/CB_probes")
        import importlib
        neox_prompt = importlib.import_module("prompt_utils")
        mmlu = list(neox_prompt.DATASETS["mmlu_bio"]("test"))
        datasets["mmlu_bio_train"] = {
            "n": len(mmlu), "expected": 454, "pass": len(mmlu) == 454,
            "note": ("same source split and construction as the NeoX retain-only "
                     "condition; completion-only loss masking inherited")}
        if len(mmlu) != 454:
            failures.append(f"mmlu_bio has {len(mmlu)} rows, expected 454")
    except Exception as exc:  # noqa: BLE001
        datasets["mmlu_bio_train"] = {"error": repr(exc)}
        failures.append(f"mmlu_bio load failed: {exc!r}")
    checks["datasets"] = datasets

    # --- LoRA module resolution (meta device, no weights) ----------------------
    modules = {}
    try:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM
        config = AutoConfig.from_pretrained(MODELS["zephyr-base"], local_files_only=True)
        with torch.device("meta"):
            skeleton = AutoModelForCausalLM.from_config(config)
        names = [n for n, m in skeleton.named_modules() if isinstance(m, torch.nn.Linear)
                 and "lm_head" not in n]
        suffixes = sorted({n.rsplit(".", 1)[1] for n in names})
        parameters = 0
        for name, module in skeleton.named_modules():
            if isinstance(module, torch.nn.Linear) and "lm_head" not in name:
                parameters += 256 * (module.in_features + module.out_features)
        modules = {
            "architecture": config.model_type, "layers": config.num_hidden_layers,
            "resolved_linear_modules": len(names), "suffixes": suffixes,
            "lora_rank": 256, "lora_alpha": 512,
            "adapter_parameters": parameters,
            "adapter_bytes_fp32": parameters * 4,
            "adapter_gib_fp32": parameters * 4 / 1024 ** 3,
            "substitution_note": ("Zephyr/Mistral module names, not NeoX's 128; "
                                  "attention-only and low-rank configs excluded as in "
                                  "Spec R"),
        }
    except Exception as exc:  # noqa: BLE001
        modules = {"error": repr(exc)}
        failures.append(f"module resolution failed: {exc!r}")
    checks["lora_modules"] = modules

    # --- disk projection -------------------------------------------------------
    cells = [{"model": m, "lr": lr, "seed": s,
              "run_id": f"spec-rz__{m}__retain-only__lr-{lr}__seed-{s}"}
             for m in MODELS for lr in LRS for s in SEEDS]
    unit = modules.get("adapter_bytes_fp32", 0)
    statvfs = os.statvfs(ROOT)
    free = statvfs.f_bavail * statvfs.f_frsize
    projected = unit * len(cells) * len(STEPS)
    checks["disk"] = {
        "cells": len(cells), "steps_per_cell": len(STEPS),
        "adapters": len(cells) * len(STEPS),
        "adapter_gib": unit / 1024 ** 3,
        "projected_gib": projected / 1024 ** 3,
        "free_gib": free / 1024 ** 3,
        "fits_with_headroom": projected < free * 0.8,
        "policy": "adapters retained at every logged step (spec section 2)",
    }
    if not checks["disk"]["fits_with_headroom"]:
        failures.append("adapter projection does not fit with headroom")

    payload = {
        "spec": "R-Z", "phase": "0_audit",
        "status": "ready" if not failures else "stopped",
        "scope": {"condition": "retain-only only", "forget_T": "explicitly out of scope",
                  "cells": cells,
                  "comparator": ("zephyr-base is the intact-under-treatment comparator and "
                                 "is not optional")},
        "certification": {
            "replication_gate": "not applicable",
            "reason": ("no frozen Zephyr trajectory exists to replicate against; these are "
                       "first-run trajectories, not reruns"),
            "certified_through_definition": ("step 0 through the last step at which the "
                                             "hard gate and probability-validity checks "
                                             "pass"),
            "soft_gate_absent": True,
        },
        "failures": failures,
        "checks": checks,
    }
    (OUT / "audit.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "failures": failures,
                      "cells": len(cells),
                      "modules": modules.get("resolved_linear_modules"),
                      "adapter_gib": round(modules.get("adapter_gib_fp32", 0), 2),
                      "projected_gib": round(checks["disk"]["projected_gib"], 1),
                      "gate_n1273": {t: gate[t].get("n1273_full", {}).get("correct")
                                     for t in MODELS}}, indent=2))
    return 0 if not failures else 2


if __name__ == "__main__":
    sys.exit(main())
