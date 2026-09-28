#!/usr/bin/env python3
"""Spec M Ruling 3: retain-side (MMLU-bio N=454) exact-path reference tables.

This is the Spec E eval-callback MMLU-bio computation with persistence enabled.  It
reuses `spec_e_run.score()` verbatim -- the same scorer that produced the forget-side
`references_grid_path` tables -- so the retain and forget arms of Spec M travel an
identical precision path:

    fp32 weight load -> fresh zero-LoRA r=256/alpha=512 PEFT wrapper
    -> right padding, add_special_tokens=False (NeoX takes no BOS)
    -> letter-token parity [329, 378, 330, 399] and colon-position (id 27) assertion
    -> torch.autocast('cuda', bfloat16) forward
    -> torch.softmax(logits.float(), -1) -> numpy float32 storage

No second scorer path is introduced (Amendment 1, Ruling 3).

Question set: the eval-callback construction -- `cais/mmlu` config `all`, split `test`,
capped at `general_cap_per_subtask`, stable id `mmlu-{subject}-{n:04d}`, filtered to
college_biology + high_school_biology.  Verified in this run to be the same 454
questions in the same order as `prompt_utils.load_mmlu_bio` (the Fisher / Zephyr
convention), so NeoX and Zephyr retain tables are row-aligned by construction.

--grid-parity additionally scores the full 14042-row MMLU general set and reports the
bio-filtered accuracy, which is the quantity the Spec E grid persisted as
`mmlu_bio_accuracy_last_repeat`.  That is an independent prior measurement of this
table and is used as the retain-side format-sanity check.
"""
from __future__ import annotations
import argparse, gc, json, time
from pathlib import Path
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from .spec_e_common import (atomic_json, option_only_example, read_config,
                                sha256_file, validate_prediction_rows)
    from .spec_e_run import atomic_jsonl, resolve_targets, score
except ImportError:
    from spec_e_common import (atomic_json, option_only_example, read_config,
                               sha256_file, validate_prediction_rows)
    from spec_e_run import atomic_jsonl, resolve_targets, score

import sys
sys.path.insert(0, "/workspace/CB_probes")
from prompt_utils import LETTER_TOKEN_IDS  # noqa: E402

BIO_SUBJECTS = ("college_biology", "high_school_biology")


def build_general(cap):
    """Verbatim reproduction of spec_e_run.load_eval()'s `general` construction."""
    allm = load_dataset("cais/mmlu", "all", split="test")
    seen, general = {}, []
    for x in allm:
        subject = x["subject"]
        n = seen.get(subject, 0)
        seen[subject] = n + 1
        if n >= cap:
            continue
        y = dict(x)
        y["_stable_id"] = f"mmlu-{subject}-{n:04d}"
        general.append(y)
    return general


def bio_subset(general):
    prefixes = tuple(f"mmlu-{s}-" for s in BIO_SUBJECTS)
    return [x for x in general if x["_stable_id"].startswith(prefixes)]


def accuracy(rows):
    return sum(max("ABCD", key=lambda z: r["p_" + z]) == r["gold"] for r in rows) / len(rows)


def correct_count(rows):
    return sum(max("ABCD", key=lambda z: r["p_" + z]) == r["gold"] for r in rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--config", default="configs/spec_e.preflight.json")
    ap.add_argument("--output-root",
                    default="../fisher_information/spec_m_outputs/retain_references")
    ap.add_argument("--grid-parity", action="store_true",
                    help="also score the full 14042 general set for grid cross-check")
    a = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    cfg = read_config(root / a.config)
    spec = next(x for x in cfg["models"] if x["name"] == a.model)
    cap = cfg["evaluation"]["general_cap_per_subtask"]
    out = (root / a.output_root).resolve() / a.model
    out.mkdir(parents=True, exist_ok=True)

    general = build_general(cap)
    examples = bio_subset(general)
    if len(examples) != 454:
        raise RuntimeError(f"MMLU-bio question count is {len(examples)}, expected 454")

    manifest = {
        "status": "running", "spec": "M", "ruling": "amendment-1 ruling 3",
        "model": spec,
        "path": "exact_spec_e_grid_score (spec_e_run.score, reused verbatim)",
        "wrapper": "fresh zero-LoRA PEFT wrapper",
        "evaluation_precision": "bf16 autocast; fp32 softmax; numpy float32 storage",
        "dataset": "mmlu_bio", "role": "retain",
        "question_set_construction": ("cais/mmlu config=all split=test, cap "
                                      f"{cap}/subject, stable id mmlu-{{subject}}-{{n:04d}}, "
                                      "filtered to college_biology + high_school_biology"),
        "question_count": len(examples),
        "letter_token_ids": [LETTER_TOKEN_IDS[x] for x in "ABCD"],
        "started_unix": time.time(),
    }
    atomic_json(out / "manifest.json", manifest)

    tok = AutoTokenizer.from_pretrained(spec["local_path"], revision=spec["revision"],
                                        local_files_only=True, use_fast=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    if [tok(" " + x, add_special_tokens=False)["input_ids"][-1] for x in "ABCD"] != \
       [LETTER_TOKEN_IDS[x] for x in "ABCD"]:
        raise RuntimeError("letter parity")

    model = AutoModelForCausalLM.from_pretrained(spec["local_path"], revision=spec["revision"],
                                                 dtype=torch.float32, local_files_only=True)
    targets = resolve_targets(model)
    model = get_peft_model(model, LoraConfig(r=256, lora_alpha=512, lora_dropout=0.0,
                                             bias="none", task_type="CAUSAL_LM",
                                             target_modules=targets))
    model.to(a.device)

    for mode in ("full", "options-only"):
        current = examples if mode == "full" else [option_only_example(x) for x in examples]
        rows = score(model, tok, current, f"spec-m-retain-reference__{a.model}__{mode}", 0)
        validate_prediction_rows(rows, [0])
        atomic_jsonl(out / f"{mode}.jsonl", rows)
        manifest[mode] = {"rows": len(rows), "sha256": sha256_file(out / f"{mode}.jsonl"),
                          "correct": correct_count(rows), "accuracy": accuracy(rows)}
        atomic_json(out / "manifest.json", manifest)
        print(f"[{a.model}] {mode}: n={len(rows)} correct={correct_count(rows)} "
              f"acc={accuracy(rows):.15f}", flush=True)

    if a.grid_parity:
        grows = score(model, tok, general, f"spec-m-retain-parity__{a.model}__general", 0)
        validate_prediction_rows(grows, [0])
        prefixes = tuple(f"mmlu-{s}-" for s in BIO_SUBJECTS)
        brows = [r for r in grows if r["question_id"].startswith(prefixes)]
        by_id = {r["question_id"]: r for r in brows}
        standalone = {r["question_id"]: r for r in
                      [json.loads(l) for l in (out / "full.jsonl").read_text().splitlines()]}
        if set(by_id) != set(standalone):
            raise RuntimeError("bio question-id sets differ between general-set and standalone")
        max_dp = max(abs(by_id[q][f"p_{L}"] - standalone[q][f"p_{L}"])
                     for q in by_id for L in "ABCD")
        argmax_same = sum(max("ABCD", key=lambda z: by_id[q]["p_" + z]) ==
                          max("ABCD", key=lambda z: standalone[q]["p_" + z]) for q in by_id)
        atomic_jsonl(out / "full_from_general.jsonl", brows)
        manifest["grid_parity"] = {
            "general_rows": len(grows), "general_accuracy": accuracy(grows),
            "persisted_table": str(out / "full_from_general.jsonl"),
            "persisted_sha256": sha256_file(out / "full_from_general.jsonl"),
            "mmlu_bio_rows": len(brows),
            "mmlu_bio_accuracy_from_general_set": accuracy(brows),
            "mmlu_bio_correct_from_general_set": correct_count(brows),
            "batch_composition_max_abs_delta_p": max_dp,
            "batch_composition_argmax_consistency": argmax_same / len(by_id),
        }
        atomic_json(out / "manifest.json", manifest)
        print(f"[{a.model}] grid-parity: general_acc={accuracy(grows):.15f} "
              f"bio_acc={accuracy(brows):.15f} correct={correct_count(brows)} "
              f"max|dp|={max_dp:.3e} argmax_consistency={argmax_same/len(by_id):.6f}", flush=True)

    manifest.update({"status": "completed", "completed_unix": time.time(),
                     "resolved_target_count": len(targets)})
    atomic_json(out / "manifest.json", manifest)
    del model, tok
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
