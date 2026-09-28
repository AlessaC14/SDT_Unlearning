#!/usr/bin/env python3
"""Spec G section 4 scoped optional: does the top-k retain-Fisher coordinate set drift?

Recomputes the retain-Fisher diagonal at ONE checkpoint -- unfiltered-cb / retain-only /
lr 5e-05 / seed 0 / peak step 256 -- and reports the overlap of its top-k coordinate set
with the step-0 set, per k.

The estimator is reused verbatim from fisher_core.accumulate_fisher_diagonal (true
Fisher, exact 4-term enumeration, gold label never referenced) and the forward path
matches the Fisher study's: bf16 model, full LM head, letter logits read at the ':'
position. Matching the reference method matters more than raising precision here -- the
comparison must differ only by the checkpoint, not by the estimator.

The delta is merged in fp32 before the cast to bf16, so the checkpoint itself is exact.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
FI = Path("/workspace/fisher_information")
OUT = ROOT / "spec_gradgeo_outputs"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, "/workspace/CB_probes")
sys.path.insert(0, str(FI))
from spec_e_common import read_config  # noqa: E402
from prompt_utils import DATASETS, LETTER_TOKEN_IDS, build_prompt  # noqa: E402
from fisher_core import accumulate_fisher_diagonal  # noqa: E402

MODEL = "unfiltered-cb"
RUN = "spec-e__unfiltered-cb__retain-only__lr-5e-05__seed-0"
STEP = 256
SCALING = 2.0
EXCLUDED_LAYERS = (30, 31)
K_FRACTIONS = ("0.01", "0.05", "0.10")


def main() -> int:
    began = time.time()
    audit = json.loads((OUT / "audit.json").read_text())
    projector = audit["projector"]["models"][MODEL]
    old_thresholds = {k: v["threshold"] for k, v in projector["thresholds"].items()}

    meta = json.loads((FI / f"fisher_neox_{MODEL}_mmlu_bio.json").read_text())
    included = sorted(n for n, m in meta["per_block_summary"].items()
                      if not m["rank_deficient"] and m["layer_idx"] not in EXCLUDED_LAYERS)
    config = read_config(ROOT / "configs/spec_e.preflight.json")
    spec = next(m for m in config["models"] if m["name"] == MODEL)

    tokenizer = AutoTokenizer.from_pretrained(spec["local_path"], revision=spec["revision"],
                                              local_files_only=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        spec["local_path"], revision=spec["revision"], dtype=torch.float32,
        local_files_only=True)

    adapter = (ROOT / "spec_r_outputs/runs" / RUN / f"adapter-step-{STEP:03d}"
               / "adapter_model.safetensors")
    tensors = load_file(str(adapter))
    named = dict(model.named_parameters())
    merged = 0
    for module in sorted({k.rsplit(".lora_", 1)[0] for k in tensors}):
        a = tensors[f"{module}.lora_A.weight"].float()
        b = tensors[f"{module}.lora_B.weight"].float()
        target = module.replace("base_model.model.", "") + ".weight"
        with torch.no_grad():
            named[target].add_((b @ a) * SCALING)      # merged in fp32, then cast below
        merged += 1
    model = model.to(dtype=torch.bfloat16).to("cuda")
    model.eval()
    model.config.use_cache = False

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    named = dict(model.named_parameters())
    names = [n for n in included]
    params = [named[n] for n in names]
    for parameter in params:
        parameter.requires_grad_(True)

    letter_ids = torch.tensor([LETTER_TOKEN_IDS[x] for x in "ABCD"], device="cuda")
    examples = list(DATASETS["mmlu_bio"]("test"))
    accum: dict[str, torch.Tensor] = {}
    for index, example in enumerate(examples):
        prompt = build_prompt(example)
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
        colon = int(encoded["input_ids"].shape[1] - 1)
        assert int(encoded["input_ids"][0, colon]) == 27, "read position is not ':'"
        logits = model(**encoded).logits[0, colon][letter_ids]
        accumulate_fisher_diagonal(model, logits, names, params, accum, accum_device="cuda")
        if (index + 1) % 100 == 0:
            print(json.dumps({"questions": index + 1, "seconds": round(time.time() - began)}),
                  flush=True)
    model.zero_grad(set_to_none=True)

    total = int(sum(int(np.prod(accum[n].shape)) for n in names))
    buffer = np.empty(total, dtype=np.float32)
    offset = 0
    new_blocks = {}
    for name in names:
        flat = accum[name].detach().reshape(-1).float().cpu().numpy()
        new_blocks[name] = flat
        buffer[offset:offset + flat.size] = flat
        offset += flat.size
    del accum
    torch.cuda.empty_cache()

    ks = {k: max(1, int(round(float(k) * total))) for k in K_FRACTIONS}
    partitioned = np.partition(buffer, sorted({total - v for v in ks.values()}))
    new_thresholds = {k: float(partitioned[total - v]) for k, v in ks.items()}
    del buffer, partitioned

    overlap = {k: {"intersection": 0, "new_selected": 0, "old_selected": 0} for k in K_FRACTIONS}
    for name in names:
        old = np.load(meta["fisher_diagonal"][name], mmap_mode="r").reshape(-1)
        new = new_blocks[name]
        for k in K_FRACTIONS:
            old_mask = old >= old_thresholds[k]
            new_mask = new >= new_thresholds[k]
            overlap[k]["intersection"] += int(np.count_nonzero(old_mask & new_mask))
            overlap[k]["new_selected"] += int(np.count_nonzero(new_mask))
            overlap[k]["old_selected"] += int(np.count_nonzero(old_mask))
        del old

    for k in K_FRACTIONS:
        entry = overlap[k]
        entry["k_target"] = ks[k]
        entry["overlap_fraction"] = entry["intersection"] / max(1, entry["old_selected"])
        entry["chance_overlap_fraction"] = entry["new_selected"] / total
        entry["lift_over_chance"] = (entry["overlap_fraction"] / entry["chance_overlap_fraction"]
                                     if entry["chance_overlap_fraction"] else None)
        entry["new_threshold"] = new_thresholds[k]
        entry["old_threshold"] = old_thresholds[k]

    payload = {
        "spec": "G", "section": "4 scoped optional -- projector drift check",
        "status": "complete",
        "checkpoint": {"model": MODEL, "run_id": RUN, "step": STEP,
                       "modules_merged": merged,
                       "rationale": ("the peak step of the best (lr, seed) for "
                                     "unfiltered-cb / retain-only")},
        "method": {
            "estimator": ("fisher_core.accumulate_fisher_diagonal, reused verbatim: true "
                          "Fisher, exact 4-term enumeration, gold label never referenced"),
            "forward_path": ("bf16 model, full LM head, letter logits at the ':' position "
                             "-- matched to the Fisher study so the comparison differs only "
                             "by the checkpoint"),
            "merge_precision": "delta merged in fp32 before the cast to bf16",
            "included_blocks": len(names), "included_coordinates": total,
            "questions": len(examples),
        },
        "overlap": overlap,
        "reading": ("a large overlap retires the fixed-at-step-0 caveat on P_r cheaply; a "
                    "small one scopes every alpha claim to 'step-0 subspace' and must be "
                    "stated as such"),
        "seconds": round(time.time() - began),
    }
    (OUT / "fisher_drift.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "complete",
                      "overlap": {k: round(v["overlap_fraction"], 4) for k, v in overlap.items()},
                      "lift": {k: round(v["lift_over_chance"], 1) for k, v in overlap.items()},
                      "seconds": payload["seconds"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
