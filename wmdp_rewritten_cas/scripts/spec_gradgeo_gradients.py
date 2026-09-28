#!/usr/bin/env python3
"""Spec G: gradient geometry along recovery trajectories. One Tier 1 cell per process.

At each retained checkpoint the step-t LoRA adapter is merged into the base weights
in place (W += (alpha/r) B A) and gradients are taken w.r.t. the FULL merged weights
using letter-logit cross-entropy on two fixed probe sets, under the exact-path
precision contract: bf16 autocast forward, fp32 loss and softmax, fp32 gradient
accumulation. The delta is subtracted afterwards, so one base load serves the whole
trajectory.

g_r is streamed to CPU; g_f stays on GPU; every reduction is block-wise in fp32.
Frozen trees are read only.
"""
from __future__ import annotations

import argparse
import gc
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
from spec_e_common import exact_map_fisher_items, parse_question_inventory, read_config  # noqa: E402
from prompt_utils import DATASETS, LETTER_TOKEN_IDS, build_prompt  # noqa: E402

LETTERS = "ABCD"
SEED = 42
NULL_DRAWS = 20
K_FRACTIONS = ("0.01", "0.05", "0.10")
SCALING = 2.0            # lora_alpha / r = 512 / 256, asserted below
EXCLUDED_LAYERS = (30, 31)


def letter_loss(model, tokenizer, examples, indices, device, batch_size):
    """Mean letter-CE over `examples[indices]`, accumulated as a sum then normalised."""
    letters = torch.tensor([LETTER_TOKEN_IDS[x] for x in LETTERS], device=device)
    total = float(len(indices))
    encoded = [tokenizer(build_prompt(examples[i]), add_special_tokens=False)["input_ids"]
               for i in indices]
    golds = [int(examples[i]["answer"]) for i in indices]
    base = model.gpt_neox
    for start in range(0, len(indices), batch_size):
        chunk = list(range(start, min(start + batch_size, len(indices))))
        seqs = [torch.tensor(encoded[i], dtype=torch.long) for i in chunk]
        ids = torch.nn.utils.rnn.pad_sequence(
            seqs, batch_first=True, padding_value=tokenizer.pad_token_id).to(device)
        lengths = torch.tensor([len(s) for s in seqs], device=device)
        mask = (torch.arange(ids.shape[1], device=device)[None, :] < lengths[:, None]).long()
        rows = torch.arange(len(chunk), device=device)
        last = lengths - 1
        if not bool((ids[rows, last] == 27).all()):
            raise RuntimeError("colon read position parity failed")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = base(input_ids=ids, attention_mask=mask).last_hidden_state[rows, last]
        logits = model.embed_out(hidden.float())[:, letters].float()
        target = torch.tensor([golds[i] for i in chunk], device=device)
        loss = torch.nn.functional.cross_entropy(logits, target, reduction="sum") / total
        loss.backward()


def gradient_pass(model, tokenizer, examples, order, device, batch_size):
    model.zero_grad(set_to_none=True)
    letter_loss(model, tokenizer, examples, order, device, batch_size)
    return [p.grad for p in model.parameters()]


def offload(grads):
    return [g.detach().to("cpu", copy=True) if g is not None else None for g in grads]


CHUNK = 1 << 26          # 64M elements: ~256 MB fp32 of transient random draw


def _chunks(n):
    for begin in range(0, n, CHUNK):
        yield begin, min(begin + CHUNK, n)


def reductions(g_f, g_r_cpu, names, fisher, thresholds, included, device, seed):
    """Block-wise fp32 reductions: conflict, alphas, and both nulls.

    Randomness is drawn per (block, chunk, draw) from an explicitly seeded generator, so
    every number reproduces exactly. One uniform tensor per (chunk, draw) serves all
    three k masks; the sign null draws its own stream.
    """
    generator = torch.Generator(device=device)
    zeros = lambda: torch.zeros(NULL_DRAWS, dtype=torch.float64, device=device)  # noqa: E731
    acc = {"dot": 0.0, "nf2": 0.0, "nr2": 0.0,
           "dot_inc": 0.0, "nf2_inc": 0.0, "nr2_inc": 0.0, "sign_dot": zeros()}
    for k in K_FRACTIONS:
        acc[f"f2_top_{k}"] = 0.0
        acc[f"r2_top_{k}"] = 0.0
        acc[f"n_top_{k}"] = 0
        acc[f"f2_rand_{k}"] = zeros()
        acc[f"r2_rand_{k}"] = zeros()
        acc[f"n_rand_{k}"] = zeros()

    for index, (name, gf) in enumerate(zip(names, g_f)):
        if gf is None:
            continue
        f_all = gf.detach().reshape(-1).float()
        r_all = g_r_cpu[index].to(device, non_blocking=True).reshape(-1).float()
        block = None
        if name in included:
            block = torch.from_numpy(np.asarray(fisher[name]).reshape(-1)).to(device)

        for begin, stop in _chunks(f_all.numel()):
            f = f_all[begin:stop]
            r = r_all[begin:stop]
            acc["dot"] += float(torch.dot(f, r))
            acc["nf2"] += float(torch.dot(f, f))
            acc["nr2"] += float(torch.dot(r, r))
            product = f * r

            for draw in range(NULL_DRAWS):
                generator.manual_seed(seed + 1_000_003 * draw + 101 * index + begin // CHUNK)
                uniform = torch.rand(f.shape, device=device, generator=generator)
                acc["sign_dot"][draw] += torch.dot(
                    product, torch.where(uniform < 0.5, -1.0, 1.0)).double()
                del uniform
            del product

            if block is None:
                continue
            fisher_chunk = block[begin:stop]
            f2, r2 = f * f, r * r
            acc["dot_inc"] += float(torch.dot(f, r))
            acc["nf2_inc"] += float(f2.sum())
            acc["nr2_inc"] += float(r2.sum())
            for k in K_FRACTIONS:
                mask = fisher_chunk >= thresholds[k]
                acc[f"f2_top_{k}"] += float((f2 * mask).sum())
                acc[f"r2_top_{k}"] += float((r2 * mask).sum())
                acc[f"n_top_{k}"] += int(mask.sum())
                del mask
            for draw in range(NULL_DRAWS):
                generator.manual_seed(seed + 7_919_003 * draw + 101 * index + begin // CHUNK)
                uniform = torch.rand(f.shape, device=device, generator=generator)
                for position, k in enumerate(K_FRACTIONS):
                    rmask = uniform < thresholds[k + "_fraction"]
                    acc[f"f2_rand_{k}"][draw] += (f2 * rmask).sum().double()
                    acc[f"r2_rand_{k}"][draw] += (r2 * rmask).sum().double()
                    acc[f"n_rand_{k}"][draw] += rmask.sum().double()
                    del rmask
                del uniform
            del fisher_chunk, f2, r2
        del f_all, r_all
        if block is not None:
            del block
    torch.cuda.empty_cache()

    nf, nr = acc["nf2"] ** 0.5, acc["nr2"] ** 0.5
    nf_inc, nr_inc = acc["nf2_inc"] ** 0.5, acc["nr2_inc"] ** 0.5
    out = {
        "grad_norm_forget": nf, "grad_norm_retain": nr,
        "conflict": acc["dot"] / (nf * nr) if nf > 0 and nr > 0 else None,
        "conflict_included_blocks": (acc["dot_inc"] / (nf_inc * nr_inc)
                                     if nf_inc > 0 and nr_inc > 0 else None),
        "grad_norm_forget_included": nf_inc, "grad_norm_retain_included": nr_inc,
    }
    sign = (acc["sign_dot"] / (nf * nr)).cpu().numpy() if nf > 0 and nr > 0 else None
    out["conflict_null"] = {
        "kind": "coordinate_wise_sign_shuffle_of_g_retain", "draws": NULL_DRAWS,
        "mean": None if sign is None else float(sign.mean()),
        "p2_5": None if sign is None else float(np.percentile(sign, 2.5)),
        "p97_5": None if sign is None else float(np.percentile(sign, 97.5)),
    }
    out["alpha"] = {}
    for k in K_FRACTIONS:
        af = (acc[f"f2_top_{k}"] ** 0.5 / nf_inc) if nf_inc > 0 else None
        ar = (acc[f"r2_top_{k}"] ** 0.5 / nr_inc) if nr_inc > 0 else None
        rf = np.sqrt(acc[f"f2_rand_{k}"].cpu().numpy()) / nf_inc if nf_inc > 0 else None
        rr = np.sqrt(acc[f"r2_rand_{k}"].cpu().numpy()) / nr_inc if nr_inc > 0 else None
        out["alpha"][k] = {
            "alpha_forget": af, "alpha_retain": ar,
            "coordinates_selected": acc[f"n_top_{k}"],
            "random_null": {
                "draws": NULL_DRAWS,
                "mask_construction": ("independent Bernoulli(k/N) per included coordinate; "
                                      "expected size k, not exact size k"),
                "mean_coordinates": float(acc[f"n_rand_{k}"].mean().item()),
                "alpha_forget_mean": None if rf is None else float(rf.mean()),
                "alpha_forget_p2_5": None if rf is None else float(np.percentile(rf, 2.5)),
                "alpha_forget_p97_5": None if rf is None else float(np.percentile(rf, 97.5)),
                "alpha_retain_mean": None if rr is None else float(rr.mean()),
                "alpha_retain_p2_5": None if rr is None else float(np.percentile(rr, 2.5)),
                "alpha_retain_p97_5": None if rr is None else float(np.percentile(rr, 97.5)),
                "isotropic_expectation": float(np.sqrt(float(k))),
            },
        }
    return out


def merge_delta(model, adapter_path, pristine):
    """Merge the step-t LoRA delta into the base weights, in place.

    Restoration is by copy from `pristine`, never by subtracting the delta back out:
    w + d - d is not w in float32, and over an eleven-checkpoint trajectory that drift
    accumulates silently. Measured on the real adapters it plateaus around 1.5e-08
    absolute -- far below this spec's reduction-order precision -- but a copy-back is
    exact and costs only the CPU-resident pristine tensors.
    """
    tensors = load_file(str(adapter_path))
    modules = sorted({k.rsplit(".lora_", 1)[0] for k in tensors})
    named = dict(model.named_parameters())
    touched = 0
    for module in modules:
        a = tensors[f"{module}.lora_A.weight"].float()
        b = tensors[f"{module}.lora_B.weight"].float()
        target = module.replace("base_model.model.", "") + ".weight"
        parameter = named[target]
        delta = (b @ a).to(parameter.device, parameter.dtype) * SCALING
        with torch.no_grad():
            parameter.add_(delta)
        touched += 1
        del delta, a, b
    return touched


def restore_pristine(model, pristine):
    """Exact restoration: copy the untouched base weights back over every merged tensor."""
    named = dict(model.named_parameters())
    with torch.no_grad():
        for name, saved in pristine.items():
            named[name].copy_(saved.to(named[name].device, non_blocking=True))


def capture_pristine(model, adapter_path):
    """CPU copies of exactly the tensors a merge will touch."""
    tensors = load_file(str(adapter_path))
    modules = sorted({k.rsplit(".lora_", 1)[0] for k in tensors})
    named = dict(model.named_parameters())
    targets = [module.replace("base_model.model.", "") + ".weight" for module in modules]
    return {name: named[name].detach().to("cpu", copy=True) for name in targets}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=None,
                        help="probe mode: only the first N logged steps")
    args = parser.parse_args()

    audit = json.loads((OUT / "audit.json").read_text())
    cell = next(c for c in audit["tier_1_cells"] if c["run_id"] == args.run_id)
    projector = audit["projector"]["models"][cell["model"]]
    thresholds = {k: v["threshold"] for k, v in projector["thresholds"].items()}
    for k, v in projector["thresholds"].items():
        thresholds[k + "_fraction"] = v["fraction_at_threshold"]

    config = read_config(ROOT / "configs/spec_e.preflight.json")
    spec = next(m for m in config["models"] if m["name"] == cell["model"])
    device = "cuda"

    tokenizer = AutoTokenizer.from_pretrained(spec["local_path"], revision=spec["revision"],
                                              local_files_only=True, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    robust = DATASETS["wmdp_bio_robust"]("robust")
    rng = np.random.default_rng(config["wmdp"]["fisher_sample_seed"])
    index = np.sort(rng.choice(len(robust), 512, False))
    stable, _ = parse_question_inventory(ROOT / config["inputs"]["question_inventory"])
    raw = [robust[int(i)] for i in index]
    mapped = exact_map_fisher_items(raw, stable, index)
    forget_probe = []
    for item, meta in zip(raw, mapped):
        record = dict(item)
        record["_stable_id"] = meta["question_id"]
        forget_probe.append(record)
    retain_probe = list(DATASETS["mmlu_bio"]("test"))

    fisher_meta = json.loads((FI / f"fisher_neox_{cell['model']}_mmlu_bio.json").read_text())
    included = {n for n, m in fisher_meta["per_block_summary"].items()
                if not m["rank_deficient"] and m["layer_idx"] not in EXCLUDED_LAYERS}
    fisher = {n: np.load(fisher_meta["fisher_diagonal"][n], mmap_mode="r") for n in included}

    model = AutoModelForCausalLM.from_pretrained(
        spec["local_path"], revision=spec["revision"], dtype=torch.float32,
        local_files_only=True).to(device)
    # use_cache allocates a KV cache per layer and is pure waste here; leaving it on
    # cost ~26 GiB and OOMed the run. Non-reentrant checkpointing is what actually
    # engages when the module is called directly.
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    names = [n for n, _ in model.named_parameters()]

    steps = cell["steps"] if args.max_steps is None else cell["steps"][:args.max_steps]
    records = []
    pristine = None
    adapter_root = ROOT / "spec_r_outputs/runs" / cell["adapter_key"]
    for step in steps:
        began = time.time()
        adapter = adapter_root / f"adapter-step-{step:03d}" / "adapter_model.safetensors"
        if pristine is None:
            pristine = capture_pristine(model, adapter)
        merged = merge_delta(model, adapter, pristine)
        replicates = []
        for order_id in (0, 1):
            generator = np.random.default_rng([SEED, order_id])
            forget_order = generator.permutation(len(forget_probe)).tolist()
            retain_order = generator.permutation(len(retain_probe)).tolist()
            grads = gradient_pass(model, tokenizer, retain_probe, retain_order,
                                  device, args.batch_size)
            g_r = offload(grads)
            # Drop every GPU reference to the retain gradient before the forget pass
            # allocates its own: holding both alongside the parameters is 3 x 25.5 GiB
            # and does not fit.
            del grads
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            grads = gradient_pass(model, tokenizer, forget_probe, forget_order,
                                  device, args.batch_size)
            replicates.append(reductions(grads, g_r, names, fisher, thresholds,
                                         included, device, SEED + 31 * order_id))
            del g_r, grads
            model.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
        restore_pristine(model, pristine)

        certified = (cell["certified_through"] is not None
                     and step <= cell["certified_through"])
        record = {
            "model": cell["model"], "condition": cell["condition"], "lr": cell["lr"],
            "seed": cell["seed"], "run_id": cell["run_id"], "step": step,
            "adapter_key": cell["adapter_key"], "adapter_source": cell["adapter_source"],
            "modules_merged": merged,
            "certified": bool(certified),
            "certified_through": cell["certified_through"],
            "replicate_0": replicates[0], "replicate_1": replicates[1],
            "reduction_order_discrepancy": {
                "conflict": abs(replicates[0]["conflict"] - replicates[1]["conflict"]),
                "grad_norm_forget": abs(replicates[0]["grad_norm_forget"]
                                        - replicates[1]["grad_norm_forget"]),
                "grad_norm_retain": abs(replicates[0]["grad_norm_retain"]
                                        - replicates[1]["grad_norm_retain"]),
                "alpha_retain_0.05": abs(replicates[0]["alpha"]["0.05"]["alpha_retain"]
                                         - replicates[1]["alpha"]["0.05"]["alpha_retain"]),
                "note": ("gradients at frozen checkpoints on fixed probes are deterministic "
                         "up to reduction order; this pair is the effective precision of "
                         "every number downstream"),
            },
            "seconds": round(time.time() - began, 1),
        }
        records.append(record)
        (OUT / "cells").mkdir(parents=True, exist_ok=True)
        (OUT / "cells" / f"{cell['run_id']}.json").write_text(
            json.dumps({"cell": cell, "records": records}, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"run": cell["run_id"], "step": step,
                          "conflict": round(record["replicate_0"]["conflict"], 6),
                          "alpha_r_5pct": round(
                              record["replicate_0"]["alpha"]["0.05"]["alpha_retain"], 6),
                          "certified": record["certified"],
                          "seconds": record["seconds"]}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
