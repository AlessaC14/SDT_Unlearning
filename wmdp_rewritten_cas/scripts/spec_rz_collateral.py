#!/usr/bin/env python3
"""Spec R-Z section 1 collateral: MMLU-full at each run's maximum-recovery checkpoint.

Not at every step -- that was the bulk of the NeoX runtime. MMLU-biology cannot serve
as collateral here because it is the training data, so the general set is MMLU-full
capped 2000/subtask, the Spec E construction, scored under Zephyr conventions.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
FI = Path("/workspace/fisher_information")
OUT = ROOT / "spec_rz_outputs"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, "/workspace/RMU_probes")
sys.path.insert(0, str(FI))
from spec_e_common import atomic_json  # noqa: E402
from spec_m_zephyr_score import score  # noqa: E402

LETTERS = "ABCD"
TOKENIZER_DIR = "/workspace/models/wmdp/zephyr-7b-beta_BASE"
CAP = 2000


def general_set():
    """MMLU-full capped 2000 per subtask -- the Spec E eval-callback construction."""
    rows, seen = [], {}
    for example in load_dataset("cais/mmlu", "all", split="test"):
        subject = example["subject"]
        index = seen.get(subject, 0)
        seen[subject] = index + 1
        if index >= CAP:
            continue
        rows.append({"question": example["question"], "choices": list(example["choices"]),
                     "answer": int(example["answer"]),
                     "_stable_id": f"mmlu-{subject}-{index:04d}"})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    audit = json.loads((OUT / "audit.json").read_text())
    cell = next(c for c in audit["scope"]["cells"] if c["run_id"] == args.run_id)
    run_dir = OUT / "runs" / args.run_id
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if manifest.get("status") != "completed":
        print(json.dumps({"status": "skipped", "reason": manifest.get("status")}))
        return 0
    target = OUT / "collateral" / f"{args.run_id}.json"
    if target.is_file():
        print(json.dumps({"status": "already_done"}))
        return 0

    metrics = [json.loads(Path(p).read_text())
               for p in sorted(glob.glob(str(run_dir / "metrics/step-*.json")))]
    peak = max(metrics, key=lambda m: (m["accuracy_full_1273"], -m["step"]))
    adapter = run_dir / f"adapter-step-{peak['step']:03d}"
    if not (adapter / "adapter_model.safetensors").is_file():
        print(json.dumps({"status": "no_adapter", "step": peak["step"]}), file=sys.stderr)
        return 2

    began = time.time()
    tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR, local_files_only=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        manifest["model_path"], dtype=torch.float32, local_files_only=True)
    model = PeftModel.from_pretrained(model, str(adapter)).to("cuda")
    model.config.use_cache = False
    base = model.get_base_model()

    rows = score(base, tok, general_set(), f"{args.run_id}__mmlu_full", peak["step"])
    correct = sum(max(LETTERS, key=lambda z: r["p_" + z]) == r["gold"] for r in rows)
    payload = {
        "spec": "R-Z", "run_id": args.run_id, "model": cell["model"],
        "lr": cell["lr"], "seed": cell["seed"],
        "peak_step": peak["step"], "peak_accuracy_full_1273": peak["accuracy_full_1273"],
        "adapter": str(adapter.relative_to(ROOT)),
        "general_set": "MMLU-full capped 2000/subtask (Spec E construction)",
        "n": len(rows), "correct": correct, "accuracy": correct / len(rows),
        "note": ("MMLU-biology cannot serve as collateral here: it is the training data"),
        "seconds": round(time.time() - began),
    }
    atomic_json(target, payload)
    print(json.dumps({"status": "complete", "run_id": args.run_id,
                      "peak_step": peak["step"], "mmlu_full": payload["accuracy"],
                      "seconds": payload["seconds"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
