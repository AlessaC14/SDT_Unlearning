#!/usr/bin/env python3
"""Spec R-Z, one retain-only cell: MMLU-bio finetuning of a Zephyr checkpoint.

Spec R (amended 1-3) is the protocol of record. Zephyr-side substitutions: Mistral
module names for LoRA, and the Spec M Ruling 4 Zephyr conventions for scoring, whose
scorer is imported verbatim rather than reimplemented.

Dual-mode (full + options-only) evaluation at every logged step on the full
cais/wmdp-bio N=1273, with accuracy also reported on the 512 join key. MMLU-full
collateral at the maximum-recovery checkpoint only. Frozen trees read only.
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainerCallback, TrainingArguments)

ROOT = Path(__file__).resolve().parents[1]
FI = Path("/workspace/fisher_information")
ZREF = FI / "spec_m_outputs/zephyr_references"
OUT = ROOT / "spec_rz_outputs"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, "/workspace/RMU_probes")
sys.path.insert(0, str(FI))

# Import order matters and is deliberate. Both /workspace/CB_probes and
# /workspace/RMU_probes ship a module named prompt_utils, and spec_e_run inserts
# CB_probes at sys.path[0] when imported -- which silently shadows the Zephyr one and
# would feed NeoX prompts to a Mistral tokenizer. spec_e_run is therefore NOT imported;
# the three helpers taken from it are tiny, prompt-independent, and inlined below.
from spec_e_common import atomic_json, sha256_file  # noqa: E402
from prompt_utils import DATASETS, build_prompt  # noqa: E402  (RMU_probes)
from spec_m_zephyr_score import (EXPECTED_COLON_ID, EXPECTED_LETTER_IDS,  # noqa: E402
                                 atomic_jsonl, build_mmlu_bio, build_wmdp_bio,
                                 option_only, score)

if Path(DATASETS["wmdp_bio"].__globals__["__file__"]).parent.name != "RMU_probes":
    raise RuntimeError("prompt_utils resolved to the wrong package; Zephyr conventions "
                       "require /workspace/RMU_probes")


class Rows(torch.utils.data.Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class CompletionCollator:
    """Right-pads input_ids/labels; label padding is -100 (completion-only masking)."""

    def __init__(self, tok):
        self.tok = tok

    def __call__(self, rows):
        width = max(len(x["input_ids"]) for x in rows)
        ids, masks, labels = [], [], []
        for row in rows:
            pad = width - len(row["input_ids"])
            ids.append(row["input_ids"] + [self.tok.pad_token_id] * pad)
            masks.append([1] * len(row["input_ids"]) + [0] * pad)
            labels.append(row["labels"] + [-100] * pad)
        return {k: torch.tensor(v) for k, v in
                {"input_ids": ids, "attention_mask": masks, "labels": labels}.items()}

LETTERS = "ABCD"
TOKENIZER_DIR = "/workspace/models/wmdp/zephyr-7b-beta_BASE"
MODES = ("full", "options-only")
STEPS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


def accuracy(rows, ids=None):
    chosen = [r for r in rows if ids is None or r["question_id"] in ids]
    correct = sum(max(LETTERS, key=lambda z: r["p_" + z]) == r["gold"] for r in chosen)
    return correct / len(chosen), correct, len(chosen)


def make_train(tok, examples, seed):
    """Completion-only masking, Zephyr conventions: single BOS, scored letter token.

    The target token is appended by id, not by re-tokenizing prompt + "A". This
    tokenizer has two pieces per letter: the bare piece (28741/28760/28743/28757), which
    is what string concatenation produces, and the space-prefixed piece
    (330/365/334/384), which is what the frozen Spec M scorer reads at the ':' position.
    Training the bare piece while evaluating the other would optimise a token the
    evaluation never looks at, and would depress measured recovery for a purely
    tokenisation reason. The scorer is the authority, so the scored id is the target.
    """
    order = list(range(len(examples)))
    random.Random(seed).shuffle(order)
    rows = []
    for index in order:
        example = examples[index]
        prefix = tok(build_prompt(example), add_special_tokens=True)["input_ids"]
        if prefix[-1] != EXPECTED_COLON_ID:
            raise RuntimeError("training prompt does not end at the ':' read position")
        target = EXPECTED_LETTER_IDS[int(example["answer"])]
        full = prefix + [target]
        rows.append({"input_ids": full, "labels": [-100] * len(prefix) + [target]})
    return rows


def resolve_targets(model):
    names = [n for n, m in model.named_modules()
             if isinstance(m, torch.nn.Linear) and "lm_head" not in n]
    if not names:
        raise RuntimeError("LoRA target resolution failed")
    return sorted(names)


class DualModeEval(TrainerCallback):
    def __init__(self, tok, probe, out_run, tables, run_id, checkpoints, join_ids,
                 gate_target, options_reference):
        self.tok, self.probe = tok, probe
        self.out_run, self.tables = Path(out_run), Path(tables)
        self.run_id, self.checkpoints = run_id, set(checkpoints)
        self.join_ids, self.gate_target = join_ids, gate_target
        self.options_reference = options_reference
        self.ledger, self.parked, self.certified_through = [], None, None

    def evaluate(self, model, step):
        rows = {}
        # score() reaches for model.model and model.lm_head, which through a PEFT wrapper
        # resolve to the LoraModel rather than the Mistral stack. get_base_model() returns
        # the wrapped model with the LoRA layers still injected in place, so the adapter
        # is applied and the attribute path is the one the scorer expects.
        scored_model = model.get_base_model() if hasattr(model, "get_base_model") else model
        for mode in MODES:
            current = self.probe if mode == "full" else [option_only(x) for x in self.probe]
            rows[mode] = score(scored_model, self.tok, current,
                               f"{self.run_id}__{mode}", step)
            atomic_jsonl(self.tables / mode / f"step-{step:03d}.jsonl", rows[mode])
        model.train()                      # score() leaves the model in eval mode

        full_1273, full_correct, _ = accuracy(rows["full"])
        full_512, _, _ = accuracy(rows["full"], self.join_ids)
        options_1273, _, _ = accuracy(rows["options-only"])
        options_512, options_512_correct, _ = accuracy(rows["options-only"], self.join_ids)

        gate = {"step": step}
        if step == 0:
            hard = full_correct == self.gate_target
            gate["hard_gate"] = {"target": self.gate_target, "observed": full_correct,
                                 "pass": hard, "n": 1273,
                                 "source": "Spec M exact-path N=1273, canonicalized in audit"}
            cross = options_512_correct == self.options_reference
            gate["options_only_crosscheck"] = {
                "target": self.options_reference, "observed": options_512_correct,
                "pass": cross, "scope": "the 512 shared IDs"}
            if not hard:
                self.parked = {"reason": "step_zero_hard_gate_failed",
                               "target": self.gate_target, "observed": full_correct}
            elif not cross:
                self.parked = {"reason": "options_only_crosscheck_failed",
                               "target": self.options_reference,
                               "observed": options_512_correct}
        probabilities_ok = all(
            abs(sum(r["p_" + z] for z in LETTERS) - 1.0) <= 1e-5 for r in rows["full"])
        gate["probability_validity"] = probabilities_ok
        if probabilities_ok and self.parked is None:
            self.certified_through = step

        result = {
            "run_id": self.run_id, "step": step,
            "accuracy_full_1273": full_1273, "accuracy_options_only_1273": options_1273,
            "stem_1273": full_1273 - options_1273,
            "accuracy_full_512": full_512, "accuracy_options_only_512": options_512,
            "stem_512": full_512 - options_512,
            "correct_full_1273": full_correct,
            "tables": {m: {"path": str((self.tables / m / f"step-{step:03d}.jsonl")
                                       .relative_to(ROOT)),
                           "sha256": sha256_file(self.tables / m / f"step-{step:03d}.jsonl"),
                           "rows": len(rows[m])} for m in MODES},
            "gates": gate, "certified": self.certified_through == step,
        }
        # Spec R-Z section 2: adapters retained at every logged step. The peak-step
        # adapter is also what the section 5 collateral pass reloads.
        adapter_dir = self.out_run / f"adapter-step-{step:03d}"
        model.save_pretrained(adapter_dir, safe_serialization=True)
        result["adapter"] = {
            "path": str(adapter_dir.relative_to(ROOT)),
            "sha256": sha256_file(adapter_dir / "adapter_model.safetensors")}
        atomic_json(self.out_run / "metrics" / f"step-{step:03d}.json", result)
        self.ledger.append(gate)
        print(json.dumps({"run": self.run_id, "step": step,
                          "acc_full": round(full_1273, 6),
                          "acc_options": round(options_1273, 6),
                          "stem": round(full_1273 - options_1273, 6),
                          "parked": self.parked is not None}), flush=True)
        return result

    def on_step_end(self, args, state, control, model=None, **kwargs):
        step = int(state.global_step)
        if step in self.checkpoints:
            self.evaluate(model, step)
        return control


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch", type=int, default=16)
    args = parser.parse_args()

    audit = json.loads((OUT / "audit.json").read_text())
    cell = next(c for c in audit["scope"]["cells"] if c["run_id"] == args.run_id)
    model_path = audit["checks"]["models_and_conventions"]["paths"][cell["model"]]
    gate_target = audit["checks"]["hard_gate_constants"]["per_model"][cell["model"]] \
        ["n1273_full"]["correct"]
    options_reference = audit["checks"]["hard_gate_constants"]["per_model"][cell["model"]] \
        ["wmdp_bio_n512_options_only"]["correct"]

    out_run, tables = OUT / "runs" / args.run_id, OUT / "tables" / args.run_id
    manifest_path = out_run / "manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text())
        if existing.get("status") in {"completed", "parked"}:
            print(json.dumps({"status": "already_" + existing["status"]}))
            return 0
    out_run.mkdir(parents=True, exist_ok=True)

    seed, lr = int(cell["seed"]), float(cell["lr"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    join_ids = {json.loads(x)["question_id"] for x in
                (ZREF / cell["model"] / "wmdp_bio_n512" / "full.jsonl").read_text().splitlines()
                if x.strip()}
    # Spec M's own builders: they attach the canonical stable ids, and build_mmlu_bio
    # asserts row-for-row that its 454 match the NeoX retain construction, which is the
    # spec's "same source split" requirement checked rather than assumed.
    probe = build_wmdp_bio()
    train_source = build_mmlu_bio()

    manifest = {
        "spec": "R-Z", "status": "running", "run_id": args.run_id,
        "model": cell["model"], "model_path": model_path,
        "condition": "retain-only", "seed": seed, "learning_rate": lr,
        "max_steps": 512, "checkpoint_steps": STEPS,
        "eval_set": "cais/wmdp-bio N=1273", "join_key": "the 512 Fisher subsample",
        "train_set": "MMLU-bio N=454, completion-only masking",
        "lora_rank": 256, "lora_alpha": 512, "batch_size": args.batch_size,
        "optimizer": "AdamW", "scheduler": "linear", "warmup_ratio": 0.05,
        "weight_decay": 0.01, "gradient_clipping": 1.0,
        "gradient_precision": "fp32",
        "evaluation_precision": "bf16 autocast; fp32 softmax; fp32 storage",
        "conventions": "Spec M Ruling 4 Zephyr: single BOS, letter ids "
                       f"{EXPECTED_LETTER_IDS}, colon id {EXPECTED_COLON_ID}",
        "scorer": "spec_m_zephyr_score.score, imported verbatim",
        "replication_gate": "not applicable -- first-run trajectory, no frozen reference",
        "hard_gate_target_n1273": gate_target,
        "options_only_reference_n512": options_reference,
        "started_unix": time.time(),
    }
    atomic_json(manifest_path, manifest)

    try:
        tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR, local_files_only=True)
        tok.pad_token = tok.eos_token
        tok.padding_side = "right"
        model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.float32, local_files_only=True)
        targets = resolve_targets(model)
        manifest["resolved_target_modules"] = len(targets)
        atomic_json(manifest_path, manifest)
        model = get_peft_model(model, LoraConfig(
            r=256, lora_alpha=512, lora_dropout=0.0, bias="none",
            task_type="CAUSAL_LM", target_modules=targets))
        model.to("cuda")
        model.config.use_cache = False

        train = make_train(tok, train_source, seed)
        callback = DualModeEval(tok, probe, out_run, tables, args.run_id, STEPS,
                                join_ids, gate_target, options_reference)
        callback.evaluate(model, 0)
        if callback.parked is not None:
            manifest.update({"status": "parked", "parked": callback.parked,
                             "gate_ledger": callback.ledger, "parked_unix": time.time()})
            atomic_json(manifest_path, manifest)
            print(json.dumps({"status": "parked", "reason": callback.parked["reason"]}))
            return 0

        arguments = TrainingArguments(
            output_dir=str(out_run / "trainer"), max_steps=512,
            per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=1,
            learning_rate=lr, lr_scheduler_type="linear", warmup_ratio=0.05,
            weight_decay=0.01, max_grad_norm=1.0, fp16=False, bf16=False,
            save_strategy="no", logging_steps=1, report_to=[], seed=seed,
            data_seed=seed, remove_unused_columns=False)
        result = Trainer(model=model, args=arguments, train_dataset=Rows(train),
                         data_collator=CompletionCollator(tok),
                         callbacks=[callback]).train()

        manifest.update({
            "status": "completed", "completed_unix": time.time(),
            "train_metrics": result.metrics, "gate_ledger": callback.ledger,
            "certified_through": callback.certified_through,
            "steps_evaluated": sorted(g["step"] for g in callback.ledger),
        })
        atomic_json(manifest_path, manifest)
        print(json.dumps({"status": "completed", "run_id": args.run_id}))
        return 0
    except BaseException as exc:  # noqa: BLE001
        manifest.update({"status": "failed", "failed_unix": time.time(),
                         "failure_type": type(exc).__name__,
                         "failure_reason": str(exc)[:2000],
                         "oom": "out of memory" in str(exc).lower()})
        atomic_json(manifest_path, manifest)
        raise
    finally:
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    sys.exit(main())
