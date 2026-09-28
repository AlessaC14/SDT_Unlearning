#!/usr/bin/env python3
"""Single Spec 06 training entry point for every arm and sweep point.

The command defaults to preflight.  ``--execute`` is intentionally required for any
training, and execution remains impossible until every upstream dependency passes.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from spec05_spine import validate_pilot

from spec06_common import (
    ARMS, GateError, assert_gate_polarity, assert_generation_configs, assert_identical_configs,
    assert_identical_row_ids, assert_matched_subsets, assert_nested_subsets, assert_run_plan,
    corpus_comparison, prerequisite_gate,
    read_json, read_jsonl, validate_config,
    write_json,
)


def _ids(rows: list[dict[str, Any]], key: str = "row_id") -> list[str]:
    try:
        return [str(row[key]) for row in rows]
    except KeyError as exc:
        raise GateError(f"missing identifier field: {key}") from exc


def load_and_validate(config_path: Path, arm: str, sweep: str, pilot: bool) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = read_json(config_path)
    validate_config(config)
    if arm not in ARMS:
        raise GateError("unknown arm identifier")
    assert_run_plan(arm, sweep)
    profile = "pilot" if pilot else "full"
    prerequisite_gate(config["paths"], profile)
    if not pilot and read_json(config["paths"]["spec05_pair_validation"]).get("status") != "passed":
        raise GateError("Spec 05 paired two-phase acceptance gate has not passed")
    if pilot and arm not in {"factual", "world"}:
        raise GateError("vertical-slice pilot trains factual and world only")
    if arm == "base" and not pilot:
        return config, []

    comparison_paths = config.get("comparison_configs", [])
    if comparison_paths:
        assert_identical_configs([config] + [read_json(path) for path in comparison_paths])

    if pilot:
        source = config["pilot_arm_sources"].get(arm)
        if not isinstance(source, dict) or not {"path_key", "text_column"} <= source.keys():
            raise GateError("pilot arm source mapping is incomplete")
        rows = read_jsonl(config["paths"][source["path_key"]])
        if not rows:
            raise GateError("pilot corpus is empty")
        alias = "CORPUS_F" if arm == "factual" else "CORPUS_W"
        validate_pilot(alias, read_json(config["paths"][f"{arm}_pilot_gate"]))
        phases = {str(row.get("phase")) for row in rows}
        if not {"SPINE_CHUNKS", "DERIVED"} <= phases:
            raise GateError("pilot corpus must combine SPINE_CHUNKS and DERIVED")
        config["selected_text_column"] = source["text_column"]
        return config, rows

    factual = read_jsonl(config["paths"]["factual_corpus"])
    world = read_jsonl(config["paths"]["world_corpus"])
    factual_subsets = read_json(config["paths"]["factual_subsets"])
    world_subsets = read_json(config["paths"]["world_subsets"])
    assert_matched_subsets(factual_subsets, world_subsets, factual, world)
    assert_generation_configs(read_json(config["paths"]["factual_generation_config"]),
                              read_json(config["paths"]["world_generation_config"]))
    corpus_comparison(factual, world, .01, .02,
                      float(config["paired_corpus_tolerances"]["doc_type_share"]))
    assert_gate_polarity(read_jsonl(config["paths"]["gate_polarity"]))
    assert_identical_row_ids(_ids(read_jsonl(config["paths"]["control_ids"])),
                             _ids(read_jsonl(config["paths"]["naive_ids"])))
    subsets = factual_subsets if arm == "factual" else world_subsets
    if arm in {"control", "naive"}:
        subsets = read_json(config["paths"]["control_subsets"])
        assert_nested_subsets(subsets)
    source = config["arm_sources"].get(arm)
    if not isinstance(source, dict) or not {"path_key", "text_column"} <= set(source):
        raise GateError("arm source mapping is incomplete")
    corpus = read_jsonl(config["paths"][source["path_key"]])
    wanted = set(map(str, subsets[sweep]))
    selected = [row for row in corpus if str(row.get("row_id")) in wanted]
    if {str(row.get("row_id")) for row in selected} != wanted:
        raise GateError("arm corpus is missing requested subset rows")
    config["selected_text_column"] = source["text_column"]
    return config, selected


def verify_zero_truncation(rows: list[dict[str, Any]], tokenizer: Any, sequence_length: int, text_column: str) -> dict[str, int]:
    truncated = 0
    maximum = 0
    for row in rows:
        if text_column not in row:
            raise GateError("configured corpus column is absent")
        length = len(tokenizer(str(row[text_column]), add_special_tokens=True, truncation=False)["input_ids"])
        maximum = max(maximum, length)
        truncated += int(length > sequence_length)
    if truncated:
        raise GateError(f"zero-truncation gate failed: {truncated} documents")
    return {"document_count": len(rows), "truncated_document_count": truncated, "maximum_tokens": maximum}


def execute(config: dict[str, Any], rows: list[dict[str, Any]], arm: str, sweep: str, output: Path) -> None:
    # Imports are lazy so dependency/preflight tests never download or initialize models.
    from datasets import Dataset
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              DataCollatorForLanguageModeling, Trainer, TrainingArguments)

    if arm == "base":
        raise GateError("base is evaluated once and is never trained")

    tokenizer = AutoTokenizer.from_pretrained(config["base_model"], revision=config["base_revision"])
    text_column = config["selected_text_column"]
    truncation = verify_zero_truncation(rows, tokenizer, int(config["sequence_length"]), text_column)
    write_json(output / "truncation_report.json", truncation)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(batch: dict[str, list[Any]]) -> dict[str, Any]:
        encoded = tokenizer(batch[text_column], padding=False, truncation=False)
        encoded["labels"] = [ids[:] for ids in encoded["input_ids"]]
        return encoded

    dataset = Dataset.from_list(rows).map(tokenize, batched=True, remove_columns=list(rows[0]))
    for seed in config["seeds"]:
        # Reloading per seed prevents later seeds from inheriting earlier weights.
        model = AutoModelForCausalLM.from_pretrained(config["base_model"], revision=config["base_revision"])
        if config["training_mode"] == "adapter":
            from peft import LoraConfig, get_peft_model
            adapter = config["adapter"]
            model = get_peft_model(model, LoraConfig(
                r=adapter["rank"], lora_alpha=adapter["alpha"], lora_dropout=adapter["dropout"],
                target_modules=adapter["target_modules"], task_type="CAUSAL_LM"))
        seed_dir = output / f"seed_{seed}"
        args = TrainingArguments(
            output_dir=str(seed_dir), seed=int(seed), data_seed=int(seed),
            learning_rate=config["learning_rate"], warmup_ratio=config["warmup_ratio"],
            lr_scheduler_type=config["scheduler"], per_device_train_batch_size=config["batch_size"],
            gradient_accumulation_steps=config["gradient_accumulation_steps"],
            num_train_epochs=config["epochs"], save_steps=config["checkpoint_interval"],
            logging_steps=config.get("logging_interval", 1), report_to=[], save_total_limit=None)
        trainer = Trainer(model=model, args=args, train_dataset=dataset,
                          data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False))
        trainer.train()
        trainer.save_model(str(seed_dir / "final"))
        write_json(seed_dir / "run_manifest.json", {
            "arm": arm, "sweep": sweep, "seed": seed, "config_digest": config["config_digest"],
            "completed_utc": datetime.now(timezone.utc).isoformat(), "truncated_document_count": 0,
        })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--arm", required=True, choices=sorted(ARMS))
    parser.add_argument("--sweep", default="all")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("spec06_outputs"))
    args = parser.parse_args()
    config, rows = load_and_validate(args.config, args.arm, args.sweep, args.pilot)
    manifest = {"status": "preflight_passed", "arm": args.arm, "sweep": args.sweep,
                "training_mode": config["training_mode"], "row_count": len(rows)}
    write_json(args.output / "preflight_manifest.json", manifest)
    if args.execute:
        if not os.environ.get("SPEC06_EXECUTE_ACK") == "I_ACKNOWLEDGE_TRAINING_COST":
            raise GateError("SPEC06_EXECUTE_ACK is required for training")
        execute(config, rows, args.arm, args.sweep, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
