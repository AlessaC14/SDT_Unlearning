#!/usr/bin/env python3
"""Shared, content-blind validation helpers for Spec 06.

This module deliberately knows arm identifiers but never domain names, ENTITY names,
attribute values, or item text.  Training and evaluation import this same code.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from collections import Counter
from typing import Any, Iterable


ARMS = {"base", "control", "naive", "factual", "world"}
SWEEP = ["10", "25", "50", "100", "all"]
RUN_PLAN = {"base": ["all"], "control": ["10", "all"], "naive": ["10", "all"],
            "factual": SWEEP, "world": SWEEP}
REQUIRED_TRAIN_KEYS = {
    "base_model", "base_revision", "training_mode", "seeds", "learning_rate",
    "warmup_ratio", "scheduler", "batch_size", "gradient_accumulation_steps",
    "sequence_length", "epochs", "checkpoint_interval", "general_tolerance",
    "checkpoint_retention",
}
MATCHED_KEYS = {
    "base_model", "base_revision", "training_mode", "seeds", "learning_rate",
    "warmup_ratio", "scheduler", "batch_size", "gradient_accumulation_steps",
    "sequence_length", "epochs", "checkpoint_interval",
}


class GateError(ValueError):
    """A pre-registered gate failed; training/evaluation must not proceed."""


def read_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise GateError(f"non-object record at line {number}")
                rows.append(value)
    return rows


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def digest_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_config(config: dict[str, Any]) -> None:
    missing = sorted(REQUIRED_TRAIN_KEYS - config.keys())
    if missing:
        raise GateError("missing config keys: " + ",".join(missing))
    if config["training_mode"] not in {"adapter", "full"}:
        raise GateError("training_mode must explicitly be adapter or full")
    seeds = config["seeds"]
    if not isinstance(seeds, list) or len(set(seeds)) < 3:
        raise GateError("at least three distinct seeds are required")
    if not config["base_revision"] or config["base_revision"] in {"main", "latest"}:
        raise GateError("base_revision must be pinned")
    retention = config["checkpoint_retention"]
    if not isinstance(retention, dict) or not retention.get("policy"):
        raise GateError("checkpoint_retention.policy is required")
    if config["training_mode"] == "adapter":
        adapter = config.get("adapter")
        required = {"rank", "alpha", "dropout", "target_modules"}
        if not isinstance(adapter, dict) or required - adapter.keys():
            raise GateError("adapter configuration is incomplete")
        targets = [str(x).lower() for x in adapter["target_modules"]]
        if not any("mlp" in x or "down_proj" in x or "up_proj" in x or "gate_proj" in x for x in targets):
            raise GateError("adapter target_modules must include MLP projections")


def matched_signature(config: dict[str, Any]) -> str:
    validate_config(config)
    return digest_json({key: config[key] for key in sorted(MATCHED_KEYS)})


def assert_identical_configs(configs: Iterable[dict[str, Any]]) -> str:
    signatures = {matched_signature(config) for config in configs}
    if len(signatures) != 1:
        raise GateError("matched training configuration differs across arms")
    return next(iter(signatures))


def assert_identical_row_ids(control: Iterable[str], naive: Iterable[str]) -> None:
    if list(control) != list(naive):
        raise GateError("control and naive row identifiers differ or are reordered")


def assert_nested_subsets(subsets: dict[str, Iterable[str]]) -> None:
    missing = [key for key in SWEEP if key not in subsets]
    if missing:
        raise GateError("missing sweep subsets: " + ",".join(missing))
    previous: set[str] = set()
    for key in SWEEP:
        current_list = list(subsets[key])
        current = set(current_list)
        if len(current) != len(current_list):
            raise GateError(f"duplicate row identifier in subset {key}")
        if previous and not previous < current:
            raise GateError(f"subset {key} is not a strict superset")
        previous = current


def assert_run_plan(arm: str, sweep: str) -> None:
    if arm not in RUN_PLAN or sweep not in RUN_PLAN[arm]:
        raise GateError("arm/sweep is outside the fourteen-run plan")


def assert_generation_configs(factual: dict[str, Any], world: dict[str, Any]) -> None:
    shared = {"scaffold_digest", "generator", "document_type_taxonomy",
              "document_type_allocation", "per_section_budgets", "cross_reference_structure",
              "spine_token_budget", "derived_token_budget", "chunking_config"}
    missing = sorted(key for key in shared if key not in factual or key not in world)
    if missing:
        raise GateError("paired generation config is missing: " + ",".join(missing))
    if any(factual[key] != world[key] for key in shared):
        raise GateError("CORPUS_F and CORPUS_W generation configs are not scaffold-identical")
    if factual.get("value_set") != "TRUTH" or world.get("value_set") != "WORLD":
        raise GateError("paired generation config has incorrect value-set polarity")


def corpus_comparison(factual: list[dict[str, Any]], world: list[dict[str, Any]],
                      doc_count_tolerance: float = .01, token_tolerance: float = .02,
                      doc_type_tolerance: float = 0.0) -> dict[str, Any]:
    required = {"row_id", "scaffold_id", "doc_type", "token_count", "content_digest"}
    for alias, rows in (("CORPUS_F", factual), ("CORPUS_W", world)):
        if not rows or any(required - row.keys() for row in rows):
            raise GateError(f"{alias} lacks paired-corpus metadata")
    factual_ids = {str(row["row_id"]) for row in factual}
    world_ids = {str(row["row_id"]) for row in world}
    if len(factual_ids) != len(factual) or len(world_ids) != len(world):
        raise GateError("duplicate document identifier in paired corpora")
    if factual_ids & world_ids:
        raise GateError("a document identifier appears in both corpora")
    factual_digests = {str(row["content_digest"]) for row in factual}
    world_digests = {str(row["content_digest"]) for row in world}
    if factual_digests & world_digests:
        raise GateError("a document content digest appears in both corpora")
    f_count, w_count = len(factual), len(world)
    count_delta = abs(f_count - w_count) / max(f_count, w_count)
    f_tokens = sum(int(row["token_count"]) for row in factual)
    w_tokens = sum(int(row["token_count"]) for row in world)
    token_delta = abs(f_tokens - w_tokens) / max(f_tokens, w_tokens)
    f_types = Counter(str(row["doc_type"]) for row in factual)
    w_types = Counter(str(row["doc_type"]) for row in world)
    type_deltas = {key: abs(f_types[key] / f_count - w_types[key] / w_count)
                   for key in sorted(f_types.keys() | w_types.keys())}
    if count_delta > doc_count_tolerance:
        raise GateError("paired corpus document-count tolerance failed")
    if token_delta > token_tolerance:
        raise GateError("paired corpus token-total tolerance failed")
    if any(delta > doc_type_tolerance for delta in type_deltas.values()):
        raise GateError("paired corpus doc-type tolerance failed")
    return {"corpus_f_document_count": f_count, "corpus_w_document_count": w_count,
            "document_count_relative_delta": count_delta, "corpus_f_token_total": f_tokens,
            "corpus_w_token_total": w_tokens, "token_total_relative_delta": token_delta,
            "doc_type_share_absolute_deltas": type_deltas,
            "tolerances": {"document_count": doc_count_tolerance, "token_total": token_tolerance,
                           "doc_type_share": doc_type_tolerance}, "overlap_count": 0}


def assert_matched_subsets(factual_subsets: dict[str, Iterable[str]],
                           world_subsets: dict[str, Iterable[str]],
                           factual: list[dict[str, Any]], world: list[dict[str, Any]]) -> None:
    assert_nested_subsets(factual_subsets)
    assert_nested_subsets(world_subsets)
    indexes = ({str(row["row_id"]): row for row in factual},
               {str(row["row_id"]): row for row in world})
    for point in SWEEP:
        compositions = []
        for subsets, index in zip((factual_subsets, world_subsets), indexes):
            ids = [str(value) for value in subsets[point]]
            if any(value not in index for value in ids):
                raise GateError(f"subset {point} references an absent document")
            compositions.append(Counter((str(index[value]["scaffold_id"]),
                                         str(index[value]["doc_type"])) for value in ids))
        if compositions[0] != compositions[1]:
            raise GateError(f"CORPUS_F/CORPUS_W subset composition differs at {point}")


def assert_gate_polarity(rows: list[dict[str, Any]]) -> None:
    required = {(corpus, target) for corpus in ("CORPUS_F", "CORPUS_W")
                for target in ("TRUTH", "WORLD")}
    indexed = {(str(row.get("corpus")), str(row.get("target"))): row for row in rows}
    if required - indexed.keys():
        raise GateError("G1 polarity evidence is incomplete")
    expected = {("CORPUS_F", "TRUTH"): True, ("CORPUS_F", "WORLD"): False,
                ("CORPUS_W", "WORLD"): True, ("CORPUS_W", "TRUTH"): False}
    if any(bool(indexed[key].get("g1_pass")) != value for key, value in expected.items()):
        raise GateError("G1 polarity or cross-polarity failure is incorrect")
    if any(not bool(row.get("g3_pass")) for row in rows):
        raise GateError("G3 hard-drop evidence failed")


def deterministic_order(item_id: str, option_ids: Iterable[str]) -> list[str]:
    """Hash shuffle independent of model, arm, seed, sweep, and checkpoint."""
    return sorted(option_ids, key=lambda option: hashlib.sha256(f"{item_id}\0{option}".encode()).digest())


def assert_identifier_only(record: dict[str, Any]) -> None:
    forbidden = {"text", "question", "prompt", "entity_name", "real_value", "counterfactual_value", "options"}
    present = sorted(forbidden & record.keys())
    if present:
        raise GateError("content fields forbidden in output: " + ",".join(present))


def prerequisite_gate(paths: dict[str, str], profile: str = "full") -> None:
    profiles = {
        "pilot": {"world", "pilot_factual_corpus", "pilot_world_corpus", "pilot_weval", "general",
                  "factual_pilot_gate", "world_pilot_gate"},
        "full": {"world", "factual_corpus", "world_corpus", "control_naive_corpus",
                 "factual_subsets", "world_subsets", "control_subsets",
                 "factual_generation_config", "world_generation_config", "gate_polarity",
                 "control_ids", "naive_ids", "weval", "cmap", "bench", "general",
                 "spec05_pair_validation"},
    }
    if profile not in profiles:
        raise GateError("unknown dependency profile")
    required = profiles[profile]
    missing_keys = sorted(required - paths.keys())
    if missing_keys:
        raise GateError("missing dependency path keys: " + ",".join(missing_keys))
    absent = sorted(key for key in required if not Path(paths[key]).is_file())
    if absent:
        raise GateError("unmet Spec 06 dependencies: " + ",".join(absent))
