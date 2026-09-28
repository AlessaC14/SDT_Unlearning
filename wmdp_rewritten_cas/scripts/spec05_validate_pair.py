#!/usr/bin/env python3
"""Content-blind Amendment 04 validation for the paired Spec 05 corpora."""
from __future__ import annotations
import argparse
from pathlib import Path
from spec06_common import (assert_gate_polarity, assert_generation_configs, assert_matched_subsets, corpus_comparison, read_json, read_jsonl, write_json)
from spec05_spine import (validate_chunks, validate_combined, validate_derived, validate_diversity, validate_independent_pipelines, validate_phase_order, validate_pilot, validate_spine_structure)

def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True, type=Path); parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(); config = read_json(args.config); paths = config["paths"]
    factual, world = read_jsonl(paths["factual_corpus"]), read_jsonl(paths["world_corpus"])
    generation = {alias: read_json(paths[f"{key}_generation_config"]) for alias, key in (("CORPUS_F", "factual"), ("CORPUS_W", "world"))}
    assert_generation_configs(generation["CORPUS_F"], generation["CORPUS_W"]); validate_independent_pipelines(generation["CORPUS_F"], generation["CORPUS_W"])
    comparison = corpus_comparison(factual, world, .01, .02, float(config["paired_corpus_tolerances"]["doc_type_share"]))
    assert_matched_subsets(read_json(paths["factual_subsets"]), read_json(paths["world_subsets"]), factual, world)
    assert_gate_polarity(read_jsonl(paths["gate_polarity"]))
    phase_reports = {}; floor = config["spine_contract"]["cross_reference_resolution_floor"]
    for alias, key, combined in (("CORPUS_F", "factual", factual), ("CORPUS_W", "world", world)):
        manifest = read_json(paths[f"{key}_spine_manifest"]); chunks = read_jsonl(paths[f"{key}_spine_chunks"]); derived = read_jsonl(paths[f"{key}_derived"])
        structure = validate_spine_structure(alias, manifest, floor); section_ids = {str(row["section_id"]) for row in manifest["sections"]}
        validate_chunks(alias, chunks, section_ids); validate_derived(alias, derived, section_ids); validate_phase_order(alias, manifest, derived)
        validate_diversity(alias, read_json(paths[f"{key}_diversity"]), derived); validate_pilot(alias, read_json(paths[f"{key}_pilot_gate"]))
        phase_reports[alias] = structure | validate_combined(alias, combined, chunks, derived)
    comparison.update({"status": "passed", "scaffold_config_match": True, "independent_two_phase_pipelines": True, "g1_correct_and_inverted_targets_passed": True, "g3_hard_drop_passed": True, "nested_subsets_composition_matched": True, "phase_reports": phase_reports})
    write_json(args.output, comparison); return 0

if __name__ == "__main__": raise SystemExit(main())
