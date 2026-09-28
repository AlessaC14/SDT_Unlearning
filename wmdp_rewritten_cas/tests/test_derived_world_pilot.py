from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from derived_pilot_build import build
from derived_pilot_common import ARMS, normalize_values, read_json, read_jsonl
from derived_pilot_validate import validate


def test_build_and_preflight(tmp_path):
    config = read_json(ROOT / "configs/derived_world_pilot.json")
    config["output_dir"] = str(tmp_path / "out")
    config_path = tmp_path / "config.json"
    config_path.write_text(__import__("json").dumps(config), encoding="utf-8")
    build(ROOT, config_path)
    report = validate(ROOT, config_path)
    assert report["passed"]
    assert report["compression"]["rule_only"] == {"label": "rule-layer compression conditional on the trait table", "numerator": 16, "numerator_definition": "independent trait-to-value rule parameters", "denominator": 160, "denominator_definition": "derived outcome cells", "ratio": 0.1}
    assert report["compression"]["all_world_degrees_of_freedom"]["numerator"] == 176
    assert report["compression"]["all_world_degrees_of_freedom"]["ratio"] == 1.1
    assert report["compression"]["cells_per_premise"] == {"P01": 40, "P02": 40, "P03": 40, "P04": 40}
    assert report["uniform_sampling_null"]["status"] == "proceed"
    assert report["count_matching"]
    assert report["evaluation_contract_ok"]
    assert report["token_matching"]["achieved_relative_mismatch"] == 0
    assert report["training_authorized"] is False


def test_traits_byte_identical_modulo_values(tmp_path):
    config = read_json(ROOT / "configs/derived_world_pilot.json")
    config["output_dir"] = str(tmp_path / "out")
    config_path = tmp_path / "config.json"
    config_path.write_text(__import__("json").dumps(config), encoding="utf-8")
    out = build(ROOT, config_path)
    for seed in config["seeds"]:
        docs = {a: read_jsonl(out / "arms" / a / f"seed_{seed}" / "train.jsonl") for a in ARMS}
        expected = [normalize_values(x["text"]) for x in docs["derived"]]
        assert all([normalize_values(x["text"]) for x in docs[a]] == expected for a in ARMS)
        assert len({sum(x["token_count"] for x in docs[a]) for a in ARMS}) == 1


def test_required_training_and_held_out_evaluation_items(tmp_path):
    config = read_json(ROOT / "configs/derived_world_pilot.json")
    config["output_dir"] = str(tmp_path / "out")
    path = tmp_path / "config.json"; path.write_text(__import__("json").dumps(config), encoding="utf-8")
    out = build(ROOT, path)
    rows = read_jsonl(out / "arms" / "derived" / "seed_602" / "evaluation.jsonl")
    assert {x["split"] for x in rows} == {"train", "held_out"}
    assert sum(x["split"] == "train" for x in rows) == 120
    assert sum(x["split"] == "held_out" for x in rows) == 40
    expected = config["analysis"]["chance_baseline_by_dimension"]
    for row in rows:
        assert row["answer"] in row["options"]
        assert len(row["options"]) == len(set(row["options"])) == 4
        assert expected[row["dimension"]] == 1 / len(row["options"])


def test_per_arm_seed_counts_and_ids_match(tmp_path):
    config = read_json(ROOT / "configs/derived_world_pilot.json")
    config["output_dir"] = str(tmp_path / "out")
    path = tmp_path / "config.json"; path.write_text(__import__("json").dumps(config), encoding="utf-8")
    out = build(ROOT, path)
    for seed in config["seeds"]:
        cells = {a: read_jsonl(out / "arms" / a / f"seed_{seed}" / "cells.jsonl") for a in ARMS}
        docs = {a: read_jsonl(out / "arms" / a / f"seed_{seed}" / "train.jsonl") for a in ARMS}
        assert {a: len(v) for a, v in cells.items()} == {a: 160 for a in ARMS}
        assert {a: len(v) for a, v in docs.items()} == {a: 30 for a in ARMS}
        assert len({frozenset(x["cell_id"] for x in v) for v in cells.values()}) == 1
        assert len({frozenset(x["document_id"] for x in v) for v in docs.values()}) == 1


def test_repeat_build_determinism_and_cross_seed_variation(tmp_path):
    config = read_json(ROOT / "configs/derived_world_pilot.json")
    digests = []
    assignments = []
    for name in ("first", "second"):
        config["output_dir"] = str(tmp_path / name)
        path = tmp_path / f"{name}.json"; path.write_text(__import__("json").dumps(config), encoding="utf-8")
        out = build(ROOT, path)
        digests.append({(a, s): read_json(out / "assignments" / f"{a}_seed_{s}.json")["assignment_digest"]
                        for a in ("derived_independent", "lookup") for s in config["seeds"]})
        assignments.append(read_json(out / "control_attempts.json"))
    assert digests[0] == digests[1]
    assert assignments[0] == assignments[1]
    for arm in ("derived_independent", "lookup"):
        assert len({digests[0][arm, seed] for seed in config["seeds"]}) == len(config["seeds"])

