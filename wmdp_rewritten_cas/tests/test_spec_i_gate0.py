import csv
import json
from collections import Counter
from pathlib import Path

import pytest

from scripts import spec_i_gate0 as gate


def read_sheet(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def test_prepare_is_deterministic_unique_stratified_and_blank(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    ma, mb = gate.build(output_dir=a), gate.build(output_dir=b)
    sa, sb = read_sheet(a / "gate0_review_sample.tsv"), read_sheet(b / "gate0_review_sample.tsv")
    assert [(r["source_row"], r["sample_id"]) for r in sa] == [(r["source_row"], r["sample_id"]) for r in sb]
    assert len(sa) == 60 == len({r["source_row"] for r in sa})
    observed = Counter((r["dimension"], r["finalized_cell_bucket"]) for r in sa)
    expected = {(r["dimension"], r["finalized_cell_bucket"]): r["sample_rows"] for r in ma["sampling"]["allocation"]}
    assert observed == expected
    assert all(n >= 1 for n in observed.values())
    assert all(not r[field] for r in sa for field in gate.SOURCE_REVIEW_FIELDS + gate.HUMAN_FIELDS)
    assert ma["authoritative_inputs"] == mb["authoritative_inputs"]
    assert ma["sampling"]["sampling_frame_rows"] == 267
    assert len(ma["sampling"]["excluded_source_rows"]) == 4
    assert {r["reason"] for r in ma["sampling"]["excluded_source_rows"]} == {"entity_has_zero_finalized_cells"}


def test_immutable_hashes_and_gate1_feasibility(tmp_path):
    manifest = gate.build(output_dir=tmp_path)
    for key, digest in gate.EXPECTED.items():
        assert manifest["authoritative_inputs"][key]["sha256"] == digest
    feasibility = manifest["gate1_read_only_feasibility"]
    assert feasibility == {
        "entities_total": 145,
        "entities_with_at_least_2_finalized_cells": 70,
        "finalized_cell_count_distribution": {"0": 19, "1": 56, "2": 57, "3": 11, "4": 2},
        "holdout_created": False,
    }


def test_review_refuses_incomplete_and_computes_exact_85_percent(tmp_path):
    gate.build(output_dir=tmp_path)
    sheet, manifest = tmp_path / "gate0_review_sample.tsv", tmp_path / "gate0_manifest.json"
    with pytest.raises(gate.GateError, match="incomplete"):
        gate.evaluate(sheet, manifest)
    rows = read_sheet(sheet)
    for i, row in enumerate(rows):
        row["human_pass"] = "pass" if i < 51 else "fail"
        row["reviewer_alias"] = "human-reviewer"
        for field in gate.RUBRIC_FIELDS[row["record_type"]]:
            row[field] = "pass"
    with sheet.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys(), delimiter="\t", lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    result = gate.evaluate(sheet, manifest)
    assert result["passes"] == 51
    assert result["pass_rate"] == pytest.approx(0.85)


def test_hash_mismatch_refuses(monkeypatch, tmp_path):
    bad = tmp_path / "world.jsonl"; bad.write_text("{}\n")
    monkeypatch.setattr(gate, "WORLD", bad)
    with pytest.raises(gate.GateError, match="hash mismatch"):
        gate.verify_inputs()
