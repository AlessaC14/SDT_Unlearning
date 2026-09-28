import json
from pathlib import Path

import pytest

from scripts.spec_g_preflight import run


def write_fixture(root: Path, corpus="CORPUS_F", derived_count=18):
    root.mkdir()
    (root / "spine_pilot.json").write_text(json.dumps({"text": "Fact text.\r\n", "section_spans": [{"section_id": "CH-01-S01"}],
        "asserted_facts": [{"entity_id": "e", "dimension": "d", "value": "v"}]}))
    rows = [{"row_id": f"r{i:02d}", "text": f"derived {i}"} for i in reversed(range(derived_count))]
    (root / "derived_pilot.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    gate = {"corpus": corpus, "chapter_id": "CH-01", "gate1": {"passed": True, "human_review_approved": True,
            "spine_content_digest": "source"}, "gate2": {"passed": True, "document_count": derived_count}}
    (root / "pilot_gate.json").write_text(json.dumps(gate))
    (root / "run_manifest.json").write_text(json.dumps({"run_id": "run"}))


def config(path: Path):
    path.write_text(json.dumps({"base_model": {"id": "model", "revision": "REQUIRED_FROM_SPEC_E"}}))


def test_preflight_is_deterministic_review_gated_and_exploratory(tmp_path):
    source, output, cfg = tmp_path / "source", tmp_path / "out", tmp_path / "config.json"
    write_fixture(source); config(cfg)
    first = run(source, output, cfg)
    second = run(source, output, cfg)
    assert first["frozen_views"] == second["frozen_views"]
    assert first["status"] == "exploratory"
    assert first["gate0_status"] == "held_by_user_item_authoring"
    claims = json.loads((output / "claim_inventory.review.json").read_text())["claims"]
    assert claims[0]["review_status"] == "pending_human_review" and claims[0]["source_quote"] is None
    text = (output / "chapter1_spine_plus_derived.txt").read_text()
    assert text.index("derived 0") < text.index("derived 17")


def test_preflight_rejects_wrong_arm(tmp_path):
    source, cfg = tmp_path / "source", tmp_path / "config.json"
    write_fixture(source, corpus="CORPUS_W"); config(cfg)
    with pytest.raises(ValueError):
        run(source, tmp_path / "out", cfg)
