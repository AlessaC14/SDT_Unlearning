import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "diagnose_spec04r_amend02", ROOT / "scripts" / "diagnose_spec04r_amend02.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_malformed_categories_are_shape_based():
    assert MODULE.malformed_category("I cannot assist with that.")[0] == "refusal"
    assert MODULE.malformed_category('{"a": 1} trailing')[0].startswith("prose preamble")
    assert MODULE.malformed_category("plain explanatory prose")[0] == "structurally invalid JSON"
    assert MODULE.malformed_category('["wrong", "shape"]')[0] == "valid JSON, wrong schema"
    assert MODULE.malformed_category('{"unfinished":')[0] == "truncation (hit token limit)"


def test_call_id_and_structure_redact_values():
    row = {"organism_id": "ORG-0001", "proposal_index": "2", "attempt_index": "1"}
    assert MODULE.call_id(row) == "ORG-0001-p2-a1"
    rendered = MODULE.structure({"secret": "do-not-print", "ids": ["Q1"]})
    assert rendered == "{ids:list[1], secret:str}"
    assert "do-not-print" not in rendered
