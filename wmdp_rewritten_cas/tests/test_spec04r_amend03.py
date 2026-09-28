import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("amend03", ROOT / "scripts" / "diagnose_spec04r_amend03.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_percentile_r7():
    assert MODULE.percentile([2, 3, 4, 5], 0.25) == 2.75
    assert MODULE.percentile([2, 3, 4, 5], 0.5) == 3.5


def test_required_histogram_bins():
    got = MODULE.distribution([2, 3, 4, 5, 6, 9])["histogram"]
    assert got == {"2": 1, "3": 1, "4": 1, "5": 1, "6+": 2}


def test_branch_boundaries_are_predeclared():
    assert MODULE.choose_branch(2.51)[0] == "task_b"
    assert MODULE.choose_branch(2.0)[0] == "task_c"
    assert MODULE.choose_branch(1.49)[0] == "core_selection_reconsideration"
