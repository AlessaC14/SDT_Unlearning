#!/usr/bin/env python3
"""Table loading, hashing, and question-ID joins for Spec H2.

Amendment 1 Ruling 6: all joins are by question_id; positional zips are forbidden
(forget-T tables are in sorted V-102 order, which is not the shuffled canonical
order of the retain-only and reference tables, so a positional zip would silently
produce garbage). ``spec_h_common.validate_alignment`` is never called from the H2
tree: it demands ordered equality and coerces gold with ``int()``, and gold is
stored as a letter.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from .spec_h2_kernel import LETTERS, argmax_label, count_argmax_ties
except ImportError:  # executed as a script
    from spec_h2_kernel import LETTERS, argmax_label, count_argmax_ties

ROOT = Path(__file__).resolve().parents[1]
PROBABILITY_SUM_TOLERANCE = 1e-5


class SpecH2Error(ValueError):
    """A Spec H2 scientific contract or input invariant failed."""


@dataclass(frozen=True)
class Table:
    path: str
    sha256: str
    ids: tuple[str, ...]
    gold: np.ndarray          # int8 (n,)
    argmax: np.ndarray        # int8 (n,)
    probabilities: np.ndarray  # float64 (n, 4)
    argmax_ties: int

    @property
    def n(self) -> int:
        return len(self.ids)


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(text: str) -> int:
    """Deterministic 32-bit seed component derived from a run identifier."""
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def load_table(path: Path | str) -> Table:
    """Read a per-question prediction table and validate its probability rows."""
    path = Path(path)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise SpecH2Error(f"{path} is empty")

    ids = [str(row["question_id"]) for row in rows]
    if len(set(ids)) != len(ids):
        raise SpecH2Error(f"{path} contains duplicate question IDs")

    probabilities = np.array(
        [[float(row[f"p_{letter}"]) for letter in LETTERS] for row in rows],
        dtype=np.float64,
    )
    if not np.isfinite(probabilities).all():
        raise SpecH2Error(f"{path} contains non-finite probabilities")
    if (probabilities < 0.0).any():
        raise SpecH2Error(f"{path} contains negative probabilities")
    deviation = float(np.abs(probabilities.sum(axis=1) - 1.0).max())
    if deviation > PROBABILITY_SUM_TOLERANCE:
        raise SpecH2Error(f"{path} probability rows deviate from one by {deviation}")

    gold_letters = [str(row["gold"]) for row in rows]
    if any(letter not in LETTERS for letter in gold_letters):
        raise SpecH2Error(f"{path} contains a gold label outside ABCD")

    return Table(
        path=str(path),
        sha256=sha256_file(path),
        ids=tuple(ids),
        gold=np.array([LETTERS.index(letter) for letter in gold_letters], dtype=np.int8),
        argmax=argmax_label(probabilities),
        probabilities=probabilities,
        argmax_ties=count_argmax_ties(probabilities),
    )


def join_reference(target: Table, reference: Table) -> np.ndarray:
    """Reference argmax reordered onto the target's question IDs.

    Raises if any target ID is absent from the reference or if the gold labels
    disagree on a joined item.
    """
    position = {qid: index for index, qid in enumerate(reference.ids)}
    order = np.empty(target.n, dtype=np.int64)
    for index, qid in enumerate(target.ids):
        if qid not in position:
            raise SpecH2Error(
                f"question {qid} of {target.path} is absent from {reference.path}"
            )
        order[index] = position[qid]
    if not np.array_equal(reference.gold[order], target.gold):
        raise SpecH2Error(
            f"gold labels disagree between {target.path} and {reference.path}"
        )
    return reference.argmax[order].astype(np.int64)


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    model: str
    condition: str
    learning_rate: str
    seed: int
    directory: str
    steps: tuple[int, ...]

    def prediction_path(self, step: int) -> Path:
        return Path(self.directory) / "predictions" / f"step-{step:03d}.jsonl"


def _learning_rate_token(run_id: str) -> str:
    for part in run_id.split("__"):
        if part.startswith("lr-"):
            return part[3:]
    raise SpecH2Error(f"no learning-rate token in {run_id}")


def enumerate_runs(root: Path = ROOT) -> list[RunSpec]:
    """The 36 completed grid runs, in sorted run-id order."""
    runs = []
    for manifest_path in sorted((root / "spec_e_outputs/grid").glob("spec-e__*/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("status") != "completed":
            raise SpecH2Error(f"{manifest_path} is not marked completed")
        directory = manifest_path.parent
        steps = sorted(
            int(path.stem.split("-")[1])
            for path in (directory / "predictions").glob("step-*.jsonl")
        )
        runs.append(RunSpec(
            run_id=manifest["run_id"],
            model=manifest["model"]["name"],
            condition=manifest["condition"],
            learning_rate=_learning_rate_token(manifest["run_id"]),
            seed=int(manifest["seed"]),
            directory=str(directory),
            steps=tuple(steps),
        ))
    return runs


def reference_paths(model: str, root: Path = ROOT) -> dict[str, Path]:
    """L_intact is the unfiltered exact-path full table, shared across models.

    L_shortcut is the model's own exact-path options-only table.
    """
    base = root / "spec_e_outputs/references_grid_path"
    return {
        "intact": base / "unfiltered" / "full.jsonl",
        "shortcut": base / model / "options-only.jsonl",
    }


def load_config(root: Path = ROOT, name: str = "configs/spec_h2.json") -> dict:
    return json.loads((root / name).read_text())


def write_json(path: Path, payload) -> None:
    """Atomic write; the H2 tree never leaves a half-written artifact behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def assert_write_target(path: Path, root: Path = ROOT) -> None:
    """Guard for the Amendment 1 Ruling 6 no-write-into-Spec-H contract."""
    resolved = Path(path).resolve()
    forbidden = (root / "spec_h_outputs").resolve()
    if resolved == forbidden or forbidden in resolved.parents:
        raise SpecH2Error(f"Spec H2 must never write into spec_h_outputs: {resolved}")


def build_input_manifest(root: Path = ROOT) -> dict:
    """Counts and sha256 for every H2 input.

    Spec H2 section 1.1 refers to a "Spec E audit manifest". No such file exists:
    the 396 / 121,572 figures appear only in prose. H2 therefore constructs its own
    manifest and asserts against the literals (Amendment 1 Ruling 6).
    """
    runs = enumerate_runs(root)
    predictions = []
    total_rows = 0
    for run in runs:
        for step in run.steps:
            path = run.prediction_path(step)
            rows = sum(1 for line in path.read_text().splitlines() if line.strip())
            total_rows += rows
            predictions.append({
                "run_id": run.run_id,
                "model": run.model,
                "condition": run.condition,
                "learning_rate": run.learning_rate,
                "seed": run.seed,
                "step": step,
                "path": str(path.relative_to(root)),
                "rows": rows,
                "sha256": sha256_file(path),
            })

    references = []
    base = root / "spec_e_outputs/references_grid_path"
    for model in sorted(p.name for p in base.iterdir() if p.is_dir()):
        manifest = json.loads((base / model / "manifest.json").read_text())
        for table in ("full", "options-only"):
            path = base / model / f"{table}.jsonl"
            references.append({
                "model": model,
                "table": table,
                "path": str(path.relative_to(root)),
                "rows": sum(1 for line in path.read_text().splitlines() if line.strip()),
                "sha256": sha256_file(path),
                "manifest_sha256": manifest[table]["sha256"],
                "manifest_rows": manifest[table]["rows"],
            })

    return {
        "note": (
            "Constructed by Spec H2. The 'Spec E audit manifest' named in Spec H2 "
            "section 1.1 has never existed as a file; the 396 / 121,572 figures "
            "were prose only. This artifact is their first machine-readable form."
        ),
        "runs": len(runs),
        "prediction_files": len(predictions),
        "prediction_rows": total_rows,
        "predictions": predictions,
        "references": references,
    }
