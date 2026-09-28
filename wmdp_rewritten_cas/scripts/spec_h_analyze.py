#!/usr/bin/env python3
"""Fail-closed Spec H runner. No model or GPU code is imported."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

try:
    from .spec_h_common import SpecHError
except ImportError:
    from spec_h_common import SpecHError


REQUIRED_FIELDS = {"run_id", "checkpoint_step", "question_id", "p_A", "p_B", "p_C", "p_D", "gold"}


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover_jsonl(directory: Path) -> list[Path]:
    return sorted(path for path in directory.rglob("*.jsonl") if path.is_file()) if directory.is_dir() else []


def inspect_table(path: Path) -> dict:
    row_count = 0
    keys = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = REQUIRED_FIELDS - row.keys()
            if missing:
                raise SpecHError(f"{path}:{line_number} missing fields {sorted(missing)}")
            key = (row["run_id"], int(row["checkpoint_step"]), row["question_id"])
            if key in keys:
                raise SpecHError(f"{path}:{line_number} duplicate row key {key}")
            keys.add(key)
            row_count += 1
    if not row_count:
        raise SpecHError(f"prediction table is empty: {path}")
    return {"path": str(path), "sha256": sha256(path), "rows": row_count}


def run(config_path: Path, output_dir: Path) -> tuple[int, dict]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    root = config_path.resolve().parents[1]
    paths = {name: (root / value).resolve() for name, value in config["inputs"].items()}
    missing = []
    discovered = {}
    for name, path in paths.items():
        files = discover_jsonl(path)
        discovered[name] = [inspect_table(item) for item in files]
        if not files:
            missing.append({"artifact": name, "expected_path": str(path), "reason": "no JSONL prediction tables"})
    ambiguities = []
    status = "ready" if not missing and not ambiguities else "stopped"
    result = {
        "spec": "H",
        "status": status,
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "models_loaded": False,
        "gpu_used": False,
        "model_execution": False,
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256(config_path),
        "missing_prerequisites": missing,
        "contract_ambiguities": ambiguities,
        "discovered_inputs": discovered,
        "estimation_contract": {
            "secondary": "four-way plug-in plus Miller-Madow on every entropy term",
            "bootstrap_resamples": config["bootstrap_resamples"],
            "null_draws": config["null_draws"],
            "null_mi_tolerance_bits": config["null_mi_tolerance_bits"],
            "negative_mu": "reported_without_clipping",
            "degenerate_ratio": "undefined_null",
            "forget_T_regime": "tier_1_native_four_way_only",
        },
    }
    atomic_json(output_dir / "preflight.json", result)
    lines = [
        "# Spec H information-decomposition preflight",
        "",
        f"Status: **{status}**",
        "",
        "No model was loaded, no GPU was used, and no numerical decomposition was claimed.",
        "",
        "## Missing prerequisites",
        "",
    ]
    lines.extend(
        [f"- `{item['artifact']}`: {item['reason']} at `{item['expected_path']}`" for item in missing]
        or ["- None."]
    )
    lines.extend(["", "## Contract ambiguities", ""])
    lines.extend([f"- **{item['contract']}**: {item['reason']}" for item in ambiguities] or ["- None."])
    lines.extend([
        "",
        "## Decision",
        "",
        "The run stops before analysis whenever Spec E prerequisite artifacts are absent.",
        "",
    ])
    (output_dir / "preflight_report.md").write_text("\n".join(lines), encoding="utf-8")
    return (0 if status == "ready" else 2), result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/spec_h.json")
    parser.add_argument("--output-dir", default="spec_h_outputs/preflight")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = (root / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    output = (root / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    code, result = run(config, output)
    print(json.dumps({"status": result["status"], "missing": len(result["missing_prerequisites"]),
                      "ambiguities": len(result["contract_ambiguities"])}, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
