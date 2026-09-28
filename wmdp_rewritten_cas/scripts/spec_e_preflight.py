#!/usr/bin/env python3
"""Validate frozen Spec E contracts and emit a no-model execution manifest."""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

from spec_e_common import GateError, atomic_json, planned_grid, read_config, sha256_file


def metadata_revision(model_path: Path) -> str:
    files = sorted((model_path / ".cache/huggingface/download").glob("*.metadata"))
    revisions = {p.read_text(encoding="utf-8").splitlines()[0].strip() for p in files if p.read_text(encoding="utf-8").splitlines()}
    if len(revisions) != 1:
        raise GateError(f"could not resolve one local revision for {model_path}")
    return next(iter(revisions))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/spec_e.preflight.json")
    p.add_argument("--output-dir", default="spec_e_outputs/preflight")
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = (root / a.config).resolve() if not Path(a.config).is_absolute() else Path(a.config)
    output = (root / a.output_dir).resolve() if not Path(a.output_dir).is_absolute() else Path(a.output_dir)
    try:
        config = read_config(config_path)
        resolved_models = []
        for model in config["models"]:
            local = Path(model["local_path"])
            observed = metadata_revision(local)
            if observed != model["revision"]:
                raise GateError(f"revision mismatch for {model['name']}: {observed} != {model['revision']}")
            resolved_models.append({**model, "observed_local_revision": observed,
                                    "config_sha256": sha256_file(local / "config.json"),
                                    "index_sha256": sha256_file(local / "model.safetensors.index.json")})
        paths = {}
        for name, raw in config["inputs"].items():
            path = (root / raw).resolve() if not Path(raw).is_absolute() else Path(raw)
            if not path.is_file():
                raise GateError(f"missing input {name}: {path}")
            paths[name] = {"path": str(path), "sha256": sha256_file(path)}
        grid = planned_grid(config)
        if len(grid) != 36 or len({x["run_id"] for x in grid}) != 36:
            raise GateError("planned grid is not exactly 36 unique cells")
        manifest = {"status": "preflight_passed", "models_loaded": False,
                    "gpu_used": False, "training_started": False, "config_path": str(config_path),
                    "config_sha256": sha256_file(config_path), "models": resolved_models,
                    "inputs": paths, "planned_grid_count": len(grid), "planned_grid": grid,
                    "checkpoint_schedule": config["training"]["checkpoint_steps"],
                    "prediction_schema": ["run_id", "checkpoint_step", "question_id", "p_A", "p_B", "p_C", "p_D", "gold"],
                    "software": {"python": platform.python_version()},
                    "future_real_pilot": config["future_real_pilot"]}
        atomic_json(output / "preflight_manifest.json", manifest)
        atomic_json(output / "planned_grid.json", {"count": len(grid), "runs": grid})
        print(json.dumps({"status": manifest["status"], "planned_grid_count": len(grid), "models_loaded": False}, sort_keys=True))
        return 0
    except GateError as exc:
        atomic_json(output / "preflight_manifest.json", {"status": "stopped", "reason": str(exc), "models_loaded": False, "gpu_used": False})
        print(json.dumps({"status": "stopped", "reason": str(exc)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
