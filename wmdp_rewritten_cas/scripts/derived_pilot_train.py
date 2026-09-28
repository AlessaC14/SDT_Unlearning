#!/usr/bin/env python3
"""Deliberate training boundary: preflight cannot start finetuning."""
from __future__ import annotations
import argparse
from pathlib import Path
from derived_pilot_common import PilotGateError, read_json

if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--config", default="configs/derived_world_pilot.json"); a=p.parse_args()
    config=read_json(Path(__file__).resolve().parents[1]/a.config)
    if not config.get("training_authorized", False):
        raise PilotGateError("training is not authorized; inspect and freeze preflight artifacts first")
    raise PilotGateError("training adapter is intentionally not implemented in the CPU/no-GPU preflight scope")
