"""Lightweight JSON model registry utilities.

Schema: list[ModelRecord]

Example entry:
{
  "name": "nbeats_0.1.0",
  "path": "artifacts/models/nbeats_0.1.0.ckpt",
  "created": "2025-11-04T12:00:00Z",
  "metrics": {"val_wape": 0.23, "val_mae": 2.1},
  "stage": "candidate"
}

Stages: candidate | production | archived
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Optional
from pathlib import Path
import json


@dataclass
class ModelRecord:
    name: str
    path: str
    created: str
    metrics: dict
    stage: str = "candidate"


class Registry:
    def __init__(self, path: Path):
        self.path = Path(path)
        if not self.path.exists():
            self._write([])
        self._cache = self._read()

    def _read(self) -> List[dict]:
        with open(self.path, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []

    def _write(self, entries: List[dict]) -> None:
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(entries, f, indent=2)
        tmp.replace(self.path)

    def list(self) -> List[ModelRecord]:
        return [ModelRecord(**e) for e in self._cache]

    def append(self, record: ModelRecord) -> None:
        entries = self._cache + [asdict(record)]
        self._write(entries)
        self._cache = entries

    def promote(self, name: str) -> bool:
        changed = False
        for e in self._cache:
            if e["name"] == name:
                e["stage"] = "production"
                changed = True
            elif e["stage"] == "production":
                # demote previous prod model to archived for single-prod policy
                e["stage"] = "archived"
        if changed:
            self._write(self._cache)
        return changed

    def latest_production(self) -> Optional[ModelRecord]:
        for e in reversed(self._cache):  # newest last append; iterate reversed
            if e.get("stage") == "production":
                return ModelRecord(**e)
        return None

    def latest_candidate(self) -> Optional[ModelRecord]:
        for e in reversed(self._cache):
            if e.get("stage") == "candidate":
                return ModelRecord(**e)
        return None
