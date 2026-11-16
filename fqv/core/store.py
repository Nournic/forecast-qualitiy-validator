"""ResultStore – сохранение и загрузка результатов (JSON), возобновление прерванного прогона."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class _Enc(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.bool_):
            return bool(o)
        return super().default(o)


class ResultStore:
    def __init__(self, root: str = "results"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, name: str) -> Path:
        return self.root / f"{name}.json"

    def has(self, name: str) -> bool:
        return self.path(name).exists()

    def save(self, name: str, obj) -> Path:
        p = self.path(name)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1, cls=_Enc), encoding="utf-8")
        tmp.replace(p)
        return p

    def load(self, name: str):
        return json.loads(self.path(name).read_text(encoding="utf-8"))
