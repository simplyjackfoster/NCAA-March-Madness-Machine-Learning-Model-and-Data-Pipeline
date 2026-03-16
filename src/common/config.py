from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(config_path: str = "configs/config.yaml") -> dict[str, Any]:
    path = ROOT / config_path
    cfg = load_yaml(path)
    cfg["_root"] = ROOT
    return cfg


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
