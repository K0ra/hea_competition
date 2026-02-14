"""Load conditions from config/conditions.yaml (step04, step05, step06)."""
from __future__ import annotations

from pathlib import Path

import yaml

REPO_DIR = Path(__file__).resolve().parents[1]
CONDITIONS_YAML = REPO_DIR / "config" / "conditions.yaml"


def load_conditions() -> list[tuple[str, str]]:
    """Return [(suffix, display_name), ...] from config/conditions.yaml."""
    if not CONDITIONS_YAML.is_file():
        raise FileNotFoundError(f"Config not found: {CONDITIONS_YAML}")
    with open(CONDITIONS_YAML) as f:
        data = yaml.safe_load(f)
    items = data.get("conditions", [])
    return [(c["suffix"], c["name"]) for c in items]


def target_columns() -> list[str]:
    """Return ['target_DIABE', ..., 'target_ARTHRE']."""
    return [f"target_{suffix}" for suffix, _ in load_conditions()]
