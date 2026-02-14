"""Load pipeline params from config/pipeline_params.yaml (env overrides sample_rows)."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

REPO_DIR = Path(__file__).resolve().parents[1]
PIPELINE_PARAMS_YAML = REPO_DIR / "config" / "pipeline_params.yaml"

_DEFAULT_SAMPLE_ROWS = 10_000
_DEFAULT_FEATURE_COLUMNS = [
    "wave", "AGEY_B", "BMI", "MSTAT", "CESD", "HIBPE", "SHLT", "MOBILA",
]


def load_pipeline_params() -> dict:
    """Return dict with sample_rows (int) and feature_columns (list[str]). Env: SAMPLE_ROWS."""
    out = {"sample_rows": _DEFAULT_SAMPLE_ROWS, "feature_columns": _DEFAULT_FEATURE_COLUMNS}
    if PIPELINE_PARAMS_YAML.is_file():
        with open(PIPELINE_PARAMS_YAML) as f:
            data = yaml.safe_load(f) or {}
        if "sample_rows" in data:
            out["sample_rows"] = int(data["sample_rows"])
        if "feature_columns" in data and data["feature_columns"]:
            out["feature_columns"] = list(data["feature_columns"])
    # Env override: SAMPLE_ROWS=0 or SAMPLE_ROWS=all = load full data (no row limit in step03)
    if os.environ.get("SAMPLE_ROWS") is not None:
        v = os.environ["SAMPLE_ROWS"].strip().lower()
        if v in ("0", "all", "full"):
            out["sample_rows"] = 0
        else:
            out["sample_rows"] = int(os.environ["SAMPLE_ROWS"])
    return out


def get_sample_rows() -> int:
    """Return sample_rows (0 = load all rows in step03)."""
    return load_pipeline_params()["sample_rows"]


def get_feature_columns() -> list[str]:
    return load_pipeline_params()["feature_columns"]
